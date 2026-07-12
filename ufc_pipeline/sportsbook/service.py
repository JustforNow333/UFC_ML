"""Orchestration for local archive, normalization, matching, and storage."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import SportsbookConfig
from .domain import NormalizationResult, NormalizedOutcome, NormalizedSnapshot
from .matching import CanonicalMatcher
from .providers.base import ProviderAdapter, ProviderSchemaError
from .providers.the_odds_api import TheOddsApiAdapter
from .storage import ArchiveResult, SportsbookStore


def provider_adapter(provider_name: str) -> ProviderAdapter:
    if provider_name == "the_odds_api":
        return TheOddsApiAdapter()
    if provider_name == "sportsdataio":
        raise NotImplementedError("SportsDataIO is interface-compatible but its Stage 2 adapter is not implemented")
    raise ValueError(f"unsupported provider: {provider_name}")


@dataclass(frozen=True)
class IngestionResult:
    provider_name: str
    request_type: str
    archive: ArchiveResult
    ingestion_batch_id: str | None
    normalization_status: str
    snapshots_seen: int
    snapshots_inserted: int
    outcomes_inserted: int
    matched_snapshots: int
    unresolved_snapshots: int
    rejected_snapshots: int
    review_rows: int
    normalization_errors: tuple[dict[str, Any], ...]
    dry_run: bool


class SportsbookIngestionService:
    def __init__(self, config: SportsbookConfig | None = None):
        self.config = config or SportsbookConfig()
        self.store = SportsbookStore(self.config)

    @staticmethod
    def load_json(path: str | Path) -> Any:
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProviderSchemaError(f"invalid JSON in {path}: {exc}") from exc

    def ingest_local_payload(
        self,
        *,
        provider_name: str,
        payload: Any,
        request_type: str,
        source_file: str | None = None,
        query_timestamp_utc: str | None = None,
        fetched_at_utc: str | None = None,
        dry_run: bool = False,
    ) -> IngestionResult:
        adapter = provider_adapter(provider_name)
        archive = self.store.archive_payload(provider_name, payload, dry_run=dry_run)
        if dry_run:
            normalized = adapter.normalize(payload, request_type=request_type, source_payload_hash=archive.payload_hash)
            matcher = CanonicalMatcher(self.config.canonical_ufc_db, sportsbook_db=self.config.database_path)
            matched, reviews = matcher.match_many(normalized.snapshots)
            return self._result(provider_name, request_type, archive, None, normalized, matched, reviews, None, True)
        batch_id = self.store.create_ingestion_batch(
            provider_name=provider_name,
            request_type=request_type,
            archive=archive,
            query_timestamp_utc=query_timestamp_utc,
            fetched_at_utc=fetched_at_utc,
            source_file=source_file,
        )
        try:
            normalized = adapter.normalize(payload, request_type=request_type, source_payload_hash=archive.payload_hash)
        except Exception as exc:
            connection = self.store.connect()
            try:
                with connection:
                    connection.execute(
                        "UPDATE provider_ingestion_batches SET response_status='failed', error_message=? WHERE ingestion_batch_id=?",
                        (str(exc), batch_id),
                    )
            finally:
                connection.close()
            raise
        discovered_query_timestamp = normalized.provider_metadata.get("historical_query_timestamp")
        if query_timestamp_utc is None and discovered_query_timestamp:
            connection = self.store.connect()
            try:
                with connection:
                    connection.execute(
                        "UPDATE provider_ingestion_batches SET query_timestamp_utc=? WHERE ingestion_batch_id=?",
                        (discovered_query_timestamp, batch_id),
                    )
            finally:
                connection.close()
        matcher = CanonicalMatcher(self.config.canonical_ufc_db, sportsbook_db=self.config.database_path)
        matched, reviews = matcher.match_many(normalized.snapshots)
        counts = self.store.store_snapshots(batch_id, matched)
        review_count = self.store.record_match_reviews(batch_id, reviews)
        return self._result(provider_name, request_type, archive, batch_id, normalized, matched, reviews, counts, False, review_count)

    @staticmethod
    def _result(
        provider_name: str,
        request_type: str,
        archive: ArchiveResult,
        batch_id: str | None,
        normalized: NormalizationResult,
        matched: tuple,
        reviews: tuple,
        counts: dict[str, int] | None,
        dry_run: bool,
        review_count: int | None = None,
    ) -> IngestionResult:
        statuses = [
            "matched" if item.canonical_bout_id else item.normalization_status
            for item in matched
        ]
        matched_count = statuses.count("matched")
        unresolved = sum(status in {"normalized", "unmatched", "ambiguous"} for status in statuses)
        rejected = len(statuses) - matched_count - unresolved
        return IngestionResult(
            provider_name=provider_name,
            request_type=request_type,
            archive=archive,
            ingestion_batch_id=batch_id,
            normalization_status="normalized_with_errors" if normalized.errors else "normalized",
            snapshots_seen=len(normalized.snapshots),
            snapshots_inserted=(counts or {}).get("snapshots_inserted", 0),
            outcomes_inserted=(counts or {}).get("outcomes_inserted", 0),
            matched_snapshots=matched_count,
            unresolved_snapshots=unresolved,
            rejected_snapshots=rejected,
            review_rows=review_count if review_count is not None else len(reviews),
            normalization_errors=normalized.errors,
            dry_run=dry_run,
        )

    def rematch_stored(self, *, ingestion_batch_id: str | None = None, dry_run: bool = False) -> dict[str, Any]:
        connection = self.store.connect(read_only=True)
        try:
            parameters: tuple[Any, ...] = ()
            where = ""
            if ingestion_batch_id:
                where = "WHERE s.ingestion_batch_id=?"
                parameters = (ingestion_batch_id,)
            snapshot_rows = connection.execute(
                f"SELECT s.* FROM market_snapshots s {where} ORDER BY s.snapshot_id",
                parameters,
            ).fetchall()
            snapshots: list[NormalizedSnapshot] = []
            for row in snapshot_rows:
                outcomes = connection.execute(
                    "SELECT * FROM outcomes WHERE snapshot_id=? ORDER BY outcome_id",
                    (row["snapshot_id"],),
                ).fetchall()
                snapshots.append(NormalizedSnapshot(
                    provider_name=row["provider_name"], provider_event_id=row["provider_event_id"],
                    provider_market_id=row["provider_market_id"], sportsbook_key=row["sportsbook_key"],
                    provider_sportsbook_key=row["provider_sportsbook_key"],
                    sportsbook_display_name=row["sportsbook_display_name"], venue_type=row["venue_type"],
                    market_type=row["market_type"], observed_at_utc=row["observed_at_utc"],
                    observed_at_original=row["observed_at_original"],
                    scheduled_event_time_utc=row["scheduled_event_time_utc"],
                    scheduled_event_time_original=row["scheduled_event_time_utc"],
                    pre_fight_or_live_status=row["pre_fight_or_live_status"], raw_event_name=row["provider_event_id"],
                    sport_key="mma_mixed_martial_arts", league_key=None,
                    outcomes=tuple(NormalizedOutcome(
                        raw_outcome_name=outcome["raw_outcome_name"], raw_opponent_name=outcome["raw_opponent_name"],
                        decimal_odds=outcome["decimal_odds"], american_odds=outcome["american_odds"],
                        raw_implied_probability=outcome["raw_implied_probability"],
                        original_odds=json.loads(outcome["original_odds_json"]),
                        original_odds_format=outcome["original_odds_format"],
                        canonical_fighter_id=outcome["canonical_fighter_id"],
                        canonical_opponent_id=outcome["canonical_opponent_id"],
                        orientation_status=outcome["orientation_status"],
                        validation_status=outcome["validation_status"],
                        validation_reason=outcome["validation_reason"],
                    ) for outcome in outcomes),
                    normalization_status=row["normalization_status"], rejection_reason=row["rejection_reason"],
                    canonical_event_id=row["canonical_event_id"], canonical_bout_id=row["canonical_bout_id"],
                    matching_method=row["matching_method"],
                    matching_diagnostics=json.loads(row["matching_diagnostics_json"] or "{}"),
                    source_payload_hash=row["source_payload_hash"],
                ))
        finally:
            connection.close()
        matcher = CanonicalMatcher(self.config.canonical_ufc_db, sportsbook_db=self.config.database_path)
        results = [matcher.match_snapshot(snapshot) for snapshot in snapshots]
        matched_count = sum(item.status == "matched" for item in results)
        if not dry_run:
            write_connection = self.store.connect()
            try:
                with write_connection:
                    for original_row, result in zip(snapshot_rows, results):
                        snapshot = result.snapshot
                        write_connection.execute(
                            """UPDATE market_snapshots SET canonical_event_id=?, canonical_bout_id=?,
                               match_status=?, matching_method=?, matching_diagnostics_json=?,
                               normalization_status=?, rejection_reason=? WHERE snapshot_id=?""",
                            (snapshot.canonical_event_id, snapshot.canonical_bout_id, result.status,
                             snapshot.matching_method, json.dumps(result.diagnostics, sort_keys=True),
                             snapshot.normalization_status, snapshot.rejection_reason, original_row["snapshot_id"]),
                        )
                        stored_outcomes = write_connection.execute(
                            "SELECT outcome_id FROM outcomes WHERE snapshot_id=? ORDER BY outcome_id",
                            (original_row["snapshot_id"],),
                        ).fetchall()
                        for stored, outcome in zip(stored_outcomes, snapshot.outcomes):
                            write_connection.execute(
                                "UPDATE outcomes SET canonical_fighter_id=?, canonical_opponent_id=?, orientation_status=? WHERE outcome_id=?",
                                (outcome.canonical_fighter_id, outcome.canonical_opponent_id, outcome.orientation_status, stored[0]),
                            )
            finally:
                write_connection.close()
        return {
            "snapshots_examined": len(results),
            "matched": matched_count,
            "unmatched": sum(item.status == "unmatched" for item in results),
            "ambiguous": sum(item.status == "ambiguous" for item in results),
            "rejected": sum(item.status.startswith("rejected_") for item in results),
            "dry_run": dry_run,
        }
