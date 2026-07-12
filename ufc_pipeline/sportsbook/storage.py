"""Immutable raw archive and transactional SQLite sportsbook storage."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import SportsbookConfig
from .domain import NormalizedSnapshot, utc_iso
from .schema import SCHEMA_VERSION, initialize_schema


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class ArchiveResult:
    provider_name: str
    payload_hash: str
    archive_path: Path
    byte_count: int
    created: bool
    dry_run: bool = False


class SportsbookStore:
    def __init__(self, config: SportsbookConfig | None = None):
        self.config = config or SportsbookConfig()

    def connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        path = self.config.database_path
        if read_only:
            connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self, *, dry_run: bool = False) -> list[str]:
        if dry_run:
            connection = sqlite3.connect(":memory:")
        else:
            connection = self.connect()
        try:
            return initialize_schema(connection, dry_run=dry_run)
        finally:
            connection.close()

    def archive_payload(self, provider_name: str, payload: Any, *, dry_run: bool = False) -> ArchiveResult:
        provider = provider_name.strip().casefold()
        if not provider or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in provider):
            raise ValueError("provider_name must be a safe lowercase identifier")
        data = canonical_json_bytes(payload)
        digest = hashlib.sha256(data).hexdigest()
        destination = self.config.raw_archive_root / provider / f"{digest}.json"
        if dry_run:
            return ArchiveResult(provider, digest, destination, len(data), destination.exists(), True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = destination.read_bytes()
            if existing != data:
                raise RuntimeError(f"content-addressed archive collision at {destination}")
            created = False
        else:
            fd, temporary_name = tempfile.mkstemp(prefix=f".{digest}.", suffix=".tmp", dir=destination.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, destination)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
            created = True
        self.initialize()
        connection = self.connect()
        try:
            with connection:
                connection.execute(
                    """INSERT OR IGNORE INTO raw_archives(
                        raw_payload_hash, provider_name, archive_path, byte_count, canonical_json, created_at_utc
                    ) VALUES (?, ?, ?, ?, 1, ?)""",
                    (digest, provider, str(destination), len(data), _now_utc()),
                )
        finally:
            connection.close()
        return ArchiveResult(provider, digest, destination, len(data), created)

    def create_ingestion_batch(
        self,
        *,
        provider_name: str,
        request_type: str,
        archive: ArchiveResult,
        query_timestamp_utc: str | None = None,
        fetched_at_utc: str | None = None,
        source_file: str | None = None,
        response_status: str = "archived",
        record_count: int = 0,
        error_message: str | None = None,
        quota_metadata: dict[str, Any] | None = None,
    ) -> str:
        fetched = utc_iso(fetched_at_utc or _now_utc())
        query = utc_iso(query_timestamp_utc) if query_timestamp_utc else None
        # The content hash already captures a historical wrapper's query
        # timestamp. Omitting caller-supplied metadata keeps re-ingestion of the
        # same raw payload idempotent even if one caller omitted --query-time.
        stable = "|".join((provider_name, request_type, archive.payload_hash))
        batch_id = "ing_" + hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]
        self.initialize()
        connection = self.connect()
        try:
            with connection:
                connection.execute(
                    """INSERT OR IGNORE INTO provider_ingestion_batches(
                        ingestion_batch_id, provider_name, request_type, query_timestamp_utc,
                        fetched_at_utc, source_file, raw_payload_hash, archive_path, response_status,
                        record_count, error_message, quota_metadata_json, schema_version, created_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        batch_id, provider_name, request_type, query, fetched, source_file,
                        archive.payload_hash, str(archive.archive_path), response_status, record_count,
                        error_message, json.dumps(quota_metadata or {}, sort_keys=True), SCHEMA_VERSION, _now_utc(),
                    ),
                )
        finally:
            connection.close()
        return batch_id

    def store_snapshots(self, ingestion_batch_id: str, snapshots: Iterable[NormalizedSnapshot]) -> dict[str, int]:
        """Atomically store normalized snapshots; duplicates are ignored."""
        self.initialize()
        connection = self.connect()
        inserted_snapshots = 0
        inserted_outcomes = 0
        try:
            with connection:
                batch = connection.execute(
                    "SELECT raw_payload_hash FROM provider_ingestion_batches WHERE ingestion_batch_id = ?",
                    (ingestion_batch_id,),
                ).fetchone()
                if batch is None:
                    raise KeyError(f"unknown ingestion batch: {ingestion_batch_id}")
                for snapshot in snapshots:
                    event_cursor = connection.execute(
                        """INSERT OR IGNORE INTO normalized_events(
                            ingestion_batch_id, provider_name, provider_event_id, raw_event_name,
                            scheduled_event_time_utc, scheduled_event_time_original, sport_key,
                            league_key, canonical_event_id, match_status, matching_method,
                            matching_diagnostics_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            ingestion_batch_id, snapshot.provider_name, snapshot.provider_event_id,
                            snapshot.raw_event_name, snapshot.scheduled_event_time_utc,
                            snapshot.scheduled_event_time_original, snapshot.sport_key, snapshot.league_key,
                            snapshot.canonical_event_id, "matched" if snapshot.canonical_event_id else "unmatched",
                            snapshot.matching_method, json.dumps(snapshot.matching_diagnostics, sort_keys=True),
                        ),
                    )
                    event = connection.execute(
                        "SELECT normalized_event_id FROM normalized_events WHERE ingestion_batch_id=? AND provider_name=? AND provider_event_id=?",
                        (ingestion_batch_id, snapshot.provider_name, snapshot.provider_event_id),
                    ).fetchone()
                    assert event is not None
                    snapshot_stable = "|".join((snapshot.provider_name, snapshot.provider_event_id, snapshot.provider_market_id, snapshot.provider_sportsbook_key, snapshot.observed_at_utc))
                    snapshot_id = "snap_" + hashlib.sha256(snapshot_stable.encode("utf-8")).hexdigest()[:32]
                    cursor = connection.execute(
                        """INSERT OR IGNORE INTO market_snapshots(
                            snapshot_id, normalized_event_id, ingestion_batch_id, provider_name,
                            provider_event_id, provider_market_id, sportsbook_key,
                            provider_sportsbook_key, sportsbook_display_name, venue_type, market_type,
                            observed_at_utc, observed_at_original, scheduled_event_time_utc,
                            pre_fight_or_live_status, canonical_event_id, canonical_bout_id,
                            normalization_status, match_status, matching_method,
                            matching_diagnostics_json, rejection_reason, source_payload_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            snapshot_id, event[0], ingestion_batch_id, snapshot.provider_name,
                            snapshot.provider_event_id, snapshot.provider_market_id, snapshot.sportsbook_key,
                            snapshot.provider_sportsbook_key, snapshot.sportsbook_display_name,
                            snapshot.venue_type, snapshot.market_type, snapshot.observed_at_utc,
                            snapshot.observed_at_original, snapshot.scheduled_event_time_utc,
                            snapshot.pre_fight_or_live_status, snapshot.canonical_event_id,
                            snapshot.canonical_bout_id, snapshot.normalization_status,
                            "matched" if snapshot.canonical_bout_id else "unmatched", snapshot.matching_method,
                            json.dumps(snapshot.matching_diagnostics, sort_keys=True), snapshot.rejection_reason,
                            snapshot.source_payload_hash or batch[0],
                        ),
                    )
                    if cursor.rowcount == 0:
                        continue
                    inserted_snapshots += 1
                    for index, outcome in enumerate(snapshot.outcomes):
                        outcome_id = "out_" + hashlib.sha256(f"{snapshot_id}|{index}|{outcome.raw_outcome_name}".encode("utf-8")).hexdigest()[:32]
                        connection.execute(
                            """INSERT INTO outcomes(
                                outcome_id, snapshot_id, raw_outcome_name, raw_opponent_name,
                                decimal_odds, american_odds, raw_implied_probability,
                                original_odds_json, original_odds_format, canonical_fighter_id,
                                canonical_opponent_id, orientation_status, validation_status,
                                validation_reason
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                outcome_id, snapshot_id, outcome.raw_outcome_name, outcome.raw_opponent_name,
                                outcome.decimal_odds, outcome.american_odds, outcome.raw_implied_probability,
                                json.dumps(outcome.original_odds, sort_keys=True), outcome.original_odds_format,
                                outcome.canonical_fighter_id, outcome.canonical_opponent_id,
                                outcome.orientation_status, outcome.validation_status, outcome.validation_reason,
                            ),
                        )
                        inserted_outcomes += 1
                connection.execute(
                    "UPDATE provider_ingestion_batches SET response_status=?, record_count=? WHERE ingestion_batch_id=?",
                    ("normalized", inserted_snapshots, ingestion_batch_id),
                )
        finally:
            connection.close()
        return {"snapshots_inserted": inserted_snapshots, "outcomes_inserted": inserted_outcomes}

    def health(self) -> dict[str, Any]:
        connection = self.connect(read_only=True)
        try:
            tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            counts = {table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in tables}
            version = connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
            return {"database_path": str(self.config.database_path), "schema_version": version, "tables": counts, "foreign_keys": bool(connection.execute("PRAGMA foreign_keys").fetchone()[0])}
        finally:
            connection.close()

    def record_match_reviews(self, ingestion_batch_id: str, reviews: Iterable[dict[str, Any]]) -> int:
        self.initialize()
        connection = self.connect()
        inserted = 0
        try:
            with connection:
                for review in reviews:
                    connection.execute(
                        """INSERT INTO match_reviews(
                            ingestion_batch_id, entity_type, raw_identifier,
                            candidate_identifiers_json, status, matching_method,
                            score, diagnostics_json, review_note, reviewed,
                            reviewed_at_utc, created_at_utc
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            ingestion_batch_id, review.get("entity_type", "snapshot"),
                            review.get("raw_identifier", "unknown"),
                            json.dumps(review.get("candidate_identifiers", []), sort_keys=True),
                            review["status"], review.get("matching_method"), review.get("score"),
                            json.dumps(review.get("diagnostics", {}), sort_keys=True),
                            review.get("review_note"), int(bool(review.get("reviewed", False))),
                            review.get("reviewed_at_utc"), _now_utc(),
                        ),
                    )
                    inserted += 1
        finally:
            connection.close()
        return inserted
