"""Conservative matching from provider records to canonical UFC identities."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

from ufc_pipeline.step6c_upcoming_feature_builder import (
    VALIDATED_FIGHTER_ALIASES,
    normalize_fighter_name,
)

from .domain import NormalizedOutcome, NormalizedSnapshot, parse_utc


MATCH_STATUSES = frozenset({
    "matched",
    "unmatched",
    "ambiguous",
    "rejected_non_ufc",
    "rejected_live_market",
    "rejected_prediction_market",
    "rejected_exchange",
    "rejected_unapproved_sportsbook",
    "rejected_invalid_odds",
    "rejected_incomplete_market",
    "rejected_cancelled_bout",
    "rejected_replacement_mismatch",
    "rejected_duplicate",
})


def _suffixless(key: str) -> str:
    return re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", key).strip()


@dataclass(frozen=True)
class FighterMatch:
    status: str
    fighter_id: int | None
    canonical_name: str | None
    method: str | None
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SnapshotMatch:
    status: str
    snapshot: NormalizedSnapshot
    diagnostics: dict[str, Any]


class CanonicalMatcher:
    """Read-only matcher against ``data/ufc.db`` plus reviewed sidecar mappings."""

    def __init__(self, canonical_db: str | Path, *, sportsbook_db: str | Path | None = None):
        self.canonical_db = Path(canonical_db)
        self.sportsbook_db = Path(sportsbook_db) if sportsbook_db else None
        self._fighters = self._load_fighters()

    def _canonical_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.canonical_db.resolve()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def _load_fighters(self) -> dict[str, list[dict[str, Any]]]:
        connection = self._canonical_connection()
        try:
            rows = connection.execute("SELECT fighter_id, name, normalized_name FROM fighters").fetchall()
        finally:
            connection.close()
        result: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            key = normalize_fighter_name(row["name"])
            result.setdefault(key, []).append({"fighter_id": int(row["fighter_id"]), "name": row["name"]})
        return result

    def _reviewed_mapping(self, provider_name: str, raw_name: str) -> FighterMatch | None:
        if not self.sportsbook_db or not self.sportsbook_db.exists():
            return None
        connection = sqlite3.connect(f"file:{self.sportsbook_db.resolve()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            row = connection.execute(
                """SELECT canonical_identifier FROM provider_mappings
                   WHERE provider_name=? AND entity_type='fighter' AND raw_identifier_normalized=?""",
                (provider_name, normalize_fighter_name(raw_name)),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        finally:
            connection.close()
        if row is None:
            return None
        try:
            fighter_id = int(row["canonical_identifier"])
        except (TypeError, ValueError):
            return FighterMatch("unmatched", None, None, None, {"reason": "reviewed mapping has invalid canonical fighter ID"})
        candidates = [item for values in self._fighters.values() for item in values if item["fighter_id"] == fighter_id]
        if len(candidates) != 1:
            return FighterMatch("unmatched", None, None, None, {"reason": "reviewed mapping references unknown canonical fighter ID"})
        return FighterMatch("matched", fighter_id, candidates[0]["name"], "reviewed_provider_mapping", {"raw_name": raw_name})

    def match_fighter(self, provider_name: str, raw_name: str) -> FighterMatch:
        reviewed = self._reviewed_mapping(provider_name, raw_name)
        if reviewed is not None:
            return reviewed
        input_key = normalize_fighter_name(raw_name)
        alias_key = VALIDATED_FIGHTER_ALIASES.get(input_key, input_key)
        candidates = self._fighters.get(alias_key, [])
        method = "reviewed_repository_alias" if alias_key != input_key else "exact_normalized_name"
        if len(candidates) == 1:
            return FighterMatch("matched", candidates[0]["fighter_id"], candidates[0]["name"], method, {"normalized_name": alias_key})
        if len(candidates) > 1:
            return FighterMatch("ambiguous", None, None, None, {"normalized_name": alias_key, "candidate_ids": [item["fighter_id"] for item in candidates]})
        suffix_key = _suffixless(alias_key)
        if suffix_key != alias_key:
            suffix_candidates = [item for key, values in self._fighters.items() if _suffixless(key) == suffix_key for item in values]
            unique = {item["fighter_id"]: item for item in suffix_candidates}
            if len(unique) == 1:
                candidate = next(iter(unique.values()))
                return FighterMatch("matched", candidate["fighter_id"], candidate["name"], "unique_suffix_normalized_name", {"normalized_name": alias_key, "suffixless_name": suffix_key})
            if len(unique) > 1:
                return FighterMatch("ambiguous", None, None, None, {"suffixless_name": suffix_key, "candidate_ids": sorted(unique)})
        return FighterMatch("unmatched", None, None, None, {"normalized_name": alias_key, "reason": "no exact, reviewed-alias, or unique suffix-normalized match"})

    def _reviewed_bout_mapping(self, provider_name: str, provider_event_id: str) -> int | None:
        if not self.sportsbook_db or not self.sportsbook_db.exists():
            return None
        connection = sqlite3.connect(f"file:{self.sportsbook_db.resolve()}?mode=ro", uri=True)
        try:
            row = connection.execute(
                """SELECT canonical_identifier FROM provider_mappings
                   WHERE provider_name=? AND entity_type='bout' AND raw_identifier_normalized=?""",
                (provider_name, normalize_fighter_name(provider_event_id)),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        finally:
            connection.close()
        return int(row[0]) if row else None

    def match_snapshot(self, snapshot: NormalizedSnapshot) -> SnapshotMatch:
        if snapshot.normalization_status.startswith("rejected_"):
            status = snapshot.normalization_status
            return SnapshotMatch(status, snapshot, {"reason": snapshot.rejection_reason})
        league = (snapshot.league_key or "").strip().casefold()
        if league and league not in {"mma", "ufc", "ultimate fighting championship", "mixed martial arts"} and "ufc" not in league:
            rejected = replace(snapshot, normalization_status="rejected_non_ufc", rejection_reason=f"provider league is not UFC: {snapshot.league_key}")
            return SnapshotMatch("rejected_non_ufc", rejected, {"league_key": snapshot.league_key})
        if snapshot.pre_fight_or_live_status != "pre_fight":
            rejected = replace(snapshot, normalization_status="rejected_live_market", rejection_reason="market is not pre-fight")
            return SnapshotMatch("rejected_live_market", rejected, {"reason": "market is not pre-fight"})
        if len(snapshot.outcomes) != 2:
            rejected = replace(snapshot, normalization_status="rejected_incomplete_market", rejection_reason="exactly two fighter outcomes are required")
            return SnapshotMatch("rejected_incomplete_market", rejected, {"reason": "outcome count"})
        fighter_matches = [self.match_fighter(snapshot.provider_name, outcome.raw_outcome_name) for outcome in snapshot.outcomes]
        if any(item.status == "ambiguous" for item in fighter_matches):
            return SnapshotMatch("ambiguous", snapshot, {"fighter_matches": [item.__dict__ for item in fighter_matches]})
        if any(item.status != "matched" for item in fighter_matches):
            return SnapshotMatch("unmatched", snapshot, {"fighter_matches": [item.__dict__ for item in fighter_matches]})
        fighter_ids = [int(item.fighter_id) for item in fighter_matches if item.fighter_id is not None]
        if fighter_ids[0] == fighter_ids[1]:
            rejected = replace(snapshot, normalization_status="rejected_incomplete_market", rejection_reason="two outcomes resolve to the same fighter")
            return SnapshotMatch("rejected_incomplete_market", rejected, {"fighter_ids": fighter_ids})
        scheduled = parse_utc(snapshot.scheduled_event_time_utc)
        date_candidates = {(scheduled + timedelta(days=offset)).date().isoformat() for offset in (-1, 0, 1)}
        reviewed_bout_id = self._reviewed_bout_mapping(snapshot.provider_name, snapshot.provider_event_id)
        connection = self._canonical_connection()
        try:
            if reviewed_bout_id is not None:
                rows = connection.execute(
                    """SELECT f.fight_id, f.date, f.event, f.fighter_a_id, f.fighter_b_id,
                            fa.name AS fighter_a_name, fb.name AS fighter_b_name
                     FROM fights f
                     JOIN fighters fa ON fa.fighter_id=f.fighter_a_id
                     JOIN fighters fb ON fb.fighter_id=f.fighter_b_id
                     WHERE f.fight_id=?""",
                    (reviewed_bout_id,),
                ).fetchall()
            else:
                placeholders = ",".join("?" for _ in date_candidates)
                rows = connection.execute(
                    f"""SELECT f.fight_id, f.date, f.event, f.fighter_a_id, f.fighter_b_id,
                                fa.name AS fighter_a_name, fb.name AS fighter_b_name
                         FROM fights f
                         JOIN fighters fa ON fa.fighter_id=f.fighter_a_id
                         JOIN fighters fb ON fb.fighter_id=f.fighter_b_id
                         WHERE f.date IN ({placeholders})
                           AND ((f.fighter_a_id=? AND f.fighter_b_id=?) OR (f.fighter_a_id=? AND f.fighter_b_id=?))""",
                    (*sorted(date_candidates), fighter_ids[0], fighter_ids[1], fighter_ids[1], fighter_ids[0]),
                ).fetchall()
        finally:
            connection.close()
        if reviewed_bout_id is not None and rows:
            mapped_pair = {int(rows[0]["fighter_a_id"]), int(rows[0]["fighter_b_id"])}
            if mapped_pair != set(fighter_ids):
                rejected = replace(snapshot, normalization_status="rejected_replacement_mismatch", rejection_reason="reviewed provider bout mapping has a different fighter pair")
                return SnapshotMatch("rejected_replacement_mismatch", rejected, {"mapped_fight_id": reviewed_bout_id, "mapped_pair": sorted(mapped_pair), "provider_pair": sorted(fighter_ids)})
        if not rows:
            return SnapshotMatch("unmatched", snapshot, {"fighter_ids": fighter_ids, "candidate_dates": sorted(date_candidates), "reason": "no canonical bout for fighter pair near scheduled date"})
        exact_date = [row for row in rows if row["date"] == scheduled.date().isoformat()]
        candidates = rows if reviewed_bout_id is not None else (exact_date or rows)
        if len(candidates) != 1:
            return SnapshotMatch("ambiguous", snapshot, {"fighter_ids": fighter_ids, "candidate_fight_ids": [row["fight_id"] for row in candidates], "reason": "repeat matchup cannot be uniquely disambiguated"})
        fight = candidates[0]
        canonical_event_stable = f"{fight['date']}|{fight['event'] or ''}"
        canonical_event_id = "ufc_event_" + hashlib.sha256(canonical_event_stable.encode("utf-8")).hexdigest()[:20]
        matched_outcomes: list[NormalizedOutcome] = []
        for index, outcome in enumerate(snapshot.outcomes):
            fighter_id = fighter_ids[index]
            opponent_id = fighter_ids[1 - index]
            matched_outcomes.append(replace(
                outcome,
                canonical_fighter_id=fighter_id,
                canonical_opponent_id=opponent_id,
                orientation_status="matched",
            ))
        diagnostics = {
            "canonical_fight_id": int(fight["fight_id"]),
            "canonical_event_name": fight["event"],
            "canonical_event_date": fight["date"],
            "fighter_matches": [item.__dict__ for item in fighter_matches],
            "date_tolerance_days": 1,
            "reviewed_bout_mapping": reviewed_bout_id is not None,
        }
        matched = replace(
            snapshot,
            outcomes=tuple(matched_outcomes),
            canonical_event_id=canonical_event_id,
            canonical_bout_id=str(fight["fight_id"]),
            matching_method="unordered_fighter_pair_plus_event_date",
            matching_diagnostics=diagnostics,
        )
        return SnapshotMatch("matched", matched, diagnostics)

    def match_many(self, snapshots: tuple[NormalizedSnapshot, ...]) -> tuple[tuple[NormalizedSnapshot, ...], tuple[dict[str, Any], ...]]:
        matched: list[NormalizedSnapshot] = []
        reviews: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for snapshot in snapshots:
            result = self.match_snapshot(snapshot)
            key = (snapshot.provider_event_id, snapshot.provider_market_id, snapshot.provider_sportsbook_key, snapshot.observed_at_utc)
            if key in seen:
                duplicate = replace(snapshot, normalization_status="rejected_duplicate", rejection_reason="duplicate provider snapshot in payload")
                matched.append(duplicate)
                reviews.append({"status": "rejected_duplicate", "raw_identifier": "|".join(key), "diagnostics": {"reason": "duplicate provider snapshot in payload"}})
                continue
            seen.add(key)
            matched.append(result.snapshot)
            if result.status != "matched":
                reviews.append({"status": result.status, "raw_identifier": "|".join(key), "diagnostics": result.diagnostics})
        return tuple(matched), tuple(reviews)


def import_reviewed_mapping(
    sportsbook_db: str | Path,
    canonical_db: str | Path,
    *,
    provider_name: str,
    entity_type: str,
    raw_identifier: str,
    canonical_identifier: str,
    created_by: str,
    review_note: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Add an explicit reviewed mapping without changing the canonical UFC DB."""
    if entity_type not in {"fighter", "bout"}:
        raise ValueError("Stage 2 mapping imports support entity_type='fighter' or 'bout'")
    canonical_connection = sqlite3.connect(f"file:{Path(canonical_db).resolve()}?mode=ro", uri=True)
    try:
        if entity_type == "fighter":
            found = canonical_connection.execute("SELECT name FROM fighters WHERE fighter_id=?", (int(canonical_identifier),)).fetchone()
        else:
            found = canonical_connection.execute("SELECT event || ' / ' || date FROM fights WHERE fight_id=?", (int(canonical_identifier),)).fetchone()
    finally:
        canonical_connection.close()
    if found is None:
        raise ValueError(f"unknown canonical {entity_type} ID: {canonical_identifier}")
    normalized = normalize_fighter_name(raw_identifier)
    result = {
        "provider_name": provider_name,
        "entity_type": entity_type,
        "raw_identifier_normalized": normalized,
        "canonical_identifier": str(canonical_identifier),
        "canonical_name": found[0],
        "created_by": created_by,
        "dry_run": dry_run,
    }
    if dry_run:
        return result
    connection = sqlite3.connect(sportsbook_db)
    try:
        with connection:
            connection.execute(
                """INSERT INTO provider_mappings(
                    provider_name, entity_type, raw_identifier_normalized,
                    canonical_identifier, mapping_method, created_by, review_note, created_at_utc
                ) VALUES (?, ?, ?, ?, 'reviewed', ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
                (provider_name, entity_type, normalized, str(canonical_identifier), created_by, review_note),
            )
    finally:
        connection.close()
    return result
