"""Timestamp-safe per-book selection, de-vigging, and sportsbook consensus."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import SportsbookConfig
from .domain import parse_utc, utc_iso
from .odds import proportional_devig
from .registry import sportsbook_by_key


@dataclass(frozen=True)
class SelectedBookProbability:
    sportsbook_key: str
    snapshot_id: str
    observed_at_utc: str
    snapshot_age_hours: float
    fair_fighter_probability: float
    fair_opponent_probability: float
    raw_fighter_probability: float
    raw_opponent_probability: float
    overround: float
    is_sharp_reference: bool


@dataclass(frozen=True)
class ConsensusResult:
    canonical_bout_id: str
    canonical_fighter_id: int
    canonical_opponent_id: int
    prediction_as_of_utc: str
    consensus_fighter_probability: float | None
    consensus_opponent_probability: float | None
    median_fair_probability: float | None
    mean_fair_probability: float | None
    minimum_fair_probability: float | None
    maximum_fair_probability: float | None
    probability_range: float | None
    probability_std: float | None
    sportsbook_count: int
    included_sportsbook_keys: tuple[str, ...]
    excluded_sportsbooks: dict[str, tuple[str, ...]]
    oldest_selected_observation: str | None
    newest_selected_observation: str | None
    maximum_snapshot_age_hours: float | None
    coverage_low: bool
    data_available: bool
    pinnacle_available: bool
    pinnacle_fair_probability: float | None
    pinnacle_minus_broad_consensus: float | None
    pinnacle_included_in_broad_consensus: bool
    selected_books: tuple[SelectedBookProbability, ...]
    consensus_policy_version: str
    selection_policy_version: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConsensusBuilder:
    def __init__(self, database_path: str | Path, *, config: SportsbookConfig | None = None):
        self.database_path = Path(database_path)
        self.config = config or SportsbookConfig(database_path=self.database_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.database_path.resolve()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def build(
        self,
        *,
        canonical_bout_id: str,
        canonical_fighter_id: int,
        canonical_opponent_id: int,
        prediction_as_of_utc: str | datetime,
        eligible_sportsbook_keys: set[str] | None = None,
        max_snapshot_age_hours: float | None = None,
        include_pinnacle_in_broad_consensus: bool | None = None,
        persist_selection: bool = False,
    ) -> ConsensusResult:
        cutoff = parse_utc(prediction_as_of_utc)
        cutoff_iso = utc_iso(cutoff)
        max_age = self.config.max_snapshot_age_hours if max_snapshot_age_hours is None else max_snapshot_age_hours
        include_pinnacle = self.config.include_pinnacle_in_broad_consensus if include_pinnacle_in_broad_consensus is None else include_pinnacle_in_broad_consensus
        connection = self._connect()
        try:
            rows = connection.execute(
                """SELECT s.*, o.outcome_id, o.raw_outcome_name, o.decimal_odds,
                          o.raw_implied_probability, o.canonical_fighter_id,
                          o.canonical_opponent_id, o.orientation_status,
                          o.validation_status
                   FROM market_snapshots s
                   JOIN outcomes o ON o.snapshot_id=s.snapshot_id
                   WHERE s.canonical_bout_id=?
                   ORDER BY s.observed_at_utc, s.snapshot_id, o.outcome_id""",
                (str(canonical_bout_id),),
            ).fetchall()
        finally:
            connection.close()
        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(row["snapshot_id"], []).append(row)
        eligible_by_book: dict[str, list[SelectedBookProbability]] = {}
        excluded: dict[str, set[str]] = {}

        def exclude(book: str, reason: str) -> None:
            excluded.setdefault(book or "unknown", set()).add(reason)

        for snapshot_id, outcome_rows in grouped.items():
            first = outcome_rows[0]
            book = first["sportsbook_key"] or first["provider_sportsbook_key"] or "unknown"
            if first["venue_type"] != "sportsbook":
                exclude(book, f"venue_type_{first['venue_type']}_prohibited")
                continue
            if not first["sportsbook_key"]:
                exclude(book, "unapproved_or_unknown_sportsbook")
                continue
            definition = sportsbook_by_key(first["sportsbook_key"])
            approved = definition.approved or book in self.config.approved_sportsbook_overrides
            if not approved:
                exclude(book, "unapproved_sportsbook")
                continue
            if eligible_sportsbook_keys is not None and book not in eligible_sportsbook_keys:
                exclude(book, "not_in_requested_eligible_set")
                continue
            if first["normalization_status"] != "normalized":
                exclude(book, first["normalization_status"])
                continue
            if first["match_status"] != "matched":
                exclude(book, f"match_status_{first['match_status']}")
                continue
            if first["market_type"] != "h2h":
                exclude(book, "unsupported_market")
                continue
            if first["pre_fight_or_live_status"] != "pre_fight":
                exclude(book, "live_or_post_start")
                continue
            observed = parse_utc(first["observed_at_utc"])
            scheduled = parse_utc(first["scheduled_event_time_utc"])
            if observed > cutoff:
                exclude(book, "observation_after_prediction_cutoff")
                continue
            if observed >= scheduled:
                exclude(book, "observation_at_or_after_scheduled_start")
                continue
            age_hours = (cutoff - observed).total_seconds() / 3600.0
            if max_age is not None and age_hours > max_age:
                exclude(book, "snapshot_stale")
                continue
            if len(outcome_rows) != 2:
                exclude(book, "incomplete_or_multiway_market")
                continue
            target = [row for row in outcome_rows if row["canonical_fighter_id"] == canonical_fighter_id and row["canonical_opponent_id"] == canonical_opponent_id]
            opponent = [row for row in outcome_rows if row["canonical_fighter_id"] == canonical_opponent_id and row["canonical_opponent_id"] == canonical_fighter_id]
            if len(target) != 1 or len(opponent) != 1:
                exclude(book, "fighter_orientation_mismatch")
                continue
            if any(row["validation_status"] != "valid" or row["orientation_status"] != "matched" for row in outcome_rows):
                exclude(book, "invalid_outcome")
                continue
            if target[0]["decimal_odds"] is None or opponent[0]["decimal_odds"] is None:
                exclude(book, "invalid_odds")
                continue
            result = proportional_devig(target[0]["decimal_odds"], opponent[0]["decimal_odds"])
            selected = SelectedBookProbability(
                sportsbook_key=book,
                snapshot_id=snapshot_id,
                observed_at_utc=utc_iso(observed),
                snapshot_age_hours=age_hours,
                fair_fighter_probability=result.fair_probability_a,
                fair_opponent_probability=result.fair_probability_b,
                raw_fighter_probability=result.raw_probability_a,
                raw_opponent_probability=result.raw_probability_b,
                overround=result.overround,
                is_sharp_reference=definition.sharp_reference,
            )
            eligible_by_book.setdefault(book, []).append(selected)

        selected_by_book = {
            book: max(items, key=lambda item: (parse_utc(item.observed_at_utc), item.snapshot_id))
            for book, items in eligible_by_book.items()
        }
        pinnacle = selected_by_book.get("pinnacle")
        broad: list[SelectedBookProbability] = []
        for book, item in sorted(selected_by_book.items()):
            definition = sportsbook_by_key(book)
            broad_eligible = definition.consensus_eligible or book in self.config.additional_consensus_sportsbook_keys
            if broad_eligible or (book == "pinnacle" and include_pinnacle):
                broad.append(item)
            elif book != "pinnacle":
                exclude(book, "not_broad_consensus_eligible")
        values = [item.fair_fighter_probability for item in broad]
        if values:
            median = float(statistics.median(values))
            mean = float(statistics.fmean(values))
            minimum = min(values)
            maximum = max(values)
            probability_range = maximum - minimum
            std = float(statistics.pstdev(values))
            oldest = min(item.observed_at_utc for item in broad)
            newest = max(item.observed_at_utc for item in broad)
            max_selected_age = max(item.snapshot_age_hours for item in broad)
        else:
            median = mean = minimum = maximum = probability_range = std = None
            oldest = newest = None
            max_selected_age = None
        result = ConsensusResult(
            canonical_bout_id=str(canonical_bout_id),
            canonical_fighter_id=int(canonical_fighter_id),
            canonical_opponent_id=int(canonical_opponent_id),
            prediction_as_of_utc=cutoff_iso,
            consensus_fighter_probability=median,
            consensus_opponent_probability=(1.0 - median) if median is not None else None,
            median_fair_probability=median,
            mean_fair_probability=mean,
            minimum_fair_probability=minimum,
            maximum_fair_probability=maximum,
            probability_range=probability_range,
            probability_std=std,
            sportsbook_count=len(broad),
            included_sportsbook_keys=tuple(item.sportsbook_key for item in broad),
            excluded_sportsbooks={key: tuple(sorted(reasons)) for key, reasons in sorted(excluded.items())},
            oldest_selected_observation=oldest,
            newest_selected_observation=newest,
            maximum_snapshot_age_hours=max_selected_age,
            coverage_low=len(broad) < self.config.minimum_normal_coverage,
            data_available=bool(broad),
            pinnacle_available=pinnacle is not None,
            pinnacle_fair_probability=pinnacle.fair_fighter_probability if pinnacle else None,
            pinnacle_minus_broad_consensus=(pinnacle.fair_fighter_probability - median) if pinnacle and median is not None and not include_pinnacle else None,
            pinnacle_included_in_broad_consensus=bool(pinnacle and include_pinnacle),
            selected_books=tuple(broad + ([pinnacle] if pinnacle is not None and pinnacle not in broad else [])),
            consensus_policy_version=self.config.consensus_policy_version,
            selection_policy_version=self.config.selection_policy_version,
        )
        if persist_selection:
            self._persist(result)
        return result

    def _persist(self, result: ConsensusResult) -> None:
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys=ON")
        created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        try:
            with connection:
                for item in result.selected_books:
                    stable = "|".join((result.canonical_bout_id, str(result.canonical_fighter_id), result.prediction_as_of_utc, item.sportsbook_key, result.selection_policy_version))
                    selection_id = "sel_" + hashlib.sha256(stable.encode("utf-8")).hexdigest()[:32]
                    connection.execute(
                        """INSERT OR IGNORE INTO selected_consensus_inputs(
                            selection_id, canonical_bout_id, canonical_fighter_id,
                            canonical_opponent_id, prediction_as_of_utc, sportsbook_key,
                            selected_snapshot_id, selected_observed_at_utc, snapshot_age_hours,
                            fair_fighter_probability, fair_opponent_probability, overround,
                            devig_method, devig_version, selection_policy_version, created_at_utc
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            selection_id, result.canonical_bout_id, result.canonical_fighter_id,
                            result.canonical_opponent_id, result.prediction_as_of_utc,
                            item.sportsbook_key, item.snapshot_id, item.observed_at_utc,
                            item.snapshot_age_hours, item.fair_fighter_probability,
                            item.fair_opponent_probability, item.overround,
                            self.config.devig_method, self.config.devig_version,
                            result.selection_policy_version, created,
                        ),
                    )
        finally:
            connection.close()
