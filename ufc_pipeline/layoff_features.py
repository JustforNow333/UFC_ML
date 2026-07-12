"""Leakage-safe inactivity features shared by historical and live builds.

The canonical ``fights`` table contains completed, decisive UFC bouts only;
draws/no-contests are dropped during ingestion because the Elo/model target is
binary.  Consequently these columns mean *time since the last stored decisive
UFC fight*, not time since every professional MMA appearance.

The legacy ``days_since_last_fight_diff`` feature remains untouched for exact
production-model reproduction.  This module implements the versioned
experimental family with a stricter rule: a prior activity date must be
strictly earlier than the current date.  Same-day records never establish
activity order, even though ``fight_id`` remains the repository tie-breaker for
other chronological features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from math import log1p
from typing import Any


SHORT_TURNAROUND_DAYS = 60
LONG_LAYOFF_DAYS = 365
EXTREME_LAYOFF_DAYS = 730

LAYOFF_BASE_FEATURES = [
    "fighter_a_layoff_days",
    "fighter_b_layoff_days",
    "layoff_days_diff",
    "abs_layoff_days_diff",
    "fighter_a_layoff_missing",
    "fighter_b_layoff_missing",
]

LAYOFF_NONLINEAR_FEATURES = [
    "fighter_a_log_layoff_days",
    "fighter_b_log_layoff_days",
    "fighter_a_short_turnaround",
    "fighter_b_short_turnaround",
    "fighter_a_long_layoff",
    "fighter_b_long_layoff",
    "fighter_a_extreme_layoff",
    "fighter_b_extreme_layoff",
]

LAYOFF_FEATURE_COLUMNS = LAYOFF_BASE_FEATURES + LAYOFF_NONLINEAR_FEATURES

# Pre-declared evaluation schemas. Candidate A follows the requested simple
# raw/signed/missing ablation; Candidate B includes the complete 14-column
# family, including the absolute difference.
LAYOFF_CANDIDATE_A_FEATURES = [
    "fighter_a_layoff_days",
    "fighter_b_layoff_days",
    "layoff_days_diff",
    "fighter_a_layoff_missing",
    "fighter_b_layoff_missing",
]
LAYOFF_CANDIDATE_B_FEATURES = list(LAYOFF_FEATURE_COLUMNS)


def parse_fight_date(value: Any) -> date | None:
    """Parse an ISO calendar date; malformed/missing values remain unknown."""
    if value is None:
        return None
    text = str(value).strip()[:10]
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


@dataclass
class LayoffHistory:
    """Completed activity dates for one canonical fighter identity."""

    completed_dates: set[date] = field(default_factory=set)

    def record_completed_fight(self, fight_date: Any) -> bool:
        parsed = parse_fight_date(fight_date)
        if parsed is None:
            return False
        before = len(self.completed_dates)
        self.completed_dates.add(parsed)  # duplicate bouts/dates do not double-count
        return len(self.completed_dates) > before

    def previous_fight_date(self, current_fight_date: Any) -> date | None:
        current = parse_fight_date(current_fight_date)
        if current is None:
            return None
        eligible = (past for past in self.completed_dates if past < current)
        return max(eligible, default=None)

    def layoff_days(self, current_fight_date: Any) -> float | None:
        current = parse_fight_date(current_fight_date)
        previous = self.previous_fight_date(current_fight_date)
        if current is None or previous is None:
            return None
        return float((current - previous).days)


def _side_features(prefix: str, days: float | None) -> dict[str, float | int | None]:
    missing = int(days is None)
    return {
        f"{prefix}_layoff_days": days,
        f"{prefix}_layoff_missing": missing,
        f"{prefix}_log_layoff_days": None if days is None else float(log1p(days)),
        f"{prefix}_short_turnaround": None if days is None else int(days < SHORT_TURNAROUND_DAYS),
        f"{prefix}_long_layoff": None if days is None else int(days >= LONG_LAYOFF_DAYS),
        f"{prefix}_extreme_layoff": None if days is None else int(days >= EXTREME_LAYOFF_DAYS),
    }


def build_layoff_feature_family(
    fighter_a_history: LayoffHistory,
    fighter_b_history: LayoffHistory,
    current_fight_date: Any,
) -> dict[str, float | int | None]:
    """Return the complete A/B layoff family from strictly earlier dates."""
    days_a = fighter_a_history.layoff_days(current_fight_date)
    days_b = fighter_b_history.layoff_days(current_fight_date)
    signed = None if days_a is None or days_b is None else days_a - days_b
    features = {
        **_side_features("fighter_a", days_a),
        **_side_features("fighter_b", days_b),
        "layoff_days_diff": signed,
        "abs_layoff_days_diff": None if signed is None else abs(signed),
    }
    # Keep one stable output order regardless of dict construction order.
    return {column: features[column] for column in LAYOFF_FEATURE_COLUMNS}


def counts_as_completed_activity(fight: dict[str, Any]) -> bool:
    """Whether a row may update inactivity history after its feature row.

    Canonical DB rows have no status field and are completed by schema. The
    synthetic Step 6C rows are explicitly excluded. Optional status metadata is
    honored for tests/future sources, with completed no-contests counted as
    physical activity if such rows are ever supplied upstream.
    """
    if fight.get("_synthetic"):
        return False
    if parse_fight_date(fight.get("date")) is None:
        return False
    if fight.get("is_completed") is False:
        return False
    status = str(fight.get("fight_status") or fight.get("status") or "").strip().casefold()
    if status in {"cancelled", "canceled", "scheduled", "postponed", "replaced", "pending"}:
        return False
    return True
