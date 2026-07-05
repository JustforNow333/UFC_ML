"""Column-mapping utilities shared by all source adapters.

Every adapter's job is to turn a raw source DataFrame into a DataFrame with
these NORMALIZED internal columns (only the first four are strictly required):

    date            - parseable date of the fight
    event           - event name (optional)
    fighter_a_name  - "red" corner when the source has red/blue
    fighter_b_name  - "blue" corner when the source has red/blue
    winner_name     - full name of the winner (or NaN for draw/NC)
    fighter_a_won   - 1 if fighter_a won, 0 if fighter_b won
    weight_class    - optional
    method          - optional (result/metadata only, never a pre-fight feature)
    round           - optional (result/metadata only, never a pre-fight feature)
    source_row_id   - row index in the raw source file, for traceability

The rest of the pipeline only ever sees these names, so adding a new source
(e.g. Greco1899/scrape_ufc_stats) later means writing one new adapter and
nothing else.
"""

from __future__ import annotations

import re

import pandas as pd

# Candidate raw column names for each normalized field, checked in order.
# Matching is case-insensitive and ignores spaces/underscores, so
# "RedFighter", "red_fighter", and "Red Fighter" all match "redfighter".
COLUMN_CANDIDATES: dict[str, list[str]] = {
    "date": ["date", "event_date", "fight_date"],
    "event": ["event", "event_name", "eventname"],
    "fighter_a_name": [
        "red_fighter", "redfighter", "r_fighter", "fighter_a",
        "fighter_red", "red_corner", "fighter1", "fighter_1",
    ],
    "fighter_b_name": [
        "blue_fighter", "bluefighter", "b_fighter", "fighter_b",
        "fighter_blue", "blue_corner", "fighter2", "fighter_2",
    ],
    "winner_name": ["winner", "winner_name", "result_winner", "result"],
    "weight_class": ["weight_class", "weightclass", "division", "weight class"],
    "method": ["method", "finish", "win_by", "finish_details_method", "victory_method"],
    "round": ["round", "finish_round", "finishround", "last_round", "total_rounds_fought"],
    # Winner/loser style sources (no red/blue corners):
    "_winner_fighter": ["winner_fighter", "winnername", "winner_full_name"],
    "_loser_fighter": ["loser_fighter", "loser", "loser_name", "losername"],
    # Per-fight tale-of-the-tape values (as-of-fight, therefore time-safe).
    # mdabbert exposes RedAge/BlueAge, RedHeightCms/..., RedReachCms/...
    "fighter_a_age": ["red_age", "r_age", "fighter_a_age"],
    "fighter_b_age": ["blue_age", "b_age", "fighter_b_age"],
    "fighter_a_height": ["red_height_cms", "r_height_cms", "red_height", "fighter_a_height"],
    "fighter_b_height": ["blue_height_cms", "b_height_cms", "blue_height", "fighter_b_height"],
    "fighter_a_reach": ["red_reach_cms", "r_reach_cms", "red_reach", "fighter_a_reach"],
    "fighter_b_reach": ["blue_reach_cms", "b_reach_cms", "blue_reach", "fighter_b_reach"],
}

_REQUIRED_FIELDS = ["date", "fighter_a_name", "fighter_b_name"]
# winner information may arrive as winner_name OR as fighter_a_won;
# validation checks for at least one of them after adaptation.


def _canon(col: str) -> str:
    """Canonical form of a raw column name for fuzzy matching."""
    return re.sub(r"[\s_]+", "", str(col).strip().lower())


def find_column(df: pd.DataFrame, field: str) -> str | None:
    """Return the raw column in `df` that maps to normalized `field`, if any."""
    canon_map = {_canon(c): c for c in df.columns}
    for candidate in COLUMN_CANDIDATES.get(field, []):
        hit = canon_map.get(_canon(candidate))
        if hit is not None:
            return hit
    return None


def build_column_mapping(df: pd.DataFrame) -> dict[str, str]:
    """Map normalized field -> raw column name for every field we can find."""
    mapping = {}
    for field in COLUMN_CANDIDATES:
        raw = find_column(df, field)
        if raw is not None:
            mapping[field] = raw
    return mapping


def missing_required_fields(mapping: dict[str, str]) -> list[str]:
    return [f for f in _REQUIRED_FIELDS if f not in mapping]


# ---------------------------------------------------------------------------
# Fighter name normalization
# ---------------------------------------------------------------------------

def clean_display_name(name: str) -> str:
    """Tidy a fighter name for display: trim + collapse repeated whitespace.

    Deliberately conservative: we do NOT lowercase, strip accents, or drop
    punctuation here, because that risks merging genuinely different fighters.
    """
    if name is None:
        return ""
    try:
        if pd.isna(name):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(name)).strip()


def name_key(name: str) -> str:
    """Internal matching key: lowercase of the cleaned display name.

    Kept intentionally minimal (case-insensitive match only) to avoid
    accidentally merging different fighters with similar names. Anything
    fancier (accent folding, nickname handling) is a conscious future step.
    """
    return clean_display_name(name).lower()
