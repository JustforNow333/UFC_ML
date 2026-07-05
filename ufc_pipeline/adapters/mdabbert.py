"""Adapter for the mdabbert "Ultimate UFC Dataset" (Kaggle).

Known quirks this adapter handles:

* Column names vary across versions: RedFighter/BlueFighter, R_fighter/B_fighter,
  red_fighter/blue_fighter, Date/date, etc. -> handled by fuzzy column mapping.
* The Winner column usually contains the corner ("Red"/"Blue"), not a name.
  We convert corner labels into the actual fighter's name.
* The dataset ships MANY precomputed/odds/rank columns. Everything except the
  basic identity + result fields is intentionally DROPPED here so that no
  leaky column can ever reach the Elo layer by accident.

Red corner -> fighter_a, Blue corner -> fighter_b (per project convention).

If a future source only has winner/loser columns (no corners), we convert it
deterministically: even source rows put the winner in the A slot, odd rows put
the winner in the B slot. This alternation is index-based (no randomness), so
output is fully reproducible and fighter_a is NOT systematically the winner.
"""

from __future__ import annotations

import pandas as pd

from ..columns import build_column_mapping, clean_display_name, missing_required_fields

_CORNER_A_LABELS = {"red", "r", "fighter_a", "a", "1"}
_CORNER_B_LABELS = {"blue", "b", "fighter_b", "b", "2"}

_OPTIONAL_FIELDS = ["event", "weight_class", "method", "round"]

# Per-fight tale-of-the-tape values. These are AS-OF-FIGHT numbers (age at
# fight time, listed height/reach for that bout), i.e. known before the
# fight -> safe to keep. Anything like averages/odds/rankings is still
# dropped because it never appears in COLUMN_CANDIDATES.
_PHYSICAL_FIELDS = [
    "fighter_a_age", "fighter_b_age",
    "fighter_a_height", "fighter_b_height",
    "fighter_a_reach", "fighter_b_reach",
]


def adapt(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw mdabbert DataFrame into internal fight columns."""
    mapping = build_column_mapping(df)

    # Winner/loser-only fallback (not typical for mdabbert, but supported
    # so the same adapter logic can serve simple mirrors of the data).
    winner_loser_mode = (
        "fighter_a_name" not in mapping
        and "_winner_fighter" in mapping
        and "_loser_fighter" in mapping
    )
    if winner_loser_mode:
        return _adapt_winner_loser(df, mapping)

    missing = missing_required_fields(mapping)
    has_winner_info = "winner_name" in mapping
    if missing or not has_winner_info:
        wanted = missing + ([] if has_winner_info else ["winner_name"])
        raise ValueError(
            "Input CSV is missing required fields.\n"
            f"  Missing normalized fields : {wanted}\n"
            f"  Columns found in the file : {list(df.columns)}\n"
            "  Expected column names include e.g. "
            "date/Date, RedFighter/red_fighter/R_fighter/fighter_a, "
            "BlueFighter/blue_fighter/B_fighter/fighter_b, "
            "Winner/winner/winner_name."
        )

    out = pd.DataFrame(index=df.index)
    out["date"] = df[mapping["date"]]
    out["fighter_a_name"] = df[mapping["fighter_a_name"]].map(clean_display_name)
    out["fighter_b_name"] = df[mapping["fighter_b_name"]].map(clean_display_name)

    # --- Winner: corner label ("Red"/"Blue") or an actual fighter name ---
    raw_winner = df[mapping["winner_name"]].map(clean_display_name)
    winner_lower = raw_winner.str.lower()
    is_a = winner_lower.isin(_CORNER_A_LABELS)
    is_b = winner_lower.isin(_CORNER_B_LABELS)
    out["winner_name"] = raw_winner
    out.loc[is_a, "winner_name"] = out.loc[is_a, "fighter_a_name"]
    out.loc[is_b, "winner_name"] = out.loc[is_b, "fighter_b_name"]
    # Draws / no-contests / unknown labels (e.g. "Draw", "") -> no winner.
    unresolved = ~(is_a | is_b) & ~winner_lower.isin(
        out["fighter_a_name"].str.lower()
    ) & ~winner_lower.isin(out["fighter_b_name"].str.lower())
    # Row-wise check is clearer for the name-match case:
    matches_a = winner_lower == out["fighter_a_name"].str.lower()
    matches_b = winner_lower == out["fighter_b_name"].str.lower()
    resolved = is_a | is_b | matches_a | matches_b
    out.loc[~resolved, "winner_name"] = None

    for field in _OPTIONAL_FIELDS:
        if field in mapping:
            out[field] = df[mapping[field]]
    for field in _PHYSICAL_FIELDS:
        if field in mapping:
            out[field] = pd.to_numeric(df[mapping[field]], errors="coerce")

    out["source_row_id"] = df.index.astype(int)
    return out


def _adapt_winner_loser(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Convert winner/loser-shaped data into fighter_a/fighter_b format.

    Deterministic alternation by source row index (documented convention):
      even row index -> winner is fighter_a
      odd  row index -> winner is fighter_b
    so fighter_a is NOT always the winner and results are reproducible.
    """
    if "date" not in mapping:
        raise ValueError(
            "Input CSV is missing required fields.\n"
            "  Missing normalized fields : ['date']\n"
            f"  Columns found in the file : {list(df.columns)}"
        )

    winners = df[mapping["_winner_fighter"]].map(clean_display_name)
    losers = df[mapping["_loser_fighter"]].map(clean_display_name)
    even = pd.Series(df.index, index=df.index).astype(int) % 2 == 0

    out = pd.DataFrame(index=df.index)
    out["date"] = df[mapping["date"]]
    out["fighter_a_name"] = winners.where(even, losers)
    out["fighter_b_name"] = losers.where(even, winners)
    out["winner_name"] = winners

    for field in _OPTIONAL_FIELDS:
        if field in mapping:
            out[field] = df[mapping[field]]

    out["source_row_id"] = df.index.astype(int)
    return out
