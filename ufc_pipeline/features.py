"""Time-safe pre-fight feature builder (Part 3).

For each fight, every feature is computed from information available BEFORE
that fight. The loop discipline mirrors the Elo engine:

    for each fight in chronological order:
        1. read fighter A's historical state   (past fights only)
        2. read fighter B's historical state   (past fights only)
        3. build the feature row (+ fighter_a_won as the target)
        4. ONLY THEN update both fighters' histories with this result

Fights are sorted by (date, fight_id) inside this module, so even a caller
that loads them out of order cannot corrupt the timeline. fight_id encodes
the original dataset order for same-date tie-breaking.

Missing-value policy (explicit, no silent filling):
  * prior_win_pct on a debut (prior_fights == 0)  -> null  (chosen over 0.5:
    "no information" should look like no information; impute at model time)
  * win_pct_last_3 / win_pct_last_5 on a debut    -> null
  * days_since_last_fight on a debut              -> null  (imputed later,
    never guessed here)
  * age/height/reach when the source lacks them   -> null
  * any diff involving a null                     -> null

Age: we only use per-fight, as-of-fight-date values.
  1. Preferred: the source's per-fight age (mdabbert RedAge/BlueAge), which
     is the fighter's age at that event -> time-safe by construction.
  2. Fallback: computed from fighters.date_of_birth at the fight date.
  3. Otherwise null. A "current age" scraped once for the whole career is
     NEVER used, because it leaks future information into old fights.

Forbidden here (enforced by the same leakage guard used by the Elo export):
odds, rankings, winner_*/loser_* features, current-fight strikes/takedowns/
control time, post-fight totals, precomputed career averages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_cls
from pathlib import Path

import pandas as pd

from .db import connect, init_schema
from .export import find_forbidden_columns
from .layoff_features import (
    LAYOFF_FEATURE_COLUMNS,
    LayoffHistory,
    build_layoff_feature_family,
    counts_as_completed_activity,
)


# ---------------------------------------------------------------------------
# Per-fighter historical state
# ---------------------------------------------------------------------------

@dataclass
class FighterHistory:
    """Everything known about a fighter from their PAST fights only."""

    results: list[int] = field(default_factory=list)  # 1 win / 0 loss, chrono
    last_fight_date: str | None = None                # ISO date of last bout

    # ---- reads (pre-fight state) ----
    def prior_record(self) -> dict:
        fights = len(self.results)
        wins = sum(self.results)
        losses = fights - wins
        win_pct = (wins / fights) if fights > 0 else None  # debut -> null
        return {
            "prior_fights": fights,
            "prior_wins": wins,
            "prior_losses": losses,
            "prior_win_pct": win_pct,
        }

    def recent_form(self, n: int) -> dict:
        """Wins in the previous `n` fights; pct over however many exist (<= n)."""
        window = self.results[-n:]
        wins = sum(window)
        pct = (wins / len(window)) if window else None  # debut -> null
        return {f"wins_last_{n}": wins, f"win_pct_last_{n}": pct}

    def days_since_last_fight(self, fight_date: str) -> float | None:
        """Days between this fight and the PREVIOUS one; null on debut."""
        if self.last_fight_date is None:
            return None
        return float(
            (date_cls.fromisoformat(fight_date)
             - date_cls.fromisoformat(self.last_fight_date)).days
        )

    # ---- write (applied only AFTER the feature row is built) ----
    def update(self, fight_date: str, won: int) -> None:
        self.results.append(int(won))
        self.last_fight_date = fight_date


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def age_at_date(date_of_birth: str | None, fight_date: str) -> float | None:
    """Age in years at the fight date, from DOB. Null if DOB is unknown."""
    if not date_of_birth:
        return None
    try:
        dob = date_cls.fromisoformat(str(date_of_birth)[:10])
    except ValueError:
        return None
    days = (date_cls.fromisoformat(fight_date) - dob).days
    return round(days / 365.25, 2) if days >= 0 else None


def diff(a: float | None, b: float | None) -> float | None:
    """fighter_a minus fighter_b; null if either side is null."""
    if a is None or b is None:
        return None
    if pd.isna(a) or pd.isna(b):
        return None
    return a - b


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

# Exact output column order (per spec).
OUTPUT_COLUMNS = [
    # identity / result
    "fight_id", "date", "event", "fighter_a", "fighter_b",
    "winner", "fighter_a_won", "weight_class", "method", "round",
    # Elo
    "fighter_a_pre_elo", "fighter_b_pre_elo", "elo_diff",
    "fighter_a_expected_win_prob", "fighter_b_expected_win_prob",
    # physical
    "fighter_a_age", "fighter_b_age", "age_diff",
    "fighter_a_height", "fighter_b_height", "height_diff",
    "fighter_a_reach", "fighter_b_reach", "reach_diff",
    # prior record
    "fighter_a_prior_fights", "fighter_b_prior_fights", "prior_fights_diff",
    "fighter_a_prior_wins", "fighter_b_prior_wins", "prior_wins_diff",
    "fighter_a_prior_losses", "fighter_b_prior_losses", "prior_losses_diff",
    "fighter_a_prior_win_pct", "fighter_b_prior_win_pct", "prior_win_pct_diff",
    # recent form
    "fighter_a_wins_last_3", "fighter_b_wins_last_3", "wins_last_3_diff",
    "fighter_a_win_pct_last_3", "fighter_b_win_pct_last_3", "win_pct_last_3_diff",
    "fighter_a_wins_last_5", "fighter_b_wins_last_5", "wins_last_5_diff",
    "fighter_a_win_pct_last_5", "fighter_b_win_pct_last_5", "win_pct_last_5_diff",
    # activity
    "fighter_a_days_since_last_fight", "fighter_b_days_since_last_fight",
    "days_since_last_fight_diff",
    # versioned experimental inactivity family (production allowlist unchanged)
    *LAYOFF_FEATURE_COLUMNS,
]


def build_feature_rows(fights: list[dict]) -> pd.DataFrame:
    """Build one time-safe feature row per fight.

    Each fight dict needs: fight_id, date (ISO), fighter_a_id, fighter_b_id,
    fighter_a / fighter_b (names), winner, fighter_a_won, plus Elo snapshot
    fields (fighter_a_pre_elo, ...) and optional metadata/physical fields.
    Sorting happens HERE, so out-of-order input is handled correctly.
    """
    ordered = sorted(fights, key=lambda f: (f["date"], f["fight_id"]))

    histories: dict[int, FighterHistory] = {}
    layoff_histories: dict[int, LayoffHistory] = {}
    rows: list[dict] = []

    for f in ordered:
        a_id, b_id = f["fighter_a_id"], f["fighter_b_id"]
        hist_a = histories.setdefault(a_id, FighterHistory())
        hist_b = histories.setdefault(b_id, FighterHistory())
        layoff_hist_a = layoff_histories.setdefault(a_id, LayoffHistory())
        layoff_hist_b = layoff_histories.setdefault(b_id, LayoffHistory())

        # ------- 1 & 2: read PRE-fight state for both fighters -------
        rec_a, rec_b = hist_a.prior_record(), hist_b.prior_record()
        f3_a, f3_b = hist_a.recent_form(3), hist_b.recent_form(3)
        f5_a, f5_b = hist_a.recent_form(5), hist_b.recent_form(5)
        days_a = hist_a.days_since_last_fight(f["date"])
        days_b = hist_b.days_since_last_fight(f["date"])
        layoff_features = build_layoff_feature_family(layoff_hist_a, layoff_hist_b, f.get("date"))

        # Age: prefer per-fight as-of value from the source; else DOB-derived.
        age_a = f.get("fighter_a_age")
        if age_a is None:
            age_a = age_at_date(f.get("fighter_a_dob"), f["date"])
        age_b = f.get("fighter_b_age")
        if age_b is None:
            age_b = age_at_date(f.get("fighter_b_dob"), f["date"])

        # ------- 3 & 4: build the row, including the target -------
        row = {
            "fight_id": f["fight_id"],
            "date": f["date"],
            "event": f.get("event"),
            "fighter_a": f["fighter_a"],
            "fighter_b": f["fighter_b"],
            "winner": f.get("winner"),
            "fighter_a_won": int(f["fighter_a_won"]),   # <- target
            "weight_class": f.get("weight_class"),
            "method": f.get("method"),
            "round": f.get("round"),
            # Elo (pre-fight, from fight_elo_snapshots)
            "fighter_a_pre_elo": f["fighter_a_pre_elo"],
            "fighter_b_pre_elo": f["fighter_b_pre_elo"],
            "elo_diff": f["fighter_a_pre_elo"] - f["fighter_b_pre_elo"],
            "fighter_a_expected_win_prob": f["fighter_a_expected_win_prob"],
            "fighter_b_expected_win_prob": f["fighter_b_expected_win_prob"],
            # physical
            "fighter_a_age": age_a,
            "fighter_b_age": age_b,
            "age_diff": diff(age_a, age_b),
            "fighter_a_height": f.get("fighter_a_height"),
            "fighter_b_height": f.get("fighter_b_height"),
            "height_diff": diff(f.get("fighter_a_height"), f.get("fighter_b_height")),
            "fighter_a_reach": f.get("fighter_a_reach"),
            "fighter_b_reach": f.get("fighter_b_reach"),
            "reach_diff": diff(f.get("fighter_a_reach"), f.get("fighter_b_reach")),
            # prior record
            "fighter_a_prior_fights": rec_a["prior_fights"],
            "fighter_b_prior_fights": rec_b["prior_fights"],
            "prior_fights_diff": rec_a["prior_fights"] - rec_b["prior_fights"],
            "fighter_a_prior_wins": rec_a["prior_wins"],
            "fighter_b_prior_wins": rec_b["prior_wins"],
            "prior_wins_diff": rec_a["prior_wins"] - rec_b["prior_wins"],
            "fighter_a_prior_losses": rec_a["prior_losses"],
            "fighter_b_prior_losses": rec_b["prior_losses"],
            "prior_losses_diff": rec_a["prior_losses"] - rec_b["prior_losses"],
            "fighter_a_prior_win_pct": rec_a["prior_win_pct"],
            "fighter_b_prior_win_pct": rec_b["prior_win_pct"],
            "prior_win_pct_diff": diff(rec_a["prior_win_pct"], rec_b["prior_win_pct"]),
            # recent form
            "fighter_a_wins_last_3": f3_a["wins_last_3"],
            "fighter_b_wins_last_3": f3_b["wins_last_3"],
            "wins_last_3_diff": f3_a["wins_last_3"] - f3_b["wins_last_3"],
            "fighter_a_win_pct_last_3": f3_a["win_pct_last_3"],
            "fighter_b_win_pct_last_3": f3_b["win_pct_last_3"],
            "win_pct_last_3_diff": diff(f3_a["win_pct_last_3"], f3_b["win_pct_last_3"]),
            "fighter_a_wins_last_5": f5_a["wins_last_5"],
            "fighter_b_wins_last_5": f5_b["wins_last_5"],
            "wins_last_5_diff": f5_a["wins_last_5"] - f5_b["wins_last_5"],
            "fighter_a_win_pct_last_5": f5_a["win_pct_last_5"],
            "fighter_b_win_pct_last_5": f5_b["win_pct_last_5"],
            "win_pct_last_5_diff": diff(f5_a["win_pct_last_5"], f5_b["win_pct_last_5"]),
            # activity
            "fighter_a_days_since_last_fight": days_a,
            "fighter_b_days_since_last_fight": days_b,
            "days_since_last_fight_diff": diff(days_a, days_b),
            **layoff_features,
        }
        rows.append(row)

        # ------- 5: ONLY NOW does the result update history -------
        hist_a.update(f["date"], int(f["fighter_a_won"]))
        hist_b.update(f["date"], 1 - int(f["fighter_a_won"]))
        if counts_as_completed_activity(f):
            layoff_hist_a.record_completed_fight(f.get("date"))
            layoff_hist_b.record_completed_fight(f.get("date"))

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


# ---------------------------------------------------------------------------
# Database plumbing
# ---------------------------------------------------------------------------

FIGHTS_WITH_ELO_QUERY = """
SELECT
    f.fight_id, f.date, f.event,
    f.fighter_a_id, f.fighter_b_id,
    fa.name AS fighter_a, fb.name AS fighter_b, fw.name AS winner,
    f.fighter_a_won, f.weight_class, f.method, f.round,
    f.fighter_a_age, f.fighter_b_age,
    -- Canonical rule (docs/greco_field_audit.md): per-fight tale-of-the-tape
    -- values (mdabbert) take precedence; static fighter values (Greco tott)
    -- only fill gaps. Deterministic, never averaged, never overwritten.
    COALESCE(f.fighter_a_height, CAST(fa.height AS REAL)) AS fighter_a_height,
    COALESCE(f.fighter_b_height, CAST(fb.height AS REAL)) AS fighter_b_height,
    COALESCE(f.fighter_a_reach,  CAST(fa.reach  AS REAL)) AS fighter_a_reach,
    COALESCE(f.fighter_b_reach,  CAST(fb.reach  AS REAL)) AS fighter_b_reach,
    fa.date_of_birth AS fighter_a_dob, fb.date_of_birth AS fighter_b_dob,
    s.fighter_a_pre_elo, s.fighter_b_pre_elo,
    s.fighter_a_expected_win_prob, s.fighter_b_expected_win_prob
FROM fights f
JOIN fighters fa ON fa.fighter_id = f.fighter_a_id
JOIN fighters fb ON fb.fighter_id = f.fighter_b_id
LEFT JOIN fighters fw ON fw.fighter_id = f.winner_id
LEFT JOIN fight_elo_snapshots s ON s.fight_id = f.fight_id
ORDER BY f.date, f.fight_id
"""


def build_features_for_db(db_path: str, output_csv: str) -> int:
    """Load fights + Elo snapshots from SQLite, build features, write CSV."""
    conn = connect(db_path)
    try:
        init_schema(conn)
        df = pd.read_sql_query(FIGHTS_WITH_ELO_QUERY, conn)
    finally:
        conn.close()

    if df.empty:
        raise RuntimeError(f"No fights found in {db_path}. Run ingest_fights.py first.")
    if df["fighter_a_pre_elo"].isna().any():
        n = int(df["fighter_a_pre_elo"].isna().sum())
        raise RuntimeError(
            f"{n} fights have no Elo snapshot. Run build_elo.py before "
            "building pre-fight features."
        )

    # object-with-None dicts so nulls stay explicit inside build_feature_rows
    fights = df.astype(object).where(pd.notna(df), None).to_dict("records")
    features = build_feature_rows(fights)

    bad = find_forbidden_columns(features.columns)
    if bad:
        raise RuntimeError(
            f"Leakage guard tripped: forbidden columns in feature output: {bad}"
        )

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_csv, index=False)
    return len(features)
