"""Step 3B: time-safe rolling striking & grappling features.

Extends the Step 3 dataset with historical stat rates for each fighter,
computed with the SAME record-before-update discipline as Elo and Step 3:

    for each fight in (date, fight_id) order:
        1. snapshot fighter A's accumulated stats  (past fights only)
        2. snapshot fighter B's accumulated stats  (past fights only)
        3. emit the feature row
        4. ONLY THEN fold this fight's stats into both accumulators

so the current fight's strikes/takedowns/control can never appear in its own
row, and future fights can never affect earlier rows. Same-date fights keep
dataset order via fight_id.

Metrics per fighter (then A-minus-B diffs):
    sig_strikes_landed_per_min   = hist sig landed / hist minutes
    sig_strikes_absorbed_per_min = hist opp sig landed / hist minutes
    sig_strike_differential      = landed_per_min - absorbed_per_min
    striking_accuracy            = hist sig landed / hist sig attempted
    striking_defense             = 1 - hist opp landed / hist opp attempted
    knockdown_rate               = hist KD / hist minutes * 15
    takedowns_per_15             = hist TD landed / hist minutes * 15
    takedown_accuracy            = hist TD landed / hist TD attempted
    takedown_defense             = 1 - hist opp TD landed / hist opp TD att
    control_time_per_15          = hist ctrl minutes / hist minutes * 15
    submission_attempts_per_15   = hist sub attempts / hist minutes * 15

Missing-history policy (explicit, never silent zeros):
  * A fighter with NO prior fights-with-stats gets null for every rolling
    metric and fighter_x_no_prior_stats = 1. Nulls are imputed later inside
    the model pipeline (train-split medians); the flag lets the model treat
    debuts differently. We deliberately do NOT fill zeros — zero means
    "lands nothing", which is a strong false claim about a debutant.
  * Ratio metrics with zero denominators (e.g. no TD attempts yet) are null.
  * A past fight with no stats row still counts toward Step 3 record
    features but contributes nothing to stat accumulators (documented).
  * Any diff with a null side is null.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .db import connect, init_schema
from .export import ALLOWED_RESULT_COLUMNS, find_forbidden_columns
from .features import FIGHTS_WITH_ELO_QUERY, build_feature_rows, diff

METRICS = [
    "sig_strikes_landed_per_min",
    "sig_strikes_absorbed_per_min",
    "sig_strike_differential",
    "striking_accuracy",
    "striking_defense",
    "knockdown_rate",
    "takedowns_per_15",
    "takedown_accuracy",
    "takedown_defense",
    "control_time_per_15",
    "submission_attempts_per_15",
]

STEP3B_COLUMNS = (
    [f"fighter_a_{m}" for m in METRICS]
    + [f"fighter_b_{m}" for m in METRICS]
    + [f"{m}_diff" for m in METRICS]
    + ["fighter_a_no_prior_stats", "fighter_b_no_prior_stats", "no_prior_stats_diff"]
)


def _ratio(num: float, den: float) -> float | None:
    if num is None or den is None or pd.isna(num) or pd.isna(den):
        return None
    return (num / den) if den > 0 else None


@dataclass
class StatsHistory:
    """Accumulated PAST fight stats for one fighter."""

    minutes: float = 0.0
    kd: float = 0.0
    sl: float = 0.0        # sig strikes landed
    sa: float = 0.0        # sig strikes attempted
    opp_sl: float = 0.0    # absorbed (opponent landed on this fighter)
    opp_sa: float = 0.0    # opponent attempts against this fighter
    tdl: float = 0.0
    tda: float = 0.0
    opp_tdl: float = 0.0   # takedowns allowed
    opp_tda: float = 0.0   # opponent takedown attempts
    sub: float = 0.0
    ctrl_seconds: float = 0.0
    fights_with_stats: int = 0

    def snapshot(self) -> dict:
        """Pre-fight rolling rates; all-null + flag if no usable history."""
        if self.fights_with_stats == 0 or self.minutes <= 0:
            snap = {m: None for m in METRICS}
            snap["no_prior_stats"] = 1
            return snap
        slpm = self.sl / self.minutes
        sapm = self.opp_sl / self.minutes
        strk_def = _ratio(self.opp_sl, self.opp_sa)
        td_def = _ratio(self.opp_tdl, self.opp_tda)
        return {
            "sig_strikes_landed_per_min": slpm,
            "sig_strikes_absorbed_per_min": sapm,
            "sig_strike_differential": slpm - sapm,
            "striking_accuracy": _ratio(self.sl, self.sa),
            "striking_defense": (1.0 - strk_def) if strk_def is not None else None,
            "knockdown_rate": self.kd / self.minutes * 15.0,
            "takedowns_per_15": self.tdl / self.minutes * 15.0,
            "takedown_accuracy": _ratio(self.tdl, self.tda),
            "takedown_defense": (1.0 - td_def) if td_def is not None else None,
            "control_time_per_15": (self.ctrl_seconds / 60.0) / self.minutes * 15.0,
            "submission_attempts_per_15": self.sub / self.minutes * 15.0,
            "no_prior_stats": 0,
        }

    def update(self, own: dict, opp: dict, minutes: float) -> None:
        """Fold ONE finished fight into history (called only after snapshot)."""
        def val(d, k):
            v = d.get(k)
            return 0.0 if v is None or pd.isna(v) else float(v)

        self.minutes += float(minutes)
        self.kd += val(own, "knockdowns")
        self.sl += val(own, "sig_str_landed")
        self.sa += val(own, "sig_str_attempted")
        self.opp_sl += val(opp, "sig_str_landed")
        self.opp_sa += val(opp, "sig_str_attempted")
        self.tdl += val(own, "td_landed")
        self.tda += val(own, "td_attempted")
        self.opp_tdl += val(opp, "td_landed")
        self.opp_tda += val(opp, "td_attempted")
        self.sub += val(own, "sub_attempts")
        self.ctrl_seconds += val(own, "ctrl_seconds")
        self.fights_with_stats += 1


def build_step3b_rows(fights: list[dict], stats: dict[tuple[int, int], dict]) -> pd.DataFrame:
    """One Step-3B row per fight. `stats[(fight_id, fighter_id)]` holds that
    fighter's offensive totals for that fight (or is absent).
    Sorting happens here; out-of-order input cannot corrupt the timeline."""
    ordered = sorted(fights, key=lambda f: (f["date"], f["fight_id"]))
    histories: dict[int, StatsHistory] = {}
    rows = []

    for f in ordered:
        a_id, b_id = f["fighter_a_id"], f["fighter_b_id"]
        hist_a = histories.setdefault(a_id, StatsHistory())
        hist_b = histories.setdefault(b_id, StatsHistory())

        # ---- 1 & 2: snapshots BEFORE this fight's stats exist anywhere ----
        snap_a, snap_b = hist_a.snapshot(), hist_b.snapshot()

        row = {"fight_id": f["fight_id"]}
        for m in METRICS:
            row[f"fighter_a_{m}"] = snap_a[m]
            row[f"fighter_b_{m}"] = snap_b[m]
            row[f"{m}_diff"] = diff(snap_a[m], snap_b[m])
        row["fighter_a_no_prior_stats"] = snap_a["no_prior_stats"]
        row["fighter_b_no_prior_stats"] = snap_b["no_prior_stats"]
        row["no_prior_stats_diff"] = (
            snap_a["no_prior_stats"] - snap_b["no_prior_stats"]
        )
        rows.append(row)

        # ---- 4: only now does the current fight enter history ----
        own_a = stats.get((f["fight_id"], a_id))
        own_b = stats.get((f["fight_id"], b_id))
        if own_a is not None and own_b is not None:
            minutes = own_a.get("minutes") or own_b.get("minutes") or 0.0
            if minutes and minutes > 0:
                hist_a.update(own_a, own_b, minutes)
                hist_b.update(own_b, own_a, minutes)
        # No stats for this fight -> it silently contributes nothing to stat
        # history (it still counts toward Step-3 record features). Documented.

    return pd.DataFrame(rows, columns=["fight_id"] + STEP3B_COLUMNS)


# ---------------------------------------------------------------------------
# Orchestration: Step 3 features + Step 3B columns -> one CSV
# ---------------------------------------------------------------------------

STATS_QUERY = """
SELECT fight_id, fighter_id, minutes, knockdowns, sig_str_landed,
       sig_str_attempted, td_landed, td_attempted, sub_attempts, ctrl_seconds
FROM fight_stats
"""


def build_step3b_for_db(
    db_path: str,
    output_csv: str,
    debug_fighters: list[str] | None = None,
) -> int:
    conn = connect(db_path)
    try:
        init_schema(conn)
        fights_df = pd.read_sql_query(FIGHTS_WITH_ELO_QUERY, conn)
        stats_df = pd.read_sql_query(STATS_QUERY, conn)
    finally:
        conn.close()

    if fights_df.empty:
        raise RuntimeError(f"No fights found in {db_path}. Run ingest_fights.py first.")
    if fights_df["fighter_a_pre_elo"].isna().any():
        raise RuntimeError("Fights without Elo snapshots found. Run build_elo.py first.")

    fights = fights_df.astype(object).where(pd.notna(fights_df), None).to_dict("records")
    stats = {
        (int(r["fight_id"]), int(r["fighter_id"])): r
        for r in stats_df.astype(object)
        .where(pd.notna(stats_df), None)
        .to_dict("records")
    }
    n_with_stats = len({fid for fid, _ in stats})
    print(
        f"Fights: {len(fights)} | fights with stats rows: {n_with_stats} "
        f"({len(fights) - n_with_stats} without; their rows get nulls/flags "
        "and they don't feed stat history)"
    )

    base = build_feature_rows(fights)                # Step 3, unchanged
    step3b = build_step3b_rows(fights, stats)        # Step 3B additions
    merged = base.merge(step3b, on="fight_id", how="left", validate="1:1")

    # Row-count validation: the expanded dataset must not gain/lose fights.
    assert len(merged) == len(base) == len(step3b), "row count changed in merge"

    # Leakage guard: 3B's historical columns legitimately contain
    # sig_str/takedown/control in their names, so they are explicitly
    # allowlisted by exact name; anything else matching a forbidden pattern
    # still trips the guard.
    bad = find_forbidden_columns(
        merged.columns, allowed=ALLOWED_RESULT_COLUMNS | set(STEP3B_COLUMNS)
    )
    if bad:
        raise RuntimeError(f"Leakage guard tripped: forbidden columns: {bad}")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    if debug_fighters:
        print_debug_timelines(merged, debug_fighters)
    return len(merged)


def print_debug_timelines(df: pd.DataFrame, fighters: list[str]) -> None:
    """Show a fighter's PRE-fight rolling stats across time (validation aid)."""
    show = [
        "sig_strikes_landed_per_min", "striking_accuracy",
        "takedowns_per_15", "control_time_per_15",
    ]
    for name in fighters:
        print(f"\nPre-fight rolling stats timeline: {name}")
        header = f"  {'date':<11}{'opponent':<22}{'side':<5}" + "".join(
            f"{s[:14]:>16}" for s in show
        ) + f"{'no_hist':>8}"
        print(header)
        sub = df[(df["fighter_a"] == name) | (df["fighter_b"] == name)]
        for _, r in sub.iterrows():
            side = "A" if r["fighter_a"] == name else "B"
            opp = r["fighter_b"] if side == "A" else r["fighter_a"]
            vals = "".join(
                f"{(r[f'fighter_{side.lower()}_{s}']):>16.3f}"
                if pd.notna(r[f"fighter_{side.lower()}_{s}"])
                else f"{'null':>16}"
                for s in show
            )
            flag = r[f"fighter_{side.lower()}_no_prior_stats"]
            print(f"  {r['date']:<11}{str(opp)[:20]:<22}{side:<5}{vals}{int(flag):>8}")
