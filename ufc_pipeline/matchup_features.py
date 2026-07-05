"""Step 3C: time-safe style-matchup interaction features.

Step 3B gave each fighter individual rolling rates (how much they land,
absorb, take down, control). Step 3C adds what those rates say about the
PAIRING: does A's striking output exceed what B historically absorbs, does
A's wrestling meet a B who historically cannot stop takedowns, does A's
reach advantage coincide with striking volume that can use it.

Time safety is inherited, not re-invented: the new "against" stats use the
exact record-before-update loop of Step 3B (snapshot both fighters' PAST
stats -> emit row -> only then fold in the current fight), so the current
fight's stats can never appear in its own row and future fights can never
affect earlier rows. Matchup features are then pure arithmetic on those
pre-fight snapshots.

DIRECTION CONVENTION (applies to every matchup feature): positive value =
advantage for fighter A. Every formula is written as (A's edge) minus
(B's mirrored edge), or as an A-minus-B difference.

New rolling "against" stats (per fighter, from opponents' offensive rows
in fight_stats — nothing is invented; the opponent totals were already
stored for the Step 3B absorbed/defense rates):
    takedowns_allowed_per_15            = hist opp TD landed / hist min * 15
    opp_takedown_attempts_per_15        = hist opp TD attempted / hist min * 15
    opp_sig_str_attempted_per_min       = hist opp sig attempted / hist min
    control_time_absorbed_per_15        = hist opp ctrl minutes / hist min * 15
    knockdowns_absorbed_per_15          = hist opp KD / hist min * 15
    submission_attempts_absorbed_per_15 = hist opp sub att / hist min * 15

Matchup features (exact formulas; a_/b_ are the fighters' PRE-fight rolling
values, reach_diff is the Step 3 physical diff):
    striking_matchup_net_advantage =
        (a_sig_landed_pm - b_sig_absorbed_pm) - (b_sig_landed_pm - a_sig_absorbed_pm)
    striking_accuracy_matchup_net_advantage =
        (a_strike_acc - b_strike_def) - (b_strike_acc - a_strike_def)
    takedown_matchup_net_advantage =
        (a_td_per15 - b_td_allowed_per15) - (b_td_per15 - a_td_allowed_per15)
    takedown_accuracy_matchup_net_advantage =
        (a_td_acc - b_td_def) - (b_td_acc - a_td_def)
    control_matchup_net_advantage =
        (a_ctrl_per15 - b_ctrl_absorbed_per15) - (b_ctrl_per15 - a_ctrl_absorbed_per15)
    knockdown_matchup_net_advantage =
        (a_kd_rate - b_kd_absorbed_per15) - (b_kd_rate - a_kd_absorbed_per15)
    submission_matchup_net_advantage =
        (a_sub_per15 - b_sub_absorbed_per15) - (b_sub_per15 - a_sub_absorbed_per15)
    reach_volume_interaction =
        reach_diff * (a_sig_landed_pm + b_sig_landed_pm) / 2
        (reach advantage amplified by how striking-heavy the pairing is; the
         naive reach_diff * volume_diff product breaks the direction rule —
         a shorter AND less active A would score positive)
    pace_pressure_advantage =
        (a_sig_landed_pm + a_td_per15) - (b_sig_landed_pm + b_td_per15)
    opponent_pressure_absorption_advantage =
        (b_sig_absorbed_pm + b_td_allowed_per15) - (a_sig_absorbed_pm + a_td_allowed_per15)
        (B historically eats more pressure than A -> positive -> A advantage)

Honesty note, documented on purpose: each *_matchup_net_advantage expands
algebraically to (aX + aY) - (bX + bY), i.e. a composite-skill difference.
Where both underlying diffs already exist in Step 3B (striking), the net
feature adds interpretability but no new information to a LINEAR model;
where one side is a NEW against-stat (takedown/control/knockdown/
submission nets), it carries genuinely new signal. reach_volume_interaction
is a true product interaction — new information for any model.

Missing-history policy (same as Step 3B, never silent zeros): a fighter
with no prior fights-with-stats has null against-stats; any matchup feature
with a null input is null; matchup_history_missing = 1 when EITHER fighter
lacks stat history (the per-side no_prior_stats flags cannot distinguish
"both have history" from "both lack it", which is exactly the case where
every matchup feature is null). Imputation happens later, inside the model
pipeline, from training-split medians.

Style archetype scores (striker_score / grappler_score / ...) are
deliberately SKIPPED in this first version: doing them honestly needs
time-safe expanding-window normalization (a fighter's "striker-ness"
relative to the era), which is real added complexity for an unproven
payoff. The matchup nets above capture the same offense-vs-vulnerability
idea with explicit units.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .db import connect, init_schema
from .export import ALLOWED_RESULT_COLUMNS, find_forbidden_columns
from .features import FIGHTS_WITH_ELO_QUERY, build_feature_rows, diff
from .stats_features import (
    STATS_QUERY,
    STEP3B_COLUMNS,
    StatsHistory,
    build_step3b_rows,
    print_debug_timelines,
)

AGAINST_METRICS = [
    "takedowns_allowed_per_15",
    "opp_takedown_attempts_per_15",
    "opp_sig_str_attempted_per_min",
    "control_time_absorbed_per_15",
    "knockdowns_absorbed_per_15",
    "submission_attempts_absorbed_per_15",
]

MATCHUP_FEATURES = [
    "striking_matchup_net_advantage",
    "striking_accuracy_matchup_net_advantage",
    "takedown_matchup_net_advantage",
    "takedown_accuracy_matchup_net_advantage",
    "control_matchup_net_advantage",
    "knockdown_matchup_net_advantage",
    "submission_matchup_net_advantage",
    "reach_volume_interaction",
    "pace_pressure_advantage",
    "opponent_pressure_absorption_advantage",
]

STEP3C_COLUMNS = (
    [f"fighter_a_{m}" for m in AGAINST_METRICS]
    + [f"fighter_b_{m}" for m in AGAINST_METRICS]
    + [f"{m}_diff" for m in AGAINST_METRICS]
    + MATCHUP_FEATURES
    + ["matchup_history_missing"]
)


@dataclass
class StyleHistory(StatsHistory):
    """Step 3B history + absorbed finishing/control accumulators.

    opp_sl/opp_sa/opp_tdl/opp_tda already live in the parent; the three
    fields below complete the 'what happened TO this fighter' picture.
    """

    opp_kd: float = 0.0
    opp_sub: float = 0.0
    opp_ctrl_seconds: float = 0.0

    def update(self, own: dict, opp: dict, minutes: float) -> None:
        super().update(own, opp, minutes)

        def val(d, k):
            v = d.get(k)
            return 0.0 if v is None or pd.isna(v) else float(v)

        self.opp_kd += val(opp, "knockdowns")
        self.opp_sub += val(opp, "sub_attempts")
        self.opp_ctrl_seconds += val(opp, "ctrl_seconds")

    def snapshot_against(self) -> dict:
        """Pre-fight rolling against-stats; all-null if no usable history."""
        if self.fights_with_stats == 0 or self.minutes <= 0:
            return {m: None for m in AGAINST_METRICS}
        return {
            "takedowns_allowed_per_15": self.opp_tdl / self.minutes * 15.0,
            "opp_takedown_attempts_per_15": self.opp_tda / self.minutes * 15.0,
            "opp_sig_str_attempted_per_min": self.opp_sa / self.minutes,
            "control_time_absorbed_per_15":
                (self.opp_ctrl_seconds / 60.0) / self.minutes * 15.0,
            "knockdowns_absorbed_per_15": self.opp_kd / self.minutes * 15.0,
            "submission_attempts_absorbed_per_15": self.opp_sub / self.minutes * 15.0,
        }


def build_against_rows(
    fights: list[dict], stats: dict[tuple[int, int], dict]
) -> pd.DataFrame:
    """One row per fight with both fighters' PRE-fight against-stats.

    Same record-before-update loop as Step 3B (see stats_features.py);
    sorting happens here so out-of-order input cannot corrupt the timeline.
    """
    ordered = sorted(fights, key=lambda f: (f["date"], f["fight_id"]))
    histories: dict[int, StyleHistory] = {}
    rows = []

    for f in ordered:
        a_id, b_id = f["fighter_a_id"], f["fighter_b_id"]
        hist_a = histories.setdefault(a_id, StyleHistory())
        hist_b = histories.setdefault(b_id, StyleHistory())

        # snapshots BEFORE this fight's stats exist anywhere
        snap_a, snap_b = hist_a.snapshot_against(), hist_b.snapshot_against()
        row = {"fight_id": f["fight_id"]}
        for m in AGAINST_METRICS:
            row[f"fighter_a_{m}"] = snap_a[m]
            row[f"fighter_b_{m}"] = snap_b[m]
            row[f"{m}_diff"] = diff(snap_a[m], snap_b[m])
        rows.append(row)

        # only now does the current fight enter history
        own_a = stats.get((f["fight_id"], a_id))
        own_b = stats.get((f["fight_id"], b_id))
        if own_a is not None and own_b is not None:
            minutes = own_a.get("minutes") or own_b.get("minutes") or 0.0
            if minutes and minutes > 0:
                hist_a.update(own_a, own_b, minutes)
                hist_b.update(own_b, own_a, minutes)

    cols = ["fight_id"] + [
        f"fighter_{s}_{m}" for m in AGAINST_METRICS for s in ("a", "b")
    ] + [f"{m}_diff" for m in AGAINST_METRICS]
    # preserve declared column order
    ordered_cols = ["fight_id"] + [c for c in STEP3C_COLUMNS if c not in MATCHUP_FEATURES
                                   and c != "matchup_history_missing"]
    return pd.DataFrame(rows, columns=ordered_cols) if rows else pd.DataFrame(columns=cols)


def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 10 matchup features from PRE-fight columns already in df.

    Pure arithmetic on snapshot columns — pandas NaN propagation implements
    the 'any null input -> null feature' policy for free. Positive always
    means fighter A advantage (see module docstring).
    """
    out = df.copy()
    a = lambda m: out[f"fighter_a_{m}"]  # noqa: E731
    b = lambda m: out[f"fighter_b_{m}"]  # noqa: E731

    out["striking_matchup_net_advantage"] = (
        (a("sig_strikes_landed_per_min") - b("sig_strikes_absorbed_per_min"))
        - (b("sig_strikes_landed_per_min") - a("sig_strikes_absorbed_per_min"))
    )
    out["striking_accuracy_matchup_net_advantage"] = (
        (a("striking_accuracy") - b("striking_defense"))
        - (b("striking_accuracy") - a("striking_defense"))
    )
    out["takedown_matchup_net_advantage"] = (
        (a("takedowns_per_15") - b("takedowns_allowed_per_15"))
        - (b("takedowns_per_15") - a("takedowns_allowed_per_15"))
    )
    out["takedown_accuracy_matchup_net_advantage"] = (
        (a("takedown_accuracy") - b("takedown_defense"))
        - (b("takedown_accuracy") - a("takedown_defense"))
    )
    out["control_matchup_net_advantage"] = (
        (a("control_time_per_15") - b("control_time_absorbed_per_15"))
        - (b("control_time_per_15") - a("control_time_absorbed_per_15"))
    )
    out["knockdown_matchup_net_advantage"] = (
        (a("knockdown_rate") - b("knockdowns_absorbed_per_15"))
        - (b("knockdown_rate") - a("knockdowns_absorbed_per_15"))
    )
    out["submission_matchup_net_advantage"] = (
        (a("submission_attempts_per_15") - b("submission_attempts_absorbed_per_15"))
        - (b("submission_attempts_per_15") - a("submission_attempts_absorbed_per_15"))
    )
    out["reach_volume_interaction"] = out["reach_diff"] * (
        (a("sig_strikes_landed_per_min") + b("sig_strikes_landed_per_min")) / 2.0
    )
    out["pace_pressure_advantage"] = (
        (a("sig_strikes_landed_per_min") + a("takedowns_per_15"))
        - (b("sig_strikes_landed_per_min") + b("takedowns_per_15"))
    )
    out["opponent_pressure_absorption_advantage"] = (
        (b("sig_strikes_absorbed_per_min") + b("takedowns_allowed_per_15"))
        - (a("sig_strikes_absorbed_per_min") + a("takedowns_allowed_per_15"))
    )
    # 1 when EITHER side lacks stat history (=> every matchup feature above
    # is null for this row). The per-side flags cannot express "either".
    out["matchup_history_missing"] = (
        (out["fighter_a_no_prior_stats"].astype(int) == 1)
        | (out["fighter_b_no_prior_stats"].astype(int) == 1)
    ).astype(int)
    return out


def report_coverage(df: pd.DataFrame) -> None:
    """Print non-null coverage for every new Step 3C column."""
    n = len(df)
    print("Step 3C feature coverage (non-null / total):")
    for col in STEP3C_COLUMNS:
        filled = int(df[col].notna().sum())
        print(f"  {col:<45} {filled:>5}/{n} ({filled / n:6.1%})")


# ---------------------------------------------------------------------------
# Orchestration: Step 3 + Step 3B + Step 3C -> one CSV
# ---------------------------------------------------------------------------

def build_step3c_for_db(
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

    base = build_feature_rows(fights)                 # Step 3, unchanged
    step3b = build_step3b_rows(fights, stats)         # Step 3B, unchanged
    against = build_against_rows(fights, stats)       # Step 3C rolling against-stats
    merged = base.merge(step3b, on="fight_id", how="left", validate="1:1")
    merged = merged.merge(against, on="fight_id", how="left", validate="1:1")
    merged = add_matchup_features(merged)             # Step 3C interactions

    # Row-count validation: the expanded dataset must not gain/lose fights.
    assert len(merged) == len(base) == len(step3b) == len(against), \
        "row count changed in merge"

    # Leakage guard: 3B/3C historical names legitimately contain
    # sig_str/takedown/control/knockdown, so they are allowlisted by exact
    # name; any OTHER column matching a forbidden pattern still trips it.
    bad = find_forbidden_columns(
        merged.columns,
        allowed=ALLOWED_RESULT_COLUMNS | set(STEP3B_COLUMNS) | set(STEP3C_COLUMNS),
    )
    if bad:
        raise RuntimeError(f"Leakage guard tripped: forbidden columns: {bad}")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    report_coverage(merged)
    if debug_fighters:
        print_debug_timelines(merged, debug_fighters)
        print_matchup_timelines(merged, debug_fighters)
    return len(merged)


def print_matchup_timelines(df: pd.DataFrame, fighters: list[str]) -> None:
    """Show matchup features for a fighter's fights (validation aid)."""
    show = [
        "striking_matchup_net_advantage", "takedown_matchup_net_advantage",
        "control_matchup_net_advantage", "reach_volume_interaction",
    ]
    for name in fighters:
        print(f"\nStep 3C matchup timeline: {name} (positive = fighter A advantage)")
        print(f"  {'date':<11}{'A':<20}{'B':<20}" + "".join(f"{s[:18]:>20}" for s in show))
        sub = df[(df["fighter_a"] == name) | (df["fighter_b"] == name)]
        for _, r in sub.iterrows():
            vals = "".join(
                f"{r[s]:>20.3f}" if pd.notna(r[s]) else f"{'null':>20}" for s in show
            )
            print(f"  {r['date']:<11}{str(r['fighter_a'])[:18]:<20}"
                  f"{str(r['fighter_b'])[:18]:<20}{vals}")
