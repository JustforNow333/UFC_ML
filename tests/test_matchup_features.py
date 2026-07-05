"""Tests for Step 3C: style-matchup features.

Covers the 15 required behaviors:
 1. Step 3C features are computed only from prior fights.
 2. Current-fight stats do not leak into the current fight's matchup features.
 3. Future fights do not affect earlier Step 3C rows.
 4. Positive matchup advantage consistently means fighter A advantage.
 5. Debut fighters do not crash the feature builder.
 6. Missing Greco stat history does not crash the feature builder.
 7. Missing values are handled consistently (null propagation + flag).
 8. Step 3C output row count matches the modelable fight count.
 9. Step 3 and Step 3B features remain present in the Step 3C output.
10. Leakage guard allows Step 3C features, rejects raw current-fight stats.
11. Model comparison includes Step 3C without overwriting Step 3B results.
12. Calibration works on Step 3C probabilities.
13. Log loss is computed from probabilities, not hard class labels.
14. The debug timeline does not crash.
15. Determinism: same input -> same Step 3C output.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufc_pipeline.matchup_features import (  # noqa: E402
    AGAINST_METRICS,
    MATCHUP_FEATURES,
    STEP3C_COLUMNS,
    StyleHistory,
    add_matchup_features,
    build_against_rows,
    print_matchup_timelines,
)
from ufc_pipeline.modeling import (  # noqa: E402
    STEP3B_MODEL_FEATURES,
    STEP3C_MODEL_FEATURES,
    check_features_allowed,
    compare_feature_sets,
)
from ufc_pipeline.stats_features import METRICS, build_step3b_rows  # noqa: E402
from tests.test_stats_features import fight, small_db, stat_row  # noqa: E402, F401


def merged_frame(fights, stats):
    """Step 3B + against rows + a reach_diff column -> matchup features,
    mirroring the real builder's merge (without needing a database)."""
    step3b = build_step3b_rows(fights, stats)
    against = build_against_rows(fights, stats)
    merged = step3b.merge(against, on="fight_id", validate="1:1")
    merged["reach_diff"] = 5.0  # constant physical edge for fighter A
    return add_matchup_features(merged)


# --------------------------------------------------------------------- 1 & 2
def test_matchup_features_use_only_prior_fights():
    fights = [fight(1, "2020-01-01", 1, 2), fight(2, "2020-03-01", 1, 3)]
    stats = {
        (1, 1): stat_row(15.0, sl=60, sa=120, tdl=3, tda=6, kd=1, sub=2, ctrl=300),
        (1, 2): stat_row(15.0, sl=30, sa=90, tdl=1, tda=4, kd=1, sub=1, ctrl=120),
        (2, 1): stat_row(10.0, sl=999, sa=999, tdl=99, tda=99),  # poison values
        (2, 3): stat_row(10.0, sl=999, sa=999, tdl=99, tda=99),
    }
    out = merged_frame(fights, stats)

    # Fight 1: both debuts -> every against-stat and matchup feature null,
    # even though fight 1 itself HAS stats (they must not leak into row 1).
    r1 = out.iloc[0]
    for m in AGAINST_METRICS:
        assert pd.isna(r1[f"fighter_a_{m}"]) and pd.isna(r1[f"fighter_b_{m}"])
    for m in MATCHUP_FEATURES:
        assert pd.isna(r1[m]), m
    assert r1["matchup_history_missing"] == 1

    # Fight 2: fighter 1's against-stats reflect ONLY fight 1 (opponent 2's
    # numbers), not the poison values of fight 2 itself.
    r2 = out.iloc[1]
    assert r2["fighter_a_takedowns_allowed_per_15"] == pytest.approx(1.0)  # 1 TD/15min
    assert r2["fighter_a_knockdowns_absorbed_per_15"] == pytest.approx(1.0)
    assert r2["fighter_a_control_time_absorbed_per_15"] == pytest.approx(2.0)  # 120s
    assert r2["fighter_a_submission_attempts_absorbed_per_15"] == pytest.approx(1.0)
    assert r2["fighter_a_opp_sig_str_attempted_per_min"] == pytest.approx(6.0)


# --------------------------------------------------------------------- 3
def test_future_fights_do_not_affect_earlier_rows():
    base_fights = [fight(1, "2020-01-01", 1, 2), fight(2, "2020-03-01", 1, 3)]
    stats = {
        (1, 1): stat_row(15.0, sl=60, sa=120, tdl=3, tda=6),
        (1, 2): stat_row(15.0, sl=30, sa=90, tdl=1, tda=4),
        (2, 1): stat_row(10.0, sl=40, sa=80, tdl=2, tda=3),
        (2, 3): stat_row(10.0, sl=20, sa=60, tdl=0, tda=1),
    }
    out_short = merged_frame(base_fights, stats)
    extra = base_fights + [fight(3, "2021-06-01", 1, 4)]
    stats_extra = dict(stats)
    stats_extra[(3, 1)] = stat_row(15.0, sl=500, sa=500, tdl=50, tda=50)
    stats_extra[(3, 4)] = stat_row(15.0, sl=500, sa=500, tdl=50, tda=50)
    out_long = merged_frame(extra, stats_extra)
    pd.testing.assert_frame_equal(out_short, out_long.iloc[:2].reset_index(drop=True))


# --------------------------------------------------------------------- 4
def test_positive_means_fighter_a_advantage():
    # Fighter 1: heavy output history; fighter 4: gets dominated historically.
    fights = [
        fight(1, "2020-01-01", 1, 2),   # builds 1's strong history
        fight(2, "2020-01-01", 4, 3),   # builds 4's weak history (4 dominated)
        fight(3, "2021-01-01", 1, 4),   # the matchup we assert on: A=1, B=4
    ]
    stats = {
        (1, 1): stat_row(15.0, sl=90, sa=120, tdl=6, tda=8, kd=3, sub=3, ctrl=600),
        (1, 2): stat_row(15.0, sl=15, sa=60, tdl=0, tda=1, kd=0, sub=0, ctrl=30),
        (2, 4): stat_row(15.0, sl=15, sa=75, tdl=0, tda=2, kd=0, sub=0, ctrl=30),
        (2, 3): stat_row(15.0, sl=75, sa=105, tdl=5, tda=7, kd=2, sub=2, ctrl=540),
    }
    out = merged_frame(fights, stats)
    r = out[out["fight_id"] == 3].iloc[0]
    # the strong-history fighter is A: every matchup feature must be positive
    for m in MATCHUP_FEATURES:
        assert r[m] > 0, f"{m} should favor fighter A, got {r[m]}"

    # and the mirrored fight (A=4, B=1) flips every sign
    fights_flipped = fights[:2] + [fight(3, "2021-01-01", 4, 1)]
    flipped = merged_frame(fights_flipped, stats)
    rf = flipped[flipped["fight_id"] == 3].iloc[0]
    for m in MATCHUP_FEATURES:
        if m == "reach_volume_interaction":
            continue  # reach_diff is a fixture constant, not fighter-derived
        assert rf[m] < 0, f"{m} should flip sign when corners swap, got {rf[m]}"
        assert rf[m] == pytest.approx(-r[m])


# --------------------------------------------------------------------- 5 & 6
def test_debuts_and_missing_stat_history_do_not_crash():
    fights = [
        fight(1, "2020-01-01", 1, 2),                  # both debut
        fight(2, "2020-06-01", 1, 3),                  # 3 debuts; 1 has history
        fight(3, "2021-01-01", 5, 6),                  # both debut, NO stats at all
    ]
    stats = {
        (1, 1): stat_row(15.0, sl=60, sa=120, tdl=3, tda=6),
        (1, 2): stat_row(15.0, sl=30, sa=90, tdl=1, tda=4),
        # fight 2 and 3 have no stats rows at all
    }
    out = merged_frame(fights, stats)          # must not crash
    assert len(out) == 3
    r2 = out.iloc[1]
    # one-sided history: fighter A has values, B null, matchup features null
    assert pd.notna(r2["fighter_a_takedowns_allowed_per_15"])
    assert pd.isna(r2["fighter_b_takedowns_allowed_per_15"])
    for m in MATCHUP_FEATURES:
        assert pd.isna(r2[m])
    assert r2["matchup_history_missing"] == 1


# --------------------------------------------------------------------- 7
def test_null_propagation_is_consistent():
    fights = [fight(1, "2020-01-01", 1, 2), fight(2, "2020-03-01", 1, 3),
              fight(3, "2020-06-01", 1, 2)]
    stats = {
        (1, 1): stat_row(15.0, sl=60, sa=120, tdl=3, tda=6),
        (1, 2): stat_row(15.0, sl=30, sa=90, tdl=1, tda=4),
        (2, 1): stat_row(10.0, sl=40, sa=80, tdl=2, tda=3),
        (2, 3): stat_row(10.0, sl=20, sa=60, tdl=0, tda=1),
        (3, 1): stat_row(15.0, sl=1, sa=2, tdl=0, tda=0),
        (3, 2): stat_row(15.0, sl=1, sa=2, tdl=0, tda=0),
    }
    out = merged_frame(fights, stats)
    r3 = out.iloc[2]  # both fighters have history now
    assert r3["matchup_history_missing"] == 0
    for m in MATCHUP_FEATURES:
        assert pd.notna(r3[m]), f"{m} should be computable with full history"
    # a matchup feature is null exactly when one of its inputs is null:
    # fighter 3 (one fight of history) vs a debutant would be null — covered
    # in the debut test; here we assert no spurious nulls appear.


# --------------------------------------------------------------------- 8 & 9 & 15
def test_full_builder_row_count_columns_and_determinism(tmp_path, small_db):
    from ufc_pipeline.features import OUTPUT_COLUMNS
    from ufc_pipeline.matchup_features import build_step3c_for_db
    from ufc_pipeline.stats_features import STEP3B_COLUMNS

    _, db = small_db
    out1 = tmp_path / "step3c_a.csv"
    out2 = tmp_path / "step3c_b.csv"
    n1 = build_step3c_for_db(str(db), str(out1))
    n2 = build_step3c_for_db(str(db), str(out2))

    df = pd.read_csv(out1)
    assert n1 == n2 == len(df)                       # row count = modelable fights
    for col in OUTPUT_COLUMNS:                       # Step 3 intact
        assert col in df.columns, f"missing Step 3 column {col}"
    for col in STEP3B_COLUMNS:                       # Step 3B intact
        assert col in df.columns, f"missing Step 3B column {col}"
    for col in STEP3C_COLUMNS:                       # Step 3C present
        assert col in df.columns, f"missing Step 3C column {col}"
    # determinism: byte-identical output
    assert out1.read_bytes() == out2.read_bytes()


# -------------------------------------------------------------------- 10
def test_leakage_guard_allows_3c_rejects_raw_stats():
    check_features_allowed(
        ["elo_diff"] + list(STEP3B_MODEL_FEATURES) + list(STEP3C_MODEL_FEATURES)
    )
    for raw in ("sig_str_landed", "td_landed", "ctrl_seconds", "knockdowns",
                "sub_attempts", "red_odds", "rank_diff", "fighter_a_post_elo",
                "current_takedown_total", "round", "method"):
        with pytest.raises(ValueError, match="Leakage guard"):
            check_features_allowed(["elo_diff", raw])


# -------------------------------------------------------------------- 11
def test_comparison_includes_3c_without_touching_3b_outputs(tmp_path):
    from tests.test_modeling import synthetic_features

    df = synthetic_features(300)
    rng = np.random.default_rng(1)
    for col in STEP3B_MODEL_FEATURES + STEP3C_MODEL_FEATURES:
        if col not in df.columns:
            df[col] = (rng.integers(0, 2, len(df))
                       if col in ("fighter_a_no_prior_stats",
                                  "fighter_b_no_prior_stats",
                                  "matchup_history_missing")
                       else rng.normal(0, 1, len(df)))
    csv = tmp_path / "step3c.csv"
    df.to_csv(csv, index=False)

    metrics_out = tmp_path / "model_comparison_step3c.json"
    preds_out = tmp_path / "model_comparison_step3c_predictions.csv"
    results = compare_feature_sets(
        input_csv=str(csv),
        test_size=0.3,
        metrics_output=str(metrics_out),
        predictions_output=str(preds_out),
        coefficients_dir=str(tmp_path),
    )
    assert set(results["models"]) == {"step3_basic", "step3_plus_3b", "step3c_matchup"}
    assert metrics_out.exists() and preds_out.exists()
    used = results["models"]["step3c_matchup"]["features_numeric"]
    assert "striking_matchup_net_advantage" in used
    assert "takedowns_allowed_per_15_diff" in used
    # 3B evaluation is unchanged by adding the 3C set (same features)
    assert "striking_matchup_net_advantage" not in \
        results["models"]["step3_plus_3b"]["features_numeric"]
    # without 3C columns in the input, the default sets are unchanged (no 3C)
    df3b = synthetic_features(300)
    csv3b = tmp_path / "step3b_only.csv"
    df3b.to_csv(csv3b, index=False)
    results3b = compare_feature_sets(
        input_csv=str(csv3b),
        test_size=0.3,
        metrics_output=str(tmp_path / "m3b.json"),
        predictions_output=str(tmp_path / "p3b.csv"),
        coefficients_dir=str(tmp_path),
    )
    assert set(results3b["models"]) == {"step3_basic", "step3_plus_3b"}


# -------------------------------------------------------------------- 12 & 13
def test_calibration_works_on_step3c_probabilities(tmp_path):
    from sklearn.metrics import log_loss

    from ufc_pipeline.calibration import run_calibration
    from ufc_pipeline.modeling import DEFAULT_NUMERIC_FEATURES
    from tests.test_calibration import biased_synthetic_features

    df = biased_synthetic_features(600)
    # attach synthetic 3C columns so the 3C feature list is exercised
    rng = np.random.default_rng(0)
    for colname in ("striking_matchup_net_advantage", "takedowns_allowed_per_15_diff",
                    "reach_volume_interaction", "pace_pressure_advantage"):
        df[colname] = rng.normal(0, 1, len(df))
    df["matchup_history_missing"] = 0
    csv = tmp_path / "features.csv"
    df.to_csv(csv, index=False)

    features = list(DEFAULT_NUMERIC_FEATURES) + [
        "striking_matchup_net_advantage", "takedowns_allowed_per_15_diff",
        "reach_volume_interaction", "pace_pressure_advantage",
        "matchup_history_missing",
    ]
    results = run_calibration(
        input_csv=str(csv),
        output_dir=str(tmp_path / "processed"),
        model_dir=str(tmp_path / "models"),
        numeric_features=features,
        output_prefix="step3c_",
        model_stem="step3c_logistic_regression",
    )
    # step3c-prefixed outputs; step3b file names untouched
    processed = tmp_path / "processed"
    assert (processed / "step3c_calibration_comparison.json").exists()
    assert not (processed / "calibration_comparison.json").exists()
    assert (tmp_path / "models" / "step3c_logistic_regression_platt_calibrated.joblib").exists()

    # log loss must come from probabilities, not 0/1 labels
    preds = pd.read_csv(processed / "step3c_calibration_predictions.csv")
    y = preds["target"].to_numpy(dtype=int)
    probs = preds["platt_probability"].to_numpy()
    assert results["models"]["platt"]["log_loss"] == pytest.approx(
        log_loss(y, probs, labels=[0, 1]), abs=1e-9
    )
    labels_ll = log_loss(y, np.clip(preds["platt_prediction"], 1e-6, 1 - 1e-6),
                         labels=[0, 1])
    assert results["models"]["platt"]["log_loss"] != pytest.approx(labels_ll, abs=1e-6)


# -------------------------------------------------------------------- 14
def test_debug_timeline_does_not_crash(capsys):
    fights = [fight(1, "2020-01-01", 1, 2), fight(2, "2020-03-01", 1, 3)]
    stats = {
        (1, 1): stat_row(15.0, sl=60, sa=120, tdl=3, tda=6),
        (1, 2): stat_row(15.0, sl=30, sa=90, tdl=1, tda=4),
    }
    out = merged_frame(fights, stats)
    out["date"] = ["2020-01-01", "2020-03-01"]
    out["fighter_a"] = ["Alpha", "Alpha"]
    out["fighter_b"] = ["Beta", "Gamma"]
    print_matchup_timelines(out, ["Alpha", "Nobody"])   # unknown name is fine
    captured = capsys.readouterr()
    assert "Step 3C matchup timeline: Alpha" in captured.out
    assert "positive = fighter A advantage" in captured.out


# ------------------------------------------------------------ unit extras
def test_style_history_accumulates_absorbed_stats():
    h = StyleHistory()
    h.update(
        stat_row(15.0, sl=10, sa=20, tdl=1, tda=2),
        stat_row(15.0, sl=30, sa=60, tdl=2, tda=5, kd=2, sub=3, ctrl=450),
        15.0,
    )
    snap = h.snapshot_against()
    assert snap["takedowns_allowed_per_15"] == pytest.approx(2.0)
    assert snap["opp_takedown_attempts_per_15"] == pytest.approx(5.0)
    assert snap["opp_sig_str_attempted_per_min"] == pytest.approx(4.0)
    assert snap["knockdowns_absorbed_per_15"] == pytest.approx(2.0)
    assert snap["submission_attempts_absorbed_per_15"] == pytest.approx(3.0)
    assert snap["control_time_absorbed_per_15"] == pytest.approx(7.5)  # 450s = 7.5min

    # 3B parent metrics still work on the subclass (one snapshot pass)
    parent = h.snapshot()
    assert parent["takedown_defense"] == pytest.approx(1 - 2 / 5)


def test_matchup_formula_values():
    """Hand-computed check of one matchup feature end to end."""
    fights = [
        fight(1, "2020-01-01", 1, 2),
        fight(2, "2020-01-01", 4, 3),
        fight(3, "2021-01-01", 1, 4),
    ]
    stats = {
        (1, 1): stat_row(15.0, sl=90, sa=120, tdl=6, tda=8),   # 1: 6 TD/15
        (1, 2): stat_row(15.0, sl=15, sa=60, tdl=0, tda=1),    # 1 allows 0
        (2, 4): stat_row(15.0, sl=15, sa=75, tdl=0, tda=2),    # 4: 0 TD/15
        (2, 3): stat_row(15.0, sl=75, sa=105, tdl=5, tda=7),   # 4 allows 5
    }
    out = merged_frame(fights, stats)
    r = out[out["fight_id"] == 3].iloc[0]
    # (a_td15 - b_allowed15) - (b_td15 - a_allowed15) = (6-5) - (0-0) = 1
    assert r["takedown_matchup_net_advantage"] == pytest.approx(1.0)
    # pace: (a_slpm + a_td15) - (b_slpm + b_td15) = (6+6) - (1+0) = 11
    assert r["pace_pressure_advantage"] == pytest.approx(11.0)
