"""Tests for Step 5A feature diagnostics (diagnostics-only, no model changes).

Covers:
 1. Feature-group mapping covers every official Step 3C feature exactly
    once, with no extras, and is deterministic.
 2. The ablation runner returns the expected schema on a reduced set.
 3. The ablation runner raises a clear error for an unknown group.
 4. The pipeline's StandardScaler is fit on the train split only.
 5. PlattCalibrator parameters depend only on the calibration rows, not on
    train/test rows (verified end-to-end through run_single_ablation).
 6. The leakage guard still rejects target/result columns.
 7. run_feature_diagnostics() writes valid JSON with the required top-level
    fields (on synthetic data, small n_repeats, reduced ablations).
 8. Coefficient report schema (columns present, ranks correct).
 9. Permutation importance schema + determinism with a fixed seed.
10. classify_group_effects sign convention on known synthetic deltas.
11. Permutation importance sign convention (strong signal -> positive
    importance) and that it never mutates the caller's test dataframe.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufc_pipeline.calibration import (  # noqa: E402
    CalibratedPipeline,
    chronological_three_way_split,
)
from ufc_pipeline.feature_diagnostics import (  # noqa: E402
    FEATURE_GROUPS,
    NOISE_THRESHOLD,
    OFFICIAL_LR_PARAMS,
    build_coefficient_report,
    check_features_allowed,
    classify_group_effects,
    get_feature_groups,
    groups_to_columns,
    official_step3c_features,
    permutation_importance_report,
    run_ablation_suite,
    run_feature_diagnostics,
    make_official_step3c_pipeline,
    run_single_ablation,
    scaling_audit,
    validate_feature_groups,
)
from ufc_pipeline.modeling import TARGET  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "benchmarks" / "official_baseline.json"


def synthetic_step3c_features(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A full-width current-official (43 numeric, no raw weight_class) shaped
    dataset, with real signal in elo_diff and a deliberate distribution
    shift in a couple of columns across eras (used by the scaling-audit
    test to prove train-only stats differ from full-dataset stats)."""
    rng = np.random.default_rng(seed)
    numeric_all, categorical_all = official_step3c_features()

    elo_diff = rng.normal(0, 80, n)
    p = 1 / (1 + 10 ** (-elo_diff / 400))
    won = (rng.random(n) < p).astype(int)
    dates = pd.date_range("2015-01-01", periods=n, freq="7D").strftime("%Y-%m-%d")

    flag_cols = {"fighter_a_no_prior_stats", "fighter_b_no_prior_stats", "matchup_history_missing"}
    era = np.arange(n) / n  # 0 (oldest) .. ~1 (newest)

    data = {
        "fight_id": np.arange(1, n + 1),
        "date": dates,
        "event": [f"UFC {i}" for i in range(n)],
        "fighter_a": [f"A{i}" for i in range(n)],
        "fighter_b": [f"B{i}" for i in range(n)],
        "winner": np.where(won == 1, [f"A{i}" for i in range(n)], [f"B{i}" for i in range(n)]),
        TARGET: won,
        "weight_class": rng.choice(["Lightweight", "Welterweight", "Bantamweight"], n),
        "fighter_a_expected_win_prob": p,
        "elo_diff": elo_diff,
    }
    for col in numeric_all:
        if col == "elo_diff":
            continue
        if col in flag_cols:
            data[col] = rng.integers(0, 2, n)
        elif col == "height_diff":
            # deliberate era shift: later fights skew taller-A, for the
            # scaling-audit test (train-only mean should differ from the
            # full-dataset mean).
            data[col] = rng.normal(0, 5, n) + era * 6.0
        else:
            data[col] = rng.normal(0, 1.0, n)

    df = pd.DataFrame(data)
    assert set(numeric_all).issubset(df.columns)
    assert categorical_all == []
    return df


def _official_log_loss() -> float:
    with open(BASELINE_PATH) as fh:
        return json.load(fh)["official_model"]["test_metrics"]["log_loss"]


# --------------------------------------------------------------------- 1
def test_feature_groups_cover_official_features_exactly_once():
    validate_feature_groups()  # must not raise
    numeric, categorical = official_step3c_features()
    official = set(numeric) | set(categorical)

    seen = []
    for cols in FEATURE_GROUPS.values():
        seen.extend(cols)
    assert len(seen) == len(set(seen)), "a column appears in more than one group"
    assert set(seen) == official, "groups must cover every official feature, no extras"


def test_feature_groups_deterministic():
    first = get_feature_groups()
    second = get_feature_groups()
    assert first == second
    # returned copies are independent of the module-level dict
    first["elo"].append("bogus")
    assert "bogus" not in FEATURE_GROUPS["elo"]


def test_groups_to_columns_unknown_group_raises():
    with pytest.raises(ValueError, match="Unknown feature group"):
        groups_to_columns(["not_a_real_group"])


# --------------------------------------------------------------------- 2
def test_ablation_runner_schema():
    df = synthetic_step3c_features(300)
    train, calib, test = chronological_three_way_split(df, train_frac=0.70, calibration_frac=0.15)
    official_ll = _official_log_loss()

    ablations = {"only_elo": ["elo"], "all": list(FEATURE_GROUPS)}
    results, artifacts = run_ablation_suite(train, calib, test, ablations, official_ll)

    for name in ablations:
        assert name in results and name in artifacts
        entry = results[name]
        for key in (
            "name", "groups", "n_features", "n_numeric", "n_categorical",
            "n_train", "n_calibration", "n_test", "uncalibrated_log_loss",
            "platt_log_loss", "platt_brier_score", "roc_auc", "accuracy",
            "delta_vs_official", "calibration_platt",
        ):
            assert key in entry, f"{key} missing from ablation {name!r}"
        assert entry["n_train"] == len(train)
        assert entry["n_calibration"] == len(calib)
        assert entry["n_test"] == len(test)
        assert 0.0 < entry["platt_log_loss"]

    helpful, harmful, neutral = classify_group_effects(results)
    # only "elo" removal ablation isn't present here, so all lists are empty
    assert helpful == [] and harmful == [] and neutral == []


def test_all_ablation_uses_current_official_elasticnet_without_weight_class():
    df = synthetic_step3c_features(300)
    train, calib, test = chronological_three_way_split(df, train_frac=0.70, calibration_frac=0.15)
    y_train = train[TARGET].astype(int).to_numpy()
    y_calib = calib[TARGET].astype(int).to_numpy()
    y_test = test[TARGET].astype(int).to_numpy()

    _entry, artifacts = run_single_ablation(
        "all", list(FEATURE_GROUPS), train, calib, test,
        y_train, y_calib, y_test, _official_log_loss(),
    )

    model = artifacts["pipeline"].named_steps["model"]
    assert artifacts["numeric"] == official_step3c_features()[0]
    assert artifacts["categorical"] == []
    assert "weight_class" not in artifacts["numeric"]
    assert model.penalty == OFFICIAL_LR_PARAMS["penalty"]
    assert model.C == pytest.approx(OFFICIAL_LR_PARAMS["C"])
    assert model.l1_ratio == pytest.approx(OFFICIAL_LR_PARAMS["l1_ratio"])
    assert model.solver == OFFICIAL_LR_PARAMS["solver"]


# --------------------------------------------------------------------- 3
def test_ablation_runner_raises_for_unknown_group():
    df = synthetic_step3c_features(200)
    train, calib, test = chronological_three_way_split(df, train_frac=0.70, calibration_frac=0.15)
    y_train = train[TARGET].astype(int).to_numpy()
    y_calib = calib[TARGET].astype(int).to_numpy()
    y_test = test[TARGET].astype(int).to_numpy()
    with pytest.raises(ValueError, match="Unknown feature group"):
        run_single_ablation(
            "bad", ["not_a_group"], train, calib, test, y_train, y_calib, y_test, 0.65
        )


# --------------------------------------------------------------------- 4
def test_scaler_fit_on_train_only():
    df = synthetic_step3c_features(400)
    numeric, categorical = official_step3c_features()
    train, calib, test = chronological_three_way_split(df, train_frac=0.70, calibration_frac=0.15)

    pipeline = make_official_step3c_pipeline(numeric)
    pipeline.fit(train[numeric + categorical], train[TARGET].astype(int))

    result = scaling_audit(pipeline, train, pd.concat([train, calib, test]), numeric)
    assert result["scaler_present"] is True
    assert result["fit_on_train_only"] is True
    # height_diff has a deliberate era shift -> full-dataset stats must differ
    # from train-only stats, proving the scaler did NOT quietly use everything.
    assert result["matches_full_dataset_stats_instead"] is False


# --------------------------------------------------------------------- 5
def test_platt_depends_only_on_calibration_rows():
    df = synthetic_step3c_features(400)
    official_ll = _official_log_loss()
    train, calib, test = chronological_three_way_split(df, train_frac=0.70, calibration_frac=0.15)
    y_train = train[TARGET].astype(int).to_numpy()
    y_calib = calib[TARGET].astype(int).to_numpy()
    y_test = test[TARGET].astype(int).to_numpy()

    # Changing TEST rows (but not train/calib) must not change Platt's fit.
    entry1, art1 = run_single_ablation(
        "elo", ["elo"], train, calib, test, y_train, y_calib, y_test, official_ll
    )
    test2 = test.copy()
    test2[TARGET] = 1 - test2[TARGET]  # flip test labels entirely
    y_test2 = test2[TARGET].astype(int).to_numpy()
    entry2, art2 = run_single_ablation(
        "elo", ["elo"], train, calib, test2, y_train, y_calib, y_test2, official_ll
    )
    assert art1["platt"].coefficients == pytest.approx(art2["platt"].coefficients)

    # Changing CALIBRATION rows must change Platt's fit.
    calib2 = calib.copy()
    calib2[TARGET] = 1 - calib2[TARGET]  # flip calibration labels entirely
    y_calib2 = calib2[TARGET].astype(int).to_numpy()
    entry3, art3 = run_single_ablation(
        "elo", ["elo"], train, calib2, test, y_train, y_calib2, y_test, official_ll
    )
    assert art3["platt"].coefficients != pytest.approx(art1["platt"].coefficients)


# --------------------------------------------------------------------- 6
def test_leakage_guard_rejects_bad_columns():
    with pytest.raises(ValueError, match="Leakage guard"):
        check_features_allowed(["elo_diff", "winner"])
    with pytest.raises(ValueError, match="Leakage guard"):
        check_features_allowed(["elo_diff", TARGET])


# --------------------------------------------------------------------- 7
def test_report_generation_writes_valid_json(tmp_path):
    df = synthetic_step3c_features(400)
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)

    reduced_ablations = {
        "all": list(FEATURE_GROUPS),
        "only_elo": ["elo"],
        "all_minus_elo": [g for g in FEATURE_GROUPS if g != "elo"],
    }
    report = run_feature_diagnostics(
        input_csv=str(csv_path),
        output_dir=str(tmp_path / "reports"),
        n_permutation_repeats=2,
        ablations=reduced_ablations,
        baseline_path=str(BASELINE_PATH),
    )

    for key in (
        "generated_at", "official_baseline", "split", "feature_audit",
        "scaling_audit", "feature_groups", "ablations", "best_ablation",
        "worst_ablation", "groups_helpful", "groups_harmful", "groups_neutral",
        "top_positive_coefficients", "top_negative_coefficients",
        "top_permutation_importances", "weak_or_noisy_features",
        "leakage_checks", "verdict",
    ):
        assert key in report, f"{key} missing from report"

    out_dir = tmp_path / "reports"
    for fname in (
        "step5a_feature_diagnostics.json",
        "step5a_feature_diagnostics.md",
        "step5a_coefficients.csv",
    ):
        assert (out_dir / fname).exists(), fname

    loaded = json.loads((out_dir / "step5a_feature_diagnostics.json").read_text())
    assert loaded["split"]["n_train"] == report["split"]["n_train"]

    assert len(report["ablations"]) == 3
    names = {a["name"] for a in report["ablations"]}
    assert names == {"all", "only_elo", "all_minus_elo"}


# --------------------------------------------------------------------- 8
def test_coefficient_report_schema():
    df = synthetic_step3c_features(300)
    numeric, categorical = official_step3c_features()
    train, calib, test = chronological_three_way_split(df, train_frac=0.70, calibration_frac=0.15)
    pipeline = make_official_step3c_pipeline(numeric)
    pipeline.fit(train[numeric + categorical], train[TARGET].astype(int))

    coef_df = build_coefficient_report(pipeline, numeric, categorical)
    for col in (
        "feature", "group", "feature_type", "coefficient", "abs_coefficient",
        "sign", "rank", "imputer_median", "scaler_mean", "scaler_std",
    ):
        assert col in coef_df.columns

    assert list(coef_df["rank"]) == list(range(1, len(coef_df) + 1))
    assert (coef_df.sort_values("rank")["abs_coefficient"].diff().dropna() <= 1e-9).all()

    numeric_rows = coef_df[coef_df["feature_type"] == "numeric"]
    assert set(numeric_rows["feature"]) == set(numeric)
    assert numeric_rows["scaler_mean"].notna().all()

    assert coef_df[coef_df["feature_type"] == "one_hot"].empty


# --------------------------------------------------------------------- 9
def test_permutation_importance_schema_and_determinism():
    df = synthetic_step3c_features(300)
    official_ll = _official_log_loss()
    numeric, categorical = official_step3c_features()
    train, calib, test = chronological_three_way_split(df, train_frac=0.70, calibration_frac=0.15)
    y_train = train[TARGET].astype(int).to_numpy()
    y_calib = calib[TARGET].astype(int).to_numpy()
    y_test = test[TARGET].astype(int).to_numpy()

    entry, art = run_single_ablation(
        "all", list(FEATURE_GROUPS), train, calib, test, y_train, y_calib, y_test, official_ll
    )
    calibrated = CalibratedPipeline(art["pipeline"], art["platt"])
    cols = art["numeric"] + art["categorical"]

    rows1, baseline1 = permutation_importance_report(
        calibrated, test, cols, y_test, n_repeats=3, seed=42
    )
    rows2, baseline2 = permutation_importance_report(
        calibrated, test, cols, y_test, n_repeats=3, seed=42
    )

    assert baseline1 == pytest.approx(baseline2)
    for r1, r2 in zip(rows1, rows2):
        assert r1["feature"] == r2["feature"]
        assert r1["mean_importance"] == pytest.approx(r2["mean_importance"])
        assert r1["rank"] == r2["rank"]

    for row in rows1:
        for key in ("feature", "group", "mean_importance", "std", "n_repeats", "scoring", "rank"):
            assert key in row
    assert [r["rank"] for r in rows1] == list(range(1, len(rows1) + 1))
    assert sorted((r["mean_importance"] for r in rows1), reverse=True) == [
        r["mean_importance"] for r in rows1
    ]


# -------------------------------------------------------------------- 10
def test_classify_group_effects_sign_convention():
    """Known synthetic deltas exercise all three buckets of the classifier."""
    all_ll = 0.6000
    ablation_results = {
        "all": {"platt_log_loss": all_ll},
        # removal made log loss WORSE by more than the threshold -> group helpful
        "all_minus_elo": {"platt_log_loss": all_ll + 0.005},
        # removal made log loss BETTER by more than the threshold -> group harmful
        "all_minus_physical": {"platt_log_loss": all_ll - 0.005},
        # removal changed log loss by less than the threshold -> neutral/noise
        "all_minus_age": {"platt_log_loss": all_ll + 0.0005},
    }
    helpful, harmful, neutral = classify_group_effects(ablation_results)
    helpful_groups = {r["group"] for r in helpful}
    harmful_groups = {r["group"] for r in harmful}
    neutral_groups = {r["group"] for r in neutral}

    assert "elo" in helpful_groups
    assert "physical" in harmful_groups
    assert "age" in neutral_groups
    # every group appears in exactly one bucket
    assert helpful_groups | harmful_groups | neutral_groups >= {"elo", "physical", "age"}
    assert not (helpful_groups & harmful_groups)
    assert not (helpful_groups & neutral_groups)
    assert not (harmful_groups & neutral_groups)

    for r in helpful:
        assert r["delta_log_loss_if_removed"] > NOISE_THRESHOLD
    for r in harmful:
        assert r["delta_log_loss_if_removed"] < -NOISE_THRESHOLD
    for r in neutral:
        assert abs(r["delta_log_loss_if_removed"]) <= NOISE_THRESHOLD


# -------------------------------------------------------------------- 11
def test_permutation_importance_sign_and_no_mutation():
    """Permuting a strong known signal (elo_diff) increases test log loss;
    the caller's test dataframe must come back unchanged."""
    df = synthetic_step3c_features(400)
    # The promoted C=0.003 elastic-net model deliberately shrinks weak signal
    # to zero. Make this fixture's single feature unambiguously predictive so
    # this remains a test of permutation importance, not regularization.
    df[TARGET] = (df["elo_diff"] > 0).astype(int)
    official_ll = _official_log_loss()
    train, calib, test = chronological_three_way_split(df, train_frac=0.70, calibration_frac=0.15)
    y_train = train[TARGET].astype(int).to_numpy()
    y_calib = calib[TARGET].astype(int).to_numpy()
    y_test = test[TARGET].astype(int).to_numpy()

    entry, art = run_single_ablation(
        "only_elo", ["elo"], train, calib, test, y_train, y_calib, y_test, official_ll
    )
    calibrated = CalibratedPipeline(art["pipeline"], art["platt"])
    cols = art["numeric"] + art["categorical"]

    test_before = test.copy(deep=True)
    rows, _ = permutation_importance_report(calibrated, test, cols, y_test, n_repeats=5, seed=42)

    # permutation_importance_report must not mutate the caller's dataframe
    pd.testing.assert_frame_equal(test, test_before)

    elo_row = next(r for r in rows if r["feature"] == "elo_diff")
    assert elo_row["mean_importance"] > 0
