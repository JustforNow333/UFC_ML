"""Tests for the probability calibration layer (Step 4B).

Covers the 10 required behaviors:
 1. Chronological three-way split order is respected (never random).
 2. The base model trains only on train rows.
 3. The calibrator fits only on calibration rows.
 4. Test metrics are computed only on test rows.
 5. Platt and isotonic output probabilities strictly inside (0, 1).
 6. Calibration does not change the prediction row count.
 7. All calibration output files are created.
 8. Log loss is computed from calibrated probabilities, not class labels.
 9. Small/empty high-confidence buckets do not crash the diagnostics.
10. Forbidden/leaky features are still rejected by the leakage guard.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufc_pipeline.calibration import (  # noqa: E402
    CalibratedPipeline,
    IsotonicCalibrator,
    PlattCalibrator,
    chronological_three_way_split,
    clip_probabilities,
    high_confidence_diagnostics,
    run_calibration,
)
from ufc_pipeline.modeling import (  # noqa: E402
    TARGET,
    make_logistic_pipeline,
)
from tests.test_modeling import synthetic_features  # noqa: E402


def biased_synthetic_features(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic dataset whose fighter-A win rate DROPS in the newest era,
    mimicking the red-corner drift that motivated calibration."""
    df = synthetic_features(n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    late = df.index >= int(n * 0.7)
    # Flip some late-era fighter-A wins to losses -> trained model becomes
    # overconfident toward A on the calibration/test windows.
    flip = late & (df[TARGET] == 1) & (rng.random(n) < 0.25)
    df.loc[flip, TARGET] = 0
    df.loc[flip, "winner"] = df.loc[flip, "fighter_b"]
    return df


def run_workflow(df, tmp_path, **kwargs):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv = tmp_path / "features.csv"
    df.to_csv(csv, index=False)
    return run_calibration(
        input_csv=str(csv),
        output_dir=str(tmp_path / "processed"),
        model_dir=str(tmp_path / "models"),
        **kwargs,
    )


# --------------------------------------------------------------------- 1
def test_three_way_split_is_chronological():
    df = biased_synthetic_features(200)
    shuffled = df.sample(frac=1.0, random_state=1)  # feed it out of order
    train, calib, test = chronological_three_way_split(
        shuffled, train_frac=0.70, calibration_frac=0.15
    )
    assert len(train) == 140 and len(calib) == 30 and len(test) == 30
    assert train["date"].max() <= calib["date"].min()  # no time overlap
    assert calib["date"].max() <= test["date"].min()
    assert list(test["date"]) == sorted(test["date"])


def test_three_way_split_by_date():
    df = biased_synthetic_features(200)
    train, calib, test = chronological_three_way_split(
        df, train_end_date="2017-01-01", calibration_end_date="2018-01-01"
    )
    assert (train["date"] < "2017-01-01").all()
    assert ((calib["date"] >= "2017-01-01") & (calib["date"] < "2018-01-01")).all()
    assert (test["date"] >= "2018-01-01").all()


def test_three_way_split_rejects_misuse():
    df = biased_synthetic_features(100)
    with pytest.raises(ValueError, match="BOTH"):
        chronological_three_way_split(df, train_end_date="2016-01-01")
    with pytest.raises(ValueError, match="earlier"):
        chronological_three_way_split(
            df, train_end_date="2018-01-01", calibration_end_date="2016-01-01"
        )
    with pytest.raises(ValueError, match="sum"):
        chronological_three_way_split(df, train_frac=0.9, calibration_frac=0.2)
    with pytest.raises(ValueError, match="Empty split"):
        chronological_three_way_split(
            df, train_end_date="1900-01-01", calibration_end_date="1900-02-01"
        )


# --------------------------------------------------------------------- 2
def test_base_model_trains_only_on_train_rows(tmp_path):
    df = biased_synthetic_features()
    results = run_workflow(df, tmp_path)
    cfg = results["config"]

    # Refit a pipeline on ONLY the train window; it must reproduce the saved
    # base model's test probabilities exactly (same data + random_state).
    train, calib, test = chronological_three_way_split(
        df[df[TARGET].notna()], train_frac=0.70, calibration_frac=0.15
    )
    assert cfg["n_train"] == len(train)
    cols = cfg["features_numeric"] + cfg["features_categorical"]
    manual = make_logistic_pipeline(cfg["features_numeric"], cfg["features_categorical"])
    manual.fit(train[cols], train[TARGET].astype(int))
    manual_probs = manual.predict_proba(test[cols])[:, 1]

    preds = pd.read_csv(Path(tmp_path) / "processed" / "calibration_predictions.csv")
    np.testing.assert_allclose(
        preds["uncalibrated_probability"].to_numpy(), manual_probs, atol=1e-9
    )


# --------------------------------------------------------------------- 3
def test_calibrator_fits_only_on_calibration_rows(tmp_path):
    df = biased_synthetic_features()
    results = run_workflow(df, tmp_path)
    cfg = results["config"]

    train, calib, test = chronological_three_way_split(
        df, train_frac=0.70, calibration_frac=0.15
    )
    assert cfg["n_calibration"] == len(calib)
    cols = cfg["features_numeric"] + cfg["features_categorical"]
    manual = make_logistic_pipeline(cfg["features_numeric"], cfg["features_categorical"])
    manual.fit(train[cols], train[TARGET].astype(int))
    calib_probs = manual.predict_proba(calib[cols])[:, 1]
    test_probs = manual.predict_proba(test[cols])[:, 1]

    # Refit calibrators on ONLY the calibration window -> identical output.
    platt = PlattCalibrator().fit(calib_probs, calib[TARGET].astype(int).to_numpy())
    iso = IsotonicCalibrator().fit(calib_probs, calib[TARGET].astype(int).to_numpy())
    preds = pd.read_csv(Path(tmp_path) / "processed" / "calibration_predictions.csv")
    np.testing.assert_allclose(
        preds["platt_probability"].to_numpy(), platt.transform(test_probs), atol=1e-9
    )
    np.testing.assert_allclose(
        preds["isotonic_probability"].to_numpy(), iso.transform(test_probs), atol=1e-9
    )


# --------------------------------------------------------------------- 4
def test_metrics_use_only_test_rows(tmp_path):
    df = biased_synthetic_features()
    results = run_workflow(df, tmp_path)
    _, _, test = chronological_three_way_split(df, train_frac=0.70, calibration_frac=0.15)

    for entry in results["models"].values():
        assert entry["n"] == len(test)
    preds = pd.read_csv(Path(tmp_path) / "processed" / "calibration_predictions.csv")
    assert len(preds) == len(test)
    assert set(preds["fight_id"]) == set(test["fight_id"])
    assert (preds["split"] == "test").all()
    # test window starts after the calibration window ends:
    assert results["config"]["test_dates"][0] >= results["config"]["calibration_dates"][1]


# --------------------------------------------------------------------- 5
def test_calibrated_probabilities_strictly_inside_unit_interval(tmp_path):
    run_workflow(biased_synthetic_features(), tmp_path)
    preds = pd.read_csv(Path(tmp_path) / "processed" / "calibration_predictions.csv")
    for col in ("uncalibrated_probability", "platt_probability", "isotonic_probability"):
        p = preds[col]
        assert p.notna().all()
        assert (p > 0.0).all() and (p < 1.0).all()  # clipped away from 0/1


def test_clip_probabilities_handles_extremes():
    p = clip_probabilities(np.array([0.0, 1.0, 0.5, -0.1, 1.1]))
    assert (p > 0).all() and (p < 1).all()
    assert p[2] == 0.5


# --------------------------------------------------------------------- 6
def test_calibration_preserves_row_count(tmp_path):
    df = biased_synthetic_features()
    results = run_workflow(df, tmp_path)
    n_test = results["config"]["n_test"]
    preds = pd.read_csv(Path(tmp_path) / "processed" / "calibration_predictions.csv")
    assert len(preds) == n_test
    for entry in results["models"].values():
        assert entry["n"] == n_test
        assert sum(b["n_predictions"] for b in entry["calibration"]) == n_test


# --------------------------------------------------------------------- 7
def test_output_files_created(tmp_path):
    run_workflow(biased_synthetic_features(), tmp_path)
    processed = Path(tmp_path) / "processed"
    models = Path(tmp_path) / "models"
    for f in ("calibration_comparison.json", "calibration_predictions.csv",
              "calibration_tables.csv"):
        assert (processed / f).exists(), f
    for f in ("step3b_logistic_regression_uncalibrated.joblib",
              "step3b_logistic_regression_platt_calibrated.joblib",
              "step3b_logistic_regression_isotonic_calibrated.joblib"):
        assert (models / f).exists(), f

    # saved calibrated pipelines actually predict calibrated probabilities
    preds = pd.read_csv(processed / "calibration_predictions.csv")
    expected_cols = [
        "fight_id", "date", "fighter_a", "fighter_b", "target",
        "uncalibrated_probability", "platt_probability", "isotonic_probability",
        "uncalibrated_prediction", "platt_prediction", "isotonic_prediction",
        "split",
    ]
    assert list(preds.columns) == expected_cols
    loaded = joblib.load(models / "step3b_logistic_regression_platt_calibrated.joblib")
    assert isinstance(loaded, CalibratedPipeline)

    tables = pd.read_csv(processed / "calibration_tables.csv")
    assert set(tables["model"]) == {"uncalibrated", "platt", "isotonic"}
    assert len(tables) == 30  # 3 models x 10 buckets


# --------------------------------------------------------------------- 8
def test_log_loss_from_probabilities_not_labels(tmp_path):
    df = biased_synthetic_features()
    results = run_workflow(df, tmp_path)
    preds = pd.read_csv(Path(tmp_path) / "processed" / "calibration_predictions.csv")
    y = preds["target"].to_numpy(dtype=int)
    for name in ("uncalibrated", "platt", "isotonic"):
        probs = preds[f"{name}_probability"].to_numpy()
        labels = preds[f"{name}_prediction"].to_numpy(dtype=float)
        reported = results["models"][name]["log_loss"]
        assert reported == pytest.approx(log_loss(y, probs, labels=[0, 1]), abs=1e-9)
        # ...and it does NOT equal log loss of the hard 0/1 class labels
        label_ll = log_loss(y, np.clip(labels, 1e-6, 1 - 1e-6), labels=[0, 1])
        assert reported != pytest.approx(label_ll, abs=1e-6)


# --------------------------------------------------------------------- 9
def test_small_and_empty_buckets_do_not_crash():
    # nothing above 0.8, nothing above 0.7 -> gaps are None, not a crash
    y = np.array([1, 0, 1, 0])
    probs = np.array([0.55, 0.45, 0.6, 0.5])
    diag = high_confidence_diagnostics(y, probs)
    assert diag["above_0.7"]["n_predictions"] == 0
    assert diag["above_0.7"]["calibration_gap"] is None
    # a single prediction above the threshold works too
    diag2 = high_confidence_diagnostics(np.array([1]), np.array([0.9]))
    assert diag2["above_0.8"]["n_predictions"] == 1
    assert diag2["above_0.8"]["calibration_gap"] == pytest.approx(-0.1)


def test_tiny_dataset_runs_end_to_end(tmp_path):
    # 60 rows -> 9-row calibration window; isotonic must not crash and the
    # verdict must flag isotonic overfit risk.
    results = run_workflow(biased_synthetic_features(60), tmp_path)
    assert results["verdict"]["isotonic_overfit_risk"] is True


# --------------------------------------------------------------------- 10
def test_forbidden_features_still_rejected(tmp_path):
    df = biased_synthetic_features()
    df["red_odds"] = 1.5  # present in the CSV but never requested -> ignored
    results = run_workflow(df, tmp_path / "clean")
    used = results["config"]["features_numeric"] + results["config"]["features_categorical"]
    assert "red_odds" not in used and TARGET not in used

    # explicitly requesting a leaky feature hard-stops before training
    with pytest.raises(ValueError, match="Leakage guard"):
        run_workflow(df, tmp_path / "leaky",
                     numeric_features=["elo_diff", "red_odds"])
    df["fighter_a_post_elo"] = 1600.0
    with pytest.raises(ValueError, match="Leakage guard"):
        run_workflow(df, tmp_path / "post",
                     numeric_features=["elo_diff", "fighter_a_post_elo"])


# ------------------------------------------------------------ extras
def test_platt_reduces_overconfidence_on_drifted_data(tmp_path):
    """On data with deliberate late-era drift, Platt must improve test log
    loss over the uncalibrated model (this is the whole point)."""
    results = run_workflow(biased_synthetic_features(1200, seed=7), tmp_path)
    m = results["models"]
    assert m["platt"]["log_loss"] < m["uncalibrated"]["log_loss"]
    unc_gap = m["uncalibrated"]["high_confidence"]["above_0.7"]["calibration_gap"]
    platt_gap = m["platt"]["high_confidence"]["above_0.7"]["calibration_gap"]
    if unc_gap is not None and platt_gap is not None:
        assert abs(platt_gap) < abs(unc_gap)


def test_calibration_is_deterministic(tmp_path):
    df = biased_synthetic_features()
    r1 = run_workflow(df, tmp_path / "a")
    r2 = run_workflow(df, tmp_path / "b")
    p1 = pd.read_csv(Path(tmp_path) / "a" / "processed" / "calibration_predictions.csv")
    p2 = pd.read_csv(Path(tmp_path) / "b" / "processed" / "calibration_predictions.csv")
    pd.testing.assert_frame_equal(p1, p2)
    assert r1["verdict"]["best_method"] == r2["verdict"]["best_method"]


def test_comparison_json_structure(tmp_path):
    run_workflow(biased_synthetic_features(), tmp_path)
    data = json.loads(
        (Path(tmp_path) / "processed" / "calibration_comparison.json").read_text()
    )
    for key in ("config", "elo_baseline", "models", "verdict"):
        assert key in data
    for name in ("uncalibrated", "platt", "isotonic"):
        entry = data["models"][name]
        for metric in ("accuracy", "log_loss", "brier_score", "roc_auc",
                       "calibration", "high_confidence"):
            assert metric in entry
