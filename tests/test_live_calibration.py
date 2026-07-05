"""Tests for rolling/live calibration (Part B).

Covers the 10 required behaviors:
 1. The rolling calibration window uses the most recent rows by date.
 2. Training rows are strictly before calibration_start.
 3. Calibration rows are within the rolling window.
 4. --as-of-date correctly changes the calibration window.
 5. The min_calibration_rows warning appears when needed.
 6. Calibrated probabilities are between 0 and 1.
 7. All live reports are written.
 8. The existing backtest calibration mode still works alongside live mode.
 9. No random split: live calibration is deterministic and chronological.
10. No future fights enter the window when --as-of-date is provided.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufc_pipeline.calibration import (  # noqa: E402
    run_calibration,
    run_live_calibration,
)
from tests.test_calibration import biased_synthetic_features  # noqa: E402


def run_live(df, tmp_path, **kwargs):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv = tmp_path / "features.csv"
    df.to_csv(csv, index=False)
    return run_live_calibration(
        input_csv=str(csv),
        output_dir=str(tmp_path / "processed"),
        model_dir=str(tmp_path / "models"),
        **kwargs,
    )


# --------------------------------------------------------------------- 1
def test_window_uses_most_recent_rows(tmp_path):
    df = biased_synthetic_features(600)  # weekly fights from 2015-01-01
    report = run_live(df, tmp_path, calibration_window_days=365,
                      min_calibration_rows=10)
    cfg = report["config"]
    latest = df["date"].max()
    assert cfg["as_of_date"] == latest
    assert cfg["calibration_dates"][1] == latest  # window ends at the newest fight
    # ~52 weekly fights fit in 365 days
    assert 50 <= cfg["n_calibration"] <= 54


# --------------------------------------------------------------------- 2 & 3
def test_train_strictly_before_window_and_calib_inside(tmp_path):
    df = biased_synthetic_features(600)
    report = run_live(df, tmp_path, calibration_window_days=365,
                      min_calibration_rows=10)
    cfg = report["config"]
    assert cfg["train_dates"][1] < cfg["calibration_start"]     # strict
    assert cfg["calibration_dates"][0] >= cfg["calibration_start"]
    assert cfg["calibration_dates"][1] <= cfg["as_of_date"]
    assert cfg["n_train"] + cfg["n_calibration"] == len(df)
    # prediction rows are exactly the window rows
    preds = pd.read_csv(Path(tmp_path) / "processed" / "live_calibration_predictions.csv")
    assert len(preds) == cfg["n_calibration"]
    assert preds["date"].min() >= cfg["calibration_start"]
    assert (preds["split"] == "calibration").all()


# --------------------------------------------------------------------- 4
def test_as_of_date_moves_the_window(tmp_path):
    df = biased_synthetic_features(600)
    r_late = run_live(df, tmp_path / "late", calibration_window_days=365,
                      min_calibration_rows=10)
    r_early = run_live(df, tmp_path / "early", calibration_window_days=365,
                       min_calibration_rows=10, as_of_date="2020-01-01")
    assert r_early["config"]["as_of_date"] == "2020-01-01"
    assert r_early["config"]["calibration_start"] == "2019-01-01"
    assert r_early["config"]["calibration_dates"][1] <= "2020-01-01"
    assert r_early["config"]["n_train"] < r_late["config"]["n_train"]


# --------------------------------------------------------------------- 5
def test_min_rows_warning(tmp_path):
    df = biased_synthetic_features(600)
    report = run_live(df, tmp_path, calibration_window_days=365,
                      min_calibration_rows=500)  # window holds only ~52
    assert any("min-calibration-rows" in w for w in report["warnings"])
    saved = json.loads(
        (Path(tmp_path) / "processed" / "live_calibration_report.json").read_text()
    )
    assert saved["warnings"] == report["warnings"]
    # and no warning when the window is big enough
    report2 = run_live(df, tmp_path / "ok", calibration_window_days=365,
                       min_calibration_rows=10)
    assert report2["warnings"] == []


# --------------------------------------------------------------------- 6
def test_probabilities_in_unit_interval(tmp_path):
    df = biased_synthetic_features(600)
    for method in ("platt", "isotonic"):
        run_live(df, tmp_path / method, calibration_window_days=730,
                 min_calibration_rows=10, method=method)
        preds = pd.read_csv(
            Path(tmp_path) / method / "processed" / "live_calibration_predictions.csv"
        )
        for col in ("raw_probability", f"{method}_probability"):
            assert preds[col].notna().all()
            assert (preds[col] > 0).all() and (preds[col] < 1).all()


# --------------------------------------------------------------------- 7
def test_live_reports_written(tmp_path):
    df = biased_synthetic_features(600)
    run_live(df, tmp_path, calibration_window_days=730, min_calibration_rows=10)
    processed = Path(tmp_path) / "processed"
    models = Path(tmp_path) / "models"
    for f in ("live_calibration_report.json", "live_calibration_table.csv",
              "live_calibration_predictions.csv"):
        assert (processed / f).exists(), f
    for f in ("step3b_logistic_regression_live_uncalibrated.joblib",
              "step3b_logistic_regression_live_platt_calibrated.joblib"):
        assert (models / f).exists(), f

    report = json.loads((processed / "live_calibration_report.json").read_text())
    assert report["mode"] == "live" and report["method"] == "platt"
    diag = report["calibration_window_diagnostics"]
    for label in ("raw", "platt"):
        for key in ("log_loss", "brier_score", "high_confidence", "calibration"):
            assert key in diag[label]
    # the required window diagnostics are all present
    for thr in ("above_0.7", "above_0.8"):
        assert thr in diag["raw"]["high_confidence"]
        assert thr in diag["platt"]["high_confidence"]
    table = pd.read_csv(processed / "live_calibration_table.csv")
    assert set(table["model"]) == {"raw", "platt"}


# --------------------------------------------------------------------- 8
def test_backtest_mode_still_works_alongside_live(tmp_path):
    df = biased_synthetic_features(600)
    csv = Path(tmp_path) / "features.csv"
    df.to_csv(csv, index=False)
    backtest = run_calibration(
        input_csv=str(csv),
        output_dir=str(Path(tmp_path) / "processed"),
        model_dir=str(Path(tmp_path) / "models"),
    )
    live = run_live_calibration(
        input_csv=str(csv),
        output_dir=str(Path(tmp_path) / "processed"),
        model_dir=str(Path(tmp_path) / "models"),
        calibration_window_days=730,
        min_calibration_rows=10,
    )
    # both output sets coexist: live_* names never overwrite backtest files
    processed = Path(tmp_path) / "processed"
    assert (processed / "calibration_comparison.json").exists()
    assert (processed / "live_calibration_report.json").exists()
    assert "models" in backtest and live["mode"] == "live"
    # backtest metrics come from a held-out test window, live has none
    assert backtest["config"]["n_test"] > 0
    assert "n_test" not in live["config"]


# --------------------------------------------------------------------- 9
def test_live_calibration_deterministic_and_chronological(tmp_path):
    df = biased_synthetic_features(600)
    r1 = run_live(df, tmp_path / "a", calibration_window_days=730,
                  min_calibration_rows=10)
    # shuffle the input rows: chronological sorting must make it identical
    shuffled = df.sample(frac=1.0, random_state=7)
    r2 = run_live(shuffled, tmp_path / "b", calibration_window_days=730,
                  min_calibration_rows=10)
    p1 = pd.read_csv(Path(tmp_path) / "a" / "processed" / "live_calibration_predictions.csv")
    p2 = pd.read_csv(Path(tmp_path) / "b" / "processed" / "live_calibration_predictions.csv")
    pd.testing.assert_frame_equal(p1, p2)
    d1 = r1["calibration_window_diagnostics"]["platt"]["log_loss"]
    d2 = r2["calibration_window_diagnostics"]["platt"]["log_loss"]
    assert d1 == pytest.approx(d2, abs=1e-12)


# -------------------------------------------------------------------- 10
def test_no_future_fights_with_as_of_date(tmp_path):
    df = biased_synthetic_features(600)
    as_of = "2020-06-01"
    report = run_live(df, tmp_path, calibration_window_days=365,
                      min_calibration_rows=10, as_of_date=as_of)
    n_future = int((df["date"] > as_of).sum())
    assert n_future > 0  # the fixture really has later fights
    assert report["config"]["future_fights_excluded"] == n_future
    assert report["config"]["n_train"] + report["config"]["n_calibration"] \
        == len(df) - n_future
    preds = pd.read_csv(Path(tmp_path) / "processed" / "live_calibration_predictions.csv")
    assert preds["date"].max() <= as_of


# ------------------------------------------------------------ extras
def test_bad_method_rejected(tmp_path):
    df = biased_synthetic_features(100)
    with pytest.raises(ValueError, match="method"):
        run_live(df, tmp_path, method="temperature")


def test_empty_window_rejected(tmp_path):
    df = biased_synthetic_features(100)
    with pytest.raises(ValueError, match="Empty live split"):
        # a window covering ALL fights leaves no training rows
        run_live(df, tmp_path, calibration_window_days=100000)
