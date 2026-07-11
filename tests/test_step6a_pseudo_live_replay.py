"""Tests for Step 6A historical pseudo-live replay.

Focus: the replay split builder uses only strictly-earlier fights, the event is
excluded from train/calibration, calibration is strictly between training and
the event, insufficient history skips, ledger + report schema, no-overwrite
guard, probability orientation, the metric/bucket/rolling calculators, target/
result leakage prevention, and official-config preservation (incl. weight_class
dropped). Fixtures are small synthetic frames; no test runs the full replay.
"""

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufc_pipeline.feature_diagnostics import official_step3c_features  # noqa: E402
from ufc_pipeline.modeling import TARGET, check_features_allowed  # noqa: E402
from ufc_pipeline.step5b_regularization_search import WEIGHT_CLASS_COLUMN  # noqa: E402
from ufc_pipeline.step6a_pseudo_live_replay import (  # noqa: E402
    ID_DATE_COLUMNS,
    MISSING_EVENT_NAME,
    OFFICIAL_LR_PARAMS,
    _row_brier,
    _row_log_loss,
    _window_metrics,
    build_replay_split,
    calibration_buckets,
    event_drift,
    fit_official_model,
    iter_event_groups,
    load_official_replay_config,
    model_classes,
    run_pseudo_live_replay,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "benchmarks" / "official_baseline.json"
REAL_CSV = REPO_ROOT / "data" / "processed" / "ufc_prefight_features_step3c.csv"


def synthetic_replay_df(n_events: int = 40, per_event: int = 12, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    numeric_all, _ = official_step3c_features()
    rows, fid = [], 1
    for e in range(n_events):
        date = (pd.Timestamp("2015-01-01") + pd.Timedelta(days=14 * e)).strftime("%Y-%m-%d")
        for _ in range(per_event):
            elo, age = rng.normal(0, 80), rng.normal(0, 5)
            p = 1 / (1 + np.exp(-(0.008 * elo - 0.06 * age)))
            won = int(rng.random() < p)
            rows.append({
                "fight_id": fid, "date": date, "event": f"UFC {e}",
                "fighter_a": f"A{fid}", "fighter_b": f"B{fid}",
                "winner": f"A{fid}" if won else f"B{fid}", TARGET: won,
                WEIGHT_CLASS_COLUMN: rng.choice(["Lightweight", "Welterweight"]),
                "fighter_a_expected_win_prob": p, "elo_diff": elo, "age_diff": age,
            })
            fid += 1
    df = pd.DataFrame(rows)
    flags = {"fighter_a_no_prior_stats", "fighter_b_no_prior_stats", "matchup_history_missing"}
    for col in numeric_all:
        if col in df.columns:
            continue
        df[col] = rng.integers(0, 2, len(df)) if col in flags else rng.normal(0, 1.0, len(df))
    for col in ("reach_diff", "height_diff"):
        if col in df:
            df.loc[df.sample(frac=0.1, random_state=1).index, col] = np.nan
    return df.sort_values(["date", "fight_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Split builder
# ---------------------------------------------------------------------------

def test_build_replay_split_uses_only_past_fights_and_excludes_event():
    df = synthetic_replay_df()
    event_date = "2015-09-10"  # event with ample warmup history
    event_ids = set(df[df["date"] == event_date]["fight_id"])
    assert event_ids  # sanity: this date really is an event
    split = build_replay_split(df, event_date, min_train_fights=100, calibration_fights=60)
    assert split is not None
    train, calib = split["training"], split["calibration"]
    # strictly earlier than the event
    assert train["date"].max() < event_date
    assert calib["date"].max() < event_date
    # event rows never appear in train or calibration
    assert not (event_ids & set(train["fight_id"]))
    assert not (event_ids & set(calib["fight_id"]))
    # train and calibration are disjoint
    assert not (set(train["fight_id"]) & set(calib["fight_id"]))


def test_calibration_is_after_training_and_is_most_recent_window():
    df = synthetic_replay_df()
    event_date = "2015-08-13"
    split = build_replay_split(df, event_date, min_train_fights=100, calibration_fights=60)
    train, calib = split["training"], split["calibration"]
    assert train["date"].max() <= calib["date"].min()  # calibration strictly after training window
    past = df[df["date"] < event_date]
    assert len(calib) == 60
    assert set(calib["fight_id"]) == set(past.iloc[-60:]["fight_id"])  # most recent 60


def test_insufficient_history_returns_none():
    df = synthetic_replay_df()
    early_event = df["date"].min()
    assert build_replay_split(df, early_event, min_train_fights=100, calibration_fights=60) is None
    # right at the boundary but not enough
    assert build_replay_split(df, "2015-01-15", min_train_fights=1000, calibration_fights=1000) is None


def test_event_groups_retain_rows_with_missing_event_name():
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-08"],
            "event": [np.nan, "UFC 2"],
            "fight_id": [1, 2],
        }
    )

    groups = list(iter_event_groups(df))

    assert [(date, event, len(rows)) for date, event, rows in groups] == [
        ("2020-01-01", MISSING_EVENT_NAME, 1),
        ("2020-01-08", "UFC 2", 1),
    ]


def test_event_drift_counts_low_history_once_per_row():
    base_numeric, _ = official_step3c_features()
    training = pd.DataFrame({feature: [0.0, 0.0] for feature in base_numeric})
    event = pd.DataFrame({feature: [0.0] for feature in base_numeric})
    event.loc[0, "fighter_a_no_prior_stats"] = 1
    event.loc[0, "fighter_b_no_prior_stats"] = 1
    event.loc[0, "matchup_history_missing"] = 1

    result = event_drift(event, training, base_numeric)

    assert result["n_low_history_flagged_rows"] == 1


# ---------------------------------------------------------------------------
# Official config
# ---------------------------------------------------------------------------

def test_official_config_loaded_and_weight_class_dropped():
    cfg = load_official_replay_config(str(BASELINE_PATH))
    assert cfg["hyperparameters"] == OFFICIAL_LR_PARAMS
    assert cfg["raw_weight_class"] == "dropped"
    assert cfg["division_interactions"] == "not_included"
    assert cfg["calibration_method"] == "platt"
    assert cfg["model_version"] == "step5c_stronger_regularized_lr_drop_weight_class_platt"
    assert isinstance(cfg["feature_schema_version"], str) and cfg["feature_schema_version"].startswith("step3c-")
    # config must not carry the whole baseline; it only reads test metrics.
    assert cfg["static_backtest_test_metrics"] is not None


def test_target_and_identifier_columns_excluded_from_features():
    base_numeric, _ = official_step3c_features()
    check_features_allowed(base_numeric)  # raises if a forbidden/result column sneaks in
    assert TARGET not in base_numeric
    assert not (ID_DATE_COLUMNS & set(base_numeric))
    assert WEIGHT_CLASS_COLUMN not in base_numeric  # dropped for the official model


# ---------------------------------------------------------------------------
# Probability orientation
# ---------------------------------------------------------------------------

def test_probability_orientation_is_p_fighter_a_wins():
    df = synthetic_replay_df()
    base_numeric, _ = official_step3c_features()
    split = build_replay_split(df, "2015-08-13", min_train_fights=100, calibration_fights=60)
    pipeline, platt = fit_official_model(split["training"], split["calibration"], base_numeric, max_iter=800)
    assert model_classes(pipeline) == [0, 1]  # column 1 of predict_proba is P(fighter_a_won == 1)


# ---------------------------------------------------------------------------
# Metric calculators
# ---------------------------------------------------------------------------

def test_row_log_loss_matches_sklearn():
    y = np.array([1, 0, 1, 0, 1])
    p = np.array([0.9, 0.2, 0.6, 0.55, 0.3])
    assert _row_log_loss(y, p).mean() == pytest.approx(log_loss(y, p, labels=[0, 1]))


def test_row_brier_matches_sklearn():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.3, 0.6, 0.1])
    assert _row_brier(y, p).mean() == pytest.approx(brier_score_loss(y, p))


def test_calibration_buckets_compute_gap_and_counts():
    # 10 predictions at 0.62 with 6 wins -> bucket 0.60-0.65
    y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    p = np.full(10, 0.62)
    buckets = {b["bucket"]: b for b in calibration_buckets(y, p)}
    b = buckets["0.60-0.65"]
    assert b["count"] == 10
    assert b["avg_predicted_prob"] == pytest.approx(0.62)
    assert b["actual_a_win_rate"] == pytest.approx(0.6)
    assert b["calibration_gap"] == pytest.approx(0.02)
    # a bucket with no predictions reports count 0 and null stats
    assert buckets["0.45-0.50"]["count"] == 0
    assert buckets["0.45-0.50"]["avg_predicted_prob"] is None


def test_window_metrics_rolling_calculation():
    y = np.array([1, 0, 1, 1, 0, 1])
    p = np.array([0.7, 0.4, 0.6, 0.55, 0.45, 0.8])
    m = _window_metrics(y, p)
    assert m["n"] == 6
    assert m["accuracy"] == pytest.approx(np.mean((p >= 0.5).astype(int) == y))
    assert m["avg_predicted_prob"] == pytest.approx(p.mean())
    assert m["actual_a_win_rate"] == pytest.approx(y.mean())
    assert m["log_loss"] == pytest.approx(log_loss(y, p, labels=[0, 1]))


# ---------------------------------------------------------------------------
# Full small run: ledger + report schema, overwrite guard, leakage
# ---------------------------------------------------------------------------

LEDGER_REQUIRED_COLUMNS = [
    "replay_id", "event_date", "event_name", "fighter_a", "fighter_b", "model_version",
    "calibration_version", "feature_schema_version", "prediction_mode", "train_start_date",
    "train_end_date", "train_rows", "calibration_start_date", "calibration_end_date",
    "calibration_rows", "predicted_probability_a", "predicted_probability_b", "target_a_win",
    "winner", "log_loss", "brier", "correct_prediction", "prediction_timestamp_utc",
    "resolved_timestamp_utc", "notes",
]


@pytest.fixture(scope="module")
def replay_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("step6a")
    csv_path = tmp / "features.csv"
    synthetic_replay_df().to_csv(csv_path, index=False)
    ledger = tmp / "reports" / "replay" / "ledger.csv"
    report = run_pseudo_live_replay(
        input_csv=str(csv_path), output_dir=str(tmp / "reports"), ledger_path=str(ledger),
        min_train_fights=150, calibration_fights=100, overwrite=True, max_iter=800,
        baseline_path=str(BASELINE_PATH),
    )
    return report, tmp, ledger, csv_path


def test_ledger_schema(replay_run):
    _report, _tmp, ledger, _csv = replay_run
    df = pd.read_csv(ledger)
    for col in LEDGER_REQUIRED_COLUMNS:
        assert col in df.columns, f"ledger missing {col}"
    assert (df["prediction_mode"] == "pseudo_live_replay").all()
    assert np.allclose(df["predicted_probability_a"] + df["predicted_probability_b"], 1.0)
    assert df["model_version"].nunique() == 1


def test_report_schema_and_baseline_preserved(replay_run):
    report, tmp, _ledger, _csv = replay_run
    for key in (
        "generated_at", "prediction_mode", "official_model", "replay_configuration",
        "date_range_replayed", "n_events_replayed", "n_fights_predicted", "n_events_skipped",
        "skipped_reasons", "overall_metrics", "calibration_buckets", "high_confidence_two_sided",
        "confidence_buckets", "performance_by_year", "rolling_metrics", "worst_events_by_log_loss",
        "best_events_by_log_loss", "comparison_to_static_backtest", "drift_data_quality",
        "leakage_protocol_checks", "interpretation_warnings",
    ):
        assert key in report, f"{key} missing from Step 6A report"
    assert report["changes_official_model"] is False
    assert report["changes_official_baseline_file"] is False
    assert report["writes_data_processed"] is False
    assert report["prediction_mode"] == "pseudo_live_replay"
    assert (tmp / "reports" / "step6a_pseudo_live_replay.json").exists()
    assert (tmp / "reports" / "step6a_pseudo_live_replay.md").exists()
    assert (tmp / "reports" / "step6a_rolling_metrics.csv").exists()
    # baseline file was only read, never rewritten
    with open(BASELINE_PATH) as fh:
        assert report["official_model"]["static_backtest_test_metrics"] == json.load(fh)["official_model"]["test_metrics"]


def test_leakage_checks_all_pass(replay_run):
    report, _tmp, _ledger, _csv = replay_run
    for name, entry in report["leakage_protocol_checks"].items():
        assert entry["status"] == "pass", f"{name} did not pass: {entry}"


def test_no_overwrite_without_flag(replay_run):
    _report, _tmp, ledger, csv = replay_run
    assert ledger.exists()
    with pytest.raises(FileExistsError):
        run_pseudo_live_replay(
            input_csv=str(csv), output_dir=str(ledger.parent.parent), ledger_path=str(ledger),
            min_train_fights=150, calibration_fights=100, overwrite=False, baseline_path=str(BASELINE_PATH),
        )


def test_run_signature_supports_required_options():
    params = set(inspect.signature(run_pseudo_live_replay).parameters)
    for opt in ("min_train_fights", "calibration_fights", "start_date", "end_date",
                "max_events", "overwrite", "random_seed", "input_csv", "output_dir"):
        assert opt in params


# ---------------------------------------------------------------------------
# Real-data quick smoke (few events)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_CSV.exists(), reason="processed Step 3C CSV not present")
def test_real_data_small_replay_runs(tmp_path):
    ledger = tmp_path / "reports" / "replay" / "ledger.csv"
    report = run_pseudo_live_replay(
        input_csv=str(REAL_CSV), output_dir=str(tmp_path / "reports"), ledger_path=str(ledger),
        min_train_fights=3000, calibration_fights=1282, max_events=3, overwrite=True,
        baseline_path=str(BASELINE_PATH),
    )
    assert report["n_events_replayed"] == 3
    assert report["n_fights_predicted"] > 0
    assert np.isfinite(report["overall_metrics"]["log_loss"])
    for entry in report["leakage_protocol_checks"].values():
        assert entry["status"] == "pass"
