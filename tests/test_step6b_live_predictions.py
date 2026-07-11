"""Tests for Step 6B forward live prediction ledger.

Covers input-schema validation, missing-feature / result-leakage / market-column
rejection, duplicate detection, append-only ledger behaviour and no-silent-
overwrite, prediction_id determinism/uniqueness, batch report schema, result
resolution (incl. reversed-order match, unmatched result/winner, already-resolved
protection), log loss / Brier calculation, the live model report schema with
small-sample warnings, drift schema, official-metadata loading, weight_class
dropped, and no-odds-column acceptance. Fixtures are tiny; the model is trained
once on a small synthetic CSV.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufc_pipeline.feature_diagnostics import official_step3c_features  # noqa: E402
from ufc_pipeline.modeling import TARGET  # noqa: E402
from ufc_pipeline.step5b_regularization_search import WEIGHT_CLASS_COLUMN  # noqa: E402
from ufc_pipeline import step6b_live_predictions as s6b  # noqa: E402
from ufc_pipeline.step6b_live_predictions import (  # noqa: E402
    LEDGER_COLUMNS,
    append_predictions,
    build_live_model_report,
    empty_ledger,
    load_official_model_config,
    make_prediction_id,
    resolve_predictions,
    run_live_predictions,
    run_resolution,
    train_official_model,
    validate_prediction_input,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "benchmarks" / "official_baseline.json"
BASE_NUMERIC, _ = official_step3c_features()
LOW_HISTORY = {"fighter_a_no_prior_stats", "fighter_b_no_prior_stats", "matchup_history_missing"}


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _feature_dict(rng) -> dict:
    elo, age = rng.normal(0, 80), rng.normal(0, 5)
    d = {"elo_diff": elo, "age_diff": age, "fighter_a_expected_win_prob": 1 / (1 + np.exp(-(0.008 * elo - 0.06 * age)))}
    for c in BASE_NUMERIC:
        if c not in d:
            d[c] = rng.integers(0, 2) if c in LOW_HISTORY else rng.normal(0, 1)
    return d


def write_training_csv(path, n=600, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        date = (pd.Timestamp("2012-01-01") + pd.Timedelta(days=3 * i)).strftime("%Y-%m-%d")
        feats = _feature_dict(rng)
        won = int(rng.random() < feats["fighter_a_expected_win_prob"])
        rows.append({"fight_id": i + 1, "date": date, "event": f"UFC {i}", "fighter_a": f"A{i}",
                     "fighter_b": f"B{i}", "winner": f"A{i}" if won else f"B{i}", TARGET: won,
                     WEIGHT_CLASS_COLUMN: rng.choice(["Lightweight", "Welterweight"]), **feats})
    pd.DataFrame(rows).to_csv(path, index=False)


def card_df(fighters, seed=99, event_date="2026-08-01", event_name="UFC 999: Test") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for fa, fb in fighters:
        rows.append({"event_date": event_date, "event_name": event_name, "fighter_a": fa, "fighter_b": fb, **_feature_dict(rng)})
    return pd.DataFrame(rows)


def ledger_row(event_date, fa, fb, status="pending", p=0.6, batch="B1", model="M1"):
    row = {c: "" for c in LEDGER_COLUMNS}
    row.update({
        "prediction_id": make_prediction_id(event_date, fa, fb, model, batch),
        "prediction_batch_id": batch, "prediction_mode": "live_forward",
        "event_date": event_date, "event_name": "E", "fighter_a": fa, "fighter_b": fb,
        "model_version": model, "calibration_version": "platt", "feature_schema_version": "step3c",
        "input_source": "x.csv", "predicted_probability_a": p, "predicted_probability_b": 1 - p,
        "prediction_timestamp_utc": "t", "status": status, "notes": "",
    })
    return row


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_valid_card_passes_validation():
    v = validate_prediction_input(card_df([("Jon", "Stipe"), ("Alex", "Jiri")]), BASE_NUMERIC)
    assert v["ok"] is True
    assert v["n_valid_rows"] == 2 and v["n_error_rows"] == 0


def test_missing_required_feature_detected():
    df = card_df([("A", "B")]).drop(columns=["elo_diff"])
    v = validate_prediction_input(df, BASE_NUMERIC)
    assert v["ok"] is False
    assert "elo_diff" in v["missing_required_features"]


def test_target_or_result_columns_rejected():
    df = card_df([("A", "B")])
    df["fighter_a_won"] = 1
    df["winner"] = "A"
    v = validate_prediction_input(df, BASE_NUMERIC)
    assert v["ok"] is False
    assert "fighter_a_won" in v["forbidden_result_columns"]
    assert "winner" in v["forbidden_result_columns"]


def test_market_columns_rejected():
    df = card_df([("A", "B")])
    df["closing_odds_a"] = 1.5
    v = validate_prediction_input(df, BASE_NUMERIC)
    assert v["ok"] is False
    assert "closing_odds_a" in v["market_columns"]


def test_per_row_errors_detected():
    df = card_df([("A", "A"), ("", "B"), ("Jon", "Stipe")])
    v = validate_prediction_input(df, BASE_NUMERIC)
    statuses = [r["status"] for r in v["row_status"]]
    assert statuses == ["error", "error", "valid"]
    assert any("fighter_a equals fighter_b" in r["reasons"] for r in v["row_status"])
    assert any("empty fighter_a" in r["reasons"] for r in v["row_status"])


def test_duplicate_fight_in_input_flagged():
    df = card_df([("Jon", "Stipe"), ("Jon", "Stipe")])
    v = validate_prediction_input(df, BASE_NUMERIC)
    assert v["row_status"][1]["status"] == "error"
    assert any("duplicate" in r for r in v["row_status"][1]["reasons"])


def test_feature_expected_win_prob_is_not_treated_as_result():
    # fighter_a_expected_win_prob contains 'win' but is a legit pre-fight feature.
    v = validate_prediction_input(card_df([("A", "B")]), BASE_NUMERIC)
    assert v["ok"] is True
    assert "fighter_a_expected_win_prob" not in v["forbidden_result_columns"]


# ---------------------------------------------------------------------------
# prediction_id + ledger append
# ---------------------------------------------------------------------------

def test_prediction_id_deterministic_and_unique():
    a = make_prediction_id("2026-08-01", "Jon", "Stipe", "M1", "B1")
    assert a == make_prediction_id("2026-08-01", "Jon", "Stipe", "M1", "B1")  # deterministic
    assert a == make_prediction_id("2026-08-01", "Stipe", "Jon", "M1", "B1")  # orientation-independent
    assert a != make_prediction_id("2026-08-01", "Jon", "Stipe", "M1", "B2")  # batch versions differ
    assert a != make_prediction_id("2026-08-01", "Jon", "Charles", "M1", "B1")  # different fight


def test_ledger_append_rejects_duplicate_by_default():
    led = empty_ledger()
    led, _ = append_predictions(led, [ledger_row("2026-01-01", "A", "B"), ledger_row("2026-01-01", "C", "D")], False, False)
    assert len(led) == 2
    led2, stats = append_predictions(led, [ledger_row("2026-01-01", "A", "B", batch="B2")], allow_duplicate=False, overwrite_pending=False)
    assert stats["n_rejected_duplicates"] == 1
    assert len(led2) == 2  # no silent overwrite / no append


def test_ledger_append_allow_duplicate_versions():
    led = empty_ledger()
    led, _ = append_predictions(led, [ledger_row("2026-01-01", "A", "B", batch="B1")], False, False)
    led2, stats = append_predictions(led, [ledger_row("2026-01-01", "A", "B", batch="B2")], allow_duplicate=True, overwrite_pending=False)
    assert stats["n_duplicates_versioned"] == 1
    assert len(led2) == 2  # original + 1 versioned copy


def test_ledger_append_overwrite_pending_replaces():
    led = empty_ledger()
    led, _ = append_predictions(led, [ledger_row("2026-01-01", "A", "B", batch="B1", p=0.6)], False, False)
    led2, stats = append_predictions(led, [ledger_row("2026-01-01", "A", "B", batch="B2", p=0.7)], allow_duplicate=False, overwrite_pending=True)
    assert stats["n_pending_overwritten"] == 1
    assert len(led2) == 1  # old pending replaced by the new one (not duplicated)
    assert float(led2[led2["status"] == "pending"]["predicted_probability_a"].iloc[-1]) == 0.7


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _results(rows):
    return pd.DataFrame(rows, columns=["event_date", "event_name", "fighter_a", "fighter_b", "winner"])


def test_resolution_scores_and_matches_reversed_order():
    led = pd.DataFrame([
        ledger_row("2026-08-01", "Jon", "Stipe", p=0.75),
        ledger_row("2026-08-01", "Alex", "Jiri", p=0.40),
    ], columns=LEDGER_COLUMNS)
    results = _results([
        ["2026-08-01", "UFC 999", "Jon", "Stipe", "Jon"],           # a wins
        ["2026-08-01", "UFC 999", "Jiri", "Alex", "Alex"],          # reversed order; a (Alex) wins
    ])
    updated, stats = resolve_predictions(led, results, result_source="r.csv")
    assert stats["n_resolved"] == 2
    jon = updated[updated["fighter_a"] == "Jon"].iloc[0]
    assert int(jon["target_a_win"]) == 1 and int(jon["correct_prediction"]) == 1
    assert float(jon["log_loss"]) == pytest.approx(log_loss([1], [0.75], labels=[0, 1]))
    assert float(jon["brier"]) == pytest.approx(brier_score_loss([1], [0.75], pos_label=1))
    alex = updated[updated["fighter_a"] == "Alex"].iloc[0]
    assert int(alex["target_a_win"]) == 1  # Alex (fighter_a) won despite reversed result order
    assert (updated["status"] == "resolved").all()


def test_resolution_reports_unmatched_and_ambiguous():
    led = pd.DataFrame([ledger_row("2026-08-01", "Jon", "Stipe", p=0.7)], columns=LEDGER_COLUMNS)
    results = _results([
        ["2026-08-01", "UFC 999", "Jon", "Stipe", "Ghost"],   # unmatched winner name
        ["2026-08-01", "UFC 999", "Zzz", "Yyy", "Zzz"],        # unmatched result (no prediction)
    ])
    updated, stats = resolve_predictions(led, results, result_source="r.csv")
    assert stats["n_resolved"] == 0
    assert len(stats["unmatched_winner"]) == 1
    assert stats["n_unmatched_results"] == 1
    assert (updated["status"] == "pending").all()  # nothing resolved


def test_already_resolved_rows_protected():
    resolved_row = ledger_row("2026-08-01", "Jon", "Stipe", status="resolved", p=0.7)
    resolved_row["target_a_win"] = 1
    led = pd.DataFrame([resolved_row], columns=LEDGER_COLUMNS)
    results = _results([["2026-08-01", "UFC 999", "Jon", "Stipe", "Stipe"]])  # would flip to 0
    updated, stats = resolve_predictions(led, results, overwrite_resolved=False)
    assert stats["n_already_resolved_skipped"] == 1
    assert stats["n_resolved"] == 0
    updated2, stats2 = resolve_predictions(led, results, overwrite_resolved=True)
    assert stats2["n_resolved"] == 1
    assert int(updated2.iloc[0]["target_a_win"]) == 0  # now overwritten


def test_resolution_requires_winner_column():
    led = pd.DataFrame([ledger_row("2026-08-01", "Jon", "Stipe")], columns=LEDGER_COLUMNS)
    bad = pd.DataFrame([{"event_date": "2026-08-01", "event_name": "E", "fighter_a": "Jon", "fighter_b": "Stipe"}])
    with pytest.raises(ValueError, match="winner"):
        resolve_predictions(led, bad)


# ---------------------------------------------------------------------------
# Official model metadata
# ---------------------------------------------------------------------------

def test_official_model_config_loaded_and_weight_class_dropped():
    cfg = load_official_model_config(str(BASELINE_PATH))
    assert cfg["hyperparameters"]["C"] == 0.003 and cfg["hyperparameters"]["l1_ratio"] == 0.1
    assert cfg["raw_weight_class"] == "dropped"
    assert cfg["calibration_method"] == "platt"
    assert cfg["model_version"] == "step5c_stronger_regularized_lr_drop_weight_class_platt"


def test_official_model_config_rejects_missing_baseline(tmp_path):
    with pytest.raises(FileNotFoundError, match="Official baseline file not found"):
        load_official_model_config(str(tmp_path / "missing.json"))


def _write_test_baseline(path: Path, frame: pd.DataFrame, n_train=42, n_calibration=9, n_test=9) -> None:
    frozen = frame.iloc[:n_train + n_calibration + n_test]
    train = frozen.iloc[:n_train]
    calibration = frozen.iloc[n_train:n_train + n_calibration]
    test = frozen.iloc[n_train + n_calibration:]
    baseline = json.loads(BASELINE_PATH.read_text())
    baseline["official_model"]["split"] = {
        "n_train": n_train,
        "n_calibration": n_calibration,
        "n_test": n_test,
        "train_dates": [str(train["date"].min()), str(train["date"].max())],
        "calibration_dates": [str(calibration["date"].min()), str(calibration["date"].max())],
        "test_dates": [str(test["date"].min()), str(test["date"].max())],
    }
    path.write_text(json.dumps(baseline))


def test_official_training_split_does_not_move_when_processed_data_grows(tmp_path):
    training_csv = tmp_path / "training.csv"
    write_training_csv(training_csv, n=70)
    frame = pd.read_csv(training_csv).sort_values(["date", "fight_id"]).reset_index(drop=True)
    baseline_path = tmp_path / "baseline.json"
    _write_test_baseline(baseline_path, frame)

    trained = train_official_model(
        str(training_csv), baseline_path=str(baseline_path), max_iter=800,
    )
    metadata = trained["training_metadata"]
    assert metadata["source_rows"] == 70
    assert metadata["benchmark_rows"] == 60
    assert metadata["excluded_post_benchmark_rows"] == 10
    assert metadata["train_rows"] == 42
    assert metadata["calibration_rows"] == 9
    assert metadata["split_source"] == "locked_official_baseline"
    assert trained["calibration_version"] == "platt_official_calib9"


def test_official_training_split_rejects_changed_frozen_window(tmp_path):
    training_csv = tmp_path / "training.csv"
    write_training_csv(training_csv, n=70)
    frame = pd.read_csv(training_csv).sort_values(["date", "fight_id"]).reset_index(drop=True)
    baseline_path = tmp_path / "baseline.json"
    _write_test_baseline(baseline_path, frame)
    changed = frame.drop(index=10).reset_index(drop=True)
    changed.to_csv(training_csv, index=False)

    with pytest.raises(ValueError, match="no longer matches its locked split"):
        train_official_model(str(training_csv), baseline_path=str(baseline_path), max_iter=800)


# ---------------------------------------------------------------------------
# Live model report schema + small-sample warnings (fabricated resolved ledger)
# ---------------------------------------------------------------------------

def _resolved_ledger(n, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p = float(np.clip(rng.random(), 0.05, 0.95))
        y = int(rng.random() < p)
        r = ledger_row(f"2026-0{1 + i % 6}-15", f"A{i}", f"B{i}", status="resolved", p=p)
        r["target_a_win"] = y
        r["winner"] = f"A{i}" if y else f"B{i}"
        r["log_loss"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        r["brier"] = (p - y) ** 2
        r["correct_prediction"] = int((p >= 0.5) == bool(y))
        rows.append(r)
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def test_live_model_report_small_sample_warning(tmp_path):
    ledger = tmp_path / "data" / "live" / "ledger.csv"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    _resolved_ledger(10).to_csv(ledger, index=False)
    report = build_live_model_report(ledger_path=str(ledger), output_dir=str(tmp_path / "reports" / "live"),
                                     baseline_path=str(BASELINE_PATH), step6a_report_path=None)
    assert report["sample_size_warning"]["level"] == "too_noisy"
    assert report["resolved_predictions"] == 10


def test_live_model_report_schema_and_usable_sample(tmp_path):
    ledger = tmp_path / "data" / "live" / "ledger.csv"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    _resolved_ledger(120).to_csv(ledger, index=False)
    report = build_live_model_report(ledger_path=str(ledger), output_dir=str(tmp_path / "reports" / "live"),
                                     baseline_path=str(BASELINE_PATH), step6a_report_path=None)
    for key in ("generated_at", "ledger_path", "total_predictions", "pending_predictions", "resolved_predictions",
                "errored_predictions", "n_events", "resolved_events", "overall_metrics", "calibration_buckets",
                "high_confidence_two_sided", "confidence_buckets", "performance_by_event", "rolling_metrics",
                "comparison_reference", "sample_size_warning", "interpretation_warnings"):
        assert key in report, f"{key} missing"
    assert report["sample_size_warning"]["level"] == "usable"
    assert report["overall_metrics"]["log_loss"] is not None
    assert report["comparison_reference"]["official_static_test_metrics"] is not None
    assert (tmp_path / "reports" / "live" / "live_model_report.json").exists()
    assert (tmp_path / "reports" / "live" / "live_model_report.md").exists()


# ---------------------------------------------------------------------------
# End-to-end: predict -> batch report -> resolve, on synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("step6b")
    tcsv = tmp / "train.csv"
    write_training_csv(tcsv)
    card = tmp / "card.csv"
    card_df([("Jon Jones", "Stipe Miocic"), ("Alex Pereira", "Jiri Prochazka"), ("Islam M", "Charles O")]).to_csv(card, index=False)
    ledger = tmp / "data" / "live" / "live_predictions.csv"
    outdir = tmp / "reports" / "live"
    report = run_live_predictions(str(card), ledger_path=str(ledger), output_dir=str(outdir),
                                  training_data=str(tcsv), baseline_path=None,
                                  prediction_batch_id="E2E", max_iter=800)
    return report, tmp, ledger, outdir, tcsv


def test_batch_report_schema_and_ledger_written(e2e):
    report, _tmp, ledger, outdir, _tcsv = e2e
    for key in ("generated_at", "prediction_batch_id", "prediction_mode", "input", "ledger_path", "model_version",
                "calibration_version", "n_fights_submitted", "n_predictions_generated", "n_failed_rows",
                "prediction_table", "append_stats", "duplicate_warnings", "data_quality_warnings",
                "drift_data_quality", "leakage_protocol_checks"):
        assert key in report, f"{key} missing from batch report"
    assert report["prediction_mode"] == "live_forward"
    assert report["n_predictions_generated"] == 3
    led = pd.read_csv(ledger)
    for col in LEDGER_COLUMNS:
        assert col in led.columns
    assert (led["status"] == "pending").sum() == 3
    assert np.allclose(led["predicted_probability_a"] + led["predicted_probability_b"], 1.0)
    assert (outdir / "live_prediction_batch_E2E.json").exists()
    assert (outdir / "live_prediction_batch_E2E.md").exists()


def test_batch_leakage_and_drift_schema(e2e):
    report, *_ = e2e
    for name in ("no_target_or_result_columns_in_input", "no_odds_or_market_columns_in_input",
                 "raw_weight_class_dropped", "probability_orientation_is_p_fighter_a_wins",
                 "official_model_metadata_preserved", "training_uses_only_official_historical_data",
                 "live_rows_not_added_to_training", "data_processed_not_modified"):
        assert report["leakage_protocol_checks"][name]["status"] == "pass"
    drift = report["drift_data_quality"]
    for key in ("n_rows", "n_out_of_range_cells", "n_rows_nan_heavy", "n_low_history_flagged_rows",
                "n_features_elevated_missingness"):
        assert key in drift


def test_e2e_resolution_and_report(e2e):
    _report, tmp, ledger, outdir, _tcsv = e2e
    results = tmp / "results.csv"
    _results([
        ["2026-08-01", "UFC 999: Test", "Jon Jones", "Stipe Miocic", "Jon Jones"],
        ["2026-08-01", "UFC 999: Test", "Alex Pereira", "Jiri Prochazka", "Jiri Prochazka"],
    ]).to_csv(results, index=False)
    res = run_resolution(str(results), ledger_path=str(ledger), output_dir=str(outdir))
    assert res["resolution_stats"]["n_resolved"] == 2
    assert res["n_pending_after"] == 1  # Islam M vs Charles O still pending
    led = pd.read_csv(ledger)
    assert (led["status"] == "resolved").sum() == 2


def test_run_predictions_rejects_result_columns_end_to_end(e2e):
    _report, tmp, _ledger, outdir, tcsv = e2e
    bad_card = tmp / "bad_card.csv"
    df = card_df([("X", "Y")])
    df["winner"] = "X"
    df.to_csv(bad_card, index=False)
    with pytest.raises(ValueError, match="result/outcome"):
        run_live_predictions(str(bad_card), ledger_path=str(tmp / "l2.csv"), output_dir=str(outdir),
                             training_data=str(tcsv), baseline_path=str(BASELINE_PATH), max_iter=800)
