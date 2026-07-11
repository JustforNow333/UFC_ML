"""Tests for Step 5D.1 / Run 4.1 nonlinear sanity audit.

Focus: the orientation check catches inverted probabilities, the parity check
catches row/target mismatches, HGB candidate generation, validation-only
selection (no final-test leakage in HGB search or blend weighting), calibration
audit schema, report schema, and official-baseline preservation. Real-data
anchor + Run 4 HGB reproduction runs when the processed CSV is present.
"""

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufc_pipeline.feature_diagnostics import official_step3c_features  # noqa: E402
from ufc_pipeline.modeling import TARGET, coerce_numeric_features  # noqa: E402
from ufc_pipeline.step5b_regularization_search import (  # noqa: E402
    WEIGHT_CLASS_COLUMN,
    build_pretest_and_official_split,
)
from ufc_pipeline.step5d_nonlinear_models import ANCHOR_NAME, build_reference_candidates, evaluate_finalist_on_test  # noqa: E402
from ufc_pipeline.step5d1_nonlinear_sanity_audit import (  # noqa: E402
    FINE_LEARNING_RATES,
    FINE_MAX_LEAF_NODES,
    RUN4_BEST_HGB_PARAMS,
    _neighbors,
    _orientation_metrics,
    alignment_report,
    audit_blend,
    build_coarse_grid,
    build_refinement_grid,
    make_hgb_params,
    reconstruct_run4_best_hgb,
    run_hgb_sensitivity_search,
    run_sanity_audit,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "benchmarks" / "official_baseline.json"
REAL_CSV = REPO_ROOT / "data" / "processed" / "ufc_prefight_features_step3c.csv"
# The frozen benchmark was computed on the data window ending 2026-05-16
# (n_train+n_calibration+n_test = 8547 rows). Step 6F promoted newer events
# (through 2026-06-27) into the official Step 3C CSV, so benchmark-reproduction
# tests must slice to this frozen window rather than the whole (grown) file.
BENCHMARK_DATA_MAX_DATE = "2026-05-16"


def synthetic_df(n: int = 900, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    numeric_all, _ = official_step3c_features()
    elo = rng.normal(0, 80, n)
    age = rng.normal(0, 5, n)
    p = 1 / (1 + np.exp(-(0.008 * elo - 0.06 * age)))
    won = (rng.random(n) < p).astype(int)
    dates = pd.date_range("2005-01-01", periods=n, freq="5D").strftime("%Y-%m-%d")
    divisions = rng.choice(["Lightweight", "Welterweight", "Bantamweight", "Featherweight", "Open Weight"],
                           n, p=[0.30, 0.28, 0.22, 0.17, 0.03]).astype(object)
    data = {
        "fight_id": np.arange(1, n + 1), "date": dates, "event": [f"E{i}" for i in range(n)],
        "fighter_a": [f"A{i}" for i in range(n)], "fighter_b": [f"B{i}" for i in range(n)],
        "winner": np.where(won == 1, [f"A{i}" for i in range(n)], [f"B{i}" for i in range(n)]),
        TARGET: won, WEIGHT_CLASS_COLUMN: divisions, "fighter_a_expected_win_prob": p,
        "elo_diff": elo, "age_diff": age,
    }
    flags = {"fighter_a_no_prior_stats", "fighter_b_no_prior_stats", "matchup_history_missing"}
    for col in numeric_all:
        if col in data:
            continue
        data[col] = rng.integers(0, 2, n) if col in flags else rng.normal(0, 1.0, n)
    df = pd.DataFrame(data)
    for col in ("reach_diff", "height_diff"):
        if col in df:
            df.loc[df.sample(frac=0.1, random_state=3).index, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

def test_orientation_metrics_flags_inverted_probabilities():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1500)
    y = (x > 0).astype(int)
    p_good = 1 / (1 + np.exp(-2.0 * x))  # well-oriented: high p ⇒ y=1
    good = _orientation_metrics(y, p_good)
    assert good["inversion_improves_log_loss"] is False
    assert good["log_loss"] < good["log_loss_inverted"]
    assert good["auc"] > 0.5 > good["auc_inverted"]

    inverted = _orientation_metrics(y, 1 - p_good)  # a real orientation bug
    assert inverted["inversion_improves_log_loss"] is True
    assert inverted["log_loss"] > inverted["log_loss_inverted"]


# ---------------------------------------------------------------------------
# Row / feature parity
# ---------------------------------------------------------------------------

def test_alignment_report_catches_row_and_target_mismatches():
    ids = [1, 2, 3, 4]
    y = [0, 1, 1, 0]
    ok = alignment_report("s", ids, y, ids, y)
    assert ok["aligned"] is True and ok["rows_match"] and ok["targets_match"]

    row_mismatch = alignment_report("s", ids, y, [1, 2, 3, 5], y)
    assert row_mismatch["rows_match"] is False
    assert row_mismatch["aligned"] is False
    assert row_mismatch["n_only_lr"] == 1 and row_mismatch["n_only_hgb"] == 1

    target_mismatch = alignment_report("s", ids, y, ids, [1, 0, 0, 1])
    assert target_mismatch["rows_match"] is True
    assert target_mismatch["targets_match"] is False
    assert target_mismatch["aligned"] is False


# ---------------------------------------------------------------------------
# HGB candidate generation + neighbourhood helper
# ---------------------------------------------------------------------------

def test_hgb_candidate_generation_and_grids():
    coarse = build_coarse_grid()
    assert len(coarse) == 32
    assert all(c["model_family"] == "hist_gradient_boosting" for c in coarse)
    assert all(c["include_raw_weight_class"] is False for c in coarse)
    assert all(c["params"]["early_stopping"] is True for c in coarse)

    best_params = make_hgb_params(0.03, 15, 100, 0.0, early_stopping=True)
    refine = build_refinement_grid(best_params, exclude={c["name"] for c in coarse})
    assert refine  # produces neighbourhood configs
    assert all(c["name"] not in {c2["name"] for c2 in coarse} for c in refine)

    # early_stopping OFF path carries an explicit max_iter.
    es_off = make_hgb_params(0.05, 7, 50, 1.0, early_stopping=False, max_iter=200)
    assert es_off["early_stopping"] is False and es_off["max_iter"] == 200

    run4 = reconstruct_run4_best_hgb(None)
    assert run4["params"] == RUN4_BEST_HGB_PARAMS
    assert run4["include_raw_weight_class"] is False


def test_neighbors_helper():
    assert _neighbors(15, FINE_MAX_LEAF_NODES) == [7, 15, 31]
    assert _neighbors(0.03, FINE_LEARNING_RATES) == [0.01, 0.03, 0.05]
    assert _neighbors(0.01, FINE_LEARNING_RATES) == [0.01, 0.03]  # low edge
    assert _neighbors(31, FINE_MAX_LEAF_NODES) == [15, 31]  # high edge


# ---------------------------------------------------------------------------
# No final-test leakage in HGB selection or blend weighting
# ---------------------------------------------------------------------------

def test_hgb_search_and_blend_have_no_test_parameters():
    for fn in (run_hgb_sensitivity_search, audit_blend):
        params = inspect.signature(fn).parameters
        assert all("test" not in name for name in params), fn.__name__


def test_blend_weight_selection_is_validation_only_and_min():
    rng = np.random.default_rng(1)
    anchor, hgb = [], []
    for _ in range(5):
        y = rng.integers(0, 2, 80)
        anchor.append({"y": y, "p_platt": np.clip(rng.random(80), 0.02, 0.98)})
        hgb.append({"y": y, "p_platt": np.clip(rng.random(80), 0.02, 0.98)})
    out = audit_blend(anchor, hgb, weights=(0.0, 0.5, 1.0))
    assert 0.0 <= out["global_best_weight_on_lr"] <= 1.0
    assert out["blend_weight_selected_on_validation_only"] is True
    assert len(out["per_fold_best_weight_on_lr"]) == 5


# ---------------------------------------------------------------------------
# Full run: schema, baseline preservation, calibration audit schema
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def audit_report(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("step5d1")
    csv_path = tmp / "features.csv"
    synthetic_df().to_csv(csv_path, index=False)
    report = run_sanity_audit(
        input_csv=str(csv_path),
        output_dir=str(tmp / "reports"),
        baseline_path=str(BASELINE_PATH),
        run4_report_path=None,
        quick=True, skip_slow=False, max_iter=800,
    )
    return report, tmp


def test_report_schema_and_baseline_preserved(audit_report):
    report, tmp = audit_report
    for key in (
        "generated_at", "official_baseline", "config", "anchor_reproduction",
        "probability_orientation_audit", "row_feature_parity_audit", "calibration_audit",
        "missing_value_audit", "hgb_sensitivity_search", "early_stopping_audit",
        "blend_audit", "feature_importance_and_distribution", "final_test_finalists",
        "leakage_protocol_checks", "verdict",
    ):
        assert key in report, f"{key} missing from Run 4.1 report"

    assert report["changes_official_model"] is False
    assert report["changes_official_baseline_file"] is False
    assert report["verdict"]["recommend_replace_official_model_now"] is False
    with open(BASELINE_PATH) as fh:
        assert report["official_baseline"] == json.load(fh)

    out_dir = tmp / "reports"
    for fname in ("step5d1_nonlinear_sanity_audit.json", "step5d1_nonlinear_sanity_audit.md",
                  "step5d1_hgb_sensitivity_results.csv", "step5d1_calibration_audit.csv"):
        assert (out_dir / fname).exists()


def test_calibration_audit_schema(audit_report):
    report, _ = audit_report
    ca = report["calibration_audit"]
    for name in ("current_official_lr", "run4_best_hgb", "best_4_1_hgb"):
        assert name in ca
        val = ca[name]["validation"]
        for method in ("uncalibrated", "platt", "isotonic"):
            assert set(val[method]) == {"log_loss", "brier", "auc"}
        assert "auc_stable_under_platt" in ca[name]
        assert ca[name]["calibrators_fit_on_calibration_window_only"] is True
        assert ca[name]["final_test_used_for_calibration"] is False


def test_hgb_search_best_is_validation_argmin(audit_report):
    report, _ = audit_report
    search = report["hgb_sensitivity_search"]
    best_ll = search["best"]["mean_val_platt_log_loss"]
    assert best_ll == min(r["mean_val_platt_log_loss"] for r in search["all_results"])


def test_final_test_finalists_are_small_set(audit_report):
    report, _ = audit_report
    ft = report["final_test_finalists"]
    assert set(ft) == {"current_official_lr", "run4_best_hgb", "best_4_1_hgb",
                       "run4_blend_reproduction", "best_4_1_blend"}


def test_leakage_checks_all_pass(audit_report):
    report, _ = audit_report
    for name, entry in report["leakage_protocol_checks"].items():
        assert entry["status"] == "pass", f"{name} did not pass: {entry}"


# ---------------------------------------------------------------------------
# Real-data reproduction of the anchors and Run 4 best HGB
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_CSV.exists(), reason="processed Step 3C CSV not present")
def test_anchor_and_run4_hgb_reproduce_on_real_split():
    baseline = json.loads(BASELINE_PATH.read_text())
    official_ll = baseline["official_model"]["test_metrics"]["log_loss"]

    base_numeric, _ = official_step3c_features()
    df = pd.read_csv(REAL_CSV)
    df = df[df[TARGET].notna() & df["fighter_a_expected_win_prob"].notna()].copy()
    df = coerce_numeric_features(df, base_numeric, context="test")
    # Reproduce on the benchmark's frozen data window, not the post-6F-promotion
    # whole file; assert the slice still equals the benchmark's recorded size.
    split = baseline["official_model"]["split"]
    n_benchmark = split["n_train"] + split["n_calibration"] + split["n_test"]
    df = df[df["date"] <= BENCHMARK_DATA_MAX_DATE].copy()
    assert len(df) == n_benchmark, f"frozen benchmark window changed: {len(df)} != {n_benchmark}"
    _pre, train, calib, test = build_pretest_and_official_split(df, train_frac=0.70, calibration_frac=0.15)

    anchor = {c["name"]: c for c in build_reference_candidates()}[ANCHOR_NAME]
    anchor_res = evaluate_finalist_on_test(anchor, base_numeric, train, calib, test, official_ll)
    assert abs(anchor_res["platt_test_log_loss"] - official_ll) < 1e-4

    run4_hgb = reconstruct_run4_best_hgb(None)
    hgb_res = evaluate_finalist_on_test(run4_hgb, base_numeric, train, calib, test, official_ll)
    assert abs(hgb_res["platt_test_log_loss"] - 0.6653879477171739) < 1e-4
