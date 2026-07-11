"""Tests for Step 5D / Run 4 nonlinear-model + blend experiment.

Focus: candidate generation, validation-only selection (no final-test
leakage), calibration-split isolation, early-stopping isolation, blend-weight
validation-only selection, report schema, official-baseline preservation, no
odds/market features, optional-dependency handling, and final-test-for-
finalists-only. Anchor/historical reproduction is checked on the real split
when the processed CSV is present.
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
from ufc_pipeline.modeling import TARGET  # noqa: E402
from ufc_pipeline.step5b_regularization_search import WEIGHT_CLASS_COLUMN  # noqa: E402
from ufc_pipeline.step5d_nonlinear_models import (  # noqa: E402
    ALLOWED_MODEL_FAMILIES,
    ANCHOR_NAME,
    FORBIDDEN_MARKET_TOKENS,
    HISTORICAL_NAME,
    _scan_for_market_tokens,
    build_histgb_grid,
    build_reference_candidates,
    build_step5d_candidates,
    detect_optional_dependencies,
    evaluate_calibrated_window,
    evaluate_finalist_on_test,
    make_histgb_model,
    run_nonlinear_models,
    select_blend_weight,
    select_nonlinear_finalists,
)
from ufc_pipeline.step5b_regularization_search import build_pretest_and_official_split  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "benchmarks" / "official_baseline.json"
REAL_CSV = REPO_ROOT / "data" / "processed" / "ufc_prefight_features_step3c.csv"
# The frozen benchmark was computed on the data window ending 2026-05-16
# (n_train+n_calibration+n_test = 8547 rows). Step 6F promoted newer events
# (through 2026-06-27) into the official Step 3C CSV, so benchmark-reproduction
# tests must slice to this frozen window rather than the whole (grown) file.
BENCHMARK_DATA_MAX_DATE = "2026-05-16"


def synthetic_step5d_df(n: int = 900, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    numeric_all, _categorical = official_step3c_features()
    elo_diff = rng.normal(0, 80, n)
    age_diff = rng.normal(0, 5, n)
    p = 1 / (1 + np.exp(-(0.008 * elo_diff - 0.06 * age_diff)))
    won = (rng.random(n) < p).astype(int)
    dates = pd.date_range("2005-01-01", periods=n, freq="5D").strftime("%Y-%m-%d")
    divisions = rng.choice(
        ["Lightweight", "Welterweight", "Bantamweight", "Featherweight", "Open Weight"],
        n, p=[0.30, 0.28, 0.22, 0.17, 0.03],
    ).astype(object)
    data = {
        "fight_id": np.arange(1, n + 1),
        "date": dates,
        "event": [f"E{i}" for i in range(n)],
        "fighter_a": [f"A{i}" for i in range(n)],
        "fighter_b": [f"B{i}" for i in range(n)],
        "winner": np.where(won == 1, [f"A{i}" for i in range(n)], [f"B{i}" for i in range(n)]),
        TARGET: won,
        WEIGHT_CLASS_COLUMN: divisions,
        "fighter_a_expected_win_prob": p,
        "elo_diff": elo_diff,
        "age_diff": age_diff,
    }
    flag_cols = {"fighter_a_no_prior_stats", "fighter_b_no_prior_stats", "matchup_history_missing"}
    for col in numeric_all:
        if col in data:
            continue
        data[col] = rng.integers(0, 2, n) if col in flag_cols else rng.normal(0, 1.0, n)
    df = pd.DataFrame(data)
    # Inject missingness so HGB native-NaN handling is exercised.
    for col in ("reach_diff", "height_diff"):
        if col in df:
            df.loc[df.sample(frac=0.1, random_state=2).index, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def test_reference_candidates_reproduce_locked_configs():
    refs = build_reference_candidates()
    assert [c["name"] for c in refs] == [ANCHOR_NAME, HISTORICAL_NAME]
    anchor = refs[0]
    assert anchor["model_family"] == "logistic_regression"
    assert anchor["include_raw_weight_class"] is False
    assert anchor["params"] == {"penalty": "elasticnet", "C": 0.003, "solver": "saga",
                                "l1_ratio": 0.1, "class_weight": None}
    assert anchor["promotable"] is False
    historical = refs[1]
    assert historical["include_raw_weight_class"] is True
    assert historical["params"]["penalty"] == "l2" and historical["params"]["C"] == 1.0
    assert historical["promotable"] is False


def test_histgb_grid_is_bounded_and_deterministic():
    grid = build_histgb_grid()
    assert len(grid) == 24
    assert build_histgb_grid() == grid  # deterministic
    assert len(build_histgb_grid(quick=True)) == 2
    for point in grid:
        assert set(point) == {"learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"}


def test_candidate_generation_families_and_no_neural_nets():
    candidates = build_step5d_candidates(quick=False, include_rf_sanity=True)
    # First two are the LR references, in order.
    assert [c["name"] for c in candidates[:2]] == [ANCHOR_NAME, HISTORICAL_NAME]
    families = {c["model_family"] for c in candidates}
    assert families <= ALLOWED_MODEL_FAMILIES
    assert "hist_gradient_boosting" in families
    # No neural-network family anywhere.
    assert not any("mlp" in c["model_family"] or "neural" in c["model_family"] for c in candidates)
    # Main HGB candidates drop weight_class; exactly one weight_class sensitivity point exists.
    main_hgb = [c for c in candidates if c["model_family"] == "hist_gradient_boosting" and c["role"] == "nonlinear"]
    sens = [c for c in candidates if c["role"] == "nonlinear_sensitivity"]
    assert all(c["include_raw_weight_class"] is False for c in main_hgb)
    assert len(sens) == 1 and sens[0]["include_raw_weight_class"] is True
    assert sens[0]["promotable"] is False  # sensitivity is not a promotion candidate
    # Every HGB candidate carries early-stopping params.
    for c in main_hgb + sens:
        assert c["params"]["early_stopping"] is True
        assert 0.0 < c["params"]["validation_fraction"] < 1.0


def test_max_candidates_always_keeps_both_references():
    candidates = build_step5d_candidates(max_candidates=3)
    assert len(candidates) == 3
    assert candidates[0]["name"] == ANCHOR_NAME
    assert candidates[1]["name"] == HISTORICAL_NAME


# ---------------------------------------------------------------------------
# No final-test leakage in selection
# ---------------------------------------------------------------------------

def test_finalist_and_blend_selection_have_no_test_parameters():
    for fn in (select_nonlinear_finalists, select_blend_weight):
        sig = inspect.signature(fn)
        assert all("test" not in name for name in sig.parameters), fn.__name__
    assert list(inspect.signature(select_nonlinear_finalists).parameters) == ["validation_results"]


def test_blend_weight_selection_is_validation_only_and_picks_min():
    rng = np.random.default_rng(0)
    anchor, nonlinear = [], []
    for _ in range(5):
        y = rng.integers(0, 2, 60)
        anchor.append({"y": y, "p_platt": np.clip(rng.random(60), 0.02, 0.98)})
        nonlinear.append({"y": y, "p_platt": np.clip(rng.random(60), 0.02, 0.98)})
    out = select_blend_weight(anchor, nonlinear, weights=(0.0, 0.25, 0.5, 0.75, 1.0))
    assert out["weights_tested"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    best = out["best"]
    assert best["mean_val_log_loss"] == min(r["mean_val_log_loss"] for r in out["results"])
    assert 0.0 <= best["weight"] <= 1.0


# ---------------------------------------------------------------------------
# Calibration-split isolation
# ---------------------------------------------------------------------------

def test_calibration_uses_calibration_window_only_not_eval_labels():
    # A fixed predictor: calibrated eval probs must depend on the calibration
    # window and eval RAW probs only, never on eval labels.
    rng = np.random.default_rng(4)
    calib = pd.DataFrame({TARGET: rng.integers(0, 2, 200), "raw": np.clip(rng.random(200), 0.02, 0.98)})
    eval_a = pd.DataFrame({TARGET: rng.integers(0, 2, 120), "raw": np.clip(rng.random(120), 0.02, 0.98)})
    eval_b = eval_a.copy()
    eval_b[TARGET] = 1 - eval_b[TARGET]  # flip eval labels only

    def predict_fn(frame):
        return frame["raw"].to_numpy()

    out_a = evaluate_calibrated_window(predict_fn, calib, eval_a)
    out_b = evaluate_calibrated_window(predict_fn, calib, eval_b)
    np.testing.assert_allclose(out_a["eval_probs_platt"], out_b["eval_probs_platt"])
    np.testing.assert_allclose(out_a["eval_probs_isotonic"], out_b["eval_probs_isotonic"])


def test_early_stopping_config_uses_internal_validation_fraction():
    model = make_histgb_model(
        {"learning_rate": 0.1, "max_leaf_nodes": 15, "min_samples_leaf": 50,
         "l2_regularization": 1.0, "max_iter": 100, "early_stopping": True,
         "validation_fraction": 0.1, "n_iter_no_change": 20},
        categorical_mask=[False, False],
        random_state=42,
    )
    assert model.early_stopping is True
    assert model.validation_fraction == 0.1  # carved from fit data, not calib/test


# ---------------------------------------------------------------------------
# No odds / market features
# ---------------------------------------------------------------------------

def test_market_token_scanner_flags_odds_but_not_official_features():
    numeric, categorical = official_step3c_features()
    assert _scan_for_market_tokens(numeric + categorical) == []
    assert _scan_for_market_tokens(["elo_diff", "closing_odds", "age_diff"]) == ["closing_odds"]
    assert "odds" in FORBIDDEN_MARKET_TOKENS and "kalshi" in FORBIDDEN_MARKET_TOKENS


# ---------------------------------------------------------------------------
# Optional dependency handling
# ---------------------------------------------------------------------------

def test_optional_dependency_detection_and_gating():
    detected = detect_optional_dependencies()
    assert set(detected) == {"xgboost", "lightgbm"}
    assert all(isinstance(v, bool) for v in detected.values())
    # When forced-skip / unavailable, no optional families are generated.
    candidates = build_step5d_candidates(optional_available={"xgboost": False, "lightgbm": False})
    families = {c["model_family"] for c in candidates}
    assert "xgboost" not in families and "lightgbm" not in families
    # If an optional lib were available, its candidate is generated (config-only).
    with_xgb = build_step5d_candidates(optional_available={"xgboost": True, "lightgbm": False})
    assert any(c["model_family"] == "xgboost" for c in with_xgb)


# ---------------------------------------------------------------------------
# Full run: schema, baseline preservation, final-test-for-finalists-only
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_report(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("step5d")
    csv_path = tmp / "features.csv"
    synthetic_step5d_df().to_csv(csv_path, index=False)
    report = run_nonlinear_models(
        input_csv=str(csv_path),
        output_dir=str(tmp / "reports"),
        baseline_path=str(BASELINE_PATH),
        run1_report_path=None, run2_report_path=None,
        run21_report_path=None, run3_report_path=None,
        quick=True, skip_slow=False, skip_blend=False,
        include_rf_sanity=True, max_iter=1000,
    )
    return report, tmp


def test_report_schema_and_baseline_preserved(synthetic_report):
    report, tmp = synthetic_report
    for key in (
        "generated_at", "official_baseline", "run1_summary", "run2_summary",
        "run21_summary", "run3_summary", "validation_protocol", "candidate_families",
        "search_space", "candidate_configurations", "validation_results",
        "calibration_comparison", "fold_by_fold_results", "finalists", "finalist_names",
        "weight_class_sensitivity", "final_test_results", "anchor_reproduction",
        "historical_reproduction", "blend_results", "blend_final_test",
        "feature_importance", "leakage_protocol_checks", "verdict",
    ):
        assert key in report, f"{key} missing from Step 5D report"

    assert report["changes_official_model"] is False
    assert report["changes_official_baseline_file"] is False
    assert report["validation_protocol"]["selection_rule"]["final_test_used_for_selection"] is False
    assert report["verdict"]["recommend_replace_official_model_now"] is False

    with open(BASELINE_PATH) as fh:
        assert report["official_baseline"] == json.load(fh)

    out_dir = tmp / "reports"
    assert (out_dir / "step5d_nonlinear_models.json").exists()
    assert (out_dir / "step5d_nonlinear_models.md").exists()
    assert (out_dir / "step5d_candidate_results.csv").exists()
    assert (out_dir / "step5d_blend_results.csv").exists()


def test_leakage_checks_all_pass(synthetic_report):
    report, _ = synthetic_report
    checks = report["leakage_protocol_checks"]
    for name in (
        "no_target_or_result_columns", "no_identifiers_or_date_as_features",
        "no_odds_or_market_features", "split_integrity",
        "calibrators_fit_on_calibration_window_only",
        "early_stopping_validation_internal_to_fit_window",
        "final_test_not_used_for_selection", "blend_weight_selected_on_validation_only",
        "no_neural_networks_or_disallowed_families",
        "optional_dependencies_not_installed_are_not_required",
        "official_baseline_file_not_modified",
    ):
        assert name in checks, f"missing leakage check {name}"
        assert checks[name]["status"] == "pass", f"{name} did not pass: {checks[name]}"


def test_final_test_scored_only_for_finalists(synthetic_report):
    report, _ = synthetic_report
    finalist_names = set(report["finalist_names"])
    final_keys = set(report["final_test_results"])
    assert final_keys == finalist_names
    # References always present; both are non-promotable baselines.
    assert {ANCHOR_NAME, HISTORICAL_NAME} <= finalist_names
    # Strictly fewer finalists than evaluated candidates: most HGB configs are
    # validated but never touch the final test.
    all_candidates = {c["name"] for c in report["candidate_configurations"]}
    assert final_keys < all_candidates
    non_finalist_hgb = [
        c["name"] for c in report["candidate_configurations"]
        if c["model_family"] == "hist_gradient_boosting" and c["name"] not in finalist_names
    ]
    assert non_finalist_hgb  # at least one HGB config never reaches the test


def test_weight_class_sensitivity_is_reported_but_not_a_finalist(synthetic_report):
    report, _ = synthetic_report
    sens = report["weight_class_sensitivity"]
    assert sens["available"] is True
    assert sens["sensitivity_candidate"] not in report["finalist_names"]
    assert "weight_class_inclusion_improvement" in sens


# ---------------------------------------------------------------------------
# Real-data reproduction of the anchor and historical baselines
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_CSV.exists(), reason="processed Step 3C CSV not present")
def test_anchor_and_historical_reproduce_benchmark_on_real_split():
    from ufc_pipeline.modeling import coerce_numeric_features

    baseline = json.loads(BASELINE_PATH.read_text())
    official_ll = baseline["official_model"]["test_metrics"]["log_loss"]
    historical_ll = baseline["historical_baselines"]["old_official_step3c_lr_platt"]["test_metrics"]["log_loss"]

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

    refs = {c["name"]: c for c in build_reference_candidates()}
    anchor_res = evaluate_finalist_on_test(refs[ANCHOR_NAME], base_numeric, train, calib, test, official_ll)
    hist_res = evaluate_finalist_on_test(refs[HISTORICAL_NAME], base_numeric, train, calib, test, historical_ll)

    assert abs(anchor_res["platt_test_log_loss"] - official_ll) < 1e-4
    assert abs(hist_res["platt_test_log_loss"] - historical_ll) < 1e-4
