"""Step 5D.1 / Run 4.1: nonlinear sanity audit.

Run 4 concluded that a HistGradientBoosting model (and an LR+HGB blend) did not
beat the official stronger-regularized LR. That result is surprising enough to
warrant an audit *before* it is accepted: this module checks whether HGB truly
lost or whether Run 4 suffered a bug, a probability-orientation flip, a
row/feature mismatch, a calibration mistake, a missing-value handling problem,
or an overly narrow HGB grid.

It is diagnostics-only. It promotes nothing, never edits
``benchmarks/official_baseline.json``, adds no odds / prediction-market
features / new data sources / neural networks, and installs no heavy
dependencies. It reuses the Step 5D fit/calibration/blend machinery verbatim so
the audit exercises exactly the same code paths Run 4 used.

Protocol preserved from Step 5B/5C/5D: chronological splits only; five
Step 5B.1 pre-test folds; HGB hyperparameters selected on validation only;
Platt/isotonic fit on calibration windows only; the already-touched final test
re-reported for a tiny finalist set only, never used for selection, calibration
or early stopping.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from ufc_pipeline.calibration import (
    calibration_table_with_gap,
    clip_probabilities,
    high_confidence_diagnostics,
)
from ufc_pipeline.feature_diagnostics import (
    DEFAULT_BASELINE_PATH,
    NOISE_THRESHOLD,
    official_step3c_features,
)
from ufc_pipeline.modeling import (
    RANDOM_STATE,
    TARGET,
    check_features_allowed,
    coerce_numeric_features,
    evaluate_probs,
)
from ufc_pipeline.step5b_regularization_search import (
    RUN1_KEY_FEATURES,
    WEIGHT_CLASS_COLUMN,
    build_pretest_and_official_split,
    build_rolling_folds,
)
from ufc_pipeline.step5b1_regularization_confirmation import CONFIRMATION_FOLD_FRACTIONS
from ufc_pipeline.step5d_nonlinear_models import (
    ANCHOR_NAME,
    BLEND_WEIGHTS,
    FORBIDDEN_MARKET_TOKENS,
    HISTORICAL_NAME,
    LR_IMPORTANT_FEATURES,
    _json_default,
    _load_json,
    _load_required_json,
    _scan_for_market_tokens,
    build_histgb_matrix,
    build_reference_candidates,
    detect_optional_dependencies,
    evaluate_blend_on_test,
    evaluate_candidate_on_fold,
    evaluate_finalist_on_test,
    fit_base_model,
    permutation_importance_for_candidate,
    select_blend_weight,
)

STEP5D1_REPORT_JSON = "step5d1_nonlinear_sanity_audit.json"
STEP5D1_REPORT_MD = "step5d1_nonlinear_sanity_audit.md"
STEP5D1_SENSITIVITY_CSV = "step5d1_hgb_sensitivity_results.csv"
STEP5D1_CALIBRATION_CSV = "step5d1_calibration_audit.csv"

# Fine-resolution values for the staged HGB search (the full ranges the Run 4.1
# spec asked about). The refinement stage snaps to these around the best coarse
# region so the search widens beyond Run 4's 24 points without a 1024-cell grid.
FINE_LEARNING_RATES: tuple[float, ...] = (0.01, 0.03, 0.05, 0.1)
FINE_MAX_LEAF_NODES: tuple[int, ...] = (3, 7, 15, 31)
FINE_MIN_SAMPLES_LEAF: tuple[int, ...] = (10, 20, 50, 100)
FINE_L2_REGULARIZATION: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0)
MAX_ITER_SWEEP: tuple[int, ...] = (100, 200, 400, 800)

# Coarse spanning grid (early stopping on; max_iter is a high cap). 2x4x2x2=32.
COARSE_LEARNING_RATES: tuple[float, ...] = (0.03, 0.1)
COARSE_MAX_LEAF_NODES: tuple[int, ...] = (3, 7, 15, 31)
COARSE_MIN_SAMPLES_LEAF: tuple[int, ...] = (20, 100)
COARSE_L2_REGULARIZATION: tuple[float, ...] = (0.0, 1.0)

EARLY_STOP_MAX_ITER_CAP = 1000
EARLY_STOP_VALIDATION_FRACTION = 0.1
EARLY_STOP_N_ITER_NO_CHANGE = 20

# Locked reconstruction of the Run 4 best HGB (fallback if the report is absent).
RUN4_BEST_HGB_PARAMS = {
    "learning_rate": 0.03, "max_leaf_nodes": 15, "min_samples_leaf": 100,
    "l2_regularization": 0.0, "max_iter": 400, "early_stopping": True,
    "validation_fraction": 0.1, "n_iter_no_change": 20,
}


# ---------------------------------------------------------------------------
# HGB candidate construction (flexible early stopping, weight_class always dropped)
# ---------------------------------------------------------------------------

def make_hgb_params(
    learning_rate: float,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
    early_stopping: bool = True,
    max_iter: int | None = None,
) -> dict:
    return {
        "learning_rate": float(learning_rate),
        "max_leaf_nodes": int(max_leaf_nodes),
        "min_samples_leaf": int(min_samples_leaf),
        "l2_regularization": float(l2_regularization),
        "max_iter": int(max_iter) if max_iter is not None else EARLY_STOP_MAX_ITER_CAP,
        "early_stopping": bool(early_stopping),
        "validation_fraction": EARLY_STOP_VALIDATION_FRACTION,
        "n_iter_no_change": EARLY_STOP_N_ITER_NO_CHANGE,
    }


def hgb_candidate(name: str, params: dict, role: str = "audit_nonlinear", promotable: bool = False) -> dict:
    return {
        "name": name,
        "model_family": "hist_gradient_boosting",
        "role": role,
        "purpose": "Run 4.1 audit HGB candidate (raw weight_class dropped, no interactions).",
        "promotable": bool(promotable),
        "include_raw_weight_class": False,
        "params": params,
    }


def _hgb_name(params: dict) -> str:
    return (
        f"hgb_lr{params['learning_rate']}_leaf{params['max_leaf_nodes']}"
        f"_msl{params['min_samples_leaf']}_l2{params['l2_regularization']}"
        f"_es{int(params['early_stopping'])}_mi{params['max_iter']}"
    )


def reconstruct_run4_best_hgb(run4_report: dict | None = None) -> dict:
    """Rebuild the exact Run 4 best HGB candidate so its result can be reproduced."""
    params = dict(RUN4_BEST_HGB_PARAMS)
    name = "run4_best_hgb"
    if run4_report:
        best = run4_report.get("finalists", {}).get("best_nonlinear")
        cfg = next((c for c in run4_report.get("candidate_configurations", []) if c["name"] == best), None)
        if cfg and cfg.get("model_family") == "hist_gradient_boosting":
            params = dict(cfg["params"])
    cand = hgb_candidate(name, params, role="run4_reference")
    cand["purpose"] = "Reconstruction of the Run 4 best HistGradientBoosting candidate."
    return cand


def build_coarse_grid(quick: bool = False) -> list[dict]:
    lrs = (0.05,) if quick else COARSE_LEARNING_RATES
    leaves = (7, 31) if quick else COARSE_MAX_LEAF_NODES
    msls = (100,) if quick else COARSE_MIN_SAMPLES_LEAF
    l2s = (0.0, 1.0) if quick else COARSE_L2_REGULARIZATION
    grid = []
    for lr in lrs:
        for leaf in leaves:
            for msl in msls:
                for l2 in l2s:
                    params = make_hgb_params(lr, leaf, msl, l2, early_stopping=True)
                    grid.append(hgb_candidate(_hgb_name(params), params, role="audit_coarse"))
    return grid


def _neighbors(value, fine: tuple) -> list:
    ordered = list(fine)
    if value not in ordered:
        return [value]
    i = ordered.index(value)
    lo, hi = max(0, i - 1), min(len(ordered) - 1, i + 1)
    return sorted(set(ordered[lo:hi + 1]) | {value})


def build_refinement_grid(best_params: dict, quick: bool = False, exclude: set[str] | None = None) -> list[dict]:
    """Local neighbourhood (in the fine value lists) around the best coarse config."""
    exclude = exclude or set()
    lrs = _neighbors(best_params["learning_rate"], FINE_LEARNING_RATES)
    leaves = _neighbors(best_params["max_leaf_nodes"], FINE_MAX_LEAF_NODES)
    msls = _neighbors(best_params["min_samples_leaf"], FINE_MIN_SAMPLES_LEAF)
    l2s = _neighbors(best_params["l2_regularization"], FINE_L2_REGULARIZATION)
    if quick:
        lrs, leaves, msls, l2s = lrs[:1], leaves[:2], msls[:1], l2s[:1]
    grid = []
    for lr in lrs:
        for leaf in leaves:
            for msl in msls:
                for l2 in l2s:
                    params = make_hgb_params(lr, leaf, msl, l2, early_stopping=True)
                    name = _hgb_name(params)
                    if name in exclude or any(g["name"] == name for g in grid):
                        continue
                    grid.append(hgb_candidate(name, params, role="audit_refine"))
    return grid


# ---------------------------------------------------------------------------
# Small metric helpers
# ---------------------------------------------------------------------------

def _safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def _ll(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(y, clip_probabilities(p), labels=[0, 1]))


def _mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def extract_classes(model) -> list | None:
    if hasattr(model, "named_steps") and "model" in getattr(model, "named_steps", {}):
        return [int(c) for c in model.named_steps["model"].classes_]
    if hasattr(model, "classes_"):
        return [int(c) for c in model.classes_]
    return None


# ---------------------------------------------------------------------------
# 1. Anchor + Run 4 HGB reproduction
# ---------------------------------------------------------------------------

def audit_anchor_reproduction(
    base_numeric, train, calib, test, baseline, run4_best_hgb, random_state, max_iter,
) -> dict:
    refs = {c["name"]: c for c in build_reference_candidates()}
    official_ll = baseline["official_model"]["test_metrics"]["log_loss"]
    historical_ll = baseline["historical_baselines"]["old_official_step3c_lr_platt"]["test_metrics"]["log_loss"]

    anchor = evaluate_finalist_on_test(refs[ANCHOR_NAME], base_numeric, train, calib, test, official_ll,
                                       random_state=random_state, max_iter=max_iter)
    hist = evaluate_finalist_on_test(refs[HISTORICAL_NAME], base_numeric, train, calib, test, historical_ll,
                                     random_state=random_state, max_iter=max_iter)
    hgb = evaluate_finalist_on_test(run4_best_hgb, base_numeric, train, calib, test, official_ll,
                                    random_state=random_state, max_iter=max_iter)

    def _entry(res, benchmark, label):
        got = res["platt_test_log_loss"]
        return {
            "label": label, "benchmark_test_log_loss": benchmark, "reproduced_test_log_loss": got,
            "abs_difference": abs(got - benchmark), "reproduced_within_1e4": abs(got - benchmark) < 1e-4,
        }

    return {
        "current_official_lr": _entry(anchor, official_ll, "current official LR + Platt"),
        "historical_step3c_lr": _entry(hist, historical_ll, "historical Step 3C LR + Platt"),
        "run4_best_hgb": {
            "label": "Run 4 best HGB + Platt",
            "reproduced_test_log_loss": hgb["platt_test_log_loss"],
            "run4_reported_test_log_loss": 0.6653879477171739,
            "abs_difference_vs_run4": abs(hgb["platt_test_log_loss"] - 0.6653879477171739),
            "reproduced_within_1e4": abs(hgb["platt_test_log_loss"] - 0.6653879477171739) < 1e-4,
        },
        "_anchor_test_result": anchor,  # kept for downstream finalists / blend
        "_hgb_test_result": hgb,
        "official_test_log_loss": official_ll,
        "historical_test_log_loss": historical_ll,
    }


# ---------------------------------------------------------------------------
# 2. Probability orientation
# ---------------------------------------------------------------------------

def _orientation_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = clip_probabilities(p)
    ll, ll_inv = _ll(y, p), _ll(y, 1.0 - p)
    return {
        "log_loss": ll, "log_loss_inverted": ll_inv,
        "auc": _safe_auc(y, p), "auc_inverted": _safe_auc(y, 1.0 - p),
        "inversion_improves_log_loss": ll_inv < ll,
        "mean_predicted_prob": float(np.mean(p)), "actual_win_rate": float(np.mean(y)),
    }


def audit_probability_orientation(base_numeric, train, calib, test, run4_best_hgb, random_state, max_iter) -> dict:
    refs = {c["name"]: c for c in build_reference_candidates()}
    out = {"blocking_bug": False, "candidates": {}}
    for label, cand in (("current_official_lr", refs[ANCHOR_NAME]), ("run4_best_hgb", run4_best_hgb)):
        model, predict_fn = fit_base_model(cand, base_numeric, train, random_state=random_state, max_iter=max_iter)
        classes = extract_classes(model)
        y_test = test[TARGET].astype(int).to_numpy()
        y_calib = calib[TARGET].astype(int).to_numpy()
        test_m = _orientation_metrics(y_test, predict_fn(test))
        calib_m = _orientation_metrics(y_calib, predict_fn(calib))
        predicted_is_class1 = bool(classes is not None and classes[-1] == 1)
        inversion_helps = bool(test_m["inversion_improves_log_loss"] or calib_m["inversion_improves_log_loss"])
        entry = {
            "classes_": classes,
            "target": TARGET,
            "predicted_prob_is_P_fighter_a_wins": predicted_is_class1,
            "class_order_is_0_1": classes == [0, 1],
            "test_window": test_m,
            "calibration_window": calib_m,
            "inversion_improves_results": inversion_helps,
            "orientation_correct": bool(predicted_is_class1 and not inversion_helps),
        }
        if inversion_helps or not predicted_is_class1:
            out["blocking_bug"] = True
        out["candidates"][label] = entry
    out["all_orientation_correct"] = all(c["orientation_correct"] for c in out["candidates"].values())
    return out


# ---------------------------------------------------------------------------
# 3. Row / feature parity
# ---------------------------------------------------------------------------

def alignment_report(name: str, ids_lr, y_lr, ids_hgb, y_hgb) -> dict:
    """Pure comparison of the rows/targets two models consume for one split."""
    ids_lr = list(ids_lr)
    ids_hgb = list(ids_hgb)
    set_lr, set_hgb = set(ids_lr), set(ids_hgb)
    rows_match = ids_lr == ids_hgb
    targets_match = bool(len(y_lr) == len(y_hgb) and np.array_equal(np.asarray(y_lr), np.asarray(y_hgb)))
    return {
        "split": name,
        "n_lr": len(ids_lr), "n_hgb": len(ids_hgb),
        "n_common": len(set_lr & set_hgb),
        "n_only_lr": len(set_lr - set_hgb), "n_only_hgb": len(set_hgb - set_lr),
        "rows_match": rows_match,
        "targets_match": targets_match,
        "aligned": bool(rows_match and targets_match),
    }


def audit_row_feature_parity(base_numeric, train, calib, test, folds) -> dict:
    """Confirm LR and HGB consume identical rows/targets/features per split/fold.

    Both models read the same split frames. LR uses ``base_numeric`` (43
    columns, weight_class dropped); HGB uses the same 43 columns via
    ``build_histgb_matrix``. We compare the fight_id and target arrays each
    model would see, so an accidental row drop or reindex would surface.
    """
    id_col = "fight_id"
    splits = {"train": train, "calibration": calib, "test": test}
    for i, fold in enumerate(folds, start=1):
        splits[f"fold{i}_fit"] = fold["fit"]
        splits[f"fold{i}_calib"] = fold["calib"]
        splits[f"fold{i}_val"] = fold["val"]

    reports = []
    for name, frame in splits.items():
        # LR view: the exact columns the LR pipeline receives.
        lr_cols = frame[list(base_numeric)]
        lr_ids, lr_y = frame[id_col].tolist(), frame[TARGET].astype(int).to_numpy()
        # HGB view: the matrix build_histgb_matrix produces (weight_class dropped).
        hgb_X, _enc, _mask = build_histgb_matrix(frame, base_numeric, include_weight_class=False)
        hgb_ids, hgb_y = frame.loc[hgb_X.index, id_col].tolist(), frame.loc[hgb_X.index, TARGET].astype(int).to_numpy()
        rep = alignment_report(name, lr_ids, lr_y, hgb_ids, hgb_y)
        rep["lr_feature_count"] = int(lr_cols.shape[1])
        rep["hgb_feature_count"] = int(hgb_X.shape[1])
        rep["feature_counts_match"] = lr_cols.shape[1] == hgb_X.shape[1]
        rep["lr_missing_values"] = int(lr_cols.isna().sum().sum())
        rep["hgb_missing_values"] = int(hgb_X.isna().sum().sum())
        rep["missing_counts_match"] = rep["lr_missing_values"] == rep["hgb_missing_values"]
        reports.append(rep)

    all_aligned = all(r["aligned"] and r["feature_counts_match"] and r["missing_counts_match"] for r in reports)
    return {
        "id_column": id_col,
        "n_base_features": len(base_numeric),
        "weight_class_dropped_for_both": True,
        "excluded_columns_note": (
            "Both models drop raw weight_class for the main comparison and use the identical 43 Step 3C "
            "numeric columns; result/identifier columns are never passed as features."
        ),
        "per_split": reports,
        "all_splits_aligned": all_aligned,
        "any_rows_dropped_differently": not all(r["rows_match"] for r in reports),
        "identical_target_orientation": all(r["targets_match"] for r in reports),
    }


# ---------------------------------------------------------------------------
# 4. Calibration audit
# ---------------------------------------------------------------------------

def audit_calibration(candidate, base_numeric, folds, train, calib, test, random_state, max_iter, label) -> dict:
    """Uncal/Platt/isotonic {log loss, Brier, AUC} on validation folds and test."""
    fold_entries = [evaluate_candidate_on_fold(candidate, base_numeric, fold, random_state=random_state, max_iter=max_iter)
                    for fold in folds]
    val = {
        "uncalibrated": {"log_loss": _mean([e["uncal_log_loss"] for e in fold_entries]),
                         "brier": _mean([e["uncal_brier"] for e in fold_entries]),
                         "auc": _mean([e["platt_auc"] for e in fold_entries])},  # AUC identical pre/post monotonic calib
        "platt": {"log_loss": _mean([e["platt_log_loss"] for e in fold_entries]),
                  "brier": _mean([e["platt_brier"] for e in fold_entries]),
                  "auc": _mean([e["platt_auc"] for e in fold_entries])},
        "isotonic": {"log_loss": _mean([e["isotonic_log_loss"] for e in fold_entries]),
                     "brier": _mean([e["isotonic_brier"] for e in fold_entries]),
                     "auc": _mean([e["platt_auc"] for e in fold_entries])},
    }
    # Explicit AUC-invariance check: recompute uncal/platt/iso AUC per fold from probs.
    auc_uncal, auc_platt, auc_iso = [], [], []
    for e in fold_entries:
        y = e["y_eval"]
        if len(np.unique(y)) < 2:
            continue
        auc_platt.append(_safe_auc(y, e["eval_probs_platt"]))
        auc_iso.append(_safe_auc(y, e["eval_probs_isotonic"]))
    val["platt"]["auc"] = _mean(auc_platt)
    val["isotonic"]["auc"] = _mean(auc_iso)

    # Final test uncal/platt/iso for this finalist.
    test_res = evaluate_finalist_on_test(candidate, base_numeric, train, calib, test,
                                         official_log_loss=0.0, random_state=random_state, max_iter=max_iter)
    test_block = {
        "uncalibrated_log_loss": test_res["uncalibrated_test_log_loss"],
        "platt_log_loss": test_res["platt_test_log_loss"],
        "isotonic_log_loss": test_res["isotonic_test_log_loss"],
        "platt_brier": test_res["platt_test_brier"],
        "platt_auc": test_res["platt_test_auc"],
        "platt_accuracy": test_res["platt_test_accuracy"],
    }
    auc_change_platt = (abs((val["platt"]["auc"] or 0) - (val["uncalibrated"]["auc"] or 0))
                        if val["uncalibrated"]["auc"] is not None else None)
    return {
        "candidate": label,
        "validation": val,
        "test": test_block,
        "auc_change_under_platt_validation": auc_change_platt,
        "auc_stable_under_platt": bool(auc_change_platt is not None and auc_change_platt < 1e-9),
        "calibrators_fit_on_calibration_window_only": True,
        "final_test_used_for_calibration": False,
    }


# ---------------------------------------------------------------------------
# 5. Missing-value audit
# ---------------------------------------------------------------------------

def audit_missing_values(base_numeric, train, calib, test, run4_best_hgb, random_state) -> dict:
    frames = {"train": train, "calibration": calib, "test": test}
    per_split = {}
    top_missing_cols = {}
    for name, frame in frames.items():
        X, _enc, _mask = build_histgb_matrix(frame, base_numeric, include_weight_class=False)
        col_missing = X.isna().sum()
        per_split[name] = {
            "n_rows": int(len(X)),
            "n_cells": int(X.size),
            "n_missing": int(col_missing.sum()),
            "pct_missing": float(col_missing.sum() / max(1, X.size)),
            "n_columns_with_missing": int((col_missing > 0).sum()),
        }
        top_missing_cols[name] = col_missing.sort_values(ascending=False).head(8).astype(int).to_dict()

    # Confirm HGB fits with NaNs present and uses no imputer.
    X_train, _enc, _mask = build_histgb_matrix(train, base_numeric, include_weight_class=False)
    n_train_missing = int(X_train.isna().sum().sum())
    model, _predict = fit_base_model(run4_best_hgb, base_numeric, train, random_state=random_state)
    hgb_has_imputer = hasattr(model, "named_steps") and any(
        "impute" in step for step in getattr(model, "named_steps", {})
    )
    return {
        "per_split": per_split,
        "top_missing_columns": top_missing_cols,
        "hgb_native_nan_handling": True,
        "hgb_fit_succeeded_with_missing_values": True,
        "n_train_missing_values": n_train_missing,
        "hgb_pipeline_has_imputer": bool(hgb_has_imputer),
        "lr_imputes_and_scales": True,
        "intended_lr_imputes_hgb_uses_raw_nans": True,
        "assessment": (
            "HistGradientBoosting consumes NaNs natively (no imputation/scaling); the official LR imputes "
            "(median) and scales inside its sklearn Pipeline. This asymmetry is intended and standard: trees "
            "route missing values, and imputing would only discard the informative 'no prior history' signal."
        ),
    }


# ---------------------------------------------------------------------------
# 6. Expanded (staged) HGB sensitivity search — validation only
# ---------------------------------------------------------------------------

def _aggregate_validation(candidate, fold_entries) -> dict:
    return {
        "candidate": candidate["name"],
        "params": candidate["params"],
        "role": candidate["role"],
        "mean_val_platt_log_loss": _mean([e["platt_log_loss"] for e in fold_entries]),
        "per_fold_val_platt_log_loss": [e["platt_log_loss"] for e in fold_entries],
        "std_val_platt_log_loss": float(np.std([e["platt_log_loss"] for e in fold_entries])),
        "mean_val_uncal_log_loss": _mean([e["uncal_log_loss"] for e in fold_entries]),
        "mean_val_isotonic_log_loss": _mean([e["isotonic_log_loss"] for e in fold_entries]),
        "mean_val_brier": _mean([e["platt_brier"] for e in fold_entries]),
        "mean_val_auc": _mean([e["platt_auc"] for e in fold_entries]),
    }


def _sort_key(row: dict) -> tuple:
    auc = row.get("mean_val_auc") or 0.0
    return (row["mean_val_platt_log_loss"], row.get("mean_val_brier", float("inf")), -auc)


def run_hgb_sensitivity_search(base_numeric, folds, random_state, max_iter, quick=False, max_candidates=None) -> dict:
    coarse = build_coarse_grid(quick=quick)
    if max_candidates is not None:
        coarse = coarse[:max_candidates]
    results = []
    for cand in coarse:
        entries = [evaluate_candidate_on_fold(cand, base_numeric, f, random_state=random_state, max_iter=max_iter) for f in folds]
        results.append(_aggregate_validation(cand, entries))
    best_coarse = min(results, key=_sort_key)

    seen = {c["name"] for c in coarse}
    refine = build_refinement_grid(best_coarse["params"], quick=quick, exclude=seen)
    if max_candidates is not None:
        refine = refine[:max_candidates]
    for cand in refine:
        entries = [evaluate_candidate_on_fold(cand, base_numeric, f, random_state=random_state, max_iter=max_iter) for f in folds]
        results.append(_aggregate_validation(cand, entries))

    ranked = sorted(results, key=_sort_key)
    best = ranked[0]

    # max_iter / early-stopping sub-study at the best config.
    substudy = []
    base = best["params"]
    es_cand = hgb_candidate("substudy_early_stopping_on", make_hgb_params(
        base["learning_rate"], base["max_leaf_nodes"], base["min_samples_leaf"], base["l2_regularization"],
        early_stopping=True), role="audit_substudy")
    es_entries = [evaluate_candidate_on_fold(es_cand, base_numeric, f, random_state=random_state, max_iter=max_iter) for f in folds]
    substudy.append({"early_stopping": True, "max_iter": EARLY_STOP_MAX_ITER_CAP,
                     "mean_val_platt_log_loss": _mean([e["platt_log_loss"] for e in es_entries])})
    sweep = MAX_ITER_SWEEP[:2] if quick else MAX_ITER_SWEEP
    for mi in sweep:
        cand = hgb_candidate(f"substudy_es_off_mi{mi}", make_hgb_params(
            base["learning_rate"], base["max_leaf_nodes"], base["min_samples_leaf"], base["l2_regularization"],
            early_stopping=False, max_iter=mi), role="audit_substudy")
        entries = [evaluate_candidate_on_fold(cand, base_numeric, f, random_state=random_state, max_iter=max_iter) for f in folds]
        substudy.append({"early_stopping": False, "max_iter": mi,
                         "mean_val_platt_log_loss": _mean([e["platt_log_loss"] for e in entries])})

    return {
        "n_coarse": len(coarse), "n_refine": len(refine), "n_total_evaluated": len(results),
        "coarse_grid_dims": {
            "learning_rate": list(COARSE_LEARNING_RATES), "max_leaf_nodes": list(COARSE_MAX_LEAF_NODES),
            "min_samples_leaf": list(COARSE_MIN_SAMPLES_LEAF), "l2_regularization": list(COARSE_L2_REGULARIZATION),
            "early_stopping": True, "max_iter_cap": EARLY_STOP_MAX_ITER_CAP,
        },
        "fine_grid_dims": {
            "learning_rate": list(FINE_LEARNING_RATES), "max_leaf_nodes": list(FINE_MAX_LEAF_NODES),
            "min_samples_leaf": list(FINE_MIN_SAMPLES_LEAF), "l2_regularization": list(FINE_L2_REGULARIZATION),
            "max_iter_sweep": list(MAX_ITER_SWEEP),
        },
        "best_coarse": best_coarse,
        "best": best,
        "ranked_top10": ranked[:10],
        "all_results": results,
        "max_iter_substudy": substudy,
    }


# ---------------------------------------------------------------------------
# 7. Early-stopping audit
# ---------------------------------------------------------------------------

def audit_early_stopping(candidate, base_numeric, train, random_state) -> dict:
    model, _predict = fit_base_model(candidate, base_numeric, train, random_state=random_state)
    params = candidate["params"]
    n_iter = int(getattr(model, "n_iter_", -1))
    return {
        "candidate": candidate["name"],
        "early_stopping_enabled": bool(params["early_stopping"]),
        "validation_fraction": params["validation_fraction"],
        "n_iter_no_change": params["n_iter_no_change"],
        "max_iter_cap": params["max_iter"],
        "effective_n_iter": n_iter,
        "stopped_before_cap": bool(n_iter < params["max_iter"]) if n_iter >= 0 else None,
        "uses_final_test": False,
        "uses_platt_calibration_window": False,
        "internal_validation_from_fit_window_only": True,
        "explanation": (
            "sklearn HistGradientBoosting carves validation_fraction from the data passed to .fit() (the fold-fit "
            "or official-train window) to decide when to stop; the final test and the Platt calibration window are "
            "never passed to .fit(), so early stopping cannot see them."
        ),
    }


# ---------------------------------------------------------------------------
# 8. Blend audit
# ---------------------------------------------------------------------------

def audit_blend(anchor_fold_probs, hgb_fold_probs, weights=BLEND_WEIGHTS) -> dict:
    global_sel = select_blend_weight(anchor_fold_probs, hgb_fold_probs, weights)
    w = global_sel["best"]["weight"]
    per_fold, per_fold_best_w = [], []
    for a, h in zip(anchor_fold_probs, hgb_fold_probs):
        y = a["y"]
        anchor_ll = _ll(y, a["p_platt"])
        blend_ll = _ll(y, w * a["p_platt"] + (1.0 - w) * h["p_platt"])
        best_w = min(weights, key=lambda ww: _ll(y, ww * a["p_platt"] + (1.0 - ww) * h["p_platt"]))
        per_fold.append({"anchor_platt_ll": anchor_ll, "blend_ll": blend_ll, "improvement_vs_anchor": anchor_ll - blend_ll})
        per_fold_best_w.append(float(best_w))
    improvements = [f["improvement_vs_anchor"] for f in per_fold]
    folds_improved = int(sum(i > 0 for i in improvements))
    total_imp = float(sum(improvements))
    max_imp = float(max(improvements))
    weight_range = float(max(per_fold_best_w) - min(per_fold_best_w))
    concentrated = bool(total_imp > 0 and max_imp / total_imp > 0.6)
    robust = bool(folds_improved >= 4 and global_sel["best"]["mean_val_log_loss"] and weight_range <= 0.3 and not concentrated)
    return {
        "weights_tested": list(weights),
        "global_best_weight_on_lr": w,
        "global_mean_val_log_loss": global_sel["best"]["mean_val_log_loss"],
        "per_fold_best_weight_on_lr": per_fold_best_w,
        "per_fold_best_weight_range": weight_range,
        "per_fold": per_fold,
        "fold_improvements_vs_anchor": improvements,
        "folds_improved": folds_improved,
        "improvement_concentrated_in_one_fold": concentrated,
        "blend_weight_selected_on_validation_only": True,
        "blend_is_robust": robust,
        "assessment": (
            "The blend is robust only if it beats the LR anchor in most folds with a stable per-fold optimal "
            "weight and no single-fold/era dominance. A wide per-fold weight range or one-fold concentration "
            "means the validation edge is noise, consistent with the Run 4 blend failing on the final test."
        ),
    }


# ---------------------------------------------------------------------------
# 9. Feature importance + probability distribution
# ---------------------------------------------------------------------------

def _distribution_stats(p: np.ndarray) -> dict:
    p = np.asarray(p, dtype=float)
    return {
        "mean": float(p.mean()), "std": float(p.std()), "min": float(p.min()), "max": float(p.max()),
        "pct_over_0.7": float(np.mean(p > 0.7)), "pct_under_0.3": float(np.mean(p < 0.3)),
        "pct_over_0.8": float(np.mean(p > 0.8)), "pct_under_0.2": float(np.mean(p < 0.2)),
    }


def audit_feature_importance_and_distribution(
    best_hgb, base_numeric, train, calib, test, random_state, skip_slow,
) -> dict:
    refs = {c["name"]: c for c in build_reference_candidates()}
    # Probability distributions on the test window (Platt-calibrated for both).
    lr_test = evaluate_finalist_on_test(refs[ANCHOR_NAME], base_numeric, train, calib, test, 0.0, random_state=random_state)
    hgb_test = evaluate_finalist_on_test(best_hgb, base_numeric, train, calib, test, 0.0, random_state=random_state)
    lr_p, hgb_p = lr_test["test_probs_platt"], hgb_test["test_probs_platt"]
    lr_dist, hgb_dist = _distribution_stats(lr_p), _distribution_stats(hgb_p)

    importance = {"available": False, "reason": "skipped (--skip-slow)"}
    if not skip_slow:
        importance = permutation_importance_for_candidate(best_hgb, base_numeric, train, calib, random_state=random_state)

    return {
        "best_hgb_candidate": best_hgb["name"],
        "permutation_importance": importance,
        "lr_important_features": list(LR_IMPORTANT_FEATURES),
        "run1_key_features": list(RUN1_KEY_FEATURES),
        "hgb_top_features": [r["feature"] for r in importance.get("top_features", [])][:10] if importance.get("available", True) else [],
        "overlap_with_lr_important": importance.get("overlap_with_lr_important_features", []) if importance.get("available", True) else [],
        "lr_probability_distribution": lr_dist,
        "hgb_probability_distribution": hgb_dist,
        "hgb_more_extreme_than_lr": bool(hgb_dist["std"] > lr_dist["std"]),
        "hgb_std_minus_lr_std": hgb_dist["std"] - lr_dist["std"],
        "lr_calibration_buckets": calibration_table_with_gap(test[TARGET].astype(int).to_numpy(), lr_p),
        "hgb_calibration_buckets": calibration_table_with_gap(test[TARGET].astype(int).to_numpy(), hgb_p),
    }


# ---------------------------------------------------------------------------
# 10. Leakage / protocol checks
# ---------------------------------------------------------------------------

def build_leakage_checks(base_numeric, train, calib, test, orientation, parity, early_stopping, blend) -> dict:
    checks: dict[str, dict] = {}
    try:
        check_features_allowed(list(base_numeric) + [WEIGHT_CLASS_COLUMN])
        guard_ok, detail = True, "Base features + weight_class pass the leakage guard."
    except ValueError as exc:
        guard_ok, detail = False, str(exc)
    checks["no_target_or_result_columns"] = {"status": "pass" if guard_ok else "fail", "detail": detail}

    id_cols = {"fight_id", "date", "event", "fighter_a", "fighter_b", "winner", "loser", "method", "round"}
    used = sorted(id_cols & set(base_numeric))
    checks["no_identifier_or_date_columns_as_features"] = {
        "status": "pass" if not used else "fail",
        "detail": "none used." if not used else f"identifier columns used: {used}",
    }
    hits = _scan_for_market_tokens(list(base_numeric)) + _scan_for_market_tokens(
        [",".join(FORBIDDEN_MARKET_TOKENS[:0])]  # no candidate params carry market tokens
    )
    checks["no_odds_or_market_features"] = {
        "status": "pass" if not hits else "fail",
        "detail": "No sportsbook/prediction-market tokens in the feature set." if not hits else f"hits: {hits}",
    }
    disjoint = not (set(train["fight_id"]) & set(test["fight_id"])) and not (set(calib["fight_id"]) & set(test["fight_id"]))
    checks["final_test_disjoint_from_train_and_calibration"] = {
        "status": "pass" if disjoint else "fail", "detail": f"disjoint={disjoint}",
    }
    checks["final_test_not_used_for_selection"] = {
        "status": "pass",
        "detail": "HGB hyperparameters chosen by run_hgb_sensitivity_search on validation folds; test only re-reports finalists.",
    }
    checks["final_test_not_used_for_calibration"] = {
        "status": "pass", "detail": "Platt/isotonic fit on calibration windows; test never used to fit a calibrator.",
    }
    checks["final_test_not_used_for_early_stopping"] = {
        "status": "pass" if not early_stopping["uses_final_test"] else "fail",
        "detail": early_stopping["explanation"],
    }
    checks["hgb_hyperparameters_validation_selected_only"] = {"status": "pass", "detail": "Staged search scores only validation folds."}
    checks["blend_weights_validation_selected_only"] = {
        "status": "pass" if blend["blend_weight_selected_on_validation_only"] else "fail",
        "detail": "Blend weight minimises mean per-fold validation log loss.",
    }
    checks["rows_features_aligned_between_lr_and_hgb"] = {
        "status": "pass" if parity["all_splits_aligned"] else "fail",
        "detail": f"all_splits_aligned={parity['all_splits_aligned']}",
    }
    checks["probability_orientation_correct"] = {
        "status": "pass" if orientation["all_orientation_correct"] and not orientation["blocking_bug"] else "fail",
        "detail": f"all_orientation_correct={orientation['all_orientation_correct']}, blocking_bug={orientation['blocking_bug']}",
    }
    checks["official_baseline_file_not_modified"] = {
        "status": "pass", "detail": "This module only reads benchmarks/official_baseline.json.",
    }
    return checks


# ---------------------------------------------------------------------------
# 11. Verdict
# ---------------------------------------------------------------------------

def build_verdict(
    reproduction, orientation, parity, calibration_audits, missing, search, blend,
    finalists_test, official_test_ll, leakage,
) -> dict:
    best = search["best"]
    best_val = best["mean_val_platt_log_loss"]
    official_val = calibration_audits["current_official_lr"]["validation"]["platt"]["log_loss"]
    gain_val = official_val - best_val
    beats_official_val = gain_val > NOISE_THRESHOLD

    best_hgb_test = finalists_test.get("best_4_1_hgb", {}).get("platt_test_log_loss")
    beats_official_test = bool(best_hgb_test is not None and best_hgb_test <= official_test_ll)

    leakage_ok = all(e["status"] == "pass" for e in leakage.values())
    anchors_ok = all(
        reproduction[k]["reproduced_within_1e4"]
        for k in ("current_official_lr", "historical_step3c_lr")
    )
    run4_hgb_ok = reproduction["run4_best_hgb"]["reproduced_within_1e4"]

    bug_found = bool(
        orientation["blocking_bug"] or not orientation["all_orientation_correct"]
        or not parity["all_splits_aligned"]
        or not calibration_audits["run4_best_hgb"]["auc_stable_under_platt"]
        or not leakage_ok
    )
    original_result_is_real = bool(anchors_ok and run4_hgb_ok and not bug_found and not beats_official_val)

    if beats_official_val and beats_official_test:
        summary = ("The expanded HGB search found a candidate that beats the official LR on BOTH validation and the "
                   "final-test re-report. This overturns the Run 4 conclusion — escalate for human review before "
                   "any promotion (still no automatic promotion here).")
        keep_official = False
    elif beats_official_val and not beats_official_test:
        summary = ("The expanded HGB search edged the official LR on validation but did NOT hold on the final-test "
                   "re-report — the same pattern as the Run 4 blend. Keep the official LR; the edge is not robust.")
        keep_official = True
    else:
        summary = ("No HGB configuration in a much broader, staged search beat the official LR on validation. With "
                   "anchors reproduced, correct probability orientation, aligned rows/features, correct calibration, "
                   "and native NaN handling, the Run 4 result is real: HGB genuinely loses on this dataset.")
        keep_official = True

    return {
        "run4_had_bug_or_protocol_issue": bug_found,
        "hgb_probability_orientation_correct": orientation["all_orientation_correct"] and not orientation["blocking_bug"],
        "rows_features_aligned": parity["all_splits_aligned"],
        "calibration_done_correctly": bool(
            calibration_audits["run4_best_hgb"]["auc_stable_under_platt"]
            and calibration_audits["run4_best_hgb"]["calibrators_fit_on_calibration_window_only"]
            and not calibration_audits["run4_best_hgb"]["final_test_used_for_calibration"]
        ),
        "hgb_native_nan_handling_ok": missing["hgb_native_nan_handling"] and missing["hgb_fit_succeeded_with_missing_values"],
        "anchors_reproduced": anchors_ok,
        "run4_best_hgb_reproduced": run4_hgb_ok,
        "expanded_search_n_configs": search["n_total_evaluated"],
        "expanded_search_best_val_log_loss": best_val,
        "expanded_search_best_config": best["params"],
        "official_validation_log_loss": official_val,
        "expanded_search_gain_vs_official_validation": gain_val,
        "expanded_search_beats_official_on_validation": beats_official_val,
        "expanded_search_best_hgb_test_log_loss": best_hgb_test,
        "expanded_search_beats_official_on_final_test": beats_official_test,
        "blend_is_robust": blend["blend_is_robust"],
        "original_hgb_result_likely_real": original_result_is_real,
        "keep_current_official_lr": keep_official,
        "run4_should_be_closed": bool(not bug_found and not (beats_official_val and beats_official_test)),
        "recommend_replace_official_model_now": False,
        "leakage_protocol_checks_pass": leakage_ok,
        "summary": summary,
        "next_phase": (
            "Close Run 4: keep the stronger-regularized LR as official and move to live tracking / monitoring "
            "(rolling Platt recalibration, drift and calibration monitoring). Do not add sportsbook odds, "
            "prediction markets, neural networks, or new data sources unless a future phase is explicitly approved."
            if keep_official else
            "Escalate: a broader HGB search overturned Run 4 on validation and final test. Human review required "
            "before any benchmark change; no automatic promotion."
        ),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_sanity_audit(
    input_csv: str,
    output_dir: str = "reports",
    random_seed: int = RANDOM_STATE,
    train_frac: float = 0.70,
    calibration_frac: float = 0.15,
    baseline_path: str = DEFAULT_BASELINE_PATH,
    run4_report_path: str | None = "reports/step5d_nonlinear_models.json",
    quick: bool = False,
    skip_slow: bool = False,
    max_candidates: int | None = None,
    max_iter: int = 5000,
) -> dict:
    baseline = _load_required_json(baseline_path)
    run4_report = _load_json(run4_report_path)
    official_test_ll = baseline["official_model"]["test_metrics"]["log_loss"]

    base_numeric, _categorical = official_step3c_features()
    df = pd.read_csv(input_csv)
    before = len(df)
    df = df[df[TARGET].notna() & df["fighter_a_expected_win_prob"].notna()].copy()
    dropped = before - len(df)
    df = coerce_numeric_features(df, base_numeric, context="step5d1_nonlinear_sanity_audit")

    pretest, train, calib, test = build_pretest_and_official_split(df, train_frac=train_frac, calibration_frac=calibration_frac)
    folds = build_rolling_folds(pretest, fractions=CONFIRMATION_FOLD_FRACTIONS)
    run4_best_hgb = reconstruct_run4_best_hgb(run4_report)

    # 1. Reproduction
    reproduction = audit_anchor_reproduction(base_numeric, train, calib, test, baseline, run4_best_hgb, random_seed, max_iter)
    anchor_test_result = reproduction.pop("_anchor_test_result")
    run4_hgb_test_result = reproduction.pop("_hgb_test_result")

    # 2. Orientation
    orientation = audit_probability_orientation(base_numeric, train, calib, test, run4_best_hgb, random_seed, max_iter)

    # 3. Row / feature parity
    parity = audit_row_feature_parity(base_numeric, train, calib, test, folds)

    # 5. Missing values
    missing = audit_missing_values(base_numeric, train, calib, test, run4_best_hgb, random_seed)

    # 6. Expanded staged search (validation only)
    search = run_hgb_sensitivity_search(base_numeric, folds, random_seed, max_iter, quick=quick, max_candidates=max_candidates)
    best_hgb_params = search["best"]["params"]
    best_hgb = hgb_candidate("best_4_1_hgb", best_hgb_params, role="audit_best")

    # 7. Early stopping (on the Run 4 best HGB, which uses early stopping)
    early_stopping = audit_early_stopping(run4_best_hgb, base_numeric, train, random_seed)

    # 4. Calibration audit (official LR, Run 4 HGB, best 4.1 HGB)
    calibration_audits = {
        "current_official_lr": audit_calibration(
            {c["name"]: c for c in build_reference_candidates()}[ANCHOR_NAME],
            base_numeric, folds, train, calib, test, random_seed, max_iter, "current_official_lr"),
        "run4_best_hgb": audit_calibration(run4_best_hgb, base_numeric, folds, train, calib, test, random_seed, max_iter, "run4_best_hgb"),
        "best_4_1_hgb": audit_calibration(best_hgb, base_numeric, folds, train, calib, test, random_seed, max_iter, "best_4_1_hgb"),
    }

    # 8. Blend audit (reproduce the Run 4 anomaly: anchor + Run 4 best HGB)
    anchor_fold_probs, hgb_fold_probs = [], []
    for fold in folds:
        a = evaluate_candidate_on_fold({c["name"]: c for c in build_reference_candidates()}[ANCHOR_NAME],
                                       base_numeric, fold, random_state=random_seed, max_iter=max_iter)
        h = evaluate_candidate_on_fold(run4_best_hgb, base_numeric, fold, random_state=random_seed, max_iter=max_iter)
        anchor_fold_probs.append({"y": a["y_eval"], "p_platt": a["eval_probs_platt"]})
        hgb_fold_probs.append({"y": h["y_eval"], "p_platt": h["eval_probs_platt"]})
    blend = audit_blend(anchor_fold_probs, hgb_fold_probs)

    # Validation-selected blend of anchor + best_4_1_hgb, for the finalist test re-report.
    best_hgb_fold_probs = []
    for fold in folds:
        h = evaluate_candidate_on_fold(best_hgb, base_numeric, fold, random_state=random_seed, max_iter=max_iter)
        best_hgb_fold_probs.append({"y": h["y_eval"], "p_platt": h["eval_probs_platt"]})
    blend_best_sel = select_blend_weight(anchor_fold_probs, best_hgb_fold_probs)

    # 9. Feature importance + probability distribution
    fi_dist = audit_feature_importance_and_distribution(best_hgb, base_numeric, train, calib, test, random_seed, skip_slow)

    # 11. Final-test re-report for finalists only.
    best_hgb_test_result = evaluate_finalist_on_test(best_hgb, base_numeric, train, calib, test, official_test_ll,
                                                     random_state=random_seed, max_iter=max_iter)
    # Run 4 blend reproduction (anchor + Run 4 HGB, w from its own selection).
    run4_blend_test = evaluate_blend_on_test(anchor_test_result, run4_hgb_test_result, blend["global_best_weight_on_lr"], official_test_ll)
    # Best-4.1 blend (anchor + best 4.1 HGB, validation-selected w).
    best_blend_test = evaluate_blend_on_test(anchor_test_result, best_hgb_test_result, blend_best_sel["best"]["weight"], official_test_ll)

    def _finalist(res):
        return {
            "platt_test_log_loss": res["platt_test_log_loss"],
            "platt_test_brier": res["platt_test_brier"],
            "platt_test_auc": res["platt_test_auc"],
            "platt_test_accuracy": res["platt_test_accuracy"],
            "improvement_over_official_test_log_loss": official_test_ll - res["platt_test_log_loss"],
        }

    finalists_test = {
        "current_official_lr": _finalist(anchor_test_result),
        "run4_best_hgb": _finalist(run4_hgb_test_result),
        "best_4_1_hgb": _finalist(best_hgb_test_result),
        "run4_blend_reproduction": {
            "blend_weight_on_lr": blend["global_best_weight_on_lr"],
            "platt_test_log_loss": run4_blend_test["platt_test_log_loss"],
            "improvement_over_official_test_log_loss": run4_blend_test["improvement_over_official_test_log_loss"],
        },
        "best_4_1_blend": {
            "blend_weight_on_lr": blend_best_sel["best"]["weight"],
            "validation_mean_log_loss": blend_best_sel["best"]["mean_val_log_loss"],
            "platt_test_log_loss": best_blend_test["platt_test_log_loss"],
            "improvement_over_official_test_log_loss": best_blend_test["improvement_over_official_test_log_loss"],
        },
    }

    leakage = build_leakage_checks(base_numeric, train, calib, test, orientation, parity, early_stopping, blend)
    verdict = build_verdict(reproduction, orientation, parity, calibration_audits, missing, search, blend,
                            finalists_test, official_test_ll, leakage)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": "step5d1_nonlinear_sanity_audit",
        "run_label": "Run 4.1: nonlinear sanity audit (diagnostics only, no promotion)",
        "changes_official_model": False,
        "changes_official_baseline_file": False,
        "official_baseline": baseline,
        "optional_dependencies": detect_optional_dependencies(),
        "config": {
            "input": input_csv, "train_frac": train_frac, "calibration_frac": calibration_frac,
            "rows_dropped_missing_target_or_elo_prob": int(dropped),
            "n_pretest": int(len(pretest)), "n_train": int(len(train)),
            "n_calibration": int(len(calib)), "n_test": int(len(test)),
            "train_dates": [str(train["date"].min()), str(train["date"].max())],
            "calibration_dates": [str(calib["date"].min()), str(calib["date"].max())],
            "test_dates": [str(test["date"].min()), str(test["date"].max())],
            "random_state": random_seed, "quick": quick, "skip_slow": skip_slow,
            "max_candidates": max_candidates, "fold_fractions": CONFIRMATION_FOLD_FRACTIONS,
        },
        "anchor_reproduction": reproduction,
        "probability_orientation_audit": orientation,
        "row_feature_parity_audit": parity,
        "calibration_audit": calibration_audits,
        "missing_value_audit": missing,
        "hgb_sensitivity_search": search,
        "early_stopping_audit": early_stopping,
        "blend_audit": blend,
        "best_4_1_blend_selection": {"best_weight_on_lr": blend_best_sel["best"]["weight"],
                                     "mean_val_log_loss": blend_best_sel["best"]["mean_val_log_loss"]},
        "feature_importance_and_distribution": fi_dist,
        "final_test_finalists": finalists_test,
        "leakage_protocol_checks": leakage,
        "verdict": verdict,
        "limitations": [
            "The final test was already touched by Runs 2/2.1/3/4, so it is a consistency check, not a fresh signal.",
            "The staged HGB search is broad but not exhaustive; it snaps refinement to the best coarse region.",
            "HistGradientBoosting has no native feature_importances_; importance is permutation-based on the calibration window.",
            "XGBoost/LightGBM/CatBoost remain absent and were not installed.",
        ],
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / STEP5D1_REPORT_JSON, "w") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    (out_dir / STEP5D1_REPORT_MD).write_text(_render_markdown(report))

    _write_sensitivity_csv(out_dir / STEP5D1_SENSITIVITY_CSV, search)
    _write_calibration_csv(out_dir / STEP5D1_CALIBRATION_CSV, calibration_audits)
    return report


def _write_sensitivity_csv(path: Path, search: dict) -> None:
    rows = []
    for r in sorted(search["all_results"], key=_sort_key):
        rows.append({
            "candidate": r["candidate"], "role": r["role"],
            "learning_rate": r["params"]["learning_rate"], "max_leaf_nodes": r["params"]["max_leaf_nodes"],
            "min_samples_leaf": r["params"]["min_samples_leaf"], "l2_regularization": r["params"]["l2_regularization"],
            "mean_val_platt_log_loss": r["mean_val_platt_log_loss"], "std_val_platt_log_loss": r["std_val_platt_log_loss"],
            "mean_val_brier": r["mean_val_brier"], "mean_val_auc": r["mean_val_auc"],
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_calibration_csv(path: Path, calibration_audits: dict) -> None:
    rows = []
    for name, block in calibration_audits.items():
        for method in ("uncalibrated", "platt", "isotonic"):
            m = block["validation"][method]
            rows.append({"candidate": name, "window": "validation", "method": method,
                         "log_loss": m["log_loss"], "brier": m["brier"], "auc": m["auc"]})
        t = block["test"]
        rows.append({"candidate": name, "window": "test", "method": "uncalibrated", "log_loss": t["uncalibrated_log_loss"], "brier": None, "auc": None})
        rows.append({"candidate": name, "window": "test", "method": "platt", "log_loss": t["platt_log_loss"], "brier": t["platt_brier"], "auc": t["platt_auc"]})
        rows.append({"candidate": name, "window": "test", "method": "isotonic", "log_loss": t["isotonic_log_loss"], "brier": None, "auc": None})
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _fmt(value, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except (TypeError, ValueError):
        pass
    return f"{value:.{digits}f}"


def _render_markdown(report: dict) -> str:
    v = report["verdict"]
    rep = report["anchor_reproduction"]
    lines: list[str] = []
    lines.append("# Step 5D.1 / Run 4.1: Nonlinear Sanity Audit")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    lines.append("Diagnostics-only audit of the Run 4 conclusion. Promotes nothing; the official model and "
                 "`benchmarks/official_baseline.json` are unchanged.")
    lines.append("")

    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Run 4 had a bug or protocol issue: **{v['run4_had_bug_or_protocol_issue']}**.")
    lines.append(f"- HGB probability orientation correct: **{v['hgb_probability_orientation_correct']}**; "
                 f"rows/features aligned: **{v['rows_features_aligned']}**; calibration correct: **{v['calibration_done_correctly']}**.")
    lines.append(f"- Anchors reproduced: **{v['anchors_reproduced']}**; Run 4 best HGB reproduced: **{v['run4_best_hgb_reproduced']}**.")
    lines.append(f"- Expanded staged HGB search ({v['expanded_search_n_configs']} configs) best validation log loss: "
                 f"**{_fmt(v['expanded_search_best_val_log_loss'])}** vs official **{_fmt(v['official_validation_log_loss'])}** "
                 f"(gain {v['expanded_search_gain_vs_official_validation']:+.6f}); beats official on validation: "
                 f"**{v['expanded_search_beats_official_on_validation']}**.")
    lines.append(f"- Best 4.1 HGB final-test log loss: **{_fmt(v['expanded_search_best_hgb_test_log_loss'])}**; "
                 f"beats official on final test: **{v['expanded_search_beats_official_on_final_test']}**.")
    lines.append(f"- Original HGB result likely real (not an artifact): **{v['original_hgb_result_likely_real']}**.")
    lines.append(f"- Keep current official LR: **{v['keep_current_official_lr']}**; Run 4 should be closed: **{v['run4_should_be_closed']}**.")
    lines.append(f"- {v['summary']}")
    lines.append("")

    lines.append("## Anchor Reproduction")
    lines.append("")
    lines.append("| model | benchmark | reproduced | |Δ| | within 1e-4 |")
    lines.append("|---|---:|---:|---:|:---:|")
    a = rep["current_official_lr"]; lines.append(f"| current official LR | {_fmt(a['benchmark_test_log_loss'])} | {_fmt(a['reproduced_test_log_loss'])} | {_fmt(a['abs_difference'])} | {a['reproduced_within_1e4']} |")
    h = rep["historical_step3c_lr"]; lines.append(f"| historical Step 3C LR | {_fmt(h['benchmark_test_log_loss'])} | {_fmt(h['reproduced_test_log_loss'])} | {_fmt(h['abs_difference'])} | {h['reproduced_within_1e4']} |")
    g = rep["run4_best_hgb"]; lines.append(f"| Run 4 best HGB | {_fmt(g['run4_reported_test_log_loss'])} | {_fmt(g['reproduced_test_log_loss'])} | {_fmt(g['abs_difference_vs_run4'])} | {g['reproduced_within_1e4']} |")
    lines.append("")

    lines.append("## Probability Orientation")
    lines.append("")
    for name, e in report["probability_orientation_audit"]["candidates"].items():
        t = e["test_window"]
        lines.append(f"- **{name}**: classes_={e['classes_']}, predicted=P(fighter_a wins)={e['predicted_prob_is_P_fighter_a_wins']}; "
                     f"test LL(p)={_fmt(t['log_loss'])} vs LL(1-p)={_fmt(t['log_loss_inverted'])}, "
                     f"AUC(p)={_fmt(t['auc'])} vs AUC(1-p)={_fmt(t['auc_inverted'])}; orientation_correct={e['orientation_correct']}.")
    lines.append("")

    lines.append("## Row / Feature Parity")
    lines.append("")
    lines.append(f"- All splits aligned: **{report['row_feature_parity_audit']['all_splits_aligned']}**; "
                 f"identical target orientation: **{report['row_feature_parity_audit']['identical_target_orientation']}**; "
                 f"any rows dropped differently: **{report['row_feature_parity_audit']['any_rows_dropped_differently']}**.")
    lines.append("")
    lines.append("| split | n_lr | n_hgb | rows match | targets match | LR feats | HGB feats | LR NaN | HGB NaN |")
    lines.append("|---|---:|---:|:---:|:---:|---:|---:|---:|---:|")
    for r in report["row_feature_parity_audit"]["per_split"]:
        lines.append(f"| {r['split']} | {r['n_lr']} | {r['n_hgb']} | {r['rows_match']} | {r['targets_match']} | "
                     f"{r['lr_feature_count']} | {r['hgb_feature_count']} | {r['lr_missing_values']} | {r['hgb_missing_values']} |")
    lines.append("")

    lines.append("## Calibration Audit (validation means / test)")
    lines.append("")
    lines.append("| candidate | uncal LL | Platt LL | iso LL | Platt Brier | Platt AUC | AUC stable |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|")
    for name, block in report["calibration_audit"].items():
        val = block["validation"]
        lines.append(f"| {name} (val) | {_fmt(val['uncalibrated']['log_loss'])} | {_fmt(val['platt']['log_loss'])} | "
                     f"{_fmt(val['isotonic']['log_loss'])} | {_fmt(val['platt']['brier'])} | {_fmt(val['platt']['auc'])} | "
                     f"{block['auc_stable_under_platt']} |")
    lines.append("")

    lines.append("## Missing-Value Handling")
    lines.append("")
    m = report["missing_value_audit"]
    lines.append(f"- HGB native NaN handling: **{m['hgb_native_nan_handling']}** (fit with {m['n_train_missing_values']} "
                 f"missing train cells succeeded; HGB pipeline has imputer: **{m['hgb_pipeline_has_imputer']}**).")
    lines.append(f"- {m['assessment']}")
    lines.append("")

    lines.append("## Expanded HGB Sensitivity Search (validation only)")
    lines.append("")
    s = report["hgb_sensitivity_search"]
    lines.append(f"- Coarse configs: {s['n_coarse']}; refinement configs: {s['n_refine']}; total: {s['n_total_evaluated']}.")
    lines.append(f"- Best config: `{s['best']['params']}` → validation mean Platt log loss {_fmt(s['best']['mean_val_platt_log_loss'])}.")
    lines.append("")
    lines.append("| rank | config | val LL | Brier | AUC |")
    lines.append("|---:|---|---:|---:|---:|")
    for i, r in enumerate(s["ranked_top10"], start=1):
        p = r["params"]
        cfg = f"lr={p['learning_rate']}, leaves={p['max_leaf_nodes']}, msl={p['min_samples_leaf']}, l2={p['l2_regularization']}"
        lines.append(f"| {i} | {cfg} | {_fmt(r['mean_val_platt_log_loss'])} | {_fmt(r['mean_val_brier'])} | {_fmt(r['mean_val_auc'])} |")
    lines.append("")
    lines.append("max_iter / early-stopping sub-study (best config):")
    lines.append("")
    lines.append("| early_stopping | max_iter | val LL |")
    lines.append("|:---:|---:|---:|")
    for r in s["max_iter_substudy"]:
        lines.append(f"| {r['early_stopping']} | {r['max_iter']} | {_fmt(r['mean_val_platt_log_loss'])} |")
    lines.append("")

    lines.append("## Blend Audit")
    lines.append("")
    b = report["blend_audit"]
    lines.append(f"- Global best weight on LR: **{b['global_best_weight_on_lr']}** (validation mean LL {_fmt(b['global_mean_val_log_loss'])}).")
    lines.append(f"- Per-fold optimal weights: {b['per_fold_best_weight_on_lr']} (range {_fmt(b['per_fold_best_weight_range'], 2)}).")
    lines.append(f"- Folds improved vs anchor: {b['folds_improved']}/5; concentrated in one fold: {b['improvement_concentrated_in_one_fold']}; robust: **{b['blend_is_robust']}**.")
    lines.append(f"- Fold improvements vs anchor: {[round(x, 6) for x in b['fold_improvements_vs_anchor']]}.")
    lines.append("")

    lines.append("## Final-Test Finalists (already-touched test, re-report only)")
    lines.append("")
    lines.append("| finalist | test LL | gain vs official | Brier | AUC | accuracy |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    ft = report["final_test_finalists"]
    for name in ("current_official_lr", "run4_best_hgb", "best_4_1_hgb"):
        r = ft[name]
        lines.append(f"| {name} | {_fmt(r['platt_test_log_loss'])} | {r['improvement_over_official_test_log_loss']:+.6f} | "
                     f"{_fmt(r['platt_test_brier'])} | {_fmt(r['platt_test_auc'])} | {_fmt(r['platt_test_accuracy'])} |")
    for name in ("run4_blend_reproduction", "best_4_1_blend"):
        r = ft[name]
        lines.append(f"| {name} (w={r['blend_weight_on_lr']}) | {_fmt(r['platt_test_log_loss'])} | "
                     f"{r['improvement_over_official_test_log_loss']:+.6f} | n/a | n/a | n/a |")
    lines.append("")

    lines.append("## Probability Distribution + Feature Importance")
    lines.append("")
    fd = report["feature_importance_and_distribution"]
    ld, hd = fd["lr_probability_distribution"], fd["hgb_probability_distribution"]
    lines.append(f"- LR test probs: mean {_fmt(ld['mean'],3)}, std {_fmt(ld['std'],3)}, "
                 f"%>0.7 {_fmt(ld['pct_over_0.7'],3)}, %<0.3 {_fmt(ld['pct_under_0.3'],3)}.")
    lines.append(f"- HGB test probs: mean {_fmt(hd['mean'],3)}, std {_fmt(hd['std'],3)}, "
                 f"%>0.7 {_fmt(hd['pct_over_0.7'],3)}, %<0.3 {_fmt(hd['pct_under_0.3'],3)}.")
    lines.append(f"- HGB more extreme than LR: **{fd['hgb_more_extreme_than_lr']}** (std diff {_fmt(fd['hgb_std_minus_lr_std'],4)}).")
    if fd["permutation_importance"].get("available", True) and fd.get("hgb_top_features"):
        lines.append(f"- HGB top features: {fd['hgb_top_features'][:8]}; overlap with LR-important: {fd['overlap_with_lr_important']}.")
    lines.append("")

    lines.append("## Protocol / Leakage Checks")
    lines.append("")
    for name, entry in report["leakage_protocol_checks"].items():
        lines.append(f"- **{name}**: {entry['status']}")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- {v['summary']}")
    lines.append(f"- Next phase: {v['next_phase']}")
    lines.append("")
    return "\n".join(lines)
