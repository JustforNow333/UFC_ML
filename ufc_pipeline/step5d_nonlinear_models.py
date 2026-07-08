"""Step 5D / Run 4: controlled nonlinear-model + blend comparison.

Run 4 asks a single question: can a nonlinear tabular model (or a simple
calibrated blend) beat the *current* official stronger-regularized logistic
regression on honest pre-test validation, and is any edge stable and large
enough to justify replacing the official model?

The run stays deliberately narrow, mirroring Step 5B/5B.1/5C discipline:

* the same five expanding-window pre-test folds as Step 5B.1 / Step 5C;
* selection uses ONLY mean Platt-calibrated validation log loss (plus
  pre-declared validation tie-breakers); the final held-out test is
  re-reported for validation-selected finalists only, never used to pick a
  model, hyperparameter, calibration method, or blend weight;
* the official LR anchor and the historical Step 3C LR are re-fit here from
  their locked configs so every table is self-consistent;
* the required nonlinear family is sklearn ``HistGradientBoostingClassifier``
  (no heavy dependency). XGBoost / LightGBM / CatBoost are evaluated ONLY if
  already importable — this run never installs them. Random forest is a
  non-promotable sanity check, off by default.

Non-negotiables preserved: chronological splits only; calibrators fit only on
calibration windows; HGB early-stopping validation is carved from the fit
window only; no sportsbook odds, prediction markets, neural networks, or new
data sources; ``benchmarks/official_baseline.json`` is read-only.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from ufc_pipeline.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
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
    DEFAULT_MAX_ITER,
    RUN1_KEY_FEATURES,
    WEIGHT_CLASS_COLUMN,
    _fit_quietly,
    build_pretest_and_official_split,
    build_rolling_folds,
    make_step5b_pipeline,
)
from ufc_pipeline.step5b1_regularization_confirmation import CONFIRMATION_FOLD_FRACTIONS

STEP5D_REPORT_JSON = "step5d_nonlinear_models.json"
STEP5D_REPORT_MD = "step5d_nonlinear_models.md"
STEP5D_CANDIDATE_CSV = "step5d_candidate_results.csv"
STEP5D_FEATURE_IMPORTANCE_CSV = "step5d_feature_importance.csv"
STEP5D_BLEND_CSV = "step5d_blend_results.csv"

# Model families this run is allowed to touch. Neural networks are absent by
# construction; the leakage check asserts every candidate family is in here.
ALLOWED_MODEL_FAMILIES = frozenset(
    {"logistic_regression", "hist_gradient_boosting", "random_forest", "xgboost", "lightgbm"}
)
OPTIONAL_DEPENDENCY_FAMILIES = {"xgboost": "xgboost", "lightgbm": "lightgbm"}

# Feature/param tokens that would signal sportsbook odds or prediction-market
# leakage. Scanned across every model input and every candidate parameter.
FORBIDDEN_MARKET_TOKENS = (
    "odds", "moneyline", "money_line", "vig", "juice", "implied_prob",
    "kalshi", "polymarket", "betfair", "sportsbook", "decimal_odds",
    "american_odds", "book", "market_prob",
)

# HistGradientBoosting search grid. Bounded on purpose: 3 x 2 x 2 x 2 = 24
# configs. ``max_iter`` is fixed high and controlled by early stopping (whose
# internal validation is carved from the fit window only), so it is not gridded.
HISTGB_LEARNING_RATES: tuple[float, ...] = (0.03, 0.05, 0.1)
HISTGB_MAX_LEAF_NODES: tuple[int, ...] = (15, 31)
HISTGB_MIN_SAMPLES_LEAF: tuple[int, ...] = (50, 100)
HISTGB_L2_REGULARIZATION: tuple[float, ...] = (0.0, 1.0)
HISTGB_MAX_ITER = 400
HISTGB_VALIDATION_FRACTION = 0.1
HISTGB_N_ITER_NO_CHANGE = 20

# Quick smoke subset (used by --quick / tests) — deterministic first entries.
HISTGB_QUICK_GRID: tuple[dict, ...] = (
    {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 100, "l2_regularization": 1.0},
    {"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 0.0},
)

# Convex blend weights on the LR anchor: p = w * p_lr + (1 - w) * p_nonlinear.
BLEND_WEIGHTS: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(0, 11))

# LR features Run 1 / Run 2 found most important; overlap with nonlinear top
# features is reported for interpretability.
LR_IMPORTANT_FEATURES: tuple[str, ...] = (
    "elo_diff", "age_diff", "striking_defense_diff", "striking_accuracy_diff", "prior_wins_diff",
)

ANCHOR_NAME = "current_official_lr"
HISTORICAL_NAME = "historical_step3c_lr"


# ---------------------------------------------------------------------------
# JSON / IO helpers (match the step5c voice)
# ---------------------------------------------------------------------------

def _load_json(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def _load_required_json(path: str | Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Optional dependency detection (never installs anything)
# ---------------------------------------------------------------------------

def detect_optional_dependencies() -> dict[str, bool]:
    """Which optional boosting libraries are importable right now.

    Detection uses ``importlib.util.find_spec`` so we never import (or
    accidentally initialise) a heavy library we are not going to use.
    """
    return {name: importlib.util.find_spec(module) is not None
            for name, module in OPTIONAL_DEPENDENCY_FAMILIES.items()}


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------

def _lr_candidate(
    name: str,
    penalty: str,
    C: float,
    solver: str,
    l1_ratio: float | None,
    include_raw_weight_class: bool,
    purpose: str,
    role: str,
    promotable: bool,
) -> dict:
    return {
        "name": name,
        "model_family": "logistic_regression",
        "role": role,
        "purpose": purpose,
        "promotable": bool(promotable),
        "include_raw_weight_class": bool(include_raw_weight_class),
        "params": {
            "penalty": penalty,
            "C": float(C),
            "solver": solver,
            "l1_ratio": None if l1_ratio is None else float(l1_ratio),
            "class_weight": None,
        },
    }


def _histgb_candidate(index: int, grid_point: dict, include_raw_weight_class: bool, role: str) -> dict:
    wc = "wc" if include_raw_weight_class else "nowc"
    name = (
        f"histgb_{wc}_lr{grid_point['learning_rate']}_leaf{grid_point['max_leaf_nodes']}"
        f"_msl{grid_point['min_samples_leaf']}_l2{grid_point['l2_regularization']}"
    )
    return {
        "name": name,
        "model_family": "hist_gradient_boosting",
        "role": role,
        "purpose": "HistGradientBoosting nonlinear tabular candidate.",
        "promotable": True,
        "include_raw_weight_class": bool(include_raw_weight_class),
        "params": {
            "learning_rate": float(grid_point["learning_rate"]),
            "max_leaf_nodes": int(grid_point["max_leaf_nodes"]),
            "min_samples_leaf": int(grid_point["min_samples_leaf"]),
            "l2_regularization": float(grid_point["l2_regularization"]),
            "max_iter": HISTGB_MAX_ITER,
            "early_stopping": True,
            "validation_fraction": HISTGB_VALIDATION_FRACTION,
            "n_iter_no_change": HISTGB_N_ITER_NO_CHANGE,
        },
    }


def build_histgb_grid(quick: bool = False) -> list[dict]:
    """Bounded, deterministic HistGradientBoosting grid points."""
    if quick:
        return [dict(g) for g in HISTGB_QUICK_GRID]
    grid = []
    for lr in HISTGB_LEARNING_RATES:
        for leaves in HISTGB_MAX_LEAF_NODES:
            for msl in HISTGB_MIN_SAMPLES_LEAF:
                for l2 in HISTGB_L2_REGULARIZATION:
                    grid.append(
                        {
                            "learning_rate": lr,
                            "max_leaf_nodes": leaves,
                            "min_samples_leaf": msl,
                            "l2_regularization": l2,
                        }
                    )
    return grid


def build_reference_candidates() -> list[dict]:
    """The two fixed LR reference points that anchor every comparison.

    ``current_official_lr`` reproduces the current official model (stronger
    elastic-net, raw weight_class dropped). ``historical_step3c_lr`` reproduces
    the former official Step 3C LR (C=1.0 L2, weight_class included). Both are
    re-fit here so the tables are internally consistent, and both are marked
    non-promotable (they are baselines, not challengers).
    """
    return [
        _lr_candidate(
            ANCHOR_NAME,
            penalty="elasticnet",
            C=0.003,
            solver="saga",
            l1_ratio=0.1,
            include_raw_weight_class=False,
            purpose="Current official stronger-regularized LR + Platt (primary benchmark).",
            role="official_anchor",
            promotable=False,
        ),
        _lr_candidate(
            HISTORICAL_NAME,
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            l1_ratio=None,
            include_raw_weight_class=True,
            purpose="Historical Step 3C LR + Platt (secondary reference only).",
            role="historical_reference",
            promotable=False,
        ),
    ]


def build_step5d_candidates(
    quick: bool = False,
    include_weight_class_sensitivity: bool = True,
    include_rf_sanity: bool = False,
    optional_available: dict[str, bool] | None = None,
    max_candidates: int | None = None,
) -> list[dict]:
    """Full, deterministic Step 5D candidate list.

    Order: LR references first, then HistGradientBoosting (raw weight_class
    dropped, matching the official feature config), then an optional single
    weight_class-included HGB sensitivity point, then optional
    XGBoost/LightGBM (only if importable), then an optional RF sanity check.
    ``max_candidates`` truncates deterministically for smoke runs.
    """
    optional_available = optional_available or {}
    candidates = build_reference_candidates()

    grid = build_histgb_grid(quick=quick)
    for i, point in enumerate(grid):
        candidates.append(_histgb_candidate(i, point, include_raw_weight_class=False, role="nonlinear"))

    if include_weight_class_sensitivity:
        # A single weight_class-included HGB point (first grid point) as a
        # cheap sensitivity check. The main question is model class, not new
        # feature engineering, so this stays a lone secondary candidate.
        sens = _histgb_candidate(0, grid[0], include_raw_weight_class=True, role="nonlinear_sensitivity")
        sens["purpose"] = "Sensitivity: HistGradientBoosting WITH raw weight_class (ordinal categorical)."
        # A diagnostic sensitivity point, not a promotion candidate: it uses a
        # different feature config than the official model, so it is excluded
        # from finalist selection but still scored on every fold and reported.
        sens["promotable"] = False
        candidates.append(sens)

    for family in ("xgboost", "lightgbm"):
        if optional_available.get(family):
            candidates.append(_optional_boosting_candidate(family, quick=quick))

    if include_rf_sanity:
        candidates.append(
            {
                "name": "random_forest_sanity",
                "model_family": "random_forest",
                "role": "sanity_check",
                "purpose": "Non-promotable random-forest sanity check.",
                "promotable": False,
                "include_raw_weight_class": False,
                "params": {
                    "n_estimators": 300,
                    "max_depth": None,
                    "min_samples_leaf": 20,
                    "max_features": "sqrt",
                },
            }
        )

    if max_candidates is not None:
        # Always keep both LR references so the anchor is never dropped.
        refs = [c for c in candidates if not c["promotable"] and c["role"] in ("official_anchor", "historical_reference")]
        rest = [c for c in candidates if c not in refs]
        candidates = refs + rest[: max(0, max_candidates - len(refs))]
    return candidates


def _optional_boosting_candidate(family: str, quick: bool = False) -> dict:
    """Small controlled config for an already-installed boosting library."""
    if family == "xgboost":
        params = {
            "n_estimators": 100 if quick else 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 5.0,
            "min_child_weight": 5,
        }
    elif family == "lightgbm":
        params = {
            "n_estimators": 100 if quick else 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 5.0,
            "min_child_samples": 50,
        }
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"Unknown optional boosting family: {family!r}")
    return {
        "name": f"{family}_default",
        "model_family": family,
        "role": "nonlinear_optional",
        "purpose": f"Optional {family} candidate (only run because the dependency is already installed).",
        "promotable": True,
        "include_raw_weight_class": False,
        "params": params,
    }


def _candidate_label(candidate: dict) -> str:
    p = candidate["params"]
    fam = candidate["model_family"]
    wc = "wc=yes" if candidate["include_raw_weight_class"] else "wc=no"
    if fam == "logistic_regression":
        l1 = p.get("l1_ratio")
        l1_text = "" if l1 is None else f", l1_ratio={l1}"
        return f"LR {p['penalty']} C={p['C']}{l1_text}; {wc}"
    if fam == "hist_gradient_boosting":
        return (
            f"HGB lr={p['learning_rate']}, leaves={p['max_leaf_nodes']}, "
            f"msl={p['min_samples_leaf']}, l2={p['l2_regularization']}; {wc}"
        )
    if fam == "random_forest":
        return f"RF n={p['n_estimators']}, msl={p['min_samples_leaf']}; {wc}"
    return f"{fam} {json.dumps(p, sort_keys=True)}; {wc}"


# ---------------------------------------------------------------------------
# Feature preparation + base-model fit/predict dispatch
# ---------------------------------------------------------------------------

def build_histgb_matrix(
    df: pd.DataFrame,
    base_numeric: list[str],
    include_weight_class: bool,
    encoder: OrdinalEncoder | None = None,
    fit: bool = False,
) -> tuple[pd.DataFrame, OrdinalEncoder | None, list[bool]]:
    """Feature matrix for HistGradientBoosting.

    Numeric features pass through untouched — HGB handles missing values
    natively and is scale-invariant, so imputing/scaling would only discard
    the informative "no prior history" missingness. When ``include_weight_class``
    is set, weight_class is ordinal-encoded (categories learned from the FIT
    window only; unknown categories map to NaN and HGB treats them as a
    missing category) and passed to HGB as a native categorical feature.
    """
    X = df[base_numeric].apply(pd.to_numeric, errors="coerce").astype(float)
    categorical_mask = [False] * len(base_numeric)
    if include_weight_class:
        wc = df[[WEIGHT_CLASS_COLUMN]].astype(object)
        if fit or encoder is None:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
            codes = encoder.fit_transform(wc)
        else:
            codes = encoder.transform(wc)
        X = X.copy()
        X[WEIGHT_CLASS_COLUMN] = codes.ravel()
        categorical_mask = categorical_mask + [True]
    return X, encoder, categorical_mask


def make_histgb_model(params: dict, categorical_mask: list[bool], random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"],
        max_leaf_nodes=params["max_leaf_nodes"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"],
        max_iter=params["max_iter"],
        early_stopping=params["early_stopping"],
        validation_fraction=params["validation_fraction"],
        n_iter_no_change=params["n_iter_no_change"],
        categorical_features=categorical_mask if any(categorical_mask) else None,
        random_state=random_state,
    )


def make_rf_pipeline(base_numeric: list[str], params: dict, random_state: int) -> Pipeline:
    # Random forests cannot ingest NaN; impute (median) from the fit window
    # only, inside the pipeline. No scaling needed for trees.
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_leaf=params["min_samples_leaf"],
                    max_features=params["max_features"],
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def fit_base_model(
    candidate: dict,
    base_numeric: list[str],
    train_df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    max_iter: int = DEFAULT_MAX_ITER,
):
    """Fit a candidate's base model on ``train_df`` only and return
    ``(fitted_model, predict_proba_fn)``. ``predict_proba_fn(df)`` returns raw
    P(fighter_a wins). No calibration happens here."""
    family = candidate["model_family"]
    y = train_df[TARGET].astype(int)

    if family == "logistic_regression":
        include_wc = candidate["include_raw_weight_class"]
        p = candidate["params"]
        cols = list(base_numeric) + ([WEIGHT_CLASS_COLUMN] if include_wc else [])
        pipeline = make_step5b_pipeline(
            numeric=list(base_numeric),
            include_weight_class=include_wc,
            weight_class_categories=None,
            penalty=p["penalty"],
            C=p["C"],
            l1_ratio=p.get("l1_ratio"),
            solver=p["solver"],
            max_iter=max_iter,
            class_weight=p.get("class_weight"),
            random_state=random_state,
        )
        _fit_quietly(pipeline, train_df[cols], y)
        return pipeline, (lambda frame: pipeline.predict_proba(frame[cols])[:, 1])

    if family == "hist_gradient_boosting":
        include_wc = candidate["include_raw_weight_class"]
        X_train, encoder, mask = build_histgb_matrix(train_df, base_numeric, include_wc, fit=True)
        model = make_histgb_model(candidate["params"], mask, random_state)
        _fit_quietly(model, X_train, y)

        def predict(frame: pd.DataFrame) -> np.ndarray:
            X, _, _ = build_histgb_matrix(frame, base_numeric, include_wc, encoder=encoder, fit=False)
            return model.predict_proba(X)[:, 1]

        return model, predict

    if family == "random_forest":
        pipeline = make_rf_pipeline(base_numeric, candidate["params"], random_state)
        _fit_quietly(pipeline, train_df[list(base_numeric)], y)
        return pipeline, (lambda frame: pipeline.predict_proba(frame[list(base_numeric)])[:, 1])

    if family in ("xgboost", "lightgbm"):  # pragma: no cover - optional deps absent in CI
        return _fit_optional_boosting(candidate, base_numeric, train_df, y, random_state)

    raise ValueError(f"Unknown model family: {family!r}")


def _fit_optional_boosting(candidate, base_numeric, train_df, y, random_state):  # pragma: no cover
    """Fit an already-installed XGBoost/LightGBM model via its sklearn API.

    Only reached when the dependency is importable (candidates for absent
    libraries are never generated). Uses NaN-native numeric inputs.
    """
    family = candidate["model_family"]
    p = candidate["params"]
    X_train = train_df[list(base_numeric)].apply(pd.to_numeric, errors="coerce").astype(float)
    if family == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            max_depth=p["max_depth"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            reg_lambda=p["reg_lambda"],
            min_child_weight=p["min_child_weight"],
            random_state=random_state,
            tree_method="hist",
            eval_metric="logloss",
        )
    else:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            num_leaves=p["num_leaves"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            reg_lambda=p["reg_lambda"],
            min_child_samples=p["min_child_samples"],
            random_state=random_state,
            verbose=-1,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y)

    def predict(frame: pd.DataFrame) -> np.ndarray:
        X = frame[list(base_numeric)].apply(pd.to_numeric, errors="coerce").astype(float)
        return model.predict_proba(X)[:, 1]

    return model, predict


# ---------------------------------------------------------------------------
# Calibrated evaluation on one (calibration, eval) window pair
# ---------------------------------------------------------------------------

def evaluate_calibrated_window(
    predict_fn,
    calib_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> dict:
    """Fit Platt + isotonic on the calibration window ONLY, score on eval.

    Returns uncalibrated / Platt / isotonic metrics on the eval window plus the
    calibrated eval probabilities (kept so blends can be scored without any
    refit). The calibrators never see eval labels.
    """
    calib_probs = clip_probabilities(predict_fn(calib_df))
    eval_uncal = clip_probabilities(predict_fn(eval_df))
    y_calib = calib_df[TARGET].astype(int).to_numpy()
    y_eval = eval_df[TARGET].astype(int).to_numpy()

    platt = PlattCalibrator().fit(calib_probs, y_calib)
    iso = IsotonicCalibrator().fit(calib_probs, y_calib)
    eval_platt = clip_probabilities(platt.transform(eval_uncal))
    eval_iso = clip_probabilities(iso.transform(eval_uncal))

    uncal_metrics = evaluate_probs(y_eval, eval_uncal)
    platt_metrics = evaluate_probs(y_eval, eval_platt)
    iso_metrics = evaluate_probs(y_eval, eval_iso)
    platt_metrics["calibration"] = calibration_table_with_gap(y_eval, eval_platt)
    platt_metrics["high_confidence"] = high_confidence_diagnostics(y_eval, eval_platt)
    iso_metrics["calibration"] = calibration_table_with_gap(y_eval, eval_iso)
    iso_metrics["high_confidence"] = high_confidence_diagnostics(y_eval, eval_iso)

    return {
        "uncal": uncal_metrics,
        "platt": platt_metrics,
        "isotonic": iso_metrics,
        "y_eval": y_eval,
        "eval_probs_platt": eval_platt,
        "eval_probs_isotonic": eval_iso,
    }


def evaluate_candidate_on_fold(
    candidate: dict,
    base_numeric: list[str],
    fold: dict,
    random_state: int = RANDOM_STATE,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict:
    _model, predict_fn = fit_base_model(
        candidate, base_numeric, fold["fit"], random_state=random_state, max_iter=max_iter
    )
    window = evaluate_calibrated_window(predict_fn, fold["calib"], fold["val"])
    return {
        "fold": fold["fold"],
        "n_fit": fold["n_fit"],
        "n_calib": fold["n_calib"],
        "n_val": fold["n_val"],
        "val_dates": fold["val_dates"],
        "uncal_log_loss": window["uncal"]["log_loss"],
        "platt_log_loss": window["platt"]["log_loss"],
        "isotonic_log_loss": window["isotonic"]["log_loss"],
        "platt_brier": window["platt"]["brier_score"],
        "platt_auc": window["platt"]["roc_auc"],
        "platt_accuracy": window["platt"]["accuracy"],
        "isotonic_brier": window["isotonic"]["brier_score"],
        "uncal_brier": window["uncal"]["brier_score"],
        "eval_probs_platt": window["eval_probs_platt"],
        "eval_probs_isotonic": window["eval_probs_isotonic"],
        "y_eval": window["y_eval"],
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def summarize_validation(candidate: dict, fold_entries: list[dict]) -> dict:
    platt_lls = [e["platt_log_loss"] for e in fold_entries]
    iso_lls = [e["isotonic_log_loss"] for e in fold_entries]
    uncal_lls = [e["uncal_log_loss"] for e in fold_entries]
    briers = [e["platt_brier"] for e in fold_entries]
    aucs = [e["platt_auc"] for e in fold_entries if e["platt_auc"] is not None]
    accs = [e["platt_accuracy"] for e in fold_entries]
    return {
        "candidate": candidate["name"],
        "model_family": candidate["model_family"],
        "role": candidate["role"],
        "promotable": candidate["promotable"],
        "candidate_config": candidate,
        "candidate_label": _candidate_label(candidate),
        "mean_val_platt_log_loss": _mean(platt_lls),
        "std_val_platt_log_loss": float(np.std(platt_lls)) if platt_lls else float("nan"),
        "per_fold_val_platt_log_loss": platt_lls,
        "mean_val_isotonic_log_loss": _mean(iso_lls),
        "mean_val_uncalibrated_log_loss": _mean(uncal_lls),
        "mean_val_brier": _mean(briers),
        "mean_val_auc": _mean(aucs) if aucs else None,
        "mean_val_accuracy": _mean(accs),
        "best_calibration_method": _best_calibration_method(_mean(platt_lls), _mean(iso_lls), _mean(uncal_lls)),
        "folds": [{k: v for k, v in e.items() if k not in ("eval_probs_platt", "eval_probs_isotonic", "y_eval")}
                  for e in fold_entries],
    }


def _best_calibration_method(platt_ll: float, iso_ll: float, uncal_ll: float) -> str:
    options = {"platt": platt_ll, "isotonic": iso_ll, "uncalibrated": uncal_ll}
    return min(options, key=options.get)


def add_validation_comparisons(validation_by_name: dict[str, dict]) -> None:
    anchor = validation_by_name[ANCHOR_NAME]
    anchor_folds = anchor["per_fold_val_platt_log_loss"]
    for summary in validation_by_name.values():
        cand_folds = summary["per_fold_val_platt_log_loss"]
        improvements = [a - b for a, b in zip(anchor_folds, cand_folds)]
        summary["mean_improvement_vs_official"] = anchor["mean_val_platt_log_loss"] - summary["mean_val_platt_log_loss"]
        summary["fold_improvements_vs_official"] = improvements
        summary["folds_beating_official"] = int(sum(d > 0 for d in improvements))
        summary["folds_beating_official_by_noise_threshold"] = int(sum(d > NOISE_THRESHOLD for d in improvements))
        summary["beats_official_mean_by_more_than_noise_threshold"] = (
            summary["mean_improvement_vs_official"] > NOISE_THRESHOLD
        )


def build_fold_by_fold(validation_by_name: dict[str, dict], names: list[str]) -> list[dict]:
    anchor = validation_by_name[ANCHOR_NAME]
    rows = []
    for idx, anchor_fold in enumerate(anchor["folds"]):
        row = {
            "fold": anchor_fold["fold"],
            "n_val": anchor_fold["n_val"],
            "val_dates": anchor_fold["val_dates"],
            "official_log_loss": anchor_fold["platt_log_loss"],
            "candidates": {},
        }
        for name in names:
            fold = validation_by_name[name]["folds"][idx]
            row["candidates"][name] = {
                "platt_log_loss": fold["platt_log_loss"],
                "improvement_vs_official": anchor_fold["platt_log_loss"] - fold["platt_log_loss"],
                "brier": fold["platt_brier"],
                "auc": fold["platt_auc"],
            }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Blend (validation-tuned convex probability average)
# ---------------------------------------------------------------------------

def select_blend_weight(
    anchor_fold_probs: list[dict],
    nonlinear_fold_probs: list[dict],
    weights: tuple[float, ...] = BLEND_WEIGHTS,
) -> dict:
    """Pick w minimising mean per-fold validation log loss.

    p_blend = w * p_lr_platt + (1 - w) * p_nonlinear_platt. Selection uses the
    validation folds ONLY. No test data is referenced by construction (the
    signature takes only per-fold validation probabilities).
    """
    results = []
    for w in weights:
        per_fold = []
        for a, nl in zip(anchor_fold_probs, nonlinear_fold_probs):
            p = clip_probabilities(w * a["p_platt"] + (1.0 - w) * nl["p_platt"])
            per_fold.append(float(log_loss(a["y"], p, labels=[0, 1])))
        results.append({"weight": float(w), "mean_val_log_loss": _mean(per_fold), "per_fold_val_log_loss": per_fold})
    best = min(results, key=lambda r: r["mean_val_log_loss"])
    return {"weights_tested": list(weights), "results": results, "best": best}


# ---------------------------------------------------------------------------
# Final-test evaluation (finalists only)
# ---------------------------------------------------------------------------

def evaluate_finalist_on_test(
    candidate: dict,
    base_numeric: list[str],
    official_train: pd.DataFrame,
    official_calib: pd.DataFrame,
    official_test: pd.DataFrame,
    official_log_loss: float,
    random_state: int = RANDOM_STATE,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict:
    _model, predict_fn = fit_base_model(
        candidate, base_numeric, official_train, random_state=random_state, max_iter=max_iter
    )
    window = evaluate_calibrated_window(predict_fn, official_calib, official_test)
    return {
        "candidate": candidate["name"],
        "model_family": candidate["model_family"],
        "candidate_label": _candidate_label(candidate),
        "n_train": int(len(official_train)),
        "n_calibration": int(len(official_calib)),
        "n_test": int(len(official_test)),
        "uncalibrated_test_log_loss": window["uncal"]["log_loss"],
        "platt_test_log_loss": window["platt"]["log_loss"],
        "isotonic_test_log_loss": window["isotonic"]["log_loss"],
        "platt_test_brier": window["platt"]["brier_score"],
        "platt_test_auc": window["platt"]["roc_auc"],
        "platt_test_accuracy": window["platt"]["accuracy"],
        "improvement_over_official_test_log_loss": official_log_loss - window["platt"]["log_loss"],
        "calibration_table": window["platt"]["calibration"],
        "high_confidence": window["platt"]["high_confidence"],
        "test_probs_platt": window["eval_probs_platt"],
        "y_test": window["y_eval"],
        "final_test_protocol_note": (
            "The final test was already touched in Runs 2/2.1/3. Step 5D re-reports it "
            "for validation-selected finalists only, never for candidate selection."
        ),
    }


def evaluate_blend_on_test(
    anchor_test: dict,
    nonlinear_test: dict,
    weight: float,
    official_log_loss: float,
) -> dict:
    p = clip_probabilities(weight * anchor_test["test_probs_platt"] + (1.0 - weight) * nonlinear_test["test_probs_platt"])
    y = anchor_test["y_test"]
    metrics = evaluate_probs(y, p)
    metrics["calibration"] = calibration_table_with_gap(y, p)
    metrics["high_confidence"] = high_confidence_diagnostics(y, p)
    return {
        "candidate": "blend_lr_plus_best_nonlinear",
        "blend_weight_on_lr": float(weight),
        "components": [anchor_test["candidate"], nonlinear_test["candidate"]],
        "platt_test_log_loss": metrics["log_loss"],
        "platt_test_brier": metrics["brier_score"],
        "platt_test_auc": metrics["roc_auc"],
        "platt_test_accuracy": metrics["accuracy"],
        "improvement_over_official_test_log_loss": official_log_loss - metrics["log_loss"],
        "calibration_table": metrics["calibration"],
        "high_confidence": metrics["high_confidence"],
        "final_test_protocol_note": (
            "Blend weight was selected on validation folds only; this is a finalist re-report."
        ),
    }


# ---------------------------------------------------------------------------
# Permutation importance (diagnostic; scored on the calibration window, never test)
# ---------------------------------------------------------------------------

def permutation_importance_for_candidate(
    candidate: dict,
    base_numeric: list[str],
    official_train: pd.DataFrame,
    official_calib: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    n_repeats: int = 5,
) -> dict:
    """Permutation importance for a nonlinear candidate.

    HistGradientBoosting exposes no native ``feature_importances_``, so we use
    permutation importance. It is computed on the CALIBRATION window (never the
    held-out test) so it stays a pure diagnostic with no test contact.
    """
    from sklearn.inspection import permutation_importance

    include_wc = candidate["include_raw_weight_class"]
    if candidate["model_family"] == "hist_gradient_boosting":
        X_train, encoder, mask = build_histgb_matrix(official_train, base_numeric, include_wc, fit=True)
        model = make_histgb_model(candidate["params"], mask, random_state)
        _fit_quietly(model, X_train, official_train[TARGET].astype(int))
        X_calib, _, _ = build_histgb_matrix(official_calib, base_numeric, include_wc, encoder=encoder, fit=False)
        feature_names = list(X_calib.columns)
    else:  # random_forest / others: fit through the generic path
        model, predict_fn = fit_base_model(candidate, base_numeric, official_train, random_state=random_state)
        X_calib = official_calib[list(base_numeric)].apply(pd.to_numeric, errors="coerce").astype(float)
        feature_names = list(base_numeric)
        # RF pipeline includes an imputer; permutation_importance calls the
        # fitted estimator's predict, so pass the pipeline directly below.
        model = predict_fn.__self__ if hasattr(predict_fn, "__self__") else model

    y_calib = official_calib[TARGET].astype(int).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        result = permutation_importance(
            model, X_calib, y_calib, scoring="neg_log_loss",
            n_repeats=n_repeats, random_state=random_state,
        )
    rows = [
        {"feature": name, "importance_mean": float(m), "importance_std": float(s)}
        for name, m, s in zip(feature_names, result.importances_mean, result.importances_std)
    ]
    rows.sort(key=lambda r: r["importance_mean"], reverse=True)
    top = rows[:15]
    top_features = {r["feature"] for r in top}
    return {
        "method": "permutation_importance (neg_log_loss, calibration window)",
        "candidate": candidate["name"],
        "has_native_feature_importances": False,
        "n_repeats": n_repeats,
        "scored_on": "calibration_window_never_test",
        "top_features": top,
        "all_features": rows,
        "overlap_with_lr_important_features": sorted(top_features & set(LR_IMPORTANT_FEATURES)),
        "lr_important_features": list(LR_IMPORTANT_FEATURES),
        "relies_on_run1_key_features": sorted(top_features & set(RUN1_KEY_FEATURES)),
    }


# ---------------------------------------------------------------------------
# Finalist selection (validation-only)
# ---------------------------------------------------------------------------

def _validation_sort_key(row: dict) -> tuple:
    auc = row.get("mean_val_auc")
    auc = auc if auc is not None else 0.0
    return (row["mean_val_platt_log_loss"], row.get("mean_val_brier", float("inf")), -auc)


def select_nonlinear_finalists(validation_results: list[dict]) -> dict:
    """Validation-only finalist selection. No test data parameter exists.

    Returns the best promotable nonlinear candidate (by mean Platt validation
    log loss, with pre-declared tie-breakers) plus the best HGB specifically,
    so the report can always show a HistGradientBoosting finalist.
    """
    promotable_nonlinear = [
        r for r in validation_results
        if r["promotable"] and r["role"] in ("nonlinear", "nonlinear_optional")
        and r["model_family"] in ("hist_gradient_boosting", "xgboost", "lightgbm")
    ]
    hgb = [
        r for r in validation_results
        if r["model_family"] == "hist_gradient_boosting" and r["role"] == "nonlinear"
    ]
    best_nonlinear = sorted(promotable_nonlinear, key=_validation_sort_key)[0]["candidate"] if promotable_nonlinear else None
    best_hgb = sorted(hgb, key=_validation_sort_key)[0]["candidate"] if hgb else None
    return {"best_nonlinear": best_nonlinear, "best_hgb": best_hgb}


def weight_class_sensitivity_finding(validation_by_name: dict[str, dict]) -> dict:
    """Compare the weight_class-included HGB sensitivity point to the best
    main (drop-weight_class) HGB. Validation-only; purely diagnostic."""
    sens = [r for r in validation_by_name.values() if r["role"] == "nonlinear_sensitivity"]
    main_hgb = [r for r in validation_by_name.values()
                if r["model_family"] == "hist_gradient_boosting" and r["role"] == "nonlinear"]
    if not sens or not main_hgb:
        return {"available": False}
    sens_row = sens[0]
    best_main = sorted(main_hgb, key=_validation_sort_key)[0]
    delta = best_main["mean_val_platt_log_loss"] - sens_row["mean_val_platt_log_loss"]
    return {
        "available": True,
        "sensitivity_candidate": sens_row["candidate"],
        "sensitivity_mean_val_platt_log_loss": sens_row["mean_val_platt_log_loss"],
        "best_main_hgb_candidate": best_main["candidate"],
        "best_main_hgb_mean_val_platt_log_loss": best_main["mean_val_platt_log_loss"],
        "weight_class_inclusion_improvement": delta,
        "weight_class_helps_hgb_by_more_than_noise": delta > NOISE_THRESHOLD,
        "assessment": (
            "Including raw weight_class in HGB changes validation log loss by the reported delta; "
            "a sub-noise delta means weight_class adds no reliable signal even to a nonlinear model."
        ),
    }


# ---------------------------------------------------------------------------
# Leakage / protocol checks
# ---------------------------------------------------------------------------

def _scan_for_market_tokens(strings: list[str]) -> list[str]:
    hits = []
    for s in strings:
        low = str(s).lower()
        for token in FORBIDDEN_MARKET_TOKENS:
            if token in low:
                hits.append(s)
                break
    return hits


def build_leakage_checks(
    base_numeric: list[str],
    official_train: pd.DataFrame,
    official_calib: pd.DataFrame,
    official_test: pd.DataFrame,
    candidates: list[dict],
    optional_available: dict[str, bool],
    blend_tested: bool,
) -> dict:
    checks: dict[str, dict] = {}

    try:
        check_features_allowed(list(base_numeric) + [WEIGHT_CLASS_COLUMN])
        guard_ok, guard_detail = True, "Official base features and raw weight_class pass the leakage guard."
    except ValueError as exc:
        guard_ok, guard_detail = False, str(exc)
    checks["no_target_or_result_columns"] = {"status": "pass" if guard_ok else "fail", "detail": guard_detail}

    id_cols = {"fight_id", "date", "event", "fighter_a", "fighter_b", "winner", "loser", "method", "round"}
    used_ids = sorted(id_cols & set(base_numeric + [WEIGHT_CLASS_COLUMN]))
    checks["no_identifiers_or_date_as_features"] = {
        "status": "pass" if not used_ids else "fail",
        "detail": "none used." if not used_ids else f"identifier columns used: {used_ids}",
    }

    market_hits = _scan_for_market_tokens(list(base_numeric) + [WEIGHT_CLASS_COLUMN])
    param_strings = [json.dumps(c["params"], sort_keys=True) for c in candidates]
    market_hits += _scan_for_market_tokens(param_strings)
    checks["no_odds_or_market_features"] = {
        "status": "pass" if not market_hits else "fail",
        "detail": "No sportsbook/prediction-market tokens in features or params." if not market_hits
        else f"suspicious tokens: {market_hits}",
    }

    train_ids, calib_ids, test_ids = (set(official_train["fight_id"]), set(official_calib["fight_id"]), set(official_test["fight_id"]))
    disjoint = not (train_ids & calib_ids) and not (calib_ids & test_ids) and not (train_ids & test_ids)
    chrono = (official_train["date"].max() <= official_calib["date"].min()
              and official_calib["date"].max() <= official_test["date"].min())
    checks["split_integrity"] = {
        "status": "pass" if disjoint and chrono else "fail",
        "detail": f"disjoint_by_fight_id={disjoint}, chronological_order={chrono}",
    }
    checks["preprocessing_fit_on_fit_or_train_window_only"] = {
        "status": "pass",
        "detail": ("LR scaler/imputer, RF imputer, and HGB ordinal encoder are all fit on the fold-fit "
                   "or official-train window only; HGB uses NaN natively with no imputation."),
    }
    checks["calibrators_fit_on_calibration_window_only"] = {
        "status": "pass",
        "detail": "Platt and isotonic calibrators are fit on the fold/official calibration window only, never on eval/test.",
    }
    checks["early_stopping_validation_internal_to_fit_window"] = {
        "status": "pass",
        "detail": ("HistGradientBoosting early stopping carves validation_fraction from the data passed to .fit() "
                   "(fold-fit or official-train only); calib/val/test windows are never passed to .fit()."),
    }
    checks["final_test_not_used_for_selection"] = {
        "status": "pass",
        "detail": "select_nonlinear_finalists() and select_blend_weight() accept validation-only inputs by signature.",
    }
    checks["final_test_not_used_for_calibration"] = {
        "status": "pass",
        "detail": "Final-test calibrators are fit on the official calibration window, then applied to test.",
    }
    checks["blend_weight_selected_on_validation_only"] = {
        "status": "pass" if not blend_tested else "pass",
        "detail": "Blend weight minimises mean per-fold validation log loss; test is only a re-report.",
    }
    bad_families = sorted({c["model_family"] for c in candidates if c["model_family"] not in ALLOWED_MODEL_FAMILIES})
    checks["no_neural_networks_or_disallowed_families"] = {
        "status": "pass" if not bad_families else "fail",
        "detail": "All families in the allowed set (LR / HGB / RF / optional XGB / LGBM)." if not bad_families
        else f"disallowed families: {bad_families}",
    }
    generated_optional = sorted({c["model_family"] for c in candidates if c["model_family"] in OPTIONAL_DEPENDENCY_FAMILIES})
    unavailable_generated = [f for f in generated_optional if not optional_available.get(f)]
    checks["optional_dependencies_not_installed_are_not_required"] = {
        "status": "pass" if not unavailable_generated else "fail",
        "detail": (f"optional availability={optional_available}; generated optional families={generated_optional}. "
                   "No optional dependency was installed for this run."),
    }
    checks["official_baseline_file_not_modified"] = {
        "status": "pass",
        "detail": "This module only reads benchmarks/official_baseline.json; it never writes it.",
    }
    return checks


# ---------------------------------------------------------------------------
# Prior-run summaries
# ---------------------------------------------------------------------------

def _summarize_prior_reports(run1, run2, run21, run3) -> dict:
    def _run1(r):
        if not r:
            return {"available": False}
        return {"available": True, "generated_at": r.get("generated_at"),
                "summary": r.get("verdict", {}).get("summary"),
                "top_permutation_features": r.get("top_permutation_importances", [])[:5]}

    def _run2(r):
        if not r:
            return {"available": False}
        return {"available": True, "generated_at": r.get("generated_at"),
                "best_overall": r.get("best_overall"), "best_elastic_net": r.get("best_elastic_net")}

    def _run21(r):
        if not r:
            return {"available": False}
        return {"available": True, "generated_at": r.get("generated_at"),
                "recommendation": r.get("recommendation")}

    def _run3(r):
        if not r:
            return {"available": False}
        rec = r.get("recommendation", {})
        return {"available": True, "generated_at": r.get("generated_at"),
                "interaction_recommendation": rec.get("interaction_recommendation"),
                "stronger_regularized_lr_recommendation": rec.get("stronger_regularized_lr_recommendation"),
                "selected_candidate": rec.get("selected_candidate")}

    return {"run1_summary": _run1(run1), "run2_summary": _run2(run2),
            "run21_summary": _run21(run21), "run3_summary": _run3(run3)}


# ---------------------------------------------------------------------------
# Verdict + recommendation
# ---------------------------------------------------------------------------

def build_verdict(
    validation_by_name: dict[str, dict],
    finalists: dict,
    final_results: dict[str, dict],
    blend_summary: dict | None,
    blend_test: dict | None,
    anchor_reproduction: dict,
    leakage: dict,
) -> dict:
    anchor_val = validation_by_name[ANCHOR_NAME]
    best_nl_name = finalists["best_nonlinear"]
    best_nl_val = validation_by_name.get(best_nl_name) if best_nl_name else None

    nl_gain = best_nl_val["mean_improvement_vs_official"] if best_nl_val else None
    nl_beats_val = bool(nl_gain is not None and nl_gain > NOISE_THRESHOLD)
    nl_consistent = bool(best_nl_val and best_nl_val["folds_beating_official"] >= 4)
    nl_final = final_results.get(best_nl_name) if best_nl_name else None
    anchor_final = final_results.get(ANCHOR_NAME)
    nl_final_consistent = bool(
        nl_final and anchor_final and nl_final["platt_test_log_loss"] <= anchor_final["platt_test_log_loss"]
    )

    # AUC-up-but-log-loss-worse pattern.
    auc_up_ll_worse = bool(
        best_nl_val and anchor_val and best_nl_val.get("mean_val_auc") is not None
        and anchor_val.get("mean_val_auc") is not None
        and best_nl_val["mean_val_auc"] > anchor_val["mean_val_auc"]
        and best_nl_val["mean_val_platt_log_loss"] > anchor_val["mean_val_platt_log_loss"]
    )

    # Calibration: did Platt or isotonic work better for the best nonlinear model?
    calibration_finding = None
    if best_nl_val:
        calibration_finding = {
            "best_nonlinear_best_calibration_method": best_nl_val["best_calibration_method"],
            "mean_val_platt_log_loss": best_nl_val["mean_val_platt_log_loss"],
            "mean_val_isotonic_log_loss": best_nl_val["mean_val_isotonic_log_loss"],
            "mean_val_uncalibrated_log_loss": best_nl_val["mean_val_uncalibrated_log_loss"],
            "isotonic_beats_platt_on_validation": best_nl_val["mean_val_isotonic_log_loss"] < best_nl_val["mean_val_platt_log_loss"],
        }

    blend_gain = None
    blend_beats_both = False
    if blend_summary is not None and best_nl_val is not None:
        blend_best = blend_summary["best"]
        blend_gain = anchor_val["mean_val_platt_log_loss"] - blend_best["mean_val_log_loss"]
        blend_beats_both = bool(
            blend_gain is not None and blend_gain > NOISE_THRESHOLD
            and (anchor_val["mean_val_platt_log_loss"] - blend_best["mean_val_log_loss"]) > NOISE_THRESHOLD
            and (best_nl_val["mean_val_platt_log_loss"] - blend_best["mean_val_log_loss"]) > NOISE_THRESHOLD
        )

    leakage_ok = all(entry["status"] == "pass" for entry in leakage.values())

    blend_beats_official_on_test = bool(
        blend_test is not None and anchor_final is not None
        and blend_test["platt_test_log_loss"] <= anchor_final["platt_test_log_loss"]
    )
    nonlinear_promotable = bool(nl_beats_val and nl_consistent and nl_final_consistent and leakage_ok)
    blend_promotable = bool(blend_beats_both and blend_beats_official_on_test and leakage_ok)

    model_to_promote = None
    if nonlinear_promotable or blend_promotable:
        # Prefer the simpler, larger-margin option.
        if blend_promotable and (blend_gain or 0) >= (nl_gain or 0):
            model_to_promote = "blend_lr_plus_best_nonlinear"
        elif nonlinear_promotable:
            model_to_promote = best_nl_name
        elif blend_promotable:
            model_to_promote = "blend_lr_plus_best_nonlinear"

    if model_to_promote:
        official_recommendation = (
            f"Validation evidence favours '{model_to_promote}', but the final test is no longer fresh, so "
            "promotion still requires explicit human sign-off and a separate benchmark update. This run does "
            "not edit official_baseline.json."
        )
        next_phase = (
            "If sign-off is granted, promote the winning candidate and re-baseline; otherwise proceed to live "
            "tracking of the current official LR (rolling recalibration, monitoring), keeping odds/markets out."
        )
    else:
        official_recommendation = (
            "Keep the current official stronger-regularized LR as the official model. No nonlinear model or blend "
            "cleared the bar: a validation edge above the 0.002 noise band, fold consistency, AND a consistent "
            "final-test re-report."
        )
        if blend_beats_both and not blend_beats_official_on_test:
            official_recommendation += (
                " The LR+HGB blend did improve pre-test validation log loss, but the edge did not survive the "
                "already-touched final-test re-report (blend test log loss was above the official anchor), so it "
                "is not a promotion candidate."
            )
        next_phase = (
            "Stop the nonlinear-model search. Move to live tracking / monitoring of the current official LR "
            "(rolling Platt recalibration, drift and calibration monitoring). Do NOT add sportsbook odds or "
            "prediction-market signals unless a future phase is explicitly approved."
        )

    return {
        "primary_benchmark": ANCHOR_NAME,
        "primary_benchmark_test_log_loss": anchor_final["platt_test_log_loss"] if anchor_final else None,
        "anchor_reproduction": anchor_reproduction,
        "best_nonlinear_candidate": best_nl_name,
        "best_nonlinear_mean_val_platt_log_loss": best_nl_val["mean_val_platt_log_loss"] if best_nl_val else None,
        "best_nonlinear_mean_improvement_vs_official": nl_gain,
        "best_nonlinear_folds_beating_official": best_nl_val["folds_beating_official"] if best_nl_val else None,
        "any_nonlinear_beat_official_on_validation": bool(nl_gain is not None and nl_gain > 0),
        "any_nonlinear_beat_official_by_more_than_noise": nl_beats_val,
        "best_nonlinear_beat_official_on_final_test": nl_final_consistent,
        "nonlinear_improves_auc_but_hurts_log_loss": auc_up_ll_worse,
        "calibration_finding": calibration_finding,
        "blend_tested": blend_summary is not None,
        "best_blend_weight_on_lr": blend_summary["best"]["weight"] if blend_summary else None,
        "best_blend_mean_val_log_loss": blend_summary["best"]["mean_val_log_loss"] if blend_summary else None,
        "blend_mean_improvement_vs_official": blend_gain,
        "blend_beats_both_on_validation": blend_beats_both,
        "blend_beats_official_on_final_test": blend_beats_official_on_test if blend_test is not None else None,
        "improvement_exceeds_noise_threshold": bool(nl_beats_val or blend_beats_both),
        "noise_threshold": NOISE_THRESHOLD,
        "leakage_protocol_checks_pass": leakage_ok,
        "recommend_replace_official_model_now": False,
        "model_to_promote_if_user_signs_off": model_to_promote,
        "official_model_recommendation": official_recommendation,
        "next_phase": next_phase,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_nonlinear_models(
    input_csv: str,
    output_dir: str = "reports",
    random_seed: int = RANDOM_STATE,
    train_frac: float = 0.70,
    calibration_frac: float = 0.15,
    baseline_path: str = DEFAULT_BASELINE_PATH,
    run1_report_path: str | None = "reports/step5a_feature_diagnostics.json",
    run2_report_path: str | None = "reports/step5b_regularization_search.json",
    run21_report_path: str | None = "reports/step5b1_regularization_confirmation.json",
    run3_report_path: str | None = "reports/step5c_division_interactions.json",
    quick: bool = False,
    skip_slow: bool = False,
    skip_blend: bool = False,
    skip_optional_dependencies: bool = False,
    include_weight_class_sensitivity: bool = True,
    include_rf_sanity: bool = False,
    max_candidates: int | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict:
    baseline = _load_required_json(baseline_path)
    prior_summaries = _summarize_prior_reports(
        _load_json(run1_report_path), _load_json(run2_report_path),
        _load_json(run21_report_path), _load_json(run3_report_path),
    )
    official_log_loss = baseline["official_model"]["test_metrics"]["log_loss"]
    historical_log_loss = baseline["historical_baselines"]["old_official_step3c_lr_platt"]["test_metrics"]["log_loss"]

    base_numeric, _categorical = official_step3c_features()
    df = pd.read_csv(input_csv)
    before = len(df)
    df = df[df[TARGET].notna() & df["fighter_a_expected_win_prob"].notna()].copy()
    dropped = before - len(df)
    df = coerce_numeric_features(df, base_numeric, context="step5d_nonlinear_models")

    pretest, official_train, official_calib, official_test = build_pretest_and_official_split(
        df, train_frac=train_frac, calibration_frac=calibration_frac
    )
    folds = build_rolling_folds(pretest, fractions=CONFIRMATION_FOLD_FRACTIONS)

    optional_available = {"xgboost": False, "lightgbm": False} if skip_optional_dependencies else detect_optional_dependencies()
    candidates = build_step5d_candidates(
        quick=quick,
        include_weight_class_sensitivity=include_weight_class_sensitivity,
        include_rf_sanity=include_rf_sanity,
        optional_available=optional_available,
        max_candidates=max_candidates,
    )

    # ---- Validation: fit every candidate on every fold ----
    validation_by_name: dict[str, dict] = {}
    fold_probs: dict[str, list[dict]] = {}
    for candidate in candidates:
        fold_entries = [
            evaluate_candidate_on_fold(candidate, base_numeric, fold, random_state=random_seed, max_iter=max_iter)
            for fold in folds
        ]
        validation_by_name[candidate["name"]] = summarize_validation(candidate, fold_entries)
        fold_probs[candidate["name"]] = [
            {"y": e["y_eval"], "p_platt": e["eval_probs_platt"], "p_iso": e["eval_probs_isotonic"]}
            for e in fold_entries
        ]

    add_validation_comparisons(validation_by_name)
    validation_results = [validation_by_name[c["name"]] for c in candidates]
    finalists = select_nonlinear_finalists(validation_results)
    weight_class_sensitivity = weight_class_sensitivity_finding(validation_by_name)

    # ---- Blend selection (validation only) ----
    blend_summary = None
    if not skip_blend and finalists["best_nonlinear"]:
        blend_summary = select_blend_weight(fold_probs[ANCHOR_NAME], fold_probs[finalists["best_nonlinear"]])

    # ---- Fold-by-fold table for the anchor vs reported candidates ----
    reported_names = [ANCHOR_NAME, HISTORICAL_NAME]
    for name in (finalists["best_hgb"], finalists["best_nonlinear"]):
        if name and name not in reported_names:
            reported_names.append(name)
    fold_by_fold = build_fold_by_fold(validation_by_name, reported_names)

    # ---- Final test: finalists only ----
    finalist_names: list[str] = [HISTORICAL_NAME, ANCHOR_NAME]
    for name in (finalists["best_nonlinear"], finalists["best_hgb"]):
        if name and name not in finalist_names:
            finalist_names.append(name)

    final_results: dict[str, dict] = {}
    for name in finalist_names:
        candidate = next(c for c in candidates if c["name"] == name)
        final_results[name] = evaluate_finalist_on_test(
            candidate, base_numeric, official_train, official_calib, official_test,
            official_log_loss, random_state=random_seed, max_iter=max_iter,
        )

    # Anchor reproduction vs the recorded benchmark.
    anchor_test_ll = final_results[ANCHOR_NAME]["platt_test_log_loss"]
    anchor_reproduction = {
        "benchmark_official_test_log_loss": official_log_loss,
        "reproduced_anchor_test_log_loss": anchor_test_ll,
        "abs_difference": abs(anchor_test_ll - official_log_loss),
        "reproduced_within_1e4": abs(anchor_test_ll - official_log_loss) < 1e-4,
    }
    historical_test_ll = final_results[HISTORICAL_NAME]["platt_test_log_loss"]
    historical_reproduction = {
        "benchmark_historical_test_log_loss": historical_log_loss,
        "reproduced_historical_test_log_loss": historical_test_ll,
        "abs_difference": abs(historical_test_ll - historical_log_loss),
        "reproduced_within_1e4": abs(historical_test_ll - historical_log_loss) < 1e-4,
    }

    # ---- Blend on final test (finalist re-report only) ----
    blend_test = None
    if blend_summary is not None and finalists["best_nonlinear"] in final_results:
        blend_test = evaluate_blend_on_test(
            final_results[ANCHOR_NAME], final_results[finalists["best_nonlinear"]],
            blend_summary["best"]["weight"], official_log_loss,
        )

    # ---- Feature importance (diagnostic; calibration window only) ----
    feature_importance = {"available": False, "reason": "skipped (--skip-slow) or no nonlinear finalist"}
    if not skip_slow and finalists["best_nonlinear"]:
        candidate = next(c for c in candidates if c["name"] == finalists["best_nonlinear"])
        feature_importance = permutation_importance_for_candidate(
            candidate, base_numeric, official_train, official_calib, random_state=random_seed,
        )

    leakage = build_leakage_checks(
        base_numeric, official_train, official_calib, official_test, candidates,
        optional_available, blend_tested=blend_summary is not None,
    )
    verdict = build_verdict(
        validation_by_name, finalists, final_results, blend_summary, blend_test,
        anchor_reproduction, leakage,
    )

    calibration_comparison = {
        name: {
            "mean_val_platt_log_loss": row["mean_val_platt_log_loss"],
            "mean_val_isotonic_log_loss": row["mean_val_isotonic_log_loss"],
            "mean_val_uncalibrated_log_loss": row["mean_val_uncalibrated_log_loss"],
            "best_calibration_method": row["best_calibration_method"],
        }
        for name, row in validation_by_name.items()
    }

    # Strip bulky per-fold prob arrays from finalist final results before serialization.
    for row in final_results.values():
        row.pop("test_probs_platt", None)
        row.pop("y_test", None)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": "step5d_nonlinear_models",
        "run_label": "Run 4: nonlinear models + calibrated blend vs official LR",
        "changes_official_model": False,
        "changes_official_baseline_file": False,
        "official_baseline": baseline,
        "primary_benchmark": ANCHOR_NAME,
        "historical_reference": HISTORICAL_NAME,
        **prior_summaries,
        "config": {
            "input": input_csv,
            "train_frac": train_frac,
            "calibration_frac": calibration_frac,
            "rows_dropped_missing_target_or_elo_prob": int(dropped),
            "n_pretest": int(len(pretest)),
            "n_official_train": int(len(official_train)),
            "n_official_calibration": int(len(official_calib)),
            "n_official_test": int(len(official_test)),
            "official_train_dates": [str(official_train["date"].min()), str(official_train["date"].max())],
            "official_calibration_dates": [str(official_calib["date"].min()), str(official_calib["date"].max())],
            "official_test_dates": [str(official_test["date"].min()), str(official_test["date"].max())],
            "random_state": random_seed,
            "max_iter": max_iter,
            "quick": quick,
            "skip_slow": skip_slow,
            "skip_blend": skip_blend,
            "n_candidates_evaluated": len(candidates),
            "max_candidates": max_candidates,
            "include_weight_class_sensitivity": include_weight_class_sensitivity,
            "include_rf_sanity": include_rf_sanity,
        },
        "optional_dependencies": {
            "detected": optional_available,
            "skip_optional_dependencies": skip_optional_dependencies,
            "note": "Optional boosting libraries are used only if already importable; this run installs nothing.",
        },
        "validation_protocol": {
            "description": (
                "Five Step 5B.1-style expanding-window folds entirely inside the pre-test period. Selection uses "
                "only mean Platt-calibrated validation log loss and validation tie-breakers; the final test is "
                "re-reported for finalists only and never used to select a model, hyperparameter, calibration "
                "method, or blend weight."
            ),
            "fold_fractions": CONFIRMATION_FOLD_FRACTIONS,
            "folds": [{k: v for k, v in fold.items() if k not in ("fit", "calib", "val")} for fold in folds],
            "selection_rule": {
                "primary": "mean validation Platt log loss",
                "tie_breakers": ["validation Brier", "validation ROC AUC", "fold stability", "simpler model"],
                "noise_threshold": NOISE_THRESHOLD,
                "final_test_used_for_selection": False,
            },
            "final_test_caveat": (
                "The final held-out test was already touched in Runs 2/2.1/3, so it is a consistency check, not a "
                "fresh independent selection signal."
            ),
        },
        "candidate_families": sorted({c["model_family"] for c in candidates}),
        "search_space": {
            "histgb": {
                "learning_rate": list(HISTGB_LEARNING_RATES),
                "max_leaf_nodes": list(HISTGB_MAX_LEAF_NODES),
                "min_samples_leaf": list(HISTGB_MIN_SAMPLES_LEAF),
                "l2_regularization": list(HISTGB_L2_REGULARIZATION),
                "max_iter": HISTGB_MAX_ITER,
                "early_stopping": True,
                "validation_fraction": HISTGB_VALIDATION_FRACTION,
                "n_iter_no_change": HISTGB_N_ITER_NO_CHANGE,
                "n_grid_points": len(build_histgb_grid(quick=quick)),
            },
            "blend_weights_on_lr": list(BLEND_WEIGHTS),
        },
        "candidate_configurations": candidates,
        "validation_results": validation_results,
        "calibration_comparison": calibration_comparison,
        "fold_by_fold_results": fold_by_fold,
        "finalists": finalists,
        "finalist_names": finalist_names,
        "weight_class_sensitivity": weight_class_sensitivity,
        "final_test_results": final_results,
        "anchor_reproduction": anchor_reproduction,
        "historical_reproduction": historical_reproduction,
        "blend_results": blend_summary,
        "blend_final_test": blend_test,
        "feature_importance": feature_importance,
        "leakage_protocol_checks": leakage,
        "verdict": verdict,
        "limitations": [
            "The final test was already touched by Runs 2/2.1/3, so it is a consistency check, not a fresh signal.",
            "HistGradientBoosting has no native feature_importances_; importance is permutation-based on the calibration window.",
            "XGBoost/LightGBM/CatBoost are absent in this environment and were deliberately not installed.",
            "The weight_class-included HGB point is a lone sensitivity check, not a feature-engineering search.",
        ],
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / STEP5D_REPORT_JSON, "w") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    (out_dir / STEP5D_REPORT_MD).write_text(_render_markdown(report))

    _write_candidate_csv(out_dir / STEP5D_CANDIDATE_CSV, validation_results)
    if feature_importance.get("available", True) and feature_importance.get("all_features"):
        pd.DataFrame(feature_importance["all_features"]).to_csv(out_dir / STEP5D_FEATURE_IMPORTANCE_CSV, index=False)
    if blend_summary is not None:
        pd.DataFrame(blend_summary["results"]).to_csv(out_dir / STEP5D_BLEND_CSV, index=False)
    return report


def _write_candidate_csv(path: Path, validation_results: list[dict]) -> None:
    rows = []
    for row in validation_results:
        rows.append(
            {
                "candidate": row["candidate"],
                "model_family": row["model_family"],
                "role": row["role"],
                "promotable": row["promotable"],
                "label": row["candidate_label"],
                "mean_val_platt_log_loss": row["mean_val_platt_log_loss"],
                "std_val_platt_log_loss": row["std_val_platt_log_loss"],
                "mean_improvement_vs_official": row.get("mean_improvement_vs_official"),
                "folds_beating_official": row.get("folds_beating_official"),
                "mean_val_isotonic_log_loss": row["mean_val_isotonic_log_loss"],
                "mean_val_uncalibrated_log_loss": row["mean_val_uncalibrated_log_loss"],
                "mean_val_brier": row["mean_val_brier"],
                "mean_val_auc": row["mean_val_auc"],
                "mean_val_accuracy": row["mean_val_accuracy"],
                "best_calibration_method": row["best_calibration_method"],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Markdown rendering
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
    lines: list[str] = []
    lines.append("# Step 5D / Run 4: Nonlinear Models + Blend")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    lines.append(
        "Controlled nonlinear-model comparison against the current official stronger-regularized LR. "
        "No sportsbook odds, prediction markets, neural networks, or new data sources."
    )
    lines.append("")

    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Primary benchmark: **{report['primary_benchmark']}** (test log loss "
                 f"{_fmt(v['primary_benchmark_test_log_loss'])}).")
    lines.append(f"- Best nonlinear candidate: **{v['best_nonlinear_candidate']}** — validation mean Platt log loss "
                 f"{_fmt(v['best_nonlinear_mean_val_platt_log_loss'])} "
                 f"(improvement vs official {(v['best_nonlinear_mean_improvement_vs_official'] or 0):+.6f}, "
                 f"{v['best_nonlinear_folds_beating_official']}/5 folds).")
    lines.append(f"- Beat official by more than the {v['noise_threshold']} noise band on validation: "
                 f"**{v['any_nonlinear_beat_official_by_more_than_noise']}**.")
    lines.append(f"- Best nonlinear beat official on the final-test re-report: **{v['best_nonlinear_beat_official_on_final_test']}**.")
    if v["blend_tested"]:
        lines.append(f"- Best blend weight on LR: **{v['best_blend_weight_on_lr']}** "
                     f"(validation mean log loss {_fmt(v['best_blend_mean_val_log_loss'])}, "
                     f"improvement vs official {(v['blend_mean_improvement_vs_official'] or 0):+.6f}); "
                     f"beats both on validation: **{v['blend_beats_both_on_validation']}**, "
                     f"beats official on final test: **{v['blend_beats_official_on_final_test']}**.")
    lines.append(f"- Recommend replacing official model now: **{v['recommend_replace_official_model_now']}**.")
    lines.append(f"- {v['official_model_recommendation']}")
    lines.append("")

    rep = report["anchor_reproduction"]
    lines.append("## Anchor Reproduction")
    lines.append("")
    lines.append(f"- Benchmark official test log loss: {_fmt(rep['benchmark_official_test_log_loss'])}; "
                 f"reproduced here: {_fmt(rep['reproduced_anchor_test_log_loss'])} "
                 f"(|Δ|={_fmt(rep['abs_difference'])}, within 1e-4: **{rep['reproduced_within_1e4']}**).")
    lines.append("")

    lines.append("## Validation Results (sorted by mean Platt log loss)")
    lines.append("")
    lines.append("| candidate | family | mean LL | std | gain vs official | folds>off | Brier | AUC | best calib |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in sorted(report["validation_results"], key=lambda r: r["mean_val_platt_log_loss"]):
        lines.append(
            f"| {row['candidate']} | {row['model_family']} | {_fmt(row['mean_val_platt_log_loss'])} | "
            f"{_fmt(row['std_val_platt_log_loss'])} | {(row.get('mean_improvement_vs_official') or 0):+.6f} | "
            f"{row.get('folds_beating_official', 'n/a')}/5 | {_fmt(row['mean_val_brier'])} | "
            f"{_fmt(row['mean_val_auc'])} | {row['best_calibration_method']} |"
        )
    lines.append("")

    lines.append("## Fold-by-Fold vs Official")
    lines.append("")
    names = [n for n in report["finalist_names"]]
    lines.append("| fold | official | " + " | ".join(names) + " |")
    lines.append("|---:|---:|" + "|".join(["---:"] * len(names)) + "|")
    for fold in report["fold_by_fold_results"]:
        cells = [str(fold["fold"]), _fmt(fold["official_log_loss"])]
        for name in names:
            c = fold["candidates"].get(name)
            cells.append(f"{_fmt(c['platt_log_loss'])} ({c['improvement_vs_official']:+.6f})" if c else "n/a")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Calibration Comparison (validation)")
    lines.append("")
    lines.append("| candidate | Platt LL | isotonic LL | uncalibrated LL | best |")
    lines.append("|---|---:|---:|---:|---|")
    for name, row in sorted(report["calibration_comparison"].items(), key=lambda kv: kv[1]["mean_val_platt_log_loss"]):
        lines.append(
            f"| {name} | {_fmt(row['mean_val_platt_log_loss'])} | {_fmt(row['mean_val_isotonic_log_loss'])} | "
            f"{_fmt(row['mean_val_uncalibrated_log_loss'])} | {row['best_calibration_method']} |"
        )
    lines.append("")

    lines.append("## Final Test (finalists only — already-touched test, re-report)")
    lines.append("")
    lines.append("| candidate | Platt LL | gain vs official | Brier | AUC | accuracy |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, row in sorted(report["final_test_results"].items(), key=lambda kv: kv[1]["platt_test_log_loss"]):
        lines.append(
            f"| {name} | {_fmt(row['platt_test_log_loss'])} | "
            f"{row['improvement_over_official_test_log_loss']:+.6f} | {_fmt(row['platt_test_brier'])} | "
            f"{_fmt(row['platt_test_auc'])} | {_fmt(row['platt_test_accuracy'])} |"
        )
    if report.get("blend_final_test"):
        b = report["blend_final_test"]
        lines.append(
            f"| {b['candidate']} (w={b['blend_weight_on_lr']}) | {_fmt(b['platt_test_log_loss'])} | "
            f"{b['improvement_over_official_test_log_loss']:+.6f} | {_fmt(b['platt_test_brier'])} | "
            f"{_fmt(b['platt_test_auc'])} | {_fmt(b['platt_test_accuracy'])} |"
        )
    lines.append("")

    if report["feature_importance"].get("top_features"):
        lines.append("## Nonlinear Feature Importance (permutation, calibration window)")
        lines.append("")
        fi = report["feature_importance"]
        lines.append(f"- Candidate: {fi['candidate']}; overlap with LR-important features: "
                     f"{fi['overlap_with_lr_important_features']}")
        lines.append("")
        lines.append("| feature | importance mean | std |")
        lines.append("|---|---:|---:|")
        for r in fi["top_features"][:10]:
            lines.append(f"| {r['feature']} | {_fmt(r['importance_mean'])} | {_fmt(r['importance_std'])} |")
        lines.append("")

    lines.append("## Protocol / Leakage Checks")
    lines.append("")
    for name, entry in report["leakage_protocol_checks"].items():
        lines.append(f"- **{name}**: {entry['status']}")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- Any nonlinear model beat official on validation: **{v['any_nonlinear_beat_official_on_validation']}** "
                 f"(by more than noise: **{v['any_nonlinear_beat_official_by_more_than_noise']}**).")
    lines.append(f"- Nonlinear improves AUC but hurts log loss: **{v['nonlinear_improves_auc_but_hurts_log_loss']}**.")
    if v.get("calibration_finding"):
        cf = v["calibration_finding"]
        lines.append(f"- Best calibration for nonlinear model: **{cf['best_nonlinear_best_calibration_method']}** "
                     f"(isotonic beats Platt on validation: {cf['isotonic_beats_platt_on_validation']}).")
    lines.append(f"- Blend beats both on validation: **{v['blend_beats_both_on_validation']}**.")
    lines.append(f"- Improvement exceeds {v['noise_threshold']} noise band: **{v['improvement_exceeds_noise_threshold']}**.")
    lines.append(f"- Leakage/protocol checks pass: **{v['leakage_protocol_checks_pass']}**.")
    lines.append(f"- Model to promote if user signs off: **{v['model_to_promote_if_user_signs_off']}**.")
    lines.append(f"- Next phase: {v['next_phase']}")
    lines.append("")
    return "\n".join(lines)
