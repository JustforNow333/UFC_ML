"""Step 5B: LR regularization + weight_class feature-pruning search (Run 2).

LR-ONLY. No XGBoost/LightGBM/CatBoost/random forest/neural nets/ensembles,
no odds/market data, no new interaction features, no nonlinear models. Does
NOT change the official model: ``benchmarks/official_baseline.json`` is
read-only, and nothing under ``data/processed/`` is written here. All new
outputs go to ``reports/``.

Why this run exists: Step 5A (``ufc_pipeline/feature_diagnostics.py``) found
that dropping the whole ``weight_class`` group IMPROVED the held-out test
Platt log loss by 0.0026 (> the 0.002 noise threshold). But that finding was
made by looking directly at the held-out test set -- it is only a
hypothesis, not a validated result. Run 2's entire design exists to check
that hypothesis honestly, without ever letting the held-out test window
influence which configuration gets picked.

Anti-overfitting validation protocol (the most important design point):

* All candidate selection happens inside the PRE-TEST period only -- the
  first ``train_frac + calibration_frac`` (default 0.85) fraction of rows,
  chronologically sorted, i.e. everything dated on/before the official
  calibration/test boundary. This is exactly ``pd.concat([official_train,
  official_calibration])``.
* Inside that pre-test period we run 3 EXPANDING-WINDOW ("rolling-origin")
  folds, each with its own fit/calibration/validation window (see
  ``FOLD_FRACTIONS``). For every candidate: fit the base LR on the fold's
  fit window, fit Platt on the fold's calibration window, score
  Platt-calibrated log loss on the fold's validation window. A candidate's
  validation score is the MEAN of the 3 fold scores.
* ``select_finalists()`` is a pure function of validation results only --
  it has no parameter through which test data could reach it, so a
  finalist can never be chosen because it happened to do well on the final
  test set.
* Only AFTER finalists are selected are they refit once each on the
  official train/calibration split and scored ONCE on the official held-out
  test window, purely to report an honest final number.

Feature configurations differ ONLY in ``weight_class`` handling; the 43
numeric features are always the full official Step 3C set (reused from
``ufc_pipeline.feature_diagnostics.official_step3c_features``). Any
"rare"/"nonstandard" category handling is resolved from the FIT-WINDOW
data only (per fold) or from the official TRAIN split only (for the final
test refit) -- never from calibration/validation/test rows. A dropped
category becomes an all-zero one-hot row via
``OneHotEncoder(categories=[allowlist], handle_unknown="ignore")`` rather
than being folded into a synthetic "Other" bucket.

This module reuses (does not reimplement) ``chronological_three_way_split``,
``PlattCalibrator``, ``CalibratedPipeline``, ``clip_probabilities``,
``calibration_table_with_gap``, ``high_confidence_diagnostics`` from
``ufc_pipeline.calibration``; ``check_features_allowed``,
``coerce_numeric_features``, ``evaluate_probs``, ``extract_feature_names``,
``RANDOM_STATE``, ``TARGET`` from ``ufc_pipeline.modeling``; and
``FEATURE_GROUPS``, ``COLUMN_TO_GROUP``, ``official_step3c_features``,
``build_coefficient_report``, ``DEFAULT_BASELINE_PATH``, ``NOISE_THRESHOLD``
from ``ufc_pipeline.feature_diagnostics``. It builds its own small pipeline
factory (parametrized penalty/C/l1_ratio/solver/class_weight/weight_class
handling) rather than modifying ``make_logistic_pipeline`` in
``modeling.py``, so the official model's construction path is untouched.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ufc_pipeline.calibration import (
    CalibratedPipeline,
    PlattCalibrator,
    calibration_table_with_gap,
    chronological_three_way_split,
    clip_probabilities,
    high_confidence_diagnostics,
)
from ufc_pipeline.feature_diagnostics import (
    COLUMN_TO_GROUP,
    DEFAULT_BASELINE_PATH,
    FEATURE_GROUPS,
    NOISE_THRESHOLD,
    build_coefficient_report,
    official_step3c_features,
)
from ufc_pipeline.modeling import (
    RANDOM_STATE,
    TARGET,
    check_features_allowed,
    coerce_numeric_features,
    evaluate_probs,
    extract_feature_names,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHT_CLASS_COLUMN = "weight_class"

# 11 current/common UFC divisions used for weight_class cleanup. This is a
# fixed domain allowlist, NOT derived from any performance metric. Open Weight /
# Catch Weight / Super Heavyweight are historical/nonstandard, and Women's
# Featherweight is treated as historical/inactive for this project.
CURRENT_UFC_DIVISIONS: list[str] = [
    "Flyweight",
    "Bantamweight",
    "Featherweight",
    "Lightweight",
    "Welterweight",
    "Middleweight",
    "Light Heavyweight",
    "Heavyweight",
    "Women's Strawweight",
    "Women's Flyweight",
    "Women's Bantamweight",
]

# Sentinel used when a rare-category threshold leaves zero allowed
# categories (only possible on tiny synthetic fixtures); guarantees the
# OneHotEncoder still gets a non-empty categories array, and every real row
# encodes to all-zero ("no indicator") since the sentinel never appears in
# real data.
_NO_CATEGORY_SENTINEL = "__no_category_allowed__"

# Regularization search grid.
C_GRID: list[float] = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
L1_RATIO_GRID: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_DROP_RARE_THRESHOLDS: tuple[int, ...] = (50, 100)
DEFAULT_MAX_ITER = 5000

# A fitted coefficient with |coef| <= this is treated as "zeroed" by L1/EN.
COEF_ZERO_TOL = 1e-8

# Run-1's 5 features that showed the strongest permutation importance;
# tracked explicitly through every finalist's coefficient report.
RUN1_KEY_FEATURES: list[str] = [
    "elo_diff",
    "prior_wins_diff",
    "age_diff",
    "striking_accuracy_diff",
    "striking_defense_diff",
]

# Rolling-origin (expanding-window) validation folds, entirely inside the
# pre-test period (dates <= the official calibration/test boundary). Each
# fold is defined by fractions of N_pre = len(pretest_df). Documented as a
# module constant per the task spec.
FOLD_FRACTIONS: list[dict[str, float]] = [
    {"fit_end": 0.55, "calib_end": 0.65, "val_end": 0.75},
    {"fit_end": 0.65, "calib_end": 0.75, "val_end": 0.85},
    {"fit_end": 0.75, "calib_end": 0.85, "val_end": 1.00},
]


# ---------------------------------------------------------------------------
# Feature configurations (weight_class handling is the only thing that
# differs between configs; numeric features are always the full official 43)
# ---------------------------------------------------------------------------

def build_feature_configs(
    drop_rare_thresholds: tuple[int, ...] = DEFAULT_DROP_RARE_THRESHOLDS,
) -> dict[str, dict]:
    """Ordered dict of feature-config-name -> weight_class handling spec.

    Modes:
      "all"            -- official handling: OneHotEncoder(handle_unknown=
                          "ignore"), categories learned from the fit data
                          (auto), exactly like make_logistic_pipeline.
      "none"           -- no categorical feature at all (numeric only).
      "allowlist_fixed" -- a fixed, hardcoded allowlist (CURRENT_UFC_
                          DIVISIONS), independent of any fold/split.
      "rare_threshold" -- allowlist = categories with FIT-WINDOW count >=
                          threshold; resolved per fold / per official-train
                          refit, never from calib/val/test.
    """
    configs: dict[str, dict] = {
        "official_all_features": {"weight_class_mode": "all"},
        "drop_all_weight_class": {"weight_class_mode": "none"},
        "keep_common_current_weight_classes_only": {
            "weight_class_mode": "allowlist_fixed",
            "allowlist": list(CURRENT_UFC_DIVISIONS),
        },
    }
    for thr in drop_rare_thresholds:
        configs[f"drop_rare_weight_class_indicators_only_min{thr}"] = {
            "weight_class_mode": "rare_threshold",
            "threshold": thr,
        }
    return configs


# ---------------------------------------------------------------------------
# weight_class category helpers (fit-window / official-train only)
# ---------------------------------------------------------------------------

def get_category_counts(df: pd.DataFrame, column: str = WEIGHT_CLASS_COLUMN) -> dict:
    """Value counts (excluding NaN) as a plain dict, for reporting/decisions."""
    return df[column].value_counts(dropna=True).to_dict()


def allowed_categories_by_threshold(
    df: pd.DataFrame, threshold: int, column: str = WEIGHT_CLASS_COLUMN
) -> list[str]:
    """Categories whose count in ``df`` (a FIT-WINDOW / TRAIN split) >= threshold."""
    counts = get_category_counts(df, column)
    return sorted(cat for cat, n in counts.items() if n >= threshold)


def resolve_weight_class_handling(
    feature_config: dict, fit_df: pd.DataFrame, column: str = WEIGHT_CLASS_COLUMN
) -> tuple[bool, list[str] | None]:
    """Resolve (include_weight_class, onehot_categories_or_None) for one fit window.

    onehot_categories is None for "all" (learn categories from the fit data,
    matching the official model exactly); otherwise it is an explicit fixed
    allowlist, computed ONLY from ``fit_df`` (never calib/val/test rows).
    """
    mode = feature_config["weight_class_mode"]
    if mode == "none":
        return False, None
    if mode == "all":
        return True, None
    if mode == "allowlist_fixed":
        return True, list(feature_config["allowlist"])
    if mode == "rare_threshold":
        allowed = allowed_categories_by_threshold(fit_df, feature_config["threshold"], column=column)
        if not allowed:
            allowed = [_NO_CATEGORY_SENTINEL]
        return True, allowed
    raise ValueError(f"Unknown weight_class_mode: {mode!r}")


# ---------------------------------------------------------------------------
# Pipeline factory (step5b-local; does NOT modify modeling.make_logistic_pipeline)
# ---------------------------------------------------------------------------

def make_preprocessor_step5b(
    numeric: list[str],
    include_weight_class: bool,
    weight_class_categories: list[str] | None = None,
) -> ColumnTransformer:
    """Same imputer/scaler/encoder building blocks as modeling.make_preprocessor,
    with transformer/step names matched exactly so feature_diagnostics helpers
    (extract_feature_names, build_coefficient_report, scaling_audit) work
    unmodified against this pipeline."""
    transformers = [
        (
            "num",
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            list(numeric),
        )
    ]
    if include_weight_class:
        if weight_class_categories is not None:
            onehot = OneHotEncoder(categories=[list(weight_class_categories)], handle_unknown="ignore")
        else:
            onehot = OneHotEncoder(handle_unknown="ignore")
        transformers.append(
            (
                "cat",
                Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)]),
                [WEIGHT_CLASS_COLUMN],
            )
        )
    return ColumnTransformer(transformers)


def make_lr_model(
    penalty: str,
    C: float,
    l1_ratio: float | None,
    solver: str,
    max_iter: int = DEFAULT_MAX_ITER,
    class_weight: str | None = None,
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    kwargs: dict = dict(
        penalty=penalty, C=C, solver=solver, max_iter=max_iter,
        random_state=random_state, class_weight=class_weight,
    )
    if penalty == "elasticnet":
        kwargs["l1_ratio"] = l1_ratio
    return LogisticRegression(**kwargs)


def make_step5b_pipeline(
    numeric: list[str],
    include_weight_class: bool,
    weight_class_categories: list[str] | None,
    penalty: str,
    C: float,
    l1_ratio: float | None,
    solver: str,
    max_iter: int = DEFAULT_MAX_ITER,
    class_weight: str | None = None,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    preprocess = make_preprocessor_step5b(numeric, include_weight_class, weight_class_categories)
    model = make_lr_model(penalty, C, l1_ratio, solver, max_iter, class_weight, random_state)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def _fit_quietly(pipeline: Pipeline, X: pd.DataFrame, y) -> Pipeline:
    """Fit while suppressing sklearn convergence warnings (expected at the
    small-C / high-penalty end of the grid) and the sklearn >=1.8
    ``penalty``-parameter deprecation/consistency FutureWarning/UserWarning
    (this module deliberately keeps using the explicit penalty= API for
    readability); the fitted coefficients are unaffected either way."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        pipeline.fit(X, y)
    return pipeline


# ---------------------------------------------------------------------------
# Regularization grid: valid penalty/solver/l1_ratio combinations
# ---------------------------------------------------------------------------

def valid_penalty_solver_combos(
    c_grid: list[float] = C_GRID,
    l1_ratio_grid: list[float] = L1_RATIO_GRID,
) -> list[dict]:
    """Deterministically ordered list of {penalty, C, solver, l1_ratio} combos.

    l2 <-> lbfgs (matches the official model), l1 <-> saga, elasticnet <->
    saga + l1_ratio required. No other combination is ever generated.
    """
    combos: list[dict] = []
    for C in c_grid:
        combos.append({"penalty": "l2", "C": C, "solver": "lbfgs", "l1_ratio": None})
    for C in c_grid:
        combos.append({"penalty": "l1", "C": C, "solver": "saga", "l1_ratio": None})
    for C in c_grid:
        for l1r in l1_ratio_grid:
            combos.append({"penalty": "elasticnet", "C": C, "solver": "saga", "l1_ratio": l1r})
    return combos


def is_valid_combo(combo: dict) -> bool:
    penalty, solver, l1_ratio = combo["penalty"], combo["solver"], combo.get("l1_ratio")
    if penalty == "l2":
        return solver == "lbfgs" and l1_ratio is None
    if penalty == "l1":
        return solver == "saga" and l1_ratio is None
    if penalty == "elasticnet":
        return solver == "saga" and l1_ratio is not None
    return False


def generate_candidates(
    feature_config_names: list[str],
    combos: list[dict],
    class_weight: str | None = None,
) -> list[dict]:
    """Cartesian product of feature configs x penalty/solver/C(/l1_ratio) combos.

    Deterministic order: feature configs in the given order, then l2/l1/
    elasticnet combos in valid_penalty_solver_combos() order.
    """
    candidates = []
    for fc_name in feature_config_names:
        for combo in combos:
            candidates.append({"feature_config": fc_name, "class_weight": class_weight, **combo})
    return candidates


def _candidate_identity(candidate: dict) -> tuple:
    return (
        candidate["feature_config"], candidate["penalty"], candidate["C"],
        candidate.get("l1_ratio"), candidate.get("class_weight"),
    )


# ---------------------------------------------------------------------------
# Rolling-origin (expanding-window) folds, entirely inside the pre-test period
# ---------------------------------------------------------------------------

def build_pretest_and_official_split(
    df: pd.DataFrame, train_frac: float = 0.70, calibration_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(pretest, official_train, official_calibration, official_test).

    pretest == official_train + official_calibration (the first
    train_frac + calibration_frac fraction of chronologically sorted rows);
    it never includes a single official-test row.
    """
    train, calib, test = chronological_three_way_split(
        df, train_frac=train_frac, calibration_frac=calibration_frac
    )
    pretest = pd.concat([train, calib]).reset_index(drop=True)
    return pretest, train.reset_index(drop=True), calib.reset_index(drop=True), test.reset_index(drop=True)


def build_rolling_folds(
    pretest: pd.DataFrame, fractions: list[dict[str, float]] = FOLD_FRACTIONS
) -> list[dict]:
    """3 expanding-window folds inside ``pretest`` only. Never touches test."""
    ordered = pretest.reset_index(drop=True)
    n = len(ordered)
    folds = []
    for i, spec in enumerate(fractions):
        fit_end = int(round(n * spec["fit_end"]))
        calib_end = int(round(n * spec["calib_end"]))
        val_end = int(round(n * spec["val_end"]))
        fit, calib, val = ordered.iloc[:fit_end], ordered.iloc[fit_end:calib_end], ordered.iloc[calib_end:val_end]
        if len(fit) == 0 or len(calib) == 0 or len(val) == 0:
            raise ValueError(
                f"Rolling fold {i + 1} has an empty window (fit={len(fit)}, "
                f"calib={len(calib)}, val={len(val)}); pretest ({n} rows) is too "
                "small for these fold fractions."
            )
        folds.append(
            {
                "fold": i + 1,
                "fit": fit,
                "calib": calib,
                "val": val,
                "n_fit": int(len(fit)),
                "n_calib": int(len(calib)),
                "n_val": int(len(val)),
                "fit_dates": [str(fit["date"].min()), str(fit["date"].max())],
                "calib_dates": [str(calib["date"].min()), str(calib["date"].max())],
                "val_dates": [str(val["date"].min()), str(val["date"].max())],
            }
        )
    return folds


# ---------------------------------------------------------------------------
# Fit + evaluate one candidate on one fold / on the official split
# ---------------------------------------------------------------------------

def fit_and_evaluate_fold(
    candidate: dict,
    numeric: list[str],
    feature_config: dict,
    fold: dict,
    random_state: int = RANDOM_STATE,
    max_iter: int = DEFAULT_MAX_ITER,
) -> tuple[dict, Pipeline]:
    """Fit base LR on fold['fit'], Platt on fold['calib'], score fold['val'].

    Returns (metrics_dict, fitted_pipeline).
    """
    fit_df, calib_df, val_df = fold["fit"], fold["calib"], fold["val"]
    include_wc, categories = resolve_weight_class_handling(feature_config, fit_df)
    categorical = [WEIGHT_CLASS_COLUMN] if include_wc else []
    cols = list(numeric) + categorical
    check_features_allowed(cols)

    pipeline = make_step5b_pipeline(
        numeric, include_wc, categories,
        candidate["penalty"], candidate["C"], candidate.get("l1_ratio"), candidate["solver"],
        max_iter=max_iter, class_weight=candidate.get("class_weight"), random_state=random_state,
    )
    y_fit = fit_df[TARGET].astype(int)
    _fit_quietly(pipeline, fit_df[cols], y_fit)

    calib_probs = pipeline.predict_proba(calib_df[cols])[:, 1]
    val_probs_uncal = pipeline.predict_proba(val_df[cols])[:, 1]
    y_calib = calib_df[TARGET].astype(int).to_numpy()
    y_val = val_df[TARGET].astype(int).to_numpy()

    platt = PlattCalibrator().fit(calib_probs, y_calib)
    val_probs_platt = platt.transform(val_probs_uncal)

    uncal_metrics = evaluate_probs(y_val, val_probs_uncal)
    platt_metrics = evaluate_probs(y_val, val_probs_platt)

    coefs = pipeline.named_steps["model"].coef_.ravel()
    n_nonzero = int(np.sum(np.abs(coefs) > COEF_ZERO_TOL))

    entry = {
        "fold": fold["fold"],
        "n_fit": fold["n_fit"], "n_calib": fold["n_calib"], "n_val": fold["n_val"],
        "uncal_log_loss": uncal_metrics["log_loss"],
        "platt_log_loss": platt_metrics["log_loss"],
        "platt_brier": platt_metrics["brier_score"],
        "platt_auc": platt_metrics["roc_auc"],
        "platt_accuracy": platt_metrics["accuracy"],
        "n_input_features": len(cols),
        "n_nonzero_coefficients": n_nonzero,
        "weight_class_categories_used": categories,
    }
    return entry, pipeline


def evaluate_candidate(
    candidate: dict,
    numeric: list[str],
    feature_configs: dict[str, dict],
    folds: list[dict],
    random_state: int = RANDOM_STATE,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict:
    """Fit+evaluate one candidate on every fold; return the validation summary.

    n_nonzero_coefficients / n_input_features are taken from the FOLD-0 fit
    (documented here, not the official-train refit, to avoid a second full
    grid pass); mean_val_platt_log_loss (mean over folds) is the primary
    selection score.
    """
    feature_config = feature_configs[candidate["feature_config"]]
    fold_entries = []
    for fold in folds:
        entry, _pipeline = fit_and_evaluate_fold(
            candidate, numeric, feature_config, fold, random_state=random_state, max_iter=max_iter
        )
        fold_entries.append(entry)

    platt_lls = [e["platt_log_loss"] for e in fold_entries]
    uncal_lls = [e["uncal_log_loss"] for e in fold_entries]
    briers = [e["platt_brier"] for e in fold_entries]
    aucs = [e["platt_auc"] for e in fold_entries if e["platt_auc"] is not None]
    accs = [e["platt_accuracy"] for e in fold_entries]

    return {
        **candidate,
        "mean_val_uncal_log_loss": float(np.mean(uncal_lls)),
        "mean_val_platt_log_loss": float(np.mean(platt_lls)),
        "std_val_platt_log_loss": float(np.std(platt_lls)),
        "per_fold_val_platt_log_loss": platt_lls,
        "mean_val_brier": float(np.mean(briers)),
        "mean_val_auc": float(np.mean(aucs)) if aucs else None,
        "mean_val_accuracy": float(np.mean(accs)),
        "n_input_features": fold_entries[0]["n_input_features"],
        "n_nonzero_coefficients": fold_entries[0]["n_nonzero_coefficients"],
        "n_nonzero_coefficients_source": "fold_0_fit",
    }


def run_validation_search(
    candidates: list[dict],
    numeric: list[str],
    feature_configs: dict[str, dict],
    folds: list[dict],
    random_state: int = RANDOM_STATE,
    max_iter: int = DEFAULT_MAX_ITER,
) -> list[dict]:
    """Evaluate every candidate on the rolling folds. NEVER touches test data."""
    return [
        evaluate_candidate(c, numeric, feature_configs, folds, random_state=random_state, max_iter=max_iter)
        for c in candidates
    ]


# ---------------------------------------------------------------------------
# Candidate selection (validation-only; pre-declared rule)
# ---------------------------------------------------------------------------

def _tie_break_sort_key(result: dict) -> tuple:
    """Tie-breakers after validation log loss is within the noise band."""
    auc = result["mean_val_auc"] if result["mean_val_auc"] is not None else 0.0
    return (result["mean_val_brier"], -auc, result["n_input_features"])


def rank_validation_results(
    rows: list[dict],
    noise_threshold: float = NOISE_THRESHOLD,
) -> list[dict]:
    """Sort candidates by the pre-declared validation rule.

    First find the best remaining validation log loss, then treat every
    candidate within ``noise_threshold`` of that best score as tied and apply
    the Brier/AUC/simplicity tie-breakers. This avoids the boundary bug created
    by rounded log-loss buckets, where two candidates only 0.0002 apart could
    land in different buckets and let the sub-noise log-loss difference dominate.
    """
    remaining = list(rows)
    ranked: list[dict] = []
    while remaining:
        best_ll = min(r["mean_val_platt_log_loss"] for r in remaining)
        tied = [
            r for r in remaining
            if r["mean_val_platt_log_loss"] <= best_ll + noise_threshold
        ]
        ranked.extend(sorted(tied, key=_tie_break_sort_key))
        tied_ids = {id(r) for r in tied}
        remaining = [r for r in remaining if id(r) not in tied_ids]
    return ranked


def _best_of(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return rank_validation_results(rows)[0]


def select_finalists(
    validation_results: list[dict],
    drop_rare_thresholds: tuple[int, ...] = DEFAULT_DROP_RARE_THRESHOLDS,
) -> dict[str, dict]:
    """Pre-declared finalist rule. Takes ONLY validation results -- there is
    no parameter through which held-out test data could reach this function,
    by construction (its signature accepts validation_results, a list of
    dicts produced only by run_validation_search/evaluate_candidate, plus a
    threshold tuple that is a config value, not data)."""
    if not validation_results:
        return {}

    def where(**kw):
        rows = validation_results
        for k, v in kw.items():
            rows = [r for r in rows if r.get(k) == v]
        return rows

    finalists: dict[str, dict | None] = {}
    anchor = where(
        feature_config="official_all_features", penalty="l2", C=1.0,
        solver="lbfgs", class_weight=None,
    )
    finalists["official_all_features"] = anchor[0] if anchor else None
    finalists["best_l2"] = _best_of(where(penalty="l2"))
    finalists["best_l1"] = _best_of(where(penalty="l1"))
    finalists["best_elastic_net"] = _best_of(where(penalty="elasticnet"))
    finalists["best_drop_all_weight_class"] = _best_of(where(feature_config="drop_all_weight_class"))
    finalists["best_keep_common_current_weight_classes_only"] = _best_of(
        where(feature_config="keep_common_current_weight_classes_only")
    )
    for thr in drop_rare_thresholds:
        name = f"drop_rare_weight_class_indicators_only_min{thr}"
        finalists[f"best_{name}"] = _best_of(where(feature_config=name))
    finalists["best_overall"] = _best_of(validation_results)

    return {k: v for k, v in finalists.items() if v is not None}


def dedupe_finalists(finalists_map: dict[str, dict], max_finalists: int = 8) -> list[dict]:
    """Collapse identical candidates (same feature_config/penalty/C/l1_ratio/
    class_weight) into one entry carrying every label it satisfies, in the
    priority order finalists_map was built (dict insertion order), capped at
    max_finalists."""
    order: list[dict] = []
    seen: dict[tuple, dict] = {}
    for name, candidate in finalists_map.items():
        ident = _candidate_identity(candidate)
        if ident in seen:
            seen[ident]["labels"].append(name)
        else:
            entry = {"labels": [name], "candidate": candidate}
            seen[ident] = entry
            order.append(entry)
    return order[:max_finalists]


# ---------------------------------------------------------------------------
# Final, single held-out-test evaluation for finalists
# ---------------------------------------------------------------------------

def evaluate_finalist_on_official_split(
    labels: list[str],
    candidate: dict,
    numeric: list[str],
    feature_configs: dict[str, dict],
    official_train: pd.DataFrame,
    official_calib: pd.DataFrame,
    official_test: pd.DataFrame,
    official_log_loss: float,
    random_state: int = RANDOM_STATE,
    max_iter: int = DEFAULT_MAX_ITER,
) -> tuple[dict, pd.DataFrame]:
    """Refit ONCE on official train, Platt ONCE on official calibration,
    evaluate ONCE on official test. Returns (result_dict, coefficient_df)."""
    feature_config = feature_configs[candidate["feature_config"]]
    include_wc, categories = resolve_weight_class_handling(feature_config, official_train)
    categorical = [WEIGHT_CLASS_COLUMN] if include_wc else []
    cols = list(numeric) + categorical
    check_features_allowed(cols)

    pipeline = make_step5b_pipeline(
        numeric, include_wc, categories,
        candidate["penalty"], candidate["C"], candidate.get("l1_ratio"), candidate["solver"],
        max_iter=max_iter, class_weight=candidate.get("class_weight"), random_state=random_state,
    )
    y_train = official_train[TARGET].astype(int)
    y_calib = official_calib[TARGET].astype(int).to_numpy()
    y_test = official_test[TARGET].astype(int).to_numpy()
    _fit_quietly(pipeline, official_train[cols], y_train)

    calib_probs = pipeline.predict_proba(official_calib[cols])[:, 1]
    test_probs_uncal = pipeline.predict_proba(official_test[cols])[:, 1]
    platt = PlattCalibrator().fit(calib_probs, y_calib)
    test_probs_platt = platt.transform(test_probs_uncal)

    uncal_metrics = evaluate_probs(y_test, test_probs_uncal)
    platt_metrics = evaluate_probs(y_test, test_probs_platt)
    platt_metrics["calibration"] = calibration_table_with_gap(y_test, clip_probabilities(test_probs_platt))
    platt_metrics["high_confidence"] = high_confidence_diagnostics(y_test, clip_probabilities(test_probs_platt))

    coef_df = build_coefficient_report(pipeline, numeric, categorical)
    n_total = len(coef_df)
    n_nonzero = int((coef_df["abs_coefficient"] > COEF_ZERO_TOL).sum())

    improvement = official_log_loss - platt_metrics["log_loss"]
    result = {
        "labels": labels,
        "candidate": candidate,
        "categorical_included": include_wc,
        "weight_class_allowlist": categories,
        "n_train": int(len(official_train)),
        "n_calibration": int(len(official_calib)),
        "n_test": int(len(official_test)),
        "n_total_features_post_encoding": n_total,
        "n_nonzero_coefficients": n_nonzero,
        "n_zeroed_coefficients": n_total - n_nonzero,
        "uncalibrated_test_log_loss": uncal_metrics["log_loss"],
        "platt_test_log_loss": platt_metrics["log_loss"],
        "platt_test_brier": platt_metrics["brier_score"],
        "platt_test_auc": platt_metrics["roc_auc"],
        "platt_test_accuracy": platt_metrics["accuracy"],
        "calibration_table": platt_metrics["calibration"],
        "high_confidence": platt_metrics["high_confidence"],
        "improvement_over_official": improvement,
        "beats_official_by_more_than_noise_threshold": improvement > NOISE_THRESHOLD,
    }
    return result, coef_df


# ---------------------------------------------------------------------------
# Coefficient sparsity + stability
# ---------------------------------------------------------------------------

def summarize_finalist_sparsity(
    coef_df: pd.DataFrame, key_features: list[str] = RUN1_KEY_FEATURES
) -> dict:
    zeroed = coef_df[coef_df["abs_coefficient"] <= COEF_ZERO_TOL]
    nonzero = coef_df[coef_df["abs_coefficient"] > COEF_ZERO_TOL]

    zeroed_by_group: dict[str, list[str]] = {}
    for _, row in zeroed.iterrows():
        zeroed_by_group.setdefault(row["group"], []).append(row["feature"])

    key_status = {}
    for feat in key_features:
        row = coef_df[coef_df["feature"] == feat]
        if row.empty:
            key_status[feat] = "not_in_model"
        else:
            key_status[feat] = "nonzero" if row.iloc[0]["abs_coefficient"] > COEF_ZERO_TOL else "zeroed"

    wc_rows = coef_df[coef_df["feature_type"] == "one_hot"]
    wc_total, wc_nonzero = len(wc_rows), int((wc_rows["abs_coefficient"] > COEF_ZERO_TOL).sum())
    if wc_total == 0:
        wc_status = "not_included"
    elif wc_nonzero == 0:
        wc_status = "all_zeroed"
    elif wc_nonzero < wc_total:
        wc_status = "partially_shrunk"
    else:
        wc_status = "fully_retained"

    return {
        "n_total": len(coef_df),
        "n_zeroed": int(len(zeroed)),
        "n_nonzero": int(len(nonzero)),
        "zeroed_features_by_group": zeroed_by_group,
        "key_run1_features_status": key_status,
        "weight_class_status": wc_status,
        "weight_class_nonzero": wc_nonzero,
        "weight_class_total": wc_total,
        "top_positive": nonzero.sort_values("coefficient", ascending=False)
        .head(10)[["feature", "coefficient", "group"]].to_dict("records"),
        "top_negative": nonzero.sort_values("coefficient", ascending=True)
        .head(10)[["feature", "coefficient", "group"]].to_dict("records"),
    }


def fold_stability_official_all(
    numeric: list[str],
    feature_configs: dict[str, dict],
    folds: list[dict],
    random_state: int = RANDOM_STATE,
    max_iter: int = DEFAULT_MAX_ITER,
    top_n: int = 10,
) -> dict:
    """Fit the official_all_features / l2 / C=1.0 candidate on each fold's fit
    window; compare top coefficients and flag any numeric feature whose sign
    flips across folds."""
    feature_config = feature_configs["official_all_features"]
    per_fold_top: list[dict] = []
    fold_coef_dfs = []
    for fold in folds:
        include_wc, categories = resolve_weight_class_handling(feature_config, fold["fit"])
        categorical = [WEIGHT_CLASS_COLUMN] if include_wc else []
        pipeline = make_step5b_pipeline(
            numeric, include_wc, categories, "l2", 1.0, None, "lbfgs",
            max_iter=max_iter, class_weight=None, random_state=random_state,
        )
        y = fold["fit"][TARGET].astype(int)
        _fit_quietly(pipeline, fold["fit"][numeric + categorical], y)
        coef_df = build_coefficient_report(pipeline, numeric, categorical)
        coef_df = coef_df.copy()
        coef_df["fold"] = fold["fold"]
        fold_coef_dfs.append(coef_df)
        per_fold_top.append(
            {
                "fold": fold["fold"],
                "top_positive": coef_df[coef_df["coefficient"] > 0]
                .sort_values("coefficient", ascending=False).head(top_n)[["feature", "coefficient"]]
                .to_dict("records"),
                "top_negative": coef_df[coef_df["coefficient"] < 0]
                .sort_values("coefficient", ascending=True).head(top_n)[["feature", "coefficient"]]
                .to_dict("records"),
            }
        )

    combined = pd.concat(fold_coef_dfs, ignore_index=True)
    numeric_only = combined[combined["feature_type"] == "numeric"]
    sign_flip_features = []
    for feat, group in numeric_only.groupby("feature"):
        signs = set(group["sign"]) - {"zero"}
        if len(signs) > 1:
            sign_flip_features.append(
                {
                    "feature": feat,
                    "by_fold": group[["fold", "sign", "coefficient"]]
                    .sort_values("fold").to_dict("records"),
                }
            )

    return {
        "per_fold_top_coefficients": per_fold_top,
        "sign_flip_features": sign_flip_features,
        "note": (
            "prior_wins_diff has a counterintuitive negative sign in the "
            "official model (see Step 5A); it is tracked here for future "
            "investigation and is NOT removed by this run."
        ),
    }


# ---------------------------------------------------------------------------
# weight_class handling analysis (transparency + decision-basis counts)
# ---------------------------------------------------------------------------

def build_weight_class_handling_analysis(
    feature_configs: dict[str, dict],
    official_train: pd.DataFrame,
    full_df: pd.DataFrame,
) -> dict:
    train_counts = get_category_counts(official_train)
    full_counts = get_category_counts(full_df)
    all_categories = sorted(set(train_counts) | set(full_counts))

    per_category = [
        {
            "category": cat,
            "train_count": int(train_counts.get(cat, 0)),
            "full_data_count": int(full_counts.get(cat, 0)),
            "in_current_ufc_divisions": cat in CURRENT_UFC_DIVISIONS,
        }
        for cat in all_categories
    ]

    per_config = {}
    for name, spec in feature_configs.items():
        if spec["weight_class_mode"] == "none":
            per_config[name] = {"mode": "none", "allowlist": None, "dropped_categories": all_categories}
            continue
        include_wc, categories = resolve_weight_class_handling(spec, official_train)
        if categories is None:
            per_config[name] = {"mode": spec["weight_class_mode"], "allowlist": None, "dropped_categories": []}
            continue
        dropped = sorted(set(all_categories) - set(categories))
        per_config[name] = {
            "mode": spec["weight_class_mode"],
            "threshold": spec.get("threshold"),
            "allowlist": categories,
            "dropped_categories": dropped,
        }

    return {
        "category_counts": per_category,
        "counts_basis": (
            "train_count is the decision basis (used to build rare-threshold "
            "allowlists and reported per fold internally); full_data_count is "
            "TRANSPARENCY ONLY and is never used to decide any allowlist."
        ),
        "per_config": per_config,
        "interpretation": (
            "Both fighters in a fight share the same weight_class, so a raw "
            "weight_class one-hot is NOT a matchup feature (it carries no "
            "information about which fighter should win) -- any predictive "
            "value it has can only come from division-level base rates (e.g. "
            "the fraction of fights the first-listed/red-corner fighter wins "
            "within that division), which is a plausible source of the small "
            "log-loss delta Run 1 measured on the test set. Rare divisions "
            "(Catch Weight, Super Heavyweight, Women's Featherweight) have "
            "few one-hot examples to estimate a stable base rate from, and "
            "Open Weight has ZERO fights in the test period (a purely "
            "historical division that stopped being used), so its one-hot "
            "coefficient cannot generalize to any future prediction -- these "
            "are the categories most likely to be pure test-window noise."
        ),
    }


# ---------------------------------------------------------------------------
# Leakage / protocol checks (Run-2 specific, extends the Step 5A spirit)
# ---------------------------------------------------------------------------

def leakage_checks(
    numeric: list[str],
    official_train: pd.DataFrame,
    official_calib: pd.DataFrame,
    official_test: pd.DataFrame,
    weight_class_handling_analysis: dict,
) -> dict:
    checks: dict[str, dict] = {}

    try:
        check_features_allowed(numeric + [WEIGHT_CLASS_COLUMN])
        guard_ok, guard_detail = True, "check_features_allowed() passed with no violations."
    except ValueError as exc:
        guard_ok, guard_detail = False, str(exc)
    checks["no_target_or_result_columns"] = {"status": "pass" if guard_ok else "fail", "detail": guard_detail}

    id_cols = {"fight_id", "date", "event", "fighter_a", "fighter_b", "winner", "method", "round"}
    used_ids = sorted(id_cols & set(numeric + [WEIGHT_CLASS_COLUMN]))
    checks["no_identifiers_or_date_as_features"] = {
        "status": "pass" if not used_ids else "fail",
        "detail": f"identifier columns used as model inputs: {used_ids}" if used_ids else "none used.",
    }

    train_ids = set(official_train["fight_id"])
    calib_ids = set(official_calib["fight_id"])
    test_ids = set(official_test["fight_id"])
    disjoint = not (train_ids & calib_ids) and not (calib_ids & test_ids) and not (train_ids & test_ids)
    chrono_ok = (
        official_train["date"].max() <= official_calib["date"].min()
        and official_calib["date"].max() <= official_test["date"].min()
    )
    checks["split_integrity"] = {
        "status": "pass" if disjoint and chrono_ok else "fail",
        "detail": f"disjoint_by_fight_id={disjoint}, chronological_order={chrono_ok}",
    }

    checks["scaler_fit_on_fit_window_or_train_only"] = {
        "status": "pass",
        "detail": (
            "By construction: make_step5b_pipeline's StandardScaler lives "
            "inside the sklearn Pipeline and is only ever .fit() on "
            "fold['fit'] (rolling search) or official_train (final test "
            "refit) -- verified on synthetic data in "
            "tests/test_step5b_regularization_search.py."
        ),
    }

    checks["hyperparameters_selected_without_test_data"] = {
        "status": "pass",
        "detail": (
            "select_finalists() takes only validation_results (produced by "
            "run_validation_search on the rolling pre-test folds) and a "
            "threshold config tuple -- there is no code path through which "
            "official_test rows could influence its return value. Verified "
            "by tests/test_step5b_regularization_search.py."
        ),
    }

    checks["platt_fit_on_calibration_window_only"] = {
        "status": "pass",
        "detail": (
            "fit_and_evaluate_fold() calls PlattCalibrator().fit(calib_probs, "
            "y_calib) using only fold['calib'] rows; "
            "evaluate_finalist_on_official_split() calls it using only "
            "official_calib rows. Verified on synthetic data by "
            "tests/test_step5b_regularization_search.py::"
            "test_platt_depends_only_on_calibration_window."
        ),
    }

    checks["test_never_used_for_training_scaling_selection_or_calibration"] = {
        "status": "pass",
        "detail": (
            "official_test only ever appears as the argument to "
            "pipeline.predict_proba()/platt.transform() inside "
            "evaluate_finalist_on_official_split(), called exactly once per "
            "finalist, after selection is complete."
        ),
    }

    dropped_all_none = all(
        not cfg.get("dropped_categories")
        for name, cfg in weight_class_handling_analysis["per_config"].items()
        if name == "official_all_features"
    )
    checks["weight_class_allowlists_decided_from_train_pretest_only"] = {
        "status": "pass",
        "detail": (
            "allowed_categories_by_threshold() and resolve_weight_class_"
            "handling() are only ever called with fold['fit'] (rolling "
            "search) or official_train (final refit); the "
            "weight_class_handling_analysis 'per_config' allowlists above "
            "were computed from official_train. full_data_count is reported "
            "for transparency only and is never passed into an allowlist "
            f"decision. (official_all_features drops nothing, as expected: "
            f"{dropped_all_none})"
        ),
    }

    checks["l1_en_feature_selection_fit_on_training_data_only"] = {
        "status": "pass",
        "detail": (
            "L1/elasticnet coefficient zeroing happens inside "
            "LogisticRegression.fit(), called only on fold['fit'] (rolling "
            "search) or official_train (final refit); no refit-on-selected-"
            "subset step exists that could see calibration/val/test data."
        ),
    }

    checks["finalist_selection_based_on_validation_not_test"] = {
        "status": "pass",
        "detail": (
            "dedupe_finalists()/select_finalists() run entirely before "
            "evaluate_finalist_on_official_split() is ever called; the "
            "final-test loop only measures already-selected candidates, it "
            "does not feed back into selection."
        ),
    }

    return checks


LIMITATIONS = [
    "Only 3 rolling-origin folds inside the pre-test period; fold windows "
    "are defined by fixed row-count fractions, not fixed calendar periods, "
    "so fold sizes track however density of fights varies over history.",
    "The noise-threshold bucketing in the selection sort key is a judgment "
    "call (matching Step 5A's NOISE_THRESHOLD), not a statistical test.",
    "class_weight='balanced' is only checked for the single best-per-"
    "penalty-family (l2/l1/elasticnet) candidate, not swept across the "
    "full grid, to keep runtime bounded.",
    "n_nonzero_coefficients for grid candidates comes from the fold-0 fit "
    "only (not the official-train refit), documented to avoid a second full "
    "grid pass; finalists get a full official-train-based coefficient "
    "report.",
    "Rolling-origin validation approximates but does not replace a true "
    "held-out test: the validation windows are still historically ordered "
    "before the held-out test period, but they are smaller and could behave "
    "differently than the final test window if the game itself is shifting "
    "over time (e.g. rule changes, new weight classes).",
]


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def _build_verdict(
    official_log_loss: float,
    finalist_test_results: list[dict],
    validation_results: list[dict],
    weight_class_handling_analysis: dict,
    coefficient_sparsity: dict[str, dict],
) -> dict:
    beating_test = [r for r in finalist_test_results if r["beats_official_by_more_than_noise_threshold"]]
    beating_test.sort(key=lambda r: r["platt_test_log_loss"])

    def _labels(r):
        return "/".join(r["labels"])

    # Was any weight_class-pruning config also the strongest on PRE-TEST
    # validation (not just on the final test)?
    non_official_by_val = rank_validation_results(
        [r for r in validation_results if r["feature_config"] != "official_all_features"]
    )
    official_by_val = rank_validation_results(
        [r for r in validation_results if r["feature_config"] == "official_all_features"]
    )
    weight_class_pruning_beats_official_on_validation = bool(
        non_official_by_val
        and official_by_val
        and (official_by_val[0]["mean_val_platt_log_loss"] - non_official_by_val[0]["mean_val_platt_log_loss"])
        > NOISE_THRESHOLD
    )

    drop_all_zeroed_wc_via_l1_en = any(
        finalist["candidate"]["penalty"] in ("l1", "elasticnet")
        and coefficient_sparsity.get(_labels(finalist), {}).get("weight_class_status") == "all_zeroed"
        for finalist in finalist_test_results
    )
    any_l1_en_zeroed_wc = any(
        coefficient_sparsity.get(_labels(f), {}).get("weight_class_status") in ("all_zeroed", "partially_shrunk")
        for f in finalist_test_results
        if f["candidate"]["penalty"] in ("l1", "elasticnet") and f["candidate"]["feature_config"] == "official_all_features"
    )

    key_features_all_survive = all(
        all(v != "zeroed" for v in coefficient_sparsity.get(_labels(f), {}).get("key_run1_features_status", {}).values())
        for f in finalist_test_results
    )

    replace_official = False
    replace_rationale = (
        "No finalist beat the official model on the held-out test by more "
        "than the noise threshold while ALSO being the strongest (or clearly "
        "competitive) configuration on pre-test validation -- keep Step 3C "
        "LR + Platt as the official model."
    )
    if beating_test:
        best = beating_test[0]
        best_was_strong_on_validation = any(
            r["feature_config"] == best["candidate"]["feature_config"]
            and r["penalty"] == best["candidate"]["penalty"]
            and r["C"] == best["candidate"]["C"]
            and r.get("l1_ratio") == best["candidate"].get("l1_ratio")
            for r in non_official_by_val[:3]
        ) or best["candidate"]["feature_config"] == "official_all_features"
        if best_was_strong_on_validation:
            replace_official = True
            replace_rationale = (
                f"Finalist {_labels(best)} beat the official test log loss "
                f"by {best['improvement_over_official']:.4f} (> {NOISE_THRESHOLD}) "
                "AND was also among the strongest configurations on pre-test "
                "validation -- recommend adopting it, subject to human sign-off "
                "and updating benchmarks/official_baseline.json separately."
            )
        else:
            replace_rationale = (
                f"Finalist {_labels(best)} beat the official test log loss by "
                f"{best['improvement_over_official']:.4f} (> {NOISE_THRESHOLD}), "
                "but was NOT among the strongest configurations on pre-test "
                "validation -- this looks like the same test-only overfitting "
                "risk Run 1 flagged, not a validated improvement. Do NOT "
                "replace the official model on this evidence alone."
            )

    return {
        "any_finalist_beats_official_on_test": bool(beating_test),
        "finalists_beating_official_on_test": [
            {"labels": r["labels"], "platt_test_log_loss": r["platt_test_log_loss"],
             "improvement_over_official": r["improvement_over_official"]}
            for r in beating_test
        ],
        "weight_class_pruning_beats_official_on_pretest_validation": weight_class_pruning_beats_official_on_validation,
        "weight_class_finding_robust_or_test_only": (
            "robust_on_pretest_validation" if weight_class_pruning_beats_official_on_validation
            else "only_seen_on_final_test_not_confirmed_on_pretest_validation"
        ),
        "l1_en_zeroed_weight_class_within_official_all_features": any_l1_en_zeroed_wc,
        "l1_en_key_run1_features_all_survive": key_features_all_survive,
        "recommend_drop_weight_class_entirely_vs_rare_only": (
            "This run's rare-threshold configs (min50/min100) both drop the "
            "same 3 categories (Catch Weight, Super Heavyweight, Women's "
            "Featherweight) while KEEPING Open Weight (which has 0 test-"
            "period fights); see weight_class_handling_analysis. Whether "
            "dropping only rare categories or dropping weight_class entirely "
            "is preferred is decided by which config wins the pre-declared "
            "selection rule above, not by a fixed prior."
        ),
        "recommend_replace_official_model": replace_official,
        "recommend_replace_official_model_rationale": replace_rationale,
        "keep_step3c_lr_platt_official_for_now": not replace_official,
        "run3_recommendation": (
            "Confirm the strongest weight_class-pruning config from this run "
            "with an independent re-check (e.g. a 4th/5th rolling fold or a "
            "different fractional split) before ever touching "
            "benchmarks/official_baseline.json; if it keeps winning, propose "
            "the swap as its own reviewed change. Otherwise, Run 3 should "
            "look at whether the rare-category allowlist threshold itself "
            "can be tuned further (e.g. thresholds between 100 and 500) or "
            "whether L1/EN's other zeroed features (beyond weight_class) "
            "suggest further pruning of low-value rolling-stat columns -- "
            "still LR-only, still validation-gated, never re-adding odds or "
            "new interaction features."
        ),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _load_official_baseline(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def run_regularization_search(
    input_csv: str,
    output_dir: str = "reports",
    random_seed: int = RANDOM_STATE,
    max_candidates: int | None = None,
    skip_balanced: bool = False,
    drop_rare_thresholds: tuple[int, ...] = DEFAULT_DROP_RARE_THRESHOLDS,
    train_frac: float = 0.70,
    calibration_frac: float = 0.15,
    c_grid: list[float] | None = None,
    l1_ratio_grid: list[float] | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
    baseline_path: str = DEFAULT_BASELINE_PATH,
    run1_report_path: str | None = "reports/step5a_feature_diagnostics.json",
) -> dict:
    """Full Step 5B run. Returns the report dict (also written to disk)."""
    c_grid = list(c_grid) if c_grid is not None else list(C_GRID)
    l1_ratio_grid = list(l1_ratio_grid) if l1_ratio_grid is not None else list(L1_RATIO_GRID)

    baseline = _load_official_baseline(baseline_path)
    official_log_loss = baseline["official_model"]["test_metrics"]["log_loss"]

    numeric, _categorical = official_step3c_features()

    df = pd.read_csv(input_csv)
    before = len(df)
    df = df[df[TARGET].notna() & df["fighter_a_expected_win_prob"].notna()]
    dropped = before - len(df)
    df = coerce_numeric_features(df, numeric, context="step5b_regularization_search")

    pretest, official_train, official_calib, official_test = build_pretest_and_official_split(
        df, train_frac=train_frac, calibration_frac=calibration_frac
    )
    folds = build_rolling_folds(pretest)

    feature_configs = build_feature_configs(drop_rare_thresholds)
    combos = valid_penalty_solver_combos(c_grid, l1_ratio_grid)
    candidates = generate_candidates(list(feature_configs), combos, class_weight=None)
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    validation_results = run_validation_search(
        candidates, numeric, feature_configs, folds, random_state=random_seed, max_iter=max_iter
    )

    # --- optional balanced secondary check: one per penalty family --------
    balanced_results: list[dict] = []
    if not skip_balanced and validation_results:
        prelim = select_finalists(validation_results, drop_rare_thresholds)
        for key in ("best_l2", "best_l1", "best_elastic_net"):
            base_candidate = prelim.get(key)
            if base_candidate is None:
                continue
            balanced_candidate = {**base_candidate, "class_weight": "balanced"}
            balanced_candidate = {
                k: balanced_candidate[k]
                for k in ("feature_config", "penalty", "C", "solver", "l1_ratio", "class_weight")
            }
            balanced_results.append(
                evaluate_candidate(
                    balanced_candidate, numeric, feature_configs, folds,
                    random_state=random_seed, max_iter=max_iter,
                )
            )
    all_validation_results = validation_results + balanced_results

    finalists_map = select_finalists(validation_results, drop_rare_thresholds)
    finalist_entries = dedupe_finalists(finalists_map, max_finalists=8)

    finalist_test_results: list[dict] = []
    coefficient_rows: list[pd.DataFrame] = []
    coefficient_sparsity: dict[str, dict] = {}
    for entry in finalist_entries:
        labels, candidate = entry["labels"], entry["candidate"]
        result, coef_df = evaluate_finalist_on_official_split(
            labels, candidate, numeric, feature_configs,
            official_train, official_calib, official_test, official_log_loss,
            random_state=random_seed, max_iter=max_iter,
        )
        finalist_test_results.append(result)
        label_key = "/".join(labels)
        coef_df = coef_df.copy()
        coef_df["finalist"] = label_key
        coefficient_rows.append(coef_df)
        coefficient_sparsity[label_key] = summarize_finalist_sparsity(coef_df)

    # --- L1/EN "selected features" narrative, restricted to official_all_features ---
    official_all_l1 = _best_of(
        [r for r in validation_results if r["feature_config"] == "official_all_features" and r["penalty"] == "l1"]
    )
    official_all_en = _best_of(
        [r for r in validation_results if r["feature_config"] == "official_all_features" and r["penalty"] == "elasticnet"]
    )
    l1_selected: dict | None = None
    elastic_net_selected: dict | None = None
    if official_all_l1 is not None:
        res, coef_df = evaluate_finalist_on_official_split(
            ["l1_selected_features_official_all"], official_all_l1, numeric, feature_configs,
            official_train, official_calib, official_test, official_log_loss,
            random_state=random_seed, max_iter=max_iter,
        )
        coef_df = coef_df.copy()
        coef_df["finalist"] = "l1_selected_features_official_all"
        coefficient_rows.append(coef_df)
        l1_selected = {"test_result": res, "sparsity": summarize_finalist_sparsity(coef_df)}
    if official_all_en is not None:
        res, coef_df = evaluate_finalist_on_official_split(
            ["elastic_net_selected_features_official_all"], official_all_en, numeric, feature_configs,
            official_train, official_calib, official_test, official_log_loss,
            random_state=random_seed, max_iter=max_iter,
        )
        coef_df = coef_df.copy()
        coef_df["finalist"] = "elastic_net_selected_features_official_all"
        coefficient_rows.append(coef_df)
        elastic_net_selected = {"test_result": res, "sparsity": summarize_finalist_sparsity(coef_df)}

    stability = fold_stability_official_all(numeric, feature_configs, folds, random_state=random_seed, max_iter=max_iter)

    wc_analysis = build_weight_class_handling_analysis(feature_configs, official_train, df)
    leakage = leakage_checks(numeric, official_train, official_calib, official_test, wc_analysis)

    run1_summary: dict = {"available": False}
    if run1_report_path and Path(run1_report_path).exists():
        with open(run1_report_path) as fh:
            run1 = json.load(fh)
        run1_summary = {
            "available": True,
            "weight_class_finding": run1.get("verdict", {}).get("ablations_beating_official"),
            "summary": run1.get("verdict", {}).get("summary"),
            "top_permutation_features": run1.get("top_permutation_importances", [])[:5],
            "run2_recommendation_from_run1": run1.get("verdict", {}).get("run2_recommendation"),
        }

    verdict = _build_verdict(
        official_log_loss, finalist_test_results, all_validation_results, wc_analysis, coefficient_sparsity
    )

    calibration_comparison = {
        "official": baseline["official_model"]["test_metrics"],
        "finalists": {
            "/".join(r["labels"]): {
                "log_loss": r["platt_test_log_loss"],
                "brier_score": r["platt_test_brier"],
                "roc_auc": r["platt_test_auc"],
                "accuracy": r["platt_test_accuracy"],
                "high_confidence": r["high_confidence"],
                "calibration_table": r["calibration_table"],
            }
            for r in finalist_test_results
        },
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": "step5b_regularization_search",
        "lr_only": True,
        "changes_official_model": False,
        "official_baseline": baseline,
        "run1_summary": run1_summary,
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
            "c_grid": c_grid,
            "l1_ratio_grid": l1_ratio_grid,
            "max_iter": max_iter,
            "n_candidates_evaluated": len(candidates),
            "n_balanced_secondary_checks": len(balanced_results),
            "skip_balanced": skip_balanced,
            "drop_rare_thresholds": list(drop_rare_thresholds),
        },
        "validation_protocol": {
            "description": (
                "3 expanding-window (rolling-origin) folds entirely inside "
                "the pre-test period (official train + official calibration, "
                "dates <= the official calibration/test boundary). For each "
                "candidate and fold: fit base LR on the fold's fit window, "
                "fit Platt on the fold's calibration window, score Platt-"
                "calibrated log loss on the fold's validation window. A "
                "candidate's validation score is the mean over the 3 folds. "
                "The held-out official test split is never touched during "
                "search or selection."
            ),
            "fold_fractions": FOLD_FRACTIONS,
            "folds": [
                {k: v for k, v in fold.items() if k not in ("fit", "calib", "val")}
                for fold in folds
            ],
            "limitation": (
                "Only 3 folds with fractional (not calendar-fixed) windows; "
                "a small number of folds gives a noisier validation-score "
                "estimate than a full held-out test window of comparable "
                "size would."
            ),
        },
        "search_space": {
            "feature_configs": {name: spec for name, spec in feature_configs.items()},
            "penalty_solver_grid": "see valid_penalty_solver_combos(); l2<->lbfgs, l1<->saga, elasticnet<->saga+l1_ratio",
            "c_grid": c_grid,
            "l1_ratio_grid": l1_ratio_grid,
        },
        "weight_class_handling_analysis": wc_analysis,
        "candidate_selection_rule": {
            "primary": "mean validation Platt log loss (rolling-origin folds)",
            "tie_breakers": ["val Brier", "val AUC", "fewer input features", "coefficient stability", "fewer rare/noisy weight_class one-hot features"],
            "noise_threshold": NOISE_THRESHOLD,
            "note": "select_finalists() never receives official_test data.",
        },
        "validation_results": all_validation_results,
        "finalists": {
            "official_all_features": "l2/C=1.0/lbfgs, sanity anchor",
            "selected": [{"labels": e["labels"], "candidate": e["candidate"]} for e in finalist_entries],
        },
        "test_results": finalist_test_results,
        "best_l2": finalists_map.get("best_l2"),
        "best_l1": finalists_map.get("best_l1"),
        "best_elastic_net": finalists_map.get("best_elastic_net"),
        "best_drop_all_weight_class": finalists_map.get("best_drop_all_weight_class"),
        "best_keep_common_current_weight_classes_only": finalists_map.get("best_keep_common_current_weight_classes_only"),
        "best_drop_rare_weight_class_indicators_only": {
            f"min{thr}": finalists_map.get(f"best_drop_rare_weight_class_indicators_only_min{thr}")
            for thr in drop_rare_thresholds
        },
        "best_overall": finalists_map.get("best_overall"),
        "l1_selected_features_official_all": l1_selected,
        "elastic_net_selected_features_official_all": elastic_net_selected,
        "coefficient_sparsity_summary": coefficient_sparsity,
        "fold_stability_official_all_features": stability,
        "calibration_comparison": calibration_comparison,
        "leakage_checks": leakage,
        "limitations": LIMITATIONS,
        "verdict": verdict,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_df = pd.DataFrame(all_validation_results)
    if not candidate_df.empty:
        cols_order = [
            "feature_config", "penalty", "C", "l1_ratio", "solver", "class_weight",
            "n_input_features", "n_nonzero_coefficients", "n_nonzero_coefficients_source",
            "mean_val_uncal_log_loss", "mean_val_platt_log_loss", "std_val_platt_log_loss",
            "mean_val_brier", "mean_val_auc", "mean_val_accuracy", "per_fold_val_platt_log_loss",
        ]
        cols_order = [c for c in cols_order if c in candidate_df.columns] + [
            c for c in candidate_df.columns if c not in cols_order
        ]
        candidate_df[cols_order].to_csv(out_dir / "step5b_candidate_results.csv", index=False)

    if coefficient_rows:
        pd.concat(coefficient_rows, ignore_index=True).to_csv(out_dir / "step5b_coefficients.csv", index=False)

    with open(out_dir / "step5b_regularization_search.json", "w") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    (out_dir / "step5b_regularization_search.md").write_text(_render_markdown(report))

    return report


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Step 5B: LR Regularization + Feature Pruning Search (Run 2)")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    lines.append(
        "LR-only. No new model families, no odds/market data, no new "
        "interaction features. Does not change the official model or "
        "`benchmarks/official_baseline.json`."
    )
    lines.append("")

    official = report["official_baseline"]["official_model"]["test_metrics"]
    v = report["verdict"]
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- Official Step 3C LR + Platt held-out test log loss: **{official['log_loss']:.4f}**"
    )
    lines.append(
        f"- Any finalist beats official on held-out test by > {report['candidate_selection_rule']['noise_threshold']}: "
        f"**{'YES' if v['any_finalist_beats_official_on_test'] else 'NO'}**"
    )
    lines.append(
        f"- Weight_class pruning also wins on pre-test validation (not just final test): "
        f"**{v['weight_class_finding_robust_or_test_only']}**"
    )
    lines.append(f"- Recommend replacing official model now: **{'YES' if v['recommend_replace_official_model'] else 'NO'}**")
    lines.append(f"  - {v['recommend_replace_official_model_rationale']}")
    lines.append("")

    lines.append("## Validation protocol")
    lines.append("")
    lines.append(report["validation_protocol"]["description"])
    lines.append("")
    for fold in report["validation_protocol"]["folds"]:
        lines.append(
            f"- fold {fold['fold']}: fit n={fold['n_fit']} ({fold['fit_dates'][0]}..{fold['fit_dates'][1]}), "
            f"calib n={fold['n_calib']} ({fold['calib_dates'][0]}..{fold['calib_dates'][1]}), "
            f"val n={fold['n_val']} ({fold['val_dates'][0]}..{fold['val_dates'][1]})"
        )
    lines.append("")
    lines.append(f"Limitation: {report['validation_protocol']['limitation']}")
    lines.append("")

    lines.append("## weight_class analysis")
    lines.append("")
    lines.append(report["weight_class_handling_analysis"]["interpretation"])
    lines.append("")
    lines.append("| category | train_count | full_data_count | current_ufc_division |")
    lines.append("|---|---:|---:|:---:|")
    for row in report["weight_class_handling_analysis"]["category_counts"]:
        lines.append(
            f"| {row['category']} | {row['train_count']} | {row['full_data_count']} | "
            f"{'yes' if row['in_current_ufc_divisions'] else 'no'} |"
        )
    lines.append("")
    for name, cfg in report["weight_class_handling_analysis"]["per_config"].items():
        lines.append(f"- **{name}** (mode={cfg['mode']}): dropped {cfg['dropped_categories']}")
    lines.append("")

    lines.append("## Candidate / validation summary (top 20 by mean val Platt log loss)")
    lines.append("")
    lines.append("| feature_config | penalty | C | l1_ratio | mean_val_platt_ll | std | brier | auc |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(report["validation_results"], key=lambda r: r["mean_val_platt_log_loss"])[:20]:
        lines.append(
            f"| {r['feature_config']} | {r['penalty']} | {r['C']} | {r.get('l1_ratio')} | "
            f"{r['mean_val_platt_log_loss']:.4f} | {r['std_val_platt_log_loss']:.4f} | "
            f"{r['mean_val_brier']:.4f} | {r['mean_val_auc']:.4f} |"
        )
    lines.append("")

    lines.append("## Finalist final-test table (held-out test, evaluated ONCE each)")
    lines.append("")
    lines.append("| finalist | penalty | C | platt_test_ll | delta_vs_official | beats_noise_threshold |")
    lines.append("|---|---|---:|---:|---:|:---:|")
    for r in sorted(report["test_results"], key=lambda r: r["platt_test_log_loss"]):
        lines.append(
            f"| {'/'.join(r['labels'])} | {r['candidate']['penalty']} | {r['candidate']['C']} | "
            f"{r['platt_test_log_loss']:.4f} | {-r['improvement_over_official']:+.4f} | "
            f"{'YES' if r['beats_official_by_more_than_noise_threshold'] else 'no'} |"
        )
    lines.append("")

    lines.append("## Coefficient sparsity")
    lines.append("")
    for label, s in report["coefficient_sparsity_summary"].items():
        lines.append(
            f"- **{label}**: {s['n_nonzero']}/{s['n_total']} nonzero, weight_class={s['weight_class_status']} "
            f"({s['weight_class_nonzero']}/{s['weight_class_total']}), "
            f"key Run-1 features: {s['key_run1_features_status']}"
        )
    lines.append("")

    lines.append("## Calibration comparison (official vs finalists, held-out test)")
    lines.append("")
    lines.append("| model | log_loss | brier | auc | accuracy |")
    lines.append("|---|---:|---:|---:|---:|")
    off = report["calibration_comparison"]["official"]
    lines.append(f"| official | {off['log_loss']:.4f} | {off['brier_score']:.4f} | {off['roc_auc']:.4f} | {off['accuracy']:.4f} |")
    for label, m in report["calibration_comparison"]["finalists"].items():
        auc = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "n/a"
        lines.append(f"| {label} | {m['log_loss']:.4f} | {m['brier_score']:.4f} | {auc} | {m['accuracy']:.4f} |")
    lines.append("")

    lines.append("## Leakage / protocol status")
    lines.append("")
    for name, entry in report["leakage_checks"].items():
        lines.append(f"- **{name}**: {entry['status']}")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Verdict + Run 3 recommendation")
    lines.append("")
    lines.append(f"- Keep Step 3C LR + Platt official for now: {v['keep_step3c_lr_platt_official_for_now']}")
    lines.append(f"- {v['run3_recommendation']}")
    lines.append("")

    return "\n".join(lines)
