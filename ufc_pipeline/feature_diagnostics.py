"""Step 5A: feature diagnostics for the current official Step 3C LR + Platt model.

This module is DIAGNOSTICS-ONLY. It never changes the official model
(``benchmarks/official_baseline.json`` is loaded as a read-only reference and
is never regenerated here), never adds odds/market features, never trains a
new model family, and never runs a hyperparameter search. It reuses the
exact same chronological three-way split, the exact same
elastic-net LR pipeline and the exact same ``PlattCalibrator`` that the
official run uses (see ``ufc_pipeline/calibration.py``), so every number
produced here is directly comparable to the current official benchmark.

What it answers:

1. Feature audit — what's a raw source column vs an engineered diff vs a
   final model input. Raw ``weight_class`` is explicitly excluded from the
   current official model.
2. Scaling audit — confirms (does not re-implement) that the pipeline's
   StandardScaler is fit on the train split only.
3. Feature groups — a deterministic, exhaustive partition of the 43 numeric
   official Step 3C inputs into interpretable groups
   (elo, physical, age, experience, recent_form, activity, striking_rolling,
   grappling_rolling, missing_flags, opponent_adjusted, matchup_style,
   matchup_style).
4. Ablations — fit/calibrate/evaluate LR+Platt on group subsets of the
   SAME chronological split used by the official run, so "does removing X
   help or hurt" is measured honestly on the held-out test window.
5. Coefficient report — standardized LR coefficients (numeric features are
   scaled inside the pipeline; one-hot columns are not).
6. Permutation importance — shuffle one raw input column at a time in the
   TEST dataframe and measure the increase in Platt-calibrated test log
   loss through the full CalibratedPipeline.
7. Leakage checks — a structured recap of the invariants the rest of the
   repo already enforces (leakage guard, chronological split, no fitting on
   test), plus what is only "enforced upstream" and not re-verified here.

Design decisions worth calling out:

* All ablations share ONE split (computed once from the full 43-feature
  coerced dataframe) so every ablation's y_train/y_calib/y_test vectors are
  row-for-row identical; only the X columns differ.
* A log-loss difference smaller than ``NOISE_THRESHOLD`` (0.002) between an
  ablation and the "all" configuration is treated as noise, not a real
  effect — this threshold is applied consistently in the helpful/harmful/
  neutral classification.
* Permutation importance measures the PLATT-CALIBRATED test log loss
  increase, not raw-model AUC/accuracy drop, so it is on the same footing
  as every other headline metric in this project.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from ufc_pipeline.calibration import (
    CalibratedPipeline,
    PlattCalibrator,
    chronological_three_way_split,
    clip_probabilities,
)
from ufc_pipeline.modeling import (
    DEFAULT_NUMERIC_FEATURES,
    RANDOM_STATE,
    STEP3B_MODEL_FEATURES,
    STEP3C_MODEL_FEATURES,
    TARGET,
    check_features_allowed,
    coerce_numeric_features,
    evaluate_probs,
    extract_feature_names,
)

DEFAULT_BASELINE_PATH = "benchmarks/official_baseline.json"

# Ablation log-loss differences smaller than this are treated as noise.
NOISE_THRESHOLD = 0.002

# Current official baseline, mirrored from benchmarks/official_baseline.json.
# Keep this local rather than importing Step 5B at module import time: Step 5B
# imports the feature-group helpers below.
OFFICIAL_LR_PARAMS = {
    "penalty": "elasticnet",
    "C": 0.003,
    "l1_ratio": 0.1,
    "solver": "saga",
    "class_weight": None,
    "max_iter": 5000,
}
OFFICIAL_MODEL_LABEL = "stronger-regularized Step 3C LR, raw weight_class dropped + Platt"

# ---------------------------------------------------------------------------
# Feature groups: an exhaustive, deterministic partition of every official
# Step 3C model input into exactly one group. Order is the report order.
# ---------------------------------------------------------------------------
FEATURE_GROUPS: dict[str, list[str]] = {
    "elo": ["elo_diff"],
    "physical": ["height_diff", "reach_diff"],
    "age": ["age_diff"],
    "experience": [
        "prior_fights_diff",
        "prior_wins_diff",
        "prior_losses_diff",
        "prior_win_pct_diff",
    ],
    "recent_form": [
        "wins_last_3_diff",
        "win_pct_last_3_diff",
        "wins_last_5_diff",
        "win_pct_last_5_diff",
    ],
    "activity": ["days_since_last_fight_diff"],
    "striking_rolling": [
        "sig_strikes_landed_per_min_diff",
        "sig_strikes_absorbed_per_min_diff",
        "sig_strike_differential_diff",
        "striking_accuracy_diff",
        "striking_defense_diff",
        "knockdown_rate_diff",
    ],
    "grappling_rolling": [
        "takedowns_per_15_diff",
        "takedown_accuracy_diff",
        "takedown_defense_diff",
        "control_time_per_15_diff",
        "submission_attempts_per_15_diff",
    ],
    "missing_flags": [
        "fighter_a_no_prior_stats",
        "fighter_b_no_prior_stats",
        "matchup_history_missing",
    ],
    "opponent_adjusted": [
        "takedowns_allowed_per_15_diff",
        "opp_takedown_attempts_per_15_diff",
        "opp_sig_str_attempted_per_min_diff",
        "control_time_absorbed_per_15_diff",
        "knockdowns_absorbed_per_15_diff",
        "submission_attempts_absorbed_per_15_diff",
    ],
    "matchup_style": [
        "striking_matchup_net_advantage",
        "striking_accuracy_matchup_net_advantage",
        "takedown_matchup_net_advantage",
        "takedown_accuracy_matchup_net_advantage",
        "control_matchup_net_advantage",
        "knockdown_matchup_net_advantage",
        "submission_matchup_net_advantage",
        "reach_volume_interaction",
        "pace_pressure_advantage",
        "opponent_pressure_absorption_advantage",
    ],
}

ALL_GROUPS: list[str] = list(FEATURE_GROUPS.keys())

# Reverse lookup: numeric model-input column name -> group name.
COLUMN_TO_GROUP: dict[str, str] = {
    col: group for group, cols in FEATURE_GROUPS.items() for col in cols
}

# Which "all_minus_<group>" ablation isolates each group's marginal value.
# (matchup_style and opponent_adjusted also have a combined ablation.)
GROUP_REMOVAL_ABLATION = {
    "elo": "all_minus_elo",
    "physical": "all_minus_physical",
    "age": "all_minus_age",
    "experience": "all_minus_experience",
    "recent_form": "all_minus_recent_form",
    "activity": "all_minus_activity",
    "striking_rolling": "all_minus_striking_rolling",
    "grappling_rolling": "all_minus_grappling_rolling",
    "matchup_style": "all_minus_matchup_style",
    "opponent_adjusted": "all_minus_opponent_adjusted",
    "missing_flags": "all_minus_missing_flags",
}


def official_step3c_features() -> tuple[list[str], list[str]]:
    """The exact current official Step 3C inputs (numeric only).

    The promoted official model intentionally drops raw ``weight_class``;
    retaining it here would silently compare diagnostics from a historical
    C=1 L2 model against the current elastic-net benchmark.
    """
    numeric = (
        list(DEFAULT_NUMERIC_FEATURES)
        + list(STEP3B_MODEL_FEATURES)
        + list(STEP3C_MODEL_FEATURES)
    )
    return numeric, []


def get_feature_groups() -> dict[str, list[str]]:
    """Deterministic copy of FEATURE_GROUPS (safe for callers to mutate)."""
    return {group: list(cols) for group, cols in FEATURE_GROUPS.items()}


def validate_feature_groups() -> None:
    """Every official feature in exactly one group; no unknown columns."""
    numeric, categorical = official_step3c_features()
    official = set(numeric) | set(categorical)

    seen: set[str] = set()
    for group, cols in FEATURE_GROUPS.items():
        for col in cols:
            if col not in official:
                raise ValueError(
                    f"feature group {group!r} references {col!r}, which is not "
                    "an official Step 3C model feature."
                )
            if col in seen:
                raise ValueError(
                    f"feature {col!r} appears in more than one feature group "
                    f"(duplicate hit while processing group {group!r})."
                )
            seen.add(col)

    missing = official - seen
    if missing:
        raise ValueError(
            "feature groups do not cover every official Step 3C feature; "
            f"missing: {sorted(missing)}"
        )


def groups_to_columns(groups: list[str]) -> tuple[list[str], list[str]]:
    """Resolve groups to current-official feature order.

    Preserving the canonical input order matters for exact reproduction of the
    official saga fit, while group membership still controls each ablation.
    """
    selected: set[str] = set()
    categorical: list[str] = []
    for group in groups:
        if group not in FEATURE_GROUPS:
            raise ValueError(
                f"Unknown feature group: {group!r}. Valid groups: "
                f"{sorted(FEATURE_GROUPS)}"
            )
        selected.update(FEATURE_GROUPS[group])
    official_numeric, _ = official_step3c_features()
    numeric = [feature for feature in official_numeric if feature in selected]
    return numeric, categorical


def _build_ablations() -> dict[str, list[str]]:
    def minus(*exclude: str) -> list[str]:
        return [g for g in ALL_GROUPS if g not in exclude]

    return {
        "all": list(ALL_GROUPS),
        "only_elo": ["elo"],
        "only_elo_plus_basic": ["elo", "physical", "age"],
        "only_basic_age_experience": ["physical", "age", "experience"],
        "only_rolling_stats": ["striking_rolling", "grappling_rolling", "missing_flags"],
        "only_striking_rolling": ["striking_rolling", "missing_flags"],
        "only_grappling_rolling": ["grappling_rolling", "missing_flags"],
        "only_matchup_style": ["matchup_style", "opponent_adjusted", "missing_flags"],
        "all_minus_elo": minus("elo"),
        "all_minus_physical": minus("physical"),
        "all_minus_age": minus("age"),
        "all_minus_experience": minus("experience"),
        "all_minus_recent_form": minus("recent_form"),
        "all_minus_activity": minus("activity"),
        "all_minus_rolling_stats": minus("striking_rolling", "grappling_rolling"),
        "all_minus_striking_rolling": minus("striking_rolling"),
        "all_minus_grappling_rolling": minus("grappling_rolling"),
        "all_minus_matchup_style": minus("matchup_style"),
        "all_minus_opponent_adjusted": minus("opponent_adjusted"),
        "all_minus_matchup_and_opponent": minus("matchup_style", "opponent_adjusted"),
        "all_minus_missing_flags": minus("missing_flags"),
    }


DEFAULT_ABLATIONS: dict[str, list[str]] = _build_ablations()


# ---------------------------------------------------------------------------
# 1. Feature audit
# ---------------------------------------------------------------------------

def feature_audit(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    """Classify every CSV column relative to the final model input set."""
    model_columns = list(numeric) + list(categorical)
    identifier_metadata = [
        c for c in
        ["fight_id", "date", "event", "fighter_a", "fighter_b", "winner", "method", "round"]
        if c in df.columns
    ]
    elo_probability_columns = [
        c for c in
        ["fighter_a_pre_elo", "fighter_b_pre_elo",
         "fighter_a_expected_win_prob", "fighter_b_expected_win_prob"]
        if c in df.columns
    ]
    raw_weight_class_excluded = ["weight_class"] if "weight_class" in df.columns else []
    excluded_other = [
        c for c in df.columns
        if c not in model_columns
        and c != TARGET
        and c not in identifier_metadata
        and c not in elo_probability_columns
        and c not in raw_weight_class_excluded
    ]
    raw_per_fighter_intermediate = sorted(
        c for c in excluded_other if c.startswith("fighter_a_") or c.startswith("fighter_b_")
    )
    other_unused = sorted(c for c in excluded_other if c not in raw_per_fighter_intermediate)

    return {
        "target_column": TARGET,
        "n_final_model_features_total": len(model_columns),
        "n_numeric": len(numeric),
        "n_categorical": len(categorical),
        "numeric_features": list(numeric),
        "categorical_features": list(categorical),
        "final_model_feature_composition": (
            "DEFAULT_NUMERIC_FEATURES (13, Step 3) + STEP3B_MODEL_FEATURES "
            "(13, rolling striking/grappling + missing flags) + "
            "STEP3C_MODEL_FEATURES (17, matchup interactions + opponent-adjusted "
            "rolling stats); raw weight_class is dropped"
        ),
        "columns_in_csv_total": int(len(df.columns)),
        "identifier_and_metadata_columns_excluded": identifier_metadata,
        "elo_and_probability_columns_excluded_from_features": elo_probability_columns,
        "raw_weight_class_excluded_from_features": raw_weight_class_excluded,
        "raw_per_fighter_intermediate_columns_excluded": raw_per_fighter_intermediate,
        "other_unused_columns": other_unused,
        "raw_vs_engineered_vs_final": {
            "raw_source": (
                "fighter_a_*/fighter_b_* per-fighter absolute values (age, height, reach, "
                "prior record, rolling per-fighter rates) -- listed in "
                "raw_per_fighter_intermediate_columns_excluded; never used directly as "
                "model inputs."
            ),
            "engineered_diff_or_flag": (
                "the *_diff columns and the three missing-history flags, computed from the "
                "raw per-fighter values -- these ARE the final model numeric features."
            ),
            "final_model": (
                "numeric_features above; identical to the current official Step 3C "
                "LR input set."
            ),
        },
    }


def make_official_step3c_pipeline(numeric: list[str]):
    """Build the current official LR pipeline without reintroducing weight_class.

    Step 5B owns the parameterized pipeline factory. Importing it lazily avoids
    a module-import cycle because that module imports this module's feature
    catalog. The returned pipeline has the same train-only imputer/scaler and
    elastic-net configuration as the benchmarked official model.
    """
    from ufc_pipeline.step5b_regularization_search import make_step5b_pipeline

    return make_step5b_pipeline(
        numeric=list(numeric),
        include_weight_class=False,
        weight_class_categories=None,
        penalty=OFFICIAL_LR_PARAMS["penalty"],
        C=OFFICIAL_LR_PARAMS["C"],
        l1_ratio=OFFICIAL_LR_PARAMS["l1_ratio"],
        solver=OFFICIAL_LR_PARAMS["solver"],
        max_iter=OFFICIAL_LR_PARAMS["max_iter"],
        class_weight=OFFICIAL_LR_PARAMS["class_weight"],
        random_state=RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# 2. Scaling audit
# ---------------------------------------------------------------------------

def scaling_audit(
    pipeline,
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    numeric: list[str],
    atol: float = 1e-8,
) -> dict:
    """Verify the pipeline's StandardScaler is fit on the TRAIN split only.

    Compares the fitted scaler's mean_/scale_ against statistics computed
    directly from the train split (post median-imputation using the SAME
    medians the imputer learned), and separately against stats computed from
    the full (train+calibration+test) dataset, to show the two would differ
    if the scaler had (incorrectly) been fit on everything.
    """
    num_pipe = pipeline.named_steps["preprocess"].named_transformers_["num"]
    imputer = num_pipe.named_steps["impute"]
    scaler = num_pipe.named_steps["scale"]
    scaler_present = type(scaler).__name__ == "StandardScaler"

    medians = dict(zip(numeric, imputer.statistics_))
    train_imputed = train_df[numeric].fillna(value=medians)
    expected_train_mean = train_imputed.mean(axis=0).to_numpy()
    expected_train_std = train_imputed.std(axis=0, ddof=0).to_numpy()
    # Mirror sklearn's StandardScaler: zero-variance columns get scale_ = 1.0
    # (not 0.0), so a constant-in-train column must not falsely look "not
    # fit on train only" just because our reference std used the raw 0.0.
    expected_train_std = np.where(expected_train_std == 0.0, 1.0, expected_train_std)

    fit_on_train_only = bool(
        np.allclose(scaler.mean_, expected_train_mean, atol=atol)
        and np.allclose(scaler.scale_, expected_train_std, atol=atol)
    )

    full_medians = full_df[numeric].median()
    full_imputed = full_df[numeric].fillna(value=full_medians)
    full_mean = full_imputed.mean(axis=0).to_numpy()
    matches_full_dataset_instead = bool(np.allclose(scaler.mean_, full_mean, atol=atol))

    return {
        "scaler_present": scaler_present,
        "fit_on_train_only": fit_on_train_only,
        "had_to_fix": False,
        "matches_full_dataset_stats_instead": matches_full_dataset_instead,
        "max_abs_mean_diff_from_train_only_stats": float(
            np.max(np.abs(scaler.mean_ - expected_train_mean))
        ),
        "max_abs_scale_diff_from_train_only_stats": float(
            np.max(np.abs(scaler.scale_ - expected_train_std))
        ),
    }


# ---------------------------------------------------------------------------
# 4. Ablations
# ---------------------------------------------------------------------------

def run_single_ablation(
    name: str,
    groups: list[str],
    train: pd.DataFrame,
    calib: pd.DataFrame,
    test: pd.DataFrame,
    y_train: np.ndarray,
    y_calib: np.ndarray,
    y_test: np.ndarray,
    official_log_loss: float,
) -> tuple[dict, dict]:
    """Fit base LR on train, Platt on calibration, evaluate both on test.

    Returns (metrics_entry, artifacts) where artifacts carries the fitted
    pipeline/calibrator/column lists so callers (e.g. the coefficient report
    and permutation importance step) can reuse the "all" ablation's model
    without refitting it.
    """
    numeric, categorical = groups_to_columns(groups)
    if not numeric and not categorical:
        raise ValueError(f"Ablation {name!r} resolves to zero columns.")
    cols = numeric + categorical
    check_features_allowed(numeric + categorical)  # leakage guard before every fit

    pipeline = make_official_step3c_pipeline(numeric)
    pipeline.fit(train[cols], y_train)
    calib_probs = pipeline.predict_proba(calib[cols])[:, 1]
    test_probs = pipeline.predict_proba(test[cols])[:, 1]

    platt = PlattCalibrator().fit(calib_probs, y_calib)
    platt_test_probs = platt.transform(test_probs)

    uncal_metrics = evaluate_probs(y_test, test_probs)
    platt_metrics = evaluate_probs(y_test, platt_test_probs)

    entry = {
        "name": name,
        "groups": list(groups),
        "n_features": len(cols),
        "n_numeric": len(numeric),
        "n_categorical": len(categorical),
        "n_train": int(len(train)),
        "n_calibration": int(len(calib)),
        "n_test": int(len(test)),
        "uncalibrated_log_loss": uncal_metrics["log_loss"],
        "platt_log_loss": platt_metrics["log_loss"],
        "platt_brier_score": platt_metrics["brier_score"],
        "roc_auc": platt_metrics["roc_auc"],
        "accuracy": platt_metrics["accuracy"],
        "delta_vs_official": platt_metrics["log_loss"] - official_log_loss,
        "calibration_platt": platt_metrics["calibration"],
    }
    artifacts = {
        "pipeline": pipeline,
        "platt": platt,
        "numeric": numeric,
        "categorical": categorical,
        "official_model_params": dict(OFFICIAL_LR_PARAMS),
    }
    return entry, artifacts


def run_ablation_suite(
    train: pd.DataFrame,
    calib: pd.DataFrame,
    test: pd.DataFrame,
    ablations: dict[str, list[str]],
    official_log_loss: float,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Run every ablation on the SAME split; return (results, artifacts)."""
    y_train = train[TARGET].astype(int).to_numpy()
    y_calib = calib[TARGET].astype(int).to_numpy()
    y_test = test[TARGET].astype(int).to_numpy()

    results: dict[str, dict] = {}
    artifacts: dict[str, dict] = {}
    for name, groups in ablations.items():
        entry, art = run_single_ablation(
            name, groups, train, calib, test, y_train, y_calib, y_test, official_log_loss
        )
        results[name] = entry
        artifacts[name] = art
    return results, artifacts


def classify_group_effects(
    ablation_results: dict[str, dict],
    noise_threshold: float = NOISE_THRESHOLD,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Classify each group as helpful/harmful/neutral from its removal ablation.

    delta = removal_platt_log_loss - all_platt_log_loss.
      delta >  +noise_threshold -> removing hurt  -> group "appears helpful"
      delta <  -noise_threshold -> removing helped -> group "appears harmful"
      otherwise                 -> "neutral/noise"
    """
    if "all" not in ablation_results:
        return [], [], []
    all_ll = ablation_results["all"]["platt_log_loss"]

    helpful, harmful, neutral = [], [], []
    for group, ablation_name in GROUP_REMOVAL_ABLATION.items():
        if ablation_name not in ablation_results:
            continue
        removal_ll = ablation_results[ablation_name]["platt_log_loss"]
        delta = removal_ll - all_ll
        record = {
            "group": group,
            "removal_ablation": ablation_name,
            "platt_log_loss_if_removed": removal_ll,
            "delta_log_loss_if_removed": delta,
        }
        if delta > noise_threshold:
            helpful.append(record)
        elif delta < -noise_threshold:
            harmful.append(record)
        else:
            neutral.append(record)
    helpful.sort(key=lambda r: r["delta_log_loss_if_removed"], reverse=True)
    harmful.sort(key=lambda r: r["delta_log_loss_if_removed"])
    neutral.sort(key=lambda r: abs(r["delta_log_loss_if_removed"]))
    return helpful, harmful, neutral


# ---------------------------------------------------------------------------
# 5. Coefficient report
# ---------------------------------------------------------------------------

def build_coefficient_report(pipeline, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Post-transform feature names/coefficients + imputer/scaler stats.

    Numeric coefficients are STANDARDIZED (StandardScaler runs inside the
    pipeline); one-hot columns are not standardized. feature_type marks the
    difference so the two are never compared as if they were on the same
    scale.
    """
    names = extract_feature_names(pipeline)
    coefs = pipeline.named_steps["model"].coef_.ravel()

    medians, means, stds = {}, {}, {}
    if numeric:
        num_pipe = pipeline.named_steps["preprocess"].named_transformers_["num"]
        medians = dict(zip(numeric, num_pipe.named_steps["impute"].statistics_))
        means = dict(zip(numeric, num_pipe.named_steps["scale"].mean_))
        stds = dict(zip(numeric, num_pipe.named_steps["scale"].scale_))
    numeric_set = set(numeric)

    rows = []
    for feat, coef in zip(names, coefs):
        if feat in numeric_set:
            group = COLUMN_TO_GROUP.get(feat, "unknown")
            feature_type = "numeric"
            imputer_median = float(medians[feat])
            scaler_mean = float(means[feat])
            scaler_std = float(stds[feat])
        else:
            # One-hot expanded categorical column, e.g. "weight_class_Lightweight".
            group = "weight_class"
            feature_type = "one_hot"
            imputer_median = scaler_mean = scaler_std = None
        rows.append(
            {
                "feature": feat,
                "group": group,
                "feature_type": feature_type,
                "coefficient": float(coef),
                "abs_coefficient": float(abs(coef)),
                "sign": "positive" if coef > 0 else ("negative" if coef < 0 else "zero"),
                "imputer_median": imputer_median,
                "scaler_mean": scaler_mean,
                "scaler_std": scaler_std,
            }
        )
    coef_df = (
        pd.DataFrame(rows)
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )
    coef_df["rank"] = np.arange(1, len(coef_df) + 1)
    return coef_df


# ---------------------------------------------------------------------------
# 6. Permutation importance
# ---------------------------------------------------------------------------

def permutation_importance_report(
    calibrated_pipeline: CalibratedPipeline,
    test_df: pd.DataFrame,
    columns: list[str],
    y_test: np.ndarray,
    n_repeats: int = 10,
    seed: int = RANDOM_STATE,
    top_n: int | None = None,
) -> tuple[list[dict], float]:
    """Permute one raw input column at a time; measure Platt test log-loss increase.

    importance = mean over repeats of (log_loss_permuted - log_loss_baseline).
    Runs through the FULL CalibratedPipeline (base LR pipeline + Platt), so
    it's on the same footing as every other test-window metric reported here.
    """
    base_cols = test_df[columns].reset_index(drop=True)
    y_test = np.asarray(y_test, dtype=int)

    baseline_probs = calibrated_pipeline.predict_proba(base_cols)[:, 1]
    baseline_ll = log_loss(y_test, clip_probabilities(baseline_probs), labels=[0, 1])

    rng = np.random.default_rng(seed)
    rows = []
    for col in columns:
        diffs = []
        for _ in range(n_repeats):
            permuted = base_cols.copy()
            permuted[col] = rng.permutation(permuted[col].to_numpy())
            probs = calibrated_pipeline.predict_proba(permuted)[:, 1]
            ll = log_loss(y_test, clip_probabilities(probs), labels=[0, 1])
            diffs.append(ll - baseline_ll)
        rows.append(
            {
                "feature": col,
                "group": COLUMN_TO_GROUP.get(col, "unknown"),
                "mean_importance": float(np.mean(diffs)),
                "std": float(np.std(diffs)),
                "n_repeats": int(n_repeats),
                "scoring": "platt_calibrated_test_log_loss_increase",
            }
        )
    rows.sort(key=lambda r: r["mean_importance"], reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    if top_n is not None:
        rows = rows[:top_n]
    return rows, float(baseline_ll)


# ---------------------------------------------------------------------------
# 7. Leakage checks
# ---------------------------------------------------------------------------

def leakage_checks(
    train: pd.DataFrame,
    calib: pd.DataFrame,
    test: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    scaling_audit_result: dict,
) -> dict:
    """Structured pass/fail recap of the project's leakage invariants."""
    checks: dict[str, dict] = {}
    features = list(numeric) + list(categorical)

    try:
        check_features_allowed(features)
        guard_detail = "check_features_allowed() passed with no violations."
        guard_ok = True
    except ValueError as exc:
        guard_detail = str(exc)
        guard_ok = False
    no_target_or_result = guard_ok and TARGET not in features and "winner" not in features
    checks["no_target_or_result_columns"] = {
        "status": "pass" if no_target_or_result else "fail",
        "detail": guard_detail,
    }

    id_cols = {"fight_id", "date", "event", "fighter_a", "fighter_b", "winner", "method", "round"}
    used_ids = sorted(id_cols & set(features))
    checks["no_identifiers_as_features"] = {
        "status": "pass" if not used_ids else "fail",
        "detail": (
            f"identifier columns used as model inputs: {used_ids}"
            if used_ids
            else "none of fight_id/date/event/fighter_a/fighter_b/winner/method/round "
            "are model inputs."
        ),
    }

    checks["raw_weight_class_dropped"] = {
        "status": "pass" if "weight_class" not in features else "fail",
        "detail": "raw weight_class is excluded from the current official model inputs.",
    }

    train_ids, calib_ids, test_ids = set(train["fight_id"]), set(calib["fight_id"]), set(test["fight_id"])
    disjoint = not (train_ids & calib_ids) and not (calib_ids & test_ids) and not (train_ids & test_ids)
    chrono_ok = (train["date"].max() <= calib["date"].min()) and (calib["date"].max() <= test["date"].min())
    checks["split_integrity"] = {
        "status": "pass" if disjoint and chrono_ok else "fail",
        "detail": (
            f"disjoint_by_fight_id={disjoint}, chronological_order={chrono_ok}, "
            f"train_max_date={train['date'].max()}, calib_min_date={calib['date'].min()}, "
            f"calib_max_date={calib['date'].max()}, test_min_date={test['date'].min()}"
        ),
    }

    checks["scaler_fit_on_train_only"] = {
        "status": "pass" if scaling_audit_result.get("fit_on_train_only") else "fail",
        "detail": scaling_audit_result,
    }

    checks["platt_fit_on_calibration_only"] = {
        "status": "pass",
        "detail": (
            "By construction: run_single_ablation() calls "
            "PlattCalibrator().fit(calib_probs, y_calib) using only calibration-window "
            "rows. Verified on synthetic data by "
            "tests/test_feature_diagnostics.py::test_platt_depends_only_on_calibration_rows."
        ),
    }

    checks["test_set_never_used_for_fitting"] = {
        "status": "pass",
        "detail": (
            "By construction: the base pipeline's .fit() is only ever called with "
            "train[cols]/y_train; test rows are only ever passed to .predict_proba()."
        ),
    }

    checks["rolling_feature_time_safety"] = {
        "status": "enforced_upstream",
        "detail": (
            "Time-safety of the rolling striking/grappling/matchup aggregates "
            "(record-before-update discipline) is enforced and tested in "
            "ufc_pipeline/stats_features.py and ufc_pipeline/matchup_features.py "
            "(see tests/test_stats_features.py, tests/test_matchup_features.py). "
            "Not independently re-verified in this diagnostics run — listed as a "
            "limitation."
        ),
    }

    checks["fighter_orientation_consistent"] = {
        "status": "documented_invariant",
        "detail": (
            "TARGET is fighter_a_won and every *_diff / matchup feature is defined "
            "as fighter_a_value - fighter_b_value (positive = fighter A advantage), "
            "per CLAUDE.md and the matchup_features.py docstring. Not independently "
            "re-derived here; treated as a stated invariant."
        ),
    }

    return checks


LIMITATIONS = [
    "Rolling-feature time-safety (striking/grappling/matchup aggregates) is enforced "
    "and tested upstream in stats_features.py/matchup_features.py, not re-verified here.",
    "Permutation importance uses independent single-column shuffles; it does not "
    "capture importance that only shows up via feature interactions (e.g. two "
    "correlated columns can each look unimportant while jointly informative).",
    "Ablation groups are fit independently; group interactions are not tested "
    "pairwise beyond the fixed ablation list.",
    "Log-loss differences below the 0.002 noise threshold are not claimed as real "
    "effects, but the threshold itself is a judgment call, not a statistical test.",
]


# ---------------------------------------------------------------------------
# 8. Orchestration
# ---------------------------------------------------------------------------

def _load_official_baseline(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _select_weak_or_noisy_features(
    coef_df: pd.DataFrame,
    permutation_rows: list[dict],
    coef_threshold_quantile: float = 0.25,
    importance_abs_threshold: float = 0.0005,
) -> list[dict]:
    """Near-zero standardized coefficient AND near-zero permutation importance."""
    if coef_df.empty or not permutation_rows:
        return []
    numeric_coefs = coef_df[coef_df["feature_type"] == "numeric"]
    if numeric_coefs.empty:
        return []
    coef_cutoff = numeric_coefs["abs_coefficient"].quantile(coef_threshold_quantile)
    weak_coef_features = set(
        numeric_coefs.loc[numeric_coefs["abs_coefficient"] <= coef_cutoff, "feature"]
    )
    perm_by_feature = {r["feature"]: r for r in permutation_rows}
    weak = []
    for feat in weak_coef_features:
        perm = perm_by_feature.get(feat)
        if perm is None:
            continue
        if abs(perm["mean_importance"]) <= importance_abs_threshold:
            row = coef_df[coef_df["feature"] == feat].iloc[0]
            weak.append(
                {
                    "feature": feat,
                    "group": row["group"],
                    "abs_coefficient": float(row["abs_coefficient"]),
                    "mean_permutation_importance": perm["mean_importance"],
                }
            )
    weak.sort(key=lambda r: (r["abs_coefficient"], abs(r["mean_permutation_importance"])))
    return weak


def run_feature_diagnostics(
    input_csv: str,
    output_dir: str = "reports",
    n_permutation_repeats: int = 10,
    train_frac: float = 0.70,
    calibration_frac: float = 0.15,
    ablations: dict[str, list[str]] | None = None,
    permutation_top_n: int | None = None,
    skip_permutation: bool = False,
    baseline_path: str = DEFAULT_BASELINE_PATH,
) -> dict:
    """Full Step 5A diagnostics run. Returns the report dict (also written to disk).

    Reuses chronological_three_way_split / the official Step 5B elastic-net
    pipeline / PlattCalibrator / check_features_allowed / coerce_numeric_features from
    the existing modeling/calibration modules; adds no new model families and
    changes nothing under data/processed/ or benchmarks/.
    """
    validate_feature_groups()
    ablations = dict(ablations) if ablations is not None else dict(DEFAULT_ABLATIONS)

    baseline = _load_official_baseline(baseline_path)
    official_log_loss = baseline["official_model"]["test_metrics"]["log_loss"]

    numeric_all, categorical_all = official_step3c_features()

    df = pd.read_csv(input_csv)
    before = len(df)
    df = df[df[TARGET].notna() & df["fighter_a_expected_win_prob"].notna()]
    dropped = before - len(df)
    df = coerce_numeric_features(df, numeric_all, context="step5a_feature_diagnostics")

    train, calib, test = chronological_three_way_split(
        df, train_frac=train_frac, calibration_frac=calibration_frac
    )

    audit = feature_audit(df, numeric_all, categorical_all)

    ablation_results, artifacts = run_ablation_suite(train, calib, test, ablations, official_log_loss)

    all_artifacts = artifacts.get("all")
    scaling_result = None
    coef_df = pd.DataFrame()
    permutation_rows: list[dict] = []
    permutation_baseline_ll = None
    if all_artifacts is not None:
        all_pipeline = all_artifacts["pipeline"]
        audit["n_features_after_onehot_expansion"] = len(extract_feature_names(all_pipeline))

        scaling_result = scaling_audit(
            all_pipeline, train, pd.concat([train, calib, test]), all_artifacts["numeric"]
        )

        coef_df = build_coefficient_report(
            all_pipeline, all_artifacts["numeric"], all_artifacts["categorical"]
        )

        if not skip_permutation:
            calibrated = CalibratedPipeline(all_pipeline, all_artifacts["platt"])
            cols = all_artifacts["numeric"] + all_artifacts["categorical"]
            y_test = test[TARGET].astype(int).to_numpy()
            permutation_rows, permutation_baseline_ll = permutation_importance_report(
                calibrated, test, cols, y_test,
                n_repeats=n_permutation_repeats, top_n=permutation_top_n,
            )
    else:
        # "all" wasn't part of a reduced ablation set (e.g. in a unit test) —
        # scaling audit / coefficients / permutation importance are skipped.
        pass

    leakage = leakage_checks(
        train, calib, test,
        numeric_all, categorical_all,
        scaling_result or {"fit_on_train_only": None, "scaler_present": None},
    )

    groups_helpful, groups_harmful, groups_neutral = classify_group_effects(ablation_results)

    sorted_ablations = sorted(ablation_results.values(), key=lambda r: r["platt_log_loss"])
    best_ablation = sorted_ablations[0] if sorted_ablations else None
    worst_ablation = sorted_ablations[-1] if sorted_ablations else None

    top_positive = []
    top_negative = []
    if not coef_df.empty:
        positive = coef_df[coef_df["coefficient"] > 0].sort_values("coefficient", ascending=False)
        negative = coef_df[coef_df["coefficient"] < 0].sort_values("coefficient", ascending=True)
        top_positive = positive.head(15).to_dict("records")
        top_negative = negative.head(15).to_dict("records")

    top_permutation = permutation_rows[:15]
    weak_or_noisy = _select_weak_or_noisy_features(coef_df, permutation_rows)

    verdict = _build_verdict(
        ablation_results, official_log_loss, groups_helpful, groups_harmful,
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "diagnostics_only": True,
        "official_baseline": baseline,
        "split": {
            "mode": "fraction",
            "train_frac": train_frac,
            "calibration_frac": calibration_frac,
            "n_train": int(len(train)),
            "n_calibration": int(len(calib)),
            "n_test": int(len(test)),
            "train_dates": [str(train["date"].min()), str(train["date"].max())],
            "calibration_dates": [str(calib["date"].min()), str(calib["date"].max())],
            "test_dates": [str(test["date"].min()), str(test["date"].max())],
            "rows_dropped_missing_target_or_elo_prob": int(dropped),
            "random_state": RANDOM_STATE,
        },
        "feature_audit": audit,
        "scaling_audit": scaling_result,
        "feature_groups": {
            "mapping": get_feature_groups(),
            "n_groups": len(FEATURE_GROUPS),
            "n_columns_covered": sum(len(v) for v in FEATURE_GROUPS.values()),
        },
        "ablations": sorted_ablations,
        "best_ablation": best_ablation,
        "worst_ablation": worst_ablation,
        "noise_threshold_log_loss": NOISE_THRESHOLD,
        "groups_helpful": groups_helpful,
        "groups_harmful": groups_harmful,
        "groups_neutral": groups_neutral,
        "top_positive_coefficients": top_positive,
        "top_negative_coefficients": top_negative,
        "permutation_importance": {
            "baseline_platt_test_log_loss": permutation_baseline_ll,
            "n_repeats": n_permutation_repeats,
            "skipped": skip_permutation,
        },
        "top_permutation_importances": top_permutation,
        "weak_or_noisy_features": weak_or_noisy,
        "leakage_checks": leakage,
        "limitations": LIMITATIONS,
        "verdict": verdict,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not coef_df.empty:
        coef_df.to_csv(out_dir / "step5a_coefficients.csv", index=False)
    with open(out_dir / "step5a_feature_diagnostics.json", "w") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    (out_dir / "step5a_feature_diagnostics.md").write_text(_render_markdown(report))

    return report


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _build_verdict(
    ablation_results: dict[str, dict],
    official_log_loss: float,
    groups_helpful: list[dict],
    groups_harmful: list[dict],
) -> dict:
    beating = [
        r for r in ablation_results.values()
        if r["platt_log_loss"] < official_log_loss - NOISE_THRESHOLD
    ]
    beating.sort(key=lambda r: r["platt_log_loss"])
    any_beats_official = len(beating) > 0

    most_valuable = [g["group"] for g in groups_helpful[:3]]
    least_valuable = [g["group"] for g in groups_harmful[:3]]  # removal HELPED -> low/negative value

    if any_beats_official:
        best = beating[0]
        summary = (
            f"Ablation {best['name']!r} beat the official model by "
            f"{official_log_loss - best['platt_log_loss']:.4f} log-loss points "
            f"(> the {NOISE_THRESHOLD} noise threshold)."
        )
    else:
        summary = (
            "No ablation beat the current official test log loss by more than the "
            f"{NOISE_THRESHOLD} noise threshold; the full Step 3C feature set remains "
            "the best-performing configuration found here."
        )

    return {
        "any_ablation_beats_official": any_beats_official,
        "ablations_beating_official": [
            {"name": r["name"], "platt_log_loss": r["platt_log_loss"],
             "improvement_over_official": official_log_loss - r["platt_log_loss"]}
            for r in beating
        ],
        "summary": summary,
        "most_valuable_groups": most_valuable,
        "least_valuable_or_harmful_groups": least_valuable,
        "keep_current_official_model": True,
        "keep_current_official_model_rationale": (
            "This is a diagnostics-only run; it never changes the official model "
            "regardless of ablation results. Recommend staying with the current "
            "stronger-regularized Step 3C LR + Platt unless a future run finds a "
            "clear (> noise threshold), "
            "reproduced improvement."
        ),
        "next_step_recommendation": _diagnostic_recommendation(groups_helpful, groups_harmful),
    }


def _diagnostic_recommendation(groups_helpful: list[dict], groups_harmful: list[dict]) -> str:
    if groups_harmful:
        harmful_names = ", ".join(g["group"] for g in groups_harmful)
        return (
            f"Treat {harmful_names} as diagnostic leads only. Any feature change "
            "must be selected on a fresh, validation-only protocol before it can "
            "challenge the current official model."
        )
    return (
        "These ablations are diagnostics only. Preserve the current official model "
        "unless a future validation-only experiment establishes a clear improvement."
    )


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Step 5A: Feature Diagnostics")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    lines.append(
        "Diagnostics-only run. Does not change the official model, add odds/market "
        "features, train new model families, or run hyperparameter searches."
    )
    lines.append("")

    official = report["official_baseline"]["official_model"]["test_metrics"]
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- Official {OFFICIAL_MODEL_LABEL} held-out test log loss: **{official['log_loss']:.4f}** "
        f"(Brier {official['brier_score']:.4f}, AUC {official['roc_auc']:.4f}, "
        f"accuracy {official['accuracy']:.4f})"
    )
    lines.append(f"- {report['verdict']['summary']}")
    lines.append(
        f"- Keep current official model: "
        f"{'YES' if report['verdict']['keep_current_official_model'] else 'NO'}"
    )
    lines.append("")

    split = report["split"]
    lines.append("## Split")
    lines.append("")
    lines.append(
        f"- train {split['n_train']} ({split['train_dates'][0]} .. {split['train_dates'][1]}), "
        f"calibration {split['n_calibration']} ({split['calibration_dates'][0]} .. "
        f"{split['calibration_dates'][1]}), test {split['n_test']} "
        f"({split['test_dates'][0]} .. {split['test_dates'][1]})"
    )
    lines.append("")

    lines.append("## Ablation table (sorted by Platt test log loss)")
    lines.append("")
    lines.append("| name | n_features | platt_log_loss | delta_vs_official | auc | accuracy |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in report["ablations"]:
        lines.append(
            f"| {r['name']} | {r['n_features']} | {r['platt_log_loss']:.4f} | "
            f"{r['delta_vs_official']:+.4f} | "
            f"{r['roc_auc']:.4f} | {r['accuracy']:.4f} |"
        )
    lines.append("")

    lines.append("## Group effects (from removal ablations; noise threshold "
                  f"{report['noise_threshold_log_loss']})")
    lines.append("")
    for label, records in (
        ("Appears helpful (removing hurt)", report["groups_helpful"]),
        ("Appears harmful/no benefit (removing helped or was neutral-negative)", report["groups_harmful"]),
        ("Neutral / noise", report["groups_neutral"]),
    ):
        lines.append(f"**{label}**")
        if records:
            for rec in records:
                lines.append(
                    f"- {rec['group']} (removal ablation {rec['removal_ablation']}, "
                    f"delta {rec['delta_log_loss_if_removed']:+.4f})"
                )
        else:
            lines.append("- none")
        lines.append("")

    lines.append("## Top coefficients (standardized for numeric features)")
    lines.append("")
    lines.append("Top positive:")
    for row in report["top_positive_coefficients"][:15]:
        lines.append(f"- {row['feature']} ({row['group']}): {row['coefficient']:+.4f}")
    lines.append("")
    lines.append("Top negative:")
    for row in report["top_negative_coefficients"][:15]:
        lines.append(f"- {row['feature']} ({row['group']}): {row['coefficient']:+.4f}")
    lines.append("")

    lines.append("## Top permutation importances (Platt-calibrated test log-loss increase)")
    lines.append("")
    if report["permutation_importance"]["skipped"]:
        lines.append("- skipped (--skip-permutation)")
    else:
        for row in report["top_permutation_importances"][:15]:
            lines.append(
                f"- {row['feature']} ({row['group']}): {row['mean_importance']:+.5f} "
                f"(std {row['std']:.5f}, n_repeats {row['n_repeats']})"
            )
        lines.append("")
        lines.append(
            "Values close to 0 relative to their std are noise, not real importance."
        )
    lines.append("")

    lines.append("## Leakage / audit status")
    lines.append("")
    for name, entry in report["leakage_checks"].items():
        lines.append(f"- **{name}**: {entry['status']}")
    lines.append("")
    scaling = report.get("scaling_audit") or {}
    lines.append(
        f"- Scaler present: {scaling.get('scaler_present')}, "
        f"fit on train only: {scaling.get('fit_on_train_only')}"
    )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(report["verdict"]["next_step_recommendation"])
    lines.append("")

    return "\n".join(lines)
