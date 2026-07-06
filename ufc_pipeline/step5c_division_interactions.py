"""Step 5C / Run 3: controlled division-interaction LR experiment.

Run 3 asks whether ``weight_class`` becomes useful only when paired with
meaningful pre-fight matchup features. It stays deliberately narrow:

* logistic regression only;
* locked/near-locked regularization from Step 5B/5B.1;
* five pre-test rolling-origin folds copied from Step 5B.1;
* division eligibility decided from pre-test rows only;
* final test re-reported for pre-declared/validation-selected finalists only.

The official benchmark remains read-only. No sportsbook odds, prediction
markets, nonlinear models, random forests, boosting, neural nets, or
ensembles are introduced here.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_pipeline.calibration import (
    PlattCalibrator,
    calibration_table_with_gap,
    clip_probabilities,
    high_confidence_diagnostics,
)
from ufc_pipeline.feature_diagnostics import (
    DEFAULT_BASELINE_PATH,
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
)
from ufc_pipeline.step5b_regularization_search import (
    COEF_ZERO_TOL,
    CURRENT_UFC_DIVISIONS,
    DEFAULT_MAX_ITER,
    RUN1_KEY_FEATURES,
    WEIGHT_CLASS_COLUMN,
    _fit_quietly,
    build_feature_configs,
    build_pretest_and_official_split,
    build_rolling_folds,
    make_step5b_pipeline,
)
from ufc_pipeline.step5b1_regularization_confirmation import CONFIRMATION_FOLD_FRACTIONS

STEP5C_REPORT_JSON = "step5c_division_interactions.json"
STEP5C_REPORT_MD = "step5c_division_interactions.md"
STEP5C_CANDIDATE_CSV = "step5c_candidate_results.csv"
STEP5C_COEFFICIENT_CSV = "step5c_interaction_coefficients.csv"

MIN_DIVISION_COUNT = 100
MIN_DIVISION_FRACTION = 0.01

NONSTANDARD_DIVISION_KEYS = {
    "openweight",
    "catchweight",
    "superheavyweight",
}

PRIORITY_INTERACTION_BASE_FEATURES: list[str] = [
    "age_diff",
    "elo_diff",
    "prior_fights_diff",
    "prior_wins_diff",
    "prior_win_pct_diff",
    "striking_accuracy_diff",
    "striking_defense_diff",
    "takedown_accuracy_diff",
    "takedown_defense_diff",
    "submission_attempts_per_15_diff",
    "submission_attempts_absorbed_per_15_diff",
    "reach_diff",
    "height_diff",
    "days_since_last_fight_diff",
    "win_pct_last_3_diff",
    "win_pct_last_5_diff",
    "takedowns_allowed_per_15_diff",
    "opp_takedown_attempts_per_15_diff",
    "opp_sig_str_attempted_per_min_diff",
    "control_time_absorbed_per_15_diff",
    "knockdowns_absorbed_per_15_diff",
    "sig_strikes_absorbed_per_min_diff",
]

TOP_INTERACTION_BASE_FEATURES: list[str] = [
    "elo_diff",
    "age_diff",
    "striking_defense_diff",
    "striking_accuracy_diff",
    "prior_wins_diff",
]

DOMAIN_INTERACTION_BASE_FEATURES: list[str] = [
    "age_diff",
    "reach_diff",
    "height_diff",
    "days_since_last_fight_diff",
    "striking_defense_diff",
    "takedown_defense_diff",
    "prior_fights_diff",
]

INTERACTION_FEATURE_SETS: dict[str, list[str]] = {
    "none": [],
    "priority": PRIORITY_INTERACTION_BASE_FEATURES,
    "top_run1": TOP_INTERACTION_BASE_FEATURES,
    "domain": DOMAIN_INTERACTION_BASE_FEATURES,
}

PREDECLARED_CANDIDATE_NAMES: tuple[str, ...] = (
    "official_baseline",
    "stronger_regularized_all_features",
    "stronger_regularized_drop_raw_weight_class",
    "interactions_no_raw_weight_class",
    "interactions_with_raw_weight_class_regularized",
    "interactions_top_features_only",
    "interactions_domain_features_only",
)


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


def normalize_division_key(value: object) -> str:
    """Normalize a division label for exclusion checks."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def sanitize_division_name(value: str) -> str:
    """Stable, readable suffix for interaction feature names."""
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", value)).strip("_")


def interaction_column_name(base_feature: str, division: str) -> str:
    return f"{base_feature}_x_weight_class_{sanitize_division_name(division)}"


def parse_interaction_column(column: str) -> tuple[str, str] | None:
    marker = "_x_weight_class_"
    if marker not in column:
        return None
    base, division = column.split(marker, 1)
    return base, division


def select_interaction_base_features(
    numeric_features: list[str],
    feature_set: str,
) -> tuple[list[str], list[str]]:
    """Resolve a pre-declared feature-set name to existing official columns."""
    if feature_set not in INTERACTION_FEATURE_SETS:
        raise ValueError(f"Unknown interaction feature set: {feature_set!r}")
    allowed = INTERACTION_FEATURE_SETS[feature_set]
    numeric_set = set(numeric_features)
    selected = [feature for feature in allowed if feature in numeric_set]
    skipped = [feature for feature in allowed if feature not in numeric_set]
    return selected, skipped


def analyze_division_eligibility(
    pretest_df: pd.DataFrame,
    full_df: pd.DataFrame,
    min_count: int = MIN_DIVISION_COUNT,
    min_fraction: float = MIN_DIVISION_FRACTION,
    current_divisions: list[str] = CURRENT_UFC_DIVISIONS,
) -> dict:
    """Pick eligible interaction divisions from pre-test counts only.

    ``full_df`` counts are transparency-only. A category can be eligible only
    when it is a current/common UFC division, is not a nonstandard historical
    bucket, and meets the pre-declared pre-test sample threshold.
    """
    threshold = max(int(min_count), int(math.ceil(len(pretest_df) * min_fraction)))
    pre_counts = pretest_df[WEIGHT_CLASS_COLUMN].value_counts(dropna=True).to_dict()
    full_counts = full_df[WEIGHT_CLASS_COLUMN].value_counts(dropna=True).to_dict()
    current_set = set(current_divisions)
    current_order = {division: i for i, division in enumerate(current_divisions)}
    all_categories = sorted(set(pre_counts) | set(full_counts))

    rows = []
    eligible = []
    excluded = []
    for category in all_categories:
        pre_count = int(pre_counts.get(category, 0))
        full_count = int(full_counts.get(category, 0))
        key = normalize_division_key(category)
        reasons = []
        if key in NONSTANDARD_DIVISION_KEYS:
            reasons.append("nonstandard_or_historical")
        if category not in current_set:
            reasons.append("not_current_ufc_division")
        if pre_count < threshold:
            reasons.append("below_pretest_sample_threshold")
        is_eligible = not reasons
        row = {
            "category": category,
            "pretest_count": pre_count,
            "full_data_count": full_count,
            "in_current_ufc_divisions": category in current_set,
            "eligible_for_interactions": is_eligible,
            "excluded_reasons": reasons,
        }
        rows.append(row)
        if is_eligible:
            eligible.append(category)
        else:
            excluded.append(row)

    eligible = sorted(eligible, key=lambda div: (current_order.get(div, 999), div))
    return {
        "threshold": {
            "min_count": int(min_count),
            "min_fraction": float(min_fraction),
            "effective_min_pretest_count": int(threshold),
            "basis": "pretest_only_official_train_plus_calibration",
        },
        "eligible_divisions": eligible,
        "excluded_divisions": excluded,
        "category_counts": rows,
        "nonstandard_division_keys": sorted(NONSTANDARD_DIVISION_KEYS),
        "note": (
            "full_data_count is reported for transparency only. It is never "
            "used to select eligible divisions or interaction columns."
        ),
    }


def build_interaction_columns(
    base_features: list[str],
    eligible_divisions: list[str],
) -> tuple[list[str], dict[str, dict]]:
    columns = []
    metadata = {}
    for division in eligible_divisions:
        for base in base_features:
            column = interaction_column_name(base, division)
            columns.append(column)
            metadata[column] = {"base_feature": base, "division": division}
    return columns, metadata


def add_division_interactions(
    df: pd.DataFrame,
    base_features: list[str],
    eligible_divisions: list[str],
    weight_class_column: str = WEIGHT_CLASS_COLUMN,
) -> tuple[pd.DataFrame, list[str], dict[str, dict]]:
    """Return a copy with deterministic ``base_feature x division`` columns.

    Missing base-feature values remain missing inside the matching division
    and are imputed by the train-only sklearn pipeline. Outside the matching
    division the interaction is exactly zero, even if the base feature is
    missing.
    """
    out = df.copy()
    columns, metadata = build_interaction_columns(base_features, eligible_divisions)
    new_columns = {}
    for column, info in metadata.items():
        indicator = out[weight_class_column].eq(info["division"]).to_numpy()
        base_values = pd.to_numeric(out[info["base_feature"]], errors="coerce").to_numpy(dtype=float)
        new_columns[column] = np.where(indicator, base_values, 0.0)
    if new_columns:
        out = pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)
    return out, columns, metadata


def separate_feature_columns(columns: list[str]) -> dict[str, list[str]]:
    raw_weight_class = [c for c in columns if c.startswith(f"{WEIGHT_CLASS_COLUMN}_")]
    interactions = [c for c in columns if parse_interaction_column(c) is not None]
    base_numeric = [c for c in columns if c not in raw_weight_class and c not in interactions]
    return {
        "base_numeric": base_numeric,
        "raw_weight_class": raw_weight_class,
        "interactions": interactions,
    }


def _candidate(
    name: str,
    penalty: str,
    C: float,
    solver: str,
    l1_ratio: float | None,
    include_raw_weight_class: bool,
    interaction_feature_set: str,
    purpose: str,
    aliases: list[str] | None = None,
) -> dict:
    return {
        "name": name,
        "purpose": purpose,
        "penalty": penalty,
        "C": float(C),
        "solver": solver,
        "l1_ratio": None if l1_ratio is None else float(l1_ratio),
        "class_weight": None,
        "include_raw_weight_class": bool(include_raw_weight_class),
        "interaction_feature_set": interaction_feature_set,
        "aliases": list(aliases or []),
        "model_family": "logistic_regression",
    }


def build_predeclared_step5c_candidates(step5b1_report: dict | None = None) -> list[dict]:
    """Fixed Run 3 candidate set. Final-test metrics are ignored."""
    en_all = None
    if step5b1_report:
        for spec in step5b1_report.get("predeclared_candidates", []):
            if spec.get("name") == "best_step5b_elastic_net_all_features":
                en_all = spec.get("candidate")
                break
    if en_all is None:
        en_all = {
            "penalty": "elasticnet",
            "C": 0.01,
            "solver": "saga",
            "l1_ratio": 0.5,
        }

    candidates = [
        _candidate(
            "official_baseline",
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            l1_ratio=None,
            include_raw_weight_class=True,
            interaction_feature_set="none",
            purpose="Official Step 3C LR + Platt anchor.",
        ),
        _candidate(
            "stronger_regularized_all_features",
            penalty=en_all["penalty"],
            C=en_all["C"],
            solver=en_all["solver"],
            l1_ratio=en_all.get("l1_ratio"),
            include_raw_weight_class=True,
            interaction_feature_set="none",
            purpose="Run 2.1 stronger-regularized all-features challenger.",
        ),
        _candidate(
            "stronger_regularized_drop_raw_weight_class",
            penalty="elasticnet",
            C=0.003,
            solver="saga",
            l1_ratio=0.1,
            include_raw_weight_class=False,
            interaction_feature_set="none",
            purpose="Locked Run 2.1 stronger-regularized no-interaction challenger.",
        ),
        _candidate(
            "interactions_no_raw_weight_class",
            penalty="elasticnet",
            C=0.003,
            solver="saga",
            l1_ratio=0.1,
            include_raw_weight_class=False,
            interaction_feature_set="priority",
            aliases=["interactions_common_divisions_only"],
            purpose="Main Run 3 interaction candidate with raw weight_class dropped.",
        ),
        _candidate(
            "interactions_with_raw_weight_class_regularized",
            penalty="elasticnet",
            C=0.003,
            solver="saga",
            l1_ratio=0.1,
            include_raw_weight_class=True,
            interaction_feature_set="priority",
            purpose="Interaction candidate allowing elastic-net to zero raw weight_class.",
        ),
        _candidate(
            "interactions_top_features_only",
            penalty="elasticnet",
            C=0.003,
            solver="saga",
            l1_ratio=0.1,
            include_raw_weight_class=False,
            interaction_feature_set="top_run1",
            purpose="Compact interaction candidate using Run 1's strongest features.",
        ),
        _candidate(
            "interactions_domain_features_only",
            penalty="elasticnet",
            C=0.003,
            solver="saga",
            l1_ratio=0.1,
            include_raw_weight_class=False,
            interaction_feature_set="domain",
            purpose="Compact interaction candidate using domain-specific division logic.",
        ),
    ]
    if [c["name"] for c in candidates] != list(PREDECLARED_CANDIDATE_NAMES):
        raise AssertionError("Run 3 candidate names changed without updating tests.")
    return candidates


def _candidate_label(candidate: dict) -> str:
    l1 = candidate.get("l1_ratio")
    l1_text = "" if l1 is None else f", l1_ratio={l1}"
    raw = "raw_wc=yes" if candidate["include_raw_weight_class"] else "raw_wc=no"
    return (
        f"{candidate['penalty']} C={candidate['C']}{l1_text}; "
        f"{raw}; interactions={candidate['interaction_feature_set']}"
    )


def _validate_interaction_inputs(base_numeric: list[str], interaction_metadata: dict[str, dict]) -> None:
    check_features_allowed(list(base_numeric) + [WEIGHT_CLASS_COLUMN])
    official_numeric, _ = official_step3c_features()
    official_numeric_set = set(official_numeric)
    bad_bases = sorted(
        {meta["base_feature"] for meta in interaction_metadata.values() if meta["base_feature"] not in official_numeric_set}
    )
    if bad_bases:
        raise ValueError(f"Interaction base features are not official Step 3C features: {bad_bases}")


def _prepare_candidate_frame(
    df: pd.DataFrame,
    candidate: dict,
    base_numeric: list[str],
    eligible_divisions: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, dict], list[str], list[str]]:
    selected_bases, skipped_bases = select_interaction_base_features(
        base_numeric, candidate["interaction_feature_set"]
    )
    if selected_bases:
        prepared, interaction_cols, interaction_metadata = add_division_interactions(
            df, selected_bases, eligible_divisions
        )
    else:
        prepared = df.copy()
        interaction_cols, interaction_metadata = [], {}
    _validate_interaction_inputs(base_numeric, interaction_metadata)
    return prepared, interaction_cols, interaction_metadata, selected_bases, skipped_bases


def _coef_report_step5c(
    pipeline,
    numeric: list[str],
    include_raw_weight_class: bool,
    interaction_cols: list[str],
    interaction_metadata: dict[str, dict],
) -> pd.DataFrame:
    categorical = [WEIGHT_CLASS_COLUMN] if include_raw_weight_class else []
    coef_df = build_coefficient_report(pipeline, numeric, categorical)
    interaction_set = set(interaction_cols)
    coef_df["is_raw_weight_class"] = coef_df["feature_type"].eq("one_hot")
    coef_df["is_interaction"] = coef_df["feature"].isin(interaction_set)
    coef_df.loc[coef_df["is_interaction"], "feature_type"] = "interaction"
    coef_df.loc[coef_df["is_interaction"], "group"] = "division_interaction"
    coef_df["interaction_base_feature"] = coef_df["feature"].map(
        lambda feature: interaction_metadata.get(feature, {}).get("base_feature")
    )
    coef_df["interaction_division"] = coef_df["feature"].map(
        lambda feature: interaction_metadata.get(feature, {}).get("division")
    )
    return coef_df


def _fit_calibrated_candidate(
    candidate: dict,
    base_numeric: list[str],
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    eligible_divisions: list[str],
    random_state: int = RANDOM_STATE,
    max_iter: int = DEFAULT_MAX_ITER,
) -> tuple[dict, pd.DataFrame, object]:
    train_prepared, interaction_cols, interaction_metadata, selected_bases, skipped_bases = _prepare_candidate_frame(
        train_df, candidate, base_numeric, eligible_divisions
    )
    calib_prepared, calib_interaction_cols, _, _, _ = _prepare_candidate_frame(
        calib_df, candidate, base_numeric, eligible_divisions
    )
    eval_prepared, eval_interaction_cols, _, _, _ = _prepare_candidate_frame(
        eval_df, candidate, base_numeric, eligible_divisions
    )
    if interaction_cols != calib_interaction_cols or interaction_cols != eval_interaction_cols:
        raise AssertionError("Interaction columns must be deterministic across split windows.")

    numeric = list(base_numeric) + list(interaction_cols)
    categorical = [WEIGHT_CLASS_COLUMN] if candidate["include_raw_weight_class"] else []
    cols = numeric + categorical

    pipeline = make_step5b_pipeline(
        numeric=numeric,
        include_weight_class=candidate["include_raw_weight_class"],
        weight_class_categories=None,
        penalty=candidate["penalty"],
        C=candidate["C"],
        l1_ratio=candidate.get("l1_ratio"),
        solver=candidate["solver"],
        max_iter=max_iter,
        class_weight=candidate.get("class_weight"),
        random_state=random_state,
    )
    _fit_quietly(pipeline, train_prepared[cols], train_prepared[TARGET].astype(int))

    calib_probs = pipeline.predict_proba(calib_prepared[cols])[:, 1]
    eval_probs_uncal = pipeline.predict_proba(eval_prepared[cols])[:, 1]
    y_calib = calib_prepared[TARGET].astype(int).to_numpy()
    y_eval = eval_prepared[TARGET].astype(int).to_numpy()
    platt = PlattCalibrator().fit(calib_probs, y_calib)
    eval_probs_platt = clip_probabilities(platt.transform(eval_probs_uncal))

    uncal_metrics = evaluate_probs(y_eval, clip_probabilities(eval_probs_uncal))
    platt_metrics = evaluate_probs(y_eval, eval_probs_platt)
    platt_metrics["calibration"] = calibration_table_with_gap(y_eval, eval_probs_platt)
    platt_metrics["high_confidence"] = high_confidence_diagnostics(y_eval, eval_probs_platt)

    coef_df = _coef_report_step5c(
        pipeline, numeric, candidate["include_raw_weight_class"], interaction_cols, interaction_metadata
    )
    n_nonzero = int((coef_df["abs_coefficient"] > COEF_ZERO_TOL).sum())
    raw_rows = coef_df[coef_df["is_raw_weight_class"]]
    interaction_rows = coef_df[coef_df["is_interaction"]]
    result = {
        "uncal_log_loss": uncal_metrics["log_loss"],
        "platt_log_loss": platt_metrics["log_loss"],
        "platt_brier": platt_metrics["brier_score"],
        "platt_auc": platt_metrics["roc_auc"],
        "platt_accuracy": platt_metrics["accuracy"],
        "calibration_table": platt_metrics["calibration"],
        "high_confidence": platt_metrics["high_confidence"],
        "n_total_features_post_encoding": int(len(coef_df)),
        "n_nonzero_coefficients": n_nonzero,
        "n_zeroed_coefficients": int(len(coef_df) - n_nonzero),
        "n_raw_weight_class_features": int(len(raw_rows)),
        "n_raw_weight_class_nonzero": int((raw_rows["abs_coefficient"] > COEF_ZERO_TOL).sum()),
        "n_interaction_features": int(len(interaction_rows)),
        "n_interaction_nonzero": int((interaction_rows["abs_coefficient"] > COEF_ZERO_TOL).sum()),
        "interaction_columns": list(interaction_cols),
        "interaction_metadata": interaction_metadata,
        "interaction_base_features_used": selected_bases,
        "interaction_base_features_skipped": skipped_bases,
        "raw_weight_class_included": bool(candidate["include_raw_weight_class"]),
    }
    return result, coef_df, pipeline


def evaluate_candidate_on_fold(
    candidate: dict,
    base_numeric: list[str],
    fold: dict,
    eligible_divisions: list[str],
    random_state: int = RANDOM_STATE,
    max_iter: int = DEFAULT_MAX_ITER,
) -> tuple[dict, pd.DataFrame]:
    metrics, coef_df, _pipeline = _fit_calibrated_candidate(
        candidate,
        base_numeric,
        fold["fit"],
        fold["calib"],
        fold["val"],
        eligible_divisions,
        random_state=random_state,
        max_iter=max_iter,
    )
    entry = {
        "fold": fold["fold"],
        "n_fit": fold["n_fit"],
        "n_calib": fold["n_calib"],
        "n_val": fold["n_val"],
        "fit_dates": fold["fit_dates"],
        "calib_dates": fold["calib_dates"],
        "val_dates": fold["val_dates"],
        **metrics,
    }
    return entry, coef_df


def _summarize_validation(candidate: dict, fold_entries: list[dict]) -> dict:
    lls = [entry["platt_log_loss"] for entry in fold_entries]
    briers = [entry["platt_brier"] for entry in fold_entries]
    aucs = [entry["platt_auc"] for entry in fold_entries if entry["platt_auc"] is not None]
    accs = [entry["platt_accuracy"] for entry in fold_entries]
    first = fold_entries[0]
    return {
        "candidate": candidate["name"],
        "candidate_config": candidate,
        "candidate_label": _candidate_label(candidate),
        "mean_val_platt_log_loss": float(np.mean(lls)),
        "std_val_platt_log_loss": float(np.std(lls)),
        "per_fold_val_platt_log_loss": lls,
        "mean_val_brier": float(np.mean(briers)),
        "mean_val_auc": float(np.mean(aucs)) if aucs else None,
        "mean_val_accuracy": float(np.mean(accs)),
        "n_total_features_post_encoding": first["n_total_features_post_encoding"],
        "n_nonzero_coefficients_fold1": first["n_nonzero_coefficients"],
        "n_raw_weight_class_nonzero_fold1": first["n_raw_weight_class_nonzero"],
        "n_interaction_features": first["n_interaction_features"],
        "n_interaction_nonzero_fold1": first["n_interaction_nonzero"],
        "interaction_base_features_used": first["interaction_base_features_used"],
        "interaction_base_features_skipped": first["interaction_base_features_skipped"],
        "folds": fold_entries,
    }


def _selection_sort_key(row: dict) -> tuple:
    auc = row.get("mean_val_auc")
    auc = auc if auc is not None else 0.0
    return (
        row["mean_val_platt_log_loss"],
        row.get("mean_val_brier", float("inf")),
        -auc,
        row.get("n_nonzero_coefficients_fold1", 10_000),
        row.get("n_raw_weight_class_nonzero_fold1", 10_000),
    )


def select_step5c_finalists(validation_results: list[dict]) -> list[str]:
    """Validation-only finalist rule; no test data parameter by construction."""
    by_name = {row["candidate"]: row for row in validation_results}
    finalists = [
        "official_baseline",
        "stronger_regularized_drop_raw_weight_class",
    ]
    interactions = [
        row for row in validation_results
        if row["candidate_config"]["interaction_feature_set"] != "none"
    ]
    compact = [
        row for row in interactions
        if row["candidate_config"]["interaction_feature_set"] in ("top_run1", "domain")
    ]
    raw_interactions = [
        row for row in interactions
        if row["candidate_config"]["include_raw_weight_class"]
    ]
    for rows in (interactions, compact, raw_interactions):
        if rows:
            finalists.append(sorted(rows, key=_selection_sort_key)[0]["candidate"])

    deduped = []
    for name in finalists:
        if name in by_name and name not in deduped:
            deduped.append(name)
    return deduped


def _add_validation_comparisons(validation_by_name: dict[str, dict]) -> None:
    official = validation_by_name["official_baseline"]
    challenger = validation_by_name["stronger_regularized_drop_raw_weight_class"]
    official_folds = official["per_fold_val_platt_log_loss"]
    challenger_folds = challenger["per_fold_val_platt_log_loss"]
    for name, summary in validation_by_name.items():
        candidate_folds = summary["per_fold_val_platt_log_loss"]
        official_improvements = [a - b for a, b in zip(official_folds, candidate_folds)]
        challenger_improvements = [a - b for a, b in zip(challenger_folds, candidate_folds)]
        summary["mean_improvement_vs_official"] = (
            official["mean_val_platt_log_loss"] - summary["mean_val_platt_log_loss"]
        )
        summary["mean_improvement_vs_challenger"] = (
            challenger["mean_val_platt_log_loss"] - summary["mean_val_platt_log_loss"]
        )
        summary["fold_improvements_vs_official"] = official_improvements
        summary["fold_improvements_vs_challenger"] = challenger_improvements
        summary["folds_beating_official"] = int(sum(delta > 0 for delta in official_improvements))
        summary["folds_beating_challenger"] = int(sum(delta > 0 for delta in challenger_improvements))
        summary["folds_beating_challenger_by_noise_threshold"] = int(
            sum(delta > NOISE_THRESHOLD for delta in challenger_improvements)
        )
        summary["beats_official_mean_by_more_than_noise_threshold"] = (
            summary["mean_improvement_vs_official"] > NOISE_THRESHOLD
        )
        summary["beats_challenger_mean_by_more_than_noise_threshold"] = (
            summary["mean_improvement_vs_challenger"] > NOISE_THRESHOLD
        )


def _build_fold_by_fold(validation_by_name: dict[str, dict]) -> list[dict]:
    official = validation_by_name["official_baseline"]
    challenger = validation_by_name["stronger_regularized_drop_raw_weight_class"]
    rows = []
    for idx, official_fold in enumerate(official["folds"]):
        challenger_fold = challenger["folds"][idx]
        row = {
            "fold": official_fold["fold"],
            "n_val": official_fold["n_val"],
            "val_dates": official_fold["val_dates"],
            "official_baseline_log_loss": official_fold["platt_log_loss"],
            "stronger_regularized_drop_raw_weight_class_log_loss": challenger_fold["platt_log_loss"],
            "candidates": {},
        }
        for name, summary in validation_by_name.items():
            fold = summary["folds"][idx]
            row["candidates"][name] = {
                "platt_log_loss": fold["platt_log_loss"],
                "improvement_vs_official": official_fold["platt_log_loss"] - fold["platt_log_loss"],
                "improvement_vs_challenger": challenger_fold["platt_log_loss"] - fold["platt_log_loss"],
                "brier": fold["platt_brier"],
                "auc": fold["platt_auc"],
                "accuracy": fold["platt_accuracy"],
            }
        rows.append(row)
    return rows


def _summarize_sparsity(coef_df: pd.DataFrame) -> dict:
    nonzero = coef_df[coef_df["abs_coefficient"] > COEF_ZERO_TOL]
    raw = coef_df[coef_df["is_raw_weight_class"]]
    interactions = coef_df[coef_df["is_interaction"]]
    nonzero_interactions = interactions[interactions["abs_coefficient"] > COEF_ZERO_TOL]
    by_division = (
        nonzero_interactions.groupby("interaction_division").size().sort_values(ascending=False).to_dict()
        if not nonzero_interactions.empty else {}
    )
    by_base = (
        nonzero_interactions.groupby("interaction_base_feature").size().sort_values(ascending=False).to_dict()
        if not nonzero_interactions.empty else {}
    )
    return {
        "n_total": int(len(coef_df)),
        "n_nonzero": int(len(nonzero)),
        "n_zeroed": int(len(coef_df) - len(nonzero)),
        "raw_weight_class_total": int(len(raw)),
        "raw_weight_class_nonzero": int((raw["abs_coefficient"] > COEF_ZERO_TOL).sum()),
        "raw_weight_class_status": (
            "not_included" if len(raw) == 0 else
            "all_zeroed" if int((raw["abs_coefficient"] > COEF_ZERO_TOL).sum()) == 0 else
            "partially_or_fully_retained"
        ),
        "interaction_total": int(len(interactions)),
        "interaction_nonzero": int(len(nonzero_interactions)),
        "interaction_status": (
            "not_included" if len(interactions) == 0 else
            "all_zeroed" if len(nonzero_interactions) == 0 else
            "some_retained"
        ),
        "nonzero_interactions_by_division": {k: int(v) for k, v in by_division.items()},
        "nonzero_interactions_by_base_feature": {k: int(v) for k, v in by_base.items()},
        "top_positive": nonzero.sort_values("coefficient", ascending=False)
        .head(10)[["feature", "coefficient", "feature_type", "group"]].to_dict("records"),
        "top_negative": nonzero.sort_values("coefficient", ascending=True)
        .head(10)[["feature", "coefficient", "feature_type", "group"]].to_dict("records"),
        "top_nonzero_interactions": nonzero_interactions.sort_values("abs_coefficient", ascending=False)
        .head(20)[
            ["feature", "coefficient", "interaction_base_feature", "interaction_division"]
        ].to_dict("records"),
    }


def _interaction_fold_stability(fold_coef_dfs: list[pd.DataFrame]) -> dict:
    if not fold_coef_dfs:
        return {
            "interactions_with_unstable_signs": [],
            "interactions_surviving_most_folds": [],
            "survival_by_base_feature": {},
            "survival_by_division": {},
        }
    rows = []
    for i, coef_df in enumerate(fold_coef_dfs, start=1):
        interactions = coef_df[coef_df["is_interaction"]].copy()
        interactions["fold"] = i
        interactions["is_nonzero"] = interactions["abs_coefficient"] > COEF_ZERO_TOL
        rows.append(interactions)
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if combined.empty:
        return {
            "interactions_with_unstable_signs": [],
            "interactions_surviving_most_folds": [],
            "survival_by_base_feature": {},
            "survival_by_division": {},
        }

    n_folds = len(fold_coef_dfs)
    most_threshold = max(1, math.ceil(n_folds / 2))
    unstable = []
    surviving = []
    for feature, group in combined.groupby("feature"):
        nonzero_group = group[group["is_nonzero"]]
        signs = sorted(set(nonzero_group["sign"]) - {"zero"})
        n_nonzero = int(len(nonzero_group))
        rec = {
            "feature": feature,
            "base_feature": group["interaction_base_feature"].iloc[0],
            "division": group["interaction_division"].iloc[0],
            "n_nonzero_folds": n_nonzero,
            "signs": signs,
            "coefficients_by_fold": group.sort_values("fold")[["fold", "coefficient", "sign"]].to_dict("records"),
        }
        if len(signs) > 1:
            unstable.append(rec)
        if n_nonzero >= most_threshold and len(signs) == 1:
            surviving.append(rec)

    surviving_df = pd.DataFrame(surviving)
    by_base = (
        surviving_df.groupby("base_feature").size().sort_values(ascending=False).to_dict()
        if not surviving_df.empty else {}
    )
    by_division = (
        surviving_df.groupby("division").size().sort_values(ascending=False).to_dict()
        if not surviving_df.empty else {}
    )
    return {
        "interactions_with_unstable_signs": unstable,
        "interactions_surviving_most_folds": sorted(
            surviving, key=lambda r: (-r["n_nonzero_folds"], r["feature"])
        )[:50],
        "survival_by_base_feature": {k: int(v) for k, v in by_base.items()},
        "survival_by_division": {k: int(v) for k, v in by_division.items()},
        "age_diff_interactions_survive": "age_diff" in by_base,
        "elo_diff_interactions_survive": "elo_diff" in by_base,
        "striking_or_grappling_interactions_survive": any(
            ("striking" in str(k) or "takedown" in str(k) or "submission" in str(k))
            for k in by_base
        ),
        "reach_or_height_interactions_survive": any(k in ("reach_diff", "height_diff") for k in by_base),
    }


def _key_feature_status(coef_df: pd.DataFrame) -> dict:
    out = {}
    for feature in RUN1_KEY_FEATURES:
        row = coef_df[coef_df["feature"] == feature]
        if row.empty:
            out[feature] = {"status": "not_in_model", "coefficient": None, "sign": "not_in_model"}
        else:
            rec = row.iloc[0]
            out[feature] = {
                "status": "nonzero" if rec["abs_coefficient"] > COEF_ZERO_TOL else "zeroed",
                "coefficient": float(rec["coefficient"]),
                "sign": rec["sign"],
            }
    return out


def _build_leakage_checks(
    base_numeric: list[str],
    official_train: pd.DataFrame,
    official_calib: pd.DataFrame,
    official_test: pd.DataFrame,
    division_analysis: dict,
    candidates: list[dict],
) -> dict:
    checks = {}
    try:
        check_features_allowed(base_numeric + [WEIGHT_CLASS_COLUMN])
        guard_ok, guard_detail = True, "Official base features and raw weight_class pass the leakage guard."
    except ValueError as exc:
        guard_ok, guard_detail = False, str(exc)
    checks["no_target_or_result_columns"] = {"status": "pass" if guard_ok else "fail", "detail": guard_detail}

    id_cols = {"fight_id", "date", "event", "fighter_a", "fighter_b", "winner", "method", "round"}
    used_ids = sorted(id_cols & set(base_numeric + [WEIGHT_CLASS_COLUMN]))
    checks["no_identifiers_or_date_as_features"] = {
        "status": "pass" if not used_ids else "fail",
        "detail": "none used." if not used_ids else f"identifier columns used: {used_ids}",
    }

    train_ids = set(official_train["fight_id"])
    calib_ids = set(official_calib["fight_id"])
    test_ids = set(official_test["fight_id"])
    disjoint = not (train_ids & calib_ids) and not (calib_ids & test_ids) and not (train_ids & test_ids)
    chrono = (
        official_train["date"].max() <= official_calib["date"].min()
        and official_calib["date"].max() <= official_test["date"].min()
    )
    checks["split_integrity"] = {
        "status": "pass" if disjoint and chrono else "fail",
        "detail": f"disjoint_by_fight_id={disjoint}, chronological_order={chrono}",
    }
    checks["scaler_fit_on_fit_window_or_train_only"] = {
        "status": "pass",
        "detail": "All numeric base and interaction columns are inside the sklearn Pipeline scaler.",
    }
    checks["platt_fit_on_calibration_window_only"] = {
        "status": "pass",
        "detail": "Every fold/final evaluation fits PlattCalibrator only on the corresponding calibration window.",
    }
    checks["final_test_not_used_for_selection_or_eligibility"] = {
        "status": "pass",
        "detail": "Eligible divisions come from pretest only; select_step5c_finalists() accepts validation results only.",
    }
    checks["eligible_divisions_selected_from_pretest_only"] = {
        "status": "pass",
        "detail": f"basis={division_analysis['threshold']['basis']}",
    }
    excluded_reasons = {
        reason
        for row in division_analysis["excluded_divisions"]
        for reason in row["excluded_reasons"]
    }
    checks["rare_nonstandard_divisions_excluded_from_interactions"] = {
        "status": "pass" if {"nonstandard_or_historical", "below_pretest_sample_threshold"} & excluded_reasons else "fail",
        "detail": f"excluded reason types: {sorted(excluded_reasons)}",
    }
    checks["interaction_features_deterministic"] = {
        "status": "pass",
        "detail": "Interaction columns are generated from predeclared base-feature order and eligible-division order.",
    }
    checks["l1_elasticnet_feature_selection_fit_on_training_only"] = {
        "status": "pass",
        "detail": "Coefficient zeroing happens inside LogisticRegression.fit() on fold fit or official train only.",
    }
    checks["final_test_for_finalists_only"] = {
        "status": "pass",
        "detail": "Final test evaluation is run only after validation finalist selection.",
    }
    checks["raw_weight_class_and_interactions_separated"] = {
        "status": "pass",
        "detail": "Coefficient reports mark raw one-hot rows separately from interaction numeric columns.",
    }
    checks["no_nonlinear_models_or_ensembles"] = {
        "status": "pass" if all(c["model_family"] == "logistic_regression" for c in candidates) else "fail",
        "detail": "All Step 5C candidates use sklearn LogisticRegression through the Step 5B pipeline factory.",
    }
    return checks


def _summarize_prior_reports(run1: dict | None, run2: dict | None, run21: dict | None) -> tuple[dict, dict, dict]:
    run1_summary = {"available": False}
    if run1:
        run1_summary = {
            "available": True,
            "generated_at": run1.get("generated_at"),
            "top_permutation_features": run1.get("top_permutation_importances", [])[:5],
            "weight_class_ablation": run1.get("verdict", {}).get("ablations_beating_official"),
            "summary": run1.get("verdict", {}).get("summary"),
        }
    run2_summary = {"available": False}
    if run2:
        best = None
        for row in run2.get("test_results", []):
            if "best_overall" in row.get("labels", []):
                best = row
                break
        run2_summary = {
            "available": True,
            "generated_at": run2.get("generated_at"),
            "best_overall": run2.get("best_overall"),
            "best_test_result": best,
            "best_l1": run2.get("best_l1"),
            "best_l2": run2.get("best_l2"),
            "best_elastic_net": run2.get("best_elastic_net"),
        }
    run21_summary = {"available": False}
    if run21:
        run21_summary = {
            "available": True,
            "generated_at": run21.get("generated_at"),
            "validation_stability": {
                name: {
                    "mean_val_platt_log_loss": row.get("mean_val_platt_log_loss"),
                    "mean_improvement_vs_official": row.get("mean_improvement_vs_official"),
                    "folds_beating_official": row.get("folds_beating_official"),
                }
                for name, row in run21.get("validation_stability", {}).items()
            },
            "recommendation": run21.get("recommendation"),
        }
    return run1_summary, run2_summary, run21_summary


def _build_raw_weight_class_verdict(validation_by_name: dict[str, dict], final_summaries: dict[str, dict]) -> dict:
    raw_candidates = [
        name for name, row in validation_by_name.items()
        if row["candidate_config"]["include_raw_weight_class"]
    ]
    raw_status_final = {
        name: final_summaries.get(name, {}).get("sparsity", {}).get("raw_weight_class_status")
        for name in raw_candidates
        if name in final_summaries
    }
    en_raw = raw_status_final.get("interactions_with_raw_weight_class_regularized")
    return {
        "raw_weight_class_candidates": raw_candidates,
        "raw_weight_class_status_in_finalists": raw_status_final,
        "elastic_net_zeroed_raw_weight_class_when_interactions_included": en_raw == "all_zeroed",
        "assessment": (
            "Raw weight_class remains weak/noisy under elastic-net if all raw one-hot coefficients are zeroed; "
            "any retained raw one-hots should be treated cautiously because both fighters share the same division."
        ),
    }


def _build_interaction_verdict(
    validation_by_name: dict[str, dict],
    final_results: dict[str, dict],
    final_summaries: dict[str, dict],
    leakage: dict,
) -> dict:
    interaction_rows = [
        row for row in validation_by_name.values()
        if row["candidate_config"]["interaction_feature_set"] != "none"
    ]
    best_interaction = sorted(interaction_rows, key=_selection_sort_key)[0] if interaction_rows else None
    challenger = validation_by_name["stronger_regularized_drop_raw_weight_class"]
    official = validation_by_name["official_baseline"]
    if best_interaction is None:
        return {"available": False}

    gain_vs_challenger = challenger["mean_val_platt_log_loss"] - best_interaction["mean_val_platt_log_loss"]
    gain_vs_official = official["mean_val_platt_log_loss"] - best_interaction["mean_val_platt_log_loss"]
    consistent_vs_challenger = best_interaction["folds_beating_challenger"] >= 4
    material_vs_challenger = gain_vs_challenger > NOISE_THRESHOLD
    final_best = final_results.get(best_interaction["candidate"])
    final_challenger = final_results.get("stronger_regularized_drop_raw_weight_class")
    final_consistent = True
    if final_best and final_challenger:
        final_consistent = final_best["platt_test_log_loss"] <= final_challenger["platt_test_log_loss"] + NOISE_THRESHOLD

    final_summary = final_summaries.get(best_interaction["candidate"], {})
    stable_interactions = final_summary.get("interaction_stability", {}).get("interactions_surviving_most_folds", [])
    leakage_ok = all(entry["status"] in ("pass", "disclosed") for entry in leakage.values())

    include_interactions_future = bool((material_vs_challenger or consistent_vs_challenger) and stable_interactions)
    return {
        "available": True,
        "best_interaction_candidate": best_interaction["candidate"],
        "best_interaction_mean_validation_log_loss": best_interaction["mean_val_platt_log_loss"],
        "gain_vs_official_validation": gain_vs_official,
        "gain_vs_stronger_regularized_challenger_validation": gain_vs_challenger,
        "folds_beating_challenger": best_interaction["folds_beating_challenger"],
        "material_gain_vs_challenger_gt_noise_threshold": material_vs_challenger,
        "consistent_vs_challenger": consistent_vs_challenger,
        "final_test_consistent_with_validation": final_consistent,
        "stable_interactions_found": bool(stable_interactions),
        "top_stable_interactions": stable_interactions[:20],
        "should_future_lr_include_division_interactions": include_interactions_future,
        "rare_nonstandard_divisions_remain_excluded": True,
        "leakage_protocol_checks_pass": leakage_ok,
        "assessment": (
            "Division interactions are useful only if they beat the stronger-regularized "
            "no-interaction challenger, not merely the official C=1.0 baseline."
        ),
    }


def _build_recommendation(
    validation_by_name: dict[str, dict],
    final_results: dict[str, dict],
    final_summaries: dict[str, dict],
    interaction_verdict: dict,
    leakage: dict,
) -> dict:
    official = validation_by_name["official_baseline"]
    challenger = validation_by_name["stronger_regularized_drop_raw_weight_class"]
    selected_name = (
        interaction_verdict.get("best_interaction_candidate")
        if interaction_verdict.get("should_future_lr_include_division_interactions")
        else "stronger_regularized_drop_raw_weight_class"
    )
    selected = validation_by_name[selected_name]
    selected_final = final_results.get(selected_name)
    official_final = final_results.get("official_baseline")
    challenger_final = final_results.get("stronger_regularized_drop_raw_weight_class")

    selected_beats_official = selected["mean_improvement_vs_official"] > NOISE_THRESHOLD
    selected_beats_challenger = (
        selected_name == "stronger_regularized_drop_raw_weight_class"
        or selected["mean_improvement_vs_challenger"] > NOISE_THRESHOLD
        or selected["folds_beating_challenger"] >= 4
    )
    calibration_not_worse = selected["mean_val_brier"] <= official["mean_val_brier"]
    if selected_final and official_final:
        calibration_not_worse = calibration_not_worse and (
            selected_final["platt_test_brier"] <= official_final["platt_test_brier"] + 0.001
        )
    final_consistent = True
    if selected_final and official_final:
        final_consistent = selected_final["platt_test_log_loss"] <= official_final["platt_test_log_loss"]
    if selected_name != "stronger_regularized_drop_raw_weight_class" and selected_final and challenger_final:
        final_consistent = final_consistent and (
            selected_final["platt_test_log_loss"] <= challenger_final["platt_test_log_loss"] + NOISE_THRESHOLD
        )

    leakage_ok = all(entry["status"] in ("pass", "disclosed") for entry in leakage.values())
    excessive_complexity = False
    if selected_name in final_summaries:
        summary = final_summaries[selected_name]["sparsity"]
        excessive_complexity = (
            summary["interaction_nonzero"] > 0
            and selected["mean_improvement_vs_challenger"] < NOISE_THRESHOLD
            and summary["interaction_nonzero"] > 30
        )

    criteria = {
        "selected_beats_official_validation_gt_noise": selected_beats_official,
        "selected_beats_stronger_regularized_challenger_or_is_challenger": selected_beats_challenger,
        "final_test_consistent": final_consistent,
        "calibration_not_worse": calibration_not_worse,
        "leakage_protocol_checks_pass": leakage_ok,
        "not_fragile_rare_category_interactions": True,
        "not_excessively_complex_relative_to_gain": not excessive_complexity,
    }
    criteria_pass = all(criteria.values())
    recommend_replace = bool(criteria_pass and selected_name != "official_baseline")
    # This final test has been touched repeatedly; require human sign-off even
    # when the criteria pass.
    if recommend_replace and selected_name == "stronger_regularized_drop_raw_weight_class":
        replacement = (
            "Stronger-regularized no-interaction LR is the best promotion "
            "candidate after Run 3, subject to human sign-off and a separate "
            "benchmark update. This run still does not edit official_baseline.json."
        )
    elif recommend_replace:
        replacement = (
            "The selected interaction LR candidate clears the conservative "
            "criteria, but benchmark promotion still requires explicit human "
            "sign-off because the final test is no longer fresh."
        )
    else:
        replacement = (
            "Do not replace the official model from this run. Keep the official "
            "baseline unchanged and carry the best LR challenger forward."
        )

    return {
        "selected_candidate": selected_name,
        "selected_candidate_config": selected["candidate_config"],
        "promotion_criteria": criteria,
        "criteria_pass": criteria_pass,
        "recommend_replace_official_model_now": False,
        "model_to_promote_if_user_signs_off": selected_name if recommend_replace else None,
        "official_model_recommendation": replacement,
        "interaction_recommendation": (
            "include_division_interactions_in_future_lr"
            if interaction_verdict.get("should_future_lr_include_division_interactions")
            else "reject_or_keep_experimental_division_interactions"
        ),
        "stronger_regularized_lr_recommendation": (
            "promote_after_signoff_or_use_as_run4_lr_baseline"
            if challenger["mean_improvement_vs_official"] > NOISE_THRESHOLD
            else "carry_as_challenger_only"
        ),
        "run4_focus": (
            "Run 4 can proceed to nonlinear models and ensembles only as a "
            "separate phase. Use the stronger-regularized LR challenger as "
            "the LR baseline, keep odds/markets out unless explicitly added "
            "in a future approved phase, and compare any nonlinear model "
            "against both official and the locked LR challenger."
        ),
    }


def run_division_interactions(
    input_csv: str,
    output_dir: str = "reports",
    random_seed: int = RANDOM_STATE,
    train_frac: float = 0.70,
    calibration_frac: float = 0.15,
    baseline_path: str = DEFAULT_BASELINE_PATH,
    run1_report_path: str | None = "reports/step5a_feature_diagnostics.json",
    run2_report_path: str | None = "reports/step5b_regularization_search.json",
    run21_report_path: str | None = "reports/step5b1_regularization_confirmation.json",
    min_division_count: int = MIN_DIVISION_COUNT,
    min_division_fraction: float = MIN_DIVISION_FRACTION,
    max_candidates: int | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict:
    baseline = _load_required_json(baseline_path)
    run1_summary, run2_summary, run21_summary = _summarize_prior_reports(
        _load_json(run1_report_path),
        _load_json(run2_report_path),
        _load_json(run21_report_path),
    )

    base_numeric, _categorical = official_step3c_features()
    df = pd.read_csv(input_csv)
    before = len(df)
    df = df[df[TARGET].notna() & df["fighter_a_expected_win_prob"].notna()].copy()
    dropped = before - len(df)
    df = coerce_numeric_features(df, base_numeric, context="step5c_division_interactions")

    pretest, official_train, official_calib, official_test = build_pretest_and_official_split(
        df, train_frac=train_frac, calibration_frac=calibration_frac
    )
    folds = build_rolling_folds(pretest, fractions=CONFIRMATION_FOLD_FRACTIONS)
    division_analysis = analyze_division_eligibility(
        pretest,
        df,
        min_count=min_division_count,
        min_fraction=min_division_fraction,
    )
    eligible_divisions = division_analysis["eligible_divisions"]
    candidates = build_predeclared_step5c_candidates(_load_json(run21_report_path))
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    validation_by_name: dict[str, dict] = {}
    coefficient_rows = []
    fold_coef_by_candidate: dict[str, list[pd.DataFrame]] = {}
    candidate_feature_report = {}

    for candidate in candidates:
        fold_entries = []
        fold_coef_dfs = []
        selected_bases, skipped_bases = select_interaction_base_features(
            base_numeric, candidate["interaction_feature_set"]
        )
        interaction_cols, _metadata = build_interaction_columns(selected_bases, eligible_divisions)
        candidate_feature_report[candidate["name"]] = {
            "interaction_feature_set": candidate["interaction_feature_set"],
            "base_features_used": selected_bases,
            "base_features_skipped": skipped_bases,
            "n_interaction_features_created": len(interaction_cols),
            "raw_weight_class_included": candidate["include_raw_weight_class"],
        }
        for fold in folds:
            entry, coef_df = evaluate_candidate_on_fold(
                candidate,
                base_numeric,
                fold,
                eligible_divisions,
                random_state=random_seed,
                max_iter=max_iter,
            )
            fold_entries.append(entry)
            fold_coef_dfs.append(coef_df)
        fold_coef_by_candidate[candidate["name"]] = fold_coef_dfs
        validation_by_name[candidate["name"]] = _summarize_validation(candidate, fold_entries)

    _add_validation_comparisons(validation_by_name)
    validation_results = [validation_by_name[c["name"]] for c in candidates if c["name"] in validation_by_name]
    finalist_names = select_step5c_finalists(validation_results)
    fold_by_fold = _build_fold_by_fold(validation_by_name)

    final_results: dict[str, dict] = {}
    final_summaries: dict[str, dict] = {}
    for candidate in candidates:
        if candidate["name"] not in finalist_names:
            continue
        metrics, coef_df, _pipeline = _fit_calibrated_candidate(
            candidate,
            base_numeric,
            official_train,
            official_calib,
            official_test,
            eligible_divisions,
            random_state=random_seed,
            max_iter=max_iter,
        )
        official_log_loss = baseline["official_model"]["test_metrics"]["log_loss"]
        metrics = {
            "candidate": candidate["name"],
            "candidate_config": candidate,
            "candidate_label": _candidate_label(candidate),
            "n_train": int(len(official_train)),
            "n_calibration": int(len(official_calib)),
            "n_test": int(len(official_test)),
            "platt_test_log_loss": metrics["platt_log_loss"],
            "platt_test_brier": metrics["platt_brier"],
            "platt_test_auc": metrics["platt_auc"],
            "platt_test_accuracy": metrics["platt_accuracy"],
            "uncalibrated_test_log_loss": metrics["uncal_log_loss"],
            "calibration_table": metrics["calibration_table"],
            "high_confidence": metrics["high_confidence"],
            "improvement_over_official_test_log_loss": official_log_loss - metrics["platt_log_loss"],
            "n_total_features_post_encoding": metrics["n_total_features_post_encoding"],
            "n_nonzero_coefficients": metrics["n_nonzero_coefficients"],
            "n_zeroed_coefficients": metrics["n_zeroed_coefficients"],
            "n_raw_weight_class_features": metrics["n_raw_weight_class_features"],
            "n_raw_weight_class_nonzero": metrics["n_raw_weight_class_nonzero"],
            "n_interaction_features": metrics["n_interaction_features"],
            "n_interaction_nonzero": metrics["n_interaction_nonzero"],
            "final_test_protocol_note": (
                "The final test was already touched in previous runs. Step 5C "
                "uses it only to re-report metrics for validation-selected "
                "finalists, never for candidate selection."
            ),
        }
        final_results[candidate["name"]] = metrics
        coef_df = coef_df.copy()
        coef_df["candidate"] = candidate["name"]
        coefficient_rows.append(coef_df)
        final_summaries[candidate["name"]] = {
            "sparsity": _summarize_sparsity(coef_df),
            "interaction_stability": _interaction_fold_stability(fold_coef_by_candidate[candidate["name"]]),
            "key_feature_status": _key_feature_status(coef_df),
        }

    leakage = _build_leakage_checks(
        base_numeric,
        official_train,
        official_calib,
        official_test,
        division_analysis,
        candidates,
    )
    raw_wc_verdict = _build_raw_weight_class_verdict(validation_by_name, final_summaries)
    interaction_verdict = _build_interaction_verdict(
        validation_by_name, final_results, final_summaries, leakage
    )
    recommendation = _build_recommendation(
        validation_by_name, final_results, final_summaries, interaction_verdict, leakage
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": "step5c_division_interactions",
        "lr_only": True,
        "changes_official_model": False,
        "changes_official_baseline_file": False,
        "official_baseline": baseline,
        "run1_summary": run1_summary,
        "run2_summary": run2_summary,
        "run21_summary": run21_summary,
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
            "n_candidates_evaluated": len(candidates),
            "max_candidates": max_candidates,
            "candidate_count_is_controlled": True,
        },
        "validation_protocol": {
            "description": (
                "Five Step 5B.1-style expanding-window folds entirely inside "
                "the pre-test period. Candidate selection uses only mean "
                "Platt-calibrated validation log loss and validation tie-"
                "breakers; final test is re-reported for finalists only."
            ),
            "fold_fractions": CONFIRMATION_FOLD_FRACTIONS,
            "folds": [
                {k: v for k, v in fold.items() if k not in ("fit", "calib", "val")}
                for fold in folds
            ],
            "selection_rule": {
                "primary": "mean validation Platt log loss",
                "tie_breakers": [
                    "validation Brier",
                    "validation ROC AUC",
                    "fewer nonzero coefficients",
                    "fewer raw weight_class coefficients",
                ],
                "noise_threshold": NOISE_THRESHOLD,
                "final_test_used_for_selection": False,
            },
        },
        "eligible_division_analysis": division_analysis,
        "base_interaction_features": {
            "priority_allowlist": PRIORITY_INTERACTION_BASE_FEATURES,
            "top_run1_allowlist": TOP_INTERACTION_BASE_FEATURES,
            "domain_allowlist": DOMAIN_INTERACTION_BASE_FEATURES,
            "by_candidate": candidate_feature_report,
        },
        "candidate_configurations": candidates,
        "validation_results": validation_results,
        "fold_by_fold_results": fold_by_fold,
        "finalist_names": finalist_names,
        "final_test_results": final_results,
        "coefficient_sparsity_summary": final_summaries,
        "interaction_stability_summary": {
            name: summary["interaction_stability"] for name, summary in final_summaries.items()
        },
        "raw_weight_class_verdict": raw_wc_verdict,
        "interaction_verdict": interaction_verdict,
        "leakage_protocol_checks": leakage,
        "promotion_criteria": recommendation["promotion_criteria"],
        "recommendation": recommendation,
        "limitations": [
            "The final test has already been touched by earlier runs, so it is not a fresh independent selection signal.",
            "Eligible divisions use pre-test sample counts; full-data counts are transparency only.",
            "Interaction features are simple one-hot x feature products, not nonlinear models.",
            "No attempt is made to rescue Open Weight, Catchweight/Catch Weight, Super Heavyweight, or rare historical divisions.",
        ],
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / STEP5C_REPORT_JSON, "w") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    (out_dir / STEP5C_REPORT_MD).write_text(_render_markdown(report))

    pd.DataFrame(validation_results).to_csv(out_dir / STEP5C_CANDIDATE_CSV, index=False)
    if coefficient_rows:
        pd.concat(coefficient_rows, ignore_index=True).to_csv(out_dir / STEP5C_COEFFICIENT_CSV, index=False)
    return report


def _fmt(value, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _render_markdown(report: dict) -> str:
    rec = report["recommendation"]
    interaction = report["interaction_verdict"]
    lines = []
    lines.append("# Step 5C: Division Interaction Features")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    lines.append(
        "LR-only controlled feature-engineering run. No nonlinear models, "
        "ensembles, sportsbook odds, or prediction-market features."
    )
    lines.append("")

    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Selected candidate: **{rec['selected_candidate']}**")
    lines.append(f"- Recommend replacing official model now: **{rec['recommend_replace_official_model_now']}**")
    lines.append(f"- {rec['official_model_recommendation']}")
    if interaction.get("available"):
        lines.append(
            f"- Best interaction candidate: **{interaction['best_interaction_candidate']}** "
            f"(validation gain vs challenger {interaction['gain_vs_stronger_regularized_challenger_validation']:+.6f})"
        )
        lines.append(
            f"- Future LR should include division interactions: "
            f"**{interaction['should_future_lr_include_division_interactions']}**"
        )
    lines.append("")

    lines.append("## Validation Results")
    lines.append("")
    lines.append("| candidate | mean LL | std | gain vs official | gain vs challenger | folds vs challenger | Brier | AUC | nonzero fold1 | int nonzero fold1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(report["validation_results"], key=lambda r: r["mean_val_platt_log_loss"]):
        lines.append(
            f"| {row['candidate']} | {_fmt(row['mean_val_platt_log_loss'])} | "
            f"{_fmt(row['std_val_platt_log_loss'])} | "
            f"{row['mean_improvement_vs_official']:+.6f} | "
            f"{row['mean_improvement_vs_challenger']:+.6f} | "
            f"{row['folds_beating_challenger']}/5 | "
            f"{_fmt(row['mean_val_brier'])} | {_fmt(row['mean_val_auc'])} | "
            f"{row['n_nonzero_coefficients_fold1']} | {row['n_interaction_nonzero_fold1']} |"
        )
    lines.append("")

    lines.append("## Fold-By-Fold")
    lines.append("")
    finalist_names = report["finalist_names"]
    header = "| fold | official | challenger | " + " | ".join(finalist_names) + " |"
    lines.append(header)
    lines.append("|---:|---:|---:|" + "|".join(["---:"] * len(finalist_names)) + "|")
    for fold in report["fold_by_fold_results"]:
        cells = [
            str(fold["fold"]),
            _fmt(fold["official_baseline_log_loss"]),
            _fmt(fold["stronger_regularized_drop_raw_weight_class_log_loss"]),
        ]
        for name in finalist_names:
            c = fold["candidates"][name]
            cells.append(f"{_fmt(c['platt_log_loss'])} ({c['improvement_vs_challenger']:+.6f})")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Final Test Finalists")
    lines.append("")
    lines.append(
        "The final test was already touched in previous runs; these are "
        "re-reports for validation-selected finalists only."
    )
    lines.append("")
    lines.append("| candidate | Platt LL | gain vs official | Brier | AUC | accuracy | nonzero | raw wc nz | interaction nz |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, row in sorted(report["final_test_results"].items(), key=lambda item: item[1]["platt_test_log_loss"]):
        lines.append(
            f"| {name} | {_fmt(row['platt_test_log_loss'])} | "
            f"{row['improvement_over_official_test_log_loss']:+.6f} | "
            f"{_fmt(row['platt_test_brier'])} | {_fmt(row['platt_test_auc'])} | "
            f"{_fmt(row['platt_test_accuracy'])} | {row['n_nonzero_coefficients']} | "
            f"{row['n_raw_weight_class_nonzero']} | {row['n_interaction_nonzero']} |"
        )
    lines.append("")

    lines.append("## Eligible Divisions")
    lines.append("")
    threshold = report["eligible_division_analysis"]["threshold"]
    lines.append(
        f"Threshold: max({threshold['min_count']}, "
        f"{threshold['min_fraction']:.2%} of pre-test rows) = "
        f"{threshold['effective_min_pretest_count']} fights."
    )
    lines.append("")
    lines.append("- Eligible: " + ", ".join(report["eligible_division_analysis"]["eligible_divisions"]))
    lines.append("")
    lines.append("| category | pretest | full data | eligible | excluded reasons |")
    lines.append("|---|---:|---:|:---:|---|")
    for row in report["eligible_division_analysis"]["category_counts"]:
        lines.append(
            f"| {row['category']} | {row['pretest_count']} | {row['full_data_count']} | "
            f"{'yes' if row['eligible_for_interactions'] else 'no'} | "
            f"{', '.join(row['excluded_reasons'])} |"
        )
    lines.append("")

    lines.append("## Surviving Interactions")
    lines.append("")
    for name, summary in report["coefficient_sparsity_summary"].items():
        sparsity = summary["sparsity"]
        lines.append(
            f"- **{name}**: interactions {sparsity['interaction_nonzero']}/"
            f"{sparsity['interaction_total']} nonzero; raw weight_class "
            f"{sparsity['raw_weight_class_status']}"
        )
        top = sparsity["top_nonzero_interactions"][:8]
        if top:
            rendered = ", ".join(
                f"{r['feature']} ({r['coefficient']:+.3f})" for r in top
            )
            lines.append(f"  - Top: {rendered}")
    lines.append("")

    lines.append("## weight_class Verdict")
    lines.append("")
    raw = report["raw_weight_class_verdict"]
    lines.append(
        f"- Elastic-net zeroed raw weight_class when interactions included: "
        f"**{raw['elastic_net_zeroed_raw_weight_class_when_interactions_included']}**"
    )
    lines.append(f"- {raw['assessment']}")
    lines.append("")

    lines.append("## Protocol Checks")
    lines.append("")
    for name, entry in report["leakage_protocol_checks"].items():
        lines.append(f"- **{name}**: {entry['status']}")
    lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"- Interaction recommendation: {rec['interaction_recommendation']}")
    lines.append(f"- Stronger LR recommendation: {rec['stronger_regularized_lr_recommendation']}")
    lines.append(f"- Run 4 focus: {rec['run4_focus']}")
    lines.append("")
    return "\n".join(lines)
