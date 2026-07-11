"""Step 5B.1: regularization confirmation check.

This is an audit pass, not a new search. It reuses the Step 5B LR pipeline
factory, chronological split helpers, Platt calibration path, and coefficient
reporting helpers, then evaluates exactly four pre-declared candidates:

* the official Step 3C LR + Platt anchor;
* the Step 5B elastic-net / drop_all_weight_class finalist;
* the best Step 5B validation-selected elastic-net using all official
  features;
* the best Step 5B validation-selected simple L1/L2 regularized candidate.

All stability checks run inside the pre-test period only. The official final
test split is re-reported for those four fixed candidates, with an explicit
protocol note that it was already touched in Step 5B and is not a fresh
independent test.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_pipeline.calibration import (
    PlattCalibrator,
    calibration_table_with_gap,
    clip_probabilities,
)
from ufc_pipeline.feature_diagnostics import (
    DEFAULT_BASELINE_PATH,
    NOISE_THRESHOLD,
    build_coefficient_report,
    official_step3c_features,
)
from ufc_pipeline.modeling import RANDOM_STATE, TARGET, coerce_numeric_features
from ufc_pipeline.step5b_regularization_search import (
    COEF_ZERO_TOL,
    DEFAULT_MAX_ITER,
    RUN1_KEY_FEATURES,
    WEIGHT_CLASS_COLUMN,
    build_feature_configs,
    build_pretest_and_official_split,
    build_rolling_folds,
    build_weight_class_handling_analysis,
    evaluate_finalist_on_official_split,
    fit_and_evaluate_fold,
    leakage_checks as step5b_leakage_checks,
    resolve_weight_class_handling,
    summarize_finalist_sparsity,
)

CONFIRMATION_FOLD_FRACTIONS: list[dict[str, float]] = [
    {"fit_end": 0.45, "calib_end": 0.55, "val_end": 0.65},
    {"fit_end": 0.50, "calib_end": 0.60, "val_end": 0.70},
    {"fit_end": 0.55, "calib_end": 0.65, "val_end": 0.75},
    {"fit_end": 0.65, "calib_end": 0.75, "val_end": 0.85},
    {"fit_end": 0.75, "calib_end": 0.85, "val_end": 1.00},
]

PREDECLARED_CANDIDATE_NAMES: tuple[str, ...] = (
    "official_baseline",
    "best_step5b_elastic_net_drop_weight_class",
    "best_step5b_elastic_net_all_features",
    "best_step5b_l2_or_l1_candidate",
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
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _candidate_from_values(
    feature_config: str,
    penalty: str,
    C: float,
    solver: str,
    l1_ratio: float | None = None,
    class_weight: str | None = None,
) -> dict:
    return {
        "feature_config": feature_config,
        "class_weight": class_weight,
        "penalty": penalty,
        "C": float(C),
        "solver": solver,
        "l1_ratio": None if l1_ratio is None else float(l1_ratio),
    }


def _candidate_from_step5b_row(row: dict) -> dict:
    return _candidate_from_values(
        feature_config=row["feature_config"],
        penalty=row["penalty"],
        C=row["C"],
        solver=row["solver"],
        l1_ratio=row.get("l1_ratio"),
        class_weight=row.get("class_weight"),
    )


def _validation_sort_key(row: dict) -> tuple:
    auc = row.get("mean_val_auc")
    auc = auc if auc is not None else 0.0
    return (
        row["mean_val_platt_log_loss"],
        row.get("mean_val_brier", float("inf")),
        -auc,
        row.get("n_input_features", 10_000),
    )


def _best_validation_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return sorted(rows, key=_validation_sort_key)[0]


def build_predeclared_confirmation_candidates(step5b_report: dict | None = None) -> list[dict]:
    """Return the fixed Step 5B.1 candidate list.

    The Step 5B report is used only to recover the exact validation-selected
    all-features elastic-net and simple L1/L2 candidates. Final-test results
    in that report are deliberately ignored.
    """
    validation_results = list((step5b_report or {}).get("validation_results", []))

    elastic_net_all = _best_validation_row(
        [
            row
            for row in validation_results
            if row.get("feature_config") == "official_all_features"
            and row.get("penalty") == "elasticnet"
            and row.get("class_weight") is None
        ]
    )
    if elastic_net_all is None:
        elastic_net_all_candidate = _candidate_from_values(
            "official_all_features", "elasticnet", 0.01, "saga", l1_ratio=0.5
        )
        elastic_net_all_source = "fallback_default_from_step5b_summary"
        elastic_net_all_step5b_validation = None
    else:
        elastic_net_all_candidate = _candidate_from_step5b_row(elastic_net_all)
        elastic_net_all_source = "step5b_validation_results"
        elastic_net_all_step5b_validation = {
            k: elastic_net_all.get(k)
            for k in (
                "mean_val_platt_log_loss",
                "std_val_platt_log_loss",
                "per_fold_val_platt_log_loss",
                "mean_val_brier",
                "mean_val_auc",
                "mean_val_accuracy",
            )
        }

    simple_rows = [
        row
        for row in ((step5b_report or {}).get("best_l1"), (step5b_report or {}).get("best_l2"))
        if row is not None and row.get("class_weight") is None
    ]
    simple_row = _best_validation_row(simple_rows)
    if simple_row is None:
        simple_candidate = _candidate_from_values("drop_all_weight_class", "l1", 0.03, "saga")
        simple_source = "fallback_default_from_step5b_summary"
        simple_step5b_validation = None
    else:
        simple_candidate = _candidate_from_step5b_row(simple_row)
        simple_source = "step5b_best_l1_or_l2_validation_result"
        simple_step5b_validation = {
            k: simple_row.get(k)
            for k in (
                "mean_val_platt_log_loss",
                "std_val_platt_log_loss",
                "per_fold_val_platt_log_loss",
                "mean_val_brier",
                "mean_val_auc",
                "mean_val_accuracy",
            )
        }

    return [
        {
            "name": "official_baseline",
            "purpose": "Existing official Step 3C LR + Platt anchor.",
            "candidate": _candidate_from_values("official_all_features", "l2", 1.0, "lbfgs"),
            "source": "official_baseline_anchor",
            "step5b_validation": None,
        },
        {
            "name": "best_step5b_elastic_net_drop_weight_class",
            "purpose": "Best selected Step 5B finalist; checks the borderline test winner.",
            "candidate": _candidate_from_values(
                "drop_all_weight_class", "elasticnet", 0.003, "saga", l1_ratio=0.1
            ),
            "source": "predeclared_from_step5b_finalist_summary",
            "step5b_validation": None,
        },
        {
            "name": "best_step5b_elastic_net_all_features",
            "purpose": (
                "Checks whether elastic-net alone performs similarly when "
                "weight_class is still available for coefficient zeroing."
            ),
            "candidate": elastic_net_all_candidate,
            "source": elastic_net_all_source,
            "step5b_validation": elastic_net_all_step5b_validation,
        },
        {
            "name": "best_step5b_l2_or_l1_candidate",
            "purpose": (
                "Checks whether the gain is broadly stronger regularization, "
                "not one fragile elastic-net setting."
            ),
            "candidate": simple_candidate,
            "source": simple_source,
            "step5b_validation": simple_step5b_validation,
        },
    ]


def _candidate_label(candidate: dict) -> str:
    l1 = candidate.get("l1_ratio")
    l1_text = "" if l1 is None else f", l1_ratio={l1}"
    cw = candidate.get("class_weight")
    cw_text = "" if cw is None else f", class_weight={cw}"
    return (
        f"{candidate['feature_config']} / {candidate['penalty']} "
        f"C={candidate['C']}{l1_text}{cw_text}"
    )


def _fold_calibration_table(
    pipeline,
    fold: dict,
    cols: list[str],
) -> list[dict]:
    calib_probs = pipeline.predict_proba(fold["calib"][cols])[:, 1]
    val_probs_uncal = pipeline.predict_proba(fold["val"][cols])[:, 1]
    y_calib = fold["calib"][TARGET].astype(int).to_numpy()
    y_val = fold["val"][TARGET].astype(int).to_numpy()
    platt = PlattCalibrator().fit(calib_probs, y_calib)
    val_probs_platt = clip_probabilities(platt.transform(val_probs_uncal))
    return calibration_table_with_gap(y_val, val_probs_platt)


def _coefficient_fold_record(
    candidate_name: str,
    fold: dict,
    coef_df: pd.DataFrame,
) -> dict:
    sparsity = summarize_finalist_sparsity(coef_df)
    zeroed_features = sorted(coef_df.loc[coef_df["abs_coefficient"] <= COEF_ZERO_TOL, "feature"].tolist())

    key_features = {}
    for feature in RUN1_KEY_FEATURES:
        row = coef_df[coef_df["feature"] == feature]
        if row.empty:
            key_features[feature] = {
                "status": "not_in_model",
                "coefficient": None,
                "sign": "not_in_model",
            }
            continue
        rec = row.iloc[0]
        key_features[feature] = {
            "status": "nonzero" if rec["abs_coefficient"] > COEF_ZERO_TOL else "zeroed",
            "coefficient": float(rec["coefficient"]),
            "sign": rec["sign"],
        }

    wc_rows = coef_df[coef_df["feature_type"] == "one_hot"]
    weight_class = {
        "status": sparsity["weight_class_status"],
        "nonzero": sparsity["weight_class_nonzero"],
        "total": sparsity["weight_class_total"],
        "zeroed_features": sorted(
            wc_rows.loc[wc_rows["abs_coefficient"] <= COEF_ZERO_TOL, "feature"].tolist()
        ),
        "retained_features": sorted(
            wc_rows.loc[wc_rows["abs_coefficient"] > COEF_ZERO_TOL, "feature"].tolist()
        ),
    }

    return {
        "candidate": candidate_name,
        "fold": fold["fold"],
        "n_total": sparsity["n_total"],
        "n_nonzero": sparsity["n_nonzero"],
        "n_zeroed": sparsity["n_zeroed"],
        "zeroed_features": zeroed_features,
        "key_features": key_features,
        "weight_class": weight_class,
    }


def _summarize_coefficient_stability(
    candidate_name: str,
    candidate: dict,
    fold_records: list[dict],
    final_coef_df: pd.DataFrame,
) -> dict:
    final_sparsity = summarize_finalist_sparsity(final_coef_df)

    zeroed_sets = [set(record["zeroed_features"]) for record in fold_records]
    zeroed_union = sorted(set().union(*zeroed_sets)) if zeroed_sets else []
    zeroed_intersection = sorted(set.intersection(*zeroed_sets)) if zeroed_sets else []

    important_features = {}
    for feature in RUN1_KEY_FEATURES:
        statuses = [record["key_features"][feature]["status"] for record in fold_records]
        signs = [record["key_features"][feature]["sign"] for record in fold_records]
        coefficients = [record["key_features"][feature]["coefficient"] for record in fold_records]
        nonzero_signs = sorted({s for s in signs if s not in ("zero", "not_in_model")})
        important_features[feature] = {
            "survives_all_folds": all(status == "nonzero" for status in statuses),
            "statuses_by_fold": statuses,
            "signs_by_fold": signs,
            "coefficients_by_fold": coefficients,
            "sign_is_stable": len(nonzero_signs) <= 1,
        }

    prior = important_features["prior_wins_diff"]
    prior_signs = [s for s in prior["signs_by_fold"] if s not in ("zero", "not_in_model")]
    if not prior["survives_all_folds"]:
        prior_note = "prior_wins_diff is zeroed or absent in at least one fold."
    elif len(set(prior_signs)) > 1:
        prior_note = "prior_wins_diff changes sign across folds, so it remains unstable."
    elif set(prior_signs) == {"negative"}:
        prior_note = "prior_wins_diff survives but remains negative/counterintuitive across folds."
    else:
        prior_note = "prior_wins_diff survives with a stable non-negative sign across folds."

    wc_statuses = [record["weight_class"]["status"] for record in fold_records]
    wc_all_zeroed_or_absent = all(status in ("all_zeroed", "not_included") for status in wc_statuses)

    return {
        "candidate": candidate_name,
        "candidate_config": candidate,
        "candidate_label": _candidate_label(candidate),
        "n_nonzero_by_fold": [record["n_nonzero"] for record in fold_records],
        "n_zeroed_by_fold": [record["n_zeroed"] for record in fold_records],
        "zeroed_features_union": zeroed_union,
        "zeroed_features_intersection": zeroed_intersection,
        "zeroed_features_by_fold": [
            {"fold": record["fold"], "features": record["zeroed_features"]} for record in fold_records
        ],
        "important_features": important_features,
        "prior_wins_diff_assessment": prior_note,
        "weight_class_status_by_fold": wc_statuses,
        "weight_class_zeroed_or_absent_all_folds": wc_all_zeroed_or_absent,
        "final_refit_sparsity": final_sparsity,
    }


def _summarize_validation(candidate_name: str, candidate: dict, fold_entries: list[dict]) -> dict:
    log_losses = [entry["platt_log_loss"] for entry in fold_entries]
    briers = [entry["platt_brier"] for entry in fold_entries]
    aucs = [entry["platt_auc"] for entry in fold_entries if entry["platt_auc"] is not None]
    accuracies = [entry["platt_accuracy"] for entry in fold_entries]
    return {
        "candidate": candidate_name,
        "candidate_config": candidate,
        "candidate_label": _candidate_label(candidate),
        "mean_val_platt_log_loss": float(np.mean(log_losses)),
        "std_val_platt_log_loss": float(np.std(log_losses)),
        "per_fold_val_platt_log_loss": log_losses,
        "mean_val_brier": float(np.mean(briers)),
        "mean_val_auc": float(np.mean(aucs)) if aucs else None,
        "mean_val_accuracy": float(np.mean(accuracies)),
        "folds": fold_entries,
    }


def _build_fold_by_fold_comparison(validation: dict[str, dict]) -> list[dict]:
    official = validation["official_baseline"]
    rows = []
    for idx, official_fold in enumerate(official["folds"]):
        row = {
            "fold": official_fold["fold"],
            "n_val": official_fold["n_val"],
            "val_dates": official_fold["val_dates"],
            "official_baseline_log_loss": official_fold["platt_log_loss"],
            "candidates": {},
        }
        for name, summary in validation.items():
            if name == "official_baseline":
                continue
            fold = summary["folds"][idx]
            improvement = official_fold["platt_log_loss"] - fold["platt_log_loss"]
            row["candidates"][name] = {
                "platt_log_loss": fold["platt_log_loss"],
                "improvement_vs_official": improvement,
                "beats_official": improvement > 0,
                "beats_official_by_more_than_noise_threshold": improvement > NOISE_THRESHOLD,
                "brier": fold["platt_brier"],
                "auc": fold["platt_auc"],
                "accuracy": fold["platt_accuracy"],
            }
        rows.append(row)
    return rows


def _add_official_comparisons(validation: dict[str, dict]) -> None:
    official = validation["official_baseline"]
    official_mean = official["mean_val_platt_log_loss"]
    official_folds = official["per_fold_val_platt_log_loss"]
    for name, summary in validation.items():
        if name == "official_baseline":
            summary["mean_improvement_vs_official"] = 0.0
            summary["fold_improvements_vs_official"] = [0.0] * len(official_folds)
            summary["folds_beating_official"] = 0
            summary["beats_official_mean_by_more_than_noise_threshold"] = False
            continue
        improvements = [
            official_ll - candidate_ll
            for official_ll, candidate_ll in zip(official_folds, summary["per_fold_val_platt_log_loss"])
        ]
        summary["mean_improvement_vs_official"] = official_mean - summary["mean_val_platt_log_loss"]
        summary["fold_improvements_vs_official"] = improvements
        summary["folds_beating_official"] = int(sum(delta > 0 for delta in improvements))
        summary["folds_beating_official_by_more_than_noise_threshold"] = int(
            sum(delta > NOISE_THRESHOLD for delta in improvements)
        )
        summary["beats_official_mean_by_more_than_noise_threshold"] = (
            summary["mean_improvement_vs_official"] > NOISE_THRESHOLD
        )


def _summarize_step5b_report(step5b_report: dict | None) -> dict:
    if not step5b_report:
        return {"available": False}

    best_test = None
    for row in step5b_report.get("test_results", []):
        labels = set(row.get("labels", []))
        if "best_overall" in labels or "best_drop_all_weight_class" in labels:
            best_test = row
            break

    validation_results = step5b_report.get("validation_results", [])
    best_drop = _best_validation_row(
        [r for r in validation_results if r.get("feature_config") == "drop_all_weight_class"]
    )
    best_all = _best_validation_row(
        [r for r in validation_results if r.get("feature_config") == "official_all_features"]
    )
    weight_class_gap = None
    if best_drop and best_all:
        weight_class_gap = best_all["mean_val_platt_log_loss"] - best_drop["mean_val_platt_log_loss"]

    return {
        "available": True,
        "generated_at": step5b_report.get("generated_at"),
        "best_selected_finalist": {
            "candidate": (best_test or {}).get("candidate"),
            "labels": (best_test or {}).get("labels"),
            "test_platt_log_loss": (best_test or {}).get("platt_test_log_loss"),
            "improvement_over_official": (best_test or {}).get("improvement_over_official"),
        },
        "best_l1": step5b_report.get("best_l1"),
        "best_l2": step5b_report.get("best_l2"),
        "best_elastic_net": step5b_report.get("best_elastic_net"),
        "best_drop_all_weight_class_validation_log_loss": (
            best_drop.get("mean_val_platt_log_loss") if best_drop else None
        ),
        "best_official_all_features_validation_log_loss": (
            best_all.get("mean_val_platt_log_loss") if best_all else None
        ),
        "drop_all_minus_official_all_validation_improvement": weight_class_gap,
        "interpretation": (
            "Step 5B already touched the official final test. Step 5B.1 "
            "therefore treats Step 5B final-test numbers as confirmatory "
            "context only, not as a fresh model-selection signal."
        ),
    }


def _build_weight_class_verdict(validation: dict[str, dict], coefficient_stability: dict[str, dict]) -> dict:
    drop = validation["best_step5b_elastic_net_drop_weight_class"]
    all_features = validation["best_step5b_elastic_net_all_features"]
    drop_vs_all = all_features["mean_val_platt_log_loss"] - drop["mean_val_platt_log_loss"]
    material_drop_gain = drop_vs_all > NOISE_THRESHOLD

    all_stability = coefficient_stability["best_step5b_elastic_net_all_features"]
    final_wc = all_stability["final_refit_sparsity"]["weight_class_status"]
    folds_zeroed = all_stability["weight_class_zeroed_or_absent_all_folds"]
    zeroed_when_included = final_wc == "all_zeroed" and folds_zeroed

    if material_drop_gain:
        robustness = "drop_all_weight_class materially improves validation versus elastic-net all_features."
        safest = "drop_weight_class_explicitly_for_the_candidate"
    elif zeroed_when_included:
        robustness = (
            "Explicit dropping is not materially better; elastic-net zeroes "
            "weight_class when it is included."
        )
        safest = "keep_weight_class_available_for_the_candidate_and_rely_on_elastic_net_zeroing"
    else:
        robustness = "The weight_class decision remains inconclusive."
        safest = "retain_official_features_for_now"

    return {
        "elastic_net_all_features_weight_class_status_final_refit": final_wc,
        "elastic_net_all_features_weight_class_status_by_fold": all_stability["weight_class_status_by_fold"],
        "elastic_net_zeroes_weight_class_when_included": zeroed_when_included,
        "drop_all_vs_elastic_net_all_features_mean_validation_log_loss_delta": drop_vs_all,
        "drop_all_materially_improves_validation": material_drop_gain,
        "weight_class_decision": "robust_drop" if material_drop_gain else "inconclusive_or_self_zeroed",
        "assessment": robustness,
        "safest_recommendation": safest,
        "current_official_feature_recommendation": "retain_official_features_for_now",
        "candidate_feature_recommendation_if_carried_forward": safest,
        "rare_division_note": (
            "Open Weight, Catchweight/Catch Weight, Super Heavyweight, and "
            "rare historical divisions remain treated as noisy/nonstandard; "
            "this run does not try to rescue them."
        ),
    }


def _build_recommendation(
    validation: dict[str, dict],
    final_tests: dict[str, dict],
    weight_class_verdict: dict,
    leakage: dict,
) -> dict:
    official_final = final_tests["official_baseline"]
    candidates = {k: v for k, v in validation.items() if k != "official_baseline"}
    best_validation_name, best_validation = sorted(
        candidates.items(), key=lambda item: item[1]["mean_val_platt_log_loss"]
    )[0]
    best_final = final_tests[best_validation_name]

    validation_gain = best_validation["mean_improvement_vs_official"]
    final_gain = official_final["platt_test_log_loss"] - best_final["platt_test_log_loss"]
    consistent = best_validation["folds_beating_official"] == len(best_validation["folds"])
    calibration_not_worse = (
        best_validation["mean_val_brier"] <= validation["official_baseline"]["mean_val_brier"]
        and best_final["platt_test_brier"] <= official_final["platt_test_brier"]
    )
    leakage_ok = all(entry["status"] in ("pass", "disclosed") for entry in leakage.values())
    fragile_weight_class = (
        best_validation_name == "best_step5b_elastic_net_drop_weight_class"
        and weight_class_verdict["drop_all_materially_improves_validation"]
    )

    promotion_criteria = {
        "validation_gain_gt_noise_threshold": validation_gain > NOISE_THRESHOLD,
        "consistent_fold_by_fold_improvement": consistent,
        "final_test_gain_gt_noise_threshold": final_gain > NOISE_THRESHOLD,
        "calibration_not_worse": calibration_not_worse,
        "not_dependent_on_fragile_weight_class_only_effect": not fragile_weight_class,
        "leakage_protocol_checks_pass": leakage_ok,
    }

    criteria_pass = all(promotion_criteria.values())
    borderline_final = final_gain < 1.5 * NOISE_THRESHOLD

    replace_now = bool(criteria_pass and not borderline_final)
    if replace_now:
        official_recommendation = (
            "Recommend replacing the official model after human sign-off; "
            "do not update benchmarks/official_baseline.json inside this run."
        )
    elif criteria_pass and borderline_final:
        official_recommendation = (
            "Do not replace the official model yet. Stronger regularization "
            "is validation-stable, but the already-touched final-test edge "
            "is still borderline, so carry the locked candidate forward."
        )
    else:
        official_recommendation = (
            "Do not replace the official model. The confirmation criteria "
            "were not all met."
        )

    elastic = validation["best_step5b_elastic_net_drop_weight_class"]
    all_features = validation["best_step5b_elastic_net_all_features"]
    simple = validation["best_step5b_l2_or_l1_candidate"]
    elastic_vs_all = all_features["mean_val_platt_log_loss"] - elastic["mean_val_platt_log_loss"]
    elastic_vs_simple = simple["mean_val_platt_log_loss"] - elastic["mean_val_platt_log_loss"]

    if elastic_vs_simple > NOISE_THRESHOLD:
        elastic_verdict = "elastic_net_clearly_beats_simple_l1_l2_on_validation"
    else:
        elastic_verdict = "elastic_net_not_clearly_better_than_simple_l1_l2"

    return {
        "best_validation_candidate": best_validation_name,
        "best_validation_candidate_config": best_validation["candidate_config"],
        "best_validation_mean_log_loss": best_validation["mean_val_platt_log_loss"],
        "best_validation_gain_vs_official": validation_gain,
        "best_validation_folds_beating_official": best_validation["folds_beating_official"],
        "already_touched_final_test_gain_vs_official": final_gain,
        "already_touched_final_test_log_loss": best_final["platt_test_log_loss"],
        "promotion_criteria": promotion_criteria,
        "criteria_pass": criteria_pass,
        "final_test_edge_is_borderline": borderline_final,
        "recommend_replace_official_model_now": replace_now,
        "official_model_recommendation": official_recommendation,
        "is_stronger_regularization_robustly_better": (
            validation_gain > NOISE_THRESHOLD and consistent and calibration_not_worse
        ),
        "is_elastic_net_clearly_better_than_tuned_l2_l1": elastic_verdict,
        "elastic_net_drop_vs_all_features_validation_log_loss_delta": elastic_vs_all,
        "elastic_net_drop_vs_simple_validation_log_loss_delta": elastic_vs_simple,
        "should_drop_weight_class_explicitly": weight_class_verdict["safest_recommendation"],
        "carry_candidate_into_run3": not replace_now,
        "run3_focus": (
            "Keep Run 3 LR-only and validation-gated. Lock the stronger-"
            "regularization candidate family, compare elastic-net all_features "
            "versus explicit weight_class drop on fresh chronological data or "
            "another predeclared stability scheme, and investigate coefficient "
            "stability for prior_wins_diff and the other sparsity-selected "
            "features before touching the official benchmark."
        ),
    }


def run_regularization_confirmation(
    input_csv: str,
    output_dir: str = "reports",
    random_seed: int = RANDOM_STATE,
    train_frac: float = 0.70,
    calibration_frac: float = 0.15,
    baseline_path: str = DEFAULT_BASELINE_PATH,
    step5b_report_path: str | None = "reports/step5b_regularization_search.json",
    fold_fractions: list[dict[str, float]] | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict:
    """Run Step 5B.1 and write JSON/Markdown reports."""
    baseline = _load_required_json(baseline_path)
    official_log_loss = baseline["official_model"]["test_metrics"]["log_loss"]
    step5b_report = _load_json(step5b_report_path)

    numeric, _categorical = official_step3c_features()
    df = pd.read_csv(input_csv)
    before = len(df)
    df = df[df[TARGET].notna() & df["fighter_a_expected_win_prob"].notna()]
    dropped = before - len(df)
    df = coerce_numeric_features(df, numeric, context="step5b1_regularization_confirmation")

    pretest, official_train, official_calib, official_test = build_pretest_and_official_split(
        df, train_frac=train_frac, calibration_frac=calibration_frac
    )
    fractions = fold_fractions if fold_fractions is not None else CONFIRMATION_FOLD_FRACTIONS
    folds = build_rolling_folds(pretest, fractions=fractions)

    feature_configs = build_feature_configs(drop_rare_thresholds=())
    candidate_specs = build_predeclared_confirmation_candidates(step5b_report)
    if [spec["name"] for spec in candidate_specs] != list(PREDECLARED_CANDIDATE_NAMES):
        raise AssertionError("Step 5B.1 candidate list changed; update tests and report protocol.")

    validation_by_name: dict[str, dict] = {}
    coefficient_stability: dict[str, dict] = {}
    final_test_by_name: dict[str, dict] = {}
    final_coefficient_rows = []

    for spec in candidate_specs:
        name = spec["name"]
        candidate = spec["candidate"]
        feature_config = feature_configs[candidate["feature_config"]]

        fold_entries = []
        coefficient_fold_records = []
        for fold in folds:
            entry, pipeline = fit_and_evaluate_fold(
                candidate,
                numeric,
                feature_config,
                fold,
                random_state=random_seed,
                max_iter=max_iter,
            )
            include_wc, categories = resolve_weight_class_handling(feature_config, fold["fit"])
            categorical = [WEIGHT_CLASS_COLUMN] if include_wc else []
            cols = list(numeric) + categorical
            entry = dict(entry)
            entry["val_dates"] = fold["val_dates"]
            entry["calibration_bucket_summary"] = _fold_calibration_table(pipeline, fold, cols)
            fold_entries.append(entry)

            coef_df = build_coefficient_report(pipeline, numeric, categorical)
            coefficient_fold_records.append(_coefficient_fold_record(name, fold, coef_df))

        validation_by_name[name] = _summarize_validation(name, candidate, fold_entries)

        final_result, final_coef_df = evaluate_finalist_on_official_split(
            [name],
            candidate,
            numeric,
            feature_configs,
            official_train,
            official_calib,
            official_test,
            official_log_loss,
            random_state=random_seed,
            max_iter=max_iter,
        )
        final_result = dict(final_result)
        final_result["final_test_protocol_note"] = (
            "This is a re-report for a pre-declared Step 5B.1 candidate. "
            "The final test window was already touched in Step 5B, so this "
            "number is not an independent fresh test and is not used to "
            "select candidates."
        )
        final_test_by_name[name] = final_result

        final_coef_df = final_coef_df.copy()
        final_coef_df["candidate"] = name
        final_coefficient_rows.append(final_coef_df)
        coefficient_stability[name] = _summarize_coefficient_stability(
            name, candidate, coefficient_fold_records, final_coef_df
        )

    _add_official_comparisons(validation_by_name)
    fold_comparison = _build_fold_by_fold_comparison(validation_by_name)

    wc_analysis = build_weight_class_handling_analysis(feature_configs, official_train, df)
    leakage = step5b_leakage_checks(numeric, official_train, official_calib, official_test, wc_analysis)
    leakage["predeclared_candidates_only"] = {
        "status": "pass",
        "detail": (
            "Step 5B.1 evaluates exactly four fixed candidates from "
            "PREDECLARED_CANDIDATE_NAMES; it does not call "
            "valid_penalty_solver_combos(), generate_candidates(), or "
            "run_validation_search()."
        ),
    }
    leakage["final_test_re_report_only_not_selection"] = {
        "status": "disclosed",
        "detail": (
            "The official final test was previously touched in Step 5B. "
            "Step 5B.1 re-reports it only for the four pre-declared "
            "candidates after validation stability has been computed."
        ),
    }

    weight_class_verdict = _build_weight_class_verdict(validation_by_name, coefficient_stability)
    recommendation = _build_recommendation(
        validation_by_name, final_test_by_name, weight_class_verdict, leakage
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": "step5b1_regularization_confirmation",
        "lr_only": True,
        "changes_official_model": False,
        "changes_official_baseline_file": False,
        "official_baseline": baseline,
        "step5b_summary": _summarize_step5b_report(step5b_report),
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
            "n_predeclared_candidates": len(candidate_specs),
            "candidate_names": [spec["name"] for spec in candidate_specs],
            "does_broad_search": False,
        },
        "validation_protocol": {
            "description": (
                "Five expanding-window confirmation folds, entirely inside "
                "the pre-test period (official train + official calibration). "
                "Each fold fits the base LR on the fold fit window, fits "
                "Platt only on the fold calibration window, and scores only "
                "the fold validation window."
            ),
            "fold_fractions": fractions,
            "folds": [
                {k: v for k, v in fold.items() if k not in ("fit", "calib", "val")}
                for fold in folds
            ],
            "selection_note": (
                "There is no candidate selection from final-test results. "
                "The candidates are fixed before this run starts."
            ),
        },
        "predeclared_candidates": candidate_specs,
        "validation_stability": validation_by_name,
        "fold_by_fold_comparison": fold_comparison,
        "final_test_metrics_re_reported": final_test_by_name,
        "coefficient_sparsity_stability": coefficient_stability,
        "weight_class_verdict": weight_class_verdict,
        "leakage_protocol_checks": leakage,
        "promotion_criteria": recommendation["promotion_criteria"],
        "recommendation": recommendation,
        "limitations": [
            "The official final test was already touched in Step 5B, so Step 5B.1 cannot create a fresh independent test result.",
            "The confirmation folds are still row-fraction chronological folds, not fixed calendar blocks.",
            "Only logistic regression regularization settings are evaluated; no nonlinear models, ensembles, odds, sportsbooks, or prediction-market features are introduced.",
            "This run does not alter benchmarks/official_baseline.json or any data/processed outputs.",
        ],
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "step5b1_regularization_confirmation.json", "w") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    (out_dir / "step5b1_regularization_confirmation.md").write_text(_render_markdown(report))
    return report


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _render_markdown(report: dict) -> str:
    rec = report["recommendation"]
    official_test = report["official_baseline"]["official_model"]["test_metrics"]
    lines: list[str] = []
    lines.append("# Step 5B.1: Regularization Confirmation Check")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    lines.append(
        "LR-only confirmation pass. No new model families, no feature engineering, "
        "no odds/market data, and no changes to `benchmarks/official_baseline.json`."
    )
    lines.append("")

    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Official held-out test Platt log loss: **{official_test['log_loss']:.6f}**")
    lines.append(
        f"- Best validation candidate: **{rec['best_validation_candidate']}** "
        f"({rec['best_validation_gain_vs_official']:.6f} mean validation gain vs official)"
    )
    lines.append(
        f"- Already-touched final-test gain for that candidate: "
        f"**{rec['already_touched_final_test_gain_vs_official']:.6f}**"
    )
    lines.append(
        f"- Stronger regularization robustly better on validation: "
        f"**{rec['is_stronger_regularization_robustly_better']}**"
    )
    lines.append(
        f"- Recommend replacing official model now: "
        f"**{rec['recommend_replace_official_model_now']}**"
    )
    lines.append(f"- {rec['official_model_recommendation']}")
    lines.append("")

    lines.append("## Pre-Declared Candidates")
    lines.append("")
    lines.append("| name | config | source |")
    lines.append("|---|---|---|")
    for spec in report["predeclared_candidates"]:
        lines.append(
            f"| {spec['name']} | {_candidate_label(spec['candidate'])} | {spec['source']} |"
        )
    lines.append("")

    lines.append("## Validation Stability")
    lines.append("")
    lines.append("| candidate | mean Platt LL | std | mean gain vs official | folds won | Brier | AUC | accuracy |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, summary in sorted(
        report["validation_stability"].items(),
        key=lambda item: item[1]["mean_val_platt_log_loss"],
    ):
        lines.append(
            f"| {name} | {_fmt(summary['mean_val_platt_log_loss'], 6)} | "
            f"{_fmt(summary['std_val_platt_log_loss'], 6)} | "
            f"{_fmt(summary['mean_improvement_vs_official'], 6)} | "
            f"{summary['folds_beating_official']}/{len(summary['folds'])} | "
            f"{_fmt(summary['mean_val_brier'], 6)} | "
            f"{_fmt(summary['mean_val_auc'], 6)} | "
            f"{_fmt(summary['mean_val_accuracy'], 6)} |"
        )
    lines.append("")

    lines.append("## Fold-By-Fold Log Loss")
    lines.append("")
    candidate_names = [name for name in report["validation_stability"] if name != "official_baseline"]
    header = "| fold | official | " + " | ".join(candidate_names) + " |"
    sep = "|---:|---:|" + "|".join(["---:"] * len(candidate_names)) + "|"
    lines.append(header)
    lines.append(sep)
    for fold in report["fold_by_fold_comparison"]:
        cells = [str(fold["fold"]), _fmt(fold["official_baseline_log_loss"], 6)]
        for name in candidate_names:
            c = fold["candidates"][name]
            cells.append(f"{_fmt(c['platt_log_loss'], 6)} ({c['improvement_vs_official']:+.6f})")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Final Test Re-Report")
    lines.append("")
    lines.append(
        "The official final test window was already touched in Step 5B. These "
        "numbers are re-reported for the four fixed candidates only and are "
        "not used for candidate selection."
    )
    lines.append("")
    lines.append("| candidate | Platt LL | gain vs official | Brier | AUC | accuracy |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, result in sorted(
        report["final_test_metrics_re_reported"].items(),
        key=lambda item: item[1]["platt_test_log_loss"],
    ):
        gain = official_test["log_loss"] - result["platt_test_log_loss"]
        lines.append(
            f"| {name} | {_fmt(result['platt_test_log_loss'], 6)} | "
            f"{gain:+.6f} | {_fmt(result['platt_test_brier'], 6)} | "
            f"{_fmt(result['platt_test_auc'], 6)} | {_fmt(result['platt_test_accuracy'], 6)} |"
        )
    lines.append("")

    lines.append("## Coefficient And Sparsity Stability")
    lines.append("")
    for name, stability in report["coefficient_sparsity_stability"].items():
        final = stability["final_refit_sparsity"]
        lines.append(
            f"- **{name}**: final nonzero {final['n_nonzero']}/{final['n_total']}; "
            f"weight_class={final['weight_class_status']} "
            f"({final['weight_class_nonzero']}/{final['weight_class_total']}); "
            f"prior_wins_diff: {stability['prior_wins_diff_assessment']}"
        )
        key_bits = []
        for feature, detail in stability["important_features"].items():
            key_bits.append(f"{feature}={'survives' if detail['survives_all_folds'] else 'not stable'}")
        lines.append(f"  - Key features: {', '.join(key_bits)}")
    lines.append("")

    lines.append("## weight_class Verdict")
    lines.append("")
    wc = report["weight_class_verdict"]
    lines.append(f"- Elastic-net zeroes weight_class when included: **{wc['elastic_net_zeroes_weight_class_when_included']}**")
    lines.append(
        f"- Drop-all vs elastic-net all-features validation delta: "
        f"**{wc['drop_all_vs_elastic_net_all_features_mean_validation_log_loss_delta']:+.6f}**"
    )
    lines.append(f"- Current official feature recommendation: **{wc['current_official_feature_recommendation']}**")
    lines.append(
        f"- Candidate feature recommendation if carried forward: "
        f"**{wc['candidate_feature_recommendation_if_carried_forward']}**"
    )
    lines.append(f"- {wc['assessment']}")
    lines.append("")

    lines.append("## Protocol Checks")
    lines.append("")
    for name, entry in report["leakage_protocol_checks"].items():
        lines.append(f"- **{name}**: {entry['status']}")
    lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"- Is stronger regularization robustly better than official? {rec['is_stronger_regularization_robustly_better']}")
    lines.append(f"- Is elastic-net clearly better than tuned L2/L1? {rec['is_elastic_net_clearly_better_than_tuned_l2_l1']}")
    lines.append(f"- Should weight_class be explicitly dropped? {rec['should_drop_weight_class_explicitly']}")
    lines.append(f"- Should the official model be replaced now? {rec['recommend_replace_official_model_now']}")
    lines.append(f"- Carry candidate into Run 3? {rec['carry_candidate_into_run3']}")
    lines.append(f"- Run 3 focus: {rec['run3_focus']}")
    lines.append("")

    return "\n".join(lines)
