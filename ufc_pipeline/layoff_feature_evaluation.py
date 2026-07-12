"""Controlled frozen-split ablation for the experimental layoff family.

This module never mutates the official benchmark, model, processed CSVs, DB,
or live ledger. It reproduces the official 43-feature anchor and fits two
versioned candidates with the identical train/calibration/test rows,
hyperparameters, Platt procedure, and random seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ufc_pipeline.calibration import (
    calibration_table_with_gap,
    clip_probabilities,
    high_confidence_diagnostics,
)
from ufc_pipeline.feature_diagnostics import DEFAULT_BASELINE_PATH, official_step3c_features
from ufc_pipeline.layoff_features import (
    LAYOFF_CANDIDATE_A_FEATURES,
    LAYOFF_CANDIDATE_B_FEATURES,
    LAYOFF_FEATURE_COLUMNS,
)
from ufc_pipeline.modeling import (
    RANDOM_STATE,
    TARGET,
    check_features_allowed,
    coerce_numeric_features,
    evaluate_probs,
    extract_feature_names,
)
from ufc_pipeline.step6a_pseudo_live_replay import feature_schema_version, fit_official_model
from ufc_pipeline.step6b_live_predictions import (
    _official_training_split,
    load_official_model_config,
)


DEFAULT_INPUT = "data/processed/experiments/layoff_20260711/ufc_prefight_features_step3c_layoff.csv"
DEFAULT_OUTPUT_DIR = "reports/layoff_experiment"
DEFAULT_ARTIFACT_DIR = "data/models/experiments/layoff_20260711"
NOISE_BAND = 0.002
MIN_BUCKET_ROWS = 30

CANDIDATES = {
    "baseline": [],
    "candidate_a": LAYOFF_CANDIDATE_A_FEATURES,
    "candidate_b": LAYOFF_CANDIDATE_B_FEATURES,
}


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(type(value).__name__)


def expected_calibration_error(table: list[dict]) -> float:
    total = sum(int(row["n_predictions"]) for row in table)
    if not total:
        return float("nan")
    return float(sum(
        int(row["n_predictions"]) * abs(float(row["gap"]))
        for row in table if row.get("gap") is not None
    ) / total)


def _layoff_bucket(row: pd.Series) -> str:
    a, b = row.get("fighter_a_layoff_days"), row.get("fighter_b_layoff_days")
    if pd.isna(a) or pd.isna(b):
        return "Missing"
    days = max(float(a), float(b))
    if days < 60:
        return "Under 60 days"
    if days < 120:
        return "60-119 days"
    if days < 240:
        return "120-239 days"
    if days < 365:
        return "240-364 days"
    if days < 730:
        return "365-729 days"
    return "730+ days"


def _metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    probabilities = clip_probabilities(probabilities)
    result = evaluate_probs(y, probabilities)
    table = calibration_table_with_gap(y, probabilities)
    result["calibration"] = table
    result["expected_calibration_error"] = expected_calibration_error(table)
    result["high_confidence"] = high_confidence_diagnostics(y, probabilities)
    return result


def _bucket_metrics(test: pd.DataFrame, y: np.ndarray, probabilities: np.ndarray) -> list[dict]:
    buckets = test.apply(_layoff_bucket, axis=1)
    order = [
        "Missing", "Under 60 days", "60-119 days", "120-239 days",
        "240-364 days", "365-729 days", "730+ days",
    ]
    rows = []
    for bucket in order:
        mask = buckets.eq(bucket).to_numpy()
        n = int(mask.sum())
        row = {"bucket": bucket, "n": n, "adequate_sample": n >= MIN_BUCKET_ROWS}
        if n:
            scored = _metrics(y[mask], probabilities[mask])
            row.update({
                "log_loss": scored["log_loss"],
                "brier_score": scored["brier_score"],
                "roc_auc": scored["roc_auc"],
                "accuracy": scored["accuracy"],
                "expected_calibration_error": scored["expected_calibration_error"],
            })
        rows.append(row)
    return rows


def _coefficients(pipeline, feature_names: list[str]) -> list[dict]:
    transformed = extract_feature_names(pipeline)
    values = pipeline.named_steps["model"].coef_.ravel()
    records = [
        {"feature": name, "coefficient": float(value), "abs_coefficient": abs(float(value))}
        for name, value in zip(transformed, values)
    ]
    wanted = set(feature_names)
    return sorted((r for r in records if r["feature"] in wanted), key=lambda r: -r["abs_coefficient"])


def run_layoff_evaluation(
    input_csv: str = DEFAULT_INPUT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    baseline_path: str = DEFAULT_BASELINE_PATH,
    overwrite: bool = False,
    random_state: int = RANDOM_STATE,
) -> dict:
    out_dir, models_dir = Path(output_dir), Path(artifact_dir)
    report_path = out_dir / "layoff_feature_ablation.json"
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"{report_path} exists; pass overwrite=True to replace experiment outputs.")
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    official, _ = official_step3c_features()
    all_features = list(official) + list(LAYOFF_FEATURE_COLUMNS)
    check_features_allowed(all_features)
    frame = pd.read_csv(input_csv)
    missing = [column for column in all_features if column not in frame.columns]
    if missing:
        raise ValueError(f"Layoff evaluation input is missing features: {missing}")
    frame = frame[frame[TARGET].notna() & frame["fighter_a_expected_win_prob"].notna()].copy()
    frame = coerce_numeric_features(frame, all_features, context="layoff_feature_evaluation")
    frame = frame.sort_values(["date", "fight_id"], kind="stable").reset_index(drop=True)

    config = load_official_model_config(baseline_path)
    train, calibration, test, split_metadata = _official_training_split(
        frame, config, train_frac=0.70, calibration_frac=0.15,
    )
    y_test = test[TARGET].astype(int).to_numpy()
    results: dict[str, dict] = {}
    prediction_frame = test[[
        "fight_id", "date", "event", "fighter_a", "fighter_b", TARGET,
        "fighter_a_layoff_days", "fighter_b_layoff_days",
    ]].copy()

    for name, additions in CANDIDATES.items():
        features = list(official) + list(additions)
        pipeline, platt = fit_official_model(train, calibration, features, random_state=random_state)
        raw = pipeline.predict_proba(test[features])[:, 1]
        probabilities = clip_probabilities(platt.transform(raw))
        metrics = _metrics(y_test, probabilities)
        prediction_frame[f"{name}_probability_a"] = probabilities
        model_version = (
            config["model_version"] if name == "baseline"
            else f"experimental_{name}_layoff_lr_platt_frozen_split_20260516"
        )
        result = {
            "name": name,
            "model_version": model_version,
            "feature_count": len(features),
            "feature_schema_version": feature_schema_version(features),
            "added_features": list(additions),
            "feature_order": features,
            "metrics": metrics,
            "layoff_bucket_metrics": _bucket_metrics(test, y_test, probabilities),
            "layoff_coefficients": _coefficients(pipeline, list(additions)),
        }
        results[name] = result
        if name != "baseline":
            artifact = {
                "pipeline": pipeline,
                "platt": platt,
                "base_numeric": features,
                "config": {
                    **config,
                    "model_version": model_version,
                    "feature_schema_version": feature_schema_version(features),
                    "base_feature_set": f"Step 3C + {name} layoff features",
                    "raw_weight_class": "dropped",
                },
                "model_version": model_version,
                "calibration_version": f"platt_official_calib{len(calibration)}",
                "feature_schema_version": feature_schema_version(features),
                "training_metadata": {
                    "training_source": input_csv,
                    "train_rows": len(train),
                    "calibration_rows": len(calibration),
                    **split_metadata,
                },
            }
            artifact_path = models_dir / f"{name}_frozen_model.joblib"
            joblib.dump(artifact, artifact_path)
            result["artifact_path"] = str(artifact_path)

    baseline_metrics = results["baseline"]["metrics"]
    for name in ("candidate_a", "candidate_b"):
        metrics = results[name]["metrics"]
        results[name]["delta_vs_baseline"] = {
            "log_loss": metrics["log_loss"] - baseline_metrics["log_loss"],
            "brier_score": metrics["brier_score"] - baseline_metrics["brier_score"],
            "roc_auc": metrics["roc_auc"] - baseline_metrics["roc_auc"],
            "accuracy": metrics["accuracy"] - baseline_metrics["accuracy"],
            "expected_calibration_error": (
                metrics["expected_calibration_error"] - baseline_metrics["expected_calibration_error"]
            ),
        }

    best_name = min(("candidate_a", "candidate_b"), key=lambda n: results[n]["metrics"]["log_loss"])
    best_delta = results[best_name]["delta_vs_baseline"]["log_loss"]
    if best_delta <= -NOISE_BAND:
        recommendation = f"Promote {best_name.replace('_', ' ').title()} only after validation-fold confirmation"
    else:
        recommendation = "Keep disabled pending more data"
    report = {
        "run": "layoff_feature_ablation",
        "input": input_csv,
        "data_source_scope": "stored decisive UFC fights only; not complete professional MMA history",
        "official_allowlist_changed": False,
        "production_model_changed": False,
        "split": {
            "train_rows": len(train), "calibration_rows": len(calibration), "test_rows": len(test),
            "train_dates": [str(train["date"].min()), str(train["date"].max())],
            "calibration_dates": [str(calibration["date"].min()), str(calibration["date"].max())],
            "test_dates": [str(test["date"].min()), str(test["date"].max())],
            **split_metadata,
        },
        "noise_band_log_loss": NOISE_BAND,
        "results": results,
        "recommendation": recommendation,
    }
    prediction_path = out_dir / "layoff_feature_benchmark_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)
    report["benchmark_predictions"] = str(prediction_path)
    report_path.write_text(json.dumps(report, indent=2, default=_json_default) + "\n")
    md_path = out_dir / "layoff_feature_ablation.md"
    md_path.write_text(render_markdown(report))
    report["report_json"] = str(report_path)
    report["report_md"] = str(md_path)
    return report


def render_markdown(report: dict) -> str:
    lines = [
        "# Layoff Feature Ablation", "",
        "Controlled reproduction of the official LR + Platt model on the locked chronological split.", "",
        f"- Data scope: {report['data_source_scope']}",
        f"- Split: {report['split']['train_rows']} train / {report['split']['calibration_rows']} calibration / "
        f"{report['split']['test_rows']} benchmark",
        f"- Recommendation: **{report['recommendation']}**", "",
        "| Model | Features | Log loss | Δ LL | Brier | Δ Brier | AUC | Accuracy | ECE | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in report["results"].items():
        m = result["metrics"]
        d = result.get("delta_vs_baseline", {})
        lines.append(
            f"| {name} | {result['feature_count']} | {m['log_loss']:.6f} | "
            f"{d.get('log_loss', 0):+.6f} | {m['brier_score']:.6f} | "
            f"{d.get('brier_score', 0):+.6f} | {m['roc_auc']:.6f} | "
            f"{m['accuracy']:.6f} | {m['expected_calibration_error']:.6f} | {m['n']} |"
        )
    lines.extend(["", "## Added features", ""])
    for name in ("candidate_a", "candidate_b"):
        lines.append(f"- **{name}:** {', '.join(report['results'][name]['added_features'])}")
    lines.extend(["", "## Layoff buckets", ""])
    for name, result in report["results"].items():
        lines.extend([f"### {name}", "", "| Bucket | n | Adequate | Log loss | Brier | AUC | Accuracy | ECE |",
                      "|---|---:|:---:|---:|---:|---:|---:|---:|"])
        for row in result["layoff_bucket_metrics"]:
            if not row["n"]:
                continue
            auc = "n/a" if row.get("roc_auc") is None else f"{row['roc_auc']:.4f}"
            lines.append(
                f"| {row['bucket']} | {row['n']} | {row['adequate_sample']} | {row['log_loss']:.4f} | "
                f"{row['brier_score']:.4f} | {auc} | {row['accuracy']:.4f} | "
                f"{row['expected_calibration_error']:.4f} |"
            )
        lines.append("")
    lines.extend([
        "## Safety", "",
        "The official feature allowlist, benchmark JSON, production model, processed files, DB, and live ledger were not changed.",
        "Candidate artifacts are versioned experiment outputs and include their exact feature order and train-fitted imputer.", "",
    ])
    return "\n".join(lines)
