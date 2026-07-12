#!/usr/bin/env python3
"""Run the versioned frozen-split layoff feature ablation."""

import argparse

import _path  # noqa: F401
from ufc_pipeline.layoff_feature_evaluation import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    run_layoff_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--baseline-path", default="benchmarks/official_baseline.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = run_layoff_evaluation(
        input_csv=args.input, output_dir=args.output_dir, artifact_dir=args.artifact_dir,
        baseline_path=args.baseline_path, overwrite=args.overwrite,
    )
    for name, result in report["results"].items():
        metrics = result["metrics"]
        delta = result.get("delta_vs_baseline", {}).get("log_loss", 0.0)
        print(f"{name}: LL={metrics['log_loss']:.6f} (delta {delta:+.6f}), "
              f"Brier={metrics['brier_score']:.6f}, AUC={metrics['roc_auc']:.6f}, "
              f"accuracy={metrics['accuracy']:.6f}, ECE={metrics['expected_calibration_error']:.6f}")
    print(f"Recommendation: {report['recommendation']}")
    print(f"Report: {report['report_md']}")


if __name__ == "__main__":
    main()
