#!/usr/bin/env python3
"""Step 5B.1: regularization confirmation check.

LR-only confirmation pass for the pre-declared Step 5B candidates. This is
not a new hyperparameter search and does not change the official model or
benchmarks/official_baseline.json.

Writes:
  <output-dir>/step5b1_regularization_confirmation.json
  <output-dir>/step5b1_regularization_confirmation.md

Usage:
  python3 scripts/run_step5b1_regularization_confirmation.py --output-dir reports
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.step5b1_regularization_confirmation import (
    RANDOM_STATE,
    run_regularization_confirmation,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/processed/ufc_prefight_features_step3c.csv")
    p.add_argument("--output-dir", default="reports")
    p.add_argument("--random-seed", type=int, default=RANDOM_STATE)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--calibration-frac", type=float, default=0.15)
    p.add_argument("--baseline-path", default="benchmarks/official_baseline.json")
    p.add_argument("--step5b-report", default="reports/step5b_regularization_search.json")
    args = p.parse_args()

    run_regularization_confirmation(
        input_csv=args.input,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        train_frac=args.train_frac,
        calibration_frac=args.calibration_frac,
        baseline_path=args.baseline_path,
        step5b_report_path=args.step5b_report,
    )
    print(f"Step 5B.1 regularization confirmation written to {args.output_dir}/")


if __name__ == "__main__":
    main()
