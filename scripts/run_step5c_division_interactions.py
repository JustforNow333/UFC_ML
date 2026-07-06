#!/usr/bin/env python3
"""Run Step 5C / Run 3 division-interaction LR experiment.

This is a controlled LR-only feature-engineering run. It does not alter the
official benchmark and writes reports under --output-dir.

Writes:
  <output-dir>/step5c_division_interactions.json
  <output-dir>/step5c_division_interactions.md
  <output-dir>/step5c_candidate_results.csv
  <output-dir>/step5c_interaction_coefficients.csv
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.step5c_division_interactions import (
    MIN_DIVISION_COUNT,
    MIN_DIVISION_FRACTION,
    RANDOM_STATE,
    run_division_interactions,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/processed/ufc_prefight_features_step3c.csv")
    p.add_argument("--output-dir", default="reports")
    p.add_argument("--random-seed", type=int, default=RANDOM_STATE)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--calibration-frac", type=float, default=0.15)
    p.add_argument("--baseline-path", default="benchmarks/official_baseline.json")
    p.add_argument("--run1-report", default="reports/step5a_feature_diagnostics.json")
    p.add_argument("--run2-report", default="reports/step5b_regularization_search.json")
    p.add_argument("--run21-report", default="reports/step5b1_regularization_confirmation.json")
    p.add_argument("--min-division-count", type=int, default=MIN_DIVISION_COUNT)
    p.add_argument("--min-division-fraction", type=float, default=MIN_DIVISION_FRACTION)
    p.add_argument("--max-candidates", type=int, default=None,
                   help="Deterministic cap for debug/CI runs; production should evaluate all predeclared candidates.")
    args = p.parse_args()

    run_division_interactions(
        input_csv=args.input,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        train_frac=args.train_frac,
        calibration_frac=args.calibration_frac,
        baseline_path=args.baseline_path,
        run1_report_path=args.run1_report,
        run2_report_path=args.run2_report,
        run21_report_path=args.run21_report,
        min_division_count=args.min_division_count,
        min_division_fraction=args.min_division_fraction,
        max_candidates=args.max_candidates,
    )
    print(f"Step 5C division-interaction reports written to {args.output_dir}/")


if __name__ == "__main__":
    main()
