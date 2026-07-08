#!/usr/bin/env python3
"""Run Step 5D / Run 4: nonlinear-model + calibrated-blend comparison.

Controlled nonlinear-model comparison against the current official
stronger-regularized LR. It does not alter the official benchmark and writes
reports under --output-dir.

Writes:
  <output-dir>/step5d_nonlinear_models.json
  <output-dir>/step5d_nonlinear_models.md
  <output-dir>/step5d_candidate_results.csv
  <output-dir>/step5d_feature_importance.csv   (unless --skip-slow)
  <output-dir>/step5d_blend_results.csv         (unless --skip-blend)
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.step5d_nonlinear_models import RANDOM_STATE, run_nonlinear_models


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
    p.add_argument("--run3-report", default="reports/step5c_division_interactions.json")
    p.add_argument("--quick", action="store_true", help="Small HGB smoke grid (fast, not for promotion).")
    p.add_argument("--skip-slow", action="store_true", help="Skip permutation importance.")
    p.add_argument("--skip-blend", action="store_true", help="Skip the LR + nonlinear blend.")
    p.add_argument("--skip-optional-dependencies", action="store_true",
                   help="Force-skip XGBoost/LightGBM even if importable.")
    p.add_argument("--no-weight-class-sensitivity", action="store_true",
                   help="Drop the single weight_class-included HGB sensitivity candidate.")
    p.add_argument("--include-rf-sanity", action="store_true",
                   help="Add a non-promotable random-forest sanity candidate.")
    p.add_argument("--max-candidates", type=int, default=None,
                   help="Deterministic cap for debug/CI runs; production evaluates the full grid.")
    args = p.parse_args()

    run_nonlinear_models(
        input_csv=args.input,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        train_frac=args.train_frac,
        calibration_frac=args.calibration_frac,
        baseline_path=args.baseline_path,
        run1_report_path=args.run1_report,
        run2_report_path=args.run2_report,
        run21_report_path=args.run21_report,
        run3_report_path=args.run3_report,
        quick=args.quick,
        skip_slow=args.skip_slow,
        skip_blend=args.skip_blend,
        skip_optional_dependencies=args.skip_optional_dependencies,
        include_weight_class_sensitivity=not args.no_weight_class_sensitivity,
        include_rf_sanity=args.include_rf_sanity,
        max_candidates=args.max_candidates,
    )
    print(f"Step 5D nonlinear-model reports written to {args.output_dir}/")


if __name__ == "__main__":
    main()
