#!/usr/bin/env python3
"""Run Step 5D.1 / Run 4.1: nonlinear sanity audit.

Diagnostics-only audit of the Run 4 conclusion (did HGB truly lose, or was
there a bug / orientation flip / row mismatch / calibration issue / missing-
value issue / overly narrow grid?). It promotes nothing and never edits
benchmarks/official_baseline.json.

Writes:
  <output-dir>/step5d1_nonlinear_sanity_audit.json
  <output-dir>/step5d1_nonlinear_sanity_audit.md
  <output-dir>/step5d1_hgb_sensitivity_results.csv
  <output-dir>/step5d1_calibration_audit.csv
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.step5d1_nonlinear_sanity_audit import RANDOM_STATE, run_sanity_audit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/processed/ufc_prefight_features_step3c.csv")
    p.add_argument("--output-dir", default="reports")
    p.add_argument("--random-seed", type=int, default=RANDOM_STATE)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--calibration-frac", type=float, default=0.15)
    p.add_argument("--baseline-path", default="benchmarks/official_baseline.json")
    p.add_argument("--run4-report", default="reports/step5d_nonlinear_models.json")
    p.add_argument("--quick", action="store_true", help="Small staged grid (fast smoke run).")
    p.add_argument("--skip-slow", action="store_true", help="Skip permutation importance.")
    p.add_argument("--max-candidates", type=int, default=None,
                   help="Deterministic cap on coarse/refine grid sizes for debug/CI.")
    args = p.parse_args()

    run_sanity_audit(
        input_csv=args.input,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        train_frac=args.train_frac,
        calibration_frac=args.calibration_frac,
        baseline_path=args.baseline_path,
        run4_report_path=args.run4_report,
        quick=args.quick,
        skip_slow=args.skip_slow,
        max_candidates=args.max_candidates,
    )
    print(f"Step 5D.1 nonlinear sanity-audit reports written to {args.output_dir}/")


if __name__ == "__main__":
    main()
