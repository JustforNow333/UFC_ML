#!/usr/bin/env python3
"""Step 5A: feature diagnostics for the current official Step 3C LR + Platt model.

Diagnostics-only: does not replace the official model, does not add
odds/market features, does not train new model families, does not run a
hyperparameter search. Reuses the same chronological 70/15/15 split, the
same elastic-net pipeline (C=0.003, l1_ratio=0.1, solver=saga, raw
weight_class dropped), and the same PlattCalibrator as the official run so
every number is directly comparable to benchmarks/official_baseline.json
(read-only).

Writes:
  <output-dir>/step5a_feature_diagnostics.json
  <output-dir>/step5a_feature_diagnostics.md
  <output-dir>/step5a_coefficients.csv

Usage:
  python scripts/run_step5a_feature_diagnostics.py
  python scripts/run_step5a_feature_diagnostics.py --skip-permutation
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.feature_diagnostics import run_feature_diagnostics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/processed/ufc_prefight_features_step3c.csv")
    p.add_argument("--output-dir", default="reports")
    p.add_argument("--permutation-repeats", type=int, default=10)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--calibration-frac", type=float, default=0.15)
    p.add_argument("--skip-permutation", action="store_true",
                   help="Skip the permutation-importance step (faster).")
    args = p.parse_args()

    run_feature_diagnostics(
        input_csv=args.input,
        output_dir=args.output_dir,
        n_permutation_repeats=args.permutation_repeats,
        train_frac=args.train_frac,
        calibration_frac=args.calibration_frac,
        skip_permutation=args.skip_permutation,
    )
    print(f"Step 5A diagnostics written to {args.output_dir}/")


if __name__ == "__main__":
    main()
