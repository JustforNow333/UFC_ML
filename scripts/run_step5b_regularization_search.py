#!/usr/bin/env python3
"""Step 5B: LR regularization + weight_class feature-pruning search (Run 2).

LR-only: no new model families, no odds/market data, no new interaction
features, no nonlinear models. Selects candidates using ONLY rolling-origin
validation folds inside the pre-test period, then confirms finalists ONCE
on the held-out official test split. Does not change the official model or
benchmarks/official_baseline.json; writes only to --output-dir.

Writes:
  <output-dir>/step5b_regularization_search.json
  <output-dir>/step5b_regularization_search.md
  <output-dir>/step5b_coefficients.csv
  <output-dir>/step5b_candidate_results.csv

Usage:
  python scripts/run_step5b_regularization_search.py
  python scripts/run_step5b_regularization_search.py --quick
  python scripts/run_step5b_regularization_search.py --max-candidates 40 --skip-balanced
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.step5b_regularization_search import (
    DEFAULT_DROP_RARE_THRESHOLDS,
    RANDOM_STATE,
    run_regularization_search,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/processed/ufc_prefight_features_step3c.csv")
    p.add_argument("--output-dir", default="reports")
    p.add_argument("--random-seed", type=int, default=RANDOM_STATE)
    p.add_argument("--max-candidates", type=int, default=None,
                   help="Cap the number of (feature_config x penalty x C[/l1_ratio]) "
                        "candidates evaluated, in deterministic grid order (debug/CI use).")
    p.add_argument("--skip-balanced", action="store_true",
                   help="Skip the class_weight='balanced' secondary check.")
    p.add_argument("--drop-rare-thresholds", type=int, nargs="+", default=list(DEFAULT_DROP_RARE_THRESHOLDS))
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--calibration-frac", type=float, default=0.15)
    p.add_argument("--quick", action="store_true",
                   help="Small grid (2 C values, 2 l1_ratio values) for a fast smoke run.")
    args = p.parse_args()

    c_grid = [0.1, 1.0] if args.quick else None
    l1_ratio_grid = [0.3, 0.7] if args.quick else None

    run_regularization_search(
        input_csv=args.input,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        max_candidates=args.max_candidates,
        skip_balanced=args.skip_balanced,
        drop_rare_thresholds=tuple(args.drop_rare_thresholds),
        train_frac=args.train_frac,
        calibration_frac=args.calibration_frac,
        c_grid=c_grid,
        l1_ratio_grid=l1_ratio_grid,
    )
    print(f"Step 5B regularization search written to {args.output_dir}/")


if __name__ == "__main__":
    main()
