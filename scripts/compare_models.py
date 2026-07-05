#!/usr/bin/env python3
"""Compare Elo baseline vs basic Step-3 model vs Step-3+3B model
on the same chronological test split.

Usage:
  python scripts/compare_models.py --input data/processed/ufc_prefight_features_step3b.csv \
      [--include-random-forest] [--split-date YYYY-MM-DD] [--test-size 0.2]
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.modeling import compare_feature_sets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/processed/ufc_prefight_features_step3b.csv")
    p.add_argument("--split-date", default=None)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--include-random-forest", action="store_true")
    p.add_argument("--metrics-output", default="data/processed/model_comparison.json")
    p.add_argument("--predictions-output",
                   default="data/processed/model_comparison_predictions.csv")
    args = p.parse_args()

    compare_feature_sets(
        input_csv=args.input,
        split_date=args.split_date,
        test_size=args.test_size,
        include_random_forest=args.include_random_forest,
        metrics_output=args.metrics_output,
        predictions_output=args.predictions_output,
    )


if __name__ == "__main__":
    main()
