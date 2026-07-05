#!/usr/bin/env python3
"""Train and evaluate prediction models on the pre-fight feature dataset.

Usage (defaults just work):
  python scripts/train_model.py --input data/processed/ufc_prefight_features.csv

Full form:
  python scripts/train_model.py \
      --input data/processed/ufc_prefight_features.csv \
      --model-output-dir data/models \
      --predictions-output data/processed/model_predictions.csv \
      --metrics-output data/processed/model_evaluation.json \
      [--split-date 2022-01-01] [--test-size 0.2] \
      [--include-random-forest] [--no-categorical] \
      [--features elo_diff age_diff reach_diff ...]
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.modeling import train_and_evaluate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/processed/ufc_prefight_features.csv")
    p.add_argument("--model-output-dir", default="data/models")
    p.add_argument("--predictions-output", default="data/processed/model_predictions.csv")
    p.add_argument("--metrics-output", default="data/processed/model_evaluation.json")
    p.add_argument("--coefficients-output",
                   default="data/processed/logistic_regression_coefficients.csv")
    p.add_argument("--rf-importances-output",
                   default="data/processed/random_forest_feature_importances.csv")
    p.add_argument("--split-date", default=None,
                   help="YYYY-MM-DD; train on fights before, test on/after. "
                        "Default: 80/20 chronological split.")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--include-random-forest", action="store_true")
    p.add_argument("--no-categorical", action="store_true",
                   help="Exclude weight_class one-hot features.")
    p.add_argument("--features", nargs="+", default=None,
                   help="Explicit feature list (validated by the leakage guard).")
    args = p.parse_args()

    train_and_evaluate(
        input_csv=args.input,
        model_output_dir=args.model_output_dir,
        predictions_output=args.predictions_output,
        metrics_output=args.metrics_output,
        coefficients_output=args.coefficients_output,
        rf_importances_output=args.rf_importances_output,
        split_date=args.split_date,
        test_size=args.test_size,
        include_random_forest=args.include_random_forest,
        use_categorical=not args.no_categorical,
        features=args.features,
    )


if __name__ == "__main__":
    main()
