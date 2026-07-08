#!/usr/bin/env python3
"""Step 6B: generate forward live predictions for an upcoming UFC card.

Reads an upcoming-card pre-fight feature CSV (same feature columns as
data/processed/ufc_prefight_features_step3c.csv, minus result/target columns),
generates official-model calibrated predictions, appends them to the durable
live ledger as `pending`, and writes a batch report. It never modifies
data/processed/ or benchmarks/official_baseline.json and adds no odds/markets.

Example:
  python scripts/run_step6b_live_predictions.py --input upcoming_card_features.csv \
      --ledger data/live/live_predictions.csv --output-dir reports/live
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.step6b_live_predictions import (
    DEFAULT_LEDGER,
    DEFAULT_LIVE_REPORT_DIR,
    DEFAULT_TRAINING_DATA,
    RANDOM_STATE,
    run_live_predictions,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Upcoming-card pre-fight feature CSV.")
    p.add_argument("--ledger", default=DEFAULT_LEDGER)
    p.add_argument("--output-dir", default=DEFAULT_LIVE_REPORT_DIR)
    p.add_argument("--training-data", default=DEFAULT_TRAINING_DATA,
                   help="Historical processed CSV used to reproduce the official model (read-only).")
    p.add_argument("--baseline-path", default="benchmarks/official_baseline.json")
    p.add_argument("--prediction-batch-id", default=None)
    p.add_argument("--overwrite-existing-pending", action="store_true", default=False,
                   help="Replace an existing PENDING prediction for the same fight.")
    p.add_argument("--allow-duplicate-predictions", action="store_true", default=False,
                   help="Append a new versioned prediction row for a fight already in the ledger.")
    p.add_argument("--random-seed", type=int, default=RANDOM_STATE)
    args = p.parse_args()

    report = run_live_predictions(
        input_csv=args.input, ledger_path=args.ledger, output_dir=args.output_dir,
        training_data=args.training_data, baseline_path=args.baseline_path,
        prediction_batch_id=args.prediction_batch_id,
        allow_duplicate_predictions=args.allow_duplicate_predictions,
        overwrite_existing_pending=args.overwrite_existing_pending,
        random_seed=args.random_seed,
    )
    print(f"Live predictions: batch {report['prediction_batch_id']} — {report['n_predictions_generated']} predictions "
          f"({report['n_failed_rows']} failed rows).")
    print(f"  ledger: {report['ledger_path']}")
    print(f"  batch report: {report['batch_report_md']}")


if __name__ == "__main__":
    main()
