#!/usr/bin/env python3
"""Calibrate the best Step 3B logistic regression. Two modes:

backtest (default) — honest final-test evaluation. Chronological 3-way
  split: base model fits on the oldest window, Platt + isotonic calibrators
  fit on the middle window, ALL metrics come from the newest window
  (nothing is ever fit there).

live — current model-building workflow for FUTURE predictions. The base
  model trains on all fights before a rolling recent window (default 730
  days ending at --as-of-date or the latest fight); the calibrator (Platt
  by default) is refit on that window; base model + calibrator are saved
  for future use. Live mode reports calibration-window diagnostics only —
  they are NOT final-test metrics.

Both modes are chronological, never random.

Usage:
  python scripts/calibrate_model.py \
      --input data/processed/ufc_prefight_features_step3b.csv \
      --output-dir data/processed --model-dir data/models
  # backtest date mode:  --train-end-date 2021-01-01 --calibration-end-date 2023-01-01
  # live mode:           --mode live --calibration-window-days 730 --method platt
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.calibration import (
    DEFAULT_CALIBRATION_WINDOW_DAYS,
    DEFAULT_MIN_CALIBRATION_ROWS,
    run_calibration,
    run_live_calibration,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/processed/ufc_prefight_features_step3b.csv")
    p.add_argument("--output-dir", default="data/processed")
    p.add_argument("--model-dir", default="data/models")
    p.add_argument("--mode", choices=["backtest", "live"], default="backtest",
                   help="backtest = honest held-out evaluation (default); "
                        "live = rolling recent-window refit for future predictions")
    # backtest-mode options
    p.add_argument("--train-frac", type=float, default=0.70,
                   help="[backtest] chronological fraction for the train window")
    p.add_argument("--calibration-frac", type=float, default=0.15,
                   help="[backtest] chronological fraction for the calibrator window")
    p.add_argument("--train-end-date", default=None,
                   help="[backtest] date mode: train = fights before this date "
                        "(requires --calibration-end-date)")
    p.add_argument("--calibration-end-date", default=None,
                   help="[backtest] date mode: calibration = fights before this "
                        "date and on/after --train-end-date; test = on/after it")
    p.add_argument("--no-step3-basic", action="store_true",
                   help="[backtest] skip the uncalibrated basic Step 3 context model")
    # live-mode options
    p.add_argument("--calibration-window-days", type=int,
                   default=DEFAULT_CALIBRATION_WINDOW_DAYS,
                   help="[live] rolling window length in days (default 730)")
    p.add_argument("--min-calibration-rows", type=int,
                   default=DEFAULT_MIN_CALIBRATION_ROWS,
                   help="[live] warn when the window has fewer rows than this")
    p.add_argument("--as-of-date", default=None,
                   help="[live] treat this date as 'today'; fights after it are "
                        "excluded everywhere (default: latest fight in the data)")
    p.add_argument("--method", choices=["platt", "isotonic"], default="platt",
                   help="[live] calibration method (default platt)")
    args = p.parse_args()

    if args.mode == "live":
        run_live_calibration(
            input_csv=args.input,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            calibration_window_days=args.calibration_window_days,
            min_calibration_rows=args.min_calibration_rows,
            as_of_date=args.as_of_date,
            method=args.method,
        )
    else:
        run_calibration(
            input_csv=args.input,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            train_frac=args.train_frac,
            calibration_frac=args.calibration_frac,
            train_end_date=args.train_end_date,
            calibration_end_date=args.calibration_end_date,
            include_step3_basic=not args.no_step3_basic,
        )


if __name__ == "__main__":
    main()
