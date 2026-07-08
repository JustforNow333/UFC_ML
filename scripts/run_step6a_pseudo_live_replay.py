#!/usr/bin/env python3
"""Run Step 6A: historical pseudo-live replay of the official model.

Event-by-event replay: for each past UFC event, re-fit the official LR + Platt
on strictly-earlier fights only and predict the event, then score it. This is a
pipeline-validation and calibration/drift tool, not a new model search. It never
modifies data/processed/ or benchmarks/official_baseline.json.

Writes:
  <output-dir>/step6a_pseudo_live_replay.json
  <output-dir>/step6a_pseudo_live_replay.md
  <output-dir>/step6a_rolling_metrics.csv
  reports/replay/step6a_pseudo_live_predictions.csv   (ledger; --output-ledger to change)

Examples:
  python scripts/run_step6a_pseudo_live_replay.py --input data/processed/ufc_prefight_features_step3c.csv \
      --output-dir reports --min-train-fights 3000 --calibration-fights 1282 --overwrite
  python scripts/run_step6a_pseudo_live_replay.py --max-events 5 --overwrite   # quick smoke
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.step6a_pseudo_live_replay import RANDOM_STATE, REPLAY_LEDGER_DEFAULT, run_pseudo_live_replay


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/processed/ufc_prefight_features_step3c.csv")
    p.add_argument("--output-dir", default="reports")
    p.add_argument("--output-ledger", default=REPLAY_LEDGER_DEFAULT,
                   help="Replay ledger CSV path (default under reports/replay/, never data/processed/).")
    p.add_argument("--baseline-path", default="benchmarks/official_baseline.json")
    p.add_argument("--min-train-fights", type=int, default=3000)
    p.add_argument("--calibration-fights", type=int, default=1282)
    p.add_argument("--start-date", default=None, help="YYYY-MM-DD; only replay events on/after this date.")
    p.add_argument("--end-date", default=None, help="YYYY-MM-DD; only replay events on/before this date.")
    p.add_argument("--max-events", type=int, default=None, help="Cap number of replayed events (quick smoke runs).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite an existing ledger.")
    p.add_argument("--random-seed", type=int, default=RANDOM_STATE)
    args = p.parse_args()

    report = run_pseudo_live_replay(
        input_csv=args.input,
        output_dir=args.output_dir,
        ledger_path=args.output_ledger,
        min_train_fights=args.min_train_fights,
        calibration_fights=args.calibration_fights,
        start_date=args.start_date,
        end_date=args.end_date,
        max_events=args.max_events,
        overwrite=args.overwrite,
        random_seed=args.random_seed,
        baseline_path=args.baseline_path,
    )
    o = report["overall_metrics"]
    print(f"Step 6A pseudo-live replay: {report['n_events_replayed']} events, {report['n_fights_predicted']} fights.")
    print(f"  overall log loss={o['log_loss']:.6f}  Brier={o['brier_score']:.6f}  AUC={o['roc_auc']:.6f}  acc={o['accuracy']:.6f}")
    print(f"  ledger: {report['ledger_path']}")
    print(f"  reports: {args.output_dir}/step6a_pseudo_live_replay.{{json,md}}")


if __name__ == "__main__":
    main()
