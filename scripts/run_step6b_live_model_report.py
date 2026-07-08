#!/usr/bin/env python3
"""Step 6B: build the live model report from the forward live ledger.

Aggregates resolved live predictions into overall/calibration/by-event/rolling
metrics and compares to the official static backtest and Step 6A pseudo-live
reference. Loudly warns when the resolved sample is too small to interpret
(< 50 too noisy, < 100 preliminary). Reads the ledger only.

Example:
  python scripts/run_step6b_live_model_report.py --ledger data/live/live_predictions.csv --output-dir reports/live
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.step6b_live_predictions import DEFAULT_LEDGER, DEFAULT_LIVE_REPORT_DIR, build_live_model_report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ledger", default=DEFAULT_LEDGER)
    p.add_argument("--output-dir", default=DEFAULT_LIVE_REPORT_DIR)
    p.add_argument("--baseline-path", default="benchmarks/official_baseline.json")
    p.add_argument("--step6a-report", default="reports/step6a_pseudo_live_replay.json")
    args = p.parse_args()

    report = build_live_model_report(
        ledger_path=args.ledger, output_dir=args.output_dir,
        baseline_path=args.baseline_path, step6a_report_path=args.step6a_report,
    )
    sw = report["sample_size_warning"]
    print(f"Live model report: {report['resolved_predictions']} resolved / {report['total_predictions']} total "
          f"({report['pending_predictions']} pending). Sample: {sw['level']}.")
    if report.get("overall_metrics"):
        o = report["overall_metrics"]
        print(f"  log loss={o['log_loss']:.6f}  Brier={o['brier_score']:.6f}  acc={o['accuracy']:.6f}")
    print(f"  report: {args.output_dir}/live_model_report.{{json,md}}")


if __name__ == "__main__":
    main()
