#!/usr/bin/env python3
"""Step 6B: resolve pending live predictions against a completed-results CSV.

The results CSV must have: event_date, event_name, fighter_a, fighter_b, winner.
Matching is exact on normalized (event_date, fighter set); the resolver fills
target_a_win/winner, computes log loss/Brier/correct, and marks rows resolved.
Already-resolved rows are preserved unless --overwrite-resolved is passed.

Example:
  python scripts/resolve_step6b_live_predictions.py --results completed_results.csv \
      --ledger data/live/live_predictions.csv --output-dir reports/live
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.step6b_live_predictions import DEFAULT_LEDGER, DEFAULT_LIVE_REPORT_DIR, run_resolution


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="Completed-results CSV.")
    p.add_argument("--ledger", default=DEFAULT_LEDGER)
    p.add_argument("--output-dir", default=DEFAULT_LIVE_REPORT_DIR)
    p.add_argument("--overwrite-resolved", action="store_true", default=False,
                   help="Re-resolve rows that are already resolved.")
    p.add_argument("--no-strict-matching", dest="strict_matching", action="store_false", default=True,
                   help="(reserved) relax exact matching; exact normalized matching is the default and only mode.")
    args = p.parse_args()

    report = run_resolution(
        results_csv=args.results, ledger_path=args.ledger, output_dir=args.output_dir,
        overwrite_resolved=args.overwrite_resolved, strict_matching=args.strict_matching,
    )
    s = report["resolution_stats"]
    print(f"Resolution: {s['n_resolved']} newly resolved; {report['n_pending_after']} still pending; "
          f"{s['n_unmatched_results']} unmatched results; {len(s['ambiguous_matches'])} ambiguous; "
          f"{s['n_already_resolved_skipped']} already-resolved skipped.")
    print(f"  ledger: {report['ledger_path']}")
    print(f"  resolution report: {report['resolution_report_json']}")


if __name__ == "__main__":
    main()
