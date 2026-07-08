#!/usr/bin/env python3
"""Step 6C: build a model-ready upcoming-card feature CSV from a matchup CSV.

Input matchup CSV: event_date, event_name, fighter_a, fighter_b (+ optional
weight_class, scheduled_rounds, fight_order, notes). Output is the exact Step 3C
feature schema Step 6B expects, recomputed from fights strictly before each
event date using the existing history DB. No scraping, no odds/markets; never
writes data/processed/ or the benchmark file.

Example:
  python scripts/build_step6c_upcoming_features.py \
      --matchups data/live/upcoming_card_matchups.csv \
      --output data/live/upcoming_card_features.csv --validate-for-step6b --overwrite
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.step6c_upcoming_feature_builder import (
    DEFAULT_HISTORY_DB,
    DEFAULT_OUTPUT,
    DEFAULT_REPORT_DIR,
    run_build,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matchups", required=True, help="Upcoming-card matchup CSV.")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output feature CSV path.")
    p.add_argument("--history-input", default=DEFAULT_HISTORY_DB, help="Historical fight DB (default data/ufc.db).")
    p.add_argument("--output-dir", default=DEFAULT_REPORT_DIR, help="Directory for the build report.")
    p.add_argument("--event-date", default=None, help="Override event_date for all matchups (YYYY-MM-DD).")
    p.add_argument("--strict-name-match", dest="strict_name_match", action="store_true", default=True)
    p.add_argument("--allow-fuzzy-match", action="store_true", default=False,
                   help="(reserved) fuzzy matching is intentionally disabled; exact normalized matching only.")
    p.add_argument("--review-matches-output", default=None, help="Optional CSV of the name-match review table.")
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--no-validate-for-step6b", dest="validate_for_step6b", action="store_false", default=True)
    p.add_argument("--run-predictions", action="store_true", default=False,
                   help="After building features, chain straight into Step 6B live predictions.")
    p.add_argument("--ledger", default="data/live/live_predictions.csv", help="Ledger for --run-predictions.")
    args = p.parse_args()

    report = run_build(
        matchups_csv=args.matchups, output=args.output, history_input=args.history_input,
        output_dir=args.output_dir, event_date=args.event_date,
        strict_name_match=args.strict_name_match, allow_fuzzy_match=args.allow_fuzzy_match,
        review_matches_output=args.review_matches_output, overwrite=args.overwrite,
        validate_for_step6b=args.validate_for_step6b,
    )
    v = report["step6b_validation"]
    print(f"Step 6C: built {report['n_feature_rows_built']} feature rows "
          f"({report['n_failed_rows']} failed) -> {report['output_feature_path']}")
    print(f"  Step 6B validation: {'PASS' if v.get('ok') else v.get('ok')}")
    if report["unmatched_fighters"]:
        print(f"  unmatched fighters: {report['unmatched_fighters']}")
    if report["ambiguous_fighters"]:
        print(f"  ambiguous fighters: {report['ambiguous_fighters']}")
    print(f"  report: {report['report_md']}")
    print(f"  next: {report['next_command']}")

    if args.run_predictions and report["n_feature_rows_built"]:
        from ufc_pipeline.step6b_live_predictions import run_live_predictions
        pred = run_live_predictions(input_csv=args.output, ledger_path=args.ledger, output_dir=args.output_dir)
        print(f"  chained Step 6B: {pred['n_predictions_generated']} predictions -> {pred['ledger_path']}")


if __name__ == "__main__":
    main()
