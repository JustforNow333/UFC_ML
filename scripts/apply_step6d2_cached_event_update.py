#!/usr/bin/env python3
"""Step 6D.2: guarded insert-only apply for one cached UFCStats event.

Dry-run is the default. The script never scrapes UFCStats and never broadens to
all missing events. To write to the DB, pass --apply explicitly; a timestamped
backup is created before any insert.

Example dry-run:
  python scripts/apply_step6d2_cached_event_update.py \
      --db data/ufc.db \
      --event-cache data/cache/ufcstats/event_31e1ea6fe6b682f8.html \
      --output-dir reports/data_update

Example apply:
  python scripts/apply_step6d2_cached_event_update.py \
      --db data/ufc.db \
      --event-cache data/cache/ufcstats/event_31e1ea6fe6b682f8.html \
      --output-dir reports/data_update \
      --apply
"""

import argparse

import _path  # noqa: F401
from ufc_pipeline.step6d2_guarded_db_apply import run_cached_event_update


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="data/ufc.db")
    p.add_argument("--event-cache", required=True)
    p.add_argument("--output-dir", default="reports/data_update")
    p.add_argument("--apply", action="store_true", default=False)
    p.add_argument(
        "--skip-step6c-verification",
        action="store_true",
        default=False,
        help="When applying, skip the post-apply Step 6C smoke build.",
    )
    args = p.parse_args()

    report = run_cached_event_update(
        db_path=args.db,
        event_cache=args.event_cache,
        output_dir=args.output_dir,
        apply=args.apply,
        run_step6c_verification=not args.skip_step6c_verification,
    )

    print("Step 6D.2 guarded cached-event update:")
    print(f"  mode: {report['mode']} | safe_to_apply: {report['safe_to_apply']}")
    print(
        f"  event: {report['event']['event_date']} - {report['event']['event_name']} "
        f"({report['event']['event_id']})"
    )
    print(
        "  planned rows: "
        f"fighters={report['planned_row_counts']['fighters']}, "
        f"fighter_source_ids={report['planned_row_counts']['fighter_source_ids']}, "
        f"fights={report['planned_row_counts']['fights']}, "
        f"fight_stats={report['planned_row_counts']['fight_stats']}, "
        f"elo_snapshots={report['planned_row_counts']['fight_elo_snapshots']}"
    )
    print(
        "  skipped/blocked: "
        f"skipped_fights={report['planned_row_counts']['skipped_fights']}, "
        f"blocked_fighters={report['planned_row_counts']['blocked_fighters']}, "
        f"blocked_fights={report['planned_row_counts']['blocked_fights']}"
    )
    if report["apply_result"]:
        ar = report["apply_result"]
        print(f"  apply_performed: {ar.get('apply_performed')}")
        if ar.get("backup_path"):
            print(f"  backup: {ar['backup_path']}")
        if ar.get("db_count_deltas"):
            print(f"  count deltas: {ar['db_count_deltas']}")
    else:
        print("  dry-run only: no DB writes performed")
    print(f"  report: {report['report_md']}")


if __name__ == "__main__":
    main()

