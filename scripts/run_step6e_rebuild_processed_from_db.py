#!/usr/bin/env python3
"""Step 6E: controlled processed-feature rebuild from the updated database.

Regenerates the Step 3 / 3B / 3C processed feature CSVs from ``data/ufc.db``
into a SEPARATE output directory, verifies them against the current official
Step 3C file, and writes a JSON + Markdown report. It never overwrites the
official processed files, never edits ``benchmarks/official_baseline.json``,
never retrains/promotes a model, and never writes to ``data/ufc.db``.

Usage:
  python scripts/run_step6e_rebuild_processed_from_db.py \
      --db data/ufc.db \
      --output-dir data/processed/rebuild_step6e_20260709 \
      --reports-dir reports/data_update/step6e \
      --no-overwrite-official
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.step6e_rebuild_processed import (
    OFFICIAL_STEP3C_PATH,
    default_output_dir,
    run_rebuild,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="data/ufc.db", help="SQLite database path (read-only here)")
    p.add_argument("--output-dir", default=None,
                   help="Directory for rebuilt CSVs (default: data/processed/rebuild_step6e_<UTC-date>)")
    p.add_argument("--reports-dir", default="reports/data_update/step6e")
    p.add_argument("--official-step3c", default=OFFICIAL_STEP3C_PATH,
                   help="Existing official Step 3C file to compare against (read-only)")
    # The official-file guard is absolute regardless of this flag; the flag is
    # recorded for provenance. --no-overwrite-official is the default; passing
    # --overwrite-official does NOT disable the hard guard (it only annotates intent).
    p.add_argument("--overwrite-official", dest="allow_overwrite_official",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Provenance flag only; --no-overwrite-official is the default and "
                        "the guard against overwriting official processed files is always enforced")
    p.add_argument("--overwrite-rebuild", action="store_true",
                   help="Allow replacing pre-existing files in the rebuild output dir")
    p.add_argument("--skip-compat", action="store_true",
                   help="Skip the optional official-model structural compatibility check")
    args = p.parse_args()

    report = run_rebuild(
        db_path=args.db,
        output_dir=args.output_dir or default_output_dir(),
        reports_dir=args.reports_dir,
        official_step3c=args.official_step3c,
        no_overwrite_official=not args.allow_overwrite_official,
        overwrite_rebuild=args.overwrite_rebuild,
        run_compat=not args.skip_compat,
    )

    s = report["schema_comparison"]
    nev = report["new_event_verification"]
    print("Step 6E processed-feature rebuild (non-destructive):")
    print(f"  db checksum: {report['db_checksum_used']} | DB unchanged: {report['db_unchanged_by_rebuild']}")
    print(f"  output dir: {report['rebuild_output_dir']}")
    print(f"  rows: old {s['old_row_count']} -> new {s['new_row_count']} (Δ {s['row_count_delta']:+d}) | "
          f"max date {s['old_max_date']} -> {s['new_max_date']}")
    print(f"  schema identical & ordered: {s['columns_identical_and_ordered']} | "
          f"all 43 features present: {s['all_official_features_present']}")
    print(f"  expected events present: {nev['all_expected_events_present']} | "
          f"future events absent: {nev['all_excluded_events_absent']}")
    print(f"  step6b input accepted: {report['step6b_input_validation']['ok']} | "
          f"model compat: {report['official_model_compatibility'].get('status')}")
    print(f"  overall_ok: {report['overall_ok']}")
    print(f"  report: {report['report_md']}")
    print("  Official processed files NOT overwritten; benchmark/model NOT modified.")


if __name__ == "__main__":
    main()
