#!/usr/bin/env python3
"""Step 6F: guarded promotion of Step 6E rebuilt processed files to official.

Dry-run by default (reports the plan + validation, touches nothing). Pass
``--apply`` to back up the current official processed files and copy the
verified rebuilt files into the official ``data/processed/`` paths. It never
retrains/promotes a model, never edits ``benchmarks/official_baseline.json``,
and never touches ``data/ufc.db``.

Usage:
  # dry-run (default): inspect + validate, no changes
  python scripts/run_step6f_promote_processed.py \
      --source-dir data/processed/rebuild_step6e_20260709

  # apply: backup then promote
  python scripts/run_step6f_promote_processed.py \
      --source-dir data/processed/rebuild_step6e_20260709 --apply
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.step6f_promote_processed import DEFAULT_SOURCE_DIR, run_promotion


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR,
                   help="Rebuilt processed dir to promote from (Step 6E output)")
    p.add_argument("--reports-dir", default="reports/data_update")
    p.add_argument("--apply", action="store_true",
                   help="Actually back up + promote (default is a dry run)")
    args = p.parse_args()

    report = run_promotion(
        source_dir=args.source_dir,
        apply=args.apply,
        reports_dir=args.reports_dir,
    )

    print(f"Step 6F guarded promotion ({report['mode']}): status = {report['status']}")
    print(f"  source: {report['source_rebuild_dir']}")
    val = report["source_validation"]
    print(f"  source validation ok: {val['ok']} "
          f"({sum(bool(v) for v in val['checks'].values())}/{len(val['checks'])} checks)")
    post = report.get("post_promotion_verification")
    if post is not None:
        print(f"  post-promotion verification ok: {post['ok']} | "
              f"official Step 3C rows {post['official_step3c_row_count']} "
              f"(max date {post['official_step3c_max_date']})")
        print(f"  DB unchanged: {post['checks']['db_unchanged']} | "
              f"benchmark unchanged: {post['checks']['benchmark_unchanged']}")
    if report.get("backup"):
        print(f"  backup dir: {report['backup'].get('backup_dir')}")
    print(f"  official files overwritten: {report['official_processed_files_overwritten']}")
    print(f"  report: {report.get('report_md')}")
    if report["status"] == "aborted_validation_failed":
        print("  VALIDATION FAILED — official files untouched. Investigate before retrying.")
    elif not args.apply and report["status"] == "dry_run":
        print("  Dry run only. Re-run with --apply to back up and promote.")


if __name__ == "__main__":
    main()
