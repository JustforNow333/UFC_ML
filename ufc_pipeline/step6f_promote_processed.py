#!/usr/bin/env python3
"""Step 6F: guarded promotion of Step 6E rebuilt processed files to official.

Purpose
-------
Step 6E rebuilt the Step 3 / 3B / 3C processed feature CSVs from the updated
``data/ufc.db`` into a separate directory and verified them. Step 6F promotes
those verified files into the official ``data/processed/`` paths — a **data
artifact** promotion only. It does NOT retrain, replace, or promote a model,
never edits ``benchmarks/official_baseline.json``, never touches
``data/ufc.db``, and never fetches UFCStats or adds odds/markets.

Guardrails (the whole point of this step)
-----------------------------------------
* **Explicit + dry-run by default.** ``run_promotion`` requires an explicit
  ``source_dir`` and only copies when ``apply=True``. Dry-run reports the full
  plan and validation without touching official files.
* **Validate before touching anything.** The rebuilt source is re-verified
  (schema parity vs the current official Step 3C, expected new events present
  with correct counts, future cards absent, no duplicate/leakage rows, no
  infinities, Step 6B input validation) *before* any backup or copy. If
  validation fails the official files are left untouched — no backup, no copy.
* **Backup before overwrite.** On apply, the current official files are copied
  into ``data/processed/backups/pre_step6f_promotion_<UTC-stamp>/`` **before**
  any official file is overwritten.
* **Verify after, auto-rollback on failure.** After copying, the now-official
  files are re-verified and DB/benchmark checksums confirmed unchanged. If
  post-promotion verification fails, the backup is restored automatically and
  the run is reported as rolled back.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ufc_pipeline.feature_diagnostics import official_step3c_features
from ufc_pipeline.step6e_rebuild_processed import (
    DEFAULT_EXCLUDED_EVENTS,
    DEFAULT_EXPECTED_NEW_EVENTS,
    OFFICIAL_PROCESSED_PATHS,
    REBUILD_BASENAMES,
    _json_default,
    _now_utc,
    check_step6b_validation,
    compare_schema,
    feature_quality_checks,
    file_md5,
    verify_new_events,
)

REPORT_STEM = "step6f_promotion_report"
BACKUP_ROOT = "data/processed/backups"
DEFAULT_SOURCE_DIR = "data/processed/rebuild_step6e_20260710_committed"
DB_PATH = "data/ufc.db"
BENCHMARK_PATH = "benchmarks/official_baseline.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _summarize_csv(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": str(path), "exists": False}
    df = pd.read_csv(p)
    return {
        "path": str(path),
        "exists": True,
        "md5": file_md5(p),
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "max_date": str(df["date"].max()) if "date" in df.columns else None,
    }


def dir_toplevel_checksum(dir_path: str | Path) -> str | None:
    """Combined md5 of the top-level files in a directory (subdirs ignored) —
    mirrors ``find <dir> -maxdepth 1 -type f -exec md5sum | sort | md5sum``."""
    d = Path(dir_path)
    if not d.exists():
        return None
    digests = sorted(f"{file_md5(f)}  {f.name}" for f in d.iterdir() if f.is_file())
    return hashlib.md5("\n".join(digests).encode()).hexdigest()


def resolve_promotion_files(source_dir: str | Path) -> list[dict]:
    """Map each official processed file to its rebuilt source, confirming exact
    filenames exist. Only the known processed feature CSVs are promoted."""
    src = Path(source_dir)
    if not src.exists() or not src.is_dir():
        raise ValueError(f"Source rebuild directory does not exist: {source_dir}")
    files = []
    missing = []
    for key, official_path in OFFICIAL_PROCESSED_PATHS.items():
        basename = REBUILD_BASENAMES[key]
        source_path = src / basename
        if not source_path.exists():
            missing.append(str(source_path))
        files.append({
            "key": key,
            "basename": basename,
            "source": str(source_path),
            "destination": official_path,
        })
    if missing:
        raise ValueError(f"Rebuilt source files missing (cannot promote): {missing}")
    return files


# ---------------------------------------------------------------------------
# Validation (before promotion)
# ---------------------------------------------------------------------------

def validate_source(
    source_step3c: str,
    official_step3c: str,
    base_numeric: list[str],
    expected_events=DEFAULT_EXPECTED_NEW_EVENTS,
    excluded_events=DEFAULT_EXCLUDED_EVENTS,
) -> dict:
    """Re-verify the rebuilt Step 3C against the CURRENT official Step 3C before
    promoting. Returns a dict with an overall ``ok`` flag and every check."""
    new_df = pd.read_csv(source_step3c)
    old_exists = Path(official_step3c).exists()
    old_df = pd.read_csv(official_step3c) if old_exists else new_df.iloc[0:0]

    schema = compare_schema(old_df, new_df, base_numeric)
    new_events = verify_new_events(new_df, base_numeric, expected_events, excluded_events)
    new_event_dates = {d for d, _, _ in expected_events}
    quality = feature_quality_checks(new_df, base_numeric, new_event_dates)
    new_events_df = new_df[new_df["date"].isin(new_event_dates)].copy()
    step6b = check_step6b_validation(new_events_df, base_numeric)

    expected_total = sum(c for _, _, c in expected_events)
    expected_max_date = max(d for d, _, _ in expected_events)

    checks = {
        "columns_identical_and_ordered": schema["columns_identical_and_ordered"],
        "all_official_features_present": schema["all_official_features_present"],
        "no_unexpected_columns": not schema["unexpected_new_columns"],
        "raw_weight_class_handling_consistent": schema["raw_weight_class_handling_consistent"],
        "row_delta_matches_expected": schema["row_count_delta"] == expected_total,
        "new_max_date_matches_expected": schema["new_max_date"] == expected_max_date,
        "all_expected_events_present": new_events["all_expected_events_present"],
        "all_excluded_events_absent": new_events["all_excluded_events_absent"],
        "leakage_guard_passes": new_events["leakage_guard_passes"],
        "no_new_event_duplicate_fight_ids": new_events["new_event_duplicate_fight_ids"] == 0,
        "no_new_event_duplicate_pairs": new_events["new_event_duplicate_pair_date_rows"] == 0,
        "no_duplicate_fight_ids": quality["duplicate_fight_ids"] == 0,
        "no_duplicate_pair_date_rows": quality["duplicate_pair_date_rows"] == 0,
        "no_infinite_values": quality["infinite_value_count_in_model_features"] == 0,
        "step6b_input_ok": step6b["ok"],
    }
    return {
        "ok": all(bool(v) for v in checks.values()),
        "checks": checks,
        "schema_comparison": schema,
        "new_event_verification": new_events,
        "feature_quality": quality,
        "step6b_input_validation": step6b,
        "expected_new_fight_total": expected_total,
        "expected_max_date": expected_max_date,
    }


# ---------------------------------------------------------------------------
# Backup / copy / rollback
# ---------------------------------------------------------------------------

def create_backup(promotion_files: list[dict], backup_dir: str | Path) -> dict:
    """Copy the CURRENT official files into ``backup_dir`` before any overwrite."""
    bdir = Path(backup_dir)
    bdir.mkdir(parents=True, exist_ok=True)
    entries = []
    for f in promotion_files:
        dest = Path(f["destination"])
        backup_path = bdir / dest.name
        if dest.exists():
            shutil.copy2(dest, backup_path)
        entries.append({
            "key": f["key"],
            "official_path": str(dest),
            "backup_path": str(backup_path),
            **({k: v for k, v in _summarize_csv(backup_path).items() if k in ("md5", "row_count", "max_date")}
               if backup_path.exists() else {"exists": False}),
        })
    return {"backup_dir": str(bdir), "files": entries}


def copy_files(promotion_files: list[dict]) -> None:
    for f in promotion_files:
        shutil.copy2(f["source"], f["destination"])


def restore_backup(backup: dict) -> None:
    for entry in backup["files"]:
        bp = Path(entry["backup_path"])
        if bp.exists():
            shutil.copy2(bp, entry["official_path"])


def rollback_instructions(backup_dir: str, promotion_files: list[dict]) -> list[str]:
    lines = [
        "To roll back this promotion, restore the backed-up official files:",
    ]
    for f in promotion_files:
        bp = Path(backup_dir) / Path(f["destination"]).name
        lines.append(f"  cp '{bp}' '{f['destination']}'")
    lines.append(
        "Then confirm each official file's md5 matches the pre-promotion "
        "'old' checksum recorded in this manifest."
    )
    return lines


# ---------------------------------------------------------------------------
# Post-promotion verification
# ---------------------------------------------------------------------------

def verify_post_promotion(
    promotion_files: list[dict],
    backup: dict,
    base_numeric: list[str],
    db_md5_before: str,
    bench_md5_before: str,
    expected_events=DEFAULT_EXPECTED_NEW_EVENTS,
    excluded_events=DEFAULT_EXCLUDED_EVENTS,
) -> dict:
    # 1. each official file is now a byte-exact copy of its rebuilt source
    copy_checks = []
    all_copied = True
    for f in promotion_files:
        src_md5 = file_md5(f["source"])
        dest_md5 = file_md5(f["destination"])
        ok = src_md5 == dest_md5
        all_copied = all_copied and ok
        copy_checks.append({"key": f["key"], "source_md5": src_md5,
                            "destination_md5": dest_md5, "exact_copy": ok})

    # 2. re-run the full verification on the now-official Step 3C vs the backup
    official_step3c = OFFICIAL_PROCESSED_PATHS["step3c"]
    backup_step3c = next(e["backup_path"] for e in backup["files"] if e["key"] == "step3c")
    source_validation = validate_source(
        official_step3c, backup_step3c, base_numeric, expected_events, excluded_events
    )

    new_df = pd.read_csv(official_step3c)
    expected_total = sum(c for _, _, c in expected_events)
    old_rows = next(e["row_count"] for e in backup["files"] if e["key"] == "step3c")
    expected_rows = old_rows + expected_total
    expected_max_date = max(d for d, _, _ in expected_events)

    db_md5_after = file_md5(DB_PATH)
    bench_md5_after = file_md5(BENCHMARK_PATH)

    checks = {
        "all_files_exact_copies_of_source": all_copied,
        "official_step3c_row_count_expected": int(len(new_df)) == expected_rows,
        "official_step3c_max_date_expected": str(new_df["date"].max()) == expected_max_date,
        "source_revalidation_ok": source_validation["ok"],
        "db_unchanged": db_md5_before == db_md5_after,
        "benchmark_unchanged": bench_md5_before == bench_md5_after,
    }
    return {
        "ok": all(bool(v) for v in checks.values()),
        "checks": checks,
        "copy_checks": copy_checks,
        "official_step3c_row_count": int(len(new_df)),
        "official_step3c_max_date": str(new_df["date"].max()),
        "expected_row_count": expected_rows,
        "expected_max_date": expected_max_date,
        "revalidation": source_validation,
        "db_md5_before": db_md5_before,
        "db_md5_after": db_md5_after,
        "benchmark_md5_before": bench_md5_before,
        "benchmark_md5_after": bench_md5_after,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_promotion(
    source_dir: str,
    apply: bool = False,
    reports_dir: str = "reports/data_update",
    official_step3c: str = OFFICIAL_PROCESSED_PATHS["step3c"],
    backup_root: str = BACKUP_ROOT,
    expected_events=DEFAULT_EXPECTED_NEW_EVENTS,
    excluded_events=DEFAULT_EXCLUDED_EVENTS,
    write_report: bool = True,
) -> dict:
    """Guarded, dry-run-by-default promotion of rebuilt processed files."""
    if not source_dir:
        raise ValueError("Step 6F requires an explicit --source-dir; refusing to promote.")

    base_numeric, _ = official_step3c_features()
    db_md5_before = file_md5(DB_PATH)
    bench_md5_before = file_md5(BENCHMARK_PATH)

    promotion_files = resolve_promotion_files(source_dir)

    # Pre-promotion inspection (never mutates anything)
    pre_inspection = {
        "official_files": {f["key"]: _summarize_csv(f["destination"]) for f in promotion_files},
        "rebuilt_files": {f["key"]: _summarize_csv(f["source"]) for f in promotion_files},
        "official_toplevel_checksum_before": dir_toplevel_checksum("data/processed"),
        "rebuild_dir_checksum": dir_toplevel_checksum(source_dir),
        "column_parity": {},
    }
    for f in promotion_files:
        off = pre_inspection["official_files"][f["key"]]
        reb = pre_inspection["rebuilt_files"][f["key"]]
        pre_inspection["column_parity"][f["key"]] = (
            off.get("column_count") == reb.get("column_count")
        )

    source_step3c = next(f["source"] for f in promotion_files if f["key"] == "step3c")
    validation = validate_source(
        source_step3c, official_step3c, base_numeric, expected_events, excluded_events,
    )

    report = {
        "step": "6F",
        "title": "Guarded promotion of rebuilt processed files",
        "generated_at": _now_utc(),
        "mode": "apply" if apply else "dry_run",
        "source_rebuild_dir": source_dir,
        "destination_official_paths": OFFICIAL_PROCESSED_PATHS,
        "promotion_files": promotion_files,
        "pre_promotion_inspection": pre_inspection,
        "source_validation": validation,
        "db_checksum": db_md5_before,
        "benchmark_checksum": bench_md5_before,
    }

    # --- Guard: never promote invalid data ---
    if not validation["ok"]:
        report["status"] = "aborted_validation_failed"
        report["backup"] = None
        report["post_promotion_verification"] = None
        report["official_processed_files_overwritten"] = False
        report["rollback_instructions"] = ["No promotion occurred; official files untouched."]
        report["statements"] = _statements(overwritten=False)
        if write_report:
            _finalize(report, reports_dir)
        return report

    if not apply:
        backup_dir = str(Path(backup_root) / f"pre_step6f_promotion_<stamp>")
        report["status"] = "dry_run"
        report["backup"] = {"backup_dir": backup_dir, "note": "created only on --apply"}
        report["post_promotion_verification"] = None
        report["official_processed_files_overwritten"] = False
        report["rollback_instructions"] = rollback_instructions(backup_dir, promotion_files)
        report["statements"] = _statements(overwritten=False)
        if write_report:
            _finalize(report, reports_dir)
        return report

    # --- Apply path: backup BEFORE overwrite ---
    backup_dir = str(Path(backup_root) / f"pre_step6f_promotion_{_stamp()}")
    backup = create_backup(promotion_files, backup_dir)
    copy_files(promotion_files)

    post = verify_post_promotion(
        promotion_files, backup, base_numeric, db_md5_before, bench_md5_before,
        expected_events, excluded_events,
    )

    report["backup"] = backup
    report["post_promotion_verification"] = post
    report["rollback_instructions"] = rollback_instructions(backup_dir, promotion_files)

    if post["ok"]:
        report["status"] = "promoted"
        report["official_processed_files_overwritten"] = True
    else:
        # auto-rollback on verification failure
        restore_backup(backup)
        report["status"] = "rolled_back_verification_failed"
        report["official_processed_files_overwritten"] = False
        report["rollback_performed"] = True

    report["statements"] = _statements(overwritten=report["official_processed_files_overwritten"])
    if write_report:
        _finalize(report, reports_dir)
    return report


def _statements(overwritten: bool) -> list[str]:
    return [
        ("Official processed files were promoted (backed up first)." if overwritten
         else "Official processed files were NOT overwritten."),
        "benchmarks/official_baseline.json was NOT modified.",
        "No model was retrained, replaced, or promoted.",
        "data/ufc.db was NOT modified.",
    ]


def _finalize(report: dict, reports_dir: str) -> None:
    json_path, md_path = write_reports(report, reports_dir)
    report["report_json"] = str(json_path)
    report["report_md"] = str(md_path)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_reports(report: dict, reports_dir: str) -> tuple[Path, Path]:
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    json_path = Path(reports_dir) / f"{REPORT_STEM}.json"
    md_path = Path(reports_dir) / f"{REPORT_STEM}.md"
    json_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _yn(value) -> str:
    return "✅" if value else "❌"


def render_markdown(report: dict) -> str:
    val = report["source_validation"]
    pre = report["pre_promotion_inspection"]
    post = report.get("post_promotion_verification")
    lines = [
        f"# Step 6F — {report['title']}",
        "",
        f"_Generated {report['generated_at']}_",
        "",
        f"**Mode:** {report['mode']}  |  **Status:** `{report['status']}`  |  "
        f"**Official files overwritten:** {_yn(report['official_processed_files_overwritten'])}",
        "",
        "## Source & destination",
        f"- Source rebuild dir: `{report['source_rebuild_dir']}`",
        f"- Rebuild dir checksum: `{pre['rebuild_dir_checksum']}`",
        f"- Official top-level checksum (before): `{pre['official_toplevel_checksum_before']}`",
        "",
        "## Files & row/date/checksum comparison",
        "| Key | Old rows | New rows | Old max date | New max date | Old md5 | New md5 | Cols match |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in report["promotion_files"]:
        k = f["key"]
        off = pre["official_files"][k]
        reb = pre["rebuilt_files"][k]
        lines.append(
            f"| {k} | {off.get('row_count')} | {reb.get('row_count')} | {off.get('max_date')} | "
            f"{reb.get('max_date')} | `{str(off.get('md5'))[:8]}` | `{str(reb.get('md5'))[:8]}` | "
            f"{_yn(pre['column_parity'][k])} |"
        )

    lines += ["", "## Source validation (before promotion)",
              f"- Overall: {_yn(val['ok'])}"]
    for name, ok in val["checks"].items():
        lines.append(f"  - {name}: {_yn(ok)}")

    if post is not None:
        lines += ["", "## Post-promotion verification",
                  f"- Overall: {_yn(post['ok'])}",
                  f"- Official Step 3C rows: {post['official_step3c_row_count']} "
                  f"(expected {post['expected_row_count']})",
                  f"- Official Step 3C max date: {post['official_step3c_max_date']} "
                  f"(expected {post['expected_max_date']})"]
        for name, ok in post["checks"].items():
            lines.append(f"  - {name}: {_yn(ok)}")
        lines += ["",
                  f"- DB md5: `{post['db_md5_before']}` → `{post['db_md5_after']}` "
                  f"({_yn(post['checks']['db_unchanged'])})",
                  f"- Benchmark md5: `{post['benchmark_md5_before']}` → `{post['benchmark_md5_after']}` "
                  f"({_yn(post['checks']['benchmark_unchanged'])})"]

    if report.get("backup"):
        bdir = report["backup"].get("backup_dir")
        lines += ["", "## Backup", f"- Backup dir: `{bdir}`"]
        for e in report["backup"].get("files", []):
            lines.append(f"  - {e['key']}: `{e.get('backup_path')}` "
                         f"(rows {e.get('row_count')}, md5 `{str(e.get('md5'))[:8]}`)")

    lines += ["", "## Rollback instructions"]
    lines += [f"    {ln}" for ln in report["rollback_instructions"]]
    lines += ["", "## Statements"]
    lines += [f"- {s}" for s in report["statements"]]
    lines.append("")
    return "\n".join(lines)
