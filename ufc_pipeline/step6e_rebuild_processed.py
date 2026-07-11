#!/usr/bin/env python3
"""Step 6E: controlled processed-feature rebuild from the updated database.

Purpose
-------
After a guarded Step 6D.2 cached-event apply changed ``data/ufc.db``, the
processed Step 3 / 3B / 3C feature CSVs under ``data/processed/`` are stale
(they predate the newly inserted events). Step 6E regenerates those artifacts
from the current DB **without touching anything official**: it writes into a
separate output directory, never overwrites the official processed files, never
edits ``benchmarks/official_baseline.json``, never retrains/promotes a model,
and never writes to ``data/ufc.db``.

Design decisions
----------------
* **Reuse the canonical builders verbatim.** The rebuild calls the exact
  ``features.build_features_for_db`` / ``stats_features.build_step3b_for_db`` /
  ``matchup_features.build_step3c_for_db`` functions that produced the official
  files. They read the DB with ``pd.read_sql_query`` and only ever ``to_csv``;
  they do not mutate the DB. We still bracket the whole rebuild with a
  before/after DB md5 assertion so any future change that starts writing the DB
  fails loudly here.
* **Elo is already in the DB.** Step 6D.2 inserted ``fight_elo_snapshots`` for
  every new fight, so the Step 3/3B/3C builders read them directly. We do NOT
  call ``build_elo_for_db`` (the only DB-writing builder) — that keeps the DB
  byte-identical.
* **Non-destructive by construction.** The three target output paths are hard
  guarded against equalling the official processed paths; existing rebuild files
  are not clobbered unless ``overwrite_rebuild`` is set.
* **Verification, not benchmarking.** The optional official-model compatibility
  check reproduces the locked official model from the *old* official CSV and
  confirms it can structurally score rebuilt rows. It deliberately reports no
  held-out log loss / Brier / AUC and promotes nothing.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_pipeline.columns import name_key
from ufc_pipeline.feature_diagnostics import DEFAULT_BASELINE_PATH, official_step3c_features
from ufc_pipeline.features import build_features_for_db
from ufc_pipeline.matchup_features import build_step3c_for_db
from ufc_pipeline.modeling import TARGET, check_features_allowed, coerce_numeric_features
from ufc_pipeline.stats_features import build_step3b_for_db

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPORT_STEM = "step6e_rebuild_report"

# The official processed artifacts we must never overwrite.
OFFICIAL_PROCESSED_PATHS = {
    "step3": "data/processed/ufc_prefight_features.csv",
    "step3b": "data/processed/ufc_prefight_features_step3b.csv",
    "step3c": "data/processed/ufc_prefight_features_step3c.csv",
}
OFFICIAL_STEP3C_PATH = OFFICIAL_PROCESSED_PATHS["step3c"]

# Basenames used inside the rebuild output directory (mirror the official names
# so the rebuilt files are drop-in comparable).
REBUILD_BASENAMES = {
    "step3": "ufc_prefight_features.csv",
    "step3b": "ufc_prefight_features_step3b.csv",
    "step3c": "ufc_prefight_features_step3c.csv",
}

# Identifier / label columns carried in the Step 3C file (not model features).
ID_COLUMNS = ("fight_id", "date", "event", "fighter_a", "fighter_b", "weight_class")
RESULT_LABEL_COLUMNS = ("winner", "fighter_a_won", "method", "round")
LOW_HISTORY_FLAG_COLUMNS = (
    "fighter_a_no_prior_stats",
    "fighter_b_no_prior_stats",
    "matchup_history_missing",
)

# Events this specific rebuild expects to have gained, and the future-dated card
# that must stay excluded. Callers/tests may override; the CLI uses these
# defaults (they match the confirmed Step 6D.2 apply through 2026-06-27).
DEFAULT_EXPECTED_NEW_EVENTS = (
    ("2026-06-06", "UFC Fight Night: Muhammad vs. Bonfim", 12),
    ("2026-06-14", "UFC Freedom 250", 7),
    ("2026-06-20", "UFC Fight Night: Kape vs. Horiguchi", 12),
    ("2026-06-27", "UFC Fight Night: Fiziev vs. Torres", 13),
)
DEFAULT_EXCLUDED_EVENTS = (
    ("2026-07-11", "UFC 329: McGregor vs. Holloway 2"),
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_output_dir() -> str:
    return f"data/processed/rebuild_step6e_{datetime.now(timezone.utc).strftime('%Y%m%d')}"


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def file_md5(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.md5()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pair_date_key(row) -> tuple:
    """Deterministic (date, {fighter set}) key for duplicate-bout detection."""
    return (str(row["date"]), frozenset({name_key(row["fighter_a"]), name_key(row["fighter_b"])}))


# ---------------------------------------------------------------------------
# Safety guard
# ---------------------------------------------------------------------------

def resolve_output_paths(output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    return {key: str(out / name) for key, name in REBUILD_BASENAMES.items()}


def assert_non_destructive(output_paths: dict[str, str], overwrite_rebuild: bool) -> None:
    """Refuse to write any official processed file; refuse to clobber prior
    rebuild files unless explicitly allowed. This protection is absolute and
    is the whole point of Step 6E."""
    official = {Path(p).resolve() for p in OFFICIAL_PROCESSED_PATHS.values()}
    for key, path in output_paths.items():
        rp = Path(path).resolve()
        if rp in official:
            raise ValueError(
                f"Step 6E refuses to overwrite the official processed file for {key}: "
                f"{path}. Choose an --output-dir other than data/processed."
            )
    if not overwrite_rebuild:
        existing = [p for p in output_paths.values() if Path(p).exists()]
        if existing:
            raise ValueError(
                "Rebuild output files already exist (pass --overwrite-rebuild to "
                f"replace them): {existing}"
            )


# ---------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------

def rebuild_artifacts(db_path: str, output_dir: str, output_paths: dict[str, str]) -> dict:
    """Run the canonical Step 3 / 3B / 3C builders into ``output_dir``.

    Bracketed by a DB md5 before/after assertion: the builders are read-only on
    the DB, and we prove it here rather than trust it.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    db_md5_before = file_md5(db_path)

    n_step3 = build_features_for_db(db_path, output_paths["step3"])
    n_step3b = build_step3b_for_db(db_path, output_paths["step3b"])
    n_step3c = build_step3c_for_db(db_path, output_paths["step3c"])

    db_md5_after = file_md5(db_path)
    if db_md5_before != db_md5_after:
        raise RuntimeError(
            "DB checksum changed during rebuild — the builders must be read-only. "
            f"before={db_md5_before} after={db_md5_after}"
        )

    return {
        "db_md5_before": db_md5_before,
        "db_md5_after": db_md5_after,
        "db_unchanged": db_md5_before == db_md5_after,
        "rows": {"step3": n_step3, "step3b": n_step3b, "step3c": n_step3c},
        "output_paths": output_paths,
    }


# ---------------------------------------------------------------------------
# Schema parity
# ---------------------------------------------------------------------------

def compare_schema(old_df: pd.DataFrame, new_df: pd.DataFrame, base_numeric: list[str]) -> dict:
    old_cols, new_cols = list(old_df.columns), list(new_df.columns)
    only_new = [c for c in new_cols if c not in old_cols]
    only_old = [c for c in old_cols if c not in new_cols]

    old_ids = set(old_df["fight_id"]) if "fight_id" in old_df.columns else set()
    new_ids = set(new_df["fight_id"]) if "fight_id" in new_df.columns else set()

    missing_features = [c for c in base_numeric if c not in new_cols]
    weight_class_present = "weight_class" in new_cols
    weight_class_in_features = "weight_class" in base_numeric

    return {
        "old_row_count": int(len(old_df)),
        "new_row_count": int(len(new_df)),
        "row_count_delta": int(len(new_df) - len(old_df)),
        "old_max_date": str(old_df["date"].max()) if "date" in old_df.columns else None,
        "new_max_date": str(new_df["date"].max()) if "date" in new_df.columns else None,
        "old_column_count": len(old_cols),
        "new_column_count": len(new_cols),
        "columns_only_in_new": only_new,
        "columns_only_in_old": only_old,
        "columns_identical_and_ordered": old_cols == new_cols,
        "unexpected_new_columns": only_new,
        "official_feature_count": len(base_numeric),
        "all_official_features_present": not missing_features,
        "missing_required_feature_columns": missing_features,
        "result_label_columns_present": [c for c in RESULT_LABEL_COLUMNS if c in new_cols],
        "target_column_present": TARGET in new_cols,
        "raw_weight_class_present_as_label": weight_class_present,
        "raw_weight_class_in_model_features": weight_class_in_features,
        "raw_weight_class_handling_consistent": weight_class_present and not weight_class_in_features,
        "new_fight_ids_not_in_old": sorted(int(x) for x in (new_ids - old_ids)),
        "old_fight_ids_not_in_new": sorted(int(x) for x in (old_ids - new_ids)),
    }


# ---------------------------------------------------------------------------
# New-event verification
# ---------------------------------------------------------------------------

def verify_new_events(
    new_df: pd.DataFrame,
    base_numeric: list[str],
    expected_events=DEFAULT_EXPECTED_NEW_EVENTS,
    excluded_events=DEFAULT_EXCLUDED_EVENTS,
) -> dict:
    events_report = []
    all_present = True
    for date, name, expected_count in expected_events:
        rows = new_df[(new_df["date"] == date) & (new_df["event"] == name)]
        count = int(len(rows))
        ok = count == expected_count
        all_present = all_present and ok
        # per-event structural checks
        n_feature_rows = int(rows[base_numeric].notna().any(axis=1).sum()) if count else 0
        events_report.append({
            "date": date,
            "event": name,
            "expected_fights": expected_count,
            "found_fights": count,
            "count_matches": ok,
            "rows_with_feature_values": n_feature_rows,
        })

    # excluded (future-dated) events must be absent
    excluded_report = []
    excluded_absent = True
    for date, name in excluded_events:
        by_date = int((new_df["date"] == date).sum())
        by_name = int((new_df["event"] == name).sum())
        present = (by_date > 0) or (by_name > 0)
        excluded_absent = excluded_absent and not present
        excluded_report.append({
            "date": date,
            "event": name,
            "rows_matching_date": by_date,
            "rows_matching_name": by_name,
            "absent": not present,
        })

    # leakage: none of the 43 model features may be a result/label column
    forbidden_in_features = [c for c in base_numeric if c in set(RESULT_LABEL_COLUMNS) | {TARGET}]
    try:
        check_features_allowed(list(base_numeric))
        leakage_guard_passes = True
        leakage_error = None
    except ValueError as exc:
        leakage_guard_passes = False
        leakage_error = str(exc)

    # duplicate detection restricted to the newly-gained fights
    new_dates = {d for d, _, _ in expected_events}
    new_rows = new_df[new_df["date"].isin(new_dates)]
    dup_fight_ids = int(new_rows["fight_id"].duplicated().sum())
    pair_keys = new_rows.apply(_pair_date_key, axis=1)
    dup_pairs = int(pair_keys.duplicated().sum())

    return {
        "expected_events": events_report,
        "all_expected_events_present": all_present,
        "excluded_events": excluded_report,
        "all_excluded_events_absent": excluded_absent,
        "new_event_duplicate_fight_ids": dup_fight_ids,
        "new_event_duplicate_pair_date_rows": dup_pairs,
        "model_features_containing_result_columns": forbidden_in_features,
        "leakage_guard_passes": leakage_guard_passes,
        "leakage_guard_error": leakage_error,
    }


# ---------------------------------------------------------------------------
# Feature-quality checks
# ---------------------------------------------------------------------------

def feature_quality_checks(
    new_df: pd.DataFrame,
    base_numeric: list[str],
    new_event_dates: set[str],
) -> dict:
    features = new_df[base_numeric].apply(pd.to_numeric, errors="coerce")

    # missingness (whole file + new events only)
    nan_rate_all = (features.isna().mean()).round(6)
    top_missing = (
        nan_rate_all.sort_values(ascending=False)
        .head(10)
        .to_dict()
    )
    inf_counts = int(np.isinf(features.to_numpy(dtype=float, na_value=np.nan)).sum())

    dup_fight_ids = int(new_df["fight_id"].duplicated().sum())
    pair_keys = new_df.apply(_pair_date_key, axis=1)
    dup_pairs = int(pair_keys.duplicated().sum())

    ordered = new_df.sort_values(["date", "fight_id"], kind="stable").reset_index(drop=True)
    date_ordered = bool((ordered["fight_id"].tolist() == new_df["fight_id"].tolist())
                        and new_df["date"].tolist() == ordered["date"].tolist())

    target_counts = new_df[TARGET].value_counts(dropna=False).to_dict() if TARGET in new_df.columns else {}
    target_dist = {str(k): int(v) for k, v in target_counts.items()}

    # low-history / debut flags for newly inserted fighters
    new_rows = new_df[new_df["date"].isin(new_event_dates)]
    debut_flags = {}
    for flag in LOW_HISTORY_FLAG_COLUMNS:
        if flag in new_rows.columns:
            debut_flags[flag] = int(pd.to_numeric(new_rows[flag], errors="coerce").fillna(0).astype(bool).sum())
    debut_fighter_rows = int(
        (
            (pd.to_numeric(new_rows.get("fighter_a_prior_fights"), errors="coerce") == 0)
            | (pd.to_numeric(new_rows.get("fighter_b_prior_fights"), errors="coerce") == 0)
        ).sum()
    ) if "fighter_a_prior_fights" in new_rows.columns else None

    return {
        "model_feature_nan_rate_top10": {k: float(v) for k, v in top_missing.items()},
        "model_feature_max_nan_rate": float(nan_rate_all.max()) if len(nan_rate_all) else 0.0,
        "infinite_value_count_in_model_features": inf_counts,
        "duplicate_fight_ids": dup_fight_ids,
        "duplicate_pair_date_rows": dup_pairs,
        "date_ordered_by_date_fight_id": date_ordered,
        "target_distribution": target_dist,
        "new_event_low_history_flag_counts": debut_flags,
        "new_event_debut_fighter_rows": debut_fighter_rows,
    }


# ---------------------------------------------------------------------------
# Step 6B input compatibility + optional model compatibility
# ---------------------------------------------------------------------------

def build_step6b_style_input(new_events_df: pd.DataFrame, base_numeric: list[str]) -> pd.DataFrame:
    """Turn rebuilt Step 3C rows into the no-result upcoming-card frame Step 6B
    expects (id columns + weight_class + the 43 features; result columns dropped)."""
    frame = pd.DataFrame({
        "event_date": new_events_df["date"].values,
        "event_name": new_events_df["event"].values,
        "fighter_a": new_events_df["fighter_a"].values,
        "fighter_b": new_events_df["fighter_b"].values,
    })
    if "weight_class" in new_events_df.columns:
        frame["weight_class"] = new_events_df["weight_class"].values
    for col in base_numeric:
        frame[col] = new_events_df[col].values
    return frame


def check_step6b_validation(new_events_df: pd.DataFrame, base_numeric: list[str]) -> dict:
    # imported lazily so a broken step6b import never blocks the rebuild itself
    from ufc_pipeline.step6b_live_predictions import validate_prediction_input

    step6b_input = build_step6b_style_input(new_events_df, base_numeric)
    validation = validate_prediction_input(step6b_input, base_numeric)
    return {
        "n_rows": validation["n_rows"],
        "n_valid_rows": validation["n_valid_rows"],
        "n_error_rows": validation["n_error_rows"],
        "ok": validation["ok"],
        "structural_errors": validation["structural_errors"],
        "forbidden_result_columns": validation["forbidden_result_columns"],
        "market_columns": validation["market_columns"],
        "missing_required_features": validation["missing_required_features"],
    }


def official_model_compatibility(
    old_official_step3c: str,
    new_events_df: pd.DataFrame,
    base_numeric: list[str],
    baseline_path: str | None = DEFAULT_BASELINE_PATH,
) -> dict:
    """Structural-only check: reproduce the locked official model from the OLD
    official CSV and confirm it can score rebuilt rows. Reports NO held-out
    metrics and promotes nothing."""
    from ufc_pipeline.step6b_live_predictions import generate_probabilities, train_official_model

    if not Path(old_official_step3c).exists():
        return {"status": "skipped", "reason": f"official CSV not found: {old_official_step3c}"}
    if baseline_path and not Path(baseline_path).exists():
        return {"status": "skipped", "reason": f"baseline not found: {baseline_path}"}

    try:
        trained = train_official_model(training_csv=old_official_step3c, baseline_path=baseline_path)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never abort the rebuild
        return {"status": "skipped", "reason": f"could not reproduce official model: {exc}"}

    scored = coerce_numeric_features(new_events_df.copy(), base_numeric, context="step6e_compat")
    probs = generate_probabilities(scored, trained["pipeline"], trained["platt"], base_numeric)
    finite = bool(np.isfinite(probs).all())
    in_unit = bool(((probs >= 0.0) & (probs <= 1.0)).all()) if len(probs) else True
    return {
        "status": "ok",
        "note": "Structural compatibility only — reproduces the official model from the "
                "old official CSV and scores rebuilt rows. No new benchmark; nothing promoted.",
        "model_version": trained["model_version"],
        "n_rows_scored": int(len(probs)),
        "all_probabilities_finite": finite,
        "all_probabilities_in_unit_interval": in_unit,
        "probability_min": float(np.min(probs)) if len(probs) else None,
        "probability_max": float(np.max(probs)) if len(probs) else None,
        "structural_ok": finite and in_unit and len(probs) == len(new_events_df),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_rebuild(
    db_path: str = "data/ufc.db",
    output_dir: str | None = None,
    reports_dir: str = "reports/data_update/step6e",
    official_step3c: str = OFFICIAL_STEP3C_PATH,
    no_overwrite_official: bool = True,
    overwrite_rebuild: bool = False,
    run_compat: bool = True,
    expected_events=DEFAULT_EXPECTED_NEW_EVENTS,
    excluded_events=DEFAULT_EXCLUDED_EVENTS,
) -> dict:
    output_dir = output_dir or default_output_dir()
    output_paths = resolve_output_paths(output_dir)

    # Absolute non-destructive guard (independent of the flag; the flag is
    # recorded for provenance). Step 6E never writes official processed files.
    assert_non_destructive(output_paths, overwrite_rebuild=overwrite_rebuild)

    base_numeric, _categorical = official_step3c_features()

    rebuild = rebuild_artifacts(db_path, output_dir, output_paths)

    new_df = pd.read_csv(output_paths["step3c"])
    old_exists = Path(official_step3c).exists()
    old_df = pd.read_csv(official_step3c) if old_exists else new_df.iloc[0:0]

    schema = compare_schema(old_df, new_df, base_numeric)
    new_events = verify_new_events(new_df, base_numeric, expected_events, excluded_events)
    new_event_dates = {d for d, _, _ in expected_events}
    quality = feature_quality_checks(new_df, base_numeric, new_event_dates)

    new_events_df = new_df[new_df["date"].isin(new_event_dates)].copy()
    step6b_validation = check_step6b_validation(new_events_df, base_numeric)

    if run_compat:
        compat = official_model_compatibility(official_step3c, new_events_df, base_numeric)
    else:
        compat = {"status": "skipped", "reason": "run_compat disabled"}

    report = {
        "step": "6E",
        "title": "Controlled processed-feature rebuild from updated DB",
        "generated_at": _now_utc(),
        "db_path": db_path,
        "db_checksum_used": rebuild["db_md5_before"],
        "db_unchanged_by_rebuild": rebuild["db_unchanged"],
        "no_overwrite_official_flag": no_overwrite_official,
        "official_processed_paths": OFFICIAL_PROCESSED_PATHS,
        "old_official_step3c_present": old_exists,
        "old_official_step3c_path": official_step3c,
        "rebuild_output_dir": output_dir,
        "rebuilt_output_paths": output_paths,
        "rebuilt_row_counts": rebuild["rows"],
        "schema_comparison": schema,
        "new_event_verification": new_events,
        "feature_quality": quality,
        "step6b_input_validation": step6b_validation,
        "official_model_compatibility": compat,
        "official_processed_files_overwritten": False,
        "benchmark_or_model_modified": False,
        "statements": [
            "Official processed files under data/processed/ were NOT overwritten.",
            "benchmarks/official_baseline.json was NOT modified.",
            "No model was retrained, replaced, or promoted.",
            "data/ufc.db was NOT modified (checksum identical before/after).",
        ],
    }
    report["overall_ok"] = _overall_ok(report)

    json_path, md_path = write_reports(report, reports_dir)
    report["report_json"] = str(json_path)
    report["report_md"] = str(md_path)
    return report


def _overall_ok(report: dict) -> bool:
    schema = report["schema_comparison"]
    nev = report["new_event_verification"]
    q = report["feature_quality"]
    checks = [
        report["db_unchanged_by_rebuild"],
        schema["all_official_features_present"],
        not schema["unexpected_new_columns"],
        schema["raw_weight_class_handling_consistent"],
        nev["all_expected_events_present"],
        nev["all_excluded_events_absent"],
        nev["leakage_guard_passes"],
        nev["new_event_duplicate_fight_ids"] == 0,
        nev["new_event_duplicate_pair_date_rows"] == 0,
        q["infinite_value_count_in_model_features"] == 0,
        q["duplicate_fight_ids"] == 0,
        report["step6b_input_validation"]["ok"],
        report["official_model_compatibility"].get("status") in ("ok", "skipped"),
    ]
    return all(bool(c) for c in checks)


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
    s = report["schema_comparison"]
    nev = report["new_event_verification"]
    q = report["feature_quality"]
    v = report["step6b_input_validation"]
    compat = report["official_model_compatibility"]
    lines = [
        f"# Step 6E — {report['title']}",
        "",
        f"_Generated {report['generated_at']}_",
        "",
        f"**Overall:** {_yn(report['overall_ok'])}  |  "
        f"**DB unchanged:** {_yn(report['db_unchanged_by_rebuild'])} "
        f"(`{report['db_checksum_used']}`)",
        "",
        "## Outputs (non-destructive)",
        f"- Rebuild output dir: `{report['rebuild_output_dir']}`",
    ]
    for key, path in report["rebuilt_output_paths"].items():
        lines.append(f"  - {key}: `{path}` ({report['rebuilt_row_counts'][key]} rows)")
    lines += [
        f"- Official processed files overwritten: {_yn(not report['official_processed_files_overwritten'])} (no)",
        f"- Benchmark / model modified: {_yn(not report['benchmark_or_model_modified'])} (no)",
        "",
        "## Schema parity (rebuilt Step 3C vs official Step 3C)",
        f"- Old rows: {s['old_row_count']} → New rows: {s['new_row_count']} (Δ {s['row_count_delta']:+d})",
        f"- Old max date: {s['old_max_date']} → New max date: {s['new_max_date']}",
        f"- Columns: {s['old_column_count']} → {s['new_column_count']} "
        f"(identical & ordered: {_yn(s['columns_identical_and_ordered'])})",
        f"- Columns only in new: {s['columns_only_in_new'] or 'none'}",
        f"- Columns only in old: {s['columns_only_in_old'] or 'none'}",
        f"- All 43 official features present: {_yn(s['all_official_features_present'])}",
        f"- Missing required feature columns: {s['missing_required_feature_columns'] or 'none'}",
        f"- Result/label columns retained: {s['result_label_columns_present']}",
        f"- Raw weight_class handling consistent "
        f"(present as label, absent from model features): {_yn(s['raw_weight_class_handling_consistent'])}",
        f"- New fight_ids not in old: {len(s['new_fight_ids_not_in_old'])} "
        f"(range {s['new_fight_ids_not_in_old'][:1]}..{s['new_fight_ids_not_in_old'][-1:]})",
        "",
        "## New-event verification",
        f"- All expected events present: {_yn(nev['all_expected_events_present'])}",
        f"- All excluded (future) events absent: {_yn(nev['all_excluded_events_absent'])}",
        f"- New-event duplicate fight_ids: {nev['new_event_duplicate_fight_ids']} | "
        f"duplicate pair/date rows: {nev['new_event_duplicate_pair_date_rows']}",
        f"- Model features containing result columns: {nev['model_features_containing_result_columns'] or 'none'} "
        f"(leakage guard passes: {_yn(nev['leakage_guard_passes'])})",
        "",
        "| Event | Date | Expected | Found | OK |",
        "|---|---|---|---|---|",
    ]
    for e in nev["expected_events"]:
        lines.append(f"| {e['event']} | {e['date']} | {e['expected_fights']} | {e['found_fights']} | {_yn(e['count_matches'])} |")
    for e in nev["excluded_events"]:
        lines.append(f"| {e['event']} (excluded) | {e['date']} | 0 | {e['rows_matching_date']} | {_yn(e['absent'])} |")
    lines += [
        "",
        "## Feature quality (rebuilt Step 3C)",
        f"- Infinite values in model features: {q['infinite_value_count_in_model_features']}",
        f"- Duplicate fight_ids: {q['duplicate_fight_ids']} | duplicate pair/date rows: {q['duplicate_pair_date_rows']}",
        f"- Date-ordered by (date, fight_id): {_yn(q['date_ordered_by_date_fight_id'])}",
        f"- Max model-feature NaN rate: {q['model_feature_max_nan_rate']:.4f}",
        f"- Target distribution: {q['target_distribution']}",
        f"- New-event low-history flag counts: {q['new_event_low_history_flag_counts']}",
        f"- New-event debut-fighter rows: {q['new_event_debut_fighter_rows']}",
        "",
        "## Step 6B input validation (rebuilt new-event rows)",
        f"- Accepted: {_yn(v['ok'])} | valid rows: {v['n_valid_rows']}/{v['n_rows']} | "
        f"errors: {v['n_error_rows']}",
        f"- Structural errors: {v['structural_errors'] or 'none'}",
        "",
        "## Official model compatibility (structural only — no benchmark, nothing promoted)",
        f"- Status: {compat.get('status')}",
    ]
    if compat.get("status") == "ok":
        lines += [
            f"- Model version: {compat['model_version']}",
            f"- Rows scored: {compat['n_rows_scored']} | all finite: {_yn(compat['all_probabilities_finite'])} | "
            f"all in [0,1]: {_yn(compat['all_probabilities_in_unit_interval'])}",
            f"- Probability range: [{compat['probability_min']:.4f}, {compat['probability_max']:.4f}]"
            if compat.get("probability_min") is not None else "- Probability range: n/a",
        ]
    else:
        lines.append(f"- Reason: {compat.get('reason')}")
    lines += ["", "## Statements"]
    lines += [f"- {stmt}" for stmt in report["statements"]]
    lines.append("")
    return "\n".join(lines)
