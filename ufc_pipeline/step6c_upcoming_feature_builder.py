"""Step 6C: build model-ready Step 3C feature rows for an upcoming card.

Takes a simple matchup CSV (event_date, event_name, fighter_a, fighter_b, and
optional weight_class) and produces the exact Step 3C feature columns Step 6B
expects — using only fights that happened strictly BEFORE the event date.

This is a *direct recomputation*, not a lossy bridge: it drives the very same
Step 3/3B/3C builders that produced ``data/processed/ufc_prefight_features_step3c.csv``
(``features.build_feature_rows``, ``stats_features.build_step3b_rows``,
``matchup_features.build_against_rows`` + ``add_matchup_features``). Each
upcoming bout is appended to the historical fight list as a *synthetic* fight
with no result; because those builders read each fighter's pre-fight state from
prior fights only and apply the result only afterwards, the synthetic row's
features are computed exactly as they would have been for a real bout on that
date — with no result/target leakage. Elo going into the card is obtained by
running the deterministic Elo engine over the strictly-earlier fights.

Boundaries (unchanged by this task): no UFCStats scraping, no odds / prediction
markets / new data sources / new model families / model retraining. It reads the
existing SQLite history (``data/ufc.db``) and writes only to the given output
path and ``reports/live/`` — never ``data/processed/`` or the benchmark file.
Debut / low-history fighters get the same missing-value treatment as training
(honest nulls, ``no_prior_stats`` flags); fighters that cannot be identified are
reported as failed rows, never guessed.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ufc_pipeline.db import connect, init_schema
from ufc_pipeline.elo import DEFAULT_K, DEFAULT_STARTING_ELO, expected_score, run_elo
from ufc_pipeline.feature_diagnostics import official_step3c_features
from ufc_pipeline.features import FIGHTS_WITH_ELO_QUERY, build_feature_rows
from ufc_pipeline.matchup_features import add_matchup_features, build_against_rows
from ufc_pipeline.modeling import TARGET
from ufc_pipeline.stats_features import STATS_QUERY, build_step3b_rows
from ufc_pipeline.step6b_live_predictions import validate_prediction_input

DEFAULT_HISTORY_DB = "data/ufc.db"
DEFAULT_OUTPUT = "data/live/upcoming_card_features.csv"
DEFAULT_REPORT_DIR = "reports/live"

REQUIRED_MATCHUP_COLUMNS = ("event_date", "event_name", "fighter_a", "fighter_b")
OPTIONAL_MATCHUP_COLUMNS = ("weight_class", "scheduled_rounds", "fight_order", "notes")
# Result/leakage columns that must NEVER appear in the built feature output.
FORBIDDEN_OUTPUT_COLUMNS = frozenset({
    "fighter_a_won", "winner", "loser", "method", "round", "result", "outcome", TARGET,
})

# Synthetic fight_ids are large negatives so they never collide with real ids
# and are obviously non-historical.
_SYNTHETIC_FIGHT_ID_BASE = -1_000_000


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(obj):
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Name normalization + matching
# ---------------------------------------------------------------------------

def normalize_fighter_name(value) -> str:
    """Deterministic name key: strip accents, lowercase, drop punctuation,
    collapse whitespace. Never fuzzy."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    s = unicodedata.normalize("NFKD", str(value))
    s = "".join(c for c in s if not unicodedata.combining(c))  # drop accents
    s = s.lower().strip()
    s = re.sub(r"[.'`’]", "", s)          # drop periods / apostrophes
    s = re.sub(r"[-_/]", " ", s)          # hyphens / slashes -> space
    s = re.sub(r"[^a-z0-9 ]", "", s)      # drop any other punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_fighter_lookup(fighters_df: pd.DataFrame) -> dict[str, list[dict]]:
    """normalized name -> list of {fighter_id, name, height, reach, date_of_birth}."""
    lookup: dict[str, list[dict]] = {}
    for _, r in fighters_df.iterrows():
        key = normalize_fighter_name(r["name"])
        lookup.setdefault(key, []).append({
            "fighter_id": int(r["fighter_id"]), "name": r["name"],
            "height": r["height"], "reach": r["reach"], "date_of_birth": r["date_of_birth"],
        })
    return lookup


def match_fighter(name, lookup: dict[str, list[dict]], allow_fuzzy: bool = False) -> dict:
    """Resolve a fighter name to a single historical fighter. Exact normalized
    match only (fuzzy is intentionally not implemented unless explicitly enabled,
    and even then stays a documented no-op placeholder)."""
    key = normalize_fighter_name(name)
    if not key:
        return {"status": "empty", "fighter_id": None, "matched_name": None, "detail": "empty name"}
    candidates = lookup.get(key, [])
    if len(candidates) == 1:
        c = candidates[0]
        return {"status": "matched", "fighter_id": c["fighter_id"], "matched_name": c["name"],
                "attrs": c, "detail": "exact normalized match"}
    if len(candidates) == 0:
        return {"status": "unmatched", "fighter_id": None, "matched_name": None,
                "detail": f"no historical fighter with normalized name '{key}'"}
    return {"status": "ambiguous", "fighter_id": None, "matched_name": None,
            "detail": f"{len(candidates)} fighters share normalized name '{key}': "
                      f"{[c['fighter_id'] for c in candidates]}; not guessing"}


# ---------------------------------------------------------------------------
# Matchup schema validation
# ---------------------------------------------------------------------------

def validate_matchup_schema(df: pd.DataFrame, event_date_override: str | None = None) -> dict:
    columns = set(df.columns)
    required = [c for c in REQUIRED_MATCHUP_COLUMNS if c != "event_date" or event_date_override is None]
    missing = [c for c in required if c not in columns]
    forbidden = sorted(FORBIDDEN_OUTPUT_COLUMNS & columns)

    row_status = []
    seen = {}
    for idx, row in df.iterrows():
        reasons = []
        ev = event_date_override if event_date_override is not None else row.get("event_date")
        try:
            pd.to_datetime(ev)
        except (ValueError, TypeError):
            reasons.append("unparseable event_date")
        fa = normalize_fighter_name(row.get("fighter_a"))
        fb = normalize_fighter_name(row.get("fighter_b"))
        if not fa:
            reasons.append("empty fighter_a")
        if not fb:
            reasons.append("empty fighter_b")
        if fa and fb and fa == fb:
            reasons.append("fighter_a equals fighter_b")
        key = (str(ev), frozenset({fa, fb}))
        if fa and fb and key in seen:
            reasons.append(f"duplicate matchup (also row {seen[key]})")
        seen.setdefault(key, idx)
        row_status.append({"row_index": int(idx), "status": "error" if reasons else "valid", "reasons": reasons})

    return {
        "ok": not missing and not forbidden,
        "missing_required_columns": missing,
        "forbidden_columns_present": forbidden,
        "row_status": row_status,
        "n_rows": int(len(df)),
    }


# ---------------------------------------------------------------------------
# History loading
# ---------------------------------------------------------------------------

def load_history(db_path: str) -> dict:
    conn = connect(db_path)
    try:
        init_schema(conn)
        fights_df = pd.read_sql_query(FIGHTS_WITH_ELO_QUERY, conn)
        stats_df = pd.read_sql_query(STATS_QUERY, conn)
        fighters_df = pd.read_sql_query(
            "SELECT fighter_id, name, normalized_name, height, reach, date_of_birth FROM fighters", conn)
    finally:
        conn.close()
    if fights_df.empty:
        raise RuntimeError(f"No fights found in {db_path}. Build the history DB first.")
    if fights_df["fighter_a_pre_elo"].isna().any():
        raise RuntimeError("Fights without Elo snapshots found in the history DB. Run build_elo.py first.")

    fights = fights_df.astype(object).where(pd.notna(fights_df), None).to_dict("records")
    stats = {
        (int(r["fight_id"]), int(r["fighter_id"])): r
        for r in stats_df.astype(object).where(pd.notna(stats_df), None).to_dict("records")
    }
    return {"fights": fights, "stats": stats, "fighters_df": fighters_df,
            "lookup": build_fighter_lookup(fighters_df)}


# ---------------------------------------------------------------------------
# Synthetic-fight construction + feature build
# ---------------------------------------------------------------------------

def _synthetic_fight(fight_id, event_date, event_name, weight_class, a, b, pre_elo_a, pre_elo_b) -> dict:
    """A time-safe stand-in for an upcoming bout. ``fighter_a_won`` is a
    placeholder used ONLY by the post-row history update (which never feeds any
    row we keep); no result information is used to build the row."""
    exp_a = expected_score(pre_elo_a, pre_elo_b)
    return {
        "fight_id": fight_id, "date": event_date, "event": event_name,
        "fighter_a_id": a["fighter_id"], "fighter_b_id": b["fighter_id"],
        "fighter_a": a["matched_name"], "fighter_b": b["matched_name"],
        "winner": None, "fighter_a_won": 0, "weight_class": weight_class,
        "method": None, "round": None,
        "fighter_a_age": None, "fighter_b_age": None,
        "fighter_a_dob": a["attrs"]["date_of_birth"], "fighter_b_dob": b["attrs"]["date_of_birth"],
        "fighter_a_height": _to_float(a["attrs"]["height"]), "fighter_b_height": _to_float(b["attrs"]["height"]),
        "fighter_a_reach": _to_float(a["attrs"]["reach"]), "fighter_b_reach": _to_float(b["attrs"]["reach"]),
        "fighter_a_pre_elo": pre_elo_a, "fighter_b_pre_elo": pre_elo_b,
        "fighter_a_expected_win_prob": exp_a, "fighter_b_expected_win_prob": 1.0 - exp_a,
        "_synthetic": True,
    }


def _to_float(v):
    try:
        return float(v) if v is not None and str(v) != "" else None
    except (ValueError, TypeError):
        return None


def build_upcoming_features(
    matchups_df: pd.DataFrame,
    db_path: str = DEFAULT_HISTORY_DB,
    event_date_override: str | None = None,
    strict_name_match: bool = True,
    allow_fuzzy_match: bool = False,
    k: float = DEFAULT_K,
    starting_elo: float = DEFAULT_STARTING_ELO,
) -> dict:
    """Return {'features': DataFrame, 'report': dict, 'match_review': list}."""
    base_numeric, _cat = official_step3c_features()
    matchups_df = matchups_df.reset_index(drop=True)  # positional row_status indexing
    schema = validate_matchup_schema(matchups_df, event_date_override)
    if not schema["ok"]:
        raise ValueError(
            "Matchup CSV failed schema validation: "
            f"missing={schema['missing_required_columns']}, forbidden={schema['forbidden_columns_present']}."
        )

    hist = load_history(db_path)
    all_fights = hist["fights"]
    stats = hist["stats"]
    lookup = hist["lookup"]

    # Resolve each matchup: event date, both fighters. Group buildable ones by date.
    resolved, match_review, failed = [], [], []
    for idx, row in matchups_df.iterrows():
        rs = schema["row_status"][idx]
        event_date = str(pd.to_datetime(event_date_override if event_date_override is not None else row["event_date"]).date())
        event_name = row.get("event_name")
        weight_class = row.get("weight_class") if "weight_class" in matchups_df.columns else None
        ma = match_fighter(row.get("fighter_a"), lookup, allow_fuzzy_match)
        mb = match_fighter(row.get("fighter_b"), lookup, allow_fuzzy_match)
        review = {
            "row_index": int(idx), "event_date": event_date, "event_name": event_name,
            "input_fighter_a": row.get("fighter_a"), "matched_fighter_a": ma["matched_name"],
            "fighter_a_id": ma["fighter_id"], "fighter_a_status": ma["status"],
            "input_fighter_b": row.get("fighter_b"), "matched_fighter_b": mb["matched_name"],
            "fighter_b_id": mb["fighter_id"], "fighter_b_status": mb["status"],
        }
        match_review.append(review)

        reasons = list(rs["reasons"])
        if ma["status"] != "matched":
            reasons.append(f"fighter_a {ma['status']}: {ma['detail']}")
        if mb["status"] != "matched":
            reasons.append(f"fighter_b {mb['status']}: {mb['detail']}")
        if reasons:
            failed.append({"row_index": int(idx), "event_date": event_date, "event_name": event_name,
                           "input_fighter_a": row.get("fighter_a"), "input_fighter_b": row.get("fighter_b"),
                           "reasons": reasons})
            continue
        resolved.append({"row_index": int(idx), "event_date": event_date, "event_name": event_name,
                         "weight_class": weight_class, "a": ma, "b": mb})

    # Build features per event date (one builder pass per card).
    built_rows = []
    time_safety = []
    syn_counter = 0
    by_date: dict[str, list[dict]] = {}
    for r in resolved:
        by_date.setdefault(r["event_date"], []).append(r)

    for event_date, cards in by_date.items():
        real_before = [f for f in all_fights if str(f["date"]) < event_date]
        max_real_date = max((str(f["date"]) for f in real_before), default=None)
        _snaps, ratings = run_elo(real_before, k=k, starting_elo=starting_elo)

        synth = []
        syn_ids = []
        for c in cards:
            syn_counter += 1
            fid = _SYNTHETIC_FIGHT_ID_BASE - syn_counter
            a, b = c["a"], c["b"]
            pre_a = ratings.get(a["fighter_id"], starting_elo)
            pre_b = ratings.get(b["fighter_id"], starting_elo)
            synth.append(_synthetic_fight(fid, event_date, c["event_name"], c["weight_class"], a, b, pre_a, pre_b))
            syn_ids.append(fid)
            c["_synthetic_fight_id"] = fid

        combined = real_before + synth
        base = build_feature_rows(combined)
        step3b = build_step3b_rows(combined, stats)
        against = build_against_rows(combined, stats)
        merged = base.merge(step3b, on="fight_id", how="left", validate="1:1")
        merged = merged.merge(against, on="fight_id", how="left", validate="1:1")
        merged = add_matchup_features(merged)

        syn_rows = merged[merged["fight_id"].isin(syn_ids)].copy()
        # Time-safety assertions for this card.
        assert max_real_date is None or max_real_date < event_date, "history leaked >= event_date"
        assert not any(str(d) >= event_date for d in
                       (f["date"] for f in real_before)), "a historical fight is not strictly before event_date"
        for c in cards:
            row = syn_rows[syn_rows["fight_id"] == c["_synthetic_fight_id"]].iloc[0]
            built_rows.append((c, row))
            time_safety.append({
                "row_index": c["row_index"], "event_date": event_date,
                "max_historical_fight_date_used": max_real_date,
                "cutoff_strictly_before_event": (max_real_date is None or max_real_date < event_date),
                "fighter_a_prior_fights": int(row["fighter_a_prior_fights"]),
                "fighter_b_prior_fights": int(row["fighter_b_prior_fights"]),
            })

    features_df = _assemble_output(built_rows, base_numeric)
    report = _build_report(matchups_df, features_df, match_review, failed, time_safety,
                           resolved, base_numeric, db_path, schema)
    return {"features": features_df, "report": report, "match_review": match_review}


def _assemble_output(built_rows: list[tuple], base_numeric: list[str]) -> pd.DataFrame:
    out_rows = []
    low_history_flags = ("fighter_a_no_prior_stats", "fighter_b_no_prior_stats", "matchup_history_missing")
    for c, row in built_rows:
        rec = {
            "event_date": c["event_date"], "event_name": c["event_name"],
            "fighter_a": row["fighter_a"], "fighter_b": row["fighter_b"],
            "weight_class": c.get("weight_class"),
        }
        for col in base_numeric:
            rec[col] = row.get(col)
        out_rows.append(rec)
    columns = ["event_date", "event_name", "fighter_a", "fighter_b", "weight_class"] + list(base_numeric)
    df = pd.DataFrame(out_rows, columns=columns)
    # Hard guarantee: no result/leakage column can escape into the feature CSV.
    leaked = sorted(FORBIDDEN_OUTPUT_COLUMNS & set(df.columns))
    if leaked:
        raise AssertionError(f"forbidden result columns leaked into output: {leaked}")
    return df


# ---------------------------------------------------------------------------
# Report + Step 6B validation
# ---------------------------------------------------------------------------

def validate_output_for_step6b(features_df: pd.DataFrame) -> dict:
    base_numeric, _ = official_step3c_features()
    v = validate_prediction_input(features_df, base_numeric)
    return {
        "ok": v["ok"],
        "missing_required_features": v["missing_required_features"],
        "forbidden_result_columns": v["forbidden_result_columns"],
        "market_columns": v["market_columns"],
        "missing_id_columns": v["missing_id_columns"],
        "extra_ignored_columns": v["extra_ignored_columns"],
        "n_valid_rows": v["n_valid_rows"],
        "n_error_rows": v["n_error_rows"],
    }


def _missing_feature_summary(features_df: pd.DataFrame, base_numeric: list[str]) -> dict:
    summary = {}
    for col in base_numeric:
        n_missing = int(features_df[col].isna().sum()) if col in features_df.columns else len(features_df)
        if n_missing:
            summary[col] = n_missing
    return summary


def _build_report(matchups_df, features_df, match_review, failed, time_safety, resolved,
                  base_numeric, db_path, schema) -> dict:
    unmatched = sorted({
        r["input_fighter_a"] for r in match_review if r["fighter_a_status"] == "unmatched"
    } | {r["input_fighter_b"] for r in match_review if r["fighter_b_status"] == "unmatched"})
    ambiguous = sorted({
        r["input_fighter_a"] for r in match_review if r["fighter_a_status"] == "ambiguous"
    } | {r["input_fighter_b"] for r in match_review if r["fighter_b_status"] == "ambiguous"})
    step6b_validation = validate_output_for_step6b(features_df) if len(features_df) else {
        "ok": None, "note": "no rows built"}
    low_history = [t for t in time_safety if t["fighter_a_prior_fights"] < 3 or t["fighter_b_prior_fights"] < 3]

    return {
        "generated_at": _now_utc(),
        "run": "step6c_upcoming_feature_builder",
        "history_input": db_path,
        "changes_data_processed": False,
        "changes_official_baseline_file": False,
        "n_matchup_rows": int(len(matchups_df)),
        "n_feature_rows_built": int(len(features_df)),
        "n_failed_rows": len(failed),
        "failed_rows": failed,
        "unmatched_fighters": unmatched,
        "ambiguous_fighters": ambiguous,
        "match_review": match_review,
        "time_safety": time_safety,
        "low_history_matchups": low_history,
        "feature_columns_generated": list(base_numeric),
        "n_feature_columns": len(base_numeric),
        "missing_feature_summary": _missing_feature_summary(features_df, base_numeric) if len(features_df) else {},
        "step6b_validation": step6b_validation,
        "data_quality_warnings": _data_quality_warnings(features_df, failed, ambiguous, low_history),
        "schema_validation": {k: v for k, v in schema.items() if k != "row_status"},
        "interpretation_notes": [
            "Features are recomputed with the exact Step 3/3B/3C builders using only fights strictly before event_date.",
            "Debut / low-history fighters get training's missing-value policy (honest nulls + no_prior_stats flags); the model pipeline imputes them.",
            "Unmatched or ambiguous fighters are reported as failed rows and never guessed.",
            "This builder does not scrape and adds no odds/markets; it reads the existing history DB only.",
        ],
    }


def _data_quality_warnings(features_df, failed, ambiguous, low_history) -> list[str]:
    w = []
    if failed:
        w.append(f"{len(failed)} matchup row(s) failed and were not built (see failed_rows).")
    if ambiguous:
        w.append(f"{len(ambiguous)} fighter name(s) are ambiguous in the history DB: {ambiguous}.")
    if low_history:
        w.append(f"{len(low_history)} built row(s) involve a fighter with < 3 prior fights (low-history/debut).")
    if len(features_df):
        allnull = [c for c in features_df.columns if features_df[c].isna().all()]
        if allnull:
            w.append(f"columns entirely null across built rows: {allnull}.")
    return w


# ---------------------------------------------------------------------------
# Orchestrator (CLI entry)
# ---------------------------------------------------------------------------

def run_build(
    matchups_csv: str,
    output: str = DEFAULT_OUTPUT,
    history_input: str = DEFAULT_HISTORY_DB,
    output_dir: str = DEFAULT_REPORT_DIR,
    event_date: str | None = None,
    strict_name_match: bool = True,
    allow_fuzzy_match: bool = False,
    review_matches_output: str | None = None,
    overwrite: bool = False,
    validate_for_step6b: bool = True,
) -> dict:
    out_path = Path(output)
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output feature file already exists at {out_path}. Pass overwrite=True (--overwrite) to replace it.")

    matchups_df = pd.read_csv(matchups_csv)
    result = build_upcoming_features(
        matchups_df, db_path=history_input, event_date_override=event_date,
        strict_name_match=strict_name_match, allow_fuzzy_match=allow_fuzzy_match,
    )
    features_df, report, match_review = result["features"], result["report"], result["match_review"]
    report["matchup_input"] = matchups_csv
    report["output_feature_path"] = str(out_path)

    if validate_for_step6b and len(features_df) and not report["step6b_validation"]["ok"]:
        raise ValueError(
            "Built features FAILED Step 6B validation: "
            f"{report['step6b_validation']}. Not writing an invalid feature file.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(out_path, index=False)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"step6c_feature_build_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    with open(out_dir / f"{stem}.json", "w") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    (out_dir / f"{stem}.md").write_text(_render_markdown(report))
    if review_matches_output:
        Path(review_matches_output).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(match_review).to_csv(review_matches_output, index=False)

    report["report_json"] = str(out_dir / f"{stem}.json")
    report["report_md"] = str(out_dir / f"{stem}.md")
    report["next_command"] = (
        f"python scripts/run_step6b_live_predictions.py --input {out_path} "
        f"--ledger data/live/live_predictions.csv --output-dir reports/live")
    return report


def _fmt_next_command(output: str) -> str:
    return (f"python scripts/run_step6b_live_predictions.py --input {output} "
            f"--ledger data/live/live_predictions.csv --output-dir reports/live")


def _render_markdown(report: dict) -> str:
    lines = ["# Step 6C: Upcoming-Card Feature Build", "", f"Generated: {report['generated_at']}", "",
             "Builds model-ready Step 3C features for an upcoming card from a simple matchup CSV, using only "
             "fights strictly before each event date. No scraping, no odds/markets. The official model and "
             "`data/processed/` are unchanged.", ""]
    lines.append(f"- Matchup input: `{report.get('matchup_input')}` | History: `{report['history_input']}`")
    lines.append(f"- Output features: `{report.get('output_feature_path')}`")
    lines.append(f"- Matchup rows: {report['n_matchup_rows']} | built: **{report['n_feature_rows_built']}** | "
                 f"failed: **{report['n_failed_rows']}**")
    v = report["step6b_validation"]
    lines.append(f"- Step 6B validation: **{'PASS' if v.get('ok') else v.get('ok')}** "
                 f"(missing features: {v.get('missing_required_features')}, forbidden: {v.get('forbidden_result_columns')})")
    lines.append(f"- Feature columns generated: {report['n_feature_columns']}")
    lines.append("")
    if report["unmatched_fighters"]:
        lines.append(f"**Unmatched fighters:** {report['unmatched_fighters']}")
    if report["ambiguous_fighters"]:
        lines.append(f"**Ambiguous fighters:** {report['ambiguous_fighters']}")
    lines.append("")
    lines.append("## Match Review")
    lines.append("")
    lines.append("| event_date | input A | matched A | status A | input B | matched B | status B |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in report["match_review"][:30]:
        lines.append(f"| {r['event_date']} | {r['input_fighter_a']} | {r['matched_fighter_a']} | {r['fighter_a_status']} | "
                     f"{r['input_fighter_b']} | {r['matched_fighter_b']} | {r['fighter_b_status']} |")
    lines.append("")
    if report["failed_rows"]:
        lines.append("## Failed Rows")
        lines.append("")
        for f in report["failed_rows"]:
            lines.append(f"- row {f['row_index']} ({f['input_fighter_a']} vs {f['input_fighter_b']}): {', '.join(f['reasons'])}")
        lines.append("")
    lines.append("## Time-Safety")
    lines.append("")
    lines.append("| event_date | max historical date used | strictly before | A prior fights | B prior fights |")
    lines.append("|---|---|:---:|---:|---:|")
    for t in report["time_safety"][:30]:
        lines.append(f"| {t['event_date']} | {t['max_historical_fight_date_used']} | {t['cutoff_strictly_before_event']} | "
                     f"{t['fighter_a_prior_fights']} | {t['fighter_b_prior_fights']} |")
    lines.append("")
    if report["missing_feature_summary"]:
        lines.append("## Missing Feature Summary (built rows)")
        lines.append("")
        for col, n in list(report["missing_feature_summary"].items())[:20]:
            lines.append(f"- {col}: {n} null")
        lines.append("")
    if report["data_quality_warnings"]:
        lines.append("## Data-Quality Warnings")
        lines.append("")
        for w in report["data_quality_warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    lines.append("## Next Step")
    lines.append("")
    lines.append("Feed the built features straight into Step 6B:")
    lines.append("")
    lines.append("```")
    lines.append(report.get("next_command", _fmt_next_command(report.get("output_feature_path", "OUTPUT.csv"))))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)
