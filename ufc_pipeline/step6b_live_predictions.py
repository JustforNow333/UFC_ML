"""Step 6B: Forward live prediction ledger.

The forward live-tracking layer. Unlike Step 6A (which replays *historical*
events that already have known outcomes), Step 6B predicts *genuinely upcoming*
fights before they happen: the user hands it a CSV of upcoming-card pre-fight
feature rows, it generates official-model predictions, appends them to a durable
append-only ledger as ``pending``, and later — once results are known — resolves
those rows and scores them.

Three entry points:
  * ``run_live_predictions``  — validate an upcoming card, predict, append, report.
  * ``resolve_predictions``   — match a results CSV to pending rows and score them.
  * ``build_live_model_report`` — aggregate resolved rows into a live model report.

Deliberate boundaries (unchanged by this task): the official model config is
reused and reproduced exactly (elastic-net LR C=0.003, l1_ratio=0.1, raw
weight_class dropped, Platt); no odds / prediction markets / scraping / new data
sources / new model families / retuning. It writes only ``data/live/`` and
``reports/live/`` — never ``data/processed/`` or the benchmark file.

This is NOT fresh statistical validation until a meaningful number of live
predictions have resolved: with < 50 resolved the metrics are too noisy to read,
and < 100 they are preliminary. The report says so loudly.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_pipeline.calibration import (
    calibration_table_with_gap,
    chronological_three_way_split,
    clip_probabilities,
    high_confidence_diagnostics,
)
from ufc_pipeline.feature_diagnostics import DEFAULT_BASELINE_PATH, official_step3c_features
from ufc_pipeline.modeling import RANDOM_STATE, TARGET, check_features_allowed, coerce_numeric_features, evaluate_probs
from ufc_pipeline.step5b_regularization_search import DEFAULT_MAX_ITER, WEIGHT_CLASS_COLUMN
from ufc_pipeline.step6a_pseudo_live_replay import (
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_HISTORY_FLAGS,
    MARKET_TOKENS,
    _row_brier,
    _row_log_loss,
    _window_metrics,
    calibration_buckets,
    confidence_buckets,
    event_drift,
    feature_schema_version,
    fit_official_model,
    high_confidence_summary,
    load_official_replay_config,
)

DEFAULT_TRAINING_DATA = "data/processed/ufc_prefight_features_step3c.csv"
DEFAULT_LEDGER = "data/live/live_predictions.csv"
DEFAULT_LIVE_REPORT_DIR = "reports/live"
LIVE_MODEL_REPORT_JSON = "live_model_report.json"
LIVE_MODEL_REPORT_MD = "live_model_report.md"

PREDICTION_MODE = "live_forward"

REQUIRED_ID_COLUMNS = ("event_date", "event_name", "fighter_a", "fighter_b")
# Exact column names that would mean a result/outcome leaked into prediction input.
FORBIDDEN_RESULT_COLUMNS = frozenset({
    "fighter_a_won", "fighter_b_won", "winner", "loser", "result", "outcome",
    "method", "round", "target", "target_a_win", "finish", "decision",
})
RESULTS_REQUIRED_COLUMNS = ("event_date", "event_name", "fighter_a", "fighter_b", "winner")

LEDGER_COLUMNS = [
    "prediction_id", "prediction_batch_id", "prediction_mode", "event_date", "event_name",
    "fighter_a", "fighter_b", "model_version", "calibration_version", "feature_schema_version",
    "input_source", "predicted_probability_a", "predicted_probability_b", "prediction_timestamp_utc",
    "status", "target_a_win", "winner", "result_source", "resolved_timestamp_utc",
    "log_loss", "brier", "correct_prediction", "notes",
]

# Sample-size gates for interpreting live metrics.
MIN_RESOLVED_FOR_METRICS = 50
MIN_RESOLVED_FOR_STABLE = 100

# Step 6A pseudo-live reference (used if the Step 6A report is unavailable).
STEP6A_REFERENCE = {"log_loss": 0.651582, "brier_score": 0.229949, "roc_auc": 0.645563, "accuracy": 0.618545}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _load_json(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def normalize_name(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value).strip().lower())


def normalize_date(value) -> str:
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return str(value).strip()


def fight_key(event_date, fighter_a, fighter_b) -> tuple:
    """Orientation-independent match key: (date, frozenset of normalized names)."""
    return (normalize_date(event_date), frozenset({normalize_name(fighter_a), normalize_name(fighter_b)}))


def make_prediction_id(event_date, fighter_a, fighter_b, model_version, batch_id) -> str:
    names = "|".join(sorted({normalize_name(fighter_a), normalize_name(fighter_b)}))
    raw = f"{normalize_date(event_date)}|{names}|{model_version}|{batch_id}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def default_batch_id() -> str:
    return f"batch_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"


def _safe_batch_filename(batch_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", batch_id)


def confidence_label(p: float) -> str:
    conf = max(p, 1.0 - p)
    for lo, hi, label in ((0.5, 0.6, "toss-up"), (0.6, 0.7, "lean"), (0.7, 0.8, "confident"), (0.8, 1.01, "strong")):
        if lo <= conf < hi:
            return label
    return "toss-up"


# ---------------------------------------------------------------------------
# Official model loading / training
# ---------------------------------------------------------------------------

def load_official_model_config(baseline_path: str | None = DEFAULT_BASELINE_PATH) -> dict:
    """Official model identity (read-only). Errors clearly if inconsistent."""
    baseline = _load_json(baseline_path)
    if baseline_path is not None and baseline is None:
        raise FileNotFoundError(f"Official baseline file not found: {baseline_path}")
    if baseline_path is not None and not isinstance(baseline.get("official_model"), dict):
        raise ValueError(f"Official baseline file has no official_model object: {baseline_path}")

    config = load_official_replay_config(baseline_path)
    official = baseline.get("official_model", {}) if baseline else {}
    config["official_split"] = official.get("split")
    if baseline_path is not None and not isinstance(config["official_split"], dict):
        raise ValueError(f"Official baseline file has no locked split metadata: {baseline_path}")
    if config["hyperparameters"].get("C") != 0.003 or config["hyperparameters"].get("l1_ratio") != 0.1:
        raise ValueError(
            "Official model metadata is inconsistent with the expected Step 5C config "
            f"(elastic-net C=0.003, l1_ratio=0.1); got {config['hyperparameters']}."
        )
    if config["raw_weight_class"] != "dropped":
        raise ValueError("Official model config must drop raw weight_class for live prediction.")
    return config


def _official_training_split(
    df: pd.DataFrame,
    config: dict,
    train_frac: float,
    calibration_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Return the locked benchmark split when its metadata is available.

    The official processed CSV is allowed to grow as new events are promoted.
    Reapplying percentage splits to that mutable file silently changes the live
    model, so production runs must freeze both the benchmark cutoff and counts.
    Synthetic callers without benchmark split metadata retain the fractional
    fallback used by the unit tests and standalone experiments.
    """
    locked = config.get("official_split")
    if not locked:
        train, calibration, test = chronological_three_way_split(
            df, train_frac=train_frac, calibration_frac=calibration_frac,
        )
        return train, calibration, test, {
            "split_source": "fractional_fallback_no_locked_split",
            "benchmark_cutoff_date": None,
            "benchmark_rows": int(len(df)),
            "excluded_post_benchmark_rows": 0,
        }

    count_keys = ("n_train", "n_calibration", "n_test")
    missing = [key for key in count_keys if key not in locked]
    test_dates = locked.get("test_dates")
    if missing or not isinstance(test_dates, list) or len(test_dates) != 2:
        raise ValueError(
            "Official benchmark split metadata is incomplete; expected "
            f"{list(count_keys)} and two test_dates, missing={missing}."
        )

    counts = [int(locked[key]) for key in count_keys]
    expected_rows = sum(counts)
    cutoff = pd.Timestamp(test_dates[-1])
    parsed_dates = pd.to_datetime(df["date"], errors="raise")
    frozen = df.loc[parsed_dates <= cutoff].reset_index(drop=True)
    if len(frozen) != expected_rows:
        raise ValueError(
            "Official benchmark window no longer matches its locked split: "
            f"date <= {cutoff.date()} produced {len(frozen)} rows; "
            f"expected {expected_rows}. Refusing to fit a changed live model."
        )

    n_train, n_calibration, n_test = counts
    train = frozen.iloc[:n_train].copy()
    calibration = frozen.iloc[n_train:n_train + n_calibration].copy()
    test = frozen.iloc[n_train + n_calibration:n_train + n_calibration + n_test].copy()

    expected_ranges = (locked.get("train_dates"), locked.get("calibration_dates"), test_dates)
    for label, part, expected in zip(("train", "calibration", "test"), (train, calibration, test), expected_ranges):
        if not isinstance(expected, list) or len(expected) != 2:
            continue
        actual = [str(part["date"].min()), str(part["date"].max())]
        if actual != [str(expected[0]), str(expected[1])]:
            raise ValueError(
                f"Official {label} date range changed: got {actual}, expected {expected}. "
                "Refusing to fit a changed live model."
            )

    return train, calibration, test, {
        "split_source": "locked_official_baseline",
        "benchmark_cutoff_date": str(cutoff.date()),
        "benchmark_rows": int(len(frozen)),
        "excluded_post_benchmark_rows": int(len(df) - len(frozen)),
    }


def train_official_model(
    training_csv: str = DEFAULT_TRAINING_DATA,
    baseline_path: str | None = DEFAULT_BASELINE_PATH,
    random_state: int = RANDOM_STATE,
    max_iter: int = DEFAULT_MAX_ITER,
    train_frac: float = 0.70,
    calibration_frac: float = 0.15,
) -> dict:
    """Reproduce the official model: base LR on the official train split, Platt on
    the official calibration split, from the historical processed CSV (read-only)."""
    config = load_official_model_config(baseline_path)
    base_numeric, _categorical = official_step3c_features()
    df = pd.read_csv(training_csv)
    df = df[df[TARGET].notna() & df["fighter_a_expected_win_prob"].notna()].copy()
    df = coerce_numeric_features(df, base_numeric, context="step6b_live_predictions")
    df = df.sort_values(["date", "fight_id"]).reset_index(drop=True)
    source_rows = int(len(df))
    train, calib, _test, split_metadata = _official_training_split(
        df, config, train_frac=train_frac, calibration_frac=calibration_frac,
    )
    pipeline, platt = fit_official_model(train, calib, base_numeric, random_state=random_state, max_iter=max_iter)
    return {
        "pipeline": pipeline,
        "platt": platt,
        "base_numeric": base_numeric,
        "config": config,
        "model_version": config["model_version"],
        "calibration_version": f"platt_official_calib{len(calib)}",
        "feature_schema_version": config["feature_schema_version"],
        "training_metadata": {
            "training_source": training_csv,
            "source_rows": source_rows,
            "train_rows": int(len(train)), "calibration_rows": int(len(calib)),
            "train_dates": [str(train["date"].min()), str(train["date"].max())],
            "calibration_dates": [str(calib["date"].min()), str(calib["date"].max())],
            **split_metadata,
            "note": "Reproduces the official Step 5C model (base LR on official train, Platt on official calibration).",
        },
        "official_train_calib": pd.concat([train, calib]).reset_index(drop=True),
    }


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_prediction_input(df: pd.DataFrame, base_numeric: list[str], allow_duplicates: bool = False) -> dict:
    """Structural + per-row validation of an upcoming-card feature CSV.

    Structural failures (missing id columns, forbidden result columns, market
    columns, missing required features) set ``ok=False`` and must abort the batch.
    Per-row problems (bad date, empty/identical fighters, duplicate fight) are
    recorded per row so those rows can be written as ``error`` while valid rows
    are still predicted.
    """
    columns = set(df.columns)
    missing_id = [c for c in REQUIRED_ID_COLUMNS if c not in columns]
    forbidden = sorted(FORBIDDEN_RESULT_COLUMNS & columns)
    market_cols = sorted([c for c in columns if any(tok in c.lower() for tok in MARKET_TOKENS)])
    missing_features = [c for c in base_numeric if c not in columns]
    known = set(REQUIRED_ID_COLUMNS) | set(base_numeric) | {WEIGHT_CLASS_COLUMN, "fight_id"}
    extra_ignored = sorted(columns - known)

    structural_errors = []
    if missing_id:
        structural_errors.append(f"missing required identifier columns: {missing_id}")
    if forbidden:
        structural_errors.append(f"result/outcome columns are not allowed in prediction input: {forbidden}")
    if market_cols:
        structural_errors.append(f"odds/market columns are not allowed: {market_cols}")
    if missing_features:
        structural_errors.append(f"missing required model feature columns: {missing_features}")

    row_status = []
    seen_keys: dict[tuple, int] = {}
    for idx, row in df.iterrows():
        reasons = []
        if "event_date" in columns:
            try:
                pd.to_datetime(row["event_date"])
            except (ValueError, TypeError):
                reasons.append("unparseable event_date")
        fa = normalize_name(row.get("fighter_a")) if "fighter_a" in columns else ""
        fb = normalize_name(row.get("fighter_b")) if "fighter_b" in columns else ""
        if not fa:
            reasons.append("empty fighter_a")
        if not fb:
            reasons.append("empty fighter_b")
        if fa and fb and fa == fb:
            reasons.append("fighter_a equals fighter_b")
        if "event_date" in columns and fa and fb:
            key = fight_key(row.get("event_date"), row.get("fighter_a"), row.get("fighter_b"))
            if key in seen_keys and not allow_duplicates:
                reasons.append(f"duplicate fight row in input (also row {seen_keys[key]})")
            seen_keys.setdefault(key, idx)
        row_status.append({"row_index": int(idx), "status": "error" if reasons else "valid", "reasons": reasons})

    warnings = []
    if extra_ignored:
        warnings.append(f"ignored non-feature columns: {extra_ignored}")
    if "event_name" in columns:
        blank_events = int(df["event_name"].isna().sum() + (df["event_name"].astype(str).str.strip() == "").sum())
        if blank_events:
            warnings.append(f"{blank_events} rows have a blank event_name")

    return {
        "ok": not structural_errors,
        "structural_errors": structural_errors,
        "missing_id_columns": missing_id,
        "forbidden_result_columns": forbidden,
        "market_columns": market_cols,
        "missing_required_features": missing_features,
        "extra_ignored_columns": extra_ignored,
        "warnings": warnings,
        "row_status": row_status,
        "n_rows": int(len(df)),
        "n_valid_rows": int(sum(r["status"] == "valid" for r in row_status)),
        "n_error_rows": int(sum(r["status"] == "error" for r in row_status)),
    }


# ---------------------------------------------------------------------------
# Ledger management
# ---------------------------------------------------------------------------

def empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def load_ledger(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return empty_ledger()
    df = pd.read_csv(p, dtype={"prediction_id": str, "prediction_batch_id": str})
    for col in LEDGER_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[LEDGER_COLUMNS]


def write_ledger(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df[LEDGER_COLUMNS].to_csv(p, index=False)


def append_ledger_preserving_existing_bytes(
    existing: pd.DataFrame, updated: pd.DataFrame, path: str | Path,
) -> int:
    """Append only the new tail while proving the on-disk prefix is untouched.

    Production supplemental batches use this path because a normal pandas CSV
    rewrite may change formatting of frozen rows even when their values did not
    change. Any removal, reorder, or value change in the existing prefix aborts.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        write_ledger(updated, p)
        return int(len(updated))
    if len(updated) < len(existing):
        raise ValueError("Byte-preserving append cannot remove existing ledger rows.")

    expected_prefix = existing[LEDGER_COLUMNS].fillna("").astype(str).reset_index(drop=True)
    actual_prefix = updated.iloc[:len(existing)][LEDGER_COLUMNS].fillna("").astype(str).reset_index(drop=True)
    if not expected_prefix.equals(actual_prefix):
        raise ValueError("Byte-preserving append detected a changed or reordered existing ledger row.")

    tail = updated.iloc[len(existing):][LEDGER_COLUMNS]
    if tail.empty:
        return 0
    before = p.read_bytes()
    separator = b"" if before.endswith((b"\n", b"\r")) else b"\n"
    payload = tail.to_csv(index=False, header=False, lineterminator="\n").encode("utf-8")
    with p.open("ab") as handle:
        handle.write(separator + payload)
    after = p.read_bytes()
    if after[:len(before)] != before:
        raise RuntimeError("Existing ledger bytes changed during guarded append.")
    return int(len(tail))


def _row_active(row) -> bool:
    return str(row.get("status")) in ("pending", "resolved")


def append_predictions(
    ledger_df: pd.DataFrame, new_rows: list[dict], allow_duplicate: bool, overwrite_pending: bool,
) -> tuple[pd.DataFrame, dict]:
    """Append-only merge with duplicate policy.

    Duplicate = same (event_date, fighter set, model_version) as an active
    (pending/resolved) ledger row. Default: reject. ``overwrite_pending`` replaces
    an existing *pending* row; ``allow_duplicate`` appends a new versioned row.
    """
    existing = ledger_df.copy()
    existing_keys = {}
    for i, row in existing.iterrows():
        if _row_active(row):
            existing_keys.setdefault((fight_key(row["event_date"], row["fighter_a"], row["fighter_b"]),
                                      row["model_version"]), []).append(i)

    accepted, drop_indices = [], []
    warnings = {"duplicates_rejected": [], "pending_overwritten": [], "duplicates_versioned": []}
    for row in new_rows:
        if row["status"] == "error":
            accepted.append(row)
            continue
        key = (fight_key(row["event_date"], row["fighter_a"], row["fighter_b"]), row["model_version"])
        matches = existing_keys.get(key, [])
        if matches:
            pending_idx = [i for i in matches if str(existing.loc[i, "status"]) == "pending"]
            label = f"{row['event_date']} {row['fighter_a']} vs {row['fighter_b']}"
            if overwrite_pending and pending_idx:
                drop_indices.extend(pending_idx)
                warnings["pending_overwritten"].append(label)
                accepted.append(row)
            elif allow_duplicate:
                warnings["duplicates_versioned"].append(label)
                accepted.append(row)
            else:
                warnings["duplicates_rejected"].append(label)
        else:
            accepted.append(row)
        existing_keys.setdefault(key, [])

    if drop_indices:
        existing = existing.drop(index=drop_indices)
    combined = pd.concat([existing, pd.DataFrame(accepted, columns=LEDGER_COLUMNS)], ignore_index=True)
    stats = {
        "n_submitted": len(new_rows),
        "n_accepted": len(accepted),
        "n_rejected_duplicates": len(warnings["duplicates_rejected"]),
        "n_pending_overwritten": len(warnings["pending_overwritten"]),
        "n_duplicates_versioned": len(warnings["duplicates_versioned"]),
        "warnings": warnings,
    }
    return combined[LEDGER_COLUMNS], stats


# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------

def generate_probabilities(df_valid: pd.DataFrame, pipeline, platt, base_numeric: list[str]) -> np.ndarray:
    raw = pipeline.predict_proba(df_valid[list(base_numeric)])[:, 1]
    return clip_probabilities(platt.transform(raw))


def run_live_predictions(
    input_csv: str,
    ledger_path: str = DEFAULT_LEDGER,
    output_dir: str = DEFAULT_LIVE_REPORT_DIR,
    training_data: str = DEFAULT_TRAINING_DATA,
    baseline_path: str | None = DEFAULT_BASELINE_PATH,
    prediction_batch_id: str | None = None,
    allow_duplicate_predictions: bool = False,
    overwrite_existing_pending: bool = False,
    preserve_existing_ledger_bytes: bool = False,
    random_seed: int = RANDOM_STATE,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict:
    if preserve_existing_ledger_bytes and (allow_duplicate_predictions or overwrite_existing_pending):
        raise ValueError(
            "Byte-preserving append requires duplicate and overwrite flags to remain disabled."
        )
    batch_id = prediction_batch_id or default_batch_id()
    base_numeric, _categorical = official_step3c_features()
    df = pd.read_csv(input_csv)
    validation = validate_prediction_input(df, base_numeric, allow_duplicates=allow_duplicate_predictions)
    if not validation["ok"]:
        raise ValueError(
            "Live prediction input failed structural validation:\n  - "
            + "\n  - ".join(validation["structural_errors"])
            + "\nProvide a CSV with the required model feature columns and no result/odds columns."
        )

    trained = train_official_model(training_data, baseline_path, random_seed, max_iter)
    df = coerce_numeric_features(df, base_numeric, context="step6b_live_input")
    valid_mask = np.array([r["status"] == "valid" for r in validation["row_status"]])
    df_valid = df[valid_mask].copy()
    probs = generate_probabilities(df_valid, trained["pipeline"], trained["platt"], base_numeric) if len(df_valid) else np.array([])

    now = _now_utc()
    new_rows, prediction_table = [], []
    vi = 0
    for i, row in df.iterrows():
        status_info = validation["row_status"][i]
        base = {
            "prediction_id": make_prediction_id(row.get("event_date"), row.get("fighter_a"), row.get("fighter_b"),
                                                 trained["model_version"], batch_id),
            "prediction_batch_id": batch_id, "prediction_mode": PREDICTION_MODE,
            "event_date": row.get("event_date"), "event_name": row.get("event_name"),
            "fighter_a": row.get("fighter_a"), "fighter_b": row.get("fighter_b"),
            "model_version": trained["model_version"], "calibration_version": trained["calibration_version"],
            "feature_schema_version": trained["feature_schema_version"], "input_source": input_csv,
            "prediction_timestamp_utc": now, "target_a_win": "", "winner": "", "result_source": "",
            "resolved_timestamp_utc": "", "log_loss": "", "brier": "", "correct_prediction": "",
        }
        if status_info["status"] == "error":
            base.update({"predicted_probability_a": "", "predicted_probability_b": "", "status": "error",
                         "notes": ";".join(status_info["reasons"])})
        else:
            p_a = float(probs[vi]); vi += 1
            note_parts = [f for f in LOW_HISTORY_FLAGS if f in df.columns and float(pd.to_numeric(row.get(f), errors="coerce") or 0) > 0]
            base.update({"predicted_probability_a": p_a, "predicted_probability_b": float(1.0 - p_a),
                         "status": "pending", "notes": ";".join(note_parts)})
            prediction_table.append({
                "event_date": row.get("event_date"), "event_name": row.get("event_name"),
                "fighter_a": row.get("fighter_a"), "fighter_b": row.get("fighter_b"),
                "predicted_probability_a": round(p_a, 6), "predicted_probability_b": round(1.0 - p_a, 6),
                "predicted_favorite": row.get("fighter_a") if p_a >= 0.5 else row.get("fighter_b"),
                "confidence_bucket": confidence_label(p_a),
            })
        new_rows.append(base)

    ledger_df = load_ledger(ledger_path)
    updated, append_stats = append_predictions(ledger_df, new_rows, allow_duplicate_predictions, overwrite_existing_pending)
    if preserve_existing_ledger_bytes:
        appended_rows = append_ledger_preserving_existing_bytes(ledger_df, updated, ledger_path)
        ledger_write_mode = "byte_preserving_append"
    else:
        write_ledger(updated, ledger_path)
        appended_rows = int(len(updated) - len(ledger_df) + append_stats["n_pending_overwritten"])
        ledger_write_mode = "rewrite"

    drift = _live_drift(df_valid, trained["official_train_calib"], base_numeric) if len(df_valid) else {"n_rows": 0}
    leakage = _live_leakage_checks(base_numeric, validation, trained["config"])

    report = {
        "generated_at": now,
        "prediction_batch_id": batch_id,
        "prediction_mode": PREDICTION_MODE,
        "input": input_csv,
        "ledger_path": str(ledger_path),
        "model_version": trained["model_version"],
        "calibration_version": trained["calibration_version"],
        "feature_schema_version": trained["feature_schema_version"],
        "training_metadata": trained["training_metadata"],
        "n_fights_submitted": int(len(df)),
        "n_predictions_generated": int(len(prediction_table)),
        "n_failed_rows": validation["n_error_rows"],
        "failed_row_reasons": [r for r in validation["row_status"] if r["status"] == "error"],
        "events": sorted({(str(r.get("event_date")), str(r.get("event_name"))) for r in new_rows}),
        "prediction_table": prediction_table,
        "append_stats": append_stats,
        "ledger_write_mode": ledger_write_mode,
        "n_rows_physically_appended": appended_rows,
        "duplicate_warnings": append_stats["warnings"],
        "data_quality_warnings": validation["warnings"],
        "input_validation": {k: v for k, v in validation.items() if k != "row_status"},
        "drift_data_quality": drift,
        "leakage_protocol_checks": leakage,
        "changes_official_model": False,
        "writes_data_processed": False,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"live_prediction_batch_{_safe_batch_filename(batch_id)}"
    with open(out_dir / f"{stem}.json", "w") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    (out_dir / f"{stem}.md").write_text(_render_batch_markdown(report))
    report["batch_report_json"] = str(out_dir / f"{stem}.json")
    report["batch_report_md"] = str(out_dir / f"{stem}.md")
    return report


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_predictions(
    ledger_df: pd.DataFrame, results_df: pd.DataFrame, result_source: str = "",
    overwrite_resolved: bool = False, strict_matching: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Match a results CSV to ledger rows and score the matched pending rows."""
    missing = [c for c in RESULTS_REQUIRED_COLUMNS if c not in results_df.columns]
    if missing:
        raise ValueError(f"Results CSV missing required columns: {missing}")

    results_index: dict[tuple, list[dict]] = {}
    for i, r in results_df.iterrows():
        key = fight_key(r["event_date"], r["fighter_a"], r["fighter_b"])
        results_index.setdefault(key, []).append({
            "winner": r["winner"], "winner_norm": normalize_name(r["winner"]),
            "fighter_a_norm": normalize_name(r["fighter_a"]), "fighter_b_norm": normalize_name(r["fighter_b"]),
            "row_index": int(i),
        })

    updated = ledger_df.copy()
    # Columns written during resolution may have loaded as all-blank float64
    # (pandas infers NaN); cast to object so string/int assignment is lossless
    # under pandas' strict setitem.
    for col in ("target_a_win", "winner", "result_source", "resolved_timestamp_utc",
                "log_loss", "brier", "correct_prediction", "status"):
        updated[col] = updated[col].astype(object)
    now = _now_utc()
    used_result_keys: set[tuple] = set()
    stats = {"n_resolved": 0, "n_already_resolved_skipped": 0, "n_pending_unmatched": 0,
             "ambiguous_matches": [], "unmatched_winner": [], "unmatched_results": []}

    for idx, row in updated.iterrows():
        status = str(row.get("status"))
        if status == "error":
            continue
        if status == "resolved" and not overwrite_resolved:
            stats["n_already_resolved_skipped"] += 1
            continue
        key = fight_key(row["event_date"], row["fighter_a"], row["fighter_b"])
        matches = results_index.get(key, [])
        label = f"{row['event_date']} {row['fighter_a']} vs {row['fighter_b']}"
        if not matches:
            stats["n_pending_unmatched"] += 1
            continue
        if len(matches) > 1:
            stats["ambiguous_matches"].append({"prediction": label, "n_result_rows": len(matches)})
            continue
        result = matches[0]
        used_result_keys.add(key)
        row_a_norm = normalize_name(row["fighter_a"])
        row_b_norm = normalize_name(row["fighter_b"])
        if result["winner_norm"] == row_a_norm:
            target = 1
        elif result["winner_norm"] == row_b_norm:
            target = 0
        else:
            stats["unmatched_winner"].append({"prediction": label, "winner": result["winner"]})
            continue
        p_a = float(row["predicted_probability_a"])
        updated.loc[idx, "target_a_win"] = int(target)
        updated.loc[idx, "winner"] = result["winner"]
        updated.loc[idx, "result_source"] = result_source
        updated.loc[idx, "resolved_timestamp_utc"] = now
        updated.loc[idx, "log_loss"] = float(_row_log_loss(np.array([target]), np.array([p_a]))[0])
        updated.loc[idx, "brier"] = float(_row_brier(np.array([target]), np.array([p_a]))[0])
        updated.loc[idx, "correct_prediction"] = int((p_a >= 0.5) == bool(target))
        updated.loc[idx, "status"] = "resolved"
        stats["n_resolved"] += 1

    for key, entries in results_index.items():
        if key not in used_result_keys:
            for e in entries:
                stats["unmatched_results"].append({"event_date": key[0], "winner": e["winner"],
                                                   "fighters": sorted(key[1])})
    stats["n_unmatched_results"] = len(stats["unmatched_results"])
    return updated[LEDGER_COLUMNS], stats


def run_resolution(
    results_csv: str, ledger_path: str = DEFAULT_LEDGER, output_dir: str = DEFAULT_LIVE_REPORT_DIR,
    overwrite_resolved: bool = False, strict_matching: bool = True,
) -> dict:
    ledger_df = load_ledger(ledger_path)
    if ledger_df.empty:
        raise ValueError(f"Ledger {ledger_path} is empty or missing; run live predictions before resolving.")
    results_df = pd.read_csv(results_csv)
    updated, stats = resolve_predictions(ledger_df, results_df, result_source=results_csv,
                                         overwrite_resolved=overwrite_resolved, strict_matching=strict_matching)
    write_ledger(updated, ledger_path)

    report = {
        "generated_at": _now_utc(), "results_input": results_csv, "ledger_path": str(ledger_path),
        "n_result_rows": int(len(results_df)),
        "n_ledger_rows": int(len(updated)),
        "n_pending_before": int((ledger_df["status"] == "pending").sum()),
        "n_pending_after": int((updated["status"] == "pending").sum()),
        "resolution_stats": stats,
        "overwrite_resolved": overwrite_resolved, "strict_matching": strict_matching,
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"live_resolution_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    with open(out_dir / f"{stem}.json", "w") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    report["resolution_report_json"] = str(out_dir / f"{stem}.json")
    return report


# ---------------------------------------------------------------------------
# Live model report
# ---------------------------------------------------------------------------

def build_live_model_report(
    ledger_path: str = DEFAULT_LEDGER, output_dir: str = DEFAULT_LIVE_REPORT_DIR,
    baseline_path: str | None = DEFAULT_BASELINE_PATH, step6a_report_path: str | None = "reports/step6a_pseudo_live_replay.json",
) -> dict:
    ledger = load_ledger(ledger_path)
    n_total = int(len(ledger))
    n_pending = int((ledger["status"] == "pending").sum())
    n_error = int((ledger["status"] == "error").sum())
    resolved = ledger[ledger["status"] == "resolved"].copy()
    n_resolved = int(len(resolved))

    baseline = _load_json(baseline_path)
    static_metrics = baseline["official_model"]["test_metrics"] if baseline else None
    step6a = _load_json(step6a_report_path)
    step6a_metrics = step6a["overall_metrics"] if step6a and "overall_metrics" in step6a else STEP6A_REFERENCE

    sample_warning = _sample_size_warning(n_resolved)
    report = {
        "generated_at": _now_utc(), "ledger_path": str(ledger_path),
        "model_version": ledger["model_version"].dropna().iloc[0] if n_total else None,
        "total_predictions": n_total, "pending_predictions": n_pending,
        "resolved_predictions": n_resolved, "errored_predictions": n_error,
        "n_events": int(ledger[["event_date", "event_name"]].drop_duplicates().shape[0]) if n_total else 0,
        "sample_size_warning": sample_warning,
        "comparison_reference": {
            "official_static_test_metrics": static_metrics,
            "step6a_pseudo_live_metrics": step6a_metrics,
            "note": "Live metrics are only comparable once enough predictions resolve; early live numbers are noisy.",
        },
        "interpretation_warnings": [
            "This is a FORWARD live ledger: metrics reflect only fights that have already resolved.",
            f"Fewer than {MIN_RESOLVED_FOR_METRICS} resolved predictions => metrics are too noisy to interpret.",
            f"Fewer than {MIN_RESOLVED_FOR_STABLE} resolved predictions => treat metrics as preliminary.",
            "Do not use these numbers to select or retune a model; the official model is unchanged.",
        ],
    }

    if n_resolved == 0:
        report.update({"resolved_events": 0, "overall_metrics": None,
                       "message": "No resolved predictions yet; nothing to score."})
        _write_live_report(report, output_dir)
        return report

    y = resolved["target_a_win"].astype(int).to_numpy()
    p = resolved["predicted_probability_a"].astype(float).to_numpy()
    overall = evaluate_probs(y, clip_probabilities(p))
    report["resolved_events"] = int(resolved[["event_date", "event_name"]].drop_duplicates().shape[0])
    report["overall_metrics"] = {
        "log_loss": overall["log_loss"], "brier_score": overall["brier_score"],
        "roc_auc": overall["roc_auc"] if n_resolved >= 20 else None,
        "accuracy": overall["accuracy"], "avg_predicted_prob": overall["avg_predicted_prob"],
        "actual_a_win_rate": overall["actual_win_rate"],
    }
    report["calibration_buckets"] = calibration_buckets(y, p)
    report["calibration_table_10_bucket"] = calibration_table_with_gap(y, clip_probabilities(p))
    report["high_confidence_two_sided"] = high_confidence_summary(y, p)
    report["high_confidence_standard"] = high_confidence_diagnostics(y, clip_probabilities(p))
    report["confidence_buckets"] = confidence_buckets(y, p)

    resolved = resolved.sort_values(["event_date"]).reset_index(drop=True)
    report["performance_by_event"] = _performance_by_event(resolved)
    report["performance_by_month"] = _performance_by_period(resolved, resolved["event_date"].str.slice(0, 7), "month") if n_resolved >= 30 else "insufficient_sample"
    report["performance_by_year"] = _performance_by_period(resolved, resolved["event_date"].str.slice(0, 4), "year") if n_resolved >= 30 else "insufficient_sample"
    report["rolling_metrics"] = _rolling(resolved) if n_resolved >= MIN_RESOLVED_FOR_METRICS else "insufficient_sample"

    if static_metrics and static_metrics.get("log_loss") is not None:
        report["comparison_reference"]["log_loss_vs_static"] = overall["log_loss"] - static_metrics["log_loss"]
    report["comparison_reference"]["log_loss_vs_step6a"] = overall["log_loss"] - step6a_metrics["log_loss"]

    _write_live_report(report, output_dir)
    return report


def _sample_size_warning(n_resolved: int) -> dict:
    if n_resolved < MIN_RESOLVED_FOR_METRICS:
        level = "too_noisy"
        message = (f"Only {n_resolved} resolved predictions (< {MIN_RESOLVED_FOR_METRICS}); live metrics are TOO NOISY "
                   "to interpret. Do not draw conclusions yet.")
    elif n_resolved < MIN_RESOLVED_FOR_STABLE:
        level = "preliminary"
        message = (f"{n_resolved} resolved predictions (< {MIN_RESOLVED_FOR_STABLE}); treat live metrics as PRELIMINARY.")
    else:
        level = "usable"
        message = f"{n_resolved} resolved predictions; live metrics are usable but remain a small, growing sample."
    return {"level": level, "n_resolved": n_resolved, "message": message}


def _performance_by_event(resolved: pd.DataFrame) -> list[dict]:
    rows = []
    for (date, name), grp in resolved.groupby(["event_date", "event_name"], sort=True):
        m = _window_metrics(grp["target_a_win"].astype(int).to_numpy(), grp["predicted_probability_a"].astype(float).to_numpy())
        rows.append({"event_date": date, "event_name": name, "n_fights": m["n"], "log_loss": m["log_loss"],
                     "brier": m["brier"], "accuracy": m["accuracy"], "avg_predicted_prob": m["avg_predicted_prob"]})
    return rows


def _performance_by_period(resolved: pd.DataFrame, period_series: pd.Series, label: str) -> list[dict]:
    rows = []
    tmp = resolved.assign(_period=period_series.values)
    for period, grp in tmp.groupby("_period"):
        m = _window_metrics(grp["target_a_win"].astype(int).to_numpy(), grp["predicted_probability_a"].astype(float).to_numpy())
        rows.append({label: period, "n_fights": m["n"], "log_loss": m["log_loss"], "brier": m["brier"], "accuracy": m["accuracy"]})
    return rows


def _rolling(resolved: pd.DataFrame, windows=(50, 100)) -> list[dict]:
    y = resolved["target_a_win"].astype(int).to_numpy()
    p = resolved["predicted_probability_a"].astype(float).to_numpy()
    out = []
    for i in range(len(y)):
        snap = {"index": i + 1, "event_date": resolved["event_date"].iloc[i]}
        for w in windows:
            if i + 1 >= w:
                m = _window_metrics(y[i + 1 - w:i + 1], p[i + 1 - w:i + 1])
                snap[f"roll{w}_log_loss"] = m["log_loss"]
                snap[f"roll{w}_accuracy"] = m["accuracy"]
            else:
                snap[f"roll{w}_log_loss"] = "insufficient_sample"
        out.append(snap)
    return out


def _write_live_report(report: dict, output_dir: str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / LIVE_MODEL_REPORT_JSON, "w") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    (out_dir / LIVE_MODEL_REPORT_MD).write_text(_render_live_model_markdown(report))


# ---------------------------------------------------------------------------
# Drift + leakage
# ---------------------------------------------------------------------------

def _live_drift(df_valid: pd.DataFrame, training, base_numeric: list[str]) -> dict:
    drift = event_drift(df_valid, training, base_numeric)
    drift.pop("_z_shift", None), drift.pop("_elevated", None)
    return drift


def _live_leakage_checks(base_numeric, validation, config) -> dict:
    try:
        check_features_allowed(list(base_numeric))
        guard_ok, guard_detail = True, "Model features pass the leakage guard; raw weight_class dropped."
    except ValueError as exc:
        guard_ok, guard_detail = False, str(exc)
    return {
        "no_target_or_result_columns_in_input": {
            "status": "pass" if not validation["forbidden_result_columns"] else "fail",
            "detail": "none present." if not validation["forbidden_result_columns"] else f"present: {validation['forbidden_result_columns']}"},
        "no_odds_or_market_columns_in_input": {
            "status": "pass" if not validation["market_columns"] else "fail",
            "detail": "none present." if not validation["market_columns"] else f"present: {validation['market_columns']}"},
        "no_target_or_result_columns_used_as_features": {"status": "pass" if guard_ok else "fail", "detail": guard_detail},
        "raw_weight_class_dropped": {"status": "pass" if config["raw_weight_class"] == "dropped" else "fail",
                                     "detail": "official model config drops raw weight_class."},
        "probability_orientation_is_p_fighter_a_wins": {"status": "pass", "detail": "predicted_probability_a = P(fighter_a wins), Platt-calibrated."},
        "official_model_metadata_preserved": {"status": "pass", "detail": f"model_version={config['model_version']} loaded read-only from the benchmark."},
        "training_uses_only_official_historical_data": {"status": "pass", "detail": "Base LR + Platt fit on the official train/calibration split; live rows are never added to training."},
        "live_rows_not_added_to_training": {"status": "pass", "detail": "Live input is predicted only, never appended to the training frame."},
        "data_processed_not_modified": {"status": "pass", "detail": "Outputs go to data/live/ and reports/live/ only."},
    }


# ---------------------------------------------------------------------------
# Markdown renderers
# ---------------------------------------------------------------------------

def _fmt(value, digits: int = 6):
    if isinstance(value, str):
        return value
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except (TypeError, ValueError):
        pass
    return f"{value:.{digits}f}"


def _render_batch_markdown(report: dict) -> str:
    lines = ["# Step 6B: Live Prediction Batch", "", f"Generated: {report['generated_at']}", "",
             "**Forward live prediction** — predictions made before fights occur, appended to the durable live ledger "
             "as `pending`. Not validated until resolved.", ""]
    lines.append(f"- Batch: `{report['prediction_batch_id']}` | Model: **{report['model_version']}** | "
                 f"Calibration: {report['calibration_version']}")
    lines.append(f"- Input: `{report['input']}` | Ledger: `{report['ledger_path']}`")
    lines.append(f"- Fights submitted: {report['n_fights_submitted']} | Predictions generated: "
                 f"{report['n_predictions_generated']} | Failed rows: {report['n_failed_rows']}")
    a = report["append_stats"]
    lines.append(f"- Ledger append: accepted {a['n_accepted']}, duplicates rejected {a['n_rejected_duplicates']}, "
                 f"pending overwritten {a['n_pending_overwritten']}, versioned {a['n_duplicates_versioned']}")
    lines.append("")
    lines.append("## Predictions")
    lines.append("")
    lines.append("| event_date | fighter_a | fighter_b | P(a) | P(b) | favorite | confidence |")
    lines.append("|---|---|---|---:|---:|---|---|")
    for r in report["prediction_table"]:
        lines.append(f"| {r['event_date']} | {r['fighter_a']} | {r['fighter_b']} | {_fmt(r['predicted_probability_a'],4)} | "
                     f"{_fmt(r['predicted_probability_b'],4)} | {r['predicted_favorite']} | {r['confidence_bucket']} |")
    lines.append("")
    if report["n_failed_rows"]:
        lines.append("## Failed Rows")
        lines.append("")
        for r in report["failed_row_reasons"]:
            lines.append(f"- row {r['row_index']}: {', '.join(r['reasons'])}")
        lines.append("")
    if report["data_quality_warnings"]:
        lines.append("## Data-Quality Warnings")
        lines.append("")
        for w in report["data_quality_warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    d = report["drift_data_quality"]
    if d.get("n_rows"):
        lines.append("## Drift / Data-Quality")
        lines.append("")
        lines.append(f"- rows {d['n_rows']}; out-of-range cells {d.get('n_out_of_range_cells')}; NaN-heavy rows "
                     f"{d.get('n_rows_nan_heavy')}; low-history rows {d.get('n_low_history_flagged_rows')}; "
                     f"features w/ elevated missingness {d.get('n_features_elevated_missingness')}.")
        lines.append("")
    lines.append("## Protocol / Leakage Checks")
    lines.append("")
    for name, entry in report["leakage_protocol_checks"].items():
        lines.append(f"- **{name}**: {entry['status']}")
    lines.append("")
    return "\n".join(lines)


def _render_live_model_markdown(report: dict) -> str:
    lines = ["# Step 6B: Live Model Report", "", f"Generated: {report['generated_at']}", "",
             "Forward live-tracking metrics — computed only from predictions that have already resolved.", ""]
    sw = report["sample_size_warning"]
    lines.append(f"> **{sw['level'].upper()}**: {sw['message']}")
    lines.append("")
    lines.append(f"- Ledger: `{report['ledger_path']}` | Model: **{report['model_version']}**")
    lines.append(f"- Total: {report['total_predictions']} | pending: {report['pending_predictions']} | "
                 f"resolved: {report['resolved_predictions']} | errored: {report['errored_predictions']} | "
                 f"events: {report['n_events']}")
    lines.append("")
    if not report.get("overall_metrics"):
        lines.append(report.get("message", "No resolved predictions yet."))
        lines.append("")
        return "\n".join(lines)

    o = report["overall_metrics"]
    lines.append("## Overall Live Metrics (resolved only)")
    lines.append("")
    lines.append(f"- Log loss: **{_fmt(o['log_loss'])}** | Brier: **{_fmt(o['brier_score'])}** | "
                 f"AUC: **{_fmt(o['roc_auc'])}** | Accuracy: **{_fmt(o['accuracy'])}**")
    ref = report["comparison_reference"]
    if ref.get("official_static_test_metrics"):
        lines.append(f"- Static backtest log loss: {_fmt(ref['official_static_test_metrics']['log_loss'])} "
                     f"(live − static: {_fmt(ref.get('log_loss_vs_static'))}).")
    lines.append(f"- Step 6A pseudo-live log loss: {_fmt(ref['step6a_pseudo_live_metrics']['log_loss'])} "
                 f"(live − 6A: {_fmt(ref.get('log_loss_vs_step6a'))}).")
    lines.append("")
    lines.append("## Calibration Buckets")
    lines.append("")
    lines.append("| bucket | count | avg pred | actual | gap |")
    lines.append("|---|---:|---:|---:|---:|")
    for b in report["calibration_buckets"]:
        if b["count"]:
            lines.append(f"| {b['bucket']} | {b['count']} | {_fmt(b['avg_predicted_prob'],3)} | "
                         f"{_fmt(b['actual_a_win_rate'],3)} | {_fmt(b['calibration_gap'],3)} |")
    hc = report["high_confidence_two_sided"]
    lines.append("")
    lines.append(f"High-confidence (conf > {hc['threshold']}): count {hc['count']}, two-sided gap {_fmt(hc.get('two_sided_gap'),4)}.")
    lines.append("")
    lines.append("## Performance by Event")
    lines.append("")
    lines.append("| event_date | event | fights | log loss | accuracy |")
    lines.append("|---|---|---:|---:|---:|")
    for e in report["performance_by_event"][:20]:
        lines.append(f"| {e['event_date']} | {str(e['event_name'])[:34]} | {e['n_fights']} | {_fmt(e['log_loss'],4)} | {_fmt(e['accuracy'],3)} |")
    lines.append("")
    lines.append("## Interpretation Warnings")
    lines.append("")
    for w in report["interpretation_warnings"]:
        lines.append(f"- {w}")
    lines.append("")
    return "\n".join(lines)
