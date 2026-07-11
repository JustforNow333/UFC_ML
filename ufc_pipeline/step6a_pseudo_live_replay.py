"""Step 6A: Historical pseudo-live replay of the official model.

This is a pipeline-validation and calibration/drift-monitoring tool, NOT a new
model search. It walks the historical fight record event by event and, for each
event, asks: *if the current official model had been run live before this card,
using only data available at the time, how would it have done?*

For every replayed event it re-fits the official base LR on the older portion of
the strictly-earlier fights, fits Platt calibration on the most recent earlier
window, predicts the event, then reveals the known outcomes and scores them. The
event being predicted is never in the training or calibration window, and no
future fight is either.

Deliberate boundaries (unchanged by this task): the official model config is
reused as-is (elastic-net LR, C=0.003, l1_ratio=0.1, raw weight_class dropped,
Platt); no odds / prediction markets / new model families / new data sources; it
never writes ``data/processed/`` or ``benchmarks/official_baseline.json``.

Interpretation caveats (also emitted into the report): this is *pseudo*-live, not
a fresh future test. It reuses the already-built Step 3C processed features, so
its time-safety rests on the upstream feature builders being pre-fight only
(which the repo enforces). It is valuable for workflow validation and
calibration/drift analysis; it is not an untouched held-out benchmark.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_pipeline.calibration import (
    PlattCalibrator,
    calibration_table_with_gap,
    clip_probabilities,
    high_confidence_diagnostics,
)
from ufc_pipeline.feature_diagnostics import DEFAULT_BASELINE_PATH, official_step3c_features
from ufc_pipeline.modeling import (
    RANDOM_STATE,
    TARGET,
    check_features_allowed,
    coerce_numeric_features,
    evaluate_probs,
)
from ufc_pipeline.step5b_regularization_search import (
    DEFAULT_MAX_ITER,
    WEIGHT_CLASS_COLUMN,
    _fit_quietly,
    make_step5b_pipeline,
)

REPLAY_LEDGER_DEFAULT = "reports/replay/step6a_pseudo_live_predictions.csv"
STEP6A_REPORT_JSON = "step6a_pseudo_live_replay.json"
STEP6A_REPORT_MD = "step6a_pseudo_live_replay.md"
STEP6A_ROLLING_CSV = "step6a_rolling_metrics.csv"

PREDICTION_MODE = "pseudo_live_replay"
MISSING_EVENT_NAME = "(event name unavailable)"

# Official model config (must match step5c_stronger_regularized_lr_drop_weight_class_platt).
OFFICIAL_LR_PARAMS = {"penalty": "elasticnet", "C": 0.003, "l1_ratio": 0.1, "solver": "saga", "class_weight": None}

# Calibration bucket edges on predicted P(fighter_a wins).
CALIBRATION_BUCKET_EDGES = (0.00, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.00)
# Confidence buckets on max(p, 1-p).
CONFIDENCE_BUCKET_EDGES = (0.50, 0.60, 0.70, 0.80, 1.00)
ROLLING_WINDOWS = (50, 100)
HIGH_CONFIDENCE_THRESHOLD = 0.70

LOW_HISTORY_FLAGS = ("fighter_a_no_prior_stats", "fighter_b_no_prior_stats", "matchup_history_missing")
ID_DATE_COLUMNS = {"fight_id", "date", "event", "fighter_a", "fighter_b", "winner", "loser", "method", "round"}
MARKET_TOKENS = ("odds", "moneyline", "vig", "implied_prob", "kalshi", "polymarket", "betfair", "sportsbook", "book")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

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


def feature_schema_version(base_numeric: list[str]) -> str:
    digest = hashlib.sha1(",".join(base_numeric).encode()).hexdigest()[:8]
    return f"step3c-{len(base_numeric)}f-{digest}"


def load_official_replay_config(baseline_path: str | None = DEFAULT_BASELINE_PATH) -> dict:
    """Official model identity for stamping every replay row (read-only)."""
    base_numeric, _categorical = official_step3c_features()
    baseline = _load_json(baseline_path)
    model_name = "step5c_stronger_regularized_lr_drop_weight_class_platt"
    static_metrics = None
    if baseline and "official_model" in baseline:
        model_name = baseline["official_model"].get("name", model_name)
        static_metrics = baseline["official_model"].get("test_metrics")
    return {
        "model_version": model_name,
        "model_family": "logistic_regression",
        "calibration_method": "platt",
        "hyperparameters": dict(OFFICIAL_LR_PARAMS),
        "raw_weight_class": "dropped",
        "division_interactions": "not_included",
        "base_feature_set": "Step 3C numeric features",
        "n_base_features": len(base_numeric),
        "feature_schema_version": feature_schema_version(base_numeric),
        "static_backtest_test_metrics": static_metrics,
        "baseline_path_read": baseline_path if baseline else None,
    }


# ---------------------------------------------------------------------------
# Data + event indexing
# ---------------------------------------------------------------------------

def load_replay_frame(input_csv: str) -> tuple[pd.DataFrame, list[str]]:
    base_numeric, _categorical = official_step3c_features()
    df = pd.read_csv(input_csv)
    df = df[df[TARGET].notna() & df["fighter_a_expected_win_prob"].notna()].copy()
    df = coerce_numeric_features(df, base_numeric, context="step6a_pseudo_live_replay")
    df = df.sort_values(["date", "fight_id"]).reset_index(drop=True)
    return df, base_numeric


def iter_event_groups(df: pd.DataFrame):
    """Yield (event_date, event_name, event_df) in chronological order.

    ``df`` is pre-sorted by (date, fight_id); ``groupby(sort=False)`` therefore
    yields events in first-appearance (chronological) order deterministically.
    Event metadata is optional upstream, so null names are grouped under a
    stable report label rather than silently dropped by pandas ``groupby``.
    """
    grouped = df.assign(event=df["event"].fillna(MISSING_EVENT_NAME))
    for (event_date, event_name), event_df in grouped.groupby(["date", "event"], sort=False):
        yield event_date, event_name, event_df


def build_replay_split(
    df: pd.DataFrame, event_date: str, min_train_fights: int, calibration_fights: int,
) -> dict | None:
    """Training + calibration windows from fights strictly BEFORE ``event_date``.

    training = all earlier fights except the most recent ``calibration_fights``;
    calibration = the most recent ``calibration_fights`` earlier fights. Returns
    ``None`` when there is not enough prior history.
    """
    past = df[df["date"] < event_date]
    if len(past) < min_train_fights + calibration_fights:
        return None
    training = past.iloc[:-calibration_fights]
    calibration = past.iloc[-calibration_fights:]
    if len(training) < min_train_fights or len(calibration) == 0:
        return None
    return {"training": training, "calibration": calibration, "n_past": len(past)}


# ---------------------------------------------------------------------------
# Fit / predict
# ---------------------------------------------------------------------------

def fit_official_model(training, calibration, base_numeric, random_state=RANDOM_STATE, max_iter=DEFAULT_MAX_ITER):
    """Fit the official base LR on training only, then Platt on calibration only."""
    pipeline = make_step5b_pipeline(
        numeric=list(base_numeric), include_weight_class=False, weight_class_categories=None,
        penalty=OFFICIAL_LR_PARAMS["penalty"], C=OFFICIAL_LR_PARAMS["C"], l1_ratio=OFFICIAL_LR_PARAMS["l1_ratio"],
        solver=OFFICIAL_LR_PARAMS["solver"], max_iter=max_iter, class_weight=OFFICIAL_LR_PARAMS["class_weight"],
        random_state=random_state,
    )
    _fit_quietly(pipeline, training[list(base_numeric)], training[TARGET].astype(int))
    calib_raw = pipeline.predict_proba(calibration[list(base_numeric)])[:, 1]
    platt = PlattCalibrator().fit(calib_raw, calibration[TARGET].astype(int).to_numpy())
    return pipeline, platt


def predict_event(pipeline, platt, event_df, base_numeric) -> np.ndarray:
    raw = pipeline.predict_proba(event_df[list(base_numeric)])[:, 1]
    return clip_probabilities(platt.transform(raw))


def model_classes(pipeline) -> list:
    return [int(c) for c in pipeline.named_steps["model"].classes_]


# ---------------------------------------------------------------------------
# Per-fight scoring
# ---------------------------------------------------------------------------

def _row_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = clip_probabilities(np.asarray(p, dtype=float))
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _row_brier(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2


# ---------------------------------------------------------------------------
# Aggregate metric helpers
# ---------------------------------------------------------------------------

def _window_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = clip_probabilities(np.asarray(p, dtype=float))
    preds = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1.0 - p)
    hc = conf > HIGH_CONFIDENCE_THRESHOLD
    hc_gap = float(conf[hc].mean() - (preds[hc] == y[hc]).mean()) if hc.any() else None
    return {
        "n": int(len(y)),
        "log_loss": float(_row_log_loss(y, p).mean()),
        "brier": float(_row_brier(y, p).mean()),
        "accuracy": float((preds == y).mean()),
        "avg_predicted_prob": float(p.mean()),
        "actual_a_win_rate": float(y.mean()),
        "high_confidence_gap": hc_gap,
    }


def calibration_buckets(y: np.ndarray, p: np.ndarray, edges=CALIBRATION_BUCKET_EDGES) -> list[dict]:
    y = np.asarray(y, dtype=int)
    p = clip_probabilities(np.asarray(p, dtype=float))
    ll = _row_log_loss(y, p)
    br = _row_brier(y, p)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Last bucket closed on the right so p == 1.0 lands somewhere.
        mask = (p >= lo) & (p < hi) if hi < edges[-1] else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n:
            rows.append({
                "bucket": f"{lo:.2f}-{hi:.2f}", "count": n,
                "avg_predicted_prob": float(p[mask].mean()),
                "actual_a_win_rate": float(y[mask].mean()),
                "calibration_gap": float(p[mask].mean() - y[mask].mean()),
                "avg_log_loss": float(ll[mask].mean()),
                "avg_brier": float(br[mask].mean()),
            })
        else:
            rows.append({"bucket": f"{lo:.2f}-{hi:.2f}", "count": 0, "avg_predicted_prob": None,
                         "actual_a_win_rate": None, "calibration_gap": None,
                         "avg_log_loss": None, "avg_brier": None})
    return rows


def confidence_buckets(y: np.ndarray, p: np.ndarray, edges=CONFIDENCE_BUCKET_EDGES) -> list[dict]:
    y = np.asarray(y, dtype=int)
    p = clip_probabilities(np.asarray(p, dtype=float))
    preds = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1.0 - p)
    correct = (preds == y).astype(int)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi) if hi < edges[-1] else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        rows.append({
            "confidence_bucket": f"{lo:.2f}-{hi:.2f}", "count": n,
            "avg_confidence": float(conf[mask].mean()) if n else None,
            "accuracy": float(correct[mask].mean()) if n else None,
            "avg_log_loss": float(_row_log_loss(y[mask], p[mask]).mean()) if n else None,
        })
    return rows


def high_confidence_summary(y: np.ndarray, p: np.ndarray, threshold=HIGH_CONFIDENCE_THRESHOLD) -> dict:
    y = np.asarray(y, dtype=int)
    p = clip_probabilities(np.asarray(p, dtype=float))
    preds = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1.0 - p)
    mask = conf > threshold
    n = int(mask.sum())
    if not n:
        return {"threshold": threshold, "count": 0, "avg_confidence": None,
                "actual_correct_rate": None, "two_sided_gap": None}
    return {
        "threshold": threshold, "count": n,
        "avg_confidence": float(conf[mask].mean()),
        "actual_correct_rate": float((preds[mask] == y[mask]).mean()),
        "two_sided_gap": float(conf[mask].mean() - (preds[mask] == y[mask]).mean()),
    }


# ---------------------------------------------------------------------------
# Drift / data-quality
# ---------------------------------------------------------------------------

def event_drift(event_df, training, base_numeric, elevated_threshold=0.10) -> dict:
    """Compare one event's feature rows to that event's training window."""
    ev = event_df[list(base_numeric)]
    tr = training[list(base_numeric)]
    ev_missing = ev.isna().mean()
    tr_missing = tr.isna().mean()
    elevated = ev_missing - tr_missing
    elevated_feats = sorted(elevated[elevated > elevated_threshold].index.tolist())

    tr_mean, tr_std = tr.mean(numeric_only=True), tr.std(numeric_only=True, ddof=0)
    ev_mean = ev.mean(numeric_only=True)
    mean_shift = (ev_mean - tr_mean)
    z_shift = mean_shift / tr_std.replace(0, np.nan)

    tr_min, tr_max = tr.min(numeric_only=True), tr.max(numeric_only=True)
    below = ev.lt(tr_min)
    above = ev.gt(tr_max)
    out_of_range_cells = int((below | above).sum().sum())

    row_missing_frac = ev.isna().mean(axis=1)
    n_nan_heavy = int((row_missing_frac > 0.5).sum())

    low_history_flags = [
        pd.to_numeric(event_df[flag], errors="coerce").fillna(0).astype(float).gt(0)
        for flag in LOW_HISTORY_FLAGS
        if flag in event_df.columns
    ]
    low_history = (
        int(pd.concat(low_history_flags, axis=1).any(axis=1).sum())
        if low_history_flags
        else 0
    )

    return {
        "n_rows": int(len(ev)),
        "avg_missing_rate_event": float(ev_missing.mean()),
        "avg_missing_rate_training": float(tr_missing.mean()),
        "n_features_elevated_missingness": len(elevated_feats),
        "elevated_missingness_features": elevated_feats[:15],
        "n_out_of_range_cells": out_of_range_cells,
        "n_rows_nan_heavy": n_nan_heavy,
        "n_low_history_flagged_rows": low_history,
        "_z_shift": z_shift,  # consumed by the global aggregator, stripped later
        "_elevated": elevated,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pseudo_live_replay(
    input_csv: str,
    output_dir: str = "reports",
    ledger_path: str | None = None,
    min_train_fights: int = 3000,
    calibration_fights: int = 1282,
    start_date: str | None = None,
    end_date: str | None = None,
    max_events: int | None = None,
    overwrite: bool = False,
    random_seed: int = RANDOM_STATE,
    baseline_path: str | None = DEFAULT_BASELINE_PATH,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict:
    out_dir = Path(output_dir)
    ledger_file = Path(ledger_path) if ledger_path else Path(REPLAY_LEDGER_DEFAULT)
    if ledger_file.exists() and not overwrite:
        raise FileExistsError(
            f"Replay ledger already exists at {ledger_file}. Pass overwrite=True (--overwrite) to replace it, "
            "or choose a different --output ledger path."
        )

    config = load_official_replay_config(baseline_path)
    calibration_version = f"platt_calib{calibration_fights}"
    df, base_numeric = load_replay_frame(input_csv)

    # Structural leakage guards that do not depend on the loop.
    guard_ok, guard_detail = True, "Base features pass the leakage guard; raw weight_class dropped."
    try:
        check_features_allowed(list(base_numeric))
    except ValueError as exc:
        guard_ok, guard_detail = False, str(exc)
    used_ids = sorted(ID_DATE_COLUMNS & set(base_numeric))
    market_hits = sorted([c for c in base_numeric if any(tok in c.lower() for tok in MARKET_TOKENS)])

    ledger_rows: list[dict] = []
    per_event: list[dict] = []
    drift_events: list[dict] = []
    z_shift_accum: dict[str, list[float]] = {f: [] for f in base_numeric}
    elevated_accum: dict[str, list[float]] = {f: [] for f in base_numeric}
    skipped = {"insufficient_history": 0, "before_start_date": 0, "after_end_date": 0}
    skipped_examples: list[dict] = []

    # Leakage accounting across events.
    leak = {
        "train_before_event": 0, "calib_before_event": 0, "train_before_calib": 0,
        "event_rows_not_in_train_or_calib": 0, "orientation_p_is_class1": 0, "events_checked": 0,
    }

    replay_index = 0
    n_events_replayed = 0
    y_stream: list[int] = []
    p_stream: list[float] = []
    rolling_snapshots: list[dict] = []

    for event_date, event_name, event_df in iter_event_groups(df):
        if start_date is not None and event_date < start_date:
            skipped["before_start_date"] += 1
            continue
        if end_date is not None and event_date > end_date:
            skipped["after_end_date"] += 1
            continue
        split = build_replay_split(df, event_date, min_train_fights, calibration_fights)
        if split is None:
            skipped["insufficient_history"] += 1
            if len(skipped_examples) < 10:
                skipped_examples.append({"event_date": event_date, "event_name": event_name,
                                         "n_past": int((df["date"] < event_date).sum()), "reason": "insufficient_history"})
            continue
        if max_events is not None and n_events_replayed >= max_events:
            break

        training, calibration = split["training"], split["calibration"]
        pipeline, platt = fit_official_model(training, calibration, base_numeric,
                                             random_state=random_seed, max_iter=max_iter)
        classes = model_classes(pipeline)
        probs_a = predict_event(pipeline, platt, event_df, base_numeric)
        y = event_df[TARGET].astype(int).to_numpy()

        # Leakage accounting for this event.
        leak["events_checked"] += 1
        if training["date"].max() < event_date:
            leak["train_before_event"] += 1
        if calibration["date"].max() < event_date:
            leak["calib_before_event"] += 1
        if training["date"].max() <= calibration["date"].min():
            leak["train_before_calib"] += 1
        event_ids = set(event_df["fight_id"])
        if not (event_ids & set(training["fight_id"])) and not (event_ids & set(calibration["fight_id"])):
            leak["event_rows_not_in_train_or_calib"] += 1
        if classes == [0, 1]:
            leak["orientation_p_is_class1"] += 1

        drift = event_drift(event_df, training, base_numeric)
        for f in base_numeric:
            zval = drift["_z_shift"].get(f)
            if zval is not None and np.isfinite(zval):
                z_shift_accum[f].append(abs(float(zval)))
            elev = drift["_elevated"].get(f)
            if elev is not None and np.isfinite(elev):
                elevated_accum[f].append(float(elev))
        drift.pop("_z_shift"), drift.pop("_elevated")

        row_ll = _row_log_loss(y, probs_a)
        row_br = _row_brier(y, probs_a)
        preds = (probs_a >= 0.5).astype(int)
        now = datetime.now(timezone.utc).isoformat()

        train_start, train_end = str(training["date"].min()), str(training["date"].max())
        calib_start, calib_end = str(calibration["date"].min()), str(calibration["date"].max())
        for i, (_, fight) in enumerate(event_df.iterrows()):
            note_parts = []
            for flag in LOW_HISTORY_FLAGS:
                if flag in event_df.columns and float(pd.to_numeric(fight[flag], errors="coerce") or 0) > 0:
                    note_parts.append(flag)
            ledger_rows.append({
                "replay_id": f"{event_date}__{int(fight['fight_id'])}",
                "event_date": event_date,
                "event_name": event_name,
                "fighter_a": fight.get("fighter_a"),
                "fighter_b": fight.get("fighter_b"),
                "model_version": config["model_version"],
                "calibration_version": calibration_version,
                "feature_schema_version": config["feature_schema_version"],
                "prediction_mode": PREDICTION_MODE,
                "train_start_date": train_start,
                "train_end_date": train_end,
                "train_rows": int(len(training)),
                "calibration_start_date": calib_start,
                "calibration_end_date": calib_end,
                "calibration_rows": int(len(calibration)),
                "predicted_probability_a": float(probs_a[i]),
                "predicted_probability_b": float(1.0 - probs_a[i]),
                "target_a_win": int(y[i]),
                "winner": fight.get("winner"),
                "log_loss": float(row_ll[i]),
                "brier": float(row_br[i]),
                "correct_prediction": int(preds[i] == y[i]),
                "prediction_timestamp_utc": now,
                "resolved_timestamp_utc": now,
                "notes": ";".join(note_parts),
            })
            replay_index += 1

        y_stream.extend(y.tolist())
        p_stream.extend(probs_a.tolist())

        ev_metrics = _window_metrics(y, probs_a)
        per_event.append({
            "event_date": event_date, "event_name": event_name, "n_fights": int(len(event_df)),
            "log_loss": ev_metrics["log_loss"], "brier": ev_metrics["brier"], "accuracy": ev_metrics["accuracy"],
            "avg_predicted_prob": ev_metrics["avg_predicted_prob"], "actual_a_win_rate": ev_metrics["actual_a_win_rate"],
            "train_rows": int(len(training)), "calibration_rows": int(len(calibration)),
            "n_low_history_flagged_rows": drift["n_low_history_flagged_rows"],
            "n_out_of_range_cells": drift["n_out_of_range_cells"],
        })
        drift_events.append({"event_date": event_date, "event_name": event_name, **drift})

        # Rolling snapshot at this event's last predicted fight.
        snap = {"event_date": event_date, "event_name": event_name, "cumulative_fights": len(y_stream)}
        for w in ROLLING_WINDOWS:
            if len(y_stream) >= w:
                wm = _window_metrics(np.array(y_stream[-w:]), np.array(p_stream[-w:]))
                snap[f"roll{w}_log_loss"] = wm["log_loss"]
                snap[f"roll{w}_brier"] = wm["brier"]
                snap[f"roll{w}_accuracy"] = wm["accuracy"]
                snap[f"roll{w}_avg_pred"] = wm["avg_predicted_prob"]
                snap[f"roll{w}_actual_rate"] = wm["actual_a_win_rate"]
                snap[f"roll{w}_high_conf_gap"] = wm["high_confidence_gap"]
            else:
                snap[f"roll{w}_log_loss"] = "insufficient_sample"
        rolling_snapshots.append(snap)
        n_events_replayed += 1

    if not ledger_rows:
        raise ValueError(
            "No events qualified for replay with the given configuration "
            f"(min_train_fights={min_train_fights}, calibration_fights={calibration_fights}, "
            f"start_date={start_date}, end_date={end_date}). Lower the warmup thresholds or widen the date range."
        )

    ledger_df = pd.DataFrame(ledger_rows)
    y_all = ledger_df["target_a_win"].to_numpy()
    p_all = ledger_df["predicted_probability_a"].to_numpy()

    overall = evaluate_probs(y_all, clip_probabilities(p_all))
    overall_calibration = calibration_table_with_gap(y_all, clip_probabilities(p_all))
    hc_standard = high_confidence_diagnostics(y_all, clip_probabilities(p_all))
    hc_two_sided = high_confidence_summary(y_all, p_all)

    # Performance by year.
    ledger_df["year"] = ledger_df["event_date"].str.slice(0, 4)
    by_year = []
    for year, grp in ledger_df.groupby("year"):
        m = _window_metrics(grp["target_a_win"].to_numpy(), grp["predicted_probability_a"].to_numpy())
        auc = evaluate_probs(grp["target_a_win"].to_numpy(), clip_probabilities(grp["predicted_probability_a"].to_numpy()))["roc_auc"]
        by_year.append({"year": year, "n_fights": m["n"], "log_loss": m["log_loss"], "brier": m["brier"],
                        "accuracy": m["accuracy"], "roc_auc": auc, "avg_predicted_prob": m["avg_predicted_prob"],
                        "actual_a_win_rate": m["actual_a_win_rate"]})

    # Worst / best events by log loss (min 6 fights to reduce single-event noise).
    events_df = pd.DataFrame(per_event)
    scored_events = events_df[events_df["n_fights"] >= 6].sort_values("log_loss")
    best_events = scored_events.head(10).to_dict("records")
    worst_events = scored_events.tail(10).iloc[::-1].to_dict("records")

    # Global drift aggregation.
    drift_global = _aggregate_drift(z_shift_accum, elevated_accum, drift_events)

    static = config.get("static_backtest_test_metrics") or {}
    comparison = {
        "static_backtest_test_metrics": static,
        "pseudo_live_metrics": {
            "log_loss": overall["log_loss"], "brier_score": overall["brier_score"],
            "roc_auc": overall["roc_auc"], "accuracy": overall["accuracy"],
        },
        "log_loss_difference_pseudo_live_minus_static": (
            overall["log_loss"] - static["log_loss"] if static.get("log_loss") is not None else None
        ),
        "note": (
            "The static backtest is a single fit scored on the 2023-11..2026-05 held-out window; this pseudo-live "
            "replay re-fits per event over a broader era (first qualifying event onward). They are not directly "
            "comparable — different eras, different (expanding) training, and the final test was already touched."
        ),
    }

    leakage_checks = _build_leakage_checks(leak, guard_ok, guard_detail, used_ids, market_hits, config)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": "step6a_pseudo_live_replay",
        "prediction_mode": PREDICTION_MODE,
        "changes_official_model": False,
        "changes_official_baseline_file": False,
        "writes_data_processed": False,
        "input": input_csv,
        "official_model": config,
        "replay_configuration": {
            "min_train_fights": min_train_fights, "calibration_fights": calibration_fights,
            "calibration_version": calibration_version, "start_date": start_date, "end_date": end_date,
            "max_events": max_events, "random_seed": random_seed, "max_iter": max_iter,
            "training_policy": "expanding history minus most recent calibration window",
            "calibration_policy": f"most recent {calibration_fights} fights strictly before each event",
        },
        "date_range_replayed": [ledger_df["event_date"].min(), ledger_df["event_date"].max()],
        "n_events_replayed": n_events_replayed,
        "n_fights_predicted": int(len(ledger_df)),
        "n_events_skipped": int(sum(skipped.values())),
        "skipped_reasons": skipped,
        "skipped_examples": skipped_examples,
        "overall_metrics": {
            "log_loss": overall["log_loss"], "brier_score": overall["brier_score"],
            "roc_auc": overall["roc_auc"], "accuracy": overall["accuracy"],
            "avg_predicted_prob": overall["avg_predicted_prob"], "actual_a_win_rate": overall["actual_win_rate"],
        },
        "calibration_buckets": calibration_buckets(y_all, p_all),
        "calibration_table_10_bucket": overall_calibration,
        "high_confidence_standard": hc_standard,
        "high_confidence_two_sided": hc_two_sided,
        "confidence_buckets": confidence_buckets(y_all, p_all),
        "performance_by_year": by_year,
        "rolling_metrics": rolling_snapshots,
        "worst_events_by_log_loss": worst_events,
        "best_events_by_log_loss": best_events,
        "comparison_to_static_backtest": comparison,
        "drift_data_quality": drift_global,
        "leakage_protocol_checks": leakage_checks,
        "interpretation_warnings": [
            "This is PSEUDO-live replay, not a fresh future live validation: it reuses the already-built Step 3C processed features.",
            "Its time-safety depends on the upstream Step 3/3B/3C feature builders being pre-fight only (enforced elsewhere in the repo); this tool re-fits only the model and Platt calibrator per event.",
            "The final held-out test split was already touched in prior runs, so this is not an untouched benchmark; it is a workflow-validation and calibration/drift tool.",
            "Do not select or retune models against these numbers; the official model is unchanged.",
            "Single-event metrics are noisy (median ~11 fights/event); read rolling and yearly aggregates instead.",
        ],
        "limitations": [
            "Features are read from the processed CSV; this tool does not recompute Elo/rolling features per event.",
            "Calibration window is fight-count based; a --calibration-window-days variant is not implemented here.",
            "Drift comparison uses each event vs its own training window, aggregated globally.",
        ],
    }

    # ---- Write outputs (never under data/processed/) ----
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_file.parent.mkdir(parents=True, exist_ok=True)
    ledger_df.drop(columns=["year"]).to_csv(ledger_file, index=False)
    with open(out_dir / STEP6A_REPORT_JSON, "w") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    (out_dir / STEP6A_REPORT_MD).write_text(_render_markdown(report, str(ledger_file)))
    pd.DataFrame(rolling_snapshots).to_csv(out_dir / STEP6A_ROLLING_CSV, index=False)

    report["ledger_path"] = str(ledger_file)
    return report


def _aggregate_drift(z_shift_accum, elevated_accum, drift_events) -> dict:
    feat_z = [{"feature": f, "mean_abs_z_shift": float(np.mean(v))} for f, v in z_shift_accum.items() if v]
    feat_z.sort(key=lambda r: r["mean_abs_z_shift"], reverse=True)
    feat_elev = [{"feature": f, "mean_missingness_elevation": float(np.mean(v))} for f, v in elevated_accum.items() if v]
    feat_elev.sort(key=lambda r: r["mean_missingness_elevation"], reverse=True)
    return {
        "n_events_analyzed": len(drift_events),
        "total_out_of_range_cells": int(sum(e["n_out_of_range_cells"] for e in drift_events)),
        "total_nan_heavy_rows": int(sum(e["n_rows_nan_heavy"] for e in drift_events)),
        "total_low_history_flagged_rows": int(sum(e["n_low_history_flagged_rows"] for e in drift_events)),
        "events_with_elevated_missingness": int(sum(e["n_features_elevated_missingness"] > 0 for e in drift_events)),
        "top_drifting_features_by_abs_z_shift": feat_z[:15],
        "top_features_by_missingness_elevation": feat_elev[:15],
        "per_event": [
            {k: v for k, v in e.items() if k not in ("elevated_missingness_features",)} for e in drift_events
        ],
    }


def _build_leakage_checks(leak, guard_ok, guard_detail, used_ids, market_hits, config) -> dict:
    n = max(1, leak["events_checked"])

    def _all(count, label):
        return {"status": "pass" if count == leak["events_checked"] else "fail",
                "detail": f"{count}/{leak['events_checked']} events {label}"}

    return {
        "no_target_or_result_columns_in_features": {"status": "pass" if guard_ok else "fail", "detail": guard_detail},
        "no_identifier_or_date_columns_in_features": {
            "status": "pass" if not used_ids else "fail",
            "detail": "none used." if not used_ids else f"used: {used_ids}"},
        "no_odds_or_market_features": {
            "status": "pass" if not market_hits else "fail",
            "detail": "no market tokens in feature set." if not market_hits else f"hits: {market_hits}"},
        "raw_weight_class_dropped": {"status": "pass" if config["raw_weight_class"] == "dropped" else "fail",
                                     "detail": "official replay config drops raw weight_class."},
        "training_strictly_before_event_date": _all(leak["train_before_event"], "have max train date < event date"),
        "calibration_strictly_before_event_date": _all(leak["calib_before_event"], "have max calibration date < event date"),
        "training_before_calibration_window": _all(leak["train_before_calib"], "have train window ending at/before calibration start"),
        "event_rows_excluded_from_train_and_calibration": _all(leak["event_rows_not_in_train_or_calib"], "have event fight_ids disjoint from train/calibration"),
        "probability_orientation_is_p_fighter_a_wins": _all(leak["orientation_p_is_class1"], "have classes_=[0,1] (predicted prob = P(fighter_a wins))"),
        "official_baseline_file_not_modified": {"status": "pass", "detail": "benchmarks/official_baseline.json is read-only here."},
        "data_processed_not_modified": {"status": "pass", "detail": "outputs go to reports/ and reports/replay/ only."},
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _fmt(value, digits: int = 6) -> str:
    if value is None or (isinstance(value, str)):
        return value if isinstance(value, str) else "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except (TypeError, ValueError):
        pass
    return f"{value:.{digits}f}"


def _render_markdown(report: dict, ledger_path: str) -> str:
    o = report["overall_metrics"]
    cfg = report["replay_configuration"]
    lines: list[str] = []
    lines.append("# Step 6A: Historical Pseudo-Live Replay")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    lines.append("**Pseudo-live replay** of the official model — a pipeline-validation and calibration/drift tool, "
                 "NOT a fresh future live test and NOT an untouched benchmark. It re-fits the official LR + Platt "
                 "per event on strictly-earlier fights and reuses the existing Step 3C processed features.")
    lines.append("")

    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Model: **{report['official_model']['model_version']}** "
                 f"(elastic-net LR C={report['official_model']['hyperparameters']['C']}, "
                 f"l1_ratio={report['official_model']['hyperparameters']['l1_ratio']}, raw weight_class dropped, Platt).")
    lines.append(f"- min_train_fights={cfg['min_train_fights']}, calibration_fights={cfg['calibration_fights']}, "
                 f"random_seed={cfg['random_seed']}.")
    lines.append(f"- Date range replayed: {report['date_range_replayed'][0]} → {report['date_range_replayed'][1]}.")
    lines.append(f"- Events replayed: **{report['n_events_replayed']}**; fights predicted: **{report['n_fights_predicted']}**; "
                 f"events skipped: **{report['n_events_skipped']}** ({report['skipped_reasons']}).")
    lines.append(f"- Ledger: `{ledger_path}`")
    lines.append("")

    lines.append("## Overall Pseudo-Live Metrics")
    lines.append("")
    lines.append(f"- Log loss: **{_fmt(o['log_loss'])}** | Brier: **{_fmt(o['brier_score'])}** | "
                 f"ROC AUC: **{_fmt(o['roc_auc'])}** | Accuracy: **{_fmt(o['accuracy'])}**")
    lines.append(f"- Avg predicted P(A wins): {_fmt(o['avg_predicted_prob'],4)} | Actual A win rate: {_fmt(o['actual_a_win_rate'],4)}")
    comp = report["comparison_to_static_backtest"]
    if comp["static_backtest_test_metrics"].get("log_loss") is not None:
        s = comp["static_backtest_test_metrics"]
        lines.append(f"- Static backtest (2023-11..2026-05): log loss {_fmt(s['log_loss'])}, Brier {_fmt(s.get('brier_score'))}, "
                     f"AUC {_fmt(s.get('roc_auc'))}, accuracy {_fmt(s.get('accuracy'))}.")
        lines.append(f"- Pseudo-live minus static log loss: **{comp['log_loss_difference_pseudo_live_minus_static']:+.6f}** "
                     f"(different era/training — not directly comparable).")
    lines.append("")

    lines.append("## Performance by Year")
    lines.append("")
    lines.append("| year | fights | log loss | Brier | accuracy | AUC | avg pred | actual rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in report["performance_by_year"]:
        lines.append(f"| {r['year']} | {r['n_fights']} | {_fmt(r['log_loss'])} | {_fmt(r['brier'])} | "
                     f"{_fmt(r['accuracy'])} | {_fmt(r['roc_auc'])} | {_fmt(r['avg_predicted_prob'],3)} | {_fmt(r['actual_a_win_rate'],3)} |")
    lines.append("")

    lines.append("## Calibration Buckets (predicted P(A wins))")
    lines.append("")
    lines.append("| bucket | count | avg pred | actual | gap | avg LL |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for b in report["calibration_buckets"]:
        lines.append(f"| {b['bucket']} | {b['count']} | {_fmt(b['avg_predicted_prob'],3)} | {_fmt(b['actual_a_win_rate'],3)} | "
                     f"{_fmt(b['calibration_gap'],3)} | {_fmt(b['avg_log_loss'],4)} |")
    hc = report["high_confidence_two_sided"]
    lines.append("")
    lines.append(f"High-confidence (conf > {hc['threshold']}): count {hc['count']}, avg confidence {_fmt(hc['avg_confidence'],3)}, "
                 f"actual correct {_fmt(hc['actual_correct_rate'],3)}, two-sided gap **{_fmt(hc['two_sided_gap'],4)}**.")
    lines.append("")

    lines.append("## Rolling Metrics (snapshot per event; sampled)")
    lines.append("")
    snaps = report["rolling_metrics"]
    sampled = snaps[:: max(1, len(snaps) // 25)] if snaps else []
    lines.append("| event_date | cum fights | roll100 LL | roll100 Brier | roll100 acc | roll100 avg pred | roll100 actual |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in sampled:
        ll = s.get("roll100_log_loss")
        lines.append(f"| {s['event_date']} | {s['cumulative_fights']} | {_fmt(ll if not isinstance(ll,str) else None,4)} | "
                     f"{_fmt(s.get('roll100_brier'),4)} | {_fmt(s.get('roll100_accuracy'),3)} | "
                     f"{_fmt(s.get('roll100_avg_pred'),3)} | {_fmt(s.get('roll100_actual_rate'),3)} |")
    lines.append("")

    lines.append("## Worst Events by Log Loss (≥6 fights)")
    lines.append("")
    lines.append("| event_date | event | fights | log loss | Brier | accuracy | avg pred |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for e in report["worst_events_by_log_loss"][:8]:
        lines.append(f"| {e['event_date']} | {str(e['event_name'])[:34]} | {e['n_fights']} | {_fmt(e['log_loss'],4)} | "
                     f"{_fmt(e['brier'],4)} | {_fmt(e['accuracy'],3)} | {_fmt(e['avg_predicted_prob'],3)} |")
    lines.append("")
    lines.append("## Best Events by Log Loss (≥6 fights)")
    lines.append("")
    lines.append("| event_date | event | fights | log loss | accuracy |")
    lines.append("|---|---|---:|---:|---:|")
    for e in report["best_events_by_log_loss"][:8]:
        lines.append(f"| {e['event_date']} | {str(e['event_name'])[:34]} | {e['n_fights']} | {_fmt(e['log_loss'],4)} | {_fmt(e['accuracy'],3)} |")
    lines.append("")

    lines.append("## Drift / Data-Quality")
    lines.append("")
    d = report["drift_data_quality"]
    lines.append(f"- Events analyzed: {d['n_events_analyzed']}; total out-of-range cells: {d['total_out_of_range_cells']}; "
                 f"NaN-heavy rows: {d['total_nan_heavy_rows']}; low-history flagged rows: {d['total_low_history_flagged_rows']}; "
                 f"events with elevated missingness: {d['events_with_elevated_missingness']}.")
    top = d["top_drifting_features_by_abs_z_shift"][:6]
    if top:
        lines.append("- Top drifting features (mean |z-shift| event vs training): "
                     + ", ".join(f"{r['feature']} ({_fmt(r['mean_abs_z_shift'],2)})" for r in top))
    lines.append("")

    lines.append("## Protocol / Leakage Checks")
    lines.append("")
    for name, entry in report["leakage_protocol_checks"].items():
        lines.append(f"- **{name}**: {entry['status']} — {entry['detail']}")
    lines.append("")

    lines.append("## Interpretation Warnings")
    lines.append("")
    for w in report["interpretation_warnings"]:
        lines.append(f"- {w}")
    lines.append("")
    return "\n".join(lines)
