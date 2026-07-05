"""Probability calibration layer (Step 4B).

The Step 3B model comparison found the best logistic regression is
OVERCONFIDENT: predictions above 0.7 won ~7 points less often than
predicted, driven by first-listed/red-corner win-rate drift between the
training era and the 2023+ test era. This module adds a recalibration
layer on top of the existing best model. It does NOT add features, does
not touch feature generation, and does not replace compare_models.py.

Design decisions:

* THREE chronological windows, never random:
    train        -> fits the base logistic regression only
    calibration  -> fits the calibrator only (Platt / isotonic)
    test         -> metrics only; nothing is ever fit here
  Fitting the calibrator on the test window would make the calibration
  numbers meaningless, so the split function hard-fails on any overlap
  or empty window.
* Platt scaling is fit on the base model's LOGITS, not raw probabilities:
  logit(p) = log(p / (1 - p)). A logistic curve over logits is the
  textbook Platt formulation and behaves better near 0/1 than a logistic
  curve over probabilities. Probabilities are clipped to [1e-6, 1-1e-6]
  before the logit transform. The Platt LogisticRegression uses a huge C
  (effectively unregularized), matching the classic maximum-likelihood fit.
* Isotonic regression is fit directly on probabilities with
  out_of_bounds="clip"; it can output exactly 0.0/1.0, so calibrated
  probabilities are clipped away from the endpoints before log loss.
* The same leakage guard as modeling.py runs before the base model trains.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ufc_pipeline.modeling import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    RANDOM_STATE,
    STEP3B_MODEL_FEATURES,
    TARGET,
    check_features_allowed,
    coerce_numeric_features,
    evaluate_probs,
    make_logistic_pipeline,
    select_features,
)

PROB_EPS = 1e-6
HIGH_CONFIDENCE_THRESHOLDS = (0.7, 0.8)
# Platt wins ties within this log-loss margin (simpler method preferred).
NEARLY_TIED_LOG_LOSS = 1e-3
# Below this many calibration rows, isotonic is flagged as overfit-prone.
ISOTONIC_MIN_CALIBRATION_ROWS = 300


def clip_probabilities(probs: np.ndarray, eps: float = PROB_EPS) -> np.ndarray:
    """Keep probabilities safely inside (0, 1) so log loss stays finite."""
    return np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)


def probs_to_logits(probs: np.ndarray) -> np.ndarray:
    p = clip_probabilities(probs)
    return np.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# Three-way chronological split
# ---------------------------------------------------------------------------

def chronological_three_way_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    calibration_frac: float = 0.15,
    train_end_date: str | None = None,
    calibration_end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Oldest fights -> train, middle window -> calibration, newest -> test.

    Two mutually exclusive modes:
    * date mode: BOTH train_end_date and calibration_end_date given.
        train:       date <  train_end_date
        calibration: train_end_date <= date < calibration_end_date
        test:        date >= calibration_end_date
    * fraction mode (default 70/15/15): cuts by chronological row count.

    Never random. Hard-fails on empty windows or misordered dates.
    """
    if (train_end_date is None) != (calibration_end_date is None):
        raise ValueError(
            "Provide BOTH --train-end-date and --calibration-end-date, "
            "or neither (fraction-based split)."
        )
    ordered = df.sort_values(["date", "fight_id"], kind="stable").reset_index(drop=True)

    if train_end_date is not None:
        if not train_end_date < calibration_end_date:
            raise ValueError(
                f"train_end_date ({train_end_date}) must be earlier than "
                f"calibration_end_date ({calibration_end_date})."
            )
        train = ordered[ordered["date"] < train_end_date]
        calib = ordered[
            (ordered["date"] >= train_end_date)
            & (ordered["date"] < calibration_end_date)
        ]
        test = ordered[ordered["date"] >= calibration_end_date]
    else:
        if train_frac <= 0 or calibration_frac <= 0 or train_frac + calibration_frac >= 1:
            raise ValueError(
                f"Need train_frac > 0, calibration_frac > 0 and their sum < 1 "
                f"(got {train_frac} + {calibration_frac})."
            )
        n = len(ordered)
        cut1 = int(round(n * train_frac))
        cut2 = int(round(n * (train_frac + calibration_frac)))
        train = ordered.iloc[:cut1]
        calib = ordered.iloc[cut1:cut2]
        test = ordered.iloc[cut2:]

    if len(train) == 0 or len(calib) == 0 or len(test) == 0:
        raise ValueError(
            f"Empty split window: {len(train)} train / {len(calib)} calibration / "
            f"{len(test)} test rows. Adjust dates or fractions."
        )
    return train, calib, test


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------

class PlattCalibrator:
    """Platt scaling: logistic regression on the base model's logits.

    Learns p_calibrated = sigmoid(a * logit(p_raw) + b) from the
    calibration window. Smooth, monotonic, two parameters -> hard to overfit.
    """

    def __init__(self) -> None:
        # Huge C = effectively unregularized, the classic Platt ML fit.
        self._lr = LogisticRegression(C=1e6, max_iter=5000, random_state=RANDOM_STATE)

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        self._lr.fit(probs_to_logits(probs).reshape(-1, 1), np.asarray(y_true, dtype=int))
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        raw = self._lr.predict_proba(probs_to_logits(probs).reshape(-1, 1))[:, 1]
        return clip_probabilities(raw)

    @property
    def coefficients(self) -> dict:
        return {
            "slope": float(self._lr.coef_.ravel()[0]),
            "intercept": float(self._lr.intercept_.ravel()[0]),
        }


class IsotonicCalibrator:
    """Isotonic regression: monotonic step-function correction p_raw -> p_cal.

    More flexible than Platt but can overfit small calibration windows
    (it fits one step per distinct violation of monotonicity).
    """

    def __init__(self) -> None:
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self._iso.fit(np.asarray(probs, dtype=float), np.asarray(y_true, dtype=int))
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        # Isotonic can emit exactly 0/1; clip so log loss stays finite.
        return clip_probabilities(self._iso.predict(np.asarray(probs, dtype=float)))


class CalibratedPipeline:
    """Saved artifact: base sklearn pipeline + fitted calibrator.

    predict_proba(X) returns the usual (n, 2) array with CALIBRATED
    P(fighter_a wins) in column 1.
    """

    def __init__(self, pipeline, calibrator) -> None:
        self.pipeline = pipeline
        self.calibrator = calibrator

    def predict_proba(self, X) -> np.ndarray:
        p = self.calibrator.transform(self.pipeline.predict_proba(X)[:, 1])
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def high_confidence_diagnostics(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: tuple[float, ...] = HIGH_CONFIDENCE_THRESHOLDS,
) -> dict:
    """For predictions above each threshold: count, avg predicted, actual, gap.

    gap = avg_predicted - actual_win_rate; positive means overconfident.
    Empty buckets report count 0 and null stats instead of crashing.
    """
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    out = {}
    for thr in thresholds:
        mask = probs > thr
        n = int(mask.sum())
        entry = {"n_predictions": n, "avg_predicted_prob": None,
                 "actual_win_rate": None, "calibration_gap": None}
        if n:
            avg, actual = float(probs[mask].mean()), float(y_true[mask].mean())
            entry.update(avg_predicted_prob=avg, actual_win_rate=actual,
                         calibration_gap=avg - actual)
        out[f"above_{thr}"] = entry
    return out


def calibration_table_with_gap(y_true: np.ndarray, probs: np.ndarray) -> list[dict]:
    """The modeling.py 10-bucket table plus a gap column (pred - actual)."""
    from ufc_pipeline.modeling import calibration_table

    rows = calibration_table(np.asarray(y_true, dtype=int), np.asarray(probs, dtype=float))
    for row in rows:
        if row["n_predictions"]:
            row["gap"] = row["avg_predicted_prob"] - row["actual_win_rate"]
        else:
            row["gap"] = None
    return rows


def _evaluate_calibrated(y_true: np.ndarray, probs: np.ndarray) -> dict:
    probs = clip_probabilities(probs)
    metrics = evaluate_probs(y_true, probs)
    metrics["calibration"] = calibration_table_with_gap(y_true, probs)
    metrics["high_confidence"] = high_confidence_diagnostics(y_true, probs)
    return metrics


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def choose_best_method(results: dict, n_calibration: int) -> dict:
    """Pick the best calibrated model, primarily by final-test log loss.

    Platt is the project default (simpler: two parameters, hard to overfit).
    Isotonic wins only when the data CLEARLY proves it better: a log-loss
    advantage beyond the near-tie margin, or a near-tie on log loss with a
    clear Brier advantage — and never on an overfit-prone small
    calibration window.
    """
    unc, platt, iso = (results["models"][k] for k in ("uncalibrated", "platt", "isotonic"))
    isotonic_overfit_risk = n_calibration < ISOTONIC_MIN_CALIBRATION_ROWS

    best = "platt"
    ll_edge = platt["log_loss"] - iso["log_loss"]      # >0 means isotonic better
    brier_edge = platt["brier_score"] - iso["brier_score"]
    if not isotonic_overfit_risk and (
        ll_edge >= NEARLY_TIED_LOG_LOSS
        or (abs(ll_edge) < NEARLY_TIED_LOG_LOSS and brier_edge >= NEARLY_TIED_LOG_LOSS)
    ):
        best = "isotonic"

    best_m = platt if best == "platt" else iso
    return {
        "best_method": best,
        "best_log_loss": best_m["log_loss"],
        "platt_improved_log_loss": platt["log_loss"] < unc["log_loss"],
        "isotonic_improved_log_loss": iso["log_loss"] < unc["log_loss"],
        "best_improved_brier": best_m["brier_score"] < unc["brier_score"],
        "overconfidence_above_0.7": {
            "uncalibrated_gap": unc["high_confidence"]["above_0.7"]["calibration_gap"],
            "best_gap": best_m["high_confidence"]["above_0.7"]["calibration_gap"],
        },
        "auc_change": (
            best_m["roc_auc"] - unc["roc_auc"]
            if best_m["roc_auc"] is not None and unc["roc_auc"] is not None
            else None
        ),
        "accuracy_change": best_m["accuracy"] - unc["accuracy"],
        "isotonic_overfit_risk": isotonic_overfit_risk,
        "n_calibration": int(n_calibration),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_calibration(
    input_csv: str,
    output_dir: str = "data/processed",
    model_dir: str = "data/models",
    train_frac: float = 0.70,
    calibration_frac: float = 0.15,
    train_end_date: str | None = None,
    calibration_end_date: str | None = None,
    numeric_features: list[str] | None = None,
    include_step3_basic: bool = True,
    output_prefix: str = "",
    model_stem: str = "step3b_logistic_regression",
) -> dict:
    """Full calibration workflow. Returns the results dict (also written to JSON).

    Train window -> base Step 3B logistic regression.
    Calibration window -> Platt + isotonic calibrators (base model frozen).
    Test window -> ALL reported metrics; nothing is fit here.
    """
    df = pd.read_csv(input_csv)
    before = len(df)
    df = df[df[TARGET].notna() & df["fighter_a_expected_win_prob"].notna()]
    dropped = before - len(df)

    train, calib, test = chronological_three_way_split(
        df,
        train_frac=train_frac,
        calibration_frac=calibration_frac,
        train_end_date=train_end_date,
        calibration_end_date=calibration_end_date,
    )
    print(
        f"Rows: {len(df)} usable ({dropped} dropped for missing target/Elo prob)\n"
        f"Chronological 3-way split:\n"
        f"  train       {len(train):>5} rows ({train['date'].min()} .. {train['date'].max()})\n"
        f"  calibration {len(calib):>5} rows ({calib['date'].min()} .. {calib['date'].max()})\n"
        f"  test        {len(test):>5} rows ({test['date'].min()} .. {test['date'].max()})"
    )

    requested_numeric = (
        list(numeric_features)
        if numeric_features is not None
        else list(DEFAULT_NUMERIC_FEATURES) + list(STEP3B_MODEL_FEATURES)
    )
    numeric, categorical, skipped = select_features(
        df, requested_numeric, list(DEFAULT_CATEGORICAL_FEATURES)
    )
    if not numeric:
        raise ValueError("No usable numeric features found in the input CSV.")
    check_features_allowed(numeric + categorical)  # hard stop BEFORE training
    df = coerce_numeric_features(df, numeric, context="calibration")
    train, calib, test = chronological_three_way_split(
        df,
        train_frac=train_frac,
        calibration_frac=calibration_frac,
        train_end_date=train_end_date,
        calibration_end_date=calibration_end_date,
    )
    if skipped:
        print(f"Features requested but unusable (skipped): {skipped}")

    cols = numeric + categorical
    y_train = train[TARGET].astype(int)
    y_calib = calib[TARGET].astype(int).to_numpy()
    y_test = test[TARGET].astype(int).to_numpy()

    # --- base model: fit on TRAIN ONLY --------------------------------------
    base = make_logistic_pipeline(numeric, categorical)
    base.fit(train[cols], y_train)
    calib_probs = base.predict_proba(calib[cols])[:, 1]
    test_probs = base.predict_proba(test[cols])[:, 1]

    # --- calibrators: fit on CALIBRATION ONLY --------------------------------
    platt = PlattCalibrator().fit(calib_probs, y_calib)
    iso = IsotonicCalibrator().fit(calib_probs, y_calib)
    platt_test = platt.transform(test_probs)
    iso_test = iso.transform(test_probs)

    # --- evaluate everything on the SAME final test window -------------------
    results: dict = {
        "config": {
            "input": input_csv,
            "split_mode": "date" if train_end_date else "fraction",
            "train_frac": train_frac,
            "calibration_frac": calibration_frac,
            "train_end_date": train_end_date,
            "calibration_end_date": calibration_end_date,
            "n_train": int(len(train)),
            "n_calibration": int(len(calib)),
            "n_test": int(len(test)),
            "train_dates": [str(train["date"].min()), str(train["date"].max())],
            "calibration_dates": [str(calib["date"].min()), str(calib["date"].max())],
            "test_dates": [str(test["date"].min()), str(test["date"].max())],
            "rows_dropped": dropped,
            "features_numeric": numeric,
            "features_categorical": categorical,
            "features_skipped": skipped,
            "platt_input": "logit(p) of base-model probability",
            "platt_coefficients": platt.coefficients,
            "random_state": RANDOM_STATE,
            "base_model_label": (
                "Step 3C LR" if model_stem.startswith("step3c")
                else "Step 3B LR" if model_stem.startswith("step3b")
                else model_stem
            ),
        },
        "elo_baseline": _evaluate_calibrated(
            y_test, test["fighter_a_expected_win_prob"].to_numpy(dtype=float)
        ),
        "models": {
            "uncalibrated": _evaluate_calibrated(y_test, test_probs),
            "platt": _evaluate_calibrated(y_test, platt_test),
            "isotonic": _evaluate_calibrated(y_test, iso_test),
        },
    }

    # Optional context model: uncalibrated basic Step 3 LR on the same split.
    if include_step3_basic:
        b_num, b_cat, _ = select_features(
            df, list(DEFAULT_NUMERIC_FEATURES), list(DEFAULT_CATEGORICAL_FEATURES)
        )
        if b_num:
            check_features_allowed(b_num + b_cat)
            df_basic = coerce_numeric_features(df, b_num, context="step3_basic_calibration")
            basic_train, _, basic_test = chronological_three_way_split(
                df_basic,
                train_frac=train_frac,
                calibration_frac=calibration_frac,
                train_end_date=train_end_date,
                calibration_end_date=calibration_end_date,
            )
            basic = make_logistic_pipeline(b_num, b_cat)
            basic.fit(basic_train[b_num + b_cat], y_train)
            results["step3_basic_uncalibrated"] = _evaluate_calibrated(
                y_test, basic.predict_proba(basic_test[b_num + b_cat])[:, 1]
            )

    results["verdict"] = choose_best_method(results, n_calibration=len(calib))

    # --- outputs --------------------------------------------------------------
    out_dir, mdl_dir = Path(output_dir), Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mdl_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(base, mdl_dir / f"{model_stem}_uncalibrated.joblib")
    joblib.dump(CalibratedPipeline(base, platt),
                mdl_dir / f"{model_stem}_platt_calibrated.joblib")
    joblib.dump(CalibratedPipeline(base, iso),
                mdl_dir / f"{model_stem}_isotonic_calibrated.joblib")

    # Predictions: TEST rows only, so downstream analysis can't accidentally
    # average in-sample calibration-window rows into the reported metrics.
    pred_df = test[["fight_id", "date", "fighter_a", "fighter_b", TARGET]].copy()
    pred_df = pred_df.rename(columns={TARGET: "target"})
    pred_df["uncalibrated_probability"] = test_probs
    pred_df["platt_probability"] = platt_test
    pred_df["isotonic_probability"] = iso_test
    for name in ("uncalibrated", "platt", "isotonic"):
        pred_df[f"{name}_prediction"] = (
            pred_df[f"{name}_probability"] >= 0.5
        ).astype(int)
    pred_df["split"] = "test"
    pred_df.to_csv(out_dir / f"{output_prefix}calibration_predictions.csv", index=False)

    # Long-format bucket tables for all three probability sets.
    table_rows = []
    for name, entry in results["models"].items():
        for b in entry["calibration"]:
            table_rows.append({"model": name, **b})
    pd.DataFrame(table_rows).to_csv(
        out_dir / f"{output_prefix}calibration_tables.csv", index=False
    )

    with open(out_dir / f"{output_prefix}calibration_comparison.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print_calibration_report(results)
    return results


# ---------------------------------------------------------------------------
# Rolling/live calibration (Part B)
# ---------------------------------------------------------------------------
#
# LIVE STRATEGY (documented, deliberately simple): the base Step 3B logistic
# regression is trained on all fights BEFORE the rolling calibration window;
# the calibrator (Platt by default) is fit on the window itself — the most
# recent `calibration_window_days` of completed fights ending at
# --as-of-date (or the latest fight in the data). The saved base model +
# calibrator pair is what you would apply to FUTURE fights.
#
# Honesty note: live mode has NO held-out test set. The diagnostics it
# reports are measured on the calibration window, where the calibrated
# numbers are in-sample for the calibrator (they will look good by
# construction). They exist to sanity-check the fit, not to grade the
# model. For honest final-test evaluation use backtest mode
# (run_calibration), which never fits anything on its test window.

DEFAULT_CALIBRATION_WINDOW_DAYS = 730
DEFAULT_MIN_CALIBRATION_ROWS = 500


def run_live_calibration(
    input_csv: str,
    output_dir: str = "data/processed",
    model_dir: str = "data/models",
    calibration_window_days: int = DEFAULT_CALIBRATION_WINDOW_DAYS,
    min_calibration_rows: int = DEFAULT_MIN_CALIBRATION_ROWS,
    as_of_date: str | None = None,
    method: str = "platt",
    numeric_features: list[str] | None = None,
) -> dict:
    """Rolling-window calibration for FUTURE predictions (live mode).

    Chronological, never random: train = fights strictly before the window;
    calibration window = the last `calibration_window_days` up to and
    including as-of-date. Fights after as-of-date are excluded everywhere.
    """
    if method not in ("platt", "isotonic"):
        raise ValueError(f"Unknown calibration method: {method!r} "
                         "(use 'platt' or 'isotonic')")

    df = pd.read_csv(input_csv)
    before = len(df)
    df = df[df[TARGET].notna() & df["fighter_a_expected_win_prob"].notna()]
    dropped = before - len(df)
    df = df.sort_values(["date", "fight_id"], kind="stable").reset_index(drop=True)

    as_of = str(as_of_date) if as_of_date is not None else str(df["date"].max())
    future_dropped = int((df["date"] > as_of).sum())
    df = df[df["date"] <= as_of]  # no future fights, ever
    if df.empty:
        raise ValueError(f"No fights on or before --as-of-date {as_of}.")

    calibration_start = str(
        (pd.Timestamp(as_of) - pd.Timedelta(days=calibration_window_days)).date()
    )
    train = df[df["date"] < calibration_start]
    calib = df[df["date"] >= calibration_start]
    if len(train) == 0 or len(calib) == 0:
        raise ValueError(
            f"Empty live split: {len(train)} train / {len(calib)} calibration "
            f"rows (window {calibration_start} .. {as_of}). Adjust "
            "--calibration-window-days or --as-of-date."
        )

    warnings: list[str] = []
    if len(calib) < min_calibration_rows:
        warnings.append(
            f"calibration window has only {len(calib)} rows "
            f"(< --min-calibration-rows {min_calibration_rows}); the "
            "calibrator may be unreliable — widen the window."
        )

    requested_numeric = (
        list(numeric_features)
        if numeric_features is not None
        else list(DEFAULT_NUMERIC_FEATURES) + list(STEP3B_MODEL_FEATURES)
    )
    numeric, categorical, skipped = select_features(
        df, requested_numeric, list(DEFAULT_CATEGORICAL_FEATURES)
    )
    if not numeric:
        raise ValueError("No usable numeric features found in the input CSV.")
    check_features_allowed(numeric + categorical)
    df = coerce_numeric_features(df, numeric, context="live_calibration")
    train = df[df["date"] < calibration_start]
    calib = df[df["date"] >= calibration_start]
    cols = numeric + categorical

    print(
        f"LIVE calibration (as of {as_of}; window {calibration_window_days} days)\n"
        f"  train       {len(train):>5} rows ({train['date'].min()} .. {train['date'].max()})\n"
        f"  calibration {len(calib):>5} rows ({calib['date'].min()} .. {calib['date'].max()})\n"
        f"  future fights excluded: {future_dropped}"
    )

    # Base model: fights strictly before the window. Calibrator: the window.
    base = make_logistic_pipeline(numeric, categorical)
    base.fit(train[cols], train[TARGET].astype(int))
    y_calib = calib[TARGET].astype(int).to_numpy()
    raw_probs = base.predict_proba(calib[cols])[:, 1]

    calibrator = PlattCalibrator() if method == "platt" else IsotonicCalibrator()
    calibrator.fit(raw_probs, y_calib)
    cal_probs = calibrator.transform(raw_probs)

    raw_metrics = _evaluate_calibrated(y_calib, raw_probs)
    cal_metrics = _evaluate_calibrated(y_calib, cal_probs)

    report: dict = {
        "mode": "live",
        "method": method,
        "note": (
            "Diagnostics are measured ON the calibration window; calibrated "
            "numbers are in-sample for the calibrator and are NOT a final-test "
            "evaluation. Use --mode backtest for honest held-out metrics."
        ),
        "config": {
            "input": input_csv,
            "calibration_window_days": calibration_window_days,
            "min_calibration_rows": min_calibration_rows,
            "as_of_date": as_of,
            "latest_fight_date": str(df["date"].max()),
            "future_fights_excluded": future_dropped,
            "calibration_start": calibration_start,
            "n_train": int(len(train)),
            "n_calibration": int(len(calib)),
            "train_dates": [str(train["date"].min()), str(train["date"].max())],
            "calibration_dates": [str(calib["date"].min()), str(calib["date"].max())],
            "rows_dropped": dropped,
            "features_numeric": numeric,
            "features_categorical": categorical,
            "features_skipped": skipped,
            "random_state": RANDOM_STATE,
        },
        "calibration_window_diagnostics": {
            "raw": raw_metrics,
            method: cal_metrics,
        },
        "warnings": warnings,
    }
    if method == "platt":
        report["config"]["platt_coefficients"] = calibrator.coefficients

    # --- outputs (live_* names so backtest outputs are never overwritten) ---
    out_dir, mdl_dir = Path(output_dir), Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mdl_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(base, mdl_dir / "step3b_logistic_regression_live_uncalibrated.joblib")
    joblib.dump(
        CalibratedPipeline(base, calibrator),
        mdl_dir / f"step3b_logistic_regression_live_{method}_calibrated.joblib",
    )

    pred_df = calib[["fight_id", "date", "fighter_a", "fighter_b", TARGET]].copy()
    pred_df = pred_df.rename(columns={TARGET: "target"})
    pred_df["raw_probability"] = raw_probs
    pred_df[f"{method}_probability"] = cal_probs
    pred_df["split"] = "calibration"  # window rows; NOT held-out test rows
    pred_df.to_csv(out_dir / "live_calibration_predictions.csv", index=False)

    table_rows = []
    for label, metrics in (("raw", raw_metrics), (method, cal_metrics)):
        for b in metrics["calibration"]:
            table_rows.append({"model": label, **b})
    pd.DataFrame(table_rows).to_csv(out_dir / "live_calibration_table.csv", index=False)

    with open(out_dir / "live_calibration_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print_live_report(report)
    return report


def print_live_report(report: dict) -> None:
    cfg = report["config"]
    diag = report["calibration_window_diagnostics"]
    method = report["method"]
    raw, cal = diag["raw"], diag[method]

    print(f"\nLive calibration diagnostics (window {cfg['calibration_start']}"
          f" .. {cfg['as_of_date']}, {cfg['n_calibration']} fights)")
    print("NOTE: calibrated numbers are in-sample for the calibrator — this is "
          "a fit sanity check, not a final-test evaluation.")
    print(f"{'':<14}{'log_loss':>9} {'brier':>8}")
    print(f"{'raw':<14}{raw['log_loss']:>9.4f} {raw['brier_score']:>8.4f}")
    print(f"{method:<14}{cal['log_loss']:>9.4f} {cal['brier_score']:>8.4f}")
    for thr in HIGH_CONFIDENCE_THRESHOLDS:
        r = raw["high_confidence"][f"above_{thr}"]
        c = cal["high_confidence"][f"above_{thr}"]
        rg = f"{r['calibration_gap']:+.3f}" if r["calibration_gap"] is not None else "n/a"
        cg = f"{c['calibration_gap']:+.3f}" if c["calibration_gap"] is not None else "n/a"
        print(f"  above {thr}: raw gap {rg} (n={r['n_predictions']}) -> "
              f"{method} gap {cg} (n={c['n_predictions']})")
    for w in report["warnings"]:
        print(f"  WARNING: {w}")
    print("Saved base model + calibrator are ready to apply to FUTURE fights.")


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

def _model_labels(results: dict) -> dict:
    base = results.get("config", {}).get("base_model_label", "Step 3B LR")
    return {
        "uncalibrated": f"{base} (uncalibrated)",
        "platt": f"{base} + Platt",
        "isotonic": f"{base} + isotonic",
    }


def print_calibration_report(results: dict) -> None:
    _MODEL_LABELS = _model_labels(results)
    def fmt(m):
        auc = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "  n/a "
        return (f"{m['accuracy']:>8.4f} {m['log_loss']:>9.4f} "
                f"{auc:>8} {m['brier_score']:>8.4f}")

    print("\nCalibration comparison (same final chronological test window)")
    print(f"{'model':<30}{'acc':>8} {'log_loss':>9} {'roc_auc':>8} {'brier':>8}")
    print(f"{'Elo baseline (untrained)':<30}{fmt(results['elo_baseline'])}")
    if "step3_basic_uncalibrated" in results:
        print(f"{'Step 3 basic LR (uncal.)':<30}"
              f"{fmt(results['step3_basic_uncalibrated'])}")
    for key, label in _MODEL_LABELS.items():
        print(f"{label:<30}{fmt(results['models'][key])}")

    print("\nHigh-confidence diagnostics (gap = avg predicted - actual; + = overconfident)")
    for thr in HIGH_CONFIDENCE_THRESHOLDS:
        print(f"  predictions above {thr}:")
        for key, label in _MODEL_LABELS.items():
            hc = results["models"][key]["high_confidence"][f"above_{thr}"]
            if hc["n_predictions"] == 0:
                print(f"    {label:<28} n=0")
                continue
            print(
                f"    {label:<28} n={hc['n_predictions']:<4} "
                f"avg pred={hc['avg_predicted_prob']:.3f} "
                f"actual={hc['actual_win_rate']:.3f} "
                f"gap={hc['calibration_gap']:+.3f}"
            )

    print("\nBucketed calibration (test window):")
    for key, label in _MODEL_LABELS.items():
        print(f"  {label}:")
        print(f"    {'bucket':>9} {'n':>5} {'avg pred':>9} {'actual':>7} {'gap':>7}")
        for b in results["models"][key]["calibration"]:
            if b["n_predictions"] == 0:
                continue
            print(
                f"    {b['bucket']:>9} {b['n_predictions']:>5} "
                f"{b['avg_predicted_prob']:>9.3f} {b['actual_win_rate']:>7.3f} "
                f"{b['gap']:>+7.3f}"
            )

    v = results["verdict"]
    unc = results["models"]["uncalibrated"]
    best = results["models"][v["best_method"]]
    print("\nVerdict:")
    print(f"  Platt improved log loss:    {'YES' if v['platt_improved_log_loss'] else 'NO'} "
          f"({results['models']['platt']['log_loss']:.4f} vs {unc['log_loss']:.4f} uncalibrated)")
    print(f"  Isotonic improved log loss: {'YES' if v['isotonic_improved_log_loss'] else 'NO'} "
          f"({results['models']['isotonic']['log_loss']:.4f} vs {unc['log_loss']:.4f} uncalibrated)")
    print(f"  Best method by test log loss: {v['best_method']} ({v['best_log_loss']:.4f})")
    g = v["overconfidence_above_0.7"]
    if g["uncalibrated_gap"] is not None and g["best_gap"] is not None:
        improved = abs(g["best_gap"]) < abs(g["uncalibrated_gap"])
        print(f"  Overconfidence above 0.7:   gap {g['uncalibrated_gap']:+.3f} -> "
              f"{g['best_gap']:+.3f} ({'REDUCED' if improved else 'NOT reduced'})")
    print(f"  Brier improved:             {'YES' if v['best_improved_brier'] else 'NO'} "
          f"({best['brier_score']:.4f} vs {unc['brier_score']:.4f})")
    if v["auc_change"] is not None:
        print(f"  AUC change:                 {v['auc_change']:+.4f} "
              "(calibration is monotonic-ish; large changes would be suspicious)")
    print(f"  Accuracy change:            {v['accuracy_change']:+.4f} "
          "(calibration reshapes probabilities, not necessarily picks)")
    if v["isotonic_overfit_risk"]:
        print(f"  NOTE: only {v['n_calibration']} calibration rows -> isotonic "
              "is overfit-prone; Platt preferred on ties.")
    safer = (v["platt_improved_log_loss"] or v["isotonic_improved_log_loss"])
    print(
        "  The calibrated model IS safer to use for probability estimates."
        if safer else
        "  Calibration did NOT improve probability quality on this test window; "
        "keep using uncalibrated probabilities."
    )
