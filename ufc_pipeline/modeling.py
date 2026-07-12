"""Model training + evaluation (Part 4).

Trains a first ML model on the time-safe pre-fight features and answers one
question: does it improve on the Elo-only baseline?

Design decisions:

* CHRONOLOGICAL split, never random. Fights are sorted by (date, fight_id);
  the model trains on older fights and is tested on newer ones. A random
  split would let the model "see the future" (e.g. train on a fighter's 2023
  form to predict their 2021 fight), inflating every metric.
* The Elo baseline is evaluated on the SAME test rows, using
  fighter_a_expected_win_prob directly as the predicted probability. It
  needs no training, so it's computed independently of the ML pipeline.
* All preprocessing (imputation, scaling, one-hot) lives INSIDE the sklearn
  Pipeline, so the exact same transformations are applied at predict time
  and nothing is fit on test data.
* A leakage guard validates every selected feature name against forbidden
  patterns BEFORE training and hard-stops on a violation.

Missing values: numeric features are median-imputed inside the pipeline
(medians learned from the training split only). Rows are NOT dropped for
missing features; any drops (e.g. missing target) are counted and printed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ufc_pipeline.layoff_features import (
    LAYOFF_CANDIDATE_A_FEATURES,
    LAYOFF_CANDIDATE_B_FEATURES,
    LAYOFF_FEATURE_COLUMNS,
)

RANDOM_STATE = 42
TARGET = "fighter_a_won"

# Default model inputs: strictly pre-fight, A-minus-B difference features.
DEFAULT_NUMERIC_FEATURES = [
    "elo_diff",
    "age_diff",
    "height_diff",
    "reach_diff",
    "prior_fights_diff",
    "prior_wins_diff",
    "prior_losses_diff",
    "prior_win_pct_diff",
    "wins_last_3_diff",
    "win_pct_last_3_diff",
    "wins_last_5_diff",
    "win_pct_last_5_diff",
    "days_since_last_fight_diff",
]
DEFAULT_CATEGORICAL_FEATURES = ["weight_class"]

# Step 3B: historical rolling striking/grappling diffs + missing-history flags.
# These names legitimately contain sig_str/takedown/control, so they are
# ALLOWLISTED BY EXACT NAME in the leakage guard: they are pre-fight rolling
# aggregates over PREVIOUS fights only (see ufc_pipeline/stats_features.py),
# not current-fight stats.
STEP3B_MODEL_FEATURES = [
    "sig_strikes_landed_per_min_diff",
    "sig_strikes_absorbed_per_min_diff",
    "sig_strike_differential_diff",
    "striking_accuracy_diff",
    "striking_defense_diff",
    "knockdown_rate_diff",
    "takedowns_per_15_diff",
    "takedown_accuracy_diff",
    "takedown_defense_diff",
    "control_time_per_15_diff",
    "submission_attempts_per_15_diff",
    "fighter_a_no_prior_stats",
    "fighter_b_no_prior_stats",
]

# Step 3C: style-matchup interactions + rolling "against" stats (see
# ufc_pipeline/matchup_features.py for the exact formulas). All are
# pre-fight rolling aggregates / interactions of them — allowlisted by
# exact name for the same reason as Step 3B. Positive = fighter A advantage.
STEP3C_MODEL_FEATURES = [
    "takedowns_allowed_per_15_diff",
    "opp_takedown_attempts_per_15_diff",
    "opp_sig_str_attempted_per_min_diff",
    "control_time_absorbed_per_15_diff",
    "knockdowns_absorbed_per_15_diff",
    "submission_attempts_absorbed_per_15_diff",
    "striking_matchup_net_advantage",
    "striking_accuracy_matchup_net_advantage",
    "takedown_matchup_net_advantage",
    "takedown_accuracy_matchup_net_advantage",
    "control_matchup_net_advantage",
    "knockdown_matchup_net_advantage",
    "submission_matchup_net_advantage",
    "reach_volume_interaction",
    "pace_pressure_advantage",
    "opponent_pressure_absorption_advantage",
    "matchup_history_missing",
]
ALLOWED_HISTORICAL_FEATURES = (
    set(STEP3B_MODEL_FEATURES)
    | set(STEP3C_MODEL_FEATURES)
    | set(LAYOFF_FEATURE_COLUMNS)
)

# Any selected input feature matching one of these substrings aborts training.
# (fighter_a_won is allowed as the TARGET only, and is checked separately.)
# td_/ctrl/knockdown/sub_att cover the RAW fight_stats column names
# (td_landed, ctrl_seconds, knockdowns, sub_attempts) so current-fight
# totals can never sneak in as model inputs under their storage names.
FORBIDDEN_FEATURE_PATTERNS = [
    "winner", "loser", "post", "method", "round", "odds", "rank",
    "sig_str", "takedown", "control", "result", "outcome",
    "td_", "ctrl", "knockdown", "sub_att",
]


def check_features_allowed(features: list[str], target: str = TARGET) -> None:
    """Leakage guard: refuse target-as-feature and forbidden name patterns."""
    violations: list[str] = []
    for feat in features:
        low = str(feat).lower()
        if feat == target:
            violations.append(f"{feat} (this is the target, not a feature)")
            continue
        if feat in ALLOWED_HISTORICAL_FEATURES:
            # Exact-name allowlist: time-safe rolling PRE-fight aggregates.
            continue
        hits = [p for p in FORBIDDEN_FEATURE_PATTERNS if p in low]
        if hits:
            violations.append(f"{feat} (matches forbidden pattern(s): {hits})")
    if violations:
        raise ValueError(
            "Leakage guard: refusing to train with forbidden input features:\n  - "
            + "\n  - ".join(violations)
            + "\nModel inputs must be known BEFORE the fight."
        )


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    split_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Older fights -> train, newer fights -> test. Never random."""
    ordered = df.sort_values(["date", "fight_id"], kind="stable").reset_index(drop=True)
    if split_date is not None:
        train = ordered[ordered["date"] < split_date]
        test = ordered[ordered["date"] >= split_date]
    else:
        cut = int(round(len(ordered) * (1.0 - test_size)))
        train, test = ordered.iloc[:cut], ordered.iloc[cut:]
    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            f"Empty split: {len(train)} train / {len(test)} test rows. "
            "Adjust --split-date or --test-size."
        )
    return train, test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calibration_table(y_true: np.ndarray, probs: np.ndarray) -> list[dict]:
    """10 fixed buckets [0.0-0.1) ... [0.9-1.0]; per-bucket count/avg/actual."""
    edges = np.linspace(0.0, 1.0, 11)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & ((probs < hi) if hi < 1.0 else (probs <= hi))
        n = int(mask.sum())
        rows.append(
            {
                "bucket": f"{lo:.1f}-{hi:.1f}",
                "n_predictions": n,
                "avg_predicted_prob": float(probs[mask].mean()) if n else None,
                "actual_win_rate": float(y_true[mask].mean()) if n else None,
            }
        )
    return rows


def evaluate_probs(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """All metrics for one set of predicted P(fighter_a wins)."""
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "log_loss": float(log_loss(y_true, probs, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probs)),
        "avg_predicted_prob": float(probs.mean()),
        "actual_win_rate": float(y_true.mean()),
        "calibration": calibration_table(y_true, probs),
    }
    # ROC AUC is undefined if the test window contains only one class.
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
    else:
        metrics["roc_auc"] = None
    return metrics


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = [
        (
            "num",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            numeric,
        )
    ]
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def make_logistic_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(numeric, categorical)),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )


def make_random_forest_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(numeric, categorical)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def extract_feature_names(pipeline: Pipeline) -> list[str]:
    """Post-transform feature names (numeric + one-hot expanded)."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    return [re.sub(r"^(num|cat)__", "", n) for n in names]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def select_features(
    df: pd.DataFrame,
    requested_numeric: list[str],
    requested_categorical: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Intersect requested features with available columns; report skips.

    Columns that exist but are ENTIRELY null are also skipped (imputation
    has nothing to learn from them) and reported as '<name> (all null)'.
    """
    skipped = [
        f for f in requested_numeric + requested_categorical if f not in df.columns
    ]
    numeric, categorical = [], []
    for f in requested_numeric:
        if f in df.columns:
            (numeric if df[f].notna().any() else skipped).append(
                f if df[f].notna().any() else f"{f} (all null)"
            )
    for f in requested_categorical:
        if f in df.columns:
            (categorical if df[f].notna().any() else skipped).append(
                f if df[f].notna().any() else f"{f} (all null)"
            )
    return numeric, categorical, skipped


def coerce_numeric_features(
    df: pd.DataFrame,
    numeric: list[str],
    context: str = "model",
) -> pd.DataFrame:
    """Return a copy with numeric features coerced, or fail on bad values.

    CSV inputs can turn a numeric feature into object dtype when one bad token
    sneaks in. Median imputation would then fail inside sklearn with a less
    useful error, so validate and coerce once at the boundary.
    """
    bad: list[str] = []
    for col in numeric:
        present = df[col].notna()
        if not present.any():
            continue
        converted = pd.to_numeric(df.loc[present, col], errors="coerce")
        invalid = present.copy()
        invalid.loc[present] = converted.isna() | ~np.isfinite(converted.to_numpy())
        if invalid.any():
            examples = df.loc[invalid, col].astype(str).head(3).tolist()
            bad.append(f"{col} (examples: {examples})")
    if bad:
        raise ValueError(
            f"[{context}] non-numeric values in numeric feature columns:\n  - "
            + "\n  - ".join(bad)
        )

    out = df.copy()
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def train_and_evaluate(
    input_csv: str,
    model_output_dir: str = "data/models",
    predictions_output: str = "data/processed/model_predictions.csv",
    metrics_output: str = "data/processed/model_evaluation.json",
    coefficients_output: str = "data/processed/logistic_regression_coefficients.csv",
    rf_importances_output: str = "data/processed/random_forest_feature_importances.csv",
    split_date: str | None = None,
    test_size: float = 0.2,
    include_random_forest: bool = False,
    use_categorical: bool = True,
    features: list[str] | None = None,
) -> dict:
    """Full Part-4 run. Returns the metrics dict (also written to JSON)."""
    df = pd.read_csv(input_csv)

    # --- feature selection -------------------------------------------------
    if features is not None:
        requested_numeric = [f for f in features if f != "weight_class"]
        requested_categorical = (
            ["weight_class"] if ("weight_class" in features and use_categorical) else []
        )
    else:
        requested_numeric = list(DEFAULT_NUMERIC_FEATURES)
        requested_categorical = (
            list(DEFAULT_CATEGORICAL_FEATURES) if use_categorical else []
        )

    numeric, categorical, skipped = select_features(
        df, requested_numeric, requested_categorical
    )
    if not numeric:
        raise ValueError("No usable numeric features found in the input CSV.")

    # --- leakage guard: hard stop BEFORE any training ----------------------
    check_features_allowed(numeric + categorical)

    print(f"Features used ({len(numeric + categorical)}): {numeric + categorical}")
    if skipped:
        print(f"Features requested but missing from input (skipped): {skipped}")

    # --- rows: report drops, never drop silently ---------------------------
    before = len(df)
    df = df[df[TARGET].notna() & df["fighter_a_expected_win_prob"].notna()]
    dropped = before - len(df)
    df = coerce_numeric_features(df, numeric, context="train_and_evaluate")
    print(f"Rows: {len(df)} usable ({dropped} dropped for missing target/Elo prob)")

    # --- chronological split ------------------------------------------------
    train, test = chronological_split(df, test_size=test_size, split_date=split_date)
    print(
        f"Chronological split: {len(train)} train "
        f"({train['date'].min()} .. {train['date'].max()}), "
        f"{len(test)} test ({test['date'].min()} .. {test['date'].max()})"
    )

    X_train, y_train = train[numeric + categorical], train[TARGET].astype(int)
    X_test, y_test = test[numeric + categorical], test[TARGET].astype(int)

    results: dict = {
        "config": {
            "input": input_csv,
            "split_date": split_date,
            "test_size": test_size,
            "features_numeric": numeric,
            "features_categorical": categorical,
            "features_skipped": skipped,
            "rows_dropped": dropped,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "random_state": RANDOM_STATE,
        }
    }

    # --- 1. Elo-only baseline (no training required) -----------------------
    elo_probs = test["fighter_a_expected_win_prob"].to_numpy(dtype=float)
    results["elo_baseline"] = evaluate_probs(y_test.to_numpy(), elo_probs)

    # --- 2. Logistic regression --------------------------------------------
    logit = make_logistic_pipeline(numeric, categorical)
    logit.fit(X_train, y_train)
    logit_probs = logit.predict_proba(X_test)[:, 1]
    results["logistic_regression"] = evaluate_probs(y_test.to_numpy(), logit_probs)

    Path(model_output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(logit, Path(model_output_dir) / "logistic_regression_prefight.joblib")

    # Coefficients: which features the model leans on (not causal claims).
    coef_names = extract_feature_names(logit)
    coefs = logit.named_steps["model"].coef_.ravel()
    coef_df = pd.DataFrame(
        {"feature": coef_names, "coefficient": coefs, "absolute_coefficient": np.abs(coefs)}
    ).sort_values("absolute_coefficient", ascending=False)
    Path(coefficients_output).parent.mkdir(parents=True, exist_ok=True)
    coef_df.to_csv(coefficients_output, index=False)

    # --- 3. Optional random forest ------------------------------------------
    rf_probs = None
    if include_random_forest:
        rf = make_random_forest_pipeline(numeric, categorical)
        rf.fit(X_train, y_train)
        rf_probs = rf.predict_proba(X_test)[:, 1]
        results["random_forest"] = evaluate_probs(y_test.to_numpy(), rf_probs)
        joblib.dump(rf, Path(model_output_dir) / "random_forest_prefight.joblib")
        rf_imp = pd.DataFrame(
            {
                "feature": extract_feature_names(rf),
                "importance": rf.named_steps["model"].feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        rf_imp.to_csv(rf_importances_output, index=False)

    # --- predictions CSV -----------------------------------------------------
    pred_df = test[
        ["fight_id", "date", "event", "fighter_a", "fighter_b", "winner", TARGET]
    ].copy()
    pred_df["elo_pred_prob"] = elo_probs
    pred_df["logistic_pred_prob"] = logit_probs
    pred_df["logistic_pred_class"] = (logit_probs >= 0.5).astype(int)
    if rf_probs is not None:
        pred_df["random_forest_pred_prob"] = rf_probs
        pred_df["random_forest_pred_class"] = (rf_probs >= 0.5).astype(int)
    Path(predictions_output).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(predictions_output, index=False)

    # --- metrics JSON ---------------------------------------------------------
    Path(metrics_output).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output, "w") as fh:
        json.dump(results, fh, indent=2)

    print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

_MODEL_LABELS = {
    "elo_baseline": "Elo baseline",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
}


def print_summary(results: dict) -> None:
    cfg = results["config"]
    print("\nModel comparison")
    print(f"(train rows: {cfg['n_train']}, test rows: {cfg['n_test']})")
    for key, label in _MODEL_LABELS.items():
        if key not in results:
            continue
        m = results[key]
        auc = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "n/a (one class)"
        print(f"\n{label}:")
        print(f"  Accuracy   : {m['accuracy']:.4f}")
        print(f"  Log loss   : {m['log_loss']:.4f}")
        print(f"  ROC AUC    : {auc}")
        print(f"  Brier score: {m['brier_score']:.4f}")
        print(f"  Avg pred p : {m['avg_predicted_prob']:.4f}"
              f"   Actual win rate: {m['actual_win_rate']:.4f}")

    if "logistic_regression" in results:
        elo_ll = results["elo_baseline"]["log_loss"]
        lr_ll = results["logistic_regression"]["log_loss"]
        verdict = "BEATS" if lr_ll < elo_ll else "does NOT beat"
        print(
            f"\nLogistic regression {verdict} the Elo baseline on log loss "
            f"({lr_ll:.4f} vs {elo_ll:.4f})."
        )
        print("Calibration by bucket (logistic regression):")
        print(f"  {'bucket':>9} {'n':>5} {'avg pred':>9} {'actual':>7}")
        for b in results["logistic_regression"]["calibration"]:
            if b["n_predictions"] == 0:
                continue
            print(
                f"  {b['bucket']:>9} {b['n_predictions']:>5} "
                f"{b['avg_predicted_prob']:>9.3f} {b['actual_win_rate']:>7.3f}"
            )


# ---------------------------------------------------------------------------
# Multi-model comparison (Step 3B evaluation): Elo vs basic vs basic+3B
# ---------------------------------------------------------------------------

def compare_feature_sets(
    input_csv: str,
    feature_sets: dict[str, dict] | None = None,
    split_date: str | None = None,
    test_size: float = 0.2,
    include_random_forest: bool = False,
    metrics_output: str = "data/processed/model_comparison.json",
    predictions_output: str = "data/processed/model_comparison_predictions.csv",
    coefficients_dir: str = "data/processed",
) -> dict:
    """Evaluate several feature sets on the SAME chronological split.

    Default sets:
      step3_basic    : the Step 3 features (+ weight_class)
      step3_plus_3b  : Step 3 features + Step 3B rolling stats + flags
      step3c_matchup : the above + Step 3C matchup features — added only
                       when the input CSV actually carries 3C columns, so
                       Step 3B-only workflows are unchanged.
    The untrained Elo baseline is always evaluated on the same test rows.
    """
    df = pd.read_csv(input_csv)

    if feature_sets is None:
        feature_sets = {
            "step3_basic": {
                "numeric": list(DEFAULT_NUMERIC_FEATURES),
                "categorical": list(DEFAULT_CATEGORICAL_FEATURES),
            },
            "step3_plus_3b": {
                "numeric": list(DEFAULT_NUMERIC_FEATURES) + list(STEP3B_MODEL_FEATURES),
                "categorical": list(DEFAULT_CATEGORICAL_FEATURES),
            },
        }
        if any(f in df.columns for f in STEP3C_MODEL_FEATURES):
            feature_sets["step3c_matchup"] = {
                "numeric": list(DEFAULT_NUMERIC_FEATURES)
                + list(STEP3B_MODEL_FEATURES)
                + list(STEP3C_MODEL_FEATURES),
                "categorical": list(DEFAULT_CATEGORICAL_FEATURES),
            }
    before = len(df)
    df = df[df[TARGET].notna() & df["fighter_a_expected_win_prob"].notna()]
    dropped = before - len(df)

    train, test = chronological_split(df, test_size=test_size, split_date=split_date)
    y_train = train[TARGET].astype(int)
    y_test = test[TARGET].astype(int).to_numpy()
    print(
        f"Rows: {len(df)} usable ({dropped} dropped) | "
        f"split: {len(train)} train / {len(test)} test "
        f"(test from {test['date'].min()})"
    )

    results: dict = {
        "config": {
            "input": input_csv,
            "split_date": split_date,
            "test_size": test_size,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "rows_dropped": dropped,
            "random_state": RANDOM_STATE,
        },
        "elo_baseline": evaluate_probs(
            y_test, test["fighter_a_expected_win_prob"].to_numpy(dtype=float)
        ),
        "models": {},
    }

    preds = test[
        ["fight_id", "date", "event", "fighter_a", "fighter_b", "winner", TARGET]
    ].copy()
    preds["elo_pred_prob"] = test["fighter_a_expected_win_prob"].to_numpy(dtype=float)

    for name, spec in feature_sets.items():
        numeric, categorical, skipped = select_features(
            df, spec.get("numeric", []), spec.get("categorical", [])
        )
        if not numeric:
            print(f"[{name}] skipped entirely: no usable numeric features")
            continue
        check_features_allowed(numeric + categorical)
        df_model = coerce_numeric_features(df, numeric, context=name)
        train_model, test_model = chronological_split(
            df_model, test_size=test_size, split_date=split_date
        )
        X_train = train_model[numeric + categorical]
        X_test = test_model[numeric + categorical]
        entry: dict = {"features_numeric": numeric,
                       "features_categorical": categorical,
                       "features_skipped": skipped}

        logit = make_logistic_pipeline(numeric, categorical)
        logit.fit(X_train, y_train)
        p = logit.predict_proba(X_test)[:, 1]
        entry["logistic_regression"] = evaluate_probs(y_test, p)
        preds[f"{name}_logistic_prob"] = p

        coef_names = extract_feature_names(logit)
        coefs = logit.named_steps["model"].coef_.ravel()
        coef_df = pd.DataFrame(
            {"feature": coef_names, "coefficient": coefs,
             "absolute_coefficient": np.abs(coefs)}
        ).sort_values("absolute_coefficient", ascending=False)
        coef_path = Path(coefficients_dir) / f"coefficients_{name}.csv"
        coef_path.parent.mkdir(parents=True, exist_ok=True)
        coef_df.to_csv(coef_path, index=False)
        entry["coefficients_csv"] = str(coef_path)

        if include_random_forest:
            rf = make_random_forest_pipeline(numeric, categorical)
            rf.fit(X_train, y_train)
            rp = rf.predict_proba(X_test)[:, 1]
            entry["random_forest"] = evaluate_probs(y_test, rp)
            preds[f"{name}_rf_prob"] = rp
            imp = pd.DataFrame(
                {"feature": extract_feature_names(rf),
                 "importance": rf.named_steps["model"].feature_importances_}
            ).sort_values("importance", ascending=False)
            imp_path = Path(coefficients_dir) / f"rf_importances_{name}.csv"
            imp.to_csv(imp_path, index=False)
            entry["rf_importances_csv"] = str(imp_path)

        results["models"][name] = entry

    Path(predictions_output).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(predictions_output, index=False)
    Path(metrics_output).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output, "w") as fh:
        json.dump(results, fh, indent=2)

    print_comparison(results)
    return results


def print_comparison(results: dict) -> None:
    def fmt(m):
        auc = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "  n/a "
        return (f"{m['accuracy']:>8.4f} {m['log_loss']:>9.4f} "
                f"{auc:>8} {m['brier_score']:>8.4f}")

    print("\nModel comparison (same chronological test split)")
    print(f"{'model':<34}{'acc':>8} {'log_loss':>9} {'roc_auc':>8} {'brier':>8}")
    print(f"{'Elo baseline (untrained)':<34}{fmt(results['elo_baseline'])}")
    for name, entry in results["models"].items():
        print(f"{'LR: ' + name:<34}{fmt(entry['logistic_regression'])}")
        if "random_forest" in entry:
            print(f"{'RF: ' + name:<34}{fmt(entry['random_forest'])}")

    names = list(results["models"])
    elo_ll = results["elo_baseline"]["log_loss"]
    # Each expanded feature set is judged against the previous one.
    for prev, curr in zip(names, names[1:]):
        prev_ll = results["models"][prev]["logistic_regression"]["log_loss"]
        curr_ll = results["models"][curr]["logistic_regression"]["log_loss"]
        verdict = "IMPROVES on" if curr_ll < prev_ll else "does NOT improve on"
        print(
            f"\n{curr} {verdict} {prev} "
            f"(log loss {curr_ll:.4f} vs {prev_ll:.4f}; Elo baseline {elo_ll:.4f})."
        )
