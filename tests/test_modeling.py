"""Tests for model training and evaluation (Part 4).

Covers the 10 required behaviors:
 1. Train/test split is chronological, not random.
 2. The target fighter_a_won is not included in X.
 3. Forbidden columns are rejected as input features.
 4. The Elo baseline is evaluated separately from the ML model.
 5. Logistic regression probabilities are between 0 and 1.
 6. model_predictions.csv has the expected columns.
 7. model_evaluation.json has metrics for Elo baseline + logistic regression.
 8. Same input + random_state -> deterministic predictions.
 9. Missing numeric values are handled by the pipeline (no crash).
10. Categorical columns are one-hot encoded safely (unseen categories OK).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufc_pipeline.modeling import (  # noqa: E402
    TARGET,
    check_features_allowed,
    chronological_split,
    evaluate_probs,
    train_and_evaluate,
)


def synthetic_features(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A small ufc_prefight_features-shaped dataset with real signal in
    elo_diff so the model has something to learn."""
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0, 80, n)
    p = 1 / (1 + 10 ** (-elo_diff / 400))
    won = (rng.random(n) < p).astype(int)
    dates = pd.date_range("2015-01-01", periods=n, freq="7D").strftime("%Y-%m-%d")
    df = pd.DataFrame(
        {
            "fight_id": np.arange(1, n + 1),
            "date": dates,
            "event": [f"UFC {i}" for i in range(n)],
            "fighter_a": [f"A{i}" for i in range(n)],
            "fighter_b": [f"B{i}" for i in range(n)],
            "winner": ["x"] * n,
            TARGET: won,
            "weight_class": rng.choice(["Lightweight", "Welterweight", "Bantamweight"], n),
            "fighter_a_expected_win_prob": p,
            "fighter_b_expected_win_prob": 1 - p,
            "elo_diff": elo_diff,
            "age_diff": rng.normal(0, 4, n),
            "height_diff": rng.normal(0, 5, n),
            "reach_diff": rng.normal(0, 6, n),
            "prior_fights_diff": rng.integers(-5, 6, n),
            "prior_wins_diff": rng.integers(-4, 5, n),
            "prior_losses_diff": rng.integers(-4, 5, n),
            "prior_win_pct_diff": rng.normal(0, 0.3, n),
            "wins_last_3_diff": rng.integers(-3, 4, n),
            "win_pct_last_3_diff": rng.normal(0, 0.4, n),
            "wins_last_5_diff": rng.integers(-5, 6, n),
            "win_pct_last_5_diff": rng.normal(0, 0.35, n),
            "days_since_last_fight_diff": rng.normal(0, 120, n),
        }
    )
    df["winner"] = np.where(won == 1, df["fighter_a"], df["fighter_b"])
    return df


def run_training(df, tmp_path, **kwargs):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv = tmp_path / "features.csv"
    df.to_csv(csv, index=False)
    return train_and_evaluate(
        input_csv=str(csv),
        model_output_dir=str(tmp_path / "models"),
        predictions_output=str(tmp_path / "preds.csv"),
        metrics_output=str(tmp_path / "metrics.json"),
        coefficients_output=str(tmp_path / "coefs.csv"),
        rf_importances_output=str(tmp_path / "rf_imp.csv"),
        **kwargs,
    )


# --------------------------------------------------------------------- 1
def test_split_is_chronological():
    df = synthetic_features(100)
    shuffled = df.sample(frac=1.0, random_state=1)  # feed it out of order
    train, test = chronological_split(shuffled, test_size=0.2)
    assert len(train) == 80 and len(test) == 20
    assert train["date"].max() <= test["date"].min()   # no time overlap
    assert list(test["date"]) == sorted(test["date"])  # newest fights are test


def test_split_by_date():
    df = synthetic_features(100)
    train, test = chronological_split(df, split_date="2016-01-01")
    assert (train["date"] < "2016-01-01").all()
    assert (test["date"] >= "2016-01-01").all()


# --------------------------------------------------------------------- 2
def test_target_not_in_model_inputs(tmp_path):
    results = run_training(synthetic_features(), tmp_path)
    used = (
        results["config"]["features_numeric"]
        + results["config"]["features_categorical"]
    )
    assert TARGET not in used
    assert "winner" not in used
    # and the guard would reject it explicitly:
    with pytest.raises(ValueError, match="target"):
        check_features_allowed(["elo_diff", TARGET])


# --------------------------------------------------------------------- 3
def test_forbidden_features_rejected(tmp_path):
    for bad in ("method", "round", "red_odds", "rank_diff",
                "fighter_a_post_elo", "winner_streak", "avg_sig_str_diff",
                "takedown_diff", "control_time_diff", "result_flag"):
        with pytest.raises(ValueError, match="Leakage guard"):
            check_features_allowed(["elo_diff", bad])
    # end-to-end: training with a forbidden feature must hard-stop
    df = synthetic_features()
    df["fighter_a_post_elo"] = 1600.0
    with pytest.raises(ValueError, match="Leakage guard"):
        run_training(df, tmp_path, features=["elo_diff", "fighter_a_post_elo"])
    # allowed defaults pass:
    check_features_allowed(["elo_diff", "age_diff", "weight_class"])


# --------------------------------------------------------------------- 4
def test_elo_baseline_evaluated_separately(tmp_path):
    results = run_training(synthetic_features(), tmp_path)
    assert "elo_baseline" in results and "logistic_regression" in results
    # Elo baseline == metrics computed directly from the Elo probabilities,
    # with no trained model involved:
    df = synthetic_features()
    _, test = chronological_split(df, test_size=0.2)
    direct = evaluate_probs(
        test[TARGET].to_numpy(), test["fighter_a_expected_win_prob"].to_numpy()
    )
    assert results["elo_baseline"]["log_loss"] == pytest.approx(direct["log_loss"])
    assert results["elo_baseline"]["accuracy"] == pytest.approx(direct["accuracy"])
    # and it is genuinely a different set of predictions than the model's:
    preds = pd.read_csv(tmp_path / "preds.csv")
    assert not np.allclose(preds["elo_pred_prob"], preds["logistic_pred_prob"])


# --------------------------------------------------------------------- 5
def test_probabilities_in_unit_interval(tmp_path):
    run_training(synthetic_features(), tmp_path, include_random_forest=True)
    preds = pd.read_csv(tmp_path / "preds.csv")
    for col in ("logistic_pred_prob", "random_forest_pred_prob"):
        assert preds[col].between(0.0, 1.0).all()


# --------------------------------------------------------------------- 6
def test_predictions_csv_columns(tmp_path):
    run_training(synthetic_features(), tmp_path, include_random_forest=True)
    preds = pd.read_csv(tmp_path / "preds.csv")
    expected = [
        "fight_id", "date", "event", "fighter_a", "fighter_b", "winner",
        TARGET, "elo_pred_prob", "logistic_pred_prob", "logistic_pred_class",
        "random_forest_pred_prob", "random_forest_pred_class",
    ]
    assert list(preds.columns) == expected
    assert set(preds["logistic_pred_class"]) <= {0, 1}


# --------------------------------------------------------------------- 7
def test_metrics_json_contents(tmp_path):
    run_training(synthetic_features(), tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    for model in ("elo_baseline", "logistic_regression"):
        assert model in metrics
        for key in ("accuracy", "log_loss", "roc_auc", "brier_score",
                    "avg_predicted_prob", "actual_win_rate", "calibration"):
            assert key in metrics[model]
    cal = metrics["logistic_regression"]["calibration"]
    assert len(cal) == 10  # fixed 0.0-0.1 ... 0.9-1.0 buckets
    assert sum(b["n_predictions"] for b in cal) == metrics["config"]["n_test"]


# --------------------------------------------------------------------- 8
def test_deterministic_predictions(tmp_path):
    df = synthetic_features()
    r1 = run_training(df, tmp_path / "run1", include_random_forest=True)
    r2 = run_training(df, tmp_path / "run2", include_random_forest=True)
    p1 = pd.read_csv(tmp_path / "run1" / "preds.csv")
    p2 = pd.read_csv(tmp_path / "run2" / "preds.csv")
    pd.testing.assert_frame_equal(p1, p2)
    assert r1["logistic_regression"]["log_loss"] == r2["logistic_regression"]["log_loss"]


# --------------------------------------------------------------------- 9
def test_missing_numeric_values_handled(tmp_path):
    df = synthetic_features()
    rng = np.random.default_rng(3)
    # punch holes in several numeric features, including whole-debut nulls
    for col in ("age_diff", "reach_diff", "days_since_last_fight_diff",
                "prior_win_pct_diff", "win_pct_last_3_diff"):
        df.loc[rng.random(len(df)) < 0.25, col] = np.nan
    results = run_training(df, tmp_path)  # must not crash
    preds = pd.read_csv(tmp_path / "preds.csv")
    assert preds["logistic_pred_prob"].notna().all()
    assert results["config"]["rows_dropped"] == 0  # imputed, not dropped


# --------------------------------------------------------------------- 10
def test_categorical_one_hot_safe(tmp_path):
    df = synthetic_features()
    # a category that appears ONLY in the test window (last 20%):
    df.loc[df.index[-10:], "weight_class"] = "Heavyweight"
    results = run_training(df, tmp_path)  # handle_unknown='ignore' -> no crash
    coefs = pd.read_csv(tmp_path / "coefs.csv")
    assert any(c.startswith("weight_class_") for c in coefs["feature"])
    # and --no-categorical really removes it
    results2 = run_training(df, tmp_path / "nocat", use_categorical=False)
    assert results2["config"]["features_categorical"] == []


# ------------------------------------------------------------ extras
def test_explicit_feature_list(tmp_path):
    results = run_training(
        synthetic_features(), tmp_path, features=["elo_diff", "age_diff"]
    )
    assert results["config"]["features_numeric"] == ["elo_diff", "age_diff"]
    coefs = pd.read_csv(tmp_path / "coefs.csv")
    assert set(coefs["feature"]) == {"elo_diff", "age_diff"}
    assert (coefs["absolute_coefficient"].diff().dropna() <= 1e-12).all()  # sorted desc


def test_missing_requested_features_skipped_not_fatal(tmp_path):
    df = synthetic_features().drop(columns=["reach_diff", "height_diff"])
    results = run_training(df, tmp_path)
    assert "reach_diff" in results["config"]["features_skipped"]
    assert "height_diff" in results["config"]["features_skipped"]
    assert "reach_diff" not in results["config"]["features_numeric"]
