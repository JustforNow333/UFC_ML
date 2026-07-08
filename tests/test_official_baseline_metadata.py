"""Tests for the promoted official benchmark metadata."""

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "benchmarks" / "official_baseline.json"


def _load_baseline() -> dict:
    with open(BASELINE_PATH) as fh:
        return json.load(fh)


def test_official_baseline_promotes_step5c_stronger_regularized_lr():
    baseline = _load_baseline()
    official = baseline["official_model"]

    assert official["name"] == "step5c_stronger_regularized_lr_drop_weight_class_platt"
    assert official["model_family"] == "logistic_regression"
    assert official["calibration_method"] == "platt"
    assert official["hyperparameters"]["penalty"] == "elasticnet"
    assert official["hyperparameters"]["C"] == pytest.approx(0.003)
    assert official["hyperparameters"]["l1_ratio"] == pytest.approx(0.1)
    assert official["feature_config"]["raw_weight_class"] == "dropped"
    assert official["feature_config"]["division_interactions"] == "not_included"

    metrics = official["test_metrics"]
    assert metrics["log_loss"] == pytest.approx(0.6419199384228512)
    assert metrics["brier_score"] == pytest.approx(0.22559535813023482)
    assert metrics["roc_auc"] == pytest.approx(0.6791349239158251)
    assert metrics["accuracy"] == pytest.approx(0.6287051482059283)

    evidence = official["validation_evidence"]
    assert evidence["old_official_mean_platt_log_loss"] == pytest.approx(0.6673284938546409)
    assert evidence["selected_candidate_mean_platt_log_loss"] == pytest.approx(0.6584660204680894)
    assert evidence["folds_won_vs_old_official"] == "5/5"

    assert baseline["promotion"]["division_interactions_promoted"] is False
    assert baseline["promotion"]["raw_weight_class_promoted"] is False
    assert "final held-out test was already touched" in official["test_protocol_note"]


def test_old_step3c_lr_platt_baseline_is_preserved_as_historical_reference():
    baseline = _load_baseline()
    old = baseline["historical_baselines"]["old_official_step3c_lr_platt"]
    old_ref = baseline["reference_points"]["old_official_step3c_lr_platt"]

    assert old["label"] == "Step 3C LR + Platt"
    assert old["test_metrics"] == old_ref["test_metrics"]
    assert old["test_metrics"]["log_loss"] == pytest.approx(0.6442244532062779)
    assert old["test_metrics"]["brier_score"] == pytest.approx(0.22671650863397674)
    assert old["test_metrics"]["roc_auc"] == pytest.approx(0.6759680308667494)
    assert old["test_metrics"]["accuracy"] == pytest.approx(0.6193447737909517)
    assert old["test_metrics"]["log_loss"] > baseline["official_model"]["test_metrics"]["log_loss"]
