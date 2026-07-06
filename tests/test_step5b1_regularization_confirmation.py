"""Tests for Step 5B.1 regularization confirmation.

The confirmation pass is intentionally narrower than Step 5B: it evaluates a
fixed candidate set on extra pre-test folds, then re-reports the already
touched final test for those fixed candidates only.
"""

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufc_pipeline.feature_diagnostics import official_step3c_features  # noqa: E402
from ufc_pipeline.modeling import TARGET  # noqa: E402
from ufc_pipeline.step5b_regularization_search import WEIGHT_CLASS_COLUMN  # noqa: E402
from ufc_pipeline.step5b1_regularization_confirmation import (  # noqa: E402
    PREDECLARED_CANDIDATE_NAMES,
    build_predeclared_confirmation_candidates,
    run_regularization_confirmation,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "benchmarks" / "official_baseline.json"


def synthetic_confirmation_df(n: int = 360, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    numeric_all, _categorical = official_step3c_features()

    won = ((np.arange(n) % 3) != 0).astype(int)
    elo_diff = np.where(won == 1, 60.0, -60.0) + rng.normal(0, 25, n)
    p = 1 / (1 + 10 ** (-elo_diff / 400))
    dates = pd.date_range("2010-01-01", periods=n, freq="7D").strftime("%Y-%m-%d")
    weight_class = rng.choice(
        ["Lightweight", "Welterweight", "Bantamweight", "Catch Weight"],
        n,
        p=[0.35, 0.35, 0.25, 0.05],
    ).astype(object)

    data = {
        "fight_id": np.arange(1, n + 1),
        "date": dates,
        "event": [f"UFC {i}" for i in range(n)],
        "fighter_a": [f"A{i}" for i in range(n)],
        "fighter_b": [f"B{i}" for i in range(n)],
        "winner": np.where(won == 1, [f"A{i}" for i in range(n)], [f"B{i}" for i in range(n)]),
        TARGET: won,
        WEIGHT_CLASS_COLUMN: weight_class,
        "fighter_a_expected_win_prob": p,
        "elo_diff": elo_diff,
    }
    flag_cols = {"fighter_a_no_prior_stats", "fighter_b_no_prior_stats", "matchup_history_missing"}
    for col in numeric_all:
        if col == "elo_diff":
            continue
        if col in flag_cols:
            data[col] = rng.integers(0, 2, n)
        else:
            data[col] = rng.normal(0, 1.0, n)

    return pd.DataFrame(data)


def _fake_step5b_row(feature_config, penalty, C, mean_ll, l1_ratio=None):
    return {
        "feature_config": feature_config,
        "class_weight": None,
        "penalty": penalty,
        "C": C,
        "solver": "lbfgs" if penalty == "l2" else "saga",
        "l1_ratio": l1_ratio,
        "mean_val_platt_log_loss": mean_ll,
        "std_val_platt_log_loss": 0.001,
        "per_fold_val_platt_log_loss": [mean_ll] * 3,
        "mean_val_brier": 0.23,
        "mean_val_auc": 0.62,
        "mean_val_accuracy": 0.60,
        "n_input_features": 44,
    }


def test_confirmation_candidates_are_predeclared_and_ignore_final_test_results():
    report = {
        "validation_results": [
            _fake_step5b_row("official_all_features", "elasticnet", 0.003, 0.660, l1_ratio=0.1),
            _fake_step5b_row("official_all_features", "elasticnet", 0.01, 0.650, l1_ratio=0.5),
        ],
        "best_l1": _fake_step5b_row("drop_all_weight_class", "l1", 0.03, 0.651),
        "best_l2": _fake_step5b_row("drop_all_weight_class", "l2", 0.003, 0.652),
        "test_results": [{"candidate": {"penalty": "l2", "C": 10.0}, "platt_test_log_loss": 0.1}],
    }
    changed_test_report = {**report, "test_results": [{"candidate": {"penalty": "l1", "C": 999.0}}]}

    candidates = build_predeclared_confirmation_candidates(report)
    candidates_changed = build_predeclared_confirmation_candidates(changed_test_report)

    assert [c["name"] for c in candidates] == list(PREDECLARED_CANDIDATE_NAMES)
    assert [c["candidate"] for c in candidates] == [c["candidate"] for c in candidates_changed]
    assert candidates[2]["candidate"]["feature_config"] == "official_all_features"
    assert candidates[2]["candidate"]["penalty"] == "elasticnet"
    assert candidates[2]["candidate"]["C"] == 0.01
    assert candidates[3]["candidate"]["penalty"] == "l1"


def test_confirmation_run_has_no_broad_search_interface():
    sig = inspect.signature(run_regularization_confirmation)
    params = set(sig.parameters)
    assert "c_grid" not in params
    assert "l1_ratio_grid" not in params
    assert "max_candidates" not in params
    assert len(PREDECLARED_CANDIDATE_NAMES) == 4


def test_confirmation_report_schema_and_baseline_metadata_preserved(tmp_path):
    csv_path = tmp_path / "features.csv"
    synthetic_confirmation_df().to_csv(csv_path, index=False)

    report = run_regularization_confirmation(
        input_csv=str(csv_path),
        output_dir=str(tmp_path / "reports"),
        baseline_path=str(BASELINE_PATH),
        step5b_report_path=None,
        max_iter=1000,
    )

    for key in (
        "generated_at",
        "run",
        "official_baseline",
        "step5b_summary",
        "config",
        "validation_protocol",
        "predeclared_candidates",
        "validation_stability",
        "fold_by_fold_comparison",
        "final_test_metrics_re_reported",
        "coefficient_sparsity_stability",
        "weight_class_verdict",
        "leakage_protocol_checks",
        "promotion_criteria",
        "recommendation",
    ):
        assert key in report, f"{key} missing from Step 5B.1 report"

    assert report["config"]["does_broad_search"] is False
    assert report["config"]["n_predeclared_candidates"] == 4
    assert list(report["validation_stability"]) == list(PREDECLARED_CANDIDATE_NAMES)
    assert all(
        "final test window was already touched"
        in result["final_test_protocol_note"]
        for result in report["final_test_metrics_re_reported"].values()
    )

    with open(BASELINE_PATH) as fh:
        expected_baseline = json.load(fh)
    assert report["official_baseline"] == expected_baseline

    out_dir = tmp_path / "reports"
    assert (out_dir / "step5b1_regularization_confirmation.json").exists()
    assert (out_dir / "step5b1_regularization_confirmation.md").exists()
