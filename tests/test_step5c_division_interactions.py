"""Tests for Step 5C division interaction experiment."""

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
from ufc_pipeline.step5c_division_interactions import (  # noqa: E402
    PREDECLARED_CANDIDATE_NAMES,
    add_division_interactions,
    analyze_division_eligibility,
    build_predeclared_step5c_candidates,
    interaction_column_name,
    run_division_interactions,
    select_interaction_base_features,
    select_step5c_finalists,
    separate_feature_columns,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "benchmarks" / "official_baseline.json"


def synthetic_step5c_df(n: int = 420, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    numeric_all, _categorical = official_step3c_features()

    elo_diff = rng.normal(0, 80, n)
    age_diff = rng.normal(0, 5, n)
    p = 1 / (1 + np.exp(-(0.008 * elo_diff - 0.06 * age_diff)))
    won = (rng.random(n) < p).astype(int)
    dates = pd.date_range("2010-01-01", periods=n, freq="7D").strftime("%Y-%m-%d")
    divisions = rng.choice(
        ["Lightweight", "Welterweight", "Bantamweight", "Open Weight", "Catch Weight"],
        n,
        p=[0.34, 0.30, 0.28, 0.03, 0.05],
    ).astype(object)

    data = {
        "fight_id": np.arange(1, n + 1),
        "date": dates,
        "event": [f"UFC {i}" for i in range(n)],
        "fighter_a": [f"A{i}" for i in range(n)],
        "fighter_b": [f"B{i}" for i in range(n)],
        "winner": np.where(won == 1, [f"A{i}" for i in range(n)], [f"B{i}" for i in range(n)]),
        TARGET: won,
        WEIGHT_CLASS_COLUMN: divisions,
        "fighter_a_expected_win_prob": p,
        "elo_diff": elo_diff,
        "age_diff": age_diff,
    }
    flag_cols = {"fighter_a_no_prior_stats", "fighter_b_no_prior_stats", "matchup_history_missing"}
    for col in numeric_all:
        if col in data:
            continue
        if col in flag_cols:
            data[col] = rng.integers(0, 2, n)
        else:
            data[col] = rng.normal(0, 1.0, n)
    return pd.DataFrame(data)


def test_eligible_division_selection_uses_pretest_not_full_data():
    pretest = pd.DataFrame(
        {
            WEIGHT_CLASS_COLUMN: (
                ["Lightweight"] * 30
                + ["Welterweight"] * 25
                + ["Women's Flyweight"] * 3
                + ["Open Weight"] * 50
                + ["Catch Weight"] * 50
                + ["Super Heavyweight"] * 50
            )
        }
    )
    full = pd.concat(
        [
            pretest,
            pd.DataFrame({WEIGHT_CLASS_COLUMN: ["Women's Flyweight"] * 100}),
        ],
        ignore_index=True,
    )

    analysis = analyze_division_eligibility(pretest, full, min_count=10, min_fraction=0.0)

    assert analysis["eligible_divisions"] == ["Lightweight", "Welterweight"]
    excluded = {row["category"]: row["excluded_reasons"] for row in analysis["excluded_divisions"]}
    assert "below_pretest_sample_threshold" in excluded["Women's Flyweight"]
    assert "nonstandard_or_historical" in excluded["Open Weight"]
    assert "nonstandard_or_historical" in excluded["Catch Weight"]
    assert "nonstandard_or_historical" in excluded["Super Heavyweight"]


def test_interaction_generation_creates_expected_columns_and_zeroes_nonmatching_rows():
    df = pd.DataFrame(
        {
            WEIGHT_CLASS_COLUMN: ["Lightweight", "Welterweight", "Lightweight"],
            "age_diff": [3.0, np.nan, np.nan],
            "elo_diff": [10.0, 20.0, 30.0],
        }
    )
    out, columns, metadata = add_division_interactions(df, ["age_diff", "elo_diff"], ["Lightweight"])

    age_col = interaction_column_name("age_diff", "Lightweight")
    elo_col = interaction_column_name("elo_diff", "Lightweight")
    assert columns == [age_col, elo_col]
    assert metadata[age_col]["base_feature"] == "age_diff"
    assert out.loc[0, age_col] == 3.0
    assert out.loc[1, age_col] == 0.0  # nonmatching division, even though base is NaN
    assert pd.isna(out.loc[2, age_col])  # matching division keeps missing base value for train-only imputer
    assert out.loc[2, elo_col] == 30.0


def test_missing_base_features_are_skipped_safely():
    selected, skipped = select_interaction_base_features(["age_diff"], "domain")
    assert selected == ["age_diff"]
    assert "reach_diff" in skipped
    assert "takedown_defense_diff" in skipped


def test_raw_weight_class_columns_are_separated_from_interactions():
    cols = [
        "age_diff",
        "weight_class_Lightweight",
        "age_diff_x_weight_class_Lightweight",
    ]
    separated = separate_feature_columns(cols)
    assert separated["base_numeric"] == ["age_diff"]
    assert separated["raw_weight_class"] == ["weight_class_Lightweight"]
    assert separated["interactions"] == ["age_diff_x_weight_class_Lightweight"]


def test_step5c_candidates_are_predeclared_and_lr_only():
    candidates = build_predeclared_step5c_candidates()
    assert [c["name"] for c in candidates] == list(PREDECLARED_CANDIDATE_NAMES)
    assert all(c["model_family"] == "logistic_regression" for c in candidates)
    assert any(c["interaction_feature_set"] == "priority" for c in candidates)
    assert any(c["include_raw_weight_class"] for c in candidates)
    assert any(not c["include_raw_weight_class"] for c in candidates)


def test_step5c_finalist_selection_has_no_test_data_parameter():
    sig = inspect.signature(select_step5c_finalists)
    assert list(sig.parameters) == ["validation_results"]
    assert all("test" not in name for name in sig.parameters)


def test_step5c_report_schema_and_baseline_preserved(tmp_path):
    csv_path = tmp_path / "features.csv"
    synthetic_step5c_df().to_csv(csv_path, index=False)

    report = run_division_interactions(
        input_csv=str(csv_path),
        output_dir=str(tmp_path / "reports"),
        baseline_path=str(BASELINE_PATH),
        run1_report_path=None,
        run2_report_path=None,
        run21_report_path=None,
        min_division_count=20,
        min_division_fraction=0.0,
        max_candidates=4,
        max_iter=1000,
    )

    for key in (
        "generated_at",
        "official_baseline",
        "run1_summary",
        "run2_summary",
        "run21_summary",
        "validation_protocol",
        "eligible_division_analysis",
        "base_interaction_features",
        "candidate_configurations",
        "validation_results",
        "fold_by_fold_results",
        "final_test_results",
        "coefficient_sparsity_summary",
        "interaction_stability_summary",
        "raw_weight_class_verdict",
        "interaction_verdict",
        "leakage_protocol_checks",
        "recommendation",
    ):
        assert key in report, f"{key} missing from Step 5C report"

    assert report["lr_only"] is True
    assert report["changes_official_baseline_file"] is False
    assert report["config"]["candidate_count_is_controlled"] is True
    assert "Lightweight" in report["eligible_division_analysis"]["eligible_divisions"]
    assert report["base_interaction_features"]["by_candidate"]["interactions_no_raw_weight_class"][
        "n_interaction_features_created"
    ] > 0
    assert report["validation_protocol"]["selection_rule"]["final_test_used_for_selection"] is False

    for check in (
        "eligible_divisions_selected_from_pretest_only",
        "interaction_features_deterministic",
        "raw_weight_class_and_interactions_separated",
        "no_nonlinear_models_or_ensembles",
    ):
        assert check in report["leakage_protocol_checks"]

    with open(BASELINE_PATH) as fh:
        assert report["official_baseline"] == json.load(fh)

    out_dir = tmp_path / "reports"
    assert (out_dir / "step5c_division_interactions.json").exists()
    assert (out_dir / "step5c_division_interactions.md").exists()
    assert (out_dir / "step5c_candidate_results.csv").exists()
    assert (out_dir / "step5c_interaction_coefficients.csv").exists()
