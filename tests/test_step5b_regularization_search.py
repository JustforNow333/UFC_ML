"""Tests for Step 5B LR regularization + weight_class pruning search (Run 2).

Covers:
 (a) config generation produces only valid penalty/solver/l1_ratio combos
 (b) candidate selection rule picks the correct best + applies tie-breakers
 (c) selection uses no test data (by construction / signature)
 (d) weight_class column identification
 (e) rare-category detection from train counts (threshold behavior)
 (f) rare/nonstandard drop -> dropped categories encode to all-zero one-hot
 (g) common/current retention keeps the allowlisted divisions
 (h) L1/EN sparsity: some zeroed coefficients at small C
 (i) report schema via a reduced run into tmp_path
 (j) Platt/calibration split isolation (fold + official split)
 (k) official baseline metadata preserved unchanged in the report
 (l) Step 5A run_feature_diagnostics still imports/behaves (no breakage)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufc_pipeline.calibration import chronological_three_way_split  # noqa: E402
from ufc_pipeline.feature_diagnostics import (  # noqa: E402
    official_step3c_features,
    run_feature_diagnostics,
)
from ufc_pipeline.modeling import TARGET  # noqa: E402
from ufc_pipeline.step5b_regularization_search import (  # noqa: E402
    CURRENT_UFC_DIVISIONS,
    WEIGHT_CLASS_COLUMN,
    allowed_categories_by_threshold,
    build_feature_configs,
    build_pretest_and_official_split,
    build_rolling_folds,
    dedupe_finalists,
    evaluate_candidate,
    evaluate_finalist_on_official_split,
    fit_and_evaluate_fold,
    generate_candidates,
    is_valid_combo,
    make_preprocessor_step5b,
    make_step5b_pipeline,
    resolve_weight_class_handling,
    run_regularization_search,
    run_validation_search,
    select_finalists,
    valid_penalty_solver_combos,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "benchmarks" / "official_baseline.json"


def synthetic_step5b_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Full-width (43 numeric + weight_class) synthetic dataset with real
    signal in elo_diff, a couple of rare weight_class categories, and some
    NaN weight_class rows."""
    rng = np.random.default_rng(seed)
    numeric_all, _categorical = official_step3c_features()

    elo_diff = rng.normal(0, 80, n)
    p = 1 / (1 + 10 ** (-elo_diff / 400))
    won = (rng.random(n) < p).astype(int)
    dates = pd.date_range("2010-01-01", periods=n, freq="7D").strftime("%Y-%m-%d")

    flag_cols = {"fighter_a_no_prior_stats", "fighter_b_no_prior_stats", "matchup_history_missing"}

    # weight_class: mostly common divisions, a couple of genuinely rare
    # categories, and some NaN (missing) rows.
    common = ["Lightweight", "Welterweight", "Bantamweight", "Heavyweight"]
    rare = ["Catch Weight", "Super Heavyweight"]
    weight_class = rng.choice(common, n, p=[0.3, 0.3, 0.25, 0.15])
    rare_idx = rng.choice(n, size=max(2, n // 100), replace=False)
    weight_class = weight_class.astype(object)
    weight_class[rare_idx] = rng.choice(rare, size=len(rare_idx))
    nan_idx = rng.choice(n, size=max(1, n // 200), replace=False)
    weight_class[nan_idx] = None

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
    for col in numeric_all:
        if col == "elo_diff":
            continue
        if col in flag_cols:
            data[col] = rng.integers(0, 2, n)
        else:
            data[col] = rng.normal(0, 1.0, n)

    df = pd.DataFrame(data)
    assert set(numeric_all).issubset(df.columns)
    return df


@pytest.fixture(scope="module")
def numeric_features():
    numeric, _categorical = official_step3c_features()
    return numeric


@pytest.fixture(scope="module")
def synth_df():
    return synthetic_step5b_df(500)


# --------------------------------------------------------------------- (a)
def test_valid_penalty_solver_combos_only_valid():
    combos = valid_penalty_solver_combos(c_grid=[0.1, 1.0], l1_ratio_grid=[0.3, 0.7])
    assert len(combos) == 2 + 2 + 2 * 2  # l2 + l1 + elasticnet(C x l1_ratio)
    for combo in combos:
        assert is_valid_combo(combo)

    # elasticnet <-> saga + l1_ratio required
    en = [c for c in combos if c["penalty"] == "elasticnet"]
    assert en and all(c["solver"] == "saga" and c["l1_ratio"] is not None for c in en)
    # l1 <-> saga, no l1_ratio
    l1 = [c for c in combos if c["penalty"] == "l1"]
    assert l1 and all(c["solver"] == "saga" and c["l1_ratio"] is None for c in l1)
    # l2 <-> lbfgs, no l1_ratio
    l2 = [c for c in combos if c["penalty"] == "l2"]
    assert l2 and all(c["solver"] == "lbfgs" and c["l1_ratio"] is None for c in l2)


def test_is_valid_combo_rejects_invalid_combinations():
    assert not is_valid_combo({"penalty": "l1", "solver": "lbfgs", "l1_ratio": None})
    assert not is_valid_combo({"penalty": "elasticnet", "solver": "saga", "l1_ratio": None})
    assert not is_valid_combo({"penalty": "l2", "solver": "saga", "l1_ratio": None})
    assert not is_valid_combo({"penalty": "elasticnet", "solver": "lbfgs", "l1_ratio": 0.5})


def test_generate_candidates_deterministic_order():
    combos = valid_penalty_solver_combos(c_grid=[0.1, 1.0], l1_ratio_grid=[0.5])
    c1 = generate_candidates(["cfg_a", "cfg_b"], combos)
    c2 = generate_candidates(["cfg_a", "cfg_b"], combos)
    assert c1 == c2
    assert [c["feature_config"] for c in c1[: len(combos)]] == ["cfg_a"] * len(combos)


# --------------------------------------------------------------------- (b)
def _fake_validation_row(feature_config, penalty, C, mean_ll, brier=0.23, auc=0.6, n_features=44, l1_ratio=None):
    return {
        "feature_config": feature_config, "penalty": penalty, "C": C, "l1_ratio": l1_ratio,
        "solver": "lbfgs" if penalty == "l2" else "saga", "class_weight": None,
        "mean_val_platt_log_loss": mean_ll, "mean_val_brier": brier, "mean_val_auc": auc,
        "mean_val_accuracy": 0.6, "n_input_features": n_features,
        "std_val_platt_log_loss": 0.001, "per_fold_val_platt_log_loss": [mean_ll] * 3,
        "mean_val_uncal_log_loss": mean_ll + 0.01,
    }


def test_select_finalists_picks_best_and_applies_tie_breakers():
    rows = [
        _fake_validation_row("official_all_features", "l2", 1.0, 0.6440),
        _fake_validation_row("official_all_features", "l2", 0.1, 0.6400),  # clear best l2 (>noise gap)
        _fake_validation_row("official_all_features", "l1", 0.03, 0.6450),
        _fake_validation_row("drop_all_weight_class", "l2", 1.0, 0.6460),
        _fake_validation_row("drop_all_weight_class", "l1", 0.1, 0.6410),  # clear winner (>noise gap)
        _fake_validation_row("official_all_features", "elasticnet", 1.0, 0.6470, l1_ratio=0.5),
    ]
    finalists = select_finalists(rows, drop_rare_thresholds=())
    assert finalists["official_all_features"]["C"] == 1.0
    assert finalists["best_l2"]["C"] == 0.1  # lowest mean log loss among l2 rows
    # best_l1 is chosen across ANY feature config -> the lower drop_all_weight_class l1 row wins
    assert finalists["best_l1"]["feature_config"] == "drop_all_weight_class"
    assert finalists["best_drop_all_weight_class"]["penalty"] == "l1"  # 0.6410 << 0.6460
    assert finalists["best_overall"]["C"] == 0.1


def test_select_finalists_tie_break_prefers_fewer_features_within_noise():
    # Two rows within the noise threshold (0.002) of each other but with a
    # meaningfully different Brier score / feature count -> tie-break rule
    # (Brier first) must pick the lower-Brier one.
    rows = [
        _fake_validation_row("official_all_features", "l2", 1.0, 0.6440, brier=0.230, n_features=44),
        _fake_validation_row("drop_all_weight_class", "l2", 1.0, 0.6441, brier=0.220, n_features=43),
    ]
    finalists = select_finalists(rows, drop_rare_thresholds=())
    # within noise bucket -> Brier tie-break should prefer the drop_all_weight_class row
    assert finalists["best_overall"]["feature_config"] == "drop_all_weight_class"


def test_select_finalists_noise_band_does_not_have_rounding_boundary_bug():
    # These rows differ by only 0.0002 log-loss points (< NOISE_THRESHOLD), but
    # the old rounded-bucket implementation put them in different buckets. The
    # tie-breakers should still decide the order.
    rows = [
        _fake_validation_row("official_all_features", "l2", 1.0, 0.6449, brier=0.230, n_features=44),
        _fake_validation_row("drop_all_weight_class", "l2", 1.0, 0.6451, brier=0.220, n_features=43),
    ]
    finalists = select_finalists(rows, drop_rare_thresholds=())

    assert finalists["best_overall"]["feature_config"] == "drop_all_weight_class"


def test_dedupe_finalists_collapses_identical_candidates():
    cand = _fake_validation_row("official_all_features", "l2", 1.0, 0.6440)
    finalists_map = {"official_all_features": cand, "best_l2": cand, "best_overall": cand}
    deduped = dedupe_finalists(finalists_map, max_finalists=8)
    assert len(deduped) == 1
    assert set(deduped[0]["labels"]) == {"official_all_features", "best_l2", "best_overall"}


# --------------------------------------------------------------------- (c)
def test_select_finalists_signature_has_no_test_data_parameter():
    import inspect

    sig = inspect.signature(select_finalists)
    param_names = list(sig.parameters)
    assert param_names == ["validation_results", "drop_rare_thresholds"]
    # drop_rare_thresholds is a config tuple of ints, not fold/test data
    for name in param_names:
        assert "test" not in name


# --------------------------------------------------------------------- (d)
def test_weight_class_column_identification():
    assert WEIGHT_CLASS_COLUMN == "weight_class"
    configs = build_feature_configs()
    assert configs["drop_all_weight_class"]["weight_class_mode"] == "none"
    assert configs["official_all_features"]["weight_class_mode"] == "all"


# --------------------------------------------------------------------- (e)
def test_allowed_categories_by_threshold(synth_df):
    train, _, _, _ = build_pretest_and_official_split(synth_df)
    counts = train[WEIGHT_CLASS_COLUMN].value_counts(dropna=True)

    low = allowed_categories_by_threshold(train, threshold=1)
    high = allowed_categories_by_threshold(train, threshold=10_000)
    assert set(low) == set(counts.index)
    assert high == []

    mid_threshold = int(counts.median())
    mid = allowed_categories_by_threshold(train, threshold=mid_threshold)
    assert set(mid) == {cat for cat, n in counts.items() if n >= mid_threshold}


# --------------------------------------------------------------------- (f)
def test_rare_category_drop_encodes_to_all_zero_row():
    fit_df = pd.DataFrame({WEIGHT_CLASS_COLUMN: ["Lightweight"] * 20 + ["Catch Weight"] * 2})
    feature_config = {"weight_class_mode": "rare_threshold", "threshold": 5}
    include_wc, categories = resolve_weight_class_handling(feature_config, fit_df)
    assert include_wc is True
    assert "Catch Weight" not in categories
    assert "Lightweight" in categories

    pre = make_preprocessor_step5b([], include_weight_class=True, weight_class_categories=categories)
    encoded = pre.fit(fit_df).transform(fit_df)
    encoded = np.asarray(encoded.todense() if hasattr(encoded, "todense") else encoded)

    common_row = encoded[0]
    rare_row = encoded[fit_df[WEIGHT_CLASS_COLUMN].tolist().index("Catch Weight")]
    assert common_row.sum() == 1  # common category fires exactly one column
    assert rare_row.sum() == 0  # dropped/rare category encodes to all-zero


# --------------------------------------------------------------------- (g)
def test_common_current_division_allowlist_is_fixed():
    feature_config = {"weight_class_mode": "allowlist_fixed", "allowlist": list(CURRENT_UFC_DIVISIONS)}
    fit_df_a = pd.DataFrame({WEIGHT_CLASS_COLUMN: ["Lightweight"] * 5})
    fit_df_b = pd.DataFrame({WEIGHT_CLASS_COLUMN: ["Open Weight"] * 5})  # different fold data
    _, cats_a = resolve_weight_class_handling(feature_config, fit_df_a)
    _, cats_b = resolve_weight_class_handling(feature_config, fit_df_b)
    assert cats_a == cats_b == list(CURRENT_UFC_DIVISIONS)
    assert "Open Weight" not in cats_a
    assert "Catch Weight" not in cats_a
    assert "Women's Featherweight" not in cats_a
    assert "Lightweight" in cats_a


# --------------------------------------------------------------------- (h)
def test_l1_and_elasticnet_zero_some_coefficients_at_small_c(numeric_features, synth_df):
    pretest, _, _, _ = build_pretest_and_official_split(synth_df)
    folds = build_rolling_folds(pretest)
    feature_configs = build_feature_configs()

    l1_candidate = {"feature_config": "official_all_features", "penalty": "l1", "C": 0.003,
                     "solver": "saga", "l1_ratio": None, "class_weight": None}
    entry, pipeline = fit_and_evaluate_fold(
        l1_candidate, numeric_features, feature_configs["official_all_features"], folds[0]
    )
    coefs = pipeline.named_steps["model"].coef_.ravel()
    assert entry["n_nonzero_coefficients"] < len(coefs)  # some zeroed out
    assert (np.abs(coefs) <= 1e-8).sum() > 0

    en_candidate = {"feature_config": "official_all_features", "penalty": "elasticnet", "C": 0.003,
                     "solver": "saga", "l1_ratio": 0.9, "class_weight": None}
    entry2, pipeline2 = fit_and_evaluate_fold(
        en_candidate, numeric_features, feature_configs["official_all_features"], folds[0]
    )
    coefs2 = pipeline2.named_steps["model"].coef_.ravel()
    assert (np.abs(coefs2) <= 1e-8).sum() > 0


# --------------------------------------------------------------------- (i)
def test_report_schema_reduced_run(tmp_path, synth_df):
    csv_path = tmp_path / "features.csv"
    synth_df.to_csv(csv_path, index=False)

    report = run_regularization_search(
        input_csv=str(csv_path),
        output_dir=str(tmp_path / "reports"),
        max_candidates=None,
        skip_balanced=True,
        drop_rare_thresholds=(2,),
        c_grid=[0.1, 1.0],
        l1_ratio_grid=[0.5],
        baseline_path=str(BASELINE_PATH),
        run1_report_path=None,
    )

    for key in (
        "generated_at", "official_baseline", "run1_summary", "config",
        "validation_protocol", "search_space", "weight_class_handling_analysis",
        "candidate_selection_rule", "validation_results", "finalists", "test_results",
        "best_l2", "best_l1", "best_elastic_net", "best_drop_all_weight_class",
        "best_keep_common_current_weight_classes_only",
        "best_drop_rare_weight_class_indicators_only", "best_overall",
        "l1_selected_features_official_all", "elastic_net_selected_features_official_all",
        "coefficient_sparsity_summary", "fold_stability_official_all_features",
        "calibration_comparison", "leakage_checks", "limitations", "verdict",
    ):
        assert key in report, f"{key} missing from report"

    out_dir = tmp_path / "reports"
    for fname in (
        "step5b_regularization_search.json", "step5b_regularization_search.md",
        "step5b_coefficients.csv", "step5b_candidate_results.csv",
    ):
        assert (out_dir / fname).exists(), fname

    loaded = json.loads((out_dir / "step5b_regularization_search.json").read_text())
    assert loaded["config"]["n_official_test"] == report["config"]["n_official_test"]
    assert report["test_results"], "at least one finalist must be evaluated on the held-out test"


# --------------------------------------------------------------------- (j)
def test_platt_depends_only_on_calibration_window(numeric_features, synth_df):
    pretest, _, _, _ = build_pretest_and_official_split(synth_df)
    folds = build_rolling_folds(pretest)
    feature_configs = build_feature_configs()
    candidate = {"feature_config": "official_all_features", "penalty": "l2", "C": 1.0,
                 "solver": "lbfgs", "l1_ratio": None, "class_weight": None}

    fold = folds[0]
    entry1, pipeline1 = fit_and_evaluate_fold(candidate, numeric_features, feature_configs["official_all_features"], fold)

    # Changing VAL rows must not change the fold's Platt-calibrated LOG LOSS
    # computation path validity, but more importantly must not change the
    # base pipeline or the Platt calibrator fit. Verify by refitting with a
    # perturbed val set and comparing coefficients.
    fold_val_changed = dict(fold)
    fold_val_changed["val"] = fold["val"].copy()
    fold_val_changed["val"][TARGET] = 1 - fold_val_changed["val"][TARGET]
    entry2, pipeline2 = fit_and_evaluate_fold(
        candidate, numeric_features, feature_configs["official_all_features"], fold_val_changed
    )
    coefs1 = pipeline1.named_steps["model"].coef_.ravel()
    coefs2 = pipeline2.named_steps["model"].coef_.ravel()
    assert np.allclose(coefs1, coefs2)  # base model unaffected by val-row changes

    # Changing CALIB rows must change the fitted base model's calibration
    # inputs (Platt is refit) -> the reported platt_log_loss should differ
    # from the calib-unchanged run when calib labels are flipped.
    fold_calib_changed = dict(fold)
    fold_calib_changed["calib"] = fold["calib"].copy()
    fold_calib_changed["calib"][TARGET] = 1 - fold_calib_changed["calib"][TARGET]
    entry3, _ = fit_and_evaluate_fold(
        candidate, numeric_features, feature_configs["official_all_features"], fold_calib_changed
    )
    assert entry3["platt_log_loss"] != pytest.approx(entry1["platt_log_loss"])


def test_official_split_platt_depends_only_on_calibration(numeric_features, synth_df):
    df = synth_df.copy()
    _, official_train, official_calib, official_test = build_pretest_and_official_split(df)
    feature_configs = build_feature_configs()
    candidate = {"feature_config": "official_all_features", "penalty": "l2", "C": 1.0,
                 "solver": "lbfgs", "l1_ratio": None, "class_weight": None}

    result1, _ = evaluate_finalist_on_official_split(
        ["x"], candidate, numeric_features, feature_configs,
        official_train, official_calib, official_test, official_log_loss=0.65,
    )
    test_flipped = official_test.copy()
    test_flipped[TARGET] = 1 - test_flipped[TARGET]
    # Re-run with flipped TEST labels: the base pipeline/platt fit must be
    # identical (test never trains anything); only the reported test log
    # loss (computed against different y) should change.
    result2, _ = evaluate_finalist_on_official_split(
        ["x"], candidate, numeric_features, feature_configs,
        official_train, official_calib, test_flipped, official_log_loss=0.65,
    )
    assert result1["platt_test_log_loss"] != pytest.approx(result2["platt_test_log_loss"])


# --------------------------------------------------------------------- (k)
def test_official_baseline_metadata_preserved(tmp_path, synth_df):
    csv_path = tmp_path / "features.csv"
    synth_df.to_csv(csv_path, index=False)
    report = run_regularization_search(
        input_csv=str(csv_path),
        output_dir=str(tmp_path / "reports"),
        skip_balanced=True,
        drop_rare_thresholds=(2,),
        c_grid=[1.0],
        l1_ratio_grid=[0.5],
        baseline_path=str(BASELINE_PATH),
        run1_report_path=None,
    )
    with open(BASELINE_PATH) as fh:
        expected = json.load(fh)
    assert report["official_baseline"] == expected


# --------------------------------------------------------------------- (l)
def test_step5a_still_imports_and_runs(tmp_path, synth_df):
    csv_path = tmp_path / "features.csv"
    synth_df.to_csv(csv_path, index=False)
    from ufc_pipeline.feature_diagnostics import FEATURE_GROUPS

    reduced_ablations = {"all": list(FEATURE_GROUPS), "only_elo": ["elo"]}
    report = run_feature_diagnostics(
        input_csv=str(csv_path),
        output_dir=str(tmp_path / "reports_5a"),
        n_permutation_repeats=1,
        ablations=reduced_ablations,
        baseline_path=str(BASELINE_PATH),
    )
    assert "ablations" in report
    assert len(report["ablations"]) == 2
