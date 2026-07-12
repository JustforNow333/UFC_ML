"""Leakage, edge-case, swapping, and schema tests for layoff features."""

from __future__ import annotations

import pandas as pd
import pytest

from ufc_pipeline.feature_diagnostics import official_step3c_features
from ufc_pipeline.features import OUTPUT_COLUMNS, build_feature_rows
from ufc_pipeline.layoff_features import (
    LAYOFF_CANDIDATE_A_FEATURES,
    LAYOFF_CANDIDATE_B_FEATURES,
    LAYOFF_FEATURE_COLUMNS,
    LayoffHistory,
    build_layoff_feature_family,
    counts_as_completed_activity,
)
from ufc_pipeline.modeling import check_features_allowed
from ufc_pipeline.step6c_upcoming_feature_builder import model_features_for_set


def fight(fight_id, date, a_id, b_id, *, a_won=1, **extra):
    row = {
        "fight_id": fight_id, "date": date, "event": f"E{fight_id}",
        "fighter_a_id": a_id, "fighter_b_id": b_id,
        "fighter_a": f"F{a_id}", "fighter_b": f"F{b_id}",
        "winner": f"F{a_id}" if a_won else f"F{b_id}", "fighter_a_won": a_won,
        "fighter_a_pre_elo": 1500.0, "fighter_b_pre_elo": 1500.0,
        "fighter_a_expected_win_prob": 0.5, "fighter_b_expected_win_prob": 0.5,
    }
    row.update(extra)
    return row


def test_normal_fighters_use_most_recent_strictly_prior_fight():
    rows = build_feature_rows([
        fight(1, "2020-01-01", 1, 3),
        fight(2, "2020-02-01", 2, 4),
        fight(3, "2020-03-01", 1, 2),
    ])
    current = rows.loc[rows["fight_id"] == 3].iloc[0]
    assert current["fighter_a_layoff_days"] == 60
    assert current["fighter_b_layoff_days"] == 29
    assert current["layoff_days_diff"] == 31
    assert current["abs_layoff_days_diff"] == 31
    assert current["fighter_a_layoff_missing"] == current["fighter_b_layoff_missing"] == 0


def test_several_prior_fights_use_latest_date_not_first_date():
    rows = build_feature_rows([
        fight(1, "2020-01-01", 1, 3),
        fight(2, "2020-02-01", 1, 4),
        fight(3, "2020-04-01", 1, 5),
    ])
    assert rows.loc[2, "fighter_a_layoff_days"] == 60


def test_debutant_is_missing_not_zero():
    first = build_feature_rows([fight(1, "2020-01-01", 1, 2)]).iloc[0]
    assert pd.isna(first["fighter_a_layoff_days"])
    assert pd.isna(first["fighter_a_log_layoff_days"])
    assert pd.isna(first["fighter_a_short_turnaround"])
    assert first["fighter_a_layoff_missing"] == 1


def test_short_long_and_extreme_thresholds_and_log():
    short = LayoffHistory(); short.record_completed_fight("2020-01-01")
    short_features = build_layoff_feature_family(short, LayoffHistory(), "2020-02-15")
    assert short_features["fighter_a_layoff_days"] == 45
    assert short_features["fighter_a_short_turnaround"] == 1
    assert short_features["fighter_a_long_layoff"] == 0

    long = LayoffHistory(); long.record_completed_fight("2020-01-01")
    long_features = build_layoff_feature_family(long, LayoffHistory(), "2022-01-01")
    assert long_features["fighter_a_layoff_days"] == 731
    assert long_features["fighter_a_long_layoff"] == 1
    assert long_features["fighter_a_extreme_layoff"] == 1
    assert long_features["fighter_a_log_layoff_days"] == pytest.approx(6.596, abs=0.001)


def test_current_and_future_fights_cannot_affect_older_rows():
    base = [
        fight(1, "2020-01-01", 1, 2),
        fight(2, "2020-03-01", 1, 3),
    ]
    old = build_feature_rows(base).set_index("fight_id")
    with_future = build_feature_rows(base + [fight(3, "2030-01-01", 1, 4)]).set_index("fight_id")
    pd.testing.assert_frame_equal(
        old[LAYOFF_FEATURE_COLUMNS], with_future.loc[old.index, LAYOFF_FEATURE_COLUMNS],
        check_dtype=False,
    )
    assert pd.isna(old.loc[1, "fighter_a_layoff_days"])
    assert old.loc[2, "fighter_a_layoff_days"] == 60


def test_duplicate_bout_date_does_not_advance_activity_twice():
    rows = build_feature_rows([
        fight(1, "2020-01-01", 1, 2),
        fight(1, "2020-01-01", 1, 2),
        fight(2, "2020-03-01", 1, 3),
    ])
    assert rows.iloc[-1]["fighter_a_layoff_days"] == 60


def test_cancelled_or_scheduled_fight_is_not_activity():
    rows = build_feature_rows([
        fight(1, "2020-01-01", 1, 2),
        fight(2, "2020-02-01", 1, 3, fight_status="cancelled"),
        fight(3, "2020-03-01", 1, 4),
    ])
    assert rows.iloc[-1]["fighter_a_layoff_days"] == 60
    assert not counts_as_completed_activity({"date": "2020-02-01", "status": "scheduled"})


def test_same_day_fights_do_not_create_an_earlier_fight_order():
    rows = build_feature_rows([
        fight(1, "2019-12-01", 1, 9),
        fight(2, "2020-01-01", 1, 2),
        fight(3, "2020-01-01", 1, 3),
    ])
    assert rows.loc[1, "fighter_a_layoff_days"] == 31
    assert rows.loc[2, "fighter_a_layoff_days"] == 31


def test_missing_or_malformed_date_is_safe_in_shared_service():
    history = LayoffHistory()
    assert not history.record_completed_fight(None)
    assert not history.record_completed_fight("bad-date")
    assert history.layoff_days("also-bad") is None
    features = build_layoff_feature_family(history, LayoffHistory(), None)
    assert features["fighter_a_layoff_missing"] == 1
    assert features["fighter_b_layoff_missing"] == 1


def test_side_swap_reverses_signed_but_not_absolute_difference():
    a, b = LayoffHistory(), LayoffHistory()
    a.record_completed_fight("2020-01-01")
    b.record_completed_fight("2020-02-01")
    ab = build_layoff_feature_family(a, b, "2020-04-01")
    ba = build_layoff_feature_family(b, a, "2020-04-01")
    assert ab["fighter_a_layoff_days"] == ba["fighter_b_layoff_days"]
    assert ab["fighter_b_layoff_days"] == ba["fighter_a_layoff_days"]
    assert ab["layoff_days_diff"] == -ba["layoff_days_diff"]
    assert ab["abs_layoff_days_diff"] == ba["abs_layoff_days_diff"]
    for suffix in ("short_turnaround", "long_layoff", "extreme_layoff", "layoff_missing"):
        assert ab[f"fighter_a_{suffix}"] == ba[f"fighter_b_{suffix}"]


def test_output_is_deterministic_and_columns_append_without_reordering_baseline():
    fights = [fight(2, "2020-03-01", 1, 3), fight(1, "2020-01-01", 1, 2)]
    first, second = build_feature_rows(fights), build_feature_rows(list(reversed(fights)))
    pd.testing.assert_frame_equal(first, second)
    assert OUTPUT_COLUMNS[-len(LAYOFF_FEATURE_COLUMNS):] == LAYOFF_FEATURE_COLUMNS
    official, _ = official_step3c_features()
    assert len(official) == 43
    assert model_features_for_set("official") == official
    assert model_features_for_set("layoff_a") == official + LAYOFF_CANDIDATE_A_FEATURES
    assert model_features_for_set("layoff_b") == official + LAYOFF_CANDIDATE_B_FEATURES
    check_features_allowed(model_features_for_set("layoff_b"))
