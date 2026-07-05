"""Tests for Step 3B: rolling striking/grappling features.

Leakage proofs:
 - current fight's stats never appear in its own row
 - future fights never affect earlier rows
 - same-date fights only see earlier same-date results
Math checks:
 - per-minute / per-15 rates and accuracy/defense on hand-computed history
Missing-history policy:
 - debut -> nulls + no_prior_stats flag (NOT zeros)
 - fights without stats rows don't feed stat history
Plumbing:
 - row counts preserved; Step 3 columns intact; deterministic
 - guard: historical names allowed as model inputs, raw stat names rejected
 - Greco converter parses 'x of y', 'm:ss', round/time -> minutes
 - end-to-end: stats ingest + 3B build + model comparison
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufc_pipeline.elo import build_elo_for_db  # noqa: E402
from ufc_pipeline.ingest import ingest_csv  # noqa: E402
from ufc_pipeline.modeling import (  # noqa: E402
    STEP3B_MODEL_FEATURES,
    check_features_allowed,
    compare_feature_sets,
)
from ufc_pipeline.stats_features import (  # noqa: E402
    METRICS,
    StatsHistory,
    build_step3b_for_db,
    build_step3b_rows,
)
from ufc_pipeline.stats_ingest import (  # noqa: E402
    _fight_minutes,
    _parse_mmss_seconds,
    _parse_of,
    convert_greco,
    ingest_stats_csv,
)


def fight(fid, date, a_id, b_id, a_won=1):
    return {
        "fight_id": fid, "date": date,
        "fighter_a_id": a_id, "fighter_b_id": b_id, "fighter_a_won": a_won,
    }


def stat_row(minutes, sl, sa, tdl, tda, kd=0, sub=0, ctrl=0):
    return {
        "minutes": minutes, "knockdowns": kd,
        "sig_str_landed": sl, "sig_str_attempted": sa,
        "td_landed": tdl, "td_attempted": tda,
        "sub_attempts": sub, "ctrl_seconds": ctrl,
    }


# ------------------------------------------------------ leakage proofs
def test_current_fight_stats_not_in_own_row():
    fights = [fight(1, "2020-01-01", 1, 2), fight(2, "2020-03-01", 1, 3)]
    stats = {
        (1, 1): stat_row(15.0, sl=60, sa=120, tdl=3, tda=6, kd=1, sub=2, ctrl=300),
        (1, 2): stat_row(15.0, sl=30, sa=90, tdl=0, tda=2),
        (2, 1): stat_row(10.0, sl=999, sa=999, tdl=99, tda=99),  # poison values
        (2, 3): stat_row(10.0, sl=999, sa=999, tdl=99, tda=99),
    }
    out = build_step3b_rows(fights, stats)
    # Fight 1: both debuts -> ALL null + flags, despite fight 1 having stats.
    r1 = out.iloc[0]
    for m in METRICS:
        assert pd.isna(r1[f"fighter_a_{m}"]) and pd.isna(r1[f"fighter_b_{m}"])
    assert r1["fighter_a_no_prior_stats"] == 1
    # Fight 2: fighter 1's row reflects EXACTLY fight 1, not fight 2's 999s.
    r2 = out.iloc[1]
    assert r2["fighter_a_sig_strikes_landed_per_min"] == pytest.approx(60 / 15)
    assert r2["fighter_a_sig_strikes_absorbed_per_min"] == pytest.approx(30 / 15)
    assert r2["fighter_a_sig_strike_differential"] == pytest.approx(2.0)
    assert r2["fighter_a_striking_accuracy"] == pytest.approx(60 / 120)
    assert r2["fighter_a_striking_defense"] == pytest.approx(1 - 30 / 90)
    assert r2["fighter_a_knockdown_rate"] == pytest.approx(1 / 15 * 15)
    assert r2["fighter_a_takedowns_per_15"] == pytest.approx(3 / 15 * 15)
    assert r2["fighter_a_takedown_accuracy"] == pytest.approx(3 / 6)
    assert r2["fighter_a_takedown_defense"] == pytest.approx(1 - 0 / 2)
    assert r2["fighter_a_control_time_per_15"] == pytest.approx((300 / 60) / 15 * 15)
    assert r2["fighter_a_submission_attempts_per_15"] == pytest.approx(2 / 15 * 15)
    assert r2["fighter_a_no_prior_stats"] == 0


def test_future_fights_do_not_affect_earlier_rows():
    base_fights = [fight(1, "2020-01-01", 1, 2), fight(2, "2020-03-01", 1, 3)]
    stats = {
        (1, 1): stat_row(15.0, 60, 120, 3, 6),
        (1, 2): stat_row(15.0, 30, 90, 0, 2),
        (2, 1): stat_row(15.0, 90, 150, 0, 0),
        (2, 3): stat_row(15.0, 45, 100, 1, 3),
    }
    with_future = build_step3b_rows(
        base_fights + [fight(3, "2020-06-01", 1, 4)], stats
    )
    without_future = build_step3b_rows(base_fights, stats)
    pd.testing.assert_frame_equal(
        with_future.iloc[:2].reset_index(drop=True), without_future
    )


def test_same_date_fights_only_see_earlier_dataset_order():
    fights = [fight(1, "2020-01-01", 1, 2), fight(2, "2020-01-01", 1, 3)]
    stats = {
        (1, 1): stat_row(15.0, 60, 120, 0, 0),
        (1, 2): stat_row(15.0, 30, 90, 0, 0),
    }
    out = build_step3b_rows(fights, stats)
    assert pd.isna(out.iloc[0]["fighter_a_sig_strikes_landed_per_min"])
    # second same-date fight sees the first one's stats
    assert out.iloc[1]["fighter_a_sig_strikes_landed_per_min"] == pytest.approx(4.0)


def test_out_of_order_input_and_determinism():
    fights = [fight(2, "2020-03-01", 1, 3), fight(1, "2020-01-01", 1, 2)]
    stats = {(1, 1): stat_row(15.0, 60, 120, 0, 0), (1, 2): stat_row(15.0, 30, 90, 0, 0)}
    a = build_step3b_rows(fights, stats)
    b = build_step3b_rows(list(reversed(fights)), stats)
    pd.testing.assert_frame_equal(a, b)
    assert list(a["fight_id"]) == [1, 2]


# ------------------------------------------------------ missing history
def test_missing_stats_are_null_with_flag_not_zero():
    # fighter 1's first fight has NO stats row -> before fight 2 they still
    # have no stat history: nulls + flag (a zero would claim "lands nothing")
    fights = [fight(1, "2020-01-01", 1, 2), fight(2, "2020-03-01", 1, 3)]
    out = build_step3b_rows(fights, stats={})
    r2 = out.iloc[1]
    assert pd.isna(r2["fighter_a_sig_strikes_landed_per_min"])
    assert r2["fighter_a_no_prior_stats"] == 1
    assert r2["no_prior_stats_diff"] == 0  # both sides missing history


def test_zero_denominator_ratios_are_null():
    h = StatsHistory()
    h.update(stat_row(15.0, sl=10, sa=20, tdl=0, tda=0),   # never attempted a TD
             stat_row(15.0, sl=5, sa=10, tdl=0, tda=0), 15.0)
    snap = h.snapshot()
    assert snap["takedown_accuracy"] is None
    assert snap["takedown_defense"] is None
    assert snap["striking_accuracy"] == pytest.approx(0.5)


def test_nan_stats_do_not_poison_history():
    h = StatsHistory()
    h.update(
        stat_row(15.0, sl=np.nan, sa=20, tdl=np.nan, tda=0),
        stat_row(15.0, sl=5, sa=np.nan, tdl=0, tda=np.nan),
        15.0,
    )
    snap = h.snapshot()
    assert snap["sig_strikes_landed_per_min"] == 0.0
    assert snap["sig_strikes_absorbed_per_min"] == pytest.approx(5 / 15)
    assert snap["striking_accuracy"] == 0.0
    assert snap["striking_defense"] is None


# ------------------------------------------------------ guard
def test_historical_features_allowed_raw_stats_rejected():
    check_features_allowed(["elo_diff"] + STEP3B_MODEL_FEATURES)  # must pass
    for bad in ("red_sig_str_landed", "blue_takedowns", "ctrl_control_time",
                "sig_str_landed", "avg_takedown_diff"):
        with pytest.raises(ValueError, match="Leakage guard"):
            check_features_allowed(["elo_diff", bad])


# ------------------------------------------------------ greco converter
def test_greco_parsers():
    assert _parse_of("37 of 102") == (37.0, 102.0)
    assert _parse_of("---") == (0.0, 0.0)
    assert _parse_mmss_seconds("2:03") == 123.0
    assert _parse_mmss_seconds("--") == 0.0
    assert _fight_minutes(3, "2:30", "3 Rnd (5-5-5)") == pytest.approx(12.5)
    assert _fight_minutes(1, "0:47", "3 Rnd (5-5-5)") == pytest.approx(47 / 60)


def test_convert_greco_end_to_end():
    events = pd.DataFrame({"EVENT": ["UFC 1"], "DATE": ["January 05, 2020"],
                           "URL": ["u"], "LOCATION": ["x"]})
    results = pd.DataFrame({
        "EVENT": ["UFC 1"], "BOUT": ["Ann Ito vs. Bo Rex"], "OUTCOME": ["W/L"],
        "WEIGHTCLASS": ["LW"], "METHOD": ["KO"], "ROUND": [2], "TIME": ["1:30"],
        "TIME FORMAT": ["3 Rnd (5-5-5)"], "REFEREE": ["r"], "URL": ["u"],
    })
    stats = pd.DataFrame({
        "EVENT": ["UFC 1"] * 4, "BOUT": ["Ann Ito vs. Bo Rex"] * 4,
        "ROUND": ["Round 1", "Round 2", "Round 1", "Round 2"],
        "FIGHTER": ["Ann Ito", "Ann Ito", "Bo Rex", "Bo Rex"],
        "KD": [0, 1, 0, 0],
        "SIG.STR.": ["20 of 40", "10 of 20", "15 of 30", "5 of 15"],
        "TD": ["1 of 2", "0 of 1", "0 of 0", "0 of 0"],
        "SUB.ATT": [0, 1, 0, 0], "CTRL": ["1:00", "0:30", "0:00", "0:10"],
    })
    wide = convert_greco(stats, results, events)
    assert len(wide) == 1
    r = wide.iloc[0]
    assert r["date"] == "2020-01-05"
    assert r["total_minutes"] == pytest.approx(6.5)   # 5 + 1:30
    assert r["red_sig_str_landed"] == 30 and r["red_sig_str_attempted"] == 60
    assert r["blue_sig_str_landed"] == 20 and r["blue_sig_str_attempted"] == 45
    assert r["red_td_landed"] == 1 and r["red_td_attempted"] == 3
    assert r["red_kd"] == 1 and r["red_sub_att"] == 1
    assert r["red_ctrl_seconds"] == 90.0 and r["blue_ctrl_seconds"] == 10.0


# ------------------------------------------------------ end-to-end via DB
@pytest.fixture
def small_db(tmp_path):
    fights_csv = tmp_path / "fights.csv"
    pd.DataFrame({
        "Date": ["2020-01-01", "2020-03-01", "2020-06-01"],
        "RedFighter": ["Ann Ito", "Ann Ito", "Bo Rex"],
        "BlueFighter": ["Bo Rex", "Cy Dax", "Cy Dax"],
        "Winner": ["Red", "Blue", "Red"],
    }).to_csv(fights_csv, index=False)
    db = tmp_path / "ufc.db"
    ingest_csv(str(fights_csv), str(db), "mdabbert")
    build_elo_for_db(str(db), k=32, starting_elo=1500)

    stats_csv = tmp_path / "stats.csv"
    pd.DataFrame({
        "date": ["2020-01-01", "2020-03-01"],
        # note: second row's corners are SWAPPED vs the fights table on purpose
        "red_fighter": ["Ann Ito", "Cy Dax"],
        "blue_fighter": ["Bo Rex", "Ann Ito"],
        "total_minutes": [15.0, 10.0],
        "red_kd": [1, 0], "blue_kd": [0, 0],
        "red_sig_str_landed": [60, 40], "blue_sig_str_landed": [30, 50],
        "red_sig_str_attempted": [120, 80], "blue_sig_str_attempted": [90, 100],
        "red_td_landed": [3, 1], "blue_td_landed": [0, 2],
        "red_td_attempted": [6, 2], "blue_td_attempted": [2, 4],
        "red_sub_att": [2, 0], "blue_sub_att": [0, 1],
        "red_ctrl_seconds": [300, 60], "blue_ctrl_seconds": [30, 120],
    }).to_csv(stats_csv, index=False)
    report = ingest_stats_csv(str(stats_csv), str(db))
    assert report.matched == 2 and report.unmatched == 0
    return tmp_path, db


def test_stats_ingest_handles_swapped_corners(small_db):
    tmp_path, db = small_db
    out = tmp_path / "f3b.csv"
    build_step3b_for_db(str(db), str(out))
    df = pd.read_csv(out)
    # Fight 3 (Bo Rex vs Cy Dax): Ann Ito's fight-2 stats were listed with
    # corners swapped in the stats CSV, so Cy Dax was CSV-red. Cy Dax's
    # history entering fight 3 must be fight 2's red column values.
    r3 = df[df["fight_id"] == 3].iloc[0]
    assert r3["fighter_b"] == "Cy Dax"
    assert r3["fighter_b_sig_strikes_landed_per_min"] == pytest.approx(40 / 10)
    assert r3["fighter_b_takedowns_per_15"] == pytest.approx(1 / 10 * 15)
    # Bo Rex entering fight 3: fight 1 blue-corner values.
    assert r3["fighter_a_sig_strikes_landed_per_min"] == pytest.approx(30 / 15)
    assert r3["fighter_a_striking_defense"] == pytest.approx(1 - 60 / 120)


def test_step3b_output_shape_and_columns(small_db):
    tmp_path, db = small_db
    out = tmp_path / "f3b.csv"
    n = build_step3b_for_db(str(db), str(out))
    df = pd.read_csv(out)
    assert n == len(df) == 3                       # row count matches fights
    # Step 3 columns intact:
    for col in ("elo_diff", "prior_win_pct_diff", "days_since_last_fight_diff",
                "fighter_a_pre_elo", "winner", "fighter_a_won"):
        assert col in df.columns
    # Step 3B columns present:
    for m in METRICS:
        assert f"{m}_diff" in df.columns
        assert f"fighter_a_{m}" in df.columns
    assert "fighter_a_no_prior_stats" in df.columns


def test_comparison_runs_and_reports_three_models(tmp_path, small_db):
    src_tmp, db = small_db
    out = src_tmp / "f3b.csv"
    build_step3b_for_db(str(db), str(out))
    # 3 fights is too few to train on -> use a synthetic expansion instead:
    rng = np.random.default_rng(0)
    n = 300
    base = pd.read_csv(out)
    rows = pd.concat([base] * (n // len(base) + 1)).head(n).reset_index(drop=True)
    rows["fight_id"] = np.arange(1, n + 1)
    rows["date"] = pd.date_range("2015-01-01", periods=n, freq="5D").strftime("%Y-%m-%d")
    elo = rng.normal(0, 100, n)
    rows["elo_diff"] = elo
    p = 1 / (1 + 10 ** (-elo / 400))
    rows["fighter_a_expected_win_prob"] = p
    rows["fighter_a_won"] = (rng.random(n) < p).astype(int)
    for m in METRICS:  # give 3B features some signal + noise
        rows[f"{m}_diff"] = 0.002 * elo + rng.normal(0, 1, n)
    rows["fighter_a_no_prior_stats"] = 0
    rows["fighter_b_no_prior_stats"] = 0
    csv = tmp_path / "features.csv"
    rows.to_csv(csv, index=False)

    results = compare_feature_sets(
        input_csv=str(csv),
        metrics_output=str(tmp_path / "cmp.json"),
        predictions_output=str(tmp_path / "cmp_preds.csv"),
        coefficients_dir=str(tmp_path),
    )
    assert "elo_baseline" in results
    assert set(results["models"]) == {"step3_basic", "step3_plus_3b"}
    for entry in results["models"].values():
        m = entry["logistic_regression"]
        assert 0 <= m["accuracy"] <= 1 and m["log_loss"] > 0
    # 3B features actually used in the second model:
    used = results["models"]["step3_plus_3b"]["features_numeric"]
    assert "sig_strikes_landed_per_min_diff" in used
    assert "sig_strikes_landed_per_min_diff" not in results["models"]["step3_basic"]["features_numeric"]
    preds = pd.read_csv(tmp_path / "cmp_preds.csv")
    for col in ("elo_pred_prob", "step3_basic_logistic_prob", "step3_plus_3b_logistic_prob"):
        assert preds[col].between(0, 1).all()
