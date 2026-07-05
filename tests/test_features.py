"""Tests for the time-safe pre-fight feature builder (Part 3).

Covers the 12 required behaviors:
 1. prior_fights excludes the current fight.
 2. prior_win_pct uses only previous fights.
 3. wins_last_3 uses only previous fights.
 4. days_since_last_fight uses the previous fight date (not current/next).
 5. Elo columns match fight_elo_snapshots pre-fight values.
 6. Difference columns are always fighter_a minus fighter_b.
 7. A fighter's first fight has prior_fights = 0.
 8. A fighter's first fight has days_since_last_fight = null.
 9. No forbidden columns in the output.
10. Deterministic output for the same input.
11. Out-of-order input is still processed chronologically.
12. The current fight result is only used after the feature row is built.
"""

import sys
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufc_pipeline.elo import build_elo_for_db  # noqa: E402
from ufc_pipeline.export import find_forbidden_columns  # noqa: E402
from ufc_pipeline.features import (  # noqa: E402
    FighterHistory,
    age_at_date,
    build_feature_rows,
    build_features_for_db,
    diff,
)
from ufc_pipeline.ingest import ingest_csv  # noqa: E402


def make_fight(fight_id, date, a_id, b_id, a_won, *, a="A", b="B", **extra):
    base = {
        "fight_id": fight_id,
        "date": date,
        "event": f"UFC {fight_id}",
        "fighter_a_id": a_id,
        "fighter_b_id": b_id,
        "fighter_a": a,
        "fighter_b": b,
        "winner": a if a_won else b,
        "fighter_a_won": a_won,
        # Elo snapshot values (arbitrary but valid for unit tests)
        "fighter_a_pre_elo": 1500.0,
        "fighter_b_pre_elo": 1500.0,
        "fighter_a_expected_win_prob": 0.5,
        "fighter_b_expected_win_prob": 0.5,
    }
    base.update(extra)
    return base


# Fighter 1 fights three times (W, W, L) against new opponents each time.
SEQUENCE = [
    make_fight(1, "2020-01-01", 1, 2, 1, a="Ann", b="Bo"),
    make_fight(2, "2020-03-01", 1, 3, 1, a="Ann", b="Cy"),
    make_fight(3, "2020-06-01", 1, 4, 0, a="Ann", b="Dee"),
    make_fight(4, "2020-09-01", 1, 5, 1, a="Ann", b="Eli"),
]


# --------------------------------------------------------------------- 1, 7
def test_prior_fights_excludes_current_fight():
    out = build_feature_rows(SEQUENCE)
    assert list(out["fighter_a_prior_fights"]) == [0, 1, 2, 3]  # never counts itself
    assert out.loc[0, "fighter_a_prior_fights"] == 0            # debut -> 0


# --------------------------------------------------------------------- 2
def test_prior_win_pct_only_previous_fights():
    out = build_feature_rows(SEQUENCE)
    # before fight 2: 1-0 -> 1.0 ; before 3: 2-0 -> 1.0 ; before 4: 2-1 -> 2/3
    assert pd.isna(out.loc[0, "fighter_a_prior_win_pct"])  # debut -> null
    assert out.loc[1, "fighter_a_prior_win_pct"] == pytest.approx(1.0)
    assert out.loc[2, "fighter_a_prior_win_pct"] == pytest.approx(1.0)
    assert out.loc[3, "fighter_a_prior_win_pct"] == pytest.approx(2 / 3)


# --------------------------------------------------------------------- 3
def test_wins_last_3_only_previous_fights():
    out = build_feature_rows(SEQUENCE)
    # windows before each fight: [], [W], [W,W], [W,W,L]
    assert list(out["fighter_a_wins_last_3"]) == [0, 1, 2, 2]
    assert pd.isna(out.loc[0, "fighter_a_win_pct_last_3"])
    assert out.loc[1, "fighter_a_win_pct_last_3"] == pytest.approx(1.0)
    assert out.loc[3, "fighter_a_win_pct_last_3"] == pytest.approx(2 / 3)
    # last-5 window before fight 4 is [W, W, L] -> 2 wins over 3 fights
    assert out.loc[3, "fighter_a_wins_last_5"] == 2
    assert out.loc[3, "fighter_a_win_pct_last_5"] == pytest.approx(2 / 3)


# --------------------------------------------------------------------- 4, 8
def test_days_since_last_fight_uses_previous_fight_date():
    out = build_feature_rows(SEQUENCE)
    assert pd.isna(out.loc[0, "fighter_a_days_since_last_fight"])  # debut -> null
    assert out.loc[1, "fighter_a_days_since_last_fight"] == 60.0   # Jan 1 -> Mar 1
    assert out.loc[2, "fighter_a_days_since_last_fight"] == 92.0   # Mar 1 -> Jun 1
    assert out.loc[3, "fighter_a_days_since_last_fight"] == 92.0   # Jun 1 -> Sep 1
    # opponents are all debuts:
    assert out["fighter_b_days_since_last_fight"].isna().all()


# --------------------------------------------------------------------- 5
def test_elo_columns_match_snapshots(tmp_path):
    csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "RedFighter": ["Ann Ito", "Ann Ito", "Cy Dax"],
            "BlueFighter": ["Bo Rex", "Cy Dax", "Bo Rex"],
            "Winner": ["Red", "Blue", "Red"],
        }
    ).to_csv(csv, index=False)
    db = tmp_path / "ufc.db"
    out = tmp_path / "features.csv"
    ingest_csv(str(csv), str(db), "mdabbert")
    build_elo_for_db(str(db), k=32, starting_elo=1500)
    build_features_for_db(str(db), str(out))

    features = pd.read_csv(out).set_index("fight_id")
    with sqlite3.connect(db) as conn:
        snaps = pd.read_sql_query(
            "SELECT * FROM fight_elo_snapshots", conn
        ).set_index("fight_id")

    for fid in snaps.index:
        for col in (
            "fighter_a_pre_elo", "fighter_b_pre_elo",
            "fighter_a_expected_win_prob", "fighter_b_expected_win_prob",
        ):
            assert features.loc[fid, col] == pytest.approx(snaps.loc[fid, col])
        assert features.loc[fid, "elo_diff"] == pytest.approx(
            snaps.loc[fid, "fighter_a_pre_elo"] - snaps.loc[fid, "fighter_b_pre_elo"]
        )


# --------------------------------------------------------------------- 6
def test_diffs_are_a_minus_b():
    fights = [
        make_fight(
            1, "2020-01-01", 1, 2, 1,
            fighter_a_pre_elo=1600.0, fighter_b_pre_elo=1500.0,
            fighter_a_age=30.0, fighter_b_age=25.0,
            fighter_a_height=180.0, fighter_b_height=175.0,
            fighter_a_reach=185.0, fighter_b_reach=190.0,
        ),
        make_fight(2, "2020-02-01", 1, 3, 1),
        make_fight(3, "2020-03-01", 1, 2, 0),
    ]
    out = build_feature_rows(fights)
    r = out.iloc[0]
    assert r["elo_diff"] == pytest.approx(100.0)
    assert r["age_diff"] == pytest.approx(5.0)
    assert r["height_diff"] == pytest.approx(5.0)
    assert r["reach_diff"] == pytest.approx(-5.0)   # sign matters: A - B
    last = out.iloc[2]  # fighter 1 is 2-0 before this; fighter 2 is 0-1
    assert last["prior_fights_diff"] == 2 - 1
    assert last["prior_wins_diff"] == 2 - 0
    assert last["prior_losses_diff"] == 0 - 1
    assert last["prior_win_pct_diff"] == pytest.approx(1.0 - 0.0)
    assert last["wins_last_3_diff"] == 2 - 0
    # null propagation: any diff with a missing side is null
    assert pd.isna(out.iloc[1]["age_diff"])
    assert diff(None, 3.0) is None and diff(3.0, None) is None
    assert diff(float("nan"), 3.0) is None and diff(3.0, float("nan")) is None


# --------------------------------------------------------------------- 9
def test_no_forbidden_columns_in_feature_output(tmp_path):
    csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "Date": ["2020-01-01"],
            "RedFighter": ["Ann Ito"],
            "BlueFighter": ["Bo Rex"],
            "Winner": ["Red"],
            "RedOdds": [-200], "BlueOdds": [170],       # leaky
            "RMatchWCRank": [4],                        # leaky
            "RedAvgSigStrLanded": [5.1],                # leaky
            "RedControlTime": [88], "BlueTDLanded": [2],  # leaky
        }
    ).to_csv(csv, index=False)
    db = tmp_path / "ufc.db"
    out = tmp_path / "features.csv"
    ingest_csv(str(csv), str(db), "mdabbert")
    build_elo_for_db(str(db), k=32, starting_elo=1500)
    build_features_for_db(str(db), str(out))

    cols = pd.read_csv(out).columns
    assert find_forbidden_columns(cols) == []
    for leaked in ("odds", "rank", "sig_str", "control", "avg"):
        assert not any(leaked in c.lower() for c in cols if c not in
                       {"winner", "fighter_a_won", "method", "round"})


# --------------------------------------------------------------------- 10
def test_deterministic_output(tmp_path):
    csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-02-01", "2020-02-01", "2020-04-01"],
            "RedFighter": ["Ann Ito", "Ann Ito", "Cy Dax", "Dee Oz"],
            "BlueFighter": ["Bo Rex", "Cy Dax", "Bo Rex", "Ann Ito"],
            "Winner": ["Red", "Blue", "Red", "Blue"],
            "RedAge": [28, 28, 31, 24],
            "BlueAge": [30, 31, 30, 28],
            "RedHeightCms": [175, 175, 180, 178],
            "BlueHeightCms": [178, 180, 178, 175],
            "RedReachCms": [180, 180, 183, 181],
            "BlueReachCms": [182, 183, 182, 180],
        }
    ).to_csv(csv, index=False)

    outputs = []
    for run in ("a", "b"):
        db = tmp_path / f"ufc_{run}.db"
        out = tmp_path / f"features_{run}.csv"
        ingest_csv(str(csv), str(db), "mdabbert")
        build_elo_for_db(str(db), k=32, starting_elo=1500)
        build_features_for_db(str(db), str(out))
        outputs.append(out.read_text())
    assert outputs[0] == outputs[1]


# --------------------------------------------------------------------- 11
def test_out_of_order_input_processed_chronologically():
    shuffled = [SEQUENCE[2], SEQUENCE[0], SEQUENCE[3], SEQUENCE[1]]
    out = build_feature_rows(shuffled)
    assert list(out["date"]) == sorted(out["date"])          # chrono output
    assert list(out["fighter_a_prior_fights"]) == [0, 1, 2, 3]
    # identical to processing the correctly-ordered input:
    assert out.equals(build_feature_rows(SEQUENCE))


def test_same_date_ties_broken_by_fight_id():
    fights = [
        make_fight(2, "2020-01-01", 1, 3, 1),  # same day, later fight_id
        make_fight(1, "2020-01-01", 1, 2, 1),  # same day, earlier fight_id
    ]
    out = build_feature_rows(fights)
    assert list(out["fight_id"]) == [1, 2]
    # fight 2's prior record already includes fight 1's result
    assert out.loc[1, "fighter_a_prior_fights"] == 1
    assert out.loc[1, "fighter_a_prior_wins"] == 1


# --------------------------------------------------------------------- 12
def test_current_result_applied_only_after_row_built():
    """If the current result leaked into its own row, a debut winner would
    show prior_wins=1 / win_pct=1.0. It must show a blank slate instead,
    while the NEXT fight must reflect it."""
    fights = [
        make_fight(1, "2020-01-01", 1, 2, 1),
        make_fight(2, "2020-02-01", 1, 3, 0),
    ]
    out = build_feature_rows(fights)
    first = out.iloc[0]
    assert first["fighter_a_prior_wins"] == 0
    assert first["fighter_a_wins_last_3"] == 0
    assert pd.isna(first["fighter_a_prior_win_pct"])
    assert first["fighter_a_won"] == 1                # target is still recorded
    second = out.iloc[1]
    assert second["fighter_a_prior_wins"] == 1        # now fight 1 counts
    assert second["fighter_a_prior_fights"] == 1


# ------------------------------------------------------------ extras
def test_fighter_history_unit():
    h = FighterHistory()
    assert h.prior_record() == {
        "prior_fights": 0, "prior_wins": 0, "prior_losses": 0, "prior_win_pct": None,
    }
    assert h.days_since_last_fight("2020-01-01") is None
    h.update("2020-01-01", 1)
    h.update("2020-02-01", 0)
    rec = h.prior_record()
    assert (rec["prior_fights"], rec["prior_wins"], rec["prior_losses"]) == (2, 1, 1)
    assert rec["prior_win_pct"] == pytest.approx(0.5)
    assert h.days_since_last_fight("2020-03-01") == 29.0


def test_age_at_date_from_dob():
    assert age_at_date("1990-01-01", "2020-01-01") == pytest.approx(30.0, abs=0.05)
    assert age_at_date(None, "2020-01-01") is None
    assert age_at_date("not-a-date", "2020-01-01") is None


def test_missing_elo_snapshots_raise(tmp_path):
    csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {"Date": ["2020-01-01"], "RedFighter": ["Ann Ito"],
         "BlueFighter": ["Bo Rex"], "Winner": ["Red"]}
    ).to_csv(csv, index=False)
    db = tmp_path / "ufc.db"
    ingest_csv(str(csv), str(db), "mdabbert")
    with pytest.raises(RuntimeError, match="no Elo snapshot"):
        build_features_for_db(str(db), str(tmp_path / "features.csv"))
