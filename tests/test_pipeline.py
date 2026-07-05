"""Tests for the ingestion + Elo baseline pipeline.

Covers the 10 required behaviors:
 1. New fighters start at 1500.
 2. Pre-fight Elo is stored before the update.
 3. Winner gains Elo, loser loses Elo.
 4. Chronological ordering is respected.
 5. Out-of-order input is sorted by date before processing.
 6. Expected win probabilities sum to ~1.
 7. Deterministic output for the same input.
 8. No forbidden columns in the exported baseline.
 9. Red maps to fighter_a, blue maps to fighter_b.
10. Winner/loser-only sources: fighter_a is not always the winner.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufc_pipeline.adapters.mdabbert import adapt, _adapt_winner_loser  # noqa: E402
from ufc_pipeline.columns import build_column_mapping, clean_display_name  # noqa: E402
from ufc_pipeline.elo import expected_score, run_elo  # noqa: E402
from ufc_pipeline.export import export_baseline, find_forbidden_columns  # noqa: E402
from ufc_pipeline.ingest import ingest_csv  # noqa: E402
from ufc_pipeline.elo import build_elo_for_db  # noqa: E402


def make_fight(fight_id, date, a, b, a_won):
    return {
        "fight_id": fight_id,
        "date": date,
        "fighter_a_id": a,
        "fighter_b_id": b,
        "fighter_a_won": a_won,
    }


# --------------------------------------------------------------------- 1
def test_new_fighter_starts_at_1500():
    snaps, _ = run_elo([make_fight(1, "2020-01-01", 10, 20, 1)])
    assert snaps[0].fighter_a_pre_elo == 1500.0
    assert snaps[0].fighter_b_pre_elo == 1500.0


# --------------------------------------------------------------------- 2
def test_pre_fight_elo_stored_before_update():
    """The snapshot's pre-Elo must be untouched by that fight's own result."""
    fights = [
        make_fight(1, "2020-01-01", 10, 20, 1),
        make_fight(2, "2020-02-01", 10, 30, 1),
    ]
    snaps, _ = run_elo(fights, k=32)
    # Fight 2: fighter 10's pre-Elo equals their post-Elo from fight 1 —
    # i.e. it reflects only PAST fights, never the current result.
    assert snaps[1].fighter_a_pre_elo == pytest.approx(snaps[0].fighter_a_post_elo)
    assert snaps[1].fighter_a_pre_elo != snaps[1].fighter_a_post_elo
    # Both debutants in fight 1: expected prob must be exactly 0.5,
    # which could not be true if the result had already been applied.
    assert snaps[0].fighter_a_expected_win_prob == pytest.approx(0.5)


# --------------------------------------------------------------------- 3
def test_winner_gains_loser_loses():
    snaps, _ = run_elo([make_fight(1, "2020-01-01", 10, 20, 1)], k=32)
    s = snaps[0]
    assert s.fighter_a_post_elo > s.fighter_a_pre_elo   # winner up
    assert s.fighter_b_post_elo < s.fighter_b_pre_elo   # loser down
    # Equal-rated fighters, K=32 -> +/- 16
    assert s.fighter_a_post_elo == pytest.approx(1516.0)
    assert s.fighter_b_post_elo == pytest.approx(1484.0)


# --------------------------------------------------------------------- 4
def test_chronological_ordering_respected():
    """A later fight must not affect an earlier fight's pre-Elo."""
    fights = [
        make_fight(1, "2020-01-01", 10, 20, 1),
        make_fight(2, "2020-06-01", 10, 20, 0),
        make_fight(3, "2021-01-01", 10, 20, 1),
    ]
    snaps, _ = run_elo(fights)
    by_id = {s.fight_id: s for s in snaps}
    assert by_id[1].fighter_a_pre_elo == 1500.0
    assert by_id[2].fighter_a_pre_elo == pytest.approx(by_id[1].fighter_a_post_elo)
    assert by_id[3].fighter_a_pre_elo == pytest.approx(by_id[2].fighter_a_post_elo)


def test_same_date_ties_broken_by_fight_id():
    fights = [
        make_fight(2, "2020-01-01", 10, 30, 1),  # same day, later fight_id
        make_fight(1, "2020-01-01", 10, 20, 1),  # same day, earlier fight_id
    ]
    snaps, _ = run_elo(fights)
    assert [s.fight_id for s in snaps] == [1, 2]
    # Fight 2's pre-Elo for fighter 10 already includes fight 1's result.
    assert snaps[1].fighter_a_pre_elo == pytest.approx(snaps[0].fighter_a_post_elo)


# --------------------------------------------------------------------- 5
def test_out_of_order_input_sorted_by_date():
    shuffled = [
        make_fight(3, "2021-01-01", 10, 20, 1),
        make_fight(1, "2019-01-01", 10, 20, 1),
        make_fight(2, "2020-01-01", 10, 20, 0),
    ]
    snaps, _ = run_elo(shuffled)
    assert [s.date for s in snaps] == ["2019-01-01", "2020-01-01", "2021-01-01"]
    assert snaps[0].fighter_a_pre_elo == 1500.0  # earliest fight is truly first


# --------------------------------------------------------------------- 6
def test_expected_probs_sum_to_one():
    fights = [
        make_fight(1, "2020-01-01", 10, 20, 1),
        make_fight(2, "2020-02-01", 10, 20, 1),
        make_fight(3, "2020-03-01", 10, 20, 0),
    ]
    snaps, _ = run_elo(fights)
    for s in snaps:
        total = s.fighter_a_expected_win_prob + s.fighter_b_expected_win_prob
        assert total == pytest.approx(1.0, abs=1e-12)
    # sanity: formula matches spec
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1600, 1500) == pytest.approx(
        1 / (1 + 10 ** ((1500 - 1600) / 400))
    )


# --------------------------------------------------------------------- 7
def test_deterministic_output(tmp_path):
    csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-02-01", "2020-02-01"],
            "RedFighter": ["Alice Silva", "Alice Silva", "Cara Jones"],
            "BlueFighter": ["Bea Nunes", "Cara Jones", "Bea Nunes"],
            "Winner": ["Red", "Blue", "Red"],
            "Event": ["UFC 1", "UFC 2", "UFC 2"],
        }
    ).to_csv(csv, index=False)

    outputs = []
    for run in ("a", "b"):
        db = tmp_path / f"ufc_{run}.db"
        out = tmp_path / f"base_{run}.csv"
        ratings = tmp_path / f"ratings_{run}.csv"
        ingest_csv(str(csv), str(db), "mdabbert")
        build_elo_for_db(str(db), k=32, starting_elo=1500)
        export_baseline(str(db), str(out), str(ratings))
        outputs.append((out.read_text(), ratings.read_text()))

    assert outputs[0] == outputs[1]


# --------------------------------------------------------------------- 8
def test_no_forbidden_columns_in_export(tmp_path):
    csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "Date": ["2020-01-01"],
            "RedFighter": ["Alice Silva"],
            "BlueFighter": ["Bea Nunes"],
            "Winner": ["Red"],
            # Leaky columns that must NOT survive to the export:
            "RedOdds": [-150],
            "BlueOdds": [130],
            "RMatchWCRank": [3],
            "RedAvgSigStrLanded": [4.5],
            "BlueAvgTDLanded": [1.2],
            "RedControlTime": [120],
        }
    ).to_csv(csv, index=False)

    db = tmp_path / "ufc.db"
    out = tmp_path / "base.csv"
    ratings = tmp_path / "ratings.csv"
    ingest_csv(str(csv), str(db), "mdabbert")
    build_elo_for_db(str(db), k=32, starting_elo=1500)
    export_baseline(str(db), str(out), str(ratings))

    cols = pd.read_csv(out).columns
    assert find_forbidden_columns(cols) == []
    # allowed result/metadata fields are still permitted:
    assert "winner" in cols and "fighter_a_won" in cols

    # And the guard itself works:
    assert find_forbidden_columns(["red_odds"]) == ["red_odds"]
    assert find_forbidden_columns(["avg_sig_str"]) == ["avg_sig_str"]
    assert find_forbidden_columns(["winner", "method", "round"]) == []


# --------------------------------------------------------------------- 9
def test_red_maps_to_a_blue_maps_to_b():
    df = pd.DataFrame(
        {
            "Date": ["2020-01-01"],
            "RedFighter": ["Alice Silva"],
            "BlueFighter": ["Bea Nunes"],
            "Winner": ["Blue"],
        }
    )
    norm = adapt(df)
    assert norm.loc[0, "fighter_a_name"] == "Alice Silva"   # red -> A
    assert norm.loc[0, "fighter_b_name"] == "Bea Nunes"     # blue -> B
    assert norm.loc[0, "winner_name"] == "Bea Nunes"        # corner label resolved


def test_winner_as_name_also_resolves():
    df = pd.DataFrame(
        {
            "date": ["2020-01-01"],
            "R_fighter": ["Alice Silva"],
            "B_fighter": ["Bea Nunes"],
            "Winner": ["Alice Silva"],
        }
    )
    norm = adapt(df)
    assert norm.loc[0, "winner_name"] == "Alice Silva"


# --------------------------------------------------------------------- 10
def test_winner_loser_source_alternates_fighter_a():
    df = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 4,
            "winner_fighter": ["W0", "W1", "W2", "W3"],
            "loser_fighter": ["L0", "L1", "L2", "L3"],
        }
    )
    mapping = build_column_mapping(df)
    norm = _adapt_winner_loser(df, mapping)
    # even rows: winner in A slot; odd rows: winner in B slot
    assert list(norm["fighter_a_name"]) == ["W0", "L1", "W2", "L3"]
    assert list(norm["winner_name"]) == ["W0", "W1", "W2", "W3"]
    # therefore fighter_a is NOT always the winner
    a_is_winner = norm["fighter_a_name"] == norm["winner_name"]
    assert not a_is_winner.all()
    # deterministic label:
    assert list(a_is_winner.astype(int)) == [1, 0, 1, 0]


# ------------------------------------------------------------ extras
def test_missing_columns_raise_clear_error():
    df = pd.DataFrame({"foo": [1], "bar": [2]})
    with pytest.raises(ValueError) as err:
        adapt(df)
    msg = str(err.value)
    assert "Missing normalized fields" in msg
    assert "foo" in msg  # shows the columns that were found


def test_nullable_missing_names_clean_to_empty_string():
    assert clean_display_name(pd.NA) == ""


def test_draws_and_bad_rows_dropped(tmp_path):
    csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-01", "not-a-date", "2020-03-01"],
            "RedFighter": ["Alice Silva", "Cara Jones", "Dee Ko", "Eve Fox"],
            "BlueFighter": ["Bea Nunes", "Cara Jones", "Fay Gu", "Gia Ho"],
            "Winner": ["Red", "Red", "Red", "Draw"],
        }
    ).to_csv(csv, index=False)
    db = tmp_path / "ufc.db"
    report = ingest_csv(str(csv), str(db), "mdabbert")
    assert report.total_rows == 4
    assert report.ingested == 1          # only the first row is fully valid
    assert report.dropped_bad_names == 1  # Cara Jones vs Cara Jones
    assert report.dropped_bad_date == 1
    assert report.dropped_no_winner == 1  # the draw
