"""Tests for Step 6C upcoming-card feature builder.

Covers name normalization/matching (exact only, unmatched, ambiguous, fuzzy off
by default), matchup schema validation, strict time-safety (no fights on/after
event_date), debut/low-history honesty, output has the Step 3C model features and
passes Step 6B validation, no result/leakage columns escape, failed-row
reporting, report schema, and the CLI run on a small synthetic DB. A tiny SQLite
fixture stands in for the history DB; no real card or network is used.
"""

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufc_pipeline.db import init_schema  # noqa: E402
from ufc_pipeline.elo import build_elo_for_db  # noqa: E402
from ufc_pipeline.feature_diagnostics import official_step3c_features  # noqa: E402
from ufc_pipeline.step6b_live_predictions import validate_prediction_input  # noqa: E402
from ufc_pipeline.step6c_upcoming_feature_builder import (  # noqa: E402
    FORBIDDEN_OUTPUT_COLUMNS,
    build_fighter_lookup,
    build_upcoming_features,
    match_fighter,
    normalize_fighter_name,
    run_build,
    validate_matchup_schema,
)

BASE_NUMERIC, _ = official_step3c_features()

FIGHTERS = [
    (1, "Jon Jones", "185", "215", "1987-07-19"),
    (2, "Stipe Miocic", "193", "203", "1982-08-19"),
    (3, "Alex Pereira", "193", "200", "1987-07-07"),
    (4, "Israel Adesanya", "193", "203", "1989-07-22"),
    (5, "Bruno Silva", "180", "185", "1990-01-01"),   # ambiguous pair (id 5)
    (6, "Bruno Silva", "175", "180", "1989-02-02"),   # ambiguous pair (id 6)
    (7, "Debut Guy", "180", "183", "1998-03-03"),     # no prior fights
]
# (fight_id, date, a_id, b_id, a_won) — winner is a when a_won==1
FIGHTS = [
    (1, "2020-01-01", 1, 2, 1),
    (2, "2020-06-01", 3, 4, 1),
    (3, "2021-01-01", 1, 3, 1),
    (4, "2021-06-01", 2, 4, 1),
    (5, "2022-01-01", 1, 4, 1),
    (6, "2022-06-01", 3, 2, 1),
    (7, "2023-01-01", 5, 6, 1),
    (8, "2024-01-01", 1, 3, 1),   # AFTER the test event_date 2023-06-01
]


@pytest.fixture(scope="module")
def history_db(tmp_path_factory):
    db = tmp_path_factory.mktemp("step6c_db") / "hist.db"
    conn = sqlite3.connect(db)
    init_schema(conn)
    conn.executemany(
        "INSERT INTO fighters (fighter_id, name, normalized_name, height, reach, stance, date_of_birth) "
        "VALUES (?,?,?,?,?,?,?)",
        [(fid, name, name.lower(), h, r, "Orthodox", dob) for fid, name, h, r, dob in FIGHTERS],
    )
    for fid, date, a, b, a_won in FIGHTS:
        winner = a if a_won == 1 else b
        conn.execute(
            "INSERT INTO fights (fight_id, date, event, fighter_a_id, fighter_b_id, winner_id, fighter_a_won, "
            "weight_class, method, round, source, source_row_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (fid, date, f"EVENT {fid}", a, b, winner, a_won, "Lightweight", "Decision", 3, "test", fid),
        )
        for fighter in (a, b):
            conn.execute(
                "INSERT INTO fight_stats (fight_id, fighter_id, minutes, knockdowns, sig_str_landed, "
                "sig_str_attempted, td_landed, td_attempted, sub_attempts, ctrl_seconds, source) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (fid, fighter, 15.0, 0, 40, 90, 1, 3, 0, 120, "test"),
            )
    conn.commit()
    conn.close()
    build_elo_for_db(str(db), k=32.0, starting_elo=1500.0)
    return str(db)


def _matchups(rows, event_date="2023-06-01"):
    return pd.DataFrame(
        [{"event_date": event_date, "event_name": "UFC Test", "fighter_a": a, "fighter_b": b, "weight_class": "Lightweight"}
         for a, b in rows]
    )


# ---------------------------------------------------------------------------
# Name normalization + matching
# ---------------------------------------------------------------------------

def test_normalize_fighter_name():
    assert normalize_fighter_name("José  Aldo Jr.") == "jose aldo jr"
    assert normalize_fighter_name("  Georges  St-Pierre ") == "georges st pierre"
    assert normalize_fighter_name("Sean O'Malley") == "sean omalley"
    assert normalize_fighter_name("KHABIB") == "khabib"
    assert normalize_fighter_name(None) == ""


def test_match_fighter_exact_unmatched_ambiguous():
    lookup = build_fighter_lookup(pd.DataFrame(
        [{"fighter_id": fid, "name": name, "height": h, "reach": r, "date_of_birth": dob}
         for fid, name, h, r, dob in FIGHTERS]))
    assert match_fighter("Jon Jones", lookup)["status"] == "matched"
    assert match_fighter("jon  jones", lookup)["fighter_id"] == 1           # normalized exact
    assert match_fighter("Ghost Fighter", lookup)["status"] == "unmatched"
    amb = match_fighter("Bruno Silva", lookup)
    assert amb["status"] == "ambiguous" and amb["fighter_id"] is None        # two Bruno Silvas


def test_fuzzy_matching_disabled_by_default():
    lookup = build_fighter_lookup(pd.DataFrame(
        [{"fighter_id": 1, "name": "Jon Jones", "height": "185", "reach": "215", "date_of_birth": "1987-07-19"}]))
    # a near-miss is NOT silently matched, with or without the (no-op) fuzzy flag
    assert match_fighter("Jon Jonez", lookup, allow_fuzzy=False)["status"] == "unmatched"
    assert match_fighter("Jon Jonez", lookup, allow_fuzzy=True)["status"] == "unmatched"


def test_reviewed_aliases_resolve_but_similar_names_do_not():
    lookup = build_fighter_lookup(pd.DataFrame([
        {"fighter_id": 35, "name": "King Green", "height": "178", "reach": "180", "date_of_birth": "1986-09-09"},
        {"fighter_id": 185, "name": "Terrance McKinney", "height": "178", "reach": "185", "date_of_birth": "1994-09-15"},
        {"fighter_id": 516, "name": "Elisha Ellison", "height": "183", "reach": "198", "date_of_birth": "1997-03-05"},
        {"fighter_id": 900, "name": "Gable Steveson", "height": "185", "reach": "200", "date_of_birth": "2000-05-31"},
    ]))
    assert match_fighter("Bobby Green", lookup)["fighter_id"] == 35
    assert match_fighter("Terrance Mickney", lookup)["fighter_id"] == 185
    assert match_fighter("Elisha Ellison", lookup)["fighter_id"] == 516
    assert match_fighter("Gable Steveson", lookup)["fighter_id"] == 900
    assert match_fighter("Gable Stevenson", lookup)["status"] == "unmatched"


# ---------------------------------------------------------------------------
# Matchup schema validation
# ---------------------------------------------------------------------------

def test_validate_matchup_schema_missing_and_forbidden():
    v = validate_matchup_schema(pd.DataFrame([{"fighter_a": "A", "fighter_b": "B"}]))
    assert v["ok"] is False and "event_date" in v["missing_required_columns"]
    v2 = validate_matchup_schema(pd.DataFrame(
        [{"event_date": "2026-01-01", "event_name": "E", "fighter_a": "A", "fighter_b": "B", "winner": "A"}]))
    assert v2["ok"] is False and "winner" in v2["forbidden_columns_present"]


def test_validate_matchup_schema_per_row_errors():
    df = pd.DataFrame([
        {"event_date": "2026-01-01", "event_name": "E", "fighter_a": "A", "fighter_b": "A"},   # a==b
        {"event_date": "not-a-date", "event_name": "E", "fighter_a": "A", "fighter_b": "B"},    # bad date
        {"event_date": "2026-01-01", "event_name": "E", "fighter_a": "", "fighter_b": "B"},     # empty
        {"event_date": "2026-01-01", "event_name": "E", "fighter_a": "X", "fighter_b": "Y"},    # ok
    ])
    v = validate_matchup_schema(df)
    statuses = [r["status"] for r in v["row_status"]]
    assert statuses == ["error", "error", "error", "valid"]


def test_validate_matchup_schema_rejects_null_event_date():
    df = pd.DataFrame([
        {"event_date": None, "event_name": "E", "fighter_a": "A", "fighter_b": "B"},
    ])

    v = validate_matchup_schema(df)

    assert v["ok"] is True  # schema exists; the bad value is a failed row
    assert v["row_status"] == [{
        "row_index": 0, "status": "error", "reasons": ["unparseable event_date"],
    }]


# ---------------------------------------------------------------------------
# End-to-end build
# ---------------------------------------------------------------------------

def test_build_produces_step6b_valid_features(history_db):
    res = build_upcoming_features(_matchups([("Jon Jones", "Stipe Miocic"), ("Alex Pereira", "Israel Adesanya")]),
                                  db_path=history_db)
    feat, rep = res["features"], res["report"]
    assert len(feat) == 2 and rep["n_failed_rows"] == 0
    # every model feature column present, no result/leakage columns escape
    assert set(BASE_NUMERIC).issubset(feat.columns)
    assert not (FORBIDDEN_OUTPUT_COLUMNS & set(feat.columns))
    # passes the exact Step 6B input validation
    v = validate_prediction_input(feat, BASE_NUMERIC)
    assert v["ok"] is True and v["n_error_rows"] == 0
    assert rep["step6b_validation"]["ok"] is True
    # output id/metadata columns
    for col in ("event_date", "event_name", "fighter_a", "fighter_b", "weight_class"):
        assert col in feat.columns


def test_time_safety_no_fights_on_or_after_event_date(history_db):
    # Jones fought f1(2020),f3(2021),f5(2022) before 2023-06-01, and f8(2024) after.
    res = build_upcoming_features(_matchups([("Jon Jones", "Stipe Miocic")], event_date="2023-06-01"), db_path=history_db)
    ts = res["report"]["time_safety"][0]
    assert ts["cutoff_strictly_before_event"] is True
    assert ts["max_historical_fight_date_used"] < "2023-06-01"
    # Jones prior fights must exclude the 2024 fight -> exactly 3 (f1,f3,f5)
    assert ts["fighter_a_prior_fights"] == 3
    # Moving the event earlier reduces the count -> proves the cutoff is applied
    res_early = build_upcoming_features(_matchups([("Jon Jones", "Stipe Miocic")], event_date="2021-03-01"), db_path=history_db)
    assert res_early["report"]["time_safety"][0]["fighter_a_prior_fights"] == 2  # only f1, f3


def test_duplicate_fighter_on_card_is_failed_without_contaminating_features(history_db):
    solo = build_upcoming_features(_matchups([("Jon Jones", "Stipe Miocic")]), db_path=history_db)
    card = _matchups([("Jon Jones", "Stipe Miocic"), ("Jon Jones", "Alex Pereira")])
    card.loc[1, "event_date"] = "2023-06-01T00:00:00"
    res = build_upcoming_features(
        card, db_path=history_db,
    )

    assert len(res["features"]) == 1
    assert res["report"]["n_failed_rows"] == 1
    failed = res["report"]["failed_rows"][0]
    assert failed["row_index"] == 1
    assert any("multiple matchups" in reason for reason in failed["reasons"])
    pd.testing.assert_series_equal(
        res["features"].iloc[0], solo["features"].iloc[0], check_names=False,
    )


def test_unresolved_matchup_does_not_reserve_a_fighter_on_the_card(history_db):
    solo = build_upcoming_features(_matchups([("Jon Jones", "Stipe Miocic")]), db_path=history_db)
    res = build_upcoming_features(
        _matchups([("Jon Jones", "Ghost Fighter"), ("Jon Jones", "Stipe Miocic")]),
        db_path=history_db,
    )

    assert len(res["features"]) == 1
    assert res["report"]["n_failed_rows"] == 1
    assert res["report"]["failed_rows"][0]["row_index"] == 0
    assert any("unmatched" in reason for reason in res["report"]["failed_rows"][0]["reasons"])
    pd.testing.assert_series_equal(
        res["features"].iloc[0], solo["features"].iloc[0], check_names=False,
    )


def test_debut_fighter_gets_null_history_not_failure(history_db):
    res = build_upcoming_features(_matchups([("Jon Jones", "Debut Guy")]), db_path=history_db)
    feat, rep = res["features"], res["report"]
    assert len(feat) == 1 and rep["n_failed_rows"] == 0  # debut is built, not failed
    row = feat.iloc[0]
    # Debut Guy (fighter_b) has no prior fights -> honest nulls + no_prior_stats flag
    assert pd.isna(row["fighter_b_win_pct_last_3"]) if "fighter_b_win_pct_last_3" in feat.columns else True
    assert row["fighter_b_no_prior_stats"] == 1
    assert rep["low_history_matchups"]  # reported as low-history


def test_unmatched_fighter_is_failed_row(history_db):
    res = build_upcoming_features(_matchups([("Jon Jones", "Ghost Fighter")]), db_path=history_db)
    assert len(res["features"]) == 0
    assert res["report"]["n_failed_rows"] == 1
    assert "Ghost Fighter" in res["report"]["unmatched_fighters"]
    assert any("unmatched" in r for f in res["report"]["failed_rows"] for r in f["reasons"])


def test_bad_event_date_is_failed_row_not_exception(history_db):
    df = _matchups([("Jon Jones", "Stipe Miocic")])
    df.loc[0, "event_date"] = "not-a-date"

    res = build_upcoming_features(df, db_path=history_db)

    assert len(res["features"]) == 0
    assert res["report"]["n_failed_rows"] == 1
    assert any("unparseable event_date" in r for r in res["report"]["failed_rows"][0]["reasons"])


def test_ambiguous_fighter_is_failed_row(history_db):
    res = build_upcoming_features(_matchups([("Jon Jones", "Bruno Silva")]), db_path=history_db)
    assert len(res["features"]) == 0
    assert res["report"]["n_failed_rows"] == 1
    assert "Bruno Silva" in res["report"]["ambiguous_fighters"]


def test_report_schema(history_db):
    res = build_upcoming_features(_matchups([("Jon Jones", "Stipe Miocic")]), db_path=history_db)
    rep = res["report"]
    for key in ("generated_at", "history_input", "n_matchup_rows", "n_feature_rows_built", "n_failed_rows",
                "failed_rows", "unmatched_fighters", "ambiguous_fighters", "match_review", "time_safety",
                "low_history_matchups", "feature_columns_generated", "missing_feature_summary",
                "step6b_validation", "data_quality_warnings", "interpretation_notes"):
        assert key in rep, f"{key} missing from report"
    assert rep["changes_data_processed"] is False


# ---------------------------------------------------------------------------
# CLI run_build
# ---------------------------------------------------------------------------

def test_run_build_writes_output_and_report_not_data_processed(history_db, tmp_path):
    matchups = tmp_path / "matchups.csv"
    _matchups([("Jon Jones", "Stipe Miocic"), ("Alex Pereira", "Israel Adesanya")]).to_csv(matchups, index=False)
    output = tmp_path / "data" / "live" / "features.csv"
    report_dir = tmp_path / "reports" / "live"
    report = run_build(str(matchups), output=str(output), history_input=history_db,
                       output_dir=str(report_dir), overwrite=True, validate_for_step6b=True)
    assert output.exists()
    built = pd.read_csv(output)
    assert len(built) == 2 and set(BASE_NUMERIC).issubset(built.columns)
    assert not (FORBIDDEN_OUTPUT_COLUMNS & set(built.columns))
    assert Path(report["report_json"]).exists() and Path(report["report_md"]).exists()
    assert "run_step6b_live_predictions" in report["next_command"]
    # output path is under the temp tree, never data/processed/
    assert "data/processed" not in str(output)


def test_run_build_no_overwrite_guard(history_db, tmp_path):
    matchups = tmp_path / "m.csv"
    _matchups([("Jon Jones", "Stipe Miocic")]).to_csv(matchups, index=False)
    output = tmp_path / "features.csv"
    run_build(str(matchups), output=str(output), history_input=history_db,
              output_dir=str(tmp_path / "rep"), overwrite=True)
    with pytest.raises(FileExistsError):
        run_build(str(matchups), output=str(output), history_input=history_db,
                  output_dir=str(tmp_path / "rep"), overwrite=False)


def test_built_features_feed_step6b_validation_directly(history_db, tmp_path):
    """The whole point: Step 6C output is a valid Step 6B input."""
    matchups = tmp_path / "m.csv"
    _matchups([("Jon Jones", "Stipe Miocic")]).to_csv(matchups, index=False)
    output = tmp_path / "features.csv"
    run_build(str(matchups), output=str(output), history_input=history_db,
              output_dir=str(tmp_path / "rep"), overwrite=True)
    built = pd.read_csv(output)
    v = validate_prediction_input(built, BASE_NUMERIC)
    assert v["ok"] is True
    assert "winner" not in built.columns and "fighter_a_won" not in built.columns


def test_layoff_historical_live_parity_and_candidate_schema(history_db):
    from ufc_pipeline.elo import run_elo
    from ufc_pipeline.features import build_feature_rows
    from ufc_pipeline.layoff_features import LAYOFF_CANDIDATE_B_FEATURES
    from ufc_pipeline.step6c_upcoming_feature_builder import load_history, model_features_for_set

    matchups = _matchups([("Jon Jones", "Stipe Miocic")], event_date="2023-06-01")
    live = build_upcoming_features(matchups, db_path=history_db, feature_set="layoff_b")["features"].iloc[0]

    history = load_history(history_db)
    real_before = [f for f in history["fights"] if str(f["date"]) < "2023-06-01"]
    _snapshots, ratings = run_elo(real_before)
    direct_fight = {
        "fight_id": -99, "date": "2023-06-01", "event": "UFC Test",
        "fighter_a_id": 1, "fighter_b_id": 2, "fighter_a": "Jon Jones", "fighter_b": "Stipe Miocic",
        "winner": None, "fighter_a_won": 0, "weight_class": "Lightweight",
        "fighter_a_pre_elo": ratings[1], "fighter_b_pre_elo": ratings[2],
        "fighter_a_expected_win_prob": 0.5, "fighter_b_expected_win_prob": 0.5,
        "_synthetic": True,
    }
    historical_equivalent = build_feature_rows(real_before + [direct_fight])
    direct = historical_equivalent[historical_equivalent["fight_id"] == -99].iloc[0]
    for column in LAYOFF_CANDIDATE_B_FEATURES:
        if pd.isna(direct[column]):
            assert pd.isna(live[column])
        else:
            assert live[column] == pytest.approx(direct[column])
    assert list(live.index[5:]) == model_features_for_set("layoff_b")
