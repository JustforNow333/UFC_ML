"""Tests for Step 6E controlled processed-feature rebuild.

Covers the non-destructive guard (never overwrite official processed files,
never clobber a prior rebuild without opt-in), schema-parity comparison
(including catching a missing model feature), new-event verification on a small
fixture, future-dated event exclusion, the Step 6B input compatibility check,
report schema, DB-unchanged assertion, and an end-to-end rebuild on a tiny
synthetic SQLite DB. No real card, network, or official file is touched.
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
from ufc_pipeline.matchup_features import build_step3c_for_db  # noqa: E402
from ufc_pipeline.step6e_rebuild_processed import (  # noqa: E402
    OFFICIAL_PROCESSED_PATHS,
    assert_non_destructive,
    build_step6b_style_input,
    check_step6b_validation,
    compare_schema,
    feature_quality_checks,
    file_md5,
    official_model_compatibility,
    resolve_output_paths,
    run_rebuild,
    verify_new_events,
    write_reports,
)

BASE_NUMERIC, _ = official_step3c_features()

FIGHTERS = [
    (1, "Jon Jones", "185", "215", "1987-07-19"),
    (2, "Stipe Miocic", "193", "203", "1982-08-19"),
    (3, "Alex Pereira", "193", "200", "1987-07-07"),
    (4, "Israel Adesanya", "193", "203", "1989-07-22"),
    (5, "Charles Oliveira", "178", "188", "1989-10-17"),
    (6, "Islam Makhachev", "178", "180", "1991-10-27"),
]
# (fight_id, date, event, a_id, b_id, a_won)
FIGHTS = [
    (1, "2020-01-01", "OLD 1", 1, 2, 1),
    (2, "2020-06-01", "OLD 2", 3, 4, 1),
    (3, "2021-01-01", "OLD 3", 1, 3, 1),
    (4, "2021-06-01", "OLD 4", 2, 4, 1),
    (5, "2022-01-01", "OLD 5", 5, 6, 1),
    (6, "2022-06-01", "OLD 6", 1, 4, 0),
    # NEW CARD (2 fights) — stands in for the "newly applied" event
    (7, "2023-06-01", "NEW CARD", 2, 3, 1),
    (8, "2023-06-01", "NEW CARD", 5, 4, 0),
]
NEW_EVENT = ("2023-06-01", "NEW CARD", 2)
FUTURE_EVENT = ("2099-12-31", "FUTURE 999")


@pytest.fixture(scope="module")
def history_db(tmp_path_factory):
    db = tmp_path_factory.mktemp("step6e_db") / "hist.db"
    conn = sqlite3.connect(db)
    init_schema(conn)
    conn.executemany(
        "INSERT INTO fighters (fighter_id, name, normalized_name, height, reach, stance, date_of_birth) "
        "VALUES (?,?,?,?,?,?,?)",
        [(fid, name, name.lower(), h, r, "Orthodox", dob) for fid, name, h, r, dob in FIGHTERS],
    )
    for fid, date, event, a, b, a_won in FIGHTS:
        winner = a if a_won == 1 else b
        conn.execute(
            "INSERT INTO fights (fight_id, date, event, fighter_a_id, fighter_b_id, winner_id, fighter_a_won, "
            "weight_class, method, round, source, source_row_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (fid, date, event, a, b, winner, a_won, "Lightweight", "Decision", 3, "test", fid),
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


@pytest.fixture(scope="module")
def full_step3c(history_db, tmp_path_factory):
    """Rebuild the full Step 3C once; use it as the 'current' reference."""
    out = tmp_path_factory.mktemp("step6e_full") / "full_step3c.csv"
    build_step3c_for_db(history_db, str(out))
    return pd.read_csv(out)


@pytest.fixture()
def old_official(full_step3c, tmp_path):
    """A pre-update 'official' Step 3C = full minus the NEW CARD rows."""
    old = full_step3c[full_step3c["date"] != NEW_EVENT[0]].copy()
    path = tmp_path / "old_official_step3c.csv"
    old.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Non-destructive guard
# --------------------------------------------------------------------------- #

def test_resolve_output_paths_uses_official_basenames(tmp_path):
    paths = resolve_output_paths(tmp_path / "rebuild")
    assert Path(paths["step3c"]).name == "ufc_prefight_features_step3c.csv"
    assert str(tmp_path) in paths["step3c"]


def test_guard_rejects_official_output_dir():
    # Pointing the rebuild at data/processed would collide with the official files.
    official_paths = resolve_output_paths("data/processed")
    assert Path(official_paths["step3c"]).resolve() == Path(OFFICIAL_PROCESSED_PATHS["step3c"]).resolve()
    with pytest.raises(ValueError, match="official processed file"):
        assert_non_destructive(official_paths, overwrite_rebuild=False)


def test_guard_refuses_to_clobber_existing_rebuild(tmp_path):
    paths = resolve_output_paths(tmp_path / "rebuild")
    Path(paths["step3c"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["step3c"]).write_text("stale")
    with pytest.raises(ValueError, match="already exist"):
        assert_non_destructive(paths, overwrite_rebuild=False)
    # opt-in allows replacement
    assert_non_destructive(paths, overwrite_rebuild=True)


# --------------------------------------------------------------------------- #
# Schema parity
# --------------------------------------------------------------------------- #

def test_compare_schema_happy_path(full_step3c):
    result = compare_schema(full_step3c, full_step3c, BASE_NUMERIC)
    assert result["row_count_delta"] == 0
    assert result["columns_identical_and_ordered"] is True
    assert result["all_official_features_present"] is True
    assert result["unexpected_new_columns"] == []
    assert result["raw_weight_class_handling_consistent"] is True


def test_compare_schema_catches_missing_feature(full_step3c):
    dropped = BASE_NUMERIC[0]
    broken = full_step3c.drop(columns=[dropped])
    result = compare_schema(full_step3c, broken, BASE_NUMERIC)
    assert result["all_official_features_present"] is False
    assert dropped in result["missing_required_feature_columns"]
    assert dropped in result["columns_only_in_old"]


def test_compare_schema_reports_row_delta(full_step3c):
    old = full_step3c[full_step3c["date"] != NEW_EVENT[0]]
    result = compare_schema(old, full_step3c, BASE_NUMERIC)
    assert result["row_count_delta"] == NEW_EVENT[2]
    assert len(result["new_fight_ids_not_in_old"]) == NEW_EVENT[2]
    assert result["old_fight_ids_not_in_new"] == []


# --------------------------------------------------------------------------- #
# New-event verification + future exclusion
# --------------------------------------------------------------------------- #

def test_verify_new_events_present(full_step3c):
    result = verify_new_events(full_step3c, BASE_NUMERIC, [NEW_EVENT], [FUTURE_EVENT])
    assert result["all_expected_events_present"] is True
    assert result["all_excluded_events_absent"] is True
    assert result["new_event_duplicate_fight_ids"] == 0
    assert result["new_event_duplicate_pair_date_rows"] == 0
    assert result["leakage_guard_passes"] is True
    assert result["model_features_containing_result_columns"] == []


def test_verify_new_events_wrong_count(full_step3c):
    result = verify_new_events(full_step3c, BASE_NUMERIC, [("2023-06-01", "NEW CARD", 3)], [FUTURE_EVENT])
    assert result["all_expected_events_present"] is False
    assert result["expected_events"][0]["found_fights"] == 2


def test_future_event_exclusion_detects_leak(full_step3c):
    # Pretend the NEW CARD date is actually a "future" card that must be absent.
    result = verify_new_events(full_step3c, BASE_NUMERIC, [], [("2023-06-01", "NEW CARD")])
    assert result["all_excluded_events_absent"] is False
    assert result["excluded_events"][0]["absent"] is False


# --------------------------------------------------------------------------- #
# Feature quality + Step 6B input
# --------------------------------------------------------------------------- #

def test_feature_quality_no_dupes_no_inf(full_step3c):
    q = feature_quality_checks(full_step3c, BASE_NUMERIC, {NEW_EVENT[0]})
    assert q["infinite_value_count_in_model_features"] == 0
    assert q["duplicate_fight_ids"] == 0
    assert q["duplicate_pair_date_rows"] == 0
    assert q["date_ordered_by_date_fight_id"] is True
    assert set(q["target_distribution"]).issubset({"0", "1"})


def test_step6b_input_has_no_forbidden_columns_and_validates(full_step3c):
    new_rows = full_step3c[full_step3c["date"] == NEW_EVENT[0]]
    frame = build_step6b_style_input(new_rows, BASE_NUMERIC)
    assert "fighter_a_won" not in frame.columns
    assert "winner" not in frame.columns
    result = check_step6b_validation(new_rows, BASE_NUMERIC)
    assert result["ok"] is True
    assert result["n_valid_rows"] == NEW_EVENT[2]
    assert result["forbidden_result_columns"] == []


def test_official_model_compat_skips_when_csv_missing(full_step3c, tmp_path):
    new_rows = full_step3c[full_step3c["date"] == NEW_EVENT[0]]
    result = official_model_compatibility(str(tmp_path / "nope.csv"), new_rows, BASE_NUMERIC)
    assert result["status"] == "skipped"


# --------------------------------------------------------------------------- #
# End-to-end rebuild + report schema
# --------------------------------------------------------------------------- #

def test_run_rebuild_writes_new_path_only_and_preserves_official(history_db, old_official, tmp_path):
    out_dir = tmp_path / "rebuild"
    reports_dir = tmp_path / "reports"
    official_md5_before = file_md5(old_official)

    report = run_rebuild(
        db_path=history_db,
        output_dir=str(out_dir),
        reports_dir=str(reports_dir),
        official_step3c=str(old_official),
        run_compat=False,
        expected_events=[NEW_EVENT],
        excluded_events=[FUTURE_EVENT],
    )

    # rebuilt files land in the new dir only
    assert (out_dir / "ufc_prefight_features_step3c.csv").exists()
    assert report["rebuilt_row_counts"]["step3c"] == len(pd.read_csv(old_official)) + NEW_EVENT[2]

    # the comparison file (our stand-in official) is byte-for-byte unchanged
    assert file_md5(old_official) == official_md5_before

    # verification results
    assert report["db_unchanged_by_rebuild"] is True
    assert report["schema_comparison"]["row_count_delta"] == NEW_EVENT[2]
    assert report["schema_comparison"]["new_max_date"] == NEW_EVENT[0]
    assert report["new_event_verification"]["all_expected_events_present"] is True
    assert report["new_event_verification"]["all_excluded_events_absent"] is True
    assert report["step6b_input_validation"]["ok"] is True
    assert report["official_processed_files_overwritten"] is False
    assert report["benchmark_or_model_modified"] is False
    assert report["overall_ok"] is True


def test_run_rebuild_refuses_official_output_dir(history_db, old_official, tmp_path):
    with pytest.raises(ValueError, match="official processed file"):
        run_rebuild(
            db_path=history_db,
            output_dir="data/processed",  # collides with official files -> guard aborts before any write
            reports_dir=str(tmp_path / "r"),
            official_step3c=str(old_official),
            run_compat=False,
            expected_events=[NEW_EVENT],
            excluded_events=[FUTURE_EVENT],
        )


def test_report_schema(history_db, old_official, tmp_path):
    report = run_rebuild(
        db_path=history_db,
        output_dir=str(tmp_path / "rb"),
        reports_dir=str(tmp_path / "rep"),
        official_step3c=str(old_official),
        run_compat=False,
        expected_events=[NEW_EVENT],
        excluded_events=[FUTURE_EVENT],
    )
    for key in (
        "step", "db_checksum_used", "db_unchanged_by_rebuild", "schema_comparison",
        "new_event_verification", "feature_quality", "step6b_input_validation",
        "official_model_compatibility", "statements", "overall_ok",
        "report_json", "report_md",
    ):
        assert key in report, f"missing report key: {key}"

    loaded = json.loads(Path(report["report_json"]).read_text())
    assert loaded["step"] == "6E"
    md = Path(report["report_md"]).read_text()
    assert "Step 6E" in md
    assert "Schema parity" in md


def test_write_reports_roundtrip(tmp_path):
    minimal = {
        "step": "6E", "title": "t", "generated_at": "now",
        "db_checksum_used": "abc", "db_unchanged_by_rebuild": True,
        "rebuild_output_dir": "d", "rebuilt_output_paths": {}, "rebuilt_row_counts": {},
        "official_processed_files_overwritten": False, "benchmark_or_model_modified": False,
        "schema_comparison": {
            "old_row_count": 1, "new_row_count": 2, "row_count_delta": 1,
            "old_max_date": "a", "new_max_date": "b", "old_column_count": 1, "new_column_count": 1,
            "columns_only_in_new": [], "columns_only_in_old": [], "columns_identical_and_ordered": True,
            "unexpected_new_columns": [], "all_official_features_present": True,
            "missing_required_feature_columns": [], "result_label_columns_present": [],
            "raw_weight_class_handling_consistent": True, "new_fight_ids_not_in_old": [1],
        },
        "new_event_verification": {
            "all_expected_events_present": True, "all_excluded_events_absent": True,
            "new_event_duplicate_fight_ids": 0, "new_event_duplicate_pair_date_rows": 0,
            "model_features_containing_result_columns": [], "leakage_guard_passes": True,
            "expected_events": [], "excluded_events": [],
        },
        "feature_quality": {
            "infinite_value_count_in_model_features": 0, "duplicate_fight_ids": 0,
            "duplicate_pair_date_rows": 0, "date_ordered_by_date_fight_id": True,
            "model_feature_max_nan_rate": 0.0, "target_distribution": {}, "model_feature_nan_rate_top10": {},
            "new_event_low_history_flag_counts": {}, "new_event_debut_fighter_rows": 0,
        },
        "step6b_input_validation": {"ok": True, "n_valid_rows": 0, "n_rows": 0, "n_error_rows": 0, "structural_errors": []},
        "official_model_compatibility": {"status": "skipped", "reason": "n/a"},
        "statements": ["ok"], "overall_ok": True,
    }
    json_path, md_path = write_reports(minimal, str(tmp_path / "rep"))
    assert json_path.exists() and md_path.exists()
    assert json.loads(json_path.read_text())["overall_ok"] is True
