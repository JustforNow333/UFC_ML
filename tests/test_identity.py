"""Tests for the fighter identity layer (Part A).

Covers the 10 required behaviors:
 1. Two fighters with the same normalized name but different Greco URLs
    do not merge.
 2. The same Greco URL always maps to the same fighter_id.
 3. A new Greco URL with a unique normalized-name match links correctly.
 4. Ambiguous normalized-name matches are reported instead of guessed.
 5. Name normalization (whitespace/case) still works when safe.
 6. Existing fighter_id values remain stable across re-registration.
 7. fight_stats rows link to the correct (split) fighter_id.
 8. Ingestion uses URL/source identity before name fallback.
 9. Unresolved identity cases do not crash the pipeline.
10. Identity reports are written and include unresolved/ambiguous cases.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufc_pipeline.db import connect, init_schema  # noqa: E402
from ufc_pipeline.identity import (  # noqa: E402
    SourceIdentity,
    assign_bouts_to_identities,
    division_weight_lbs,
    load_greco_identities,
    register_greco_identities,
    url_slug,
    write_identity_report,
)
from ufc_pipeline.ingest import ingest_csv  # noqa: E402
from ufc_pipeline.stats_ingest import ingest_stats_csv  # noqa: E402

URL = "http://ufcstats.com/fighter-details/"


def make_tott_csv(path: Path, rows: list[dict]) -> Path:
    defaults = {"HEIGHT": "5' 11\"", "WEIGHT": "155 lbs.", "REACH": '72"',
                "STANCE": "Orthodox", "DOB": "Jan 01, 1990"}
    df = pd.DataFrame([{**defaults, **r} for r in rows])
    df = df[["FIGHTER", "HEIGHT", "WEIGHT", "REACH", "STANCE", "DOB", "URL"]]
    df.to_csv(path, index=False)
    return path


def make_fights_csv(path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def register(tmp_path: Path, db: Path, tott_rows: list[dict]):
    tott = make_tott_csv(tmp_path / "tott.csv", tott_rows)
    identities = load_greco_identities(str(tott))
    conn = connect(db)
    try:
        init_schema(conn)
        report = register_greco_identities(conn, identities)
    finally:
        conn.close()
    return report


# Same-name pair mirroring the real Bruno Silva case.
BRUNO_TOTT = [
    {"FIGHTER": "Bruno Silva", "WEIGHT": "125 lbs.", "DOB": "Mar 16, 1990",
     "URL": URL + "aaa111"},
    {"FIGHTER": "Bruno Silva", "WEIGHT": "185 lbs.", "DOB": "Jul 13, 1989",
     "URL": URL + "bbb222"},
]
BRUNO_FIGHTS = [
    {"date": "2020-01-11", "event": "UFC 1", "red_fighter": "Bruno Silva",
     "blue_fighter": "Opp One", "winner": "Bruno Silva",
     "weight_class": "Flyweight"},
    {"date": "2020-06-20", "event": "UFC 2", "red_fighter": "Bruno Silva",
     "blue_fighter": "Opp Two", "winner": "Opp Two",
     "weight_class": "Middleweight"},
    {"date": "2021-03-13", "event": "UFC 3", "red_fighter": "Opp Three",
     "blue_fighter": "Bruno Silva", "winner": "Bruno Silva",
     "weight_class": "Flyweight"},
]


# --------------------------------------------------------------------- 1
def test_same_name_different_urls_do_not_merge(tmp_path):
    db = tmp_path / "ufc.db"
    register(tmp_path, db, BRUNO_TOTT)
    fights = make_fights_csv(tmp_path / "fights.csv", BRUNO_FIGHTS)
    report = ingest_csv(str(fights), str(db), source="greco")
    assert report.ingested == 3

    conn = connect(db)
    brunos = conn.execute(
        "SELECT fighter_id FROM fighters WHERE normalized_name = 'bruno silva'"
    ).fetchall()
    assert len(brunos) == 2, "flyweight and middleweight Bruno must be two fighters"

    # each identity linked to its own fighter, and fights partitioned by division
    links = dict(conn.execute(
        "SELECT source_fighter_id, fighter_id FROM fighter_source_ids "
        "WHERE fighter_id IS NOT NULL"
    ))
    assert links["aaa111"] != links["bbb222"]
    fly = conn.execute(
        "SELECT COUNT(*) FROM fights WHERE (fighter_a_id = ? OR fighter_b_id = ?)",
        (links["aaa111"], links["aaa111"]),
    ).fetchone()[0]
    mid = conn.execute(
        "SELECT COUNT(*) FROM fights WHERE (fighter_a_id = ? OR fighter_b_id = ?)",
        (links["bbb222"], links["bbb222"]),
    ).fetchone()[0]
    assert (fly, mid) == (2, 1)
    conn.close()


# --------------------------------------------------------------------- 2
def test_same_url_always_same_fighter_id(tmp_path):
    db = tmp_path / "ufc.db"
    register(tmp_path, db, BRUNO_TOTT)
    fights = make_fights_csv(tmp_path / "fights.csv", BRUNO_FIGHTS)
    ingest_csv(str(fights), str(db), source="greco")

    conn = connect(db)
    fid_before = conn.execute(
        "SELECT fighter_id FROM fighter_source_ids WHERE source_fighter_id='aaa111'"
    ).fetchone()[0]
    conn.close()

    # re-register + ingest MORE fights for the same URL identity
    register(tmp_path, db, BRUNO_TOTT)
    more = make_fights_csv(tmp_path / "more.csv", [
        {"date": "2022-05-07", "event": "UFC 4", "red_fighter": "Bruno Silva",
         "blue_fighter": "Opp Four", "winner": "Opp Four",
         "weight_class": "Flyweight"},
    ])
    ingest_csv(str(more), str(db), source="greco")

    conn = connect(db)
    fid_after = conn.execute(
        "SELECT fighter_id FROM fighter_source_ids WHERE source_fighter_id='aaa111'"
    ).fetchone()[0]
    new_fight_fighters = conn.execute(
        "SELECT fighter_a_id FROM fights WHERE date='2022-05-07'"
    ).fetchone()
    conn.close()
    assert fid_before == fid_after
    assert new_fight_fighters[0] == fid_before


# --------------------------------------------------------------------- 3
def test_new_url_unique_name_links_to_existing_fighter(tmp_path):
    db = tmp_path / "ufc.db"
    # fighter exists first (ingested without any identity registered)
    fights = make_fights_csv(tmp_path / "fights.csv", [
        {"date": "2019-02-02", "event": "UFC 0", "red_fighter": "Max Holloway",
         "blue_fighter": "Some Guy", "winner": "Max Holloway",
         "weight_class": "Featherweight"},
    ])
    ingest_csv(str(fights), str(db), source="greco")
    conn = connect(db)
    fid = conn.execute(
        "SELECT fighter_id FROM fighters WHERE normalized_name='max holloway'"
    ).fetchone()[0]
    conn.close()

    report = register(tmp_path, db, [
        {"FIGHTER": "Max Holloway", "WEIGHT": "145 lbs.", "URL": URL + "max001"},
    ])
    assert report.newly_linked_by_name == 1
    conn = connect(db)
    linked = conn.execute(
        "SELECT fighter_id, resolution FROM fighter_source_ids "
        "WHERE source_fighter_id='max001'"
    ).fetchone()
    conn.close()
    assert linked == (fid, "linked")


# --------------------------------------------------------------------- 4
def test_ambiguous_name_reported_not_guessed(tmp_path):
    db = tmp_path / "ufc.db"
    report = register(tmp_path, db, BRUNO_TOTT)
    assert report.ambiguous_same_name == 2
    assert "bruno silva" in report.duplicate_normalized_names
    assert any(c["normalized_name"] == "bruno silva"
               for c in report.same_name_not_merged)
    conn = connect(db)
    resolutions = [r for (r,) in conn.execute(
        "SELECT resolution FROM fighter_source_ids"
    )]
    conn.close()
    assert resolutions == ["ambiguous", "ambiguous"]  # neither linked by name


def test_catchweight_same_name_bout_is_unresolved(tmp_path):
    """A same-name bout with NO usable evidence (catch weight, both DOBs
    plausible, no nearby assigned bouts) must be reported, not guessed."""
    db = tmp_path / "ufc.db"
    register(tmp_path, db, [
        {"FIGHTER": "Sam Same", "WEIGHT": "155 lbs.", "DOB": "Jan 01, 1990",
         "URL": URL + "s1"},
        {"FIGHTER": "Sam Same", "WEIGHT": "170 lbs.", "DOB": "Feb 02, 1991",
         "URL": URL + "s2"},
    ])
    fights = make_fights_csv(tmp_path / "fights.csv", [
        {"date": "2020-01-01", "event": "UFC X", "red_fighter": "Sam Same",
         "blue_fighter": "Other Guy", "winner": "Sam Same",
         "weight_class": "Catch Weight"},
    ])
    report = ingest_csv(str(fights), str(db), source="greco")  # must not crash
    assert report.ingested == 1
    assert len(report.identity_unresolved) == 1
    assert report.identity_unresolved[0]["name"] == "sam same"
    # the fallback fighter is NOT linked to either identity
    conn = connect(db)
    linked = conn.execute(
        "SELECT COUNT(*) FROM fighter_source_ids WHERE fighter_id IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    assert linked == 0


# --------------------------------------------------------------------- 5
def test_name_normalization_still_safe(tmp_path):
    db = tmp_path / "ufc.db"
    register(tmp_path, db, [
        {"FIGHTER": "  JosE   Aldo ", "URL": URL + "aldo1"},
    ])
    fights = make_fights_csv(tmp_path / "fights.csv", [
        {"date": "2015-12-12", "event": "UFC 194", "red_fighter": "Jose Aldo",
         "blue_fighter": "Conor McGregor", "winner": "Conor McGregor",
         "weight_class": "Featherweight"},
    ])
    ingest_csv(str(fights), str(db), source="greco")
    conn = connect(db)
    # whitespace-collapsed, case-insensitive match linked the identity
    row = conn.execute(
        "SELECT f.normalized_name FROM fighter_source_ids s "
        "JOIN fighters f ON f.fighter_id = s.fighter_id "
        "WHERE s.source_fighter_id = 'aldo1'"
    ).fetchone()
    n_aldos = conn.execute(
        "SELECT COUNT(*) FROM fighters WHERE normalized_name='jose aldo'"
    ).fetchone()[0]
    conn.close()
    assert row == ("jose aldo",)
    assert n_aldos == 1  # no duplicate canonical fighter created


# --------------------------------------------------------------------- 6
def test_existing_fighter_ids_stable_across_reregistration(tmp_path):
    db = tmp_path / "ufc.db"
    register(tmp_path, db, BRUNO_TOTT)
    fights = make_fights_csv(tmp_path / "fights.csv", BRUNO_FIGHTS)
    ingest_csv(str(fights), str(db), source="greco")
    conn = connect(db)
    before = conn.execute(
        "SELECT fighter_id, normalized_name FROM fighters ORDER BY fighter_id"
    ).fetchall()
    links_before = conn.execute(
        "SELECT source_fighter_id, fighter_id FROM fighter_source_ids "
        "ORDER BY source_fighter_id"
    ).fetchall()
    conn.close()

    register(tmp_path, db, BRUNO_TOTT)  # idempotent re-registration

    conn = connect(db)
    after = conn.execute(
        "SELECT fighter_id, normalized_name FROM fighters ORDER BY fighter_id"
    ).fetchall()
    links_after = conn.execute(
        "SELECT source_fighter_id, fighter_id FROM fighter_source_ids "
        "ORDER BY source_fighter_id"
    ).fetchall()
    conn.close()
    assert before == after
    assert links_before == links_after


# --------------------------------------------------------------------- 7
def test_fight_stats_link_to_correct_split_fighter(tmp_path):
    db = tmp_path / "ufc.db"
    register(tmp_path, db, BRUNO_TOTT)
    fights = make_fights_csv(tmp_path / "fights.csv", BRUNO_FIGHTS)
    ingest_csv(str(fights), str(db), source="greco")

    stats = pd.DataFrame([
        {"date": "2020-01-11", "red_fighter": "Bruno Silva",
         "blue_fighter": "Opp One", "total_minutes": 15.0,
         "red_kd": 1, "blue_kd": 0,
         "red_sig_str_landed": 50, "blue_sig_str_landed": 30,
         "red_sig_str_attempted": 90, "blue_sig_str_attempted": 70,
         "red_td_landed": 2, "blue_td_landed": 0,
         "red_td_attempted": 4, "blue_td_attempted": 1,
         "red_sub_att": 0, "blue_sub_att": 1,
         "red_ctrl_seconds": 120, "blue_ctrl_seconds": 30},
    ])
    stats_csv = tmp_path / "stats.csv"
    stats.to_csv(stats_csv, index=False)
    report = ingest_stats_csv(str(stats_csv), str(db))
    assert report.matched == 1

    conn = connect(db)
    flyweight_bruno = conn.execute(
        "SELECT fighter_id FROM fighter_source_ids WHERE source_fighter_id='aaa111'"
    ).fetchone()[0]
    stat_fighters = [f for (f,) in conn.execute(
        """SELECT fighter_id FROM fight_stats fs
           JOIN fights f ON f.fight_id = fs.fight_id WHERE f.date='2020-01-11'"""
    )]
    conn.close()
    # the 2020-01-11 flyweight bout's stats belong to the FLYWEIGHT Bruno
    assert flyweight_bruno in stat_fighters


# --------------------------------------------------------------------- 8
def test_url_identity_used_before_name_fallback(tmp_path):
    db = tmp_path / "ufc.db"
    register(tmp_path, db, [
        {"FIGHTER": "Solo Fighter", "WEIGHT": "155 lbs.", "URL": URL + "solo1"},
    ])
    fights = make_fights_csv(tmp_path / "fights.csv", [
        {"date": "2020-01-01", "event": "UFC X", "red_fighter": "Solo Fighter",
         "blue_fighter": "No Identity Guy", "winner": "Solo Fighter",
         "weight_class": "Lightweight"},
    ])
    report = ingest_csv(str(fights), str(db), source="greco")
    # exactly one slot (Solo Fighter) resolved via identity; the opponent
    # has no registered identity and used the name path
    assert report.identity_resolved_slots == 1
    assert report.identity_new_links == 1
    conn = connect(db)
    linked = conn.execute(
        "SELECT fighter_id FROM fighter_source_ids WHERE source_fighter_id='solo1'"
    ).fetchone()[0]
    assert linked is not None
    # second ingest run: the URL link is reused, no new fighter created
    more = make_fights_csv(tmp_path / "more.csv", [
        {"date": "2021-01-01", "event": "UFC Y", "red_fighter": "Solo Fighter",
         "blue_fighter": "Another Guy", "winner": "Another Guy",
         "weight_class": "Lightweight"},
    ])
    ingest_csv(str(more), str(db), source="greco")
    n = conn.execute(
        "SELECT COUNT(*) FROM fighters WHERE normalized_name='solo fighter'"
    ).fetchone()[0]
    conn.close()
    assert n == 1


# --------------------------------------------------------------------- 9
def test_unresolved_identities_do_not_crash_full_flow(tmp_path):
    db = tmp_path / "ufc.db"
    register(tmp_path, db, [
        {"FIGHTER": "Sam Same", "WEIGHT": "--", "DOB": "--", "URL": URL + "s1"},
        {"FIGHTER": "Sam Same", "WEIGHT": "--", "DOB": "--", "URL": URL + "s2"},
    ])
    fights = make_fights_csv(tmp_path / "fights.csv", [
        {"date": "2020-01-01", "event": "UFC X", "red_fighter": "Sam Same",
         "blue_fighter": "Clean Name", "winner": "Sam Same",
         "weight_class": "Catch Weight"},
    ])
    report = ingest_csv(str(fights), str(db), source="greco")
    assert report.ingested == 1
    assert len(report.identity_unresolved) == 1
    # stats for the unresolved bout still ingest (fight exists via fallback)
    stats = pd.DataFrame([{
        "date": "2020-01-01", "red_fighter": "Sam Same",
        "blue_fighter": "Clean Name", "total_minutes": 15.0,
        "red_kd": 0, "blue_kd": 0,
        "red_sig_str_landed": 10, "blue_sig_str_landed": 5,
        "red_sig_str_attempted": 20, "blue_sig_str_attempted": 15,
        "red_td_landed": 0, "blue_td_landed": 0,
        "red_td_attempted": 0, "blue_td_attempted": 0,
        "red_sub_att": 0, "blue_sub_att": 0,
        "red_ctrl_seconds": 0, "blue_ctrl_seconds": 0,
    }])
    stats_csv = tmp_path / "stats.csv"
    stats.to_csv(stats_csv, index=False)
    sreport = ingest_stats_csv(str(stats_csv), str(db))
    assert sreport.matched == 1


# -------------------------------------------------------------------- 10
def test_identity_report_written_with_unresolved_and_ambiguous(tmp_path):
    db = tmp_path / "ufc.db"
    report = register(tmp_path, db, BRUNO_TOTT + [
        {"FIGHTER": "Unique Guy", "URL": URL + "u1"},
    ])
    out = tmp_path / "identity_report.json"
    write_identity_report(report, str(out))
    data = json.loads(out.read_text())
    assert data["total_identities"] == 3
    assert data["ambiguous_same_name"] == 2
    assert data["duplicate_normalized_names"] == ["bruno silva"]
    assert data["same_name_not_merged"][0]["normalized_name"] == "bruno silva"
    for key in ("already_linked", "newly_linked_by_name", "unlinked_not_in_db",
                "linked_examples", "unresolved_examples"):
        assert key in data


# ------------------------------------------------------------ unit extras
def test_url_slug_and_division_weight():
    assert url_slug("http://ufcstats.com/fighter-details/93fe7332d16c6ad9") \
        == "93fe7332d16c6ad9"
    assert url_slug("http://ufcstats.com/fighter-details/abc/") == "abc"
    assert url_slug(None) is None and url_slug("") is None
    assert division_weight_lbs("Flyweight") == 125.0
    assert division_weight_lbs("Women's Flyweight") == 125.0
    assert division_weight_lbs("Catch Weight") is None
    assert division_weight_lbs(None) is None


def test_continuity_evidence_resolves_one_off_division_move():
    """Mirrors the real Jean Silva case: a featherweight's one-off
    lightweight bout must go to the ACTIVE fighter (career continuity),
    not to the candidate whose listed weight is numerically closer."""
    cands = [
        SourceIdentity(slug="old", url=None, raw_name="Jean Silva",
                       key="jean silva", weight_lbs=160.0, dob="1977-10-08",
                       height_cm=None, reach_cm=None),
        SourceIdentity(slug="lord", url=None, raw_name="Jean Silva",
                       key="jean silva", weight_lbs=145.0, dob="1996-12-27",
                       height_cm=None, reach_cm=None),
    ]
    bouts = [
        ("b1", "2024-01-13", "Featherweight"),
        ("b2", "2024-06-29", "Featherweight"),
        ("b3", "2024-07-13", "Lightweight"),   # the one-off move up
        ("b4", "2025-02-22", "Featherweight"),
    ]
    result = assign_bouts_to_identities(bouts, cands)
    assert result == {"b1": "lord", "b2": "lord", "b3": "lord", "b4": "lord"}


def test_dob_plausibility_used_when_weight_and_continuity_fail():
    cands = [
        SourceIdentity(slug="young", url=None, raw_name="X Y", key="x y",
                       weight_lbs=None, dob="1995-01-01",
                       height_cm=None, reach_cm=None),
        SourceIdentity(slug="ancient", url=None, raw_name="X Y", key="x y",
                       weight_lbs=None, dob="1950-01-01",
                       height_cm=None, reach_cm=None),
    ]
    result = assign_bouts_to_identities([("b1", "2020-01-01", None)], cands)
    assert result == {"b1": "young"}  # the 1950 fighter would be 70


def test_legacy_db_migration_drops_unique_constraint(tmp_path):
    """A database created with the old UNIQUE(normalized_name) schema must
    migrate in place, preserving fighter_id values."""
    import sqlite3

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """CREATE TABLE fighters (
               fighter_id      INTEGER PRIMARY KEY AUTOINCREMENT,
               name            TEXT NOT NULL,
               normalized_name TEXT NOT NULL UNIQUE,
               height TEXT, reach TEXT, stance TEXT, date_of_birth TEXT,
               created_at TEXT DEFAULT (datetime('now')),
               updated_at TEXT DEFAULT (datetime('now')));
           INSERT INTO fighters (name, normalized_name)
               VALUES ('A B', 'a b'), ('C D', 'c d');"""
    )
    conn.commit()
    conn.close()

    conn = connect(db)
    init_schema(conn)
    ids = conn.execute(
        "SELECT fighter_id, normalized_name FROM fighters ORDER BY fighter_id"
    ).fetchall()
    assert ids == [(1, "a b"), (2, "c d")]
    # duplicate normalized names are now allowed (the whole point)
    conn.execute("INSERT INTO fighters (name, normalized_name) VALUES ('A B','a b')")
    conn.commit()
    n = conn.execute(
        "SELECT COUNT(*) FROM fighters WHERE normalized_name='a b'"
    ).fetchone()[0]
    conn.close()
    assert n == 2
