"""Tests for the Greco central-database expansion.

Covers:
 - tott parsing (height feet'inches" -> cm, reach inches -> cm, DOB -> ISO)
 - fill-only provenance: existing fighter fields are never overwritten
 - Greco fights conversion: OUTCOME -> winner name; draws/NC -> no winner
 - canonical COALESCE rule: per-fight physicals win, static tott fills gaps
 - deterministic source selection is documented per field (audit doc exists)
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufc_pipeline.db import connect, init_schema  # noqa: E402
from ufc_pipeline.elo import build_elo_for_db  # noqa: E402
from ufc_pipeline.features import build_features_for_db  # noqa: E402
from ufc_pipeline.fighter_details import (  # noqa: E402
    ingest_fighter_tott,
    parse_dob_iso,
    parse_height_cm,
    parse_reach_cm,
)
from ufc_pipeline.ingest import ingest_csv  # noqa: E402
from ufc_pipeline.stats_ingest import convert_greco_fights  # noqa: E402


def test_tott_parsers():
    assert parse_height_cm("5' 11\"") == pytest.approx(180.3, abs=0.1)
    assert parse_height_cm("6' 0\"") == pytest.approx(182.9, abs=0.1)
    assert parse_height_cm("--") is None and parse_height_cm(None) is None
    assert parse_reach_cm('72"') == pytest.approx(182.9, abs=0.1)
    assert parse_reach_cm("--") is None
    assert parse_dob_iso("Jul 15, 1988") == "1988-07-15"
    assert parse_dob_iso("--") is None and parse_dob_iso("") is None
    assert parse_height_cm(pd.NA) is None
    assert parse_reach_cm(pd.NA) is None
    assert parse_dob_iso(pd.NA) is None


@pytest.fixture
def greco_style_db(tmp_path):
    fights_csv = tmp_path / "fights.csv"
    pd.DataFrame({
        "date": ["2020-01-01", "2020-06-01"],
        "red_fighter": ["Ann Ito", "Ann Ito"],
        "blue_fighter": ["Bo Rex", "Cy Dax"],
        "winner": ["Ann Ito", "Cy Dax"],
        # per-fight physicals present ONLY for the first fight:
        "red_height_cms": [175.0, None], "blue_height_cms": [180.0, None],
        "red_reach_cms": [178.0, None], "blue_reach_cms": [183.0, None],
    }).to_csv(fights_csv, index=False)
    db = tmp_path / "ufc.db"
    ingest_csv(str(fights_csv), str(db), "greco")
    build_elo_for_db(str(db), k=32, starting_elo=1500)
    return tmp_path, db


def test_tott_fill_only_never_overwrites(greco_style_db):
    tmp_path, db = greco_style_db
    conn = connect(db)
    init_schema(conn)
    conn.execute(
        "UPDATE fighters SET height = 199.9 WHERE normalized_name = 'ann ito'"
    )
    conn.commit()
    conn.close()

    tott = tmp_path / "tott.csv"
    pd.DataFrame({
        "FIGHTER": ["Ann Ito", "Bo Rex", "Unknown Person"],
        "HEIGHT": ["5' 6\"", "6' 1\"", "5' 9\""],
        "WEIGHT": ["145 lbs."] * 3,
        "REACH": ['66"', '75"', "--"],
        "STANCE": ["Orthodox", "Southpaw", ""],
        "DOB": ["Jan 02, 1992", "Mar 04, 1990", "--"],
        "URL": ["u"] * 3,
    }).to_csv(tott, index=False)
    report = ingest_fighter_tott(str(tott), str(db))
    assert report.matched == 2  # Unknown Person isn't in the DB

    conn = connect(db)
    rows = dict(conn.execute("SELECT normalized_name, height FROM fighters"))
    dobs = dict(conn.execute("SELECT normalized_name, date_of_birth FROM fighters"))
    conn.close()
    assert float(rows["ann ito"]) == pytest.approx(199.9)   # NOT overwritten
    assert float(rows["bo rex"]) == pytest.approx(185.4, abs=0.1)  # filled
    assert dobs["ann ito"] == "1992-01-02"                  # null -> filled
    assert report.skipped_existing >= 1


def test_convert_greco_fights_outcomes():
    events = pd.DataFrame({"EVENT": ["UFC 9"], "DATE": ["May 17, 1996"],
                           "URL": ["u"], "LOCATION": ["x"]})
    results = pd.DataFrame({
        "EVENT": ["UFC 9 "] * 4,
        "BOUT": ["Ann Ito vs. Bo Rex", "Cy Dax vs. Dee Oz",
                 "Eli Fox vs. Gia Ho", "Hal Ip vs. Ida Jo"],
        "OUTCOME": ["W/L", "L/W", "D/D", "NC/NC"],
        "WEIGHTCLASS": ["Lightweight Bout"] * 4,
        "METHOD": ["KO/TKO "] * 4, "ROUND": [1] * 4, "TIME": ["2:00"] * 4,
        "TIME FORMAT": ["3 Rnd (5-5-5)"] * 4, "REFEREE": ["r"] * 4,
        "DETAILS": [""] * 4, "URL": ["u"] * 4,
    })
    out = convert_greco_fights(results, events)
    assert len(out) == 4
    assert out.loc[0, "winner"] == "Ann Ito"   # W/L -> red (first-listed)
    assert out.loc[1, "winner"] == "Dee Oz"    # L/W -> blue
    assert out.loc[2, "winner"] == ""          # draw -> no winner
    assert out.loc[3, "winner"] == ""          # NC -> no winner
    assert (out["date"] == "1996-05-17").all()
    assert out.loc[0, "method"] == "KO/TKO"    # trailing space stripped


def test_coalesce_prefers_per_fight_over_static(greco_style_db, tmp_path):
    src_tmp, db = greco_style_db
    # Give the fighters static tott values that DIFFER from per-fight ones.
    tott = src_tmp / "tott.csv"
    pd.DataFrame({
        "FIGHTER": ["Ann Ito", "Bo Rex", "Cy Dax"],
        "HEIGHT": ["5' 0\"", "5' 0\"", "5' 0\""],       # 152.4 cm static
        "WEIGHT": ["145 lbs."] * 3,
        "REACH": ['60"', '60"', '60"'],                  # 152.4 cm static
        "STANCE": ["Orthodox"] * 3, "DOB": ["Jan 01, 1990"] * 3, "URL": ["u"] * 3,
    }).to_csv(tott, index=False)
    ingest_fighter_tott(str(tott), str(db))

    out = tmp_path / "features.csv"
    build_features_for_db(str(db), str(out))
    df = pd.read_csv(out).sort_values("fight_id").reset_index(drop=True)
    # Fight 1 HAS per-fight values -> they win over static 152.4:
    assert df.loc[0, "fighter_a_height"] == pytest.approx(175.0)
    assert df.loc[0, "fighter_b_height"] == pytest.approx(180.0)
    # Fight 2 has NO per-fight values -> static tott fills the gap:
    assert df.loc[1, "fighter_a_height"] == pytest.approx(152.4)
    assert df.loc[1, "fighter_b_height"] == pytest.approx(152.4)
    # DOB-derived age fallback also works (no per-fight age in this source):
    assert df.loc[0, "fighter_a_age"] == pytest.approx(30.0, abs=0.1)


def test_audit_doc_exists_and_documents_choices():
    doc = Path(__file__).resolve().parent.parent / "docs" / "greco_field_audit.md"
    text = doc.read_text()
    for needle in ("Category A", "Category B", "Category C",
                   "COALESCE", "FILL-ONLY", "fill-only", "provenance"):
        assert needle.lower() in text.lower()


def test_normalize_weight_class():
    from ufc_pipeline.stats_ingest import normalize_weight_class as nwc
    assert nwc("UFC Women's Flyweight Title Bout") == "Women's Flyweight"
    assert nwc("Light Heavyweight Bout") == "Light Heavyweight"
    assert nwc("Heavyweight Bout") == "Heavyweight"
    assert nwc(
        "Ultimate Fighter Brazil 2 Welterweight Tournament Title Bout"
    ) == "Welterweight"
    assert nwc("UFC Superfight Championship Bout") is None
    assert nwc("Lightweight") == "Lightweight"      # clean labels pass through
    assert nwc("") is None and nwc(None) is None
