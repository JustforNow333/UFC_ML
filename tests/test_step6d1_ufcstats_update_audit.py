"""Tests for Step 6D.1 UFCStats data-update audit (read-only).

All parsing is exercised against small static HTML fixtures; the audit uses an
injected fetcher so no test touches the network. Covers event/fight/fighter/
result parsing, date parsing, challenge detection, exact normalized name
matching (+ ambiguity), URL-preferred fighter matching, missing event/fight/
fighter detection, stale-result detection, report schema, offline fetch
behaviour, and — crucially — that the dry run performs NO writes to the DB.
"""

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufc_pipeline.db import init_schema  # noqa: E402
from ufc_pipeline.step6d_ufcstats_update_audit import (  # noqa: E402
    COMPLETED_EVENTS_URL,
    compare_to_local,
    fetch_html,
    looks_like_challenge,
    match_local_fighter,
    normalize_event_name,
    parse_event_page,
    parse_events_list,
    parse_ufcstats_date,
    read_local_db,
    run_audit,
)

EVENT_A_URL = "http://ufcstats.com/event-details/eventA"
EVENT_B_URL = "http://ufcstats.com/event-details/eventB"

EVENTS_LIST_HTML = f"""
<table class="b-statistics__table-events"><tbody>
<tr class="b-statistics__table-row"><th>Name/date</th></tr>
<tr class="b-statistics__table-row"><td class="b-statistics__table-col"><i class="b-statistics__table-content">
  <a href="{EVENT_A_URL}" class="b-link b-link_style_black">UFC Local: A</a>
  <span class="b-statistics__date">January 1, 2024</span></i></td></tr>
<tr class="b-statistics__table-row"><td class="b-statistics__table-col"><i class="b-statistics__table-content">
  <a href="{EVENT_B_URL}" class="b-link b-link_style_black">UFC New: B</a>
  <span class="b-statistics__date">June 1, 2024</span></i></td></tr>
</tbody></table>
"""


def _fight_row(fight_id, flag, a_url, a_name, b_url, b_name):
    return f"""
<tr class="b-fight-details__table-row" data-link="http://ufcstats.com/fight-details/{fight_id}">
  <td class="b-fight-details__table-col"><p class="b-fight-details__table-text">
    <i class="b-flag b-flag_style_green"><i class="b-flag__inner"><i class="b-flag__text">{flag}</i></i></i></p></td>
  <td class="b-fight-details__table-col l-page_align_left">
    <p class="b-fight-details__table-text"><a href="{a_url}" class="b-link">{a_name}</a></p>
    <p class="b-fight-details__table-text"><a href="{b_url}" class="b-link">{b_name}</a></p></td>
  <td><p class="b-fight-details__table-text">Lightweight</p></td>
  <td><p class="b-fight-details__table-text">Decision</p></td></tr>
"""


EVENT_A_HTML = (
    '<h2 class="b-content__title"><span class="b-content__title-highlight">UFC Local: A</span></h2>'
    '<table class="b-fight-details__table"><tbody>'
    + _fight_row("fa1", "win", "http://ufcstats.com/fighter-details/jones01", "Jon Jones",
                 "http://ufcstats.com/fighter-details/stipe01", "Stipe Miocic")
    + _fight_row("fa2", "win", "http://ufcstats.com/fighter-details/pereira01", "Alex Pereira",
                 "http://ufcstats.com/fighter-details/adesanya01", "Israel Adesanya")
    + _fight_row("fa3", "win", "http://ufcstats.com/fighter-details/oliveira01", "Charles Oliveira",
                 "http://ufcstats.com/fighter-details/islam01", "Islam Makhachev")
    + "</tbody></table>"
)

EVENT_B_HTML = (
    '<h2 class="b-content__title"><span class="b-content__title-highlight">UFC New: B</span></h2>'
    '<table class="b-fight-details__table"><tbody>'
    + _fight_row("fb1", "win", "http://ufcstats.com/fighter-details/pereira01", "Alex Pereira",
                 "http://ufcstats.com/fighter-details/bruno99", "Bruno Silva")
    + "</tbody></table>"
)


@pytest.fixture(scope="module")
def audit_db(tmp_path_factory):
    db = tmp_path_factory.mktemp("step6d1_db") / "hist.db"
    conn = sqlite3.connect(db)
    init_schema(conn)
    fighters = [(1, "Jon Jones"), (2, "Stipe Miocic"), (3, "Alex Pereira"),
                (4, "Bruno Silva"), (5, "Bruno Silva"), (6, "Israel Adesanya")]
    conn.executemany(
        "INSERT INTO fighters (fighter_id, name, normalized_name) VALUES (?,?,?)",
        [(fid, name, name.lower()) for fid, name in fighters])
    # stable UFCStats urls for 1,2,3,6 (not for the two Bruno Silvas)
    urls = {1: "jones01", 2: "stipe01", 3: "pereira01", 6: "adesanya01"}
    conn.executemany(
        "INSERT INTO fighter_source_ids (fighter_id, source, source_fighter_id, source_fighter_url, raw_name, normalized_name) "
        "VALUES (?,?,?,?,?,?)",
        [(fid, "greco", u, f"http://ufcstats.com/fighter-details/{u}", "", "") for fid, u in urls.items()])
    # local "UFC Local: A": Jones beat Stipe (matches fetched); Pereira vs Adesanya recorded with
    # Adesanya as winner while fetched says Pereira won -> stale/mismatched result.
    conn.execute("INSERT INTO fights (fight_id, date, event, fighter_a_id, fighter_b_id, winner_id, fighter_a_won, "
                 "source, source_row_id) VALUES (1,'2024-01-01','UFC Local: A',1,2,1,1,'test',1)")
    conn.execute("INSERT INTO fights (fight_id, date, event, fighter_a_id, fighter_b_id, winner_id, fighter_a_won, "
                 "source, source_row_id) VALUES (2,'2024-01-01','UFC Local: A',3,6,6,0,'test',2)")
    conn.commit()
    conn.close()
    return str(db)


def _fake_fetcher(mapping):
    def _fetch(url):
        html = mapping.get(url)
        if html is None:
            return {"url": url, "html": None, "source": "offline_miss", "status": "not_cached", "bytes": 0}
        return {"url": url, "html": html, "source": "cache", "status": "ok", "bytes": len(html)}
    return _fetch


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_events_list():
    events = parse_events_list(EVENTS_LIST_HTML)
    assert [(e["event_id"], e["event_name"], e["date"]) for e in events] == [
        ("eventA", "UFC Local: A", "2024-01-01"),
        ("eventB", "UFC New: B", "2024-06-01"),
    ]


def test_parse_event_page_fights_and_fighters_and_result():
    page = parse_event_page(EVENT_A_HTML)
    assert page["event_title"] == "UFC Local: A"
    assert len(page["fights"]) == 3
    f0 = page["fights"][0]
    assert f0["fight_id"] == "fa1"
    assert f0["fighter_a"]["name"] == "Jon Jones" and f0["fighter_a"]["url"].endswith("jones01")
    assert f0["result_status"] == "win" and f0["winner_name"] == "Jon Jones" and f0["is_completed"]
    # 6 unique fighters across the 3 fights
    assert len(page["fighters"]) == 6


def test_parse_draw_has_no_winner():
    html = ('<tr data-link="http://ufcstats.com/fight-details/x">'
            '<td><p><i class="b-flag__text">draw</i></p></td>'
            '<td><p><a href="http://ufcstats.com/fighter-details/a1">A</a></p>'
            '<p><a href="http://ufcstats.com/fighter-details/b1">B</a></p></td></tr>')
    f = parse_event_page(html)["fights"][0]
    assert f["result_status"] == "draw" and f["winner_name"] is None and f["is_completed"]


def test_compare_flags_local_winner_when_fetched_result_is_draw():
    event_url = "http://ufcstats.com/event-details/drawEvent"
    event = {"event_name": "UFC Draw", "date": "2024-02-01", "event_url": event_url}
    page = parse_event_page(
        '<h2 class="b-content__title"><span class="b-content__title-highlight">UFC Draw</span></h2>'
        '<table class="b-fight-details__table"><tbody>'
        + _fight_row("draw1", "draw", "http://ufcstats.com/fighter-details/a1", "Fighter A",
                     "http://ufcstats.com/fighter-details/b1", "Fighter B")
        + "</tbody></table>"
    )
    fight_key = ("2024-02-01", frozenset({
        "http://ufcstats.com/fighter-details/a1",
        "http://ufcstats.com/fighter-details/b1",
    }))
    local = {
        "event_keys": {(normalize_event_name("UFC Draw"), "2024-02-01")},
        "fight_url_keys": {fight_key: {"event": "UFC Draw"}},
        "fight_name_keys": {},
        "result_by_url_key": {fight_key: "http://ufcstats.com/fighter-details/a1"},
        "result_by_name_key": {},
        "local_fighter_urls": set(),
        "name_to_ids": {},
    }

    comparison = compare_to_local([event], {event_url: page}, local)

    assert len(comparison["stale_or_mismatched_results"]) == 1
    assert comparison["stale_or_mismatched_results"][0]["matchup"].startswith("Fighter A vs Fighter B")


def test_parse_date_and_challenge_and_event_norm():
    assert parse_ufcstats_date("April 13, 2024") == "2024-04-13"
    assert parse_ufcstats_date("Apr 13, 2024") == "2024-04-13"
    assert parse_ufcstats_date("garbage") is None
    assert looks_like_challenge("<html>Checking your browser…</html>") is True
    assert looks_like_challenge(None) is True
    assert looks_like_challenge(EVENT_A_HTML) is False
    assert normalize_event_name("UFC Fight Night: X vs. Y") == normalize_event_name("UFC Fight Night: X vs Y")


# ---------------------------------------------------------------------------
# Fighter matching
# ---------------------------------------------------------------------------

def test_match_local_fighter_url_name_ambiguous_unmatched(audit_db):
    local = read_local_db(audit_db)
    assert match_local_fighter({"name": "Jon Jones", "url": "http://ufcstats.com/fighter-details/jones01"}, local)["status"] == "matched_by_url"
    # no stored url, unique name -> matched_by_name (weaker)
    assert match_local_fighter({"name": "Israel Adesanya", "url": "http://ufcstats.com/fighter-details/UNSEEN"}, local)["status"] == "matched_by_name"
    # ambiguous: two local Bruno Silvas, url not stored
    assert match_local_fighter({"name": "Bruno Silva", "url": "http://ufcstats.com/fighter-details/bruno99"}, local)["status"] == "ambiguous"
    assert match_local_fighter({"name": "Nobody", "url": "http://ufcstats.com/fighter-details/none"}, local)["status"] == "unmatched"


# ---------------------------------------------------------------------------
# Read-only DB
# ---------------------------------------------------------------------------

def test_read_local_db_is_read_only(audit_db):
    local = read_local_db(audit_db)
    assert local["counts"]["events"] == 1 and local["counts"]["fighters"] == 6
    assert "fights" in local["schema"] and "fighter_source_ids" in local["schema"]
    # a write through mode=ro must be rejected
    con = sqlite3.connect(f"file:{Path(audit_db).as_posix()}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("INSERT INTO fighters (fighter_id, name) VALUES (999, 'Hacker')")
    con.close()


# ---------------------------------------------------------------------------
# Full audit (dry run, injected fetcher)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def audit_report(audit_db, tmp_path_factory):
    out = tmp_path_factory.mktemp("step6d1_out")
    fetcher = _fake_fetcher({COMPLETED_EVENTS_URL: EVENTS_LIST_HTML, EVENT_A_URL: EVENT_A_HTML, EVENT_B_URL: EVENT_B_HTML})
    report = run_audit(db_path=audit_db, output_dir=str(out), max_events=10, fetcher=fetcher)
    return report, out


def test_audit_detects_missing_events_fights_fighters_and_stale(audit_report):
    report, _ = audit_report
    assert report["n_fetched_events"] == 2
    assert report["events_present_locally"] == 1
    # UFC New: B is not local
    assert any(e["event_name"] == "UFC New: B" for e in report["missing_local_events"])
    # Oliveira vs Islam (not local) and the event-B fight are missing
    missing_matchups = {m["matchup"].split(" (")[0] for m in report["missing_local_fights"]}
    assert "Charles Oliveira vs Islam Makhachev" in missing_matchups
    # unmatched fighters reported
    missing_fighter_names = {f["name"] for f in report["missing_local_fighters"]}
    assert {"Charles Oliveira", "Islam Makhachev"} <= missing_fighter_names
    # Bruno Silva is ambiguous
    assert any(a["name"] == "Bruno Silva" for a in report["ambiguous_fighter_names"])
    # Pereira vs Adesanya matched locally but the recorded winner disagrees -> stale/mismatched
    assert any("Pereira" in s["matchup"] for s in report["stale_or_mismatched_results"])
    # Jones vs Stipe result matches -> NOT stale
    assert not any("Jon Jones" in s["matchup"] for s in report["stale_or_mismatched_results"])


def test_audit_report_schema_and_no_write(audit_report):
    report, out = audit_report
    for key in ("generated_at", "mode", "db_writes_performed", "database_modified", "local_db_path",
                "db_schema_summary", "db_schema_assumptions", "local_counts", "scraping", "pages_fetched",
                "fetch_details", "fetch_available", "n_fetched_events", "n_fetched_fights", "n_fetched_fighters",
                "missing_local_events", "missing_local_fights", "missing_local_fighters",
                "stale_or_mismatched_results", "ambiguous_fighter_names", "unmatched_fighter_names",
                "upcoming_events", "recommended_next_action", "no_write_statement", "limitations"):
        assert key in report, f"{key} missing from audit report"
    assert report["mode"] == "read_only_dry_run"
    assert report["db_writes_performed"] is False and report["database_modified"] is False
    assert (out / "step6d1_ufcstats_audit.json").exists()
    assert (out / "step6d1_ufcstats_audit.md").exists()


def test_audit_does_not_modify_db(audit_db, tmp_path):
    before = hashlib.md5(Path(audit_db).read_bytes()).hexdigest()
    fetcher = _fake_fetcher({COMPLETED_EVENTS_URL: EVENTS_LIST_HTML, EVENT_A_URL: EVENT_A_HTML, EVENT_B_URL: EVENT_B_HTML})
    run_audit(db_path=audit_db, output_dir=str(tmp_path), max_events=10, fetcher=fetcher)
    after = hashlib.md5(Path(audit_db).read_bytes()).hexdigest()
    assert before == after, "audit must not modify the database"


def test_audit_handles_blocked_fetch_gracefully(audit_db, tmp_path):
    # fetcher returns nothing (JS challenge / offline) -> audit still produces a report, no crash, no writes
    report = run_audit(db_path=audit_db, output_dir=str(tmp_path), fetcher=lambda url:
                       {"url": url, "html": None, "source": "blocked", "status": "js_challenge", "bytes": 100})
    assert report["fetch_available"] is False
    assert report["n_fetched_events"] == 0
    assert "cache" in report["recommended_next_action"].lower()
    assert report["db_writes_performed"] is False


# ---------------------------------------------------------------------------
# Fetcher offline behaviour
# ---------------------------------------------------------------------------

def test_fetch_html_offline_cache_only_and_cache_roundtrip(tmp_path):
    cache = tmp_path / "cache"
    # offline + no cache -> offline_miss (never hits network)
    res = fetch_html("http://ufcstats.com/x", cache_dir=str(cache), offline_cache_only=True, delay=0)
    assert res["source"] == "offline_miss" and res["html"] is None
    # seed a cache file and confirm it is read
    cache.mkdir(parents=True, exist_ok=True)
    (cache / (hashlib.sha1(b"http://ufcstats.com/x").hexdigest() + ".html")).write_text("<html>event-details cached</html>")
    res2 = fetch_html("http://ufcstats.com/x", cache_dir=str(cache), offline_cache_only=True, delay=0)
    assert res2["source"] == "cache" and "cached" in res2["html"]
