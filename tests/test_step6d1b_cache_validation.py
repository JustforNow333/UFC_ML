"""Tests for Step 6D.1b: real-cached-HTML validation helpers.

Covers page-type classification, friendly cache filenames + offline lookup, the
cache validator (recognises pages, detects challenge HTML, warns on empty/unknown
caches), offline-cache-only making no network call, and the cache-source-quality
fields added to the audit report. Static fixtures only; never hits the network
and never writes a database.
"""

import hashlib
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufc_pipeline.db import init_schema  # noqa: E402
from ufc_pipeline.step6d_ufcstats_update_audit import (  # noqa: E402
    COMPLETED_EVENTS_URL,
    detect_page_type,
    expected_cache_filenames,
    fetch_html,
    run_audit,
    suggested_cache_filename,
    validate_cache,
)
import ufc_pipeline.step6d_ufcstats_update_audit as audit_mod  # noqa: E402

EVENTS_HTML = (
    '<table class="b-statistics__table-events"><tbody>'
    '<tr class="b-statistics__table-row"><td class="b-statistics__table-col"><i class="b-statistics__table-content">'
    '<a href="http://ufcstats.com/event-details/aaa111" class="b-link">UFC Local: A</a>'
    '<span class="b-statistics__date">January 1, 2024</span></i></td></tr></tbody></table>'
)
EVENT_HTML = (
    '<h2 class="b-content__title"><span class="b-content__title-highlight">UFC Local: A</span></h2>'
    '<table class="b-fight-details__table"><tbody>'
    '<tr class="b-fight-details__table-row" data-link="http://ufcstats.com/fight-details/f1">'
    '<td><p><i class="b-flag__text">win</i></p></td>'
    '<td class="l-page_align_left"><p><a href="http://ufcstats.com/fighter-details/jones01">Jon Jones</a></p>'
    '<p><a href="http://ufcstats.com/fighter-details/stipe01">Stipe Miocic</a></p></td></tr></tbody></table>'
)
EVENT_HTML_WITH_FIGHTERISH_MARKERS = (
    '<!-- saved from url=(0050)http://ufcstats.com/event-details/abc123 -->'
    '<h2 class="b-content__title"><span class="b-content__title-highlight">UFC Local: A</span></h2>'
    '<div class="b-list__info-box b-list__info-box_style_large-width">event metadata</div>'
    '<table class="b-fight-details__table"><tbody>'
    '<tr class="b-fight-details__table-row" data-link="http://ufcstats.com/fight-details/f1">'
    '<td><p><i class="b-flag__text">win</i></p></td>'
    '<td class="l-page_align_left"><p><a href="http://ufcstats.com/fighter-details/jones01">Jon Jones</a></p>'
    '<p><a href="http://ufcstats.com/fighter-details/stipe01">Stipe Miocic</a></p></td></tr></tbody></table>'
)
CHALLENGE_HTML = "<html><head><title>Loading…</title></head><body><p>Checking your browser…</p></body></html>"
FIGHTER_HTML = "<html><body>Record: 20-3-0 STANCE: Orthodox — fighter-details/self Career statistics</body></html>"


# ---------------------------------------------------------------------------
# Page-type classification + filenames
# ---------------------------------------------------------------------------

def test_detect_page_type():
    assert detect_page_type(EVENTS_HTML) == "completed_events"
    completed_with_upcoming_nav = (
        "<!-- saved from url=(0056)http://ufcstats.com/statistics/events/completed?page=all -->"
        '<a href="http://ufcstats.com/statistics/events/upcoming">Upcoming</a>'
        + EVENTS_HTML
    )
    upcoming_saved = (
        "<!-- saved from url=(0055)http://ufcstats.com/statistics/events/upcoming?page=all -->"
        + EVENTS_HTML
    )
    assert detect_page_type(completed_with_upcoming_nav) == "completed_events"
    assert detect_page_type(upcoming_saved) == "upcoming_events"
    assert detect_page_type(EVENT_HTML) == "event_detail"
    assert detect_page_type(EVENT_HTML_WITH_FIGHTERISH_MARKERS) == "event_detail"
    assert detect_page_type(EVENT_HTML, filename="event_abc123.html") == "event_detail"
    assert detect_page_type(CHALLENGE_HTML) == "browser_challenge"
    assert detect_page_type(CHALLENGE_HTML + EVENT_HTML, filename="event_abc123.html") == "browser_challenge"
    assert detect_page_type(FIGHTER_HTML) == "fighter_detail"
    assert detect_page_type("<html>totally unrelated page</html>") == "unknown"
    assert detect_page_type(None) == "unknown"


def test_expected_and_suggested_cache_filenames():
    names = expected_cache_filenames(COMPLETED_EVENTS_URL)
    assert names[0].endswith(".html") and len(names[0]) == 45  # sha1 + .html
    assert "completed_events.html" in names
    ev = expected_cache_filenames("http://ufcstats.com/event-details/abc123")
    assert "event_abc123.html" in ev
    assert suggested_cache_filename("http://ufcstats.com/event-details/abc123") == "event_abc123.html"


# ---------------------------------------------------------------------------
# Cache validator
# ---------------------------------------------------------------------------

def test_validate_cache_recognizes_pages_and_is_ready(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "completed_events.html").write_text(EVENTS_HTML)
    (cache / "event_aaa111.html").write_text(EVENT_HTML)
    result = validate_cache(str(cache))
    assert result["cache_dir_exists"] is True and result["n_html_files"] == 2
    assert result["page_types_detected"].get("completed_events") == 1
    assert result["page_types_detected"].get("event_detail") == 1
    assert result["has_completed_events_page"] and result["has_event_detail_page"]
    assert result["ready_for_offline_audit"] is True
    # parse previews present
    ev = next(f for f in result["files"] if f["page_type"] == "event_detail")
    assert ev["fights_parsed"] == 1 and ev["fighters_parsed"] == 2


def test_validate_cache_detects_challenge_and_warns(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "completed_events.html").write_text(CHALLENGE_HTML)  # a challenge, not real content
    result = validate_cache(str(cache))
    assert result["challenge_pages_detected"] == ["completed_events.html"]
    assert result["ready_for_offline_audit"] is False
    assert any("browser-challenge" in w for w in result["warnings"])


def test_validate_cache_empty_and_missing(tmp_path):
    empty = validate_cache(str(tmp_path / "cache"))  # does not exist
    assert empty["cache_dir_exists"] is False and empty["ready_for_offline_audit"] is False
    assert any("does not exist" in w for w in empty["warnings"])
    assert empty["recommended_files_to_save"]  # tells the user what to save
    (tmp_path / "cache").mkdir()
    only_events = tmp_path / "cache"
    (only_events / "completed_events.html").write_text(EVENTS_HTML)
    res = validate_cache(str(only_events))
    assert res["has_completed_events_page"] and not res["has_event_detail_page"]
    assert any("no event-detail pages" in w for w in res["warnings"])


def test_validate_cache_takes_no_db_argument():
    import inspect
    assert list(inspect.signature(validate_cache).parameters) == ["cache_dir"]


# ---------------------------------------------------------------------------
# Friendly-name lookup + offline-cache-only makes no network call
# ---------------------------------------------------------------------------

def test_friendly_name_cache_lookup(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "completed_events.html").write_text(EVENTS_HTML)  # NOT the sha1 name
    res = fetch_html(COMPLETED_EVENTS_URL, cache_dir=str(cache), offline_cache_only=True, delay=0)
    assert res["source"] == "cache" and res["cache_file"] == "completed_events.html"
    assert res["html"] is not None


def test_offline_cache_only_makes_no_network_call(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("network must not be contacted in offline-cache-only mode")
    monkeypatch.setattr(audit_mod.urllib.request, "urlopen", _boom)
    # miss -> offline_miss, no network
    res = fetch_html("http://ufcstats.com/x", cache_dir=str(tmp_path), offline_cache_only=True, delay=0)
    assert res["source"] == "offline_miss"
    # a full audit in offline mode also makes no network call
    report = run_audit(db_path="data/ufc.db", output_dir=str(tmp_path / "out"),
                       cache_dir=str(tmp_path / "empty"), offline_cache_only=True, max_events=2)
    assert report["fetch_available"] is False


def test_cached_challenge_html_is_rejected(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "completed_events.html").write_text(CHALLENGE_HTML)
    res = fetch_html(COMPLETED_EVENTS_URL, cache_dir=str(cache), offline_cache_only=True, delay=0)
    assert res["source"] == "cache_challenge" and res["html"] is None


# ---------------------------------------------------------------------------
# Audit report cache-source quality fields + no DB writes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mini_db(tmp_path_factory):
    db = tmp_path_factory.mktemp("6d1b_db") / "hist.db"
    conn = sqlite3.connect(db)
    init_schema(conn)
    conn.execute("INSERT INTO fighters (fighter_id, name, normalized_name) VALUES (1,'Jon Jones','jon jones')")
    conn.execute("INSERT INTO fighters (fighter_id, name, normalized_name) VALUES (2,'Stipe Miocic','stipe miocic')")
    conn.execute("INSERT INTO fights (fight_id, date, event, fighter_a_id, fighter_b_id, winner_id, fighter_a_won, "
                 "source, source_row_id) VALUES (1,'2024-01-01','UFC Local: A',1,2,1,1,'test',1)")
    conn.commit()
    conn.close()
    return str(db)


def test_audit_report_includes_cache_source_quality(mini_db, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "completed_events.html").write_text(EVENTS_HTML)
    (cache / "event_aaa111.html").write_text(EVENT_HTML)
    report = run_audit(db_path=mini_db, output_dir=str(tmp_path / "out"),
                       cache_dir=str(cache), offline_cache_only=True, include_upcoming=True, max_events=5)
    csq = report["cache_source_quality"]
    for key in ("offline_cache_only", "cache_dir", "cache_files_used", "expected_missing",
                "page_types_detected", "challenge_pages_detected", "parse_warnings",
                "cache_validation", "recommendation"):
        assert key in csq, f"{key} missing from cache_source_quality"
    assert csq["offline_cache_only"] is True
    assert "completed_events.html" in csq["cache_files_used"]
    # upcoming page was requested but not cached -> reported as expected_missing
    assert any(m["suggested_filename"] == "upcoming_events.html" for m in csq["expected_missing"])
    # the audit consumed the cached event page and matched the local fight -> not missing
    assert report["events_present_locally"] == 1


def test_audit_offline_does_not_modify_db(mini_db, tmp_path):
    before = hashlib.md5(Path(mini_db).read_bytes()).hexdigest()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "completed_events.html").write_text(EVENTS_HTML)
    run_audit(db_path=mini_db, output_dir=str(tmp_path / "out"), cache_dir=str(cache),
              offline_cache_only=True, max_events=5)
    assert hashlib.md5(Path(mini_db).read_bytes()).hexdigest() == before
