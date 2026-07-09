"""Step 6D.1: UFCStats data-update audit (read-only / dry-run).

Checks UFCStats against the local ``data/ufc.db`` and reports what events,
fights, fighters, and results look missing or stale — so a human can decide
whether the history DB needs refreshing before Step 6C builds upcoming-card
features. This step **never modifies the database**: it is a dry-run audit only.
A guarded ``--apply`` writer is deliberately left for a future Step 6D.2.

Scraping posture (conservative on purpose):
  * stdlib only (``urllib`` + ``html.parser``); no new dependencies, no
    sportsbook odds, no prediction markets.
  * a clear User-Agent, a polite inter-request delay, and a ``--max-events`` cap.
  * an on-disk HTML cache (``--cache-dir``) and an ``--offline-cache-only`` mode
    so the audit runs deterministically from cached/provided pages.
  * UFCStats currently serves a JavaScript "checking your browser" challenge to
    plain HTTP clients. This module does NOT try to defeat it — it detects the
    challenge and degrades gracefully, telling the user to supply cached HTML
    (saved from a real browser / headless fetch out of band). Parsing then runs
    on that cached HTML.

Matching posture: prefer stable UFCStats URLs/IDs (fighter URLs are already
stored in ``fighter_source_ids``); fall back to exact normalized name matching
and REPORT ambiguities. Never silently fuzzy-match. Never overwrite local rows.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from ufc_pipeline.step6c_upcoming_feature_builder import normalize_fighter_name

BASE_URL = "http://ufcstats.com"
COMPLETED_EVENTS_URL = f"{BASE_URL}/statistics/events/completed?page=all"
UPCOMING_EVENTS_URL = f"{BASE_URL}/statistics/events/upcoming?page=all"
USER_AGENT = "Mozilla/5.0 (compatible; ufc-elo-audit/0.1; +personal-research; read-only)"
DEFAULT_REQUEST_DELAY = 1.5  # seconds between live requests (polite)
DEFAULT_TIMEOUT = 15

REPORT_JSON = "step6d1_ufcstats_audit.json"
REPORT_MD = "step6d1_ufcstats_audit.md"
DEFAULT_OUTPUT_DIR = "reports/data_update"

# UFCStats ids are 16-char lowercase hex, but keep the id class permissive so a
# format change (or a test fixture) does not silently drop rows.
EVENT_DETAILS_RE = re.compile(r"event-details/([a-zA-Z0-9]+)", re.I)
FIGHTER_DETAILS_RE = re.compile(r"fighter-details/([a-zA-Z0-9]+)", re.I)
FIGHT_DETAILS_RE = re.compile(r"fight-details/([a-zA-Z0-9]+)", re.I)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(obj):
    try:
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
    except TypeError:
        pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def normalize_event_name(value) -> str:
    if value is None:
        return ""
    s = str(value).lower().strip()
    s = re.sub(r"[.:]", "", s)
    s = re.sub(r"\bvs\b", "vs", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_ufcstats_date(raw) -> str | None:
    """'April 13, 2024' -> '2024-04-13'. Returns None if unparseable."""
    if not raw:
        return None
    raw = str(raw).strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw[:10]).strftime("%Y-%m-%d")
    except ValueError:
        return None


def looks_like_challenge(html: str | None) -> bool:
    """Detect the JS 'checking your browser' interstitial / an empty shell."""
    if not html:
        return True
    low = html.lower()
    if "checking your browser" in low or "this site requires javascript" in low:
        return True
    # A real UFCStats listing/detail page always references these paths.
    if len(html) < 4000 and not any(t in low for t in ("event-details", "fighter-details", "fight-details")):
        return True
    return False


def detect_page_type(html: str | None, filename: str | None = None) -> str:
    """Heuristic classifier for a cached UFCStats HTML file (content, not filename).

    Returns one of: browser_challenge, completed_events, upcoming_events,
    event_detail, fighter_detail, events_list, unknown.
    """
    if not html:
        return "unknown"
    low = html.lower()
    # Classification uses EXPLICIT challenge markers (not the broader fetch-path
    # heuristic), so a small generic page classifies as 'unknown', not challenge.
    if "checking your browser" in low or "this site requires javascript" in low or "enable javascript" in low:
        return "browser_challenge"
    saved_from_completed = "statistics/events/completed" in low[:1000]
    saved_from_upcoming = "statistics/events/upcoming" in low[:1000]
    saved_from_event = "event-details/" in low[:1000]
    saved_from_fighter = "fighter-details/" in low[:1000]
    filename_is_event = bool(filename and re.match(r"event_[a-zA-Z0-9]+\.html$", filename))
    has_events_table = "b-statistics__table-events" in low or low.count("event-details/") >= 3
    has_fight_table = "b-fight-details__table" in low or "fight-details/" in low
    has_title = "b-content__title-highlight" in low
    fighter_markers = any(t in low for t in ("stance:", "record:", "b-list__info-box", "career statistics", "date of birth"))
    if has_events_table and "b-fight-details__table" not in low:
        if saved_from_upcoming and not saved_from_completed:
            return "upcoming_events"
        return "completed_events"

    # Event pages contain many fighter links and a b-list__info-box event-summary
    # block, so they must be recognized before the broader fighter-page check.
    if (saved_from_event or filename_is_event or (has_fight_table and has_title)):
        try:
            page = parse_event_page(html)
        except Exception:  # pragma: no cover - defensive against malformed HTML
            page = {"event_title": None, "fights": []}
        if page.get("event_title") and page.get("fights"):
            return "event_detail"
    if saved_from_fighter:
        return "fighter_detail"
    if fighter_markers and "fighter-details" in low and not has_title:
        return "fighter_detail"
    if has_fight_table and has_title:
        return "event_detail"
    if fighter_markers:
        return "fighter_detail"
    if "event-details/" in low:
        return "events_list"
    return "unknown"


# ---------------------------------------------------------------------------
# HTML parsing (pure functions over HTML strings — the tested surface)
# ---------------------------------------------------------------------------

class _EventsListParser(HTMLParser):
    """Collect (event_url, event_name) anchors and b-statistics__date spans in
    document order, so each event can be paired with the date that follows it."""

    def __init__(self):
        super().__init__()
        self.tokens: list[tuple[str, str]] = []  # ("event", url) or ("name", text) or ("date", text)
        self._in_event_anchor = False
        self._cur_url = None
        self._in_date_span = False
        self._text_buf = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and a.get("href") and EVENT_DETAILS_RE.search(a["href"]):
            self._in_event_anchor = True
            self._cur_url = a["href"].strip()
            self._text_buf = []
        elif tag == "span" and "b-statistics__date" in (a.get("class") or ""):
            self._in_date_span = True
            self._text_buf = []

    def handle_data(self, data):
        if self._in_event_anchor or self._in_date_span:
            self._text_buf.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._in_event_anchor:
            self.tokens.append(("event", self._cur_url))
            self.tokens.append(("name", "".join(self._text_buf).strip()))
            self._in_event_anchor = False
        elif tag == "span" and self._in_date_span:
            self.tokens.append(("date", "".join(self._text_buf).strip()))
            self._in_date_span = False


def parse_events_list(html: str) -> list[dict]:
    """Parse a completed/upcoming events listing page into event dicts."""
    p = _EventsListParser()
    p.feed(html or "")
    events, i = [], 0
    toks = p.tokens
    while i < len(toks):
        if toks[i][0] == "event":
            url = toks[i][1]
            name = toks[i + 1][1] if i + 1 < len(toks) and toks[i + 1][0] == "name" else ""
            date_raw = None
            j = i + 2
            while j < len(toks) and toks[j][0] != "event":
                if toks[j][0] == "date":
                    date_raw = toks[j][1]
                    break
                j += 1
            m = EVENT_DETAILS_RE.search(url)
            events.append({
                "event_id": m.group(1) if m else None, "event_url": url, "event_name": name,
                "date": parse_ufcstats_date(date_raw), "date_raw": date_raw,
            })
            i += 2
        else:
            i += 1
    return events


class _EventPageParser(HTMLParser):
    """Walk the fight table: collect, per row, its fight-details link, the W/L
    flag text, and the two fighter anchors in order."""

    def __init__(self):
        super().__init__()
        self.event_title = None
        self._in_title = False
        self._title_buf = []
        self.rows: list[dict] = []
        self._cur = None
        self._in_fighter_anchor = False
        self._fighter_url = None
        self._buf = []
        self._in_flag = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class") or ""
        if tag in ("h2", "span") and "b-content__title-highlight" in cls:
            self._in_title = True
            self._title_buf = []
        if tag == "tr" and a.get("data-link") and FIGHT_DETAILS_RE.search(a.get("data-link", "")):
            self._cur = {"fight_url": a["data-link"].strip(), "fighters": [], "flags": []}
        elif tag == "tr" and "b-fight-details__table-row" in cls and "b-fight-details__table-row__head" not in cls:
            # rows without data-link (defensive): still open a row container
            self._cur = self._cur or {"fight_url": None, "fighters": [], "flags": []}
        if tag == "a" and a.get("href") and FIGHTER_DETAILS_RE.search(a["href"]) and self._cur is not None:
            self._in_fighter_anchor = True
            self._fighter_url = a["href"].strip()
            self._buf = []
        if tag == "i" and "b-flag__text" in cls and self._cur is not None:
            self._in_flag = True
            self._buf = []

    def handle_data(self, data):
        if self._in_title or self._in_fighter_anchor or self._in_flag:
            self._buf_append(data)

    def _buf_append(self, data):
        if self._in_title:
            self._title_buf.append(data)
        else:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag in ("h2", "span") and self._in_title:
            self.event_title = "".join(self._title_buf).strip() or self.event_title
            self._in_title = False
        if tag == "a" and self._in_fighter_anchor:
            m = FIGHTER_DETAILS_RE.search(self._fighter_url)
            self._cur["fighters"].append({
                "name": "".join(self._buf).strip(), "url": self._fighter_url,
                "fighter_id": m.group(1) if m else None,
            })
            self._in_fighter_anchor = False
        if tag == "i" and self._in_flag:
            self._cur["flags"].append("".join(self._buf).strip().lower())
            self._in_flag = False
        if tag == "tr" and self._cur is not None:
            if self._cur["fighters"]:
                self.rows.append(self._cur)
            self._cur = None


def parse_event_page(html: str) -> dict:
    """Parse an event-details page into {event_title, fights:[...], fighters:[...]}."""
    p = _EventPageParser()
    p.feed(html or "")
    fights, fighters = [], []
    seen_fighter_urls = set()
    for row in p.rows:
        fs = row["fighters"]
        if len(fs) < 2:
            continue
        a, b = fs[0], fs[1]
        # UFCStats lists the winner first; the flag column reads win/loss/draw/nc.
        flags = row["flags"]
        status = flags[0] if flags else None
        if status == "win":
            winner_url, winner_name = a["url"], a["name"]
        elif status in ("draw", "nc", "no contest"):
            winner_url, winner_name = None, None
        else:
            winner_url, winner_name = None, None  # upcoming / unknown
        m = FIGHT_DETAILS_RE.search(row["fight_url"] or "")
        fights.append({
            "fight_id": m.group(1) if m else None, "fight_url": row["fight_url"],
            "fighter_a": a, "fighter_b": b,
            "result_status": status, "winner_url": winner_url, "winner_name": winner_name,
            "is_completed": status in ("win", "draw", "nc", "no contest"),
        })
        for f in (a, b):
            if f["url"] not in seen_fighter_urls:
                seen_fighter_urls.add(f["url"])
                fighters.append(f)
    return {"event_title": p.event_title, "fights": fights, "fighters": fighters}


# ---------------------------------------------------------------------------
# Cache validation (offline; classify + preview manually saved HTML)
# ---------------------------------------------------------------------------

# The four pages the audit most wants cached before a first real run.
RECOMMENDED_CACHE_FILES = [
    ("completed_events.html", "the completed-events listing page (statistics/events/completed?page=all)"),
    ("upcoming_events.html", "the upcoming-events listing page (statistics/events/upcoming?page=all)"),
    ("event_<id>.html", "one RECENT completed event-detail page (event-details/<id>)"),
    ("event_<id>.html", "one OLDER known event-detail page, to sanity-check historical parsing"),
]


def validate_cache(cache_dir: str | None) -> dict:
    """Classify every cached .html file and preview parse counts. Read-only, no
    network, no DB. Used by the validator CLI and folded into the audit report."""
    p = Path(cache_dir) if cache_dir else None
    exists = bool(p and p.exists())
    files = sorted(p.glob("*.html")) if exists else []
    entries, page_type_counts, warnings = [], {}, []
    challenge_files = []
    for f in files:
        html = f.read_text(encoding="utf-8", errors="ignore")
        ptype = detect_page_type(html, filename=f.name)
        page_type_counts[ptype] = page_type_counts.get(ptype, 0) + 1
        entry = {"file": f.name, "bytes": len(html), "page_type": ptype}
        try:
            if ptype in ("completed_events", "upcoming_events", "events_list"):
                entry["events_parsed"] = len(parse_events_list(html))
            elif ptype == "event_detail":
                page = parse_event_page(html)
                entry.update(event_title=page["event_title"], fights_parsed=len(page["fights"]),
                             fighters_parsed=len(page["fighters"]))
        except Exception as exc:  # pragma: no cover - defensive against odd HTML
            entry["parse_error"] = f"{type(exc).__name__}: {exc}"
            warnings.append(f"{f.name}: parse error ({type(exc).__name__})")
        if ptype == "browser_challenge":
            challenge_files.append(f.name)
        entries.append(entry)

    if not exists:
        warnings.append(f"cache dir '{cache_dir}' does not exist — create it and save UFCStats HTML there.")
    elif not files:
        warnings.append(f"cache dir '{cache_dir}' contains no .html files — save the recommended pages there.")
    if challenge_files:
        warnings.append(f"{len(challenge_files)} cached file(s) are UFCStats browser-challenge pages, not real "
                        f"content (re-save from a logged-in/rendered browser tab): {challenge_files}")
    if files and not any(e["page_type"] in ("completed_events", "events_list") for e in entries):
        warnings.append("no completed-events listing page detected — save it first (completed_events.html).")
    if files and not any(e["page_type"] == "event_detail" for e in entries):
        warnings.append("no event-detail pages detected — save at least one recent and one older event page.")

    has_completed = any(e["page_type"] in ("completed_events", "events_list") for e in entries)
    has_event = any(e["page_type"] == "event_detail" for e in entries)
    ready = bool(files) and has_completed and has_event and not challenge_files
    recommendation = (
        "Cache looks ready: run the offline-cache audit next." if ready else
        "Save the recommended pages (see below) from a normal browser into the cache dir, then re-validate. "
        "The project intentionally does not bypass the UFCStats browser challenge.")
    return {
        "cache_dir": cache_dir, "cache_dir_exists": exists, "n_html_files": len(files),
        "files": entries, "page_types_detected": page_type_counts,
        "challenge_pages_detected": challenge_files, "warnings": warnings,
        "has_completed_events_page": has_completed, "has_event_detail_page": has_event,
        "ready_for_offline_audit": ready,
        "recommended_files_to_save": [{"filename": n, "description": d} for n, d in RECOMMENDED_CACHE_FILES],
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Fetching (cache-first, polite, graceful on the JS challenge)
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: str, url: str) -> Path:
    return Path(cache_dir) / (hashlib.sha1(url.encode()).hexdigest() + ".html")


def expected_cache_filenames(url: str) -> list[str]:
    """Candidate cache filenames for a URL, in lookup priority.

    The live fetcher writes the opaque ``sha1(url).html`` name, but a human
    saving pages from a browser can't compute that — so we ALSO accept friendly
    names (``completed_events.html``, ``event_<id>.html``, ...). Lookup tries the
    sha1 name first (round-trips the live cache) then the friendly aliases.
    """
    names = [hashlib.sha1(url.encode()).hexdigest() + ".html"]
    if url == COMPLETED_EVENTS_URL:
        names += ["completed_events.html", "completed.html"]
    elif url == UPCOMING_EVENTS_URL:
        names += ["upcoming_events.html", "upcoming.html"]
    m = EVENT_DETAILS_RE.search(url)
    if m:
        names += [f"event_{m.group(1)}.html", f"event-details_{m.group(1)}.html", f"{m.group(1)}.html"]
    m2 = FIGHTER_DETAILS_RE.search(url)
    if m2:
        names += [f"fighter_{m2.group(1)}.html", f"fighter-details_{m2.group(1)}.html"]
    return names


def suggested_cache_filename(url: str) -> str:
    """The human-friendly filename we recommend the user save a URL's HTML as."""
    names = expected_cache_filenames(url)
    return names[1] if len(names) > 1 else names[0]


def fetch_html(
    url: str, cache_dir: str | None = None, offline_cache_only: bool = False,
    delay: float = DEFAULT_REQUEST_DELAY, timeout: int = DEFAULT_TIMEOUT, user_agent: str = USER_AGENT,
) -> dict:
    """Return {'url', 'html', 'source', 'status', 'bytes'}.

    ``source`` is one of cache / live / offline_miss / blocked / error. The audit
    treats blocked/offline_miss/error as "no usable HTML" and reports it.
    """
    if cache_dir:
        for fname in expected_cache_filenames(url):
            cp = Path(cache_dir) / fname
            if cp.exists():
                html = cp.read_text(encoding="utf-8", errors="ignore")
                source = "cache_challenge" if looks_like_challenge(html) else "cache"
                return {"url": url, "html": None if source == "cache_challenge" else html,
                        "source": source, "status": "js_challenge" if source == "cache_challenge" else "ok",
                        "bytes": len(html), "cache_file": fname}
    if offline_cache_only:
        return {"url": url, "html": None, "source": "offline_miss", "status": "not_cached", "bytes": 0,
                "cache_file": None, "suggested_filename": suggested_cache_filename(url)}
    try:
        if delay:
            time.sleep(delay)
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", "ignore")
            code = getattr(resp, "status", 200)
        if looks_like_challenge(html):
            return {"url": url, "html": None, "source": "blocked", "status": "js_challenge",
                    "bytes": len(html), "cache_file": None, "suggested_filename": suggested_cache_filename(url)}
        if cache_dir:
            cp = _cache_path(cache_dir, url)
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(html, encoding="utf-8")
        return {"url": url, "html": html, "source": "live", "status": str(code), "bytes": len(html),
                "cache_file": _cache_path(cache_dir, url).name if cache_dir else None}
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        return {"url": url, "html": None, "source": "error", "status": type(exc).__name__, "bytes": 0,
                "cache_file": None, "suggested_filename": suggested_cache_filename(url)}


def make_fetcher(cache_dir, offline_cache_only, delay):
    def _fetch(url):
        return fetch_html(url, cache_dir=cache_dir, offline_cache_only=offline_cache_only, delay=delay)
    return _fetch


# ---------------------------------------------------------------------------
# Local DB (READ-ONLY)
# ---------------------------------------------------------------------------

def read_local_db(db_path: str) -> dict:
    """Read events/fights/fighters from the local DB in READ-ONLY mode."""
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        schema = {}
        for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            schema[table] = {"columns": cols, "row_count": int(n)}

        fighter_url_by_id = {}
        for fid, url in conn.execute(
            "SELECT fighter_id, source_fighter_url FROM fighter_source_ids "
            "WHERE fighter_id IS NOT NULL AND source_fighter_url IS NOT NULL"
        ):
            fighter_url_by_id.setdefault(int(fid), url)
        local_fighter_urls = set(fighter_url_by_id.values())

        name_to_ids: dict[str, list[int]] = {}
        for fid, name in conn.execute("SELECT fighter_id, name FROM fighters"):
            name_to_ids.setdefault(normalize_fighter_name(name), []).append(int(fid))

        event_keys, event_rows = set(), []
        for name, date in conn.execute("SELECT DISTINCT event, date FROM fights ORDER BY date"):
            event_keys.add((normalize_event_name(name), str(date)))
            event_rows.append({"event_name": name, "date": str(date)})

        fights = conn.execute(
            "SELECT f.date, f.event, f.fighter_a_id, f.fighter_b_id, f.winner_id, "
            "fa.name a_name, fb.name b_name, fw.name w_name "
            "FROM fights f JOIN fighters fa ON fa.fighter_id=f.fighter_a_id "
            "JOIN fighters fb ON fb.fighter_id=f.fighter_b_id "
            "LEFT JOIN fighters fw ON fw.fighter_id=f.winner_id"
        ).fetchall()
        fight_url_keys, fight_name_keys, result_by_url_key, result_by_name_key = {}, {}, {}, {}
        for date, event, a_id, b_id, w_id, a_name, b_name, w_name in fights:
            a_url, b_url = fighter_url_by_id.get(int(a_id)), fighter_url_by_id.get(int(b_id))
            w_url = fighter_url_by_id.get(int(w_id)) if w_id is not None else None
            if a_url and b_url:
                key = (str(date), frozenset({a_url, b_url}))
                fight_url_keys[key] = {"event": event}
                result_by_url_key[key] = w_url
            nkey = (str(date), frozenset({normalize_fighter_name(a_name), normalize_fighter_name(b_name)}))
            fight_name_keys[nkey] = {"event": event}
            result_by_name_key[nkey] = normalize_fighter_name(w_name) if w_name else None
    finally:
        conn.close()
    return {
        "schema": schema,
        "fighter_url_by_id": fighter_url_by_id,
        "local_fighter_urls": local_fighter_urls,
        "name_to_ids": name_to_ids,
        "event_keys": event_keys, "event_rows": event_rows,
        "fight_url_keys": fight_url_keys, "fight_name_keys": fight_name_keys,
        "result_by_url_key": result_by_url_key, "result_by_name_key": result_by_name_key,
        "counts": {
            "events": len(event_keys), "fights": len(fights),
            "fighters": schema.get("fighters", {}).get("row_count", 0),
            "fighter_urls": len(local_fighter_urls),
        },
    }


# ---------------------------------------------------------------------------
# Comparison (pure)
# ---------------------------------------------------------------------------

def match_local_fighter(fighter: dict, local: dict) -> dict:
    """Prefer the stable UFCStats URL; fall back to exact normalized name."""
    url = fighter.get("url")
    if url and url in local["local_fighter_urls"]:
        return {"status": "matched_by_url", "detail": url}
    norm = normalize_fighter_name(fighter.get("name"))
    ids = local["name_to_ids"].get(norm, [])
    if len(ids) == 1:
        return {"status": "matched_by_name", "detail": f"fighter_id={ids[0]} (url not stored locally)"}
    if len(ids) > 1:
        return {"status": "ambiguous", "detail": f"{len(ids)} local fighters share normalized name '{norm}'"}
    return {"status": "unmatched", "detail": f"no local fighter with url {url} or name '{norm}'"}


def compare_to_local(fetched_events: list[dict], event_pages: dict[str, dict], local: dict,
                     upcoming_events: list[dict] | None = None) -> dict:
    missing_events, events_present = [], 0
    for ev in fetched_events:
        key = (normalize_event_name(ev["event_name"]), ev.get("date"))
        if key in local["event_keys"]:
            events_present += 1
        else:
            missing_events.append({"event_name": ev["event_name"], "date": ev.get("date"),
                                   "event_url": ev.get("event_url")})

    missing_fights, stale_results, missing_fighters, ambiguous, unmatched = [], [], [], [], []
    seen_fighter_urls = set()
    n_fetched_fights = 0
    for ev in fetched_events:
        page = event_pages.get(ev.get("event_url"))
        if not page:
            continue
        for fight in page["fights"]:
            n_fetched_fights += 1
            a_url, b_url = fight["fighter_a"]["url"], fight["fighter_b"]["url"]
            url_key = (ev.get("date"), frozenset({a_url, b_url}))
            name_key = (ev.get("date"), frozenset({
                normalize_fighter_name(fight["fighter_a"]["name"]),
                normalize_fighter_name(fight["fighter_b"]["name"])}))
            matched_url = url_key in local["fight_url_keys"]
            matched_name = name_key in local["fight_name_keys"]
            label = f"{fight['fighter_a']['name']} vs {fight['fighter_b']['name']} ({ev.get('date')})"
            if not matched_url and not matched_name:
                missing_fights.append({"matchup": label, "event": ev["event_name"], "fight_url": fight.get("fight_url")})
            elif fight["is_completed"]:
                local_winner = (local["result_by_url_key"].get(url_key) if matched_url
                                else local["result_by_name_key"].get(name_key))
                if matched_url:
                    fetched_winner = fight["winner_url"]
                else:
                    fetched_winner = normalize_fighter_name(fight["winner_name"]) if fight["winner_name"] else None
                mismatch = local_winner != fetched_winner
                if mismatch:
                    stale_results.append({"matchup": label, "event": ev["event_name"],
                                          "local_winner_missing": local_winner is None,
                                          "matched_by": "url" if matched_url else "name"})
        for fighter in page["fighters"]:
            if fighter["url"] in seen_fighter_urls:
                continue
            seen_fighter_urls.add(fighter["url"])
            m = match_local_fighter(fighter, local)
            if m["status"] == "unmatched":
                missing_fighters.append({"name": fighter["name"], "url": fighter["url"], "detail": m["detail"]})
            elif m["status"] == "ambiguous":
                ambiguous.append({"name": fighter["name"], "url": fighter["url"], "detail": m["detail"]})

    upcoming_report = []
    for ev in (upcoming_events or []):
        key = (normalize_event_name(ev["event_name"]), ev.get("date"))
        upcoming_report.append({"event_name": ev["event_name"], "date": ev.get("date"),
                                "event_url": ev.get("event_url"), "already_in_local_db": key in local["event_keys"]})

    return {
        "n_fetched_fights": n_fetched_fights,
        "n_unique_fetched_fighters": len(seen_fighter_urls),
        "events_present_locally": events_present,
        "missing_local_events": missing_events,
        "missing_local_fights": missing_fights,
        "missing_local_fighters": missing_fighters,
        "stale_or_mismatched_results": stale_results,
        "ambiguous_fighter_names": ambiguous,
        "unmatched_fighter_names": unmatched,
        "upcoming_events": upcoming_report,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_audit(
    db_path: str = "data/ufc.db",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_events: int = 10,
    include_upcoming: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    cache_dir: str | None = None,
    offline_cache_only: bool = False,
    delay: float = DEFAULT_REQUEST_DELAY,
    fetcher=None,
) -> dict:
    fetch = fetcher or make_fetcher(cache_dir, offline_cache_only, delay)
    pages_fetched, fetch_meta = [], []
    local = read_local_db(db_path)

    def _fetch(url):
        res = fetch(url)
        meta = {k: v for k, v in res.items() if k != "html"}
        if res.get("html"):
            meta["page_type"] = detect_page_type(res["html"])
            pages_fetched.append(url)
        fetch_meta.append(meta)
        return res.get("html")

    events_html = _fetch(COMPLETED_EVENTS_URL)
    fetched_events = parse_events_list(events_html) if events_html else []
    if start_date:
        fetched_events = [e for e in fetched_events if (e.get("date") or "9999") >= start_date]
    if end_date:
        fetched_events = [e for e in fetched_events if (e.get("date") or "0000") <= end_date]
    fetched_events = fetched_events[:max_events]

    event_pages = {}
    for ev in fetched_events:
        html = _fetch(ev["event_url"])
        if html:
            event_pages[ev["event_url"]] = parse_event_page(html)

    upcoming_events = []
    if include_upcoming:
        up_html = _fetch(UPCOMING_EVENTS_URL)
        upcoming_events = parse_events_list(up_html) if up_html else []
        upcoming_events = upcoming_events[:max_events]
        for ev in upcoming_events:
            html = _fetch(ev["event_url"])
            if html:
                event_pages[ev["event_url"]] = parse_event_page(html)

    comparison = compare_to_local(fetched_events, event_pages, local, upcoming_events)

    fetch_ok = any(m["source"] in ("cache", "live") for m in fetch_meta)
    blocked = [m for m in fetch_meta if m["source"] in ("blocked", "offline_miss", "error", "cache_challenge")]
    n_fetched_fighters = sum(len(p["fighters"]) for p in event_pages.values())

    # ---- Cache-source quality (6D.1b): where the HTML came from + what is missing ----
    page_types_from_fetch = {}
    for m in fetch_meta:
        pt = m.get("page_type", "no_content")
        page_types_from_fetch[pt] = page_types_from_fetch.get(pt, 0) + 1
    parse_warnings = []
    if events_html is not None and not fetched_events:
        parse_warnings.append("the completed-events page returned HTML but parsed 0 events (stale/altered markup?).")
    for ev in fetched_events:
        page = event_pages.get(ev["event_url"])
        if page is not None and not page["fights"]:
            parse_warnings.append(f"event page parsed 0 fights: {ev['event_name']} ({ev.get('event_url')}).")
    cache_source_quality = {
        "offline_cache_only": offline_cache_only,
        "cache_dir": cache_dir,
        "cache_files_used": [m.get("cache_file") for m in fetch_meta
                             if m.get("source") in ("cache", "cache_challenge") and m.get("cache_file")],
        "expected_missing": [
            {"url": m["url"], "suggested_filename": m.get("suggested_filename") or suggested_cache_filename(m["url"]),
             "reason": m["source"]}
            for m in fetch_meta if m["source"] in ("offline_miss", "blocked", "error", "cache_challenge")
        ],
        "page_types_detected": page_types_from_fetch,
        "challenge_pages_detected": [m["url"] for m in fetch_meta
                                     if m["source"] in ("blocked", "cache_challenge") or m.get("page_type") == "browser_challenge"],
        "parse_warnings": parse_warnings,
        "cache_validation": validate_cache(cache_dir) if cache_dir else None,
    }
    cache_source_quality["recommendation"] = (
        cache_source_quality["cache_validation"]["recommendation"] if cache_source_quality["cache_validation"]
        else "Provide --cache-dir with browser-saved UFCStats HTML and run with --offline-cache-only.")

    report = {
        "generated_at": _now_utc(),
        "run": "step6d1_ufcstats_audit",
        "mode": "read_only_dry_run",
        "db_writes_performed": False,
        "database_modified": False,
        "local_db_path": db_path,
        "db_schema_summary": local["schema"],
        "db_schema_assumptions": [
            "There is no dedicated events table; local events are DISTINCT (event, date) from the fights table.",
            "fighter_source_ids.source_fighter_url holds stable UFCStats fighter URLs (preferred match key).",
            "Fights/events have no stored UFCStats URLs, so they are matched by (date, fighter set) and (name, date).",
            "winner_id on fights gives the local result; a NULL winner_id means no local result recorded.",
        ],
        "local_counts": local["counts"],
        "scraping": {
            "base_url": BASE_URL, "user_agent": USER_AGENT, "request_delay_seconds": delay,
            "cache_dir": cache_dir, "offline_cache_only": offline_cache_only, "max_events": max_events,
            "include_upcoming": include_upcoming, "start_date": start_date, "end_date": end_date,
        },
        "pages_fetched": pages_fetched,
        "fetch_details": fetch_meta,
        "fetch_available": fetch_ok,
        "blocked_or_unavailable_fetches": blocked,
        "n_fetched_events": len(fetched_events),
        "n_fetched_fights": comparison["n_fetched_fights"],
        "n_fetched_fighters": n_fetched_fighters,
        "n_upcoming_events": len(upcoming_events),
        **{k: v for k, v in comparison.items() if k != "n_fetched_fights"},
        "cache_source_quality": cache_source_quality,
        "recommended_next_action": _recommend(fetch_ok, blocked, comparison),
        "no_write_statement": (
            "This audit is READ-ONLY. It opened data/ufc.db in SQLite mode=ro and performed NO inserts, "
            "updates, or deletes. Applying changes is deferred to a future, explicitly approved Step 6D.2."
        ),
        "limitations": [
            "UFCStats serves a JavaScript browser-challenge to plain HTTP clients; this module does not bypass it. "
            "Run with --cache-dir pointing at HTML saved from a real browser / headless fetch, or --offline-cache-only.",
            "Events/fights are name+date matched (no stored UFCStats URLs), so renamed events can look 'missing'.",
            "The winner-first parsing convention follows UFCStats' event table; verify against cached HTML if the site changes.",
            "The local DB in this repo is a synthetic dataset, so a live diff against the real site is expected to be noisy.",
        ],
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / REPORT_JSON, "w") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    (out_dir / REPORT_MD).write_text(_render_markdown(report))
    report["report_json"] = str(out_dir / REPORT_JSON)
    report["report_md"] = str(out_dir / REPORT_MD)
    return report


def _recommend(fetch_ok, blocked, comparison) -> str:
    if not fetch_ok:
        return ("No UFCStats HTML could be fetched (JS challenge / offline / no cache). Save the completed-events "
                "page and event pages from a real browser into --cache-dir and re-run with --offline-cache-only, "
                "then review before any future Step 6D.2 apply.")
    n_missing = len(comparison["missing_local_events"]) + len(comparison["missing_local_fights"])
    if n_missing == 0 and not comparison["stale_or_mismatched_results"]:
        return "Local DB appears current for the fetched range. No refresh needed now. No DB writes were made."
    return ("Local DB looks stale: review missing events/fights/fighters and any mismatched results below, then "
            "consider a guarded Step 6D.2 apply (not implemented yet). No DB writes were made in this audit.")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _render_markdown(report: dict) -> str:
    lines = ["# Step 6D.1: UFCStats Data-Update Audit (read-only)", "",
             f"Generated: {report['generated_at']}", "",
             "**Read-only dry run.** No inserts/updates/deletes were performed on `data/ufc.db`. "
             "A guarded apply is deferred to a future Step 6D.2.", ""]
    lines.append(f"- Local DB: `{report['local_db_path']}` — events {report['local_counts']['events']}, "
                 f"fights {report['local_counts']['fights']}, fighters {report['local_counts']['fighters']} "
                 f"({report['local_counts']['fighter_urls']} with stable UFCStats URLs)")
    lines.append(f"- Fetch available: **{report['fetch_available']}** | pages fetched: {len(report['pages_fetched'])} | "
                 f"blocked/unavailable: {len(report['blocked_or_unavailable_fetches'])}")
    lines.append(f"- Fetched events: **{report['n_fetched_events']}** | fights: **{report['n_fetched_fights']}** | "
                 f"fighters: **{report['n_fetched_fighters']}** | upcoming: {report['n_upcoming_events']}")
    lines.append("")
    lines.append(f"> **Recommendation:** {report['recommended_next_action']}")
    lines.append("")
    lines.append(f"_{report['no_write_statement']}_")
    lines.append("")

    lines.append("## DB Schema Summary")
    lines.append("")
    for table, info in report["db_schema_summary"].items():
        lines.append(f"- **{table}** ({info['row_count']} rows): {', '.join(info['columns'])}")
    lines.append("")
    lines.append("Assumptions:")
    for a in report["db_schema_assumptions"]:
        lines.append(f"- {a}")
    lines.append("")

    def _section(title, items, render):
        lines.append(f"## {title} ({len(items)})")
        lines.append("")
        if not items:
            lines.append("- none")
        for it in items[:40]:
            lines.append(f"- {render(it)}")
        lines.append("")

    _section("Missing Local Events", report["missing_local_events"],
             lambda e: f"{e['date']} — {e['event_name']} ({e.get('event_url')})")
    _section("Missing Local Fights", report["missing_local_fights"],
             lambda f: f"{f['matchup']} — {f['event']}")
    _section("Missing Local Fighters", report["missing_local_fighters"],
             lambda f: f"{f['name']} ({f['url']})")
    _section("Stale / Mismatched Results", report["stale_or_mismatched_results"],
             lambda r: f"{r['matchup']} — {r['event']} (local result missing: {r['local_winner_missing']}, matched by {r['matched_by']})")
    _section("Ambiguous Fighter Names", report["ambiguous_fighter_names"],
             lambda a: f"{a['name']}: {a['detail']}")
    _section("Unmatched Fighter Names", report["unmatched_fighter_names"],
             lambda u: f"{u.get('name')}: {u.get('detail')}")
    if report["scraping"]["include_upcoming"]:
        _section("Upcoming Events", report["upcoming_events"],
                 lambda e: f"{e['date']} — {e['event_name']} (already local: {e['already_in_local_db']})")

    csq = report.get("cache_source_quality") or {}
    lines.append("## Cache-Source Quality")
    lines.append("")
    lines.append(f"- offline_cache_only: **{csq.get('offline_cache_only')}** | cache_dir: `{csq.get('cache_dir')}`")
    lines.append(f"- cache files used: {csq.get('cache_files_used')}")
    lines.append(f"- page types detected: {csq.get('page_types_detected')}")
    lines.append(f"- challenge pages detected: {len(csq.get('challenge_pages_detected') or [])}")
    if csq.get("expected_missing"):
        lines.append("- **Missing pages to save next:**")
        for m in csq["expected_missing"][:15]:
            lines.append(f"  - save `{m['suggested_filename']}` for {m['url']} ({m['reason']})")
    for w in (csq.get("parse_warnings") or []):
        lines.append(f"- ⚠ parse warning: {w}")
    cv = csq.get("cache_validation")
    if cv:
        lines.append(f"- cache readiness: **{cv.get('ready_for_offline_audit')}** — {cv.get('recommendation')}")
    lines.append("")

    lines.append("## Fetch Details")
    lines.append("")
    for m in report["fetch_details"][:30]:
        lines.append(f"- [{m['source']}/{m['status']}] {m['url']} ({m['bytes']} bytes)")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for lim in report["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")
    return "\n".join(lines)
