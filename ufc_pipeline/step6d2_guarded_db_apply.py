"""Step 6D.2: guarded insert-only DB apply for one cached UFCStats event.

This module is deliberately narrow:

* It reads one browser-saved UFCStats event-detail HTML file.
* It plans inserts for that event only.
* Dry-run is the default and opens the DB read-only.
* ``apply=True`` is required for writes, and creates a DB backup first.
* It never updates or deletes existing rows.

The first supported use case is adding a missing completed event after the
current end of the local history DB, so Step 6C can later build upcoming-card
features against a refreshed history without rewriting official processed
outputs.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from ufc_pipeline.columns import name_key
from ufc_pipeline.elo import DEFAULT_K, DEFAULT_STARTING_ELO, run_elo
from ufc_pipeline.step6c_upcoming_feature_builder import run_build
from ufc_pipeline.step6d_ufcstats_update_audit import (
    EVENT_DETAILS_RE,
    FIGHT_DETAILS_RE,
    FIGHTER_DETAILS_RE,
    detect_page_type,
    normalize_event_name,
    parse_ufcstats_date,
)

APPLY_SOURCE = "ufcstats_step6d2"
REPORT_STEM = "step6d2_cached_event_apply"
OPTIONAL_PAGE_FIELDS_NOT_AVAILABLE = [
    "sig_str_attempted",
    "td_attempted",
    "ctrl_seconds",
    "fighter_a_age",
    "fighter_b_age",
    "fighter_a_height",
    "fighter_b_height",
    "fighter_a_reach",
    "fighter_b_reach",
    "fight_url",
    "ufcstats_fight_id",
    "method_detail",
]

REQUIRED_SCHEMA = {
    "fighters": {
        "fighter_id", "name", "normalized_name", "height", "reach",
        "stance", "date_of_birth",
    },
    "fighter_source_ids": {
        "fighter_id", "source", "source_fighter_id", "source_fighter_url",
        "raw_name", "normalized_name", "resolution", "resolution_note",
        "first_seen_date", "last_seen_date",
    },
    "fights": {
        "fight_id", "date", "event", "fighter_a_id", "fighter_b_id",
        "winner_id", "fighter_a_won", "weight_class", "method", "round",
        "source", "source_row_id", "fighter_a_age", "fighter_b_age",
        "fighter_a_height", "fighter_b_height", "fighter_a_reach",
        "fighter_b_reach",
    },
    "fight_stats": {
        "fight_id", "fighter_id", "minutes", "knockdowns",
        "sig_str_landed", "sig_str_attempted", "td_landed",
        "td_attempted", "sub_attempts", "ctrl_seconds", "source",
    },
    "fight_elo_snapshots": {
        "fight_id", "date", "fighter_a_id", "fighter_b_id",
        "fighter_a_pre_elo", "fighter_b_pre_elo", "elo_diff",
        "fighter_a_expected_win_prob", "fighter_b_expected_win_prob",
        "fighter_a_post_elo", "fighter_b_post_elo", "fighter_a_won",
    },
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _clean_text(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _readonly_connect(db_path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)


def _write_connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _split_cell(cell: str | None) -> list[str]:
    return [p.strip() for p in (cell or "").split("|") if p.strip()]


def _to_int(value) -> int | None:
    value = _clean_text(value)
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _fight_minutes(round_value: str | None, time_value: str | None) -> float | None:
    round_int = _to_int(round_value)
    if round_int is None or round_int <= 0:
        return None
    m = re.match(r"^\s*(\d+):(\d{2})\s*$", str(time_value or ""))
    if not m:
        return None
    minutes = int(m.group(1))
    seconds = int(m.group(2))
    return float((round_int - 1) * 5 + minutes + seconds / 60.0)


def _method_from_raw(raw: str | None) -> tuple[str | None, str | None]:
    parts = _split_cell(raw)
    if not parts:
        return None, None
    primary = parts[0]
    detail = " / ".join(parts[1:]) if len(parts) > 1 else None
    method_map = {
        "U-DEC": "Decision - Unanimous",
        "S-DEC": "Decision - Split",
        "M-DEC": "Decision - Majority",
        "KO/TKO": "KO/TKO",
        "SUB": "Submission",
        "DQ": "DQ",
        "TKO - Doctor's Stoppage": "TKO - Doctor's Stoppage",
    }
    return method_map.get(primary, primary), detail


class _DetailedEventParser(HTMLParser):
    """Extract event metadata and event-table rows from a UFCStats detail page."""

    def __init__(self):
        super().__init__()
        self.event_title: str | None = None
        self.info_items: list[str] = []
        self.rows: list[dict] = []
        self._in_title = False
        self._title_buf: list[str] = []
        self._in_info_item = False
        self._info_buf: list[str] = []
        self._row: dict | None = None
        self._in_cell = False
        self._cell_buf: list[str] = []
        self._in_fighter_anchor = False
        self._fighter_url: str | None = None
        self._fighter_buf: list[str] = []
        self._in_flag = False
        self._flag_buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        cls = attrs_d.get("class") or ""
        if tag in ("h2", "span") and "b-content__title-highlight" in cls:
            self._in_title = True
            self._title_buf = []
        if tag == "li":
            self._in_info_item = True
            self._info_buf = []
        if tag == "tr" and attrs_d.get("data-link") and FIGHT_DETAILS_RE.search(attrs_d["data-link"]):
            self._row = {
                "fight_url": attrs_d["data-link"].strip(),
                "cells": [],
                "fighters": [],
                "flags": [],
            }
        if self._row is not None and tag == "td":
            self._in_cell = True
            self._cell_buf = []
        if self._row is not None and tag == "a" and attrs_d.get("href") and FIGHTER_DETAILS_RE.search(attrs_d["href"]):
            self._in_fighter_anchor = True
            self._fighter_url = attrs_d["href"].strip()
            self._fighter_buf = []
        if self._row is not None and tag == "i" and "b-flag__text" in cls:
            self._in_flag = True
            self._flag_buf = []

    def handle_data(self, data):
        text = _clean_text(data)
        if not text:
            return
        if self._in_title:
            self._title_buf.append(text)
        if self._in_info_item:
            self._info_buf.append(text)
        if self._in_cell:
            self._cell_buf.append(text)
        if self._in_fighter_anchor:
            self._fighter_buf.append(text)
        if self._in_flag:
            self._flag_buf.append(text)

    def handle_endtag(self, tag):
        if tag in ("h2", "span") and self._in_title:
            self.event_title = _clean_text(" ".join(self._title_buf)) or self.event_title
            self._in_title = False
        if tag == "li" and self._in_info_item:
            item = _clean_text(" ".join(self._info_buf))
            if item:
                self.info_items.append(item)
            self._in_info_item = False
        if tag == "a" and self._in_fighter_anchor and self._row is not None:
            name = _clean_text(" ".join(self._fighter_buf))
            url = self._fighter_url
            m = FIGHTER_DETAILS_RE.search(url or "")
            if name and url and m:
                self._row["fighters"].append({
                    "name": name,
                    "url": url,
                    "fighter_id": m.group(1),
                })
            self._in_fighter_anchor = False
            self._fighter_url = None
        if tag == "i" and self._in_flag and self._row is not None:
            flag = _clean_text(" ".join(self._flag_buf)).lower()
            if flag:
                self._row["flags"].append(flag)
            self._in_flag = False
        if tag == "td" and self._in_cell and self._row is not None:
            self._row["cells"].append(_clean_text(" | ".join(self._cell_buf)))
            self._in_cell = False
        if tag == "tr" and self._row is not None:
            if self._row["fighters"] or self._row["cells"]:
                self.rows.append(self._row)
            self._row = None


def parse_cached_event_detail(html: str, event_cache: str | Path | None = None) -> dict:
    """Parse a cached UFCStats event page into DB-apply ready fields."""
    parser = _DetailedEventParser()
    parser.feed(html or "")
    event_id = None
    event_url = None
    if event_cache:
        m = re.search(r"event_([a-zA-Z0-9]+)", Path(event_cache).name)
        if m:
            event_id = m.group(1)
            event_url = f"http://ufcstats.com/event-details/{event_id}"
    if event_id is None:
        m = EVENT_DETAILS_RE.search(html or "")
        if m:
            event_id = m.group(1)
            event_url = f"http://ufcstats.com/event-details/{event_id}"

    date_raw = None
    location = None
    for item in parser.info_items:
        m = re.search(r"\bDate:\s*(.+)$", item, re.I)
        if m:
            date_raw = m.group(1).strip()
        m = re.search(r"\bLocation:\s*(.+)$", item, re.I)
        if m:
            location = m.group(1).strip()

    fights = []
    fighters = []
    seen_urls = set()
    for idx, row in enumerate(parser.rows, start=1):
        row_fighters = row.get("fighters") or []
        if len(row_fighters) < 2:
            fights.append({
                "row_index": idx,
                "status": "blocked",
                "block_reason": "row did not contain two UFCStats fighter-detail links",
                "fight_url": row.get("fight_url"),
            })
            continue
        a, b = row_fighters[0], row_fighters[1]
        cells = row.get("cells") or []
        kd_a, kd_b = _pair_from_cell(cells, 2)
        str_a, str_b = _pair_from_cell(cells, 3)
        td_a, td_b = _pair_from_cell(cells, 4)
        sub_a, sub_b = _pair_from_cell(cells, 5)
        weight_class = cells[6] if len(cells) > 6 else None
        method, method_detail = _method_from_raw(cells[7] if len(cells) > 7 else None)
        round_value = cells[8] if len(cells) > 8 else None
        time_value = cells[9] if len(cells) > 9 else None
        status = (row.get("flags") or [None])[0]
        fight_url = row.get("fight_url")
        m = FIGHT_DETAILS_RE.search(fight_url or "")
        winner_url = a["url"] if status == "win" else None
        winner_name = a["name"] if status == "win" else None
        fight = {
            "row_index": idx,
            "fight_id": m.group(1) if m else None,
            "fight_url": fight_url,
            "fighter_a": a,
            "fighter_b": b,
            "result_status": status,
            "winner_url": winner_url,
            "winner_name": winner_name,
            "is_completed": status in ("win", "draw", "nc", "no contest"),
            "weight_class": weight_class,
            "method": method,
            "method_detail": method_detail,
            "round": str(_to_int(round_value)) if _to_int(round_value) is not None else None,
            "time": time_value,
            "minutes": _fight_minutes(round_value, time_value),
            "stats": {
                "fighter_a": {
                    "knockdowns": kd_a,
                    "sig_str_landed": str_a,
                    "sig_str_attempted": None,
                    "td_landed": td_a,
                    "td_attempted": None,
                    "sub_attempts": sub_a,
                    "ctrl_seconds": None,
                },
                "fighter_b": {
                    "knockdowns": kd_b,
                    "sig_str_landed": str_b,
                    "sig_str_attempted": None,
                    "td_landed": td_b,
                    "td_attempted": None,
                    "sub_attempts": sub_b,
                    "ctrl_seconds": None,
                },
            },
        }
        fights.append(fight)
        for fighter in (a, b):
            if fighter["url"] not in seen_urls:
                seen_urls.add(fighter["url"])
                fighters.append(fighter)

    return {
        "event_id": event_id,
        "event_url": event_url,
        "event_title": parser.event_title,
        "date_raw": date_raw,
        "date": parse_ufcstats_date(date_raw),
        "location": location,
        "fights": fights,
        "fighters": fighters,
        "page_type": detect_page_type(html, filename=Path(event_cache).name if event_cache else None),
    }


def _pair_from_cell(cells: list[str], idx: int) -> tuple[int | None, int | None]:
    if len(cells) <= idx:
        return None, None
    parts = _split_cell(cells[idx])
    if len(parts) < 2:
        return None, None
    return _to_int(parts[0]), _to_int(parts[1])


def _schema_state(conn: sqlite3.Connection) -> dict:
    tables = {}
    for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        tables[table] = {"columns": cols, "row_count": int(count)}
    missing = []
    for table, required in REQUIRED_SCHEMA.items():
        have = set(tables.get(table, {}).get("columns", []))
        if table not in tables:
            missing.append({"table": table, "missing_columns": sorted(required)})
        else:
            miss = sorted(required - have)
            if miss:
                missing.append({"table": table, "missing_columns": miss})
    return {"tables": tables, "missing_requirements": missing, "ok": not missing}


def _read_db_state(conn: sqlite3.Connection) -> dict:
    schema = _schema_state(conn)
    linked_by_url: dict[str, list[dict]] = {}
    for row in conn.execute(
        """SELECT s.source_fighter_url, s.source, s.source_fighter_id,
                  f.fighter_id, f.name, f.normalized_name
           FROM fighter_source_ids s
           JOIN fighters f ON f.fighter_id = s.fighter_id
           WHERE s.fighter_id IS NOT NULL AND s.source_fighter_url IS NOT NULL"""
    ):
        url, source, source_id, fid, name, norm = row
        linked_by_url.setdefault(url, []).append({
            "source": source,
            "source_fighter_id": source_id,
            "fighter_id": int(fid),
            "name": name,
            "normalized_name": norm,
        })

    source_rows = {}
    for row in conn.execute(
        """SELECT source_fighter_id, fighter_id, source_fighter_url, raw_name,
                  normalized_name, resolution
           FROM fighter_source_ids
           WHERE source = ?""",
        (APPLY_SOURCE,),
    ):
        source_id, fid, url, raw_name, norm, resolution = row
        source_rows[source_id] = {
            "fighter_id": int(fid) if fid is not None else None,
            "source_fighter_url": url,
            "raw_name": raw_name,
            "normalized_name": norm,
            "resolution": resolution,
        }

    unlinked_urls = {}
    for row in conn.execute(
        """SELECT source, source_fighter_id, source_fighter_url, raw_name,
                  normalized_name, resolution
           FROM fighter_source_ids
           WHERE fighter_id IS NULL AND source_fighter_url IS NOT NULL"""
    ):
        source, source_id, url, raw_name, norm, resolution = row
        unlinked_urls.setdefault(url, []).append({
            "source": source,
            "source_fighter_id": source_id,
            "raw_name": raw_name,
            "normalized_name": norm,
            "resolution": resolution,
        })

    fighters_by_name: dict[str, list[dict]] = {}
    for fid, name, _norm in conn.execute("SELECT fighter_id, name, normalized_name FROM fighters"):
        normalized_name = name_key(name)
        fighters_by_name.setdefault(normalized_name, []).append({
            "fighter_id": int(fid),
            "name": name,
            "normalized_name": normalized_name,
        })

    event_keys = set()
    for event, fight_date in conn.execute("SELECT DISTINCT event, date FROM fights"):
        event_keys.add((normalize_event_name(event), str(fight_date)))

    fight_pairs = {}
    for fight_id, fight_date, event, a_id, b_id in conn.execute(
        "SELECT fight_id, date, event, fighter_a_id, fighter_b_id FROM fights"
    ):
        fight_pairs[(str(fight_date), frozenset({int(a_id), int(b_id)}))] = {
            "fight_id": int(fight_id),
            "event": event,
        }

    max_date = conn.execute("SELECT MAX(date) FROM fights").fetchone()[0]
    max_fight_id = conn.execute("SELECT MAX(fight_id) FROM fights").fetchone()[0]
    return {
        "schema": schema,
        "linked_by_url": linked_by_url,
        "source_rows": source_rows,
        "unlinked_urls": unlinked_urls,
        "fighters_by_name": fighters_by_name,
        "event_keys": event_keys,
        "fight_pairs": fight_pairs,
        "max_existing_fight_date": max_date,
        "max_existing_fight_id": int(max_fight_id or 0),
        "counts": {
            table: info["row_count"]
            for table, info in schema["tables"].items()
            if table in REQUIRED_SCHEMA
        },
    }


def _resolve_fighter(fighter: dict, state: dict) -> dict:
    name = _clean_text(fighter.get("name"))
    url = _clean_text(fighter.get("url"))
    source_id = _clean_text(fighter.get("fighter_id"))
    norm = name_key(name)
    if not name or not url or not source_id or not norm:
        return {
            "status": "blocked_missing_required_fields",
            "name": name,
            "url": url,
            "source_fighter_id": source_id,
            "normalized_name": norm,
            "reason": "fighter name, URL, source id, and normalized name are required",
        }

    linked = state["linked_by_url"].get(url, [])
    linked_ids = sorted({r["fighter_id"] for r in linked})
    if len(linked_ids) == 1:
        row = linked[0]
        return {
            "status": "existing_by_url",
            "fighter_id": linked_ids[0],
            "name": name,
            "url": url,
            "source_fighter_id": source_id,
            "normalized_name": norm,
            "matched_existing_name": row["name"],
            "source_identity_action": "skip_existing_linked_url",
        }
    if len(linked_ids) > 1:
        return {
            "status": "blocked_ambiguous_url",
            "name": name,
            "url": url,
            "source_fighter_id": source_id,
            "normalized_name": norm,
            "reason": f"UFCStats URL is linked to multiple local fighter ids: {linked_ids}",
        }

    apply_row = state["source_rows"].get(source_id)
    if apply_row and apply_row["fighter_id"] is None:
        return {
            "status": "blocked_unlinked_apply_source_identity",
            "name": name,
            "url": url,
            "source_fighter_id": source_id,
            "normalized_name": norm,
            "reason": f"{APPLY_SOURCE} source identity exists but is unlinked; insert-only apply will not update it",
        }

    matches = state["fighters_by_name"].get(norm, [])
    if len(matches) == 1:
        return {
            "status": "existing_by_exact_name",
            "fighter_id": matches[0]["fighter_id"],
            "name": name,
            "url": url,
            "source_fighter_id": source_id,
            "normalized_name": norm,
            "matched_existing_name": matches[0]["name"],
            "source_identity_action": "insert_linked_source_identity",
            "unlinked_source_identities": state["unlinked_urls"].get(url, []),
        }
    if len(matches) > 1:
        return {
            "status": "blocked_ambiguous_name",
            "name": name,
            "url": url,
            "source_fighter_id": source_id,
            "normalized_name": norm,
            "reason": f"{len(matches)} local fighters share normalized name '{norm}'",
            "candidate_fighter_ids": [m["fighter_id"] for m in matches],
        }

    return {
        "status": "insert_new_fighter",
        "planned_fighter_ref": f"new:{source_id}",
        "name": name,
        "url": url,
        "source_fighter_id": source_id,
        "normalized_name": norm,
        "source_identity_action": "insert_new_linked_source_identity",
        "unlinked_source_identities": state["unlinked_urls"].get(url, []),
    }


def build_apply_plan(db_path: str, event_cache: str, output_dir: str = "reports/data_update") -> dict:
    event_path = Path(event_cache)
    html = event_path.read_text(encoding="utf-8", errors="ignore")
    event = parse_cached_event_detail(html, event_cache=event_path)
    conn = _readonly_connect(db_path)
    try:
        state = _read_db_state(conn)
    finally:
        conn.close()
    return _build_plan_from_state(db_path, event_cache, output_dir, event, state)


def _build_plan_from_state(db_path: str, event_cache: str, output_dir: str, event: dict, state: dict) -> dict:
    required_missing = []
    blocks = []
    warnings = []
    if event["page_type"] != "event_detail":
        required_missing.append({"field": "page_type", "detail": f"expected event_detail, got {event['page_type']}"})
    if not event.get("event_title"):
        required_missing.append({"field": "event_title", "detail": "event title was not parsed"})
    if not event.get("date"):
        required_missing.append({"field": "event_date", "detail": "event date was not parsed"})
    if not event.get("event_id"):
        required_missing.append({"field": "event_id", "detail": "event id was not found from filename or HTML"})
    parsed_fights = [f for f in event.get("fights", []) if f.get("fighter_a") and f.get("fighter_b")]
    if not parsed_fights:
        required_missing.append({"field": "fights", "detail": "no usable fight rows were parsed"})

    if not state["schema"]["ok"]:
        blocks.append({"type": "schema", "detail": state["schema"]["missing_requirements"]})

    event_date = event.get("date")
    max_date = state.get("max_existing_fight_date")
    if event_date and max_date and event_date < max_date:
        blocks.append({
            "type": "non_append_history",
            "detail": (
                f"selected event date {event_date} is before existing max fight date {max_date}; "
                "correct Elo would require updating existing future snapshots, which Step 6D.2 prototype forbids"
            ),
        })
    elif event_date and max_date and event_date == max_date:
        warnings.append(
            "selected event has the same date as existing fights; new fight_ids will be appended after existing same-date fights"
        )

    fighter_resolutions = {}
    fighters_to_insert = []
    source_ids_to_insert = []
    skipped_source_ids = []
    blocked_fighters = []
    for fighter in event.get("fighters", []):
        res = _resolve_fighter(fighter, state)
        fighter_resolutions[fighter["url"]] = res
        if res["status"].startswith("blocked"):
            blocked_fighters.append(res)
            continue
        if res["status"] == "insert_new_fighter":
            fighters_to_insert.append({
                "name": res["name"],
                "normalized_name": res["normalized_name"],
                "source_fighter_id": res["source_fighter_id"],
                "source_fighter_url": res["url"],
                "planned_fighter_ref": res["planned_fighter_ref"],
                "unlinked_source_identities_left_unchanged": res.get("unlinked_source_identities", []),
            })
        if res.get("source_identity_action", "").startswith("insert"):
            source_ids_to_insert.append({
                "source": APPLY_SOURCE,
                "source_fighter_id": res["source_fighter_id"],
                "source_fighter_url": res["url"],
                "raw_name": res["name"],
                "normalized_name": res["normalized_name"],
                "fighter_id": res.get("fighter_id"),
                "planned_fighter_ref": res.get("planned_fighter_ref"),
                "resolution": "linked",
                "first_seen_date": event_date,
                "last_seen_date": event_date,
            })
        else:
            skipped_source_ids.append({
                "source_fighter_id": res["source_fighter_id"],
                "source_fighter_url": res["url"],
                "reason": res.get("source_identity_action"),
                "fighter_id": res.get("fighter_id"),
            })

    fights_to_insert = []
    fight_stats_to_insert = []
    fight_elo_snapshots_to_insert = []
    skipped_fights = []
    blocked_fights = []
    planned_next_fight_id = state["max_existing_fight_id"]
    for fight in event.get("fights", []):
        if not (fight.get("fighter_a") and fight.get("fighter_b")):
            blocked_fights.append({
                "row_index": fight.get("row_index"),
                "fight_url": fight.get("fight_url"),
                "reason": fight.get("block_reason", "missing fighter links"),
            })
            continue
        reasons = []
        if fight.get("result_status") != "win":
            reasons.append(f"non-decisive or unsupported result status: {fight.get('result_status')}")
        if fight.get("minutes") is None:
            reasons.append("round/time could not be converted into fight minutes required by fight_stats")
        for side in ("fighter_a", "fighter_b"):
            res = fighter_resolutions.get(fight[side]["url"])
            if not res or res["status"].startswith("blocked"):
                reasons.append(f"{side} identity blocked: {(res or {}).get('reason', 'unresolved')}")
        if reasons:
            blocked_fights.append({
                "row_index": fight.get("row_index"),
                "matchup": _matchup_label(fight),
                "fight_url": fight.get("fight_url"),
                "reasons": reasons,
            })
            continue
        res_a = fighter_resolutions[fight["fighter_a"]["url"]]
        res_b = fighter_resolutions[fight["fighter_b"]["url"]]
        a_id = res_a.get("fighter_id")
        b_id = res_b.get("fighter_id")
        existing = None
        if a_id is not None and b_id is not None and event_date:
            existing = state["fight_pairs"].get((event_date, frozenset({a_id, b_id})))
        if existing:
            skipped_fights.append({
                "matchup": _matchup_label(fight),
                "reason": "fight already exists for date and fighter pair",
                "existing_fight_id": existing["fight_id"],
                "existing_event": existing["event"],
            })
            continue
        planned_next_fight_id += 1
        planned_fight_ref = f"new_fight:{planned_next_fight_id}"
        fight_row = {
            "planned_fight_ref": planned_fight_ref,
            "estimated_fight_id": planned_next_fight_id,
            "date": event_date,
            "event": event.get("event_title"),
            "fighter_a": fight["fighter_a"]["name"],
            "fighter_b": fight["fighter_b"]["name"],
            "fighter_a_id": a_id,
            "fighter_b_id": b_id,
            "fighter_a_ref": res_a.get("planned_fighter_ref"),
            "fighter_b_ref": res_b.get("planned_fighter_ref"),
            "winner": fight["winner_name"],
            "winner_id": a_id,
            "winner_ref": res_a.get("planned_fighter_ref"),
            "fighter_a_won": 1,
            "weight_class": fight.get("weight_class"),
            "method": fight.get("method"),
            "method_detail_not_persisted": fight.get("method_detail"),
            "round": fight.get("round"),
            "time": fight.get("time"),
            "minutes": fight.get("minutes"),
            "source": APPLY_SOURCE,
            "fight_url_not_persisted": fight.get("fight_url"),
            "ufcstats_fight_id_not_persisted": fight.get("fight_id"),
        }
        fights_to_insert.append(fight_row)
        for side in ("fighter_a", "fighter_b"):
            res = res_a if side == "fighter_a" else res_b
            stats = fight["stats"][side]
            fight_stats_to_insert.append({
                "planned_fight_ref": planned_fight_ref,
                "fighter_name": fight[side]["name"],
                "fighter_id": res.get("fighter_id"),
                "fighter_ref": res.get("planned_fighter_ref"),
                "minutes": fight.get("minutes"),
                **stats,
                "source": APPLY_SOURCE,
            })
        fight_elo_snapshots_to_insert.append({
            "planned_fight_ref": planned_fight_ref,
            "detail": "computed during apply after actual fight_id values are assigned",
        })

    event_key = (normalize_event_name(event.get("event_title")), event_date)
    event_exists = event_key in state["event_keys"]
    target_tables = []
    if fighters_to_insert:
        target_tables.append("fighters")
    if source_ids_to_insert:
        target_tables.append("fighter_source_ids")
    if fights_to_insert:
        target_tables += ["fights", "fight_stats", "fight_elo_snapshots"]
    safe = (
        not required_missing
        and not blocks
        and not blocked_fighters
        and not blocked_fights
    )
    return {
        "generated_at": _now_utc(),
        "run": "step6d2_guarded_cached_event_apply",
        "mode": "dry_run_plan",
        "db_path": str(db_path),
        "event_cache": str(event_cache),
        "output_dir": str(output_dir),
        "apply_source": APPLY_SOURCE,
        "dry_run_default": True,
        "apply_requires_explicit_flag": True,
        "event": {
            "event_id": event.get("event_id"),
            "event_url": event.get("event_url"),
            "event_name": event.get("event_title"),
            "event_date": event_date,
            "date_raw": event.get("date_raw"),
            "location": event.get("location"),
            "page_type": event.get("page_type"),
            "parsed_fights": len(parsed_fights),
            "parsed_fighters": len(event.get("fighters", [])),
            "exists_locally_by_event_date": event_exists,
            "event_table_note": "no dedicated events table exists; event is materialized through fights rows",
        },
        "db_counts_before": state["counts"],
        "max_existing_fight_date": max_date,
        "schema_check": state["schema"],
        "required_fields_missing": required_missing,
        "optional_fields_not_available_from_event_page": OPTIONAL_PAGE_FIELDS_NOT_AVAILABLE,
        "safety_blocks": blocks,
        "safety_warnings": warnings,
        "fighter_resolution": list(fighter_resolutions.values()),
        "fighters_to_insert": fighters_to_insert,
        "fighter_source_ids_to_insert": source_ids_to_insert,
        "fighter_source_ids_skipped": skipped_source_ids,
        "fights_to_insert": fights_to_insert,
        "fight_stats_to_insert": fight_stats_to_insert,
        "fight_elo_snapshots_to_insert": fight_elo_snapshots_to_insert,
        "rows_skipped": {
            "fighter_source_ids": skipped_source_ids,
            "fights": skipped_fights,
        },
        "rows_blocked": {
            "fighters": blocked_fighters,
            "fights": blocked_fights,
        },
        "target_tables": sorted(set(target_tables)),
        "planned_row_counts": {
            "fighters": len(fighters_to_insert),
            "fighter_source_ids": len(source_ids_to_insert),
            "fights": len(fights_to_insert),
            "fight_stats": len(fight_stats_to_insert),
            "fight_elo_snapshots": len(fight_elo_snapshots_to_insert),
            "skipped_fights": len(skipped_fights),
            "blocked_fighters": len(blocked_fighters),
            "blocked_fights": len(blocked_fights),
        },
        "safe_to_apply": safe,
        "safety_summary": (
            "safe to apply with --apply" if safe else
            "blocked: review required_fields_missing, safety_blocks, and rows_blocked"
        ),
        "apply_result": None,
        "no_write_statement": (
            "Dry-run plan only. The database was opened read-only and no inserts, updates, or deletes were performed."
        ),
    }


def _matchup_label(fight: dict) -> str:
    a = (fight.get("fighter_a") or {}).get("name")
    b = (fight.get("fighter_b") or {}).get("name")
    return f"{a} vs {b}"


def backup_db(db_path: str | Path) -> str:
    src = Path(db_path)
    backup = src.with_name(f"{src.name}.step6d2_backup_{_stamp()}")
    shutil.copy2(src, backup)
    return str(backup)


def run_cached_event_update(
    db_path: str,
    event_cache: str,
    output_dir: str = "reports/data_update",
    apply: bool = False,
    run_step6c_verification: bool = True,
) -> dict:
    """Build a plan, optionally apply it, and always write JSON/MD reports."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = build_apply_plan(db_path=db_path, event_cache=event_cache, output_dir=output_dir)
    if apply and plan["safe_to_apply"]:
        apply_result = _apply_plan(db_path, event_cache, plan, out_dir, run_step6c_verification)
        plan["mode"] = "apply"
        plan["apply_result"] = apply_result
        plan["no_write_statement"] = None
    elif apply:
        plan["mode"] = "apply_blocked"
        plan["apply_result"] = {
            "apply_requested": True,
            "apply_performed": False,
            "reason": "plan was not safe to apply",
        }

    report_stem = f"{REPORT_STEM}_{_stamp()}"
    json_path = out_dir / f"{report_stem}.json"
    md_path = out_dir / f"{report_stem}.md"
    json_path.write_text(json.dumps(plan, indent=2, default=_json_default), encoding="utf-8")
    md_path.write_text(_render_markdown(plan), encoding="utf-8")
    plan["report_json"] = str(json_path)
    plan["report_md"] = str(md_path)
    return plan


def _apply_plan(
    db_path: str,
    event_cache: str,
    plan: dict,
    output_dir: Path,
    run_step6c_verification: bool,
) -> dict:
    backup_path = backup_db(db_path)
    conn = _write_connect(db_path)
    inserted = {
        "fighters": [],
        "fighter_source_ids": [],
        "fights": [],
        "fight_stats": [],
        "fight_elo_snapshots": [],
    }
    runtime_skipped = {"fights": [], "fighter_source_ids": []}
    try:
        before_counts = _count_required_tables(conn)
        conn.execute("BEGIN")
        schema = _schema_state(conn)
        if not schema["ok"]:
            raise RuntimeError(f"schema assumptions failed before apply: {schema['missing_requirements']}")

        fighter_ids_by_url = _current_linked_fighter_ids_by_url(conn)
        fighter_ids_by_ref = {}

        for item in plan["fighters_to_insert"]:
            cur = conn.execute(
                "INSERT INTO fighters (name, normalized_name) VALUES (?, ?)",
                (item["name"], item["normalized_name"]),
            )
            fid = int(cur.lastrowid)
            fighter_ids_by_ref[item["planned_fighter_ref"]] = fid
            fighter_ids_by_url[item["source_fighter_url"]] = fid
            inserted["fighters"].append({
                "fighter_id": fid,
                "name": item["name"],
                "source_fighter_url": item["source_fighter_url"],
            })

        for item in plan["fighter_source_ids_to_insert"]:
            source_id = item["source_fighter_id"]
            exists = conn.execute(
                "SELECT id, fighter_id FROM fighter_source_ids WHERE source = ? AND source_fighter_id = ?",
                (APPLY_SOURCE, source_id),
            ).fetchone()
            if exists is not None:
                runtime_skipped["fighter_source_ids"].append({
                    "source_fighter_id": source_id,
                    "reason": "source identity already exists at apply time",
                })
                continue
            fid = item.get("fighter_id")
            if fid is None:
                fid = fighter_ids_by_ref.get(item.get("planned_fighter_ref"))
            if fid is None:
                raise RuntimeError(f"could not resolve fighter_id for source identity {source_id}")
            cur = conn.execute(
                """INSERT INTO fighter_source_ids
                   (fighter_id, source, source_fighter_id, source_fighter_url,
                    raw_name, normalized_name, resolution, resolution_note,
                    first_seen_date, last_seen_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fid, APPLY_SOURCE, source_id, item["source_fighter_url"],
                    item["raw_name"], item["normalized_name"], "linked",
                    "linked by Step 6D.2 guarded cached-event apply",
                    item["first_seen_date"], item["last_seen_date"],
                ),
            )
            inserted["fighter_source_ids"].append({
                "id": int(cur.lastrowid),
                "fighter_id": fid,
                "source_fighter_id": source_id,
                "source_fighter_url": item["source_fighter_url"],
            })

        fight_id_by_ref = {}
        for item in plan["fights_to_insert"]:
            a_id = item.get("fighter_a_id") or fighter_ids_by_ref.get(item.get("fighter_a_ref"))
            b_id = item.get("fighter_b_id") or fighter_ids_by_ref.get(item.get("fighter_b_ref"))
            winner_id = item.get("winner_id") or fighter_ids_by_ref.get(item.get("winner_ref"))
            if a_id is None or b_id is None or winner_id is None:
                raise RuntimeError(f"could not resolve fighter ids for fight {item['fighter_a']} vs {item['fighter_b']}")
            existing = conn.execute(
                """SELECT fight_id, event FROM fights
                   WHERE date = ?
                     AND ((fighter_a_id = ? AND fighter_b_id = ?)
                          OR (fighter_a_id = ? AND fighter_b_id = ?))
                   LIMIT 1""",
                (item["date"], a_id, b_id, b_id, a_id),
            ).fetchone()
            if existing is not None:
                runtime_skipped["fights"].append({
                    "planned_fight_ref": item["planned_fight_ref"],
                    "existing_fight_id": int(existing[0]),
                    "existing_event": existing[1],
                })
                continue
            cur = conn.execute(
                """INSERT INTO fights
                   (date, event, fighter_a_id, fighter_b_id, winner_id,
                    fighter_a_won, weight_class, method, round, source,
                    source_row_id, fighter_a_age, fighter_b_age,
                    fighter_a_height, fighter_b_height, fighter_a_reach,
                    fighter_b_reach)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    item["date"], item["event"], a_id, b_id, winner_id,
                    int(item["fighter_a_won"]), item["weight_class"],
                    item["method"], item["round"], APPLY_SOURCE,
                    None, None, None, None, None, None, None,
                ),
            )
            fight_id = int(cur.lastrowid)
            fight_id_by_ref[item["planned_fight_ref"]] = fight_id
            inserted["fights"].append({
                "fight_id": fight_id,
                "date": item["date"],
                "event": item["event"],
                "matchup": f"{item['fighter_a']} vs {item['fighter_b']}",
            })

        for item in plan["fight_stats_to_insert"]:
            fight_id = fight_id_by_ref.get(item["planned_fight_ref"])
            if fight_id is None:
                continue
            fighter_id = item.get("fighter_id") or fighter_ids_by_ref.get(item.get("fighter_ref"))
            if fighter_id is None:
                raise RuntimeError(f"could not resolve fighter_id for fight_stats {item}")
            conn.execute(
                """INSERT INTO fight_stats
                   (fight_id, fighter_id, minutes, knockdowns,
                    sig_str_landed, sig_str_attempted, td_landed,
                    td_attempted, sub_attempts, ctrl_seconds, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fight_id, fighter_id, item["minutes"], item["knockdowns"],
                    item["sig_str_landed"], item["sig_str_attempted"],
                    item["td_landed"], item["td_attempted"], item["sub_attempts"],
                    item["ctrl_seconds"], APPLY_SOURCE,
                ),
            )
            inserted["fight_stats"].append({
                "fight_id": fight_id,
                "fighter_id": fighter_id,
                "fighter_name": item["fighter_name"],
            })

        inserted_fight_ids = {r["fight_id"] for r in inserted["fights"]}
        if inserted_fight_ids:
            fights = [
                dict(zip(("fight_id", "date", "fighter_a_id", "fighter_b_id", "fighter_a_won"), row))
                for row in conn.execute(
                    "SELECT fight_id, date, fighter_a_id, fighter_b_id, fighter_a_won FROM fights"
                )
            ]
            snapshots, _ratings = run_elo(fights, k=DEFAULT_K, starting_elo=DEFAULT_STARTING_ELO)
            for s in snapshots:
                if s.fight_id not in inserted_fight_ids:
                    continue
                conn.execute(
                    """INSERT INTO fight_elo_snapshots
                       (fight_id, date, fighter_a_id, fighter_b_id,
                        fighter_a_pre_elo, fighter_b_pre_elo, elo_diff,
                        fighter_a_expected_win_prob, fighter_b_expected_win_prob,
                        fighter_a_post_elo, fighter_b_post_elo, fighter_a_won)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        s.fight_id, s.date, s.fighter_a_id, s.fighter_b_id,
                        s.fighter_a_pre_elo, s.fighter_b_pre_elo, s.elo_diff,
                        s.fighter_a_expected_win_prob, s.fighter_b_expected_win_prob,
                        s.fighter_a_post_elo, s.fighter_b_post_elo, s.fighter_a_won,
                    ),
                )
                inserted["fight_elo_snapshots"].append({"fight_id": int(s.fight_id)})

        conn.commit()
        after_counts = _count_required_tables(conn)
        verification = _verify_apply(conn, plan, inserted)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    step6c_verification = None
    if run_step6c_verification and inserted["fights"]:
        step6c_verification = _run_step6c_verification(db_path, plan, output_dir)

    return {
        "apply_requested": True,
        "apply_performed": True,
        "backup_path": backup_path,
        "db_counts_before": before_counts,
        "db_counts_after": after_counts,
        "db_count_deltas": {
            table: after_counts.get(table, 0) - before_counts.get(table, 0)
            for table in sorted(set(before_counts) | set(after_counts))
        },
        "inserted": inserted,
        "runtime_skipped": runtime_skipped,
        "verification": verification,
        "step6c_verification": step6c_verification,
    }


def _current_linked_fighter_ids_by_url(conn: sqlite3.Connection) -> dict[str, int]:
    out = {}
    for url, fid in conn.execute(
        """SELECT source_fighter_url, fighter_id FROM fighter_source_ids
           WHERE fighter_id IS NOT NULL AND source_fighter_url IS NOT NULL"""
    ):
        out[url] = int(fid)
    return out


def _count_required_tables(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in REQUIRED_SCHEMA
    }


def _verify_apply(conn: sqlite3.Connection, plan: dict, inserted: dict) -> dict:
    event_name = plan["event"]["event_name"]
    event_date = plan["event"]["event_date"]
    event_count = conn.execute(
        "SELECT COUNT(*) FROM fights WHERE event = ? AND date = ?",
        (event_name, event_date),
    ).fetchone()[0]
    inserted_fight_ids = [r["fight_id"] for r in inserted["fights"]]
    stats_count = 0
    snapshot_count = 0
    if inserted_fight_ids:
        placeholders = ",".join("?" for _ in inserted_fight_ids)
        stats_count = conn.execute(
            f"SELECT COUNT(*) FROM fight_stats WHERE fight_id IN ({placeholders})",
            inserted_fight_ids,
        ).fetchone()[0]
        snapshot_count = conn.execute(
            f"SELECT COUNT(*) FROM fight_elo_snapshots WHERE fight_id IN ({placeholders})",
            inserted_fight_ids,
        ).fetchone()[0]
    return {
        "event_fights_found": int(event_count),
        "inserted_fight_stats_rows_found": int(stats_count),
        "inserted_elo_snapshots_found": int(snapshot_count),
        "ok": (
            int(event_count) >= len(inserted["fights"])
            and int(stats_count) == 2 * len(inserted["fights"])
            and int(snapshot_count) == len(inserted["fights"])
        ),
    }


def _run_step6c_verification(db_path: str, plan: dict, output_dir: Path) -> dict:
    first = plan["fights_to_insert"][0]
    event_date = date_cls.fromisoformat(plan["event"]["event_date"]) + timedelta(days=1)
    stamp = _stamp()
    matchups_path = output_dir / f"step6d2_step6c_verification_matchups_{stamp}.csv"
    features_path = output_dir / f"step6d2_step6c_verification_features_{stamp}.csv"
    pd.DataFrame([
        {
            "event_date": event_date.isoformat(),
            "event_name": "Step 6D.2 verification matchup",
            "fighter_a": first["fighter_a"],
            "fighter_b": first["fighter_b"],
            "weight_class": first["weight_class"],
        }
    ]).to_csv(matchups_path, index=False)
    report = run_build(
        matchups_csv=str(matchups_path),
        output=str(features_path),
        history_input=db_path,
        output_dir=str(output_dir),
        overwrite=False,
        validate_for_step6b=True,
    )
    return {
        "matchups_csv": str(matchups_path),
        "features_csv": str(features_path),
        "report_json": report.get("report_json"),
        "report_md": report.get("report_md"),
        "n_feature_rows_built": report.get("n_feature_rows_built"),
        "n_failed_rows": report.get("n_failed_rows"),
        "step6b_validation": report.get("step6b_validation"),
        "writes_data_processed": False,
    }


def _render_markdown(report: dict) -> str:
    lines = [
        "# Step 6D.2: Guarded Cached Event Apply",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"- Mode: `{report['mode']}`",
        f"- DB: `{report['db_path']}`",
        f"- Event cache: `{report['event_cache']}`",
        f"- Apply source: `{report['apply_source']}`",
        f"- Safe to apply: **{report['safe_to_apply']}**",
        f"- Summary: {report['safety_summary']}",
        "",
        "## Event",
        "",
        f"- {report['event']['event_date']} - {report['event']['event_name']} ({report['event']['event_id']})",
        f"- Parsed fights: {report['event']['parsed_fights']} | parsed fighters: {report['event']['parsed_fighters']}",
        f"- Local event exists by event/date: {report['event']['exists_locally_by_event_date']}",
        "",
        "## Planned Rows",
        "",
        f"- fighters: {report['planned_row_counts']['fighters']}",
        f"- fighter_source_ids: {report['planned_row_counts']['fighter_source_ids']}",
        f"- fights: {report['planned_row_counts']['fights']}",
        f"- fight_stats: {report['planned_row_counts']['fight_stats']}",
        f"- fight_elo_snapshots: {report['planned_row_counts']['fight_elo_snapshots']}",
        f"- skipped fights: {report['planned_row_counts']['skipped_fights']}",
        f"- blocked fighters: {report['planned_row_counts']['blocked_fighters']}",
        f"- blocked fights: {report['planned_row_counts']['blocked_fights']}",
        "",
        f"Target tables: `{', '.join(report['target_tables'])}`",
        "",
    ]
    if report["required_fields_missing"]:
        lines += ["## Required Fields Missing", ""]
        for item in report["required_fields_missing"]:
            lines.append(f"- {item}")
        lines.append("")
    if report["safety_blocks"]:
        lines += ["## Safety Blocks", ""]
        for item in report["safety_blocks"]:
            lines.append(f"- {item}")
        lines.append("")
    if report["rows_blocked"]["fighters"] or report["rows_blocked"]["fights"]:
        lines += ["## Rows Blocked", ""]
        for item in report["rows_blocked"]["fighters"]:
            lines.append(f"- fighter: {item}")
        for item in report["rows_blocked"]["fights"]:
            lines.append(f"- fight: {item}")
        lines.append("")
    lines += ["## Fighters To Insert", ""]
    if not report["fighters_to_insert"]:
        lines.append("- none")
    for item in report["fighters_to_insert"]:
        lines.append(f"- {item['name']} ({item['source_fighter_url']})")
    lines += ["", "## Fights To Insert", ""]
    if not report["fights_to_insert"]:
        lines.append("- none")
    for item in report["fights_to_insert"]:
        lines.append(
            f"- {item['date']} - {item['fighter_a']} vs {item['fighter_b']} "
            f"({item['weight_class']}, {item['method']}, R{item['round']} {item['time']})"
        )
    lines += ["", "## Optional Fields Not Available From Event Page", ""]
    for field in report["optional_fields_not_available_from_event_page"]:
        lines.append(f"- {field}")
    if report.get("apply_result"):
        ar = report["apply_result"]
        lines += ["", "## Apply Result", ""]
        lines.append(f"- apply performed: {ar.get('apply_performed')}")
        if ar.get("backup_path"):
            lines.append(f"- backup: `{ar['backup_path']}`")
        if ar.get("db_count_deltas"):
            lines.append(f"- count deltas: {ar['db_count_deltas']}")
        if ar.get("verification"):
            lines.append(f"- verification: {ar['verification']}")
        if ar.get("step6c_verification"):
            lines.append(f"- Step 6C verification: {ar['step6c_verification']}")
    if report.get("no_write_statement"):
        lines += ["", f"_{report['no_write_statement']}_"]
    lines.append("")
    return "\n".join(lines)
