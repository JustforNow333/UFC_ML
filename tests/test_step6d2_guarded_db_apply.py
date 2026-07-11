import json
import sqlite3
from pathlib import Path

import pytest

from ufc_pipeline.db import init_schema
from ufc_pipeline.columns import name_key
from ufc_pipeline.step6c_upcoming_feature_builder import normalize_fighter_name
from ufc_pipeline.step6d2_guarded_db_apply import (
    APPLY_SOURCE,
    build_apply_plan,
    parse_cached_event_detail,
    run_cached_event_update,
)


def _fighter_id(name: str) -> str:
    return {
        "Alpha Fighter": "aaaaaaaaaaaaaaaa",
        "Beta Fighter": "bbbbbbbbbbbbbbbb",
        "New Prospect": "cccccccccccccccc",
        "Second New": "dddddddddddddddd",
        "Sean O'Malley": "eeeeeeeeeeeeeeee",
    }[name]


def _fighter_url(name: str) -> str:
    return f"http://ufcstats.com/fighter-details/{_fighter_id(name)}"


def _fight_row(
    fight_id: str,
    a: str,
    b: str,
    kd: str = "1|0",
    strikes: str = "25|10",
    td: str = "2|0",
    sub: str = "0|1",
    weight_class: str = "Lightweight",
    method: str = "U-DEC",
    round_value: str = "3",
    time_value: str = "5:00",
) -> str:
    def cell(text: str) -> str:
        return "".join(f"<p class='b-fight-details__table-text'>{part}</p>" for part in text.split("|"))

    return f"""
    <tr class="b-fight-details__table-row js-fight-details-click"
        data-link="http://ufcstats.com/fight-details/{fight_id}">
      <td><i class="b-flag__text">win</i></td>
      <td>
        <a href="{_fighter_url(a)}">{a}</a>
        <a href="{_fighter_url(b)}">{b}</a>
      </td>
      <td>{cell(kd)}</td>
      <td>{cell(strikes)}</td>
      <td>{cell(td)}</td>
      <td>{cell(sub)}</td>
      <td>{cell(weight_class)}</td>
      <td>{cell(method)}</td>
      <td>{cell(round_value)}</td>
      <td>{cell(time_value)}</td>
    </tr>
    """


def _event_html(rows: str, date_text: str = "June 27, 2026") -> str:
    return f"""
    <html>
      <body>
        <h2 class="b-content__title-highlight">UFC Test Event</h2>
        <ul class="b-list__box-list">
          <li>Date: {date_text}</li>
          <li>Location: Test City</li>
        </ul>
        <table class="b-fight-details__table b-fight-details__table_type_event-details">
          <tbody>{rows}</tbody>
        </table>
      </body>
    </html>
    """


def _write_event(tmp_path: Path, rows: str, name: str = "event_1111222233334444.html") -> Path:
    path = tmp_path / name
    path.write_text(_event_html(rows), encoding="utf-8")
    return path


def _make_db(tmp_path: Path, fighters: list[tuple[int, str]], source_links: bool = True) -> Path:
    db = tmp_path / "test.db"
    conn = sqlite3.connect(db)
    init_schema(conn)
    for fid, name in fighters:
        conn.execute(
            "INSERT INTO fighters (fighter_id, name, normalized_name, height, reach) VALUES (?, ?, ?, ?, ?)",
            (fid, name, name_key(name), "180", "185"),
        )
        if source_links and name in ("Alpha Fighter", "Beta Fighter"):
            conn.execute(
                """INSERT INTO fighter_source_ids
                   (fighter_id, source, source_fighter_id, source_fighter_url,
                    raw_name, normalized_name, resolution)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    fid,
                    "greco",
                    _fighter_id(name),
                    _fighter_url(name),
                    name,
                    normalize_fighter_name(name),
                    "linked",
                ),
            )
    conn.commit()
    conn.close()
    return db


def _counts(db: Path) -> dict[str, int]:
    conn = sqlite3.connect(db)
    try:
        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("fighters", "fighter_source_ids", "fights", "fight_stats", "fight_elo_snapshots")
        }
    finally:
        conn.close()


def test_parse_cached_event_detail_extracts_result_and_stats(tmp_path):
    event = _write_event(tmp_path, _fight_row("f100", "Alpha Fighter", "Beta Fighter", method="KO/TKO|Punches", round_value="2", time_value="0:15"))

    parsed = parse_cached_event_detail(event.read_text(), event_cache=event)

    assert parsed["event_title"] == "UFC Test Event"
    assert parsed["date"] == "2026-06-27"
    assert parsed["fights"][0]["winner_name"] == "Alpha Fighter"
    assert parsed["fights"][0]["method"] == "KO/TKO"
    assert parsed["fights"][0]["method_detail"] == "Punches"
    assert parsed["fights"][0]["minutes"] == 5.25
    assert parsed["fights"][0]["stats"]["fighter_a"]["sig_str_landed"] == 25
    assert parsed["fights"][0]["stats"]["fighter_b"]["td_landed"] == 0


def test_dry_run_does_not_write(tmp_path):
    db = _make_db(tmp_path, [(1, "Alpha Fighter"), (2, "Beta Fighter")])
    event = _write_event(tmp_path, _fight_row("f100", "Alpha Fighter", "Beta Fighter"))
    before = _counts(db)

    report = run_cached_event_update(str(db), str(event), output_dir=str(tmp_path / "reports"), apply=False)

    assert _counts(db) == before
    assert report["mode"] == "dry_run_plan"
    assert report["safe_to_apply"] is True
    assert report["planned_row_counts"]["fights"] == 1
    assert report["apply_result"] is None
    assert "data/ufc.db" not in report["db_path"]


def test_apply_creates_backup_and_inserts_only(tmp_path):
    db = _make_db(tmp_path, [(1, "Alpha Fighter"), (2, "Beta Fighter")])
    event = _write_event(
        tmp_path,
        _fight_row("f100", "Alpha Fighter", "Beta Fighter")
        + _fight_row("f200", "New Prospect", "Beta Fighter", method="SUB|Armbar", round_value="1", time_value="2:30"),
    )

    report = run_cached_event_update(
        str(db),
        str(event),
        output_dir=str(tmp_path / "reports"),
        apply=True,
        run_step6c_verification=False,
    )

    ar = report["apply_result"]
    assert ar["apply_performed"] is True
    assert Path(ar["backup_path"]).exists()
    assert ar["db_count_deltas"] == {
        "fight_elo_snapshots": 2,
        "fight_stats": 4,
        "fighter_source_ids": 1,
        "fighters": 1,
        "fights": 2,
    }
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT height FROM fighters WHERE fighter_id = 1").fetchone()[0] == "180"
        assert conn.execute("SELECT COUNT(*) FROM fights WHERE event = 'UFC Test Event'").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM fight_stats").fetchone()[0] == 4
        assert conn.execute("SELECT COUNT(*) FROM fight_elo_snapshots").fetchone()[0] == 2
        source = conn.execute(
            "SELECT source FROM fighter_source_ids WHERE source_fighter_id = ?",
            (_fighter_id("New Prospect"),),
        ).fetchone()[0]
        assert source == APPLY_SOURCE
    finally:
        conn.close()


def test_duplicate_apply_is_safely_skipped(tmp_path):
    db = _make_db(tmp_path, [(1, "Alpha Fighter"), (2, "Beta Fighter")])
    event = _write_event(tmp_path, _fight_row("f100", "Alpha Fighter", "Beta Fighter"))

    first = run_cached_event_update(str(db), str(event), output_dir=str(tmp_path / "reports"), apply=True, run_step6c_verification=False)
    second = run_cached_event_update(str(db), str(event), output_dir=str(tmp_path / "reports"), apply=True, run_step6c_verification=False)

    assert first["apply_result"]["db_count_deltas"]["fights"] == 1
    assert second["planned_row_counts"]["fights"] == 0
    assert second["planned_row_counts"]["skipped_fights"] == 1
    assert second["apply_result"]["db_count_deltas"] == {
        "fight_elo_snapshots": 0,
        "fight_stats": 0,
        "fighter_source_ids": 0,
        "fighters": 0,
        "fights": 0,
    }


def test_apostrophe_name_uses_production_key_for_exact_fallback(tmp_path):
    db = _make_db(tmp_path, [(1, "Sean O'Malley"), (2, "Beta Fighter")], source_links=False)
    event = _write_event(tmp_path, _fight_row("f100", "Sean O'Malley", "Beta Fighter"))

    report = run_cached_event_update(
        str(db), str(event), output_dir=str(tmp_path / "reports"),
        apply=True, run_step6c_verification=False,
    )

    assert report["apply_result"]["db_count_deltas"]["fighters"] == 0
    resolution = next(
        row for row in report["fighter_resolution"]
        if row["source_fighter_id"] == _fighter_id("Sean O'Malley")
    )
    assert resolution["status"] == "existing_by_exact_name"
    assert resolution["fighter_id"] == 1
    conn = sqlite3.connect(db)
    try:
        stored_key = conn.execute(
            "SELECT normalized_name FROM fighter_source_ids WHERE source_fighter_id = ?",
            (_fighter_id("Sean O'Malley"),),
        ).fetchone()[0]
        assert stored_key == name_key("Sean O'Malley")
    finally:
        conn.close()


def test_ambiguous_fighter_blocks_apply_without_backup(tmp_path):
    db = _make_db(
        tmp_path,
        [(1, "Alpha Fighter"), (2, "Alpha Fighter"), (3, "Beta Fighter")],
        source_links=False,
    )
    event = _write_event(tmp_path, _fight_row("f100", "Alpha Fighter", "Beta Fighter"))
    before = _counts(db)

    report = run_cached_event_update(str(db), str(event), output_dir=str(tmp_path / "reports"), apply=True)

    assert report["safe_to_apply"] is False
    assert report["mode"] == "apply_blocked"
    assert report["apply_result"]["apply_performed"] is False
    assert report["planned_row_counts"]["blocked_fighters"] == 1
    assert _counts(db) == before
    assert not list(tmp_path.glob("test.db.step6d2_backup_*"))


def test_required_fighter_fields_block_apply(tmp_path):
    db = _make_db(tmp_path, [(1, "Alpha Fighter"), (2, "Beta Fighter")])
    malformed_row = """
    <tr data-link="http://ufcstats.com/fight-details/f100">
      <td><i class="b-flag__text">win</i></td>
      <td><a href="http://ufcstats.com/fighter-details/aaaaaaaaaaaaaaaa">Alpha Fighter</a></td>
    </tr>
    """
    event = _write_event(tmp_path, malformed_row)

    report = run_cached_event_update(str(db), str(event), output_dir=str(tmp_path / "reports"), apply=True)

    assert report["safe_to_apply"] is False
    assert report["apply_result"]["apply_performed"] is False
    assert any(item["field"] == "fights" for item in report["required_fields_missing"])


def test_changelog_report_schema(tmp_path):
    db = _make_db(tmp_path, [(1, "Alpha Fighter"), (2, "Beta Fighter")])
    event = _write_event(tmp_path, _fight_row("f100", "Alpha Fighter", "Beta Fighter"))

    report = run_cached_event_update(str(db), str(event), output_dir=str(tmp_path / "reports"), apply=False)
    report_json = Path(report["report_json"])
    loaded = json.loads(report_json.read_text())

    assert report_json.exists()
    assert Path(report["report_md"]).exists()
    assert loaded["run"] == "step6d2_guarded_cached_event_apply"
    assert loaded["dry_run_default"] is True
    assert loaded["apply_requires_explicit_flag"] is True
    assert loaded["target_tables"] == ["fight_elo_snapshots", "fight_stats", "fights"]
    assert loaded["planned_row_counts"]["fights"] == 1


def test_mid_history_event_blocks_insert_only_elo_apply(tmp_path):
    db = _make_db(tmp_path, [(1, "Alpha Fighter"), (2, "Beta Fighter")])
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            """INSERT INTO fights
               (fight_id, date, event, fighter_a_id, fighter_b_id, winner_id,
                fighter_a_won, weight_class, method, round, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (1, "2026-07-01", "Later Event", 1, 2, 1, 1, "Lightweight", "Decision", "3", "test"),
        )
        conn.commit()
    finally:
        conn.close()
    event = _write_event(tmp_path, _fight_row("f100", "Alpha Fighter", "Beta Fighter"))

    plan = build_apply_plan(str(db), str(event), output_dir=str(tmp_path / "reports"))

    assert plan["safe_to_apply"] is False
    assert any(block["type"] == "non_append_history" for block in plan["safety_blocks"])
