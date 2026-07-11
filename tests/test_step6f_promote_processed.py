"""Tests for Step 6F guarded promotion of rebuilt processed files.

Covers: promotion requires an explicit source dir; the backup dir is created
(capturing pre-overwrite content) before any official file is overwritten;
official files are NOT overwritten when validation fails; the manifest schema;
DB-unchanged and benchmark-unchanged checks; and that rollback instructions are
emitted. A tiny synthetic SQLite DB feeds the real Step 3/3B/3C builders; the
official/DB/benchmark paths are monkeypatched to a temp sandbox so no real
artifact is touched.
"""

import json
import shutil
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufc_pipeline.db import init_schema  # noqa: E402
from ufc_pipeline.elo import build_elo_for_db  # noqa: E402
from ufc_pipeline.features import build_features_for_db  # noqa: E402
from ufc_pipeline.stats_features import build_step3b_for_db  # noqa: E402
from ufc_pipeline.matchup_features import build_step3c_for_db  # noqa: E402
from ufc_pipeline import step6f_promote_processed as s6f  # noqa: E402
from ufc_pipeline.step6f_promote_processed import (  # noqa: E402
    resolve_promotion_files,
    restore_backup,
    run_promotion,
)

FIGHTERS = [
    (1, "Jon Jones", "185", "215", "1987-07-19"),
    (2, "Stipe Miocic", "193", "203", "1982-08-19"),
    (3, "Alex Pereira", "193", "200", "1987-07-07"),
    (4, "Israel Adesanya", "193", "203", "1989-07-22"),
    (5, "Charles Oliveira", "178", "188", "1989-10-17"),
    (6, "Islam Makhachev", "178", "180", "1991-10-27"),
]
FIGHTS = [
    (1, "2020-01-01", "OLD 1", 1, 2, 1),
    (2, "2020-06-01", "OLD 2", 3, 4, 1),
    (3, "2021-01-01", "OLD 3", 1, 3, 1),
    (4, "2021-06-01", "OLD 4", 2, 4, 1),
    (5, "2022-01-01", "OLD 5", 5, 6, 1),
    (6, "2022-06-01", "OLD 6", 1, 4, 0),
    (7, "2023-06-01", "NEW CARD", 2, 3, 1),
    (8, "2023-06-01", "NEW CARD", 5, 4, 0),
]
NEW_EVENT = ("2023-06-01", "NEW CARD", 2)
FUTURE_EVENT = ("2099-12-31", "FUTURE 999")
BASENAMES = {
    "step3": "ufc_prefight_features.csv",
    "step3b": "ufc_prefight_features_step3b.csv",
    "step3c": "ufc_prefight_features_step3c.csv",
}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Build full Step 3/3B/3C into a 'rebuilt' dir; derive an 'old official'
    (full minus the NEW CARD rows) for each file."""
    root = tmp_path_factory.mktemp("s6f_built")
    db = root / "hist.db"
    conn = sqlite3.connect(db)
    init_schema(conn)
    conn.executemany(
        "INSERT INTO fighters (fighter_id, name, normalized_name, height, reach, stance, date_of_birth) "
        "VALUES (?,?,?,?,?,?,?)",
        [(fid, n, n.lower(), h, r, "Orthodox", dob) for fid, n, h, r, dob in FIGHTERS],
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

    rebuilt = root / "rebuilt"
    rebuilt.mkdir()
    build_features_for_db(str(db), str(rebuilt / BASENAMES["step3"]))
    build_step3b_for_db(str(db), str(rebuilt / BASENAMES["step3b"]))
    build_step3c_for_db(str(db), str(rebuilt / BASENAMES["step3c"]))

    old = root / "old_official"
    old.mkdir()
    for name in BASENAMES.values():
        df = pd.read_csv(rebuilt / name)
        df[df["date"] != NEW_EVENT[0]].to_csv(old / name, index=False)
    return {"rebuilt": rebuilt, "old": old}


@pytest.fixture()
def sandbox(built, tmp_path, monkeypatch):
    """A fresh official dir (copy of old_official) + fake DB/benchmark, all wired
    into the module via monkeypatch so no real artifact is touched."""
    official_dir = tmp_path / "processed"
    official_dir.mkdir()
    for name in BASENAMES.values():
        shutil.copy2(built["old"] / name, official_dir / name)

    official_paths = {k: str(official_dir / v) for k, v in BASENAMES.items()}
    db = tmp_path / "ufc.db"
    db.write_bytes(b"fake-db-bytes")
    bench = tmp_path / "official_baseline.json"
    bench.write_text('{"model": "fake"}')

    monkeypatch.setattr(s6f, "OFFICIAL_PROCESSED_PATHS", official_paths)
    monkeypatch.setattr(s6f, "DB_PATH", str(db))
    monkeypatch.setattr(s6f, "BENCHMARK_PATH", str(bench))
    return {
        "official_dir": official_dir, "official_paths": official_paths,
        "db": db, "bench": bench, "backup_root": str(tmp_path / "backups"),
        "reports": str(tmp_path / "reports"),
    }


def _run(sandbox, built, source_dir, apply):
    return run_promotion(
        source_dir=str(source_dir),
        apply=apply,
        reports_dir=sandbox["reports"],
        official_step3c=sandbox["official_paths"]["step3c"],
        backup_root=sandbox["backup_root"],
        expected_events=[NEW_EVENT],
        excluded_events=[FUTURE_EVENT],
    )


# --------------------------------------------------------------------------- #
# Explicit source requirement
# --------------------------------------------------------------------------- #

def test_promotion_requires_explicit_source():
    with pytest.raises(ValueError, match="explicit"):
        run_promotion(source_dir="", apply=False, write_report=False)


def test_resolve_requires_existing_source(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        resolve_promotion_files(tmp_path / "nope")


def test_resolve_requires_all_files(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / BASENAMES["step3c"]).write_text("x")  # only 1 of 3
    with pytest.raises(ValueError, match="missing"):
        resolve_promotion_files(tmp_path / "src")


# --------------------------------------------------------------------------- #
# Happy-path apply: backup-before-overwrite, exact copy, manifest, unchanged checks
# --------------------------------------------------------------------------- #

def test_apply_promotes_and_backs_up_before_overwrite(sandbox, built):
    old_md5 = s6f.file_md5(sandbox["official_paths"]["step3c"])
    source_md5 = s6f.file_md5(built["rebuilt"] / BASENAMES["step3c"])
    assert old_md5 != source_md5  # sanity: promotion will actually change the file

    report = _run(sandbox, built, built["rebuilt"], apply=True)

    assert report["status"] == "promoted"
    assert report["source_validation"]["ok"] is True
    assert report["post_promotion_verification"]["ok"] is True
    assert report["official_processed_files_overwritten"] is True

    # official is now an exact copy of the rebuilt source
    assert s6f.file_md5(sandbox["official_paths"]["step3c"]) == source_md5

    # backup captured the PRE-overwrite (old) content -> backup happened first
    backup_dir = Path(report["backup"]["backup_dir"])
    assert backup_dir.exists()
    assert s6f.file_md5(backup_dir / BASENAMES["step3c"]) == old_md5

    # unchanged checks
    assert report["post_promotion_verification"]["checks"]["db_unchanged"] is True
    assert report["post_promotion_verification"]["checks"]["benchmark_unchanged"] is True
    assert report["post_promotion_verification"]["official_step3c_row_count"] == \
        report["post_promotion_verification"]["expected_row_count"]


def test_dry_run_touches_nothing(sandbox, built):
    before = s6f.file_md5(sandbox["official_paths"]["step3c"])
    report = _run(sandbox, built, built["rebuilt"], apply=False)
    assert report["status"] == "dry_run"
    assert report["official_processed_files_overwritten"] is False
    assert s6f.file_md5(sandbox["official_paths"]["step3c"]) == before
    assert not Path(sandbox["backup_root"]).exists()


# --------------------------------------------------------------------------- #
# Validation failure guard
# --------------------------------------------------------------------------- #

def test_official_not_overwritten_when_validation_fails(sandbox, built):
    # Point promotion at the OLD dir (no NEW CARD) -> row delta 0, events absent.
    before = s6f.file_md5(sandbox["official_paths"]["step3c"])
    report = _run(sandbox, built, built["old"], apply=True)

    assert report["status"] == "aborted_validation_failed"
    assert report["source_validation"]["ok"] is False
    assert report["official_processed_files_overwritten"] is False
    # official file untouched, no backup created
    assert s6f.file_md5(sandbox["official_paths"]["step3c"]) == before
    assert not Path(sandbox["backup_root"]).exists()


# --------------------------------------------------------------------------- #
# Manifest schema + rollback + restore
# --------------------------------------------------------------------------- #

def test_manifest_schema_and_rollback(sandbox, built):
    report = _run(sandbox, built, built["rebuilt"], apply=True)
    for key in (
        "step", "generated_at", "mode", "source_rebuild_dir", "destination_official_paths",
        "promotion_files", "pre_promotion_inspection", "source_validation",
        "post_promotion_verification", "backup", "rollback_instructions",
        "statements", "status", "report_json", "report_md",
    ):
        assert key in report, f"missing manifest key: {key}"

    loaded = json.loads(Path(report["report_json"]).read_text())
    assert loaded["step"] == "6F"
    md = Path(report["report_md"]).read_text()
    assert "Rollback instructions" in md and "Step 6F" in md

    # rollback instruction exists and references the backup dir
    assert report["rollback_instructions"]
    assert any("cp " in line for line in report["rollback_instructions"])
    assert any(report["backup"]["backup_dir"] in line for line in report["rollback_instructions"])


def test_restore_backup_reverts_official(sandbox, built):
    old_md5 = s6f.file_md5(sandbox["official_paths"]["step3c"])
    report = _run(sandbox, built, built["rebuilt"], apply=True)
    # after promotion the official differs from old
    assert s6f.file_md5(sandbox["official_paths"]["step3c"]) != old_md5
    # rolling back restores the old content exactly
    restore_backup(report["backup"])
    assert s6f.file_md5(sandbox["official_paths"]["step3c"]) == old_md5
