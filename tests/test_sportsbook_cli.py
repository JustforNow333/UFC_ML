import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ufc_pipeline.sportsbook.config import SportsbookConfig
from ufc_pipeline.sportsbook.service import SportsbookIngestionService


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sportsbook" / "the_odds_api_historical.json"
UFC_DB = ROOT / "data" / "ufc.db"


def _run(script, *args, check=True):
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if check and completed.returncode != 0:
        raise AssertionError(f"{script} failed\nstdout={completed.stdout}\nstderr={completed.stderr}")
    return completed


def test_storage_init_dry_run_and_apply_then_inspect(tmp_path):
    db = tmp_path / "odds.db"
    preview = json.loads(_run("init_sportsbook_storage.py", "--db", db, "--raw-dir", tmp_path / "raw", "--dry-run").stdout)
    assert preview["dry_run"]
    assert preview["migration_statement_count"] > 0
    assert not db.exists()
    applied = json.loads(_run("init_sportsbook_storage.py", "--db", db, "--raw-dir", tmp_path / "raw").stdout)
    assert applied["health"]["schema_version"] == 1
    inspected = json.loads(_run("inspect_sportsbook_database.py", "--db", db).stdout)
    assert inspected["tables"]["market_snapshots"] == 0


def test_archive_cli_is_idempotent(tmp_path):
    args = ("--provider", "the_odds_api", "--input", FIXTURE, "--request-type", "historical", "--db", tmp_path / "odds.db", "--raw-dir", tmp_path / "raw")
    first = json.loads(_run("archive_sportsbook_response.py", *args).stdout)
    second = json.loads(_run("archive_sportsbook_response.py", *args).stdout)
    assert first["archive"]["payload_hash"] == second["archive"]["payload_hash"]
    assert first["archive"]["created"] is True
    assert second["archive"]["created"] is False
    assert first["ingestion_batch_id"] == second["ingestion_batch_id"]


def test_normalize_match_consensus_and_health_clis(tmp_path):
    db = tmp_path / "odds.db"
    normalized = json.loads(_run(
        "normalize_sportsbook_response.py",
        "--provider", "the_odds_api", "--input", FIXTURE,
        "--request-type", "historical", "--query-timestamp-utc", "2024-04-12T02:00:00Z",
        "--db", db, "--raw-dir", tmp_path / "raw", "--ufc-db", UFC_DB,
    ).stdout)
    assert normalized["matched_snapshots"] == 4
    rematched = json.loads(_run("match_sportsbook_records.py", "--db", db, "--ufc-db", UFC_DB, "--dry-run").stdout)
    assert rematched["matched"] == 4
    consensus = json.loads(_run(
        "build_sportsbook_consensus.py", "--db", db,
        "--canonical-bout-id", "1073", "--canonical-fighter-id", "250",
        "--canonical-opponent-id", "362", "--prediction-as-of-utc", "2024-04-12T02:00:00Z",
        "--dry-run",
    ).stdout)
    assert consensus["sportsbook_count"] == 2
    assert consensus["pinnacle_available"] is True
    assert "polymarket" not in consensus["included_sportsbook_keys"]


def test_mapping_cli_dry_run_validates_without_writing(tmp_path):
    db = tmp_path / "odds.db"
    output = json.loads(_run(
        "import_sportsbook_mapping.py", "--provider", "the_odds_api",
        "--entity-type", "fighter", "--raw-identifier", "The Highlight",
        "--canonical-identifier", "362", "--created-by", "pytest",
        "--db", db, "--ufc-db", UFC_DB, "--dry-run",
    ).stdout)
    assert output["canonical_name"] == "Justin Gaethje"
    assert output["dry_run"] is True
    assert not db.exists()


def test_partial_provider_failure_stores_valid_records_and_preserves_archive(tmp_path):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["data"][0]["bookmakers"].append({"key": "draftkings", "title": "Broken", "markets": "not-a-list"})
    config = SportsbookConfig(
        database_path=tmp_path / "odds.db", raw_archive_root=tmp_path / "raw",
        canonical_ufc_db=UFC_DB, reports_dir=tmp_path / "reports",
    )
    result = SportsbookIngestionService(config).ingest_local_payload(
        provider_name="the_odds_api", payload=payload, request_type="historical",
        query_timestamp_utc="2024-04-12T02:00:00Z",
    )
    assert result.normalization_status == "normalized_with_errors"
    assert result.snapshots_inserted == 5
    assert result.matched_snapshots == 4
    archive_bytes = result.archive.archive_path.read_bytes()
    assert hashlib.sha256(archive_bytes).hexdigest() == result.archive.payload_hash


def test_normalization_failure_archives_payload_and_marks_batch_failed(tmp_path):
    config = SportsbookConfig(
        database_path=tmp_path / "odds.db", raw_archive_root=tmp_path / "raw",
        canonical_ufc_db=UFC_DB,
    )
    service = SportsbookIngestionService(config)
    with pytest.raises(Exception):
        service.ingest_local_payload(
            provider_name="the_odds_api", payload={"bad": "shape"}, request_type="historical",
        )
    connection = service.store.connect(read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM raw_archives").fetchone()[0] == 1
        row = connection.execute("SELECT response_status, error_message FROM provider_ingestion_batches").fetchone()
    finally:
        connection.close()
    assert row[0] == "failed"
    assert "historical response" in row[1]
