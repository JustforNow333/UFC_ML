import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from ufc_pipeline.sportsbook.config import SportsbookConfig
from ufc_pipeline.sportsbook.domain import NormalizedOutcome, NormalizedSnapshot
from ufc_pipeline.sportsbook.storage import SportsbookStore, canonical_json_bytes


@pytest.fixture
def store(tmp_path):
    config = SportsbookConfig(
        database_path=tmp_path / "sportsbook_odds.db",
        raw_archive_root=tmp_path / "raw",
        canonical_ufc_db=tmp_path / "ufc.db",
        reports_dir=tmp_path / "reports",
    )
    result = SportsbookStore(config)
    result.initialize()
    return result


def _snapshot(payload_hash):
    outcomes = (
        NormalizedOutcome("Fighter One", "Fighter Two", 1.8, None, 1 / 1.8, 1.8, "decimal"),
        NormalizedOutcome("Fighter Two", "Fighter One", 2.1, None, 1 / 2.1, 2.1, "decimal"),
    )
    return NormalizedSnapshot(
        provider_name="the_odds_api",
        provider_event_id="event-1",
        provider_market_id="event-1:draftkings:h2h",
        sportsbook_key="draftkings",
        provider_sportsbook_key="draftkings",
        sportsbook_display_name="DraftKings",
        venue_type="sportsbook",
        market_type="h2h",
        observed_at_utc="2026-07-10T00:00:00Z",
        observed_at_original="2026-07-10T00:00:00Z",
        scheduled_event_time_utc="2026-07-12T00:00:00Z",
        scheduled_event_time_original="2026-07-12T00:00:00Z",
        pre_fight_or_live_status="pre_fight",
        raw_event_name="Fighter One vs Fighter Two",
        sport_key="mma_mixed_martial_arts",
        league_key="MMA",
        outcomes=outcomes,
        source_payload_hash=payload_hash,
    )


def test_schema_is_isolated_and_has_foreign_keys(store):
    health = store.health()
    assert health["schema_version"] == 1
    assert health["foreign_keys"]
    assert "market_snapshots" in health["tables"]
    assert health["tables"]["sportsbook_registry"] >= 11


def test_raw_payload_archive_is_canonical_hashed_and_idempotent(store):
    first = store.archive_payload("the_odds_api", {"z": 1, "a": [2]})
    second = store.archive_payload("the_odds_api", {"a": [2], "z": 1})
    expected = hashlib.sha256(canonical_json_bytes({"z": 1, "a": [2]})).hexdigest()
    assert first.payload_hash == expected
    assert first.created
    assert not second.created
    assert first.archive_path == second.archive_path
    assert json.loads(first.archive_path.read_text(encoding="utf-8")) == {"a": [2], "z": 1}
    assert store.health()["tables"]["raw_archives"] == 1


def test_different_payload_has_different_immutable_archive(store):
    first = store.archive_payload("the_odds_api", {"value": 1})
    second = store.archive_payload("the_odds_api", {"value": 2})
    assert first.payload_hash != second.payload_hash
    assert first.archive_path.read_bytes() != second.archive_path.read_bytes()
    assert store.health()["tables"]["raw_archives"] == 2


def test_archive_collision_never_overwrites_existing_content(store):
    archive = store.archive_payload("the_odds_api", {"value": 1})
    archive.archive_path.write_text("different", encoding="utf-8")
    with pytest.raises(RuntimeError, match="collision"):
        store.archive_payload("the_odds_api", {"value": 1})
    assert archive.archive_path.read_text(encoding="utf-8") == "different"


def test_dry_run_does_not_create_archive_or_database(tmp_path):
    store = SportsbookStore(SportsbookConfig(database_path=tmp_path / "odds.db", raw_archive_root=tmp_path / "raw"))
    result = store.archive_payload("the_odds_api", {"value": 1}, dry_run=True)
    assert result.dry_run
    assert not result.archive_path.exists()
    assert not store.config.database_path.exists()


def test_batch_and_normalized_inserts_are_idempotent(store):
    archive = store.archive_payload("the_odds_api", {"events": []})
    batch = store.create_ingestion_batch(provider_name="the_odds_api", request_type="historical", archive=archive, query_timestamp_utc="2026-07-10T00:00:00Z")
    same_batch = store.create_ingestion_batch(provider_name="the_odds_api", request_type="historical", archive=archive)
    assert batch == same_batch
    assert store.store_snapshots(batch, [_snapshot(archive.payload_hash)]) == {"snapshots_inserted": 1, "outcomes_inserted": 2}
    assert store.store_snapshots(batch, [_snapshot(archive.payload_hash)]) == {"snapshots_inserted": 0, "outcomes_inserted": 0}


def test_snapshot_transaction_rolls_back_on_failure(store):
    archive = store.archive_payload("the_odds_api", {"events": []})
    batch = store.create_ingestion_batch(provider_name="the_odds_api", request_type="current", archive=archive)
    valid = _snapshot(archive.payload_hash)
    invalid = replace(valid, provider_event_id="event-2", source_payload_hash="missing-hash")
    with pytest.raises(sqlite3.IntegrityError):
        store.store_snapshots(batch, [valid, invalid])
    health = store.health()
    assert health["tables"]["market_snapshots"] == 0
    assert health["tables"]["outcomes"] == 0


def test_failed_normalization_can_be_recorded_without_losing_raw_payload(store):
    archive = store.archive_payload("the_odds_api", {"bad": "shape"})
    store.create_ingestion_batch(
        provider_name="the_odds_api",
        request_type="historical",
        archive=archive,
        response_status="failed",
        error_message="missing data",
    )
    connection = store.connect(read_only=True)
    try:
        batch = connection.execute("SELECT response_status, error_message FROM provider_ingestion_batches").fetchone()
        raw = connection.execute("SELECT raw_payload_hash FROM raw_archives").fetchone()
    finally:
        connection.close()
    assert tuple(batch) == ("failed", "missing data")
    assert raw[0] == archive.payload_hash
