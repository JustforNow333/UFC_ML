"""SQLite schema and migrations for the isolated sportsbook store."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict

from .registry import EXCLUDED_SOURCES, SPORTSBOOKS


SCHEMA_VERSION = 1


DDL_V1 = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_registry (
    provider_source_key TEXT PRIMARY KEY,
    canonical_sportsbook_key TEXT,
    venue_type TEXT NOT NULL,
    approved INTEGER NOT NULL CHECK (approved IN (0, 1)),
    notes TEXT
);

CREATE TABLE IF NOT EXISTS sportsbook_registry (
    canonical_sportsbook_key TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    venue_type TEXT NOT NULL,
    approved INTEGER NOT NULL CHECK (approved IN (0, 1)),
    consensus_eligible INTEGER NOT NULL CHECK (consensus_eligible IN (0, 1)),
    sharp_reference INTEGER NOT NULL CHECK (sharp_reference IN (0, 1)),
    secondary_candidate INTEGER NOT NULL CHECK (secondary_candidate IN (0, 1)),
    provider_confirmed INTEGER NOT NULL CHECK (provider_confirmed IN (0, 1)),
    parent_operator TEXT,
    active_from TEXT,
    active_to TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS raw_archives (
    raw_payload_hash TEXT PRIMARY KEY,
    provider_name TEXT NOT NULL,
    archive_path TEXT NOT NULL UNIQUE,
    byte_count INTEGER NOT NULL,
    canonical_json INTEGER NOT NULL CHECK (canonical_json IN (0, 1)),
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provider_ingestion_batches (
    ingestion_batch_id TEXT PRIMARY KEY,
    provider_name TEXT NOT NULL,
    request_type TEXT NOT NULL,
    query_timestamp_utc TEXT,
    fetched_at_utc TEXT NOT NULL,
    source_file TEXT,
    raw_payload_hash TEXT NOT NULL REFERENCES raw_archives(raw_payload_hash),
    archive_path TEXT NOT NULL,
    response_status TEXT NOT NULL,
    record_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    quota_metadata_json TEXT,
    schema_version INTEGER NOT NULL,
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS normalized_events (
    normalized_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ingestion_batch_id TEXT NOT NULL REFERENCES provider_ingestion_batches(ingestion_batch_id),
    provider_name TEXT NOT NULL,
    provider_event_id TEXT NOT NULL,
    raw_event_name TEXT NOT NULL,
    scheduled_event_time_utc TEXT NOT NULL,
    scheduled_event_time_original TEXT,
    sport_key TEXT NOT NULL,
    league_key TEXT,
    canonical_event_id TEXT,
    match_status TEXT NOT NULL DEFAULT 'unmatched',
    matching_method TEXT,
    matching_diagnostics_json TEXT,
    UNIQUE (ingestion_batch_id, provider_name, provider_event_id)
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    normalized_event_id INTEGER NOT NULL REFERENCES normalized_events(normalized_event_id),
    ingestion_batch_id TEXT NOT NULL REFERENCES provider_ingestion_batches(ingestion_batch_id),
    provider_name TEXT NOT NULL,
    provider_event_id TEXT NOT NULL,
    provider_market_id TEXT NOT NULL,
    sportsbook_key TEXT,
    provider_sportsbook_key TEXT NOT NULL,
    sportsbook_display_name TEXT NOT NULL,
    venue_type TEXT NOT NULL,
    market_type TEXT NOT NULL,
    observed_at_utc TEXT NOT NULL,
    observed_at_original TEXT,
    scheduled_event_time_utc TEXT NOT NULL,
    pre_fight_or_live_status TEXT NOT NULL,
    canonical_event_id TEXT,
    canonical_bout_id TEXT,
    normalization_status TEXT NOT NULL,
    match_status TEXT NOT NULL DEFAULT 'unmatched',
    matching_method TEXT,
    matching_diagnostics_json TEXT,
    rejection_reason TEXT,
    source_payload_hash TEXT NOT NULL REFERENCES raw_archives(raw_payload_hash),
    UNIQUE (provider_name, provider_event_id, provider_market_id, provider_sportsbook_key, observed_at_utc)
);

CREATE TABLE IF NOT EXISTS outcomes (
    outcome_id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL REFERENCES market_snapshots(snapshot_id) ON DELETE CASCADE,
    raw_outcome_name TEXT NOT NULL,
    raw_opponent_name TEXT,
    decimal_odds REAL,
    american_odds REAL,
    raw_implied_probability REAL,
    original_odds_json TEXT,
    original_odds_format TEXT NOT NULL,
    canonical_fighter_id INTEGER,
    canonical_opponent_id INTEGER,
    orientation_status TEXT NOT NULL,
    validation_status TEXT NOT NULL,
    validation_reason TEXT,
    UNIQUE (snapshot_id, raw_outcome_name)
);

CREATE TABLE IF NOT EXISTS match_reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ingestion_batch_id TEXT REFERENCES provider_ingestion_batches(ingestion_batch_id),
    entity_type TEXT NOT NULL,
    raw_identifier TEXT NOT NULL,
    candidate_identifiers_json TEXT,
    status TEXT NOT NULL,
    matching_method TEXT,
    score REAL,
    diagnostics_json TEXT,
    review_note TEXT,
    reviewed INTEGER NOT NULL DEFAULT 0 CHECK (reviewed IN (0, 1)),
    reviewed_at_utc TEXT,
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provider_mappings (
    mapping_id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    raw_identifier_normalized TEXT NOT NULL,
    canonical_identifier TEXT NOT NULL,
    mapping_method TEXT NOT NULL DEFAULT 'reviewed',
    created_by TEXT NOT NULL,
    review_note TEXT,
    created_at_utc TEXT NOT NULL,
    UNIQUE (provider_name, entity_type, raw_identifier_normalized)
);

CREATE TABLE IF NOT EXISTS selected_consensus_inputs (
    selection_id TEXT PRIMARY KEY,
    canonical_bout_id TEXT NOT NULL,
    canonical_fighter_id INTEGER NOT NULL,
    canonical_opponent_id INTEGER NOT NULL,
    prediction_as_of_utc TEXT NOT NULL,
    sportsbook_key TEXT NOT NULL,
    selected_snapshot_id TEXT NOT NULL REFERENCES market_snapshots(snapshot_id),
    selected_observed_at_utc TEXT NOT NULL,
    snapshot_age_hours REAL NOT NULL,
    fair_fighter_probability REAL NOT NULL,
    fair_opponent_probability REAL NOT NULL,
    overround REAL NOT NULL,
    devig_method TEXT NOT NULL,
    devig_version TEXT NOT NULL,
    selection_policy_version TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    UNIQUE (canonical_bout_id, canonical_fighter_id, prediction_as_of_utc, sportsbook_key, selection_policy_version)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_bout_time
    ON market_snapshots(canonical_bout_id, observed_at_utc);
CREATE INDEX IF NOT EXISTS idx_snapshots_book_time
    ON market_snapshots(sportsbook_key, observed_at_utc);
CREATE INDEX IF NOT EXISTS idx_outcomes_snapshot
    ON outcomes(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_reviews_status
    ON match_reviews(status, reviewed);
"""


def initialize_schema(connection: sqlite3.Connection, *, dry_run: bool = False) -> list[str]:
    statements = [statement.strip() for statement in DDL_V1.split(";\n") if statement.strip()]
    if dry_run:
        return statements
    connection.execute("PRAGMA foreign_keys = ON")
    with connection:
        connection.executescript(DDL_V1)
        connection.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at_utc) VALUES (?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
            (SCHEMA_VERSION,),
        )
        for book in SPORTSBOOKS:
            connection.execute(
                """INSERT OR REPLACE INTO sportsbook_registry(
                    canonical_sportsbook_key, display_name, venue_type, approved,
                    consensus_eligible, sharp_reference, secondary_candidate,
                    provider_confirmed, parent_operator, active_from, active_to, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    book.canonical_key, book.display_name, book.venue_type.value,
                    int(book.approved), int(book.consensus_eligible), int(book.sharp_reference),
                    int(book.secondary_candidate), int(book.provider_confirmed), book.parent_operator,
                    book.active_from, book.active_to, book.notes,
                ),
            )
            for provider_key in (book.canonical_key, *book.provider_specific_keys):
                connection.execute(
                    "INSERT OR REPLACE INTO source_registry(provider_source_key, canonical_sportsbook_key, venue_type, approved, notes) VALUES (?, ?, ?, ?, ?)",
                    (provider_key.casefold(), book.canonical_key, book.venue_type.value, int(book.approved), book.notes),
                )
        for provider_key, venue_type in EXCLUDED_SOURCES.items():
            connection.execute(
                "INSERT OR REPLACE INTO source_registry(provider_source_key, canonical_sportsbook_key, venue_type, approved, notes) VALUES (?, NULL, ?, 0, ?)",
                (provider_key, venue_type.value, "Explicitly prohibited from traditional-sportsbook consensus."),
            )
    return statements
