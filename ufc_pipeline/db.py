"""SQLite schema + small helpers.

Tables:
  fighters            - one row per canonical fighter. normalized_name is a
                        MATCHING KEY, not an identity: two different fighters
                        can share a name (e.g. the flyweight and middleweight
                        Bruno Silvas), so it is deliberately NOT unique. True
                        identity lives in fighter_source_ids.
  fighter_source_ids  - durable source-specific fighter identities (e.g.
                        UFCStats fighter-details URLs from Greco). One source
                        URL maps to exactly one canonical fighter; a canonical
                        fighter may have IDs from several sources.
  fights              - one row per bout, normalized, in source order
  fight_stats         - per-fight per-fighter offensive totals (Step 3B)
  fight_elo_snapshots - pre/post Elo per fight, written by the Elo builder
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS fighters (
    fighter_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    height          TEXT,
    reach           TEXT,
    stance          TEXT,
    date_of_birth   TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fighters_normalized ON fighters(normalized_name);

-- Source-specific fighter identity (Part A of the identity layer).
-- source_fighter_id is the stable slug/hash from the source URL (for
-- UFCStats: the hex id in /fighter-details/<id>). UNIQUE(source,
-- source_fighter_id) guarantees one URL -> at most one canonical fighter.
-- fighter_id is NULL until the identity is linked; listed weight / DOB /
-- height / reach are kept as disambiguation evidence and for debugging.
-- resolution: 'unlinked' (fighter not in DB yet), 'linked' (matched),
--             'ambiguous' (same-name group needing evidence), 'unresolved'
--             (evidence insufficient; reported, never guessed).
CREATE TABLE IF NOT EXISTS fighter_source_ids (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    fighter_id         INTEGER REFERENCES fighters(fighter_id),
    source             TEXT NOT NULL,
    source_fighter_id  TEXT NOT NULL,
    source_fighter_url TEXT,
    raw_name           TEXT,
    normalized_name    TEXT,
    listed_weight_lbs  REAL,
    date_of_birth      TEXT,
    height_cm          REAL,
    reach_cm           REAL,
    resolution         TEXT NOT NULL DEFAULT 'unlinked',
    resolution_note    TEXT,
    first_seen_date    TEXT,
    last_seen_date     TEXT,
    created_at         TEXT DEFAULT (datetime('now')),
    updated_at         TEXT DEFAULT (datetime('now')),
    UNIQUE (source, source_fighter_id)
);

CREATE INDEX IF NOT EXISTS idx_fsi_name ON fighter_source_ids(source, normalized_name);
CREATE INDEX IF NOT EXISTS idx_fsi_fighter ON fighter_source_ids(fighter_id);

CREATE TABLE IF NOT EXISTS fights (
    fight_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    date           TEXT NOT NULL,           -- ISO yyyy-mm-dd
    event          TEXT,
    fighter_a_id   INTEGER NOT NULL REFERENCES fighters(fighter_id),
    fighter_b_id   INTEGER NOT NULL REFERENCES fighters(fighter_id),
    winner_id      INTEGER REFERENCES fighters(fighter_id),
    fighter_a_won  INTEGER NOT NULL CHECK (fighter_a_won IN (0, 1)),
    weight_class   TEXT,
    method         TEXT,
    round          TEXT,
    source         TEXT,
    source_row_id  INTEGER,
    -- Per-fight "tale of the tape" values from the source, recorded AS OF
    -- the fight (age at fight time, listed height/reach). These are known
    -- before the bout, so they are time-safe pre-fight features.
    fighter_a_age    REAL,
    fighter_b_age    REAL,
    fighter_a_height REAL,
    fighter_b_height REAL,
    fighter_a_reach  REAL,
    fighter_b_reach  REAL
);

-- fight_id order encodes the original dataset order, which we rely on to
-- break ties between fights on the same date (see elo.py).
CREATE INDEX IF NOT EXISTS idx_fights_date ON fights(date, fight_id);

-- Per-fight, per-fighter OFFENSIVE totals from the stats source (Step 3B).
-- A fight's stats are only ever used to build features for LATER fights;
-- the rolling-stats builder enforces that, same as the Elo engine.
-- "Absorbed"/defense numbers come from joining the opponent's row.
CREATE TABLE IF NOT EXISTS fight_stats (
    fight_id          INTEGER NOT NULL REFERENCES fights(fight_id),
    fighter_id        INTEGER NOT NULL REFERENCES fighters(fighter_id),
    minutes           REAL NOT NULL,      -- total fight minutes (same both rows)
    knockdowns        REAL,
    sig_str_landed    REAL,
    sig_str_attempted REAL,
    td_landed         REAL,
    td_attempted      REAL,
    sub_attempts      REAL,
    ctrl_seconds      REAL,
    source            TEXT,               -- provenance (e.g. 'greco')
    PRIMARY KEY (fight_id, fighter_id)
);

CREATE TABLE IF NOT EXISTS fight_elo_snapshots (
    fight_id                     INTEGER PRIMARY KEY REFERENCES fights(fight_id),
    date                         TEXT NOT NULL,
    fighter_a_id                 INTEGER NOT NULL,
    fighter_b_id                 INTEGER NOT NULL,
    fighter_a_pre_elo            REAL NOT NULL,
    fighter_b_pre_elo            REAL NOT NULL,
    elo_diff                     REAL NOT NULL,
    fighter_a_expected_win_prob  REAL NOT NULL,
    fighter_b_expected_win_prob  REAL NOT NULL,
    fighter_a_post_elo           REAL NOT NULL,
    fighter_b_post_elo           REAL NOT NULL,
    fighter_a_won                INTEGER NOT NULL
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


_FIGHTS_MIGRATION_COLUMNS = [
    "fighter_a_age", "fighter_b_age",
    "fighter_a_height", "fighter_b_height",
    "fighter_a_reach", "fighter_b_reach",
]


def _normalized_name_is_unique(conn: sqlite3.Connection) -> bool:
    """True if fighters.normalized_name still carries the legacy UNIQUE
    constraint (databases created before the identity layer)."""
    for row in conn.execute("PRAGMA index_list(fighters)"):
        # row: (seq, name, unique, origin, partial); origin 'u' = constraint
        if row[2] and row[3] == "u":
            cols = [r[2] for r in conn.execute(f"PRAGMA index_info('{row[1]}')")]
            if cols == ["normalized_name"]:
                return True
    return False


def _drop_normalized_name_unique(conn: sqlite3.Connection) -> None:
    """Rebuild the fighters table without UNIQUE(normalized_name).

    SQLite cannot drop a constraint in place, so this is the standard
    copy/drop/rename dance. Foreign keys are switched off for the swap so
    fights/fight_stats references survive; fighter_id values are preserved
    exactly (existing mappings must never change during migration).
    """
    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.executescript(
        """
        CREATE TABLE fighters_migrated (
            fighter_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            height          TEXT,
            reach           TEXT,
            stance          TEXT,
            date_of_birth   TEXT,
            created_at      TEXT DEFAULT (datetime('now')),
            updated_at      TEXT DEFAULT (datetime('now'))
        );
        INSERT INTO fighters_migrated
            (fighter_id, name, normalized_name, height, reach, stance,
             date_of_birth, created_at, updated_at)
            SELECT fighter_id, name, normalized_name, height, reach, stance,
                   date_of_birth, created_at, updated_at FROM fighters;
        DROP TABLE fighters;
        ALTER TABLE fighters_migrated RENAME TO fighters;
        CREATE INDEX IF NOT EXISTS idx_fighters_normalized
            ON fighters(normalized_name);
        """
    )
    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON")


def init_schema(conn: sqlite3.Connection) -> None:
    # Migration first: executescript's CREATE TABLE IF NOT EXISTS would skip
    # an existing legacy fighters table, so drop its UNIQUE constraint here.
    fighters_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fighters'"
    ).fetchone()
    if fighters_exists and _normalized_name_is_unique(conn):
        _drop_normalized_name_unique(conn)

    conn.executescript(SCHEMA)
    # Lightweight migration: databases created before the physical columns
    # were added get them via ALTER TABLE (SQLite ignores nothing here, so
    # we check what exists first).
    existing = {row[1] for row in conn.execute("PRAGMA table_info(fights)")}
    for col in _FIGHTS_MIGRATION_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE fights ADD COLUMN {col} REAL")
    stats_cols = {row[1] for row in conn.execute("PRAGMA table_info(fight_stats)")}
    if stats_cols and "source" not in stats_cols:
        conn.execute("ALTER TABLE fight_stats ADD COLUMN source TEXT")
    conn.commit()
