#!/usr/bin/env python3
"""Create or preview the isolated Stage 2 sportsbook SQLite schema."""

import argparse
import json

import _path  # noqa: F401
from ufc_pipeline.sportsbook.config import SportsbookConfig
from ufc_pipeline.sportsbook.storage import SportsbookStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/sportsbook/sportsbook_odds.db")
    parser.add_argument("--raw-dir", default="data/sportsbook/raw")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    store = SportsbookStore(SportsbookConfig(database_path=args.db, raw_archive_root=args.raw_dir))
    statements = store.initialize(dry_run=args.dry_run)
    result = {"database": args.db, "dry_run": args.dry_run, "migration_statement_count": len(statements)}
    if not args.dry_run:
        result["health"] = store.health()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
