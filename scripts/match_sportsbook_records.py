#!/usr/bin/env python3
"""Re-run conservative canonical matching for stored sportsbook snapshots."""

import argparse
import json

import _path  # noqa: F401
from ufc_pipeline.sportsbook.config import SportsbookConfig
from ufc_pipeline.sportsbook.service import SportsbookIngestionService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/sportsbook/sportsbook_odds.db")
    parser.add_argument("--ufc-db", default="data/ufc.db")
    parser.add_argument("--ingestion-batch-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = SportsbookConfig(database_path=args.db, canonical_ufc_db=args.ufc_db)
    result = SportsbookIngestionService(config).rematch_stored(
        ingestion_batch_id=args.ingestion_batch_id, dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
