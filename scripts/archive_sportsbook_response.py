#!/usr/bin/env python3
"""Archive a local provider JSON response immutably and idempotently."""

import argparse
import json
from dataclasses import asdict

import _path  # noqa: F401
from ufc_pipeline.sportsbook.config import SportsbookConfig
from ufc_pipeline.sportsbook.service import SportsbookIngestionService
from ufc_pipeline.sportsbook.storage import SportsbookStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=("the_odds_api", "sportsdataio"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--request-type", choices=("current", "historical"), required=True)
    parser.add_argument("--query-timestamp-utc")
    parser.add_argument("--db", default="data/sportsbook/sportsbook_odds.db")
    parser.add_argument("--raw-dir", default="data/sportsbook/raw")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = SportsbookConfig(database_path=args.db, raw_archive_root=args.raw_dir)
    payload = SportsbookIngestionService.load_json(args.input)
    store = SportsbookStore(config)
    archive = store.archive_payload(args.provider, payload, dry_run=args.dry_run)
    batch_id = None
    if not args.dry_run:
        batch_id = store.create_ingestion_batch(
            provider_name=args.provider, request_type=args.request_type,
            archive=archive, query_timestamp_utc=args.query_timestamp_utc,
            source_file=args.input,
        )
    print(json.dumps({"archive": {**asdict(archive), "archive_path": str(archive.archive_path)}, "ingestion_batch_id": batch_id}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
