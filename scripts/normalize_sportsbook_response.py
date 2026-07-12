#!/usr/bin/env python3
"""Archive, normalize, conservatively match, and store a local provider response."""

import argparse
import json
from dataclasses import asdict

import _path  # noqa: F401
from ufc_pipeline.sportsbook.config import SportsbookConfig
from ufc_pipeline.sportsbook.service import SportsbookIngestionService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=("the_odds_api",))
    parser.add_argument("--input", required=True)
    parser.add_argument("--request-type", choices=("current", "historical"), required=True)
    parser.add_argument("--query-timestamp-utc")
    parser.add_argument("--db", default="data/sportsbook/sportsbook_odds.db")
    parser.add_argument("--raw-dir", default="data/sportsbook/raw")
    parser.add_argument("--ufc-db", default="data/ufc.db")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = SportsbookConfig(database_path=args.db, raw_archive_root=args.raw_dir, canonical_ufc_db=args.ufc_db)
    service = SportsbookIngestionService(config)
    result = service.ingest_local_payload(
        provider_name=args.provider,
        payload=service.load_json(args.input),
        request_type=args.request_type,
        source_file=args.input,
        query_timestamp_utc=args.query_timestamp_utc,
        dry_run=args.dry_run,
    )
    output = asdict(result)
    output["archive"]["archive_path"] = str(result.archive.archive_path)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
