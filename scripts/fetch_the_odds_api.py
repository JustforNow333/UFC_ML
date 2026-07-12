#!/usr/bin/env python3
"""Explicitly gated The Odds API fetch; no request occurs without safety flags."""

import argparse
import json

import _path  # noqa: F401
from ufc_pipeline.sportsbook.config import SportsbookConfig
from ufc_pipeline.sportsbook.providers.the_odds_api import TheOddsApiAdapter
from ufc_pipeline.sportsbook.storage import SportsbookStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-type", required=True, choices=("current", "historical", "sports"))
    parser.add_argument("--as-of-utc", help="Required for historical requests.")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--allow-billable-request", action="store_true")
    parser.add_argument("--db", default="data/sportsbook/sportsbook_odds.db")
    parser.add_argument("--raw-dir", default="data/sportsbook/raw")
    args = parser.parse_args()
    adapter = TheOddsApiAdapter()
    if args.request_type == "historical":
        if not args.as_of_utc:
            parser.error("--as-of-utc is required for historical requests")
        response = adapter.fetch_historical_odds(args.as_of_utc, allow_network=args.allow_network, allow_billable_request=args.allow_billable_request)
    elif args.request_type == "current":
        response = adapter.fetch_current_odds(allow_network=args.allow_network)
    else:
        response = adapter.list_supported_sports(allow_network=args.allow_network)
    store = SportsbookStore(SportsbookConfig(database_path=args.db, raw_archive_root=args.raw_dir))
    archive = store.archive_payload("the_odds_api", response.payload)
    batch_id = store.create_ingestion_batch(
        provider_name="the_odds_api", request_type=args.request_type,
        archive=archive, query_timestamp_utc=response.query_timestamp_utc,
        response_status="archived", quota_metadata=adapter.quota_metadata(response.headers),
    )
    print(json.dumps({
        "archive_path": str(archive.archive_path), "payload_hash": archive.payload_hash,
        "ingestion_batch_id": batch_id, "status_code": response.status_code,
        "quota": adapter.quota_metadata(response.headers),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
