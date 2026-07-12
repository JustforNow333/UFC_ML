#!/usr/bin/env python3
"""Import one explicit reviewed provider fighter or bout mapping."""

import argparse
import json

import _path  # noqa: F401
from ufc_pipeline.sportsbook.config import SportsbookConfig
from ufc_pipeline.sportsbook.matching import import_reviewed_mapping
from ufc_pipeline.sportsbook.storage import SportsbookStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--entity-type", required=True, choices=("fighter", "bout"))
    parser.add_argument("--raw-identifier", required=True)
    parser.add_argument("--canonical-identifier", required=True)
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--review-note")
    parser.add_argument("--db", default="data/sportsbook/sportsbook_odds.db")
    parser.add_argument("--ufc-db", default="data/ufc.db")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    SportsbookStore(SportsbookConfig(database_path=args.db)).initialize(dry_run=args.dry_run)
    result = import_reviewed_mapping(
        args.db, args.ufc_db, provider_name=args.provider,
        entity_type=args.entity_type, raw_identifier=args.raw_identifier,
        canonical_identifier=args.canonical_identifier, created_by=args.created_by,
        review_note=args.review_note, dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
