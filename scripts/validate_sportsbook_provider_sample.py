#!/usr/bin/env python3
"""Validate a local provider sample for Stage 2 schema/timestamp compatibility."""

import argparse
import json

import _path  # noqa: F401
from ufc_pipeline.sportsbook.config import SportsbookConfig
from ufc_pipeline.sportsbook.sample_validation import SampleValidator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=("the_odds_api",))
    parser.add_argument("--input", required=True)
    parser.add_argument("--request-type", default="historical", choices=("historical", "current"))
    parser.add_argument("--sample-provenance", default="user_supplied_unverified", choices=("user_supplied_unverified", "real_provider_sample", "synthetic_fixture", "saved_documentation_example"))
    parser.add_argument("--raw-retention-permitted", action="store_true", help="Confirm the user's provider terms permit local raw-payload retention.")
    parser.add_argument("--db", default="data/sportsbook/sportsbook_odds.db")
    parser.add_argument("--raw-dir", default="data/sportsbook/raw")
    parser.add_argument("--ufc-db", default="data/ufc.db")
    parser.add_argument("--output-dir", default="reports/sportsbook")
    parser.add_argument("--dry-run", action="store_true", help="Validate and write reports without archiving or changing the sidecar database.")
    args = parser.parse_args()
    config = SportsbookConfig(database_path=args.db, raw_archive_root=args.raw_dir, canonical_ufc_db=args.ufc_db, reports_dir=args.output_dir)
    result = SampleValidator(config).validate(
        provider_name=args.provider, input_path=args.input, request_type=args.request_type,
        sample_provenance=args.sample_provenance,
        raw_retention_permitted=args.raw_retention_permitted, dry_run=args.dry_run,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
