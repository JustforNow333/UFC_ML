#!/usr/bin/env python3
"""Build a deterministic timestamp-safe sportsbook consensus for one canonical bout."""

import argparse
import json

import _path  # noqa: F401
from ufc_pipeline.sportsbook.consensus import ConsensusBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/sportsbook/sportsbook_odds.db")
    parser.add_argument("--canonical-bout-id", required=True)
    parser.add_argument("--canonical-fighter-id", required=True, type=int)
    parser.add_argument("--canonical-opponent-id", required=True, type=int)
    parser.add_argument("--prediction-as-of-utc", required=True)
    parser.add_argument("--max-snapshot-age-hours", type=float)
    parser.add_argument("--include-pinnacle-in-broad-consensus", action="store_true")
    parser.add_argument("--persist-selection", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build without persisting selected-snapshot references.")
    args = parser.parse_args()
    if args.dry_run and args.persist_selection:
        parser.error("--dry-run and --persist-selection cannot be combined")
    result = ConsensusBuilder(args.db).build(
        canonical_bout_id=args.canonical_bout_id,
        canonical_fighter_id=args.canonical_fighter_id,
        canonical_opponent_id=args.canonical_opponent_id,
        prediction_as_of_utc=args.prediction_as_of_utc,
        max_snapshot_age_hours=args.max_snapshot_age_hours,
        include_pinnacle_in_broad_consensus=args.include_pinnacle_in_broad_consensus,
        persist_selection=args.persist_selection,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
