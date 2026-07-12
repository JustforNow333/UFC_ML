#!/usr/bin/env python3
"""Print isolated sportsbook database counts and health without secrets."""

import argparse
import json

import _path  # noqa: F401
from ufc_pipeline.sportsbook.config import SportsbookConfig
from ufc_pipeline.sportsbook.storage import SportsbookStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/sportsbook/sportsbook_odds.db")
    args = parser.parse_args()
    print(json.dumps(SportsbookStore(SportsbookConfig(database_path=args.db)).health(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
