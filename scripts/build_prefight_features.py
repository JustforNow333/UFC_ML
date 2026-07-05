#!/usr/bin/env python3
"""Build the time-safe pre-fight feature dataset.

Usage:
  python scripts/build_prefight_features.py --db data/ufc.db \
      --output data/processed/ufc_prefight_features.csv

Requires ingest_fights.py and build_elo.py to have been run first.
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.features import build_features_for_db


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/ufc.db", help="SQLite database path")
    p.add_argument("--output", default="data/processed/ufc_prefight_features.csv")
    args = p.parse_args()

    n = build_features_for_db(args.db, args.output)
    print(f"Wrote {n} pre-fight feature rows -> {args.output}")


if __name__ == "__main__":
    main()
