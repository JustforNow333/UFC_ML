#!/usr/bin/env python3
"""Convert Greco1899 fight results + events into a bout-level fights CSV
that can be ingested as the fights base (when no mdabbert CSV is available).

Usage:
  python scripts/convert_greco_fights.py \
      --results data/raw/greco/ufc_fight_results.csv \
      --events data/raw/greco/ufc_event_details.csv \
      --output data/raw/greco_fights.csv
Then:
  python scripts/ingest_fights.py --input data/raw/greco_fights.csv --db data/ufc.db --source greco
"""
import argparse

import pandas as pd

import _path  # noqa: F401
from ufc_pipeline.stats_ingest import convert_greco_fights


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--output", default="data/raw/greco_fights.csv")
    args = p.parse_args()

    fights = convert_greco_fights(pd.read_csv(args.results), pd.read_csv(args.events))
    fights.to_csv(args.output, index=False)
    decisive = (fights["winner"].fillna("") != "").sum()
    print(f"Wrote {len(fights)} bouts -> {args.output} "
          f"({decisive} decisive, {len(fights) - decisive} draw/NC)")


if __name__ == "__main__":
    main()
