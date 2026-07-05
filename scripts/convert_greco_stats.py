#!/usr/bin/env python3
"""Convert Greco1899/scrape_ufc_stats files into the wide per-fight stats CSV.

Download from https://github.com/Greco1899/scrape_ufc_stats :
  ufc_fight_stats.csv, ufc_fight_results.csv, ufc_event_details.csv

Usage:
  python scripts/convert_greco_stats.py \
      --stats data/raw/ufc_fight_stats.csv \
      --results data/raw/ufc_fight_results.csv \
      --events data/raw/ufc_event_details.csv \
      --output data/raw/fight_stats_wide.csv
"""
import argparse

import pandas as pd

import _path  # noqa: F401
from ufc_pipeline.stats_ingest import convert_greco


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stats", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--output", default="data/raw/fight_stats_wide.csv")
    args = p.parse_args()

    wide = convert_greco(
        pd.read_csv(args.stats), pd.read_csv(args.results), pd.read_csv(args.events)
    )
    wide.to_csv(args.output, index=False)
    print(f"Wrote {len(wide)} wide stats rows -> {args.output}")


if __name__ == "__main__":
    main()
