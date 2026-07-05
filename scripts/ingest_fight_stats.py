#!/usr/bin/env python3
"""Ingest a wide per-fight stats CSV into the fight_stats table.

Usage:
  python scripts/ingest_fight_stats.py --input data/raw/fight_stats_wide.csv --db data/ufc.db
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.stats_ingest import ingest_stats_csv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Wide per-fight stats CSV")
    p.add_argument("--db", default="data/ufc.db")
    args = p.parse_args()

    report = ingest_stats_csv(args.input, args.db)
    print(report.summary())


if __name__ == "__main__":
    main()
