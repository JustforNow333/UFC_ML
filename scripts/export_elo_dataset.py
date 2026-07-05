#!/usr/bin/env python3
"""Export the Elo baseline dataset and current fighter ratings to CSV.

Usage:
  python scripts/export_elo_dataset.py --db data/ufc.db \
      --output data/processed/ufc_elo_baseline.csv \
      --ratings-output data/processed/current_fighter_elos.csv
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.export import export_baseline


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/ufc.db", help="SQLite database path")
    p.add_argument("--output", default="data/processed/ufc_elo_baseline.csv")
    p.add_argument("--ratings-output", default="data/processed/current_fighter_elos.csv")
    args = p.parse_args()

    n_fights, n_fighters = export_baseline(args.db, args.output, args.ratings_output)
    print(f"Wrote {n_fights} fights  -> {args.output}")
    print(f"Wrote {n_fighters} fighters -> {args.ratings_output}")


if __name__ == "__main__":
    main()
