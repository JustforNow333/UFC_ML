#!/usr/bin/env python3
"""Build the expanded (Step 3 + Step 3B) pre-fight feature dataset.

Usage:
  python scripts/build_prefight_features_step3b.py --db data/ufc.db \
      --output data/processed/ufc_prefight_features_step3b.csv \
      [--debug-fighters "Fighter One" "Fighter Two"]
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.stats_features import build_step3b_for_db


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/ufc.db")
    p.add_argument("--output", default="data/processed/ufc_prefight_features_step3b.csv")
    p.add_argument("--debug-fighters", nargs="+", default=None,
                   help="Print these fighters' pre-fight rolling stats timelines")
    args = p.parse_args()

    n = build_step3b_for_db(args.db, args.output, debug_fighters=args.debug_fighters)
    print(f"Wrote {n} expanded feature rows -> {args.output}")


if __name__ == "__main__":
    main()
