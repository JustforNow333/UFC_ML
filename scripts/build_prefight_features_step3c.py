#!/usr/bin/env python3
"""Build the Step 3 + 3B + 3C (style-matchup) pre-fight feature dataset.

Includes every Step 3 and Step 3B column unchanged, plus rolling "against"
stats (takedowns allowed, control/knockdowns/submissions absorbed, ...) and
style-matchup interaction features. Positive matchup values always mean a
fighter A advantage. Writes to its own versioned output; never overwrites
the Step 3B dataset.

Usage:
  python scripts/build_prefight_features_step3c.py --db data/ufc.db \
      --output data/processed/ufc_prefight_features_step3c.csv \
      [--debug-fighters "Fighter One" "Fighter Two"]
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.matchup_features import build_step3c_for_db


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/ufc.db")
    p.add_argument("--output", default="data/processed/ufc_prefight_features_step3c.csv")
    p.add_argument("--debug-fighters", nargs="+", default=None,
                   help="Print these fighters' rolling-stat and matchup timelines")
    args = p.parse_args()

    n = build_step3c_for_db(args.db, args.output, debug_fighters=args.debug_fighters)
    print(f"Wrote {n} Step 3C feature rows -> {args.output}")


if __name__ == "__main__":
    main()
