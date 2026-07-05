#!/usr/bin/env python3
"""Build time-safe Elo snapshots for every fight in the database.

Usage:
  python scripts/build_elo.py --db data/ufc.db --k 32 --starting-elo 1500
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.elo import DEFAULT_K, DEFAULT_STARTING_ELO, build_elo_for_db


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/ufc.db", help="SQLite database path")
    p.add_argument("--k", type=float, default=DEFAULT_K, help="Elo K-factor")
    p.add_argument("--starting-elo", type=float, default=DEFAULT_STARTING_ELO,
                   help="Starting Elo for new fighters")
    args = p.parse_args()

    n = build_elo_for_db(args.db, k=args.k, starting_elo=args.starting_elo)
    print(f"Elo snapshots built for {n} fights "
          f"(K={args.k}, starting Elo={args.starting_elo}).")


if __name__ == "__main__":
    main()
