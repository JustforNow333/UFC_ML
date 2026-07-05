#!/usr/bin/env python3
"""Ingest Greco fighter tale-of-the-tape (height/reach/stance/DOB) into the
fighters table. FILL-ONLY: existing values are never overwritten.

Usage:
  python scripts/ingest_fighter_details.py --input data/raw/greco/ufc_fighter_tott.csv --db data/ufc.db
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.fighter_details import ingest_fighter_tott


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--db", default="data/ufc.db")
    args = p.parse_args()

    print(ingest_fighter_tott(args.input, args.db).summary())


if __name__ == "__main__":
    main()
