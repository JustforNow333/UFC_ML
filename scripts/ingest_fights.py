#!/usr/bin/env python3
"""Ingest a raw fights CSV into the SQLite database.

Usage:
  python scripts/ingest_fights.py --input data/raw/mdabbert_ufc.csv --db data/ufc.db --source mdabbert
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.ingest import ingest_csv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Path to the raw source CSV")
    p.add_argument("--db", default="data/ufc.db", help="SQLite database path")
    p.add_argument("--source", default="mdabbert", help="Source adapter name")
    args = p.parse_args()

    report = ingest_csv(args.input, args.db, args.source)
    print(report.summary())


if __name__ == "__main__":
    main()
