#!/usr/bin/env python3
"""Register Greco/UFCStats fighter URL identities in fighter_source_ids.

Run this BEFORE ingesting fights so ingestion can resolve every bout
fighter through a durable source identity (and split same-name fighters
instead of merging them). Safe to re-run any time: known URLs keep their
fighter_id links; re-running after ingestion refreshes the report.

Usage:
  python scripts/build_fighter_identities.py \
      --tott data/raw/greco/ufc_fighter_tott.csv --db data/ufc.db \
      --report data/processed/fighter_identity_report.json
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.db import connect, init_schema
from ufc_pipeline.identity import (
    load_greco_identities,
    register_greco_identities,
    write_identity_report,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tott", default="data/raw/greco/ufc_fighter_tott.csv")
    p.add_argument("--db", default="data/ufc.db")
    p.add_argument("--report", default="data/processed/fighter_identity_report.json")
    args = p.parse_args()

    identities = load_greco_identities(args.tott)
    conn = connect(args.db)
    try:
        init_schema(conn)
        report = register_greco_identities(conn, identities)
    finally:
        conn.close()
    write_identity_report(report, args.report)
    print(report.summary())
    print(f"\nIdentity report written to {args.report}")


if __name__ == "__main__":
    main()
