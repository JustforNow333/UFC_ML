#!/usr/bin/env python3
"""Step 6D.1: read-only UFCStats data-update audit (dry run).

Compares UFCStats (fetched live, or from a --cache-dir of saved HTML) against the
local data/ufc.db and reports missing/stale events, fights, fighters, and
results. It NEVER modifies the database — a guarded apply is deferred to a
future Step 6D.2. No sportsbook odds, no prediction markets, no new heavy deps.

UFCStats serves a JavaScript browser-challenge to plain HTTP clients; if a live
fetch is blocked, save the pages from a real browser into --cache-dir and re-run
with --offline-cache-only.

Example:
  python scripts/run_step6d1_ufcstats_audit.py --db data/ufc.db \
      --output-dir reports/data_update --max-events 10 --include-upcoming
"""
import argparse

import _path  # noqa: F401
from ufc_pipeline.step6d_ufcstats_update_audit import DEFAULT_OUTPUT_DIR, DEFAULT_REQUEST_DELAY, run_audit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="data/ufc.db")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-events", type=int, default=10, help="Bounded cap on events audited (safe default).")
    p.add_argument("--include-upcoming", action="store_true", default=False)
    p.add_argument("--start-date", default=None, help="Only audit fetched events on/after this date (YYYY-MM-DD).")
    p.add_argument("--end-date", default=None, help="Only audit fetched events on/before this date (YYYY-MM-DD).")
    p.add_argument("--cache-dir", default=None, help="Directory of cached UFCStats HTML (read + write-through).")
    p.add_argument("--offline-cache-only", action="store_true", default=False,
                   help="Never hit the network; use only cached HTML.")
    p.add_argument("--delay", type=float, default=DEFAULT_REQUEST_DELAY, help="Polite delay between live requests (s).")
    args = p.parse_args()

    report = run_audit(
        db_path=args.db, output_dir=args.output_dir, max_events=args.max_events,
        include_upcoming=args.include_upcoming, start_date=args.start_date, end_date=args.end_date,
        cache_dir=args.cache_dir, offline_cache_only=args.offline_cache_only, delay=args.delay,
    )
    print(f"Step 6D.1 UFCStats audit (READ-ONLY, no DB writes):")
    print(f"  fetch available: {report['fetch_available']} | events fetched: {report['n_fetched_events']} | "
          f"fights: {report['n_fetched_fights']} | fighters: {report['n_fetched_fighters']}")
    print(f"  missing local: events={len(report['missing_local_events'])}, fights={len(report['missing_local_fights'])}, "
          f"fighters={len(report['missing_local_fighters'])} | stale results={len(report['stale_or_mismatched_results'])}")
    print(f"  ambiguous names={len(report['ambiguous_fighter_names'])} | upcoming={report['n_upcoming_events']}")
    print(f"  -> {report['recommended_next_action']}")
    print(f"  report: {report['report_md']}")


if __name__ == "__main__":
    main()
