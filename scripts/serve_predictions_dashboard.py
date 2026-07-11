#!/usr/bin/env python3
"""Serve the read-only upcoming UFC predictions dashboard."""

import argparse
import logging
from pathlib import Path

import _path  # noqa: F401
from ufc_pipeline.predictions_dashboard import (
    DEFAULT_FEATURE_ROOT,
    DEFAULT_LEDGER_PATH,
    DEFAULT_STATIC_DIR,
    REPO_ROOT,
    DashboardConfig,
    serve_dashboard,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; loopback by default.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    parser.add_argument("--static-dir", default=str(DEFAULT_STATIC_DIR))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = DashboardConfig(
        ledger_path=Path(args.ledger),
        feature_root=Path(args.feature_root),
        static_dir=Path(args.static_dir),
        repo_root=REPO_ROOT,
    )
    serve_dashboard(config, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
