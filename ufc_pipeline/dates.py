"""Date parsing helpers shared by ingestion/conversion code."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=1)
def _supports_format_mixed() -> bool:
    """True when this pandas version supports ``format="mixed"`` parsing."""
    try:
        parsed = pd.to_datetime(["2000-01-02"], errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return False
    return bool(pd.notna(parsed[0]))


def to_datetime_mixed(values, *, errors: str = "raise"):
    """Parse scalar/array dates across pandas 1.x and 2.x.

    Pandas 2 needs ``format="mixed"`` for arrays containing multiple date
    formats, while pandas 1 treats that value as a literal strptime format.
    """
    if _supports_format_mixed():
        return pd.to_datetime(values, errors=errors, format="mixed")
    return pd.to_datetime(values, errors=errors)


def to_iso_date_or_none(value) -> str | None:
    """Return ``YYYY-MM-DD`` for a parseable scalar date, otherwise ``None``."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    parsed = to_datetime_mixed(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return str(parsed.date())
