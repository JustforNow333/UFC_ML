"""Source adapters.

Each adapter exposes:  adapt(df: pd.DataFrame) -> pd.DataFrame
returning the normalized internal columns (see ufc_pipeline.columns).

To add a new source later (Greco1899/scrape_ufc_stats, UFC-DataLab, ...),
write one adapt() function and register it here.
"""

from . import greco, mdabbert

ADAPTERS = {
    "mdabbert": mdabbert.adapt,
    "greco": greco.adapt,   # fights converted by scripts/convert_greco_fights.py
}


def get_adapter(source: str):
    try:
        return ADAPTERS[source]
    except KeyError:
        raise ValueError(
            f"Unknown source '{source}'. Available sources: {sorted(ADAPTERS)}"
        ) from None
