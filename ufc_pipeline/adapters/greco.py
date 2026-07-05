"""Adapter for fights converted from Greco1899/scrape_ufc_stats.

`scripts/convert_greco_fights.py` joins `ufc_fight_results.csv` +
`ufc_event_details.csv` into a wide bout-level CSV (date, event,
red_fighter, blue_fighter, winner-as-name, weight_class, method, round).
That CSV uses the same generic red/blue column conventions as the mdabbert
adapter's fuzzy mapping, so the normalization logic is shared. Registering
it as its own source keeps provenance clean: fights.source = 'greco'.

Use ONE fights base per database build (mdabbert preferred when available,
Greco otherwise) — ingesting both would create duplicate bouts.
"""

from .mdabbert import adapt  # shared column-mapped normalization

__all__ = ["adapt"]
