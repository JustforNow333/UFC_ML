"""Fighter tale-of-the-tape ingestion from Greco `ufc_fighter_tott.csv`.

Parses HEIGHT (feet'inches"), REACH (inches"), STANCE, DOB into the
fighters table, converted to cm / ISO dates.

Provenance rule (Category B in docs/greco_field_audit.md): FILL-ONLY.
A fighter's existing height/reach/stance/date_of_birth is NEVER overwritten;
only NULL fields are filled. Per-fight mdabbert values (stored on the fights
table) always take precedence over these static values at feature time via
COALESCE, so mdabbert data is never displaced by Greco data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .columns import clean_display_name, name_key
from .dates import to_iso_date_or_none
from .db import connect, init_schema


def _text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def parse_height_cm(value) -> float | None:
    """`5' 11"` -> 180.34 cm; '--'/missing -> None."""
    m = re.match(r"\s*(\d+)'\s*(\d+)", _text(value))
    if not m:
        return None
    inches = int(m.group(1)) * 12 + int(m.group(2))
    return round(inches * 2.54, 1)


def parse_reach_cm(value) -> float | None:
    """`72"` -> 182.88 cm; '--'/missing -> None."""
    text = _text(value)
    m = re.match(r"\s*(\d+(?:\.\d+)?)", text)
    if not m or text.startswith("-"):
        return None
    return round(float(m.group(1)) * 2.54, 1)


def parse_dob_iso(value) -> str | None:
    """'Jul 15, 1988' -> '1988-07-15'; '--'/missing -> None."""
    s = _text(value)
    if not s or s.startswith("-"):
        return None
    return to_iso_date_or_none(s)


@dataclass
class TottReport:
    tott_rows: int = 0
    fighters_in_db: int = 0
    matched: int = 0
    matched_by_url: int = 0
    matched_by_name: int = 0
    skipped_ambiguous_name: int = 0
    fields_filled: int = 0
    skipped_existing: int = 0

    def summary(self) -> str:
        return (
            f"tott rows in source file  : {self.tott_rows}\n"
            f"fighters in database      : {self.fighters_in_db}\n"
            f"matched (total)           : {self.matched}\n"
            f"  via source URL identity : {self.matched_by_url}\n"
            f"  via unique name fallback: {self.matched_by_name}\n"
            f"skipped (ambiguous name)  : {self.skipped_ambiguous_name} "
            "(same-name group without a linked URL; never guessed)\n"
            f"fields filled (were null) : {self.fields_filled}\n"
            f"fields left untouched     : {self.skipped_existing} "
            "(already had a value; never overwritten)"
        )


def ingest_fighter_tott(csv_path: str, db_path: str) -> TottReport:
    """URL-first tott ingestion (identity layer, Part A).

    Resolution per row: (1) the row's fighter URL, if linked in
    fighter_source_ids, gives the fighter_id directly — this is what stops
    e.g. the 1965 kickboxer Michael McDonald's DOB landing on the 1991
    bantamweight's row; (2) otherwise a unique normalized-name match is
    used ONLY when the name is unambiguous (one fighter row, and not a
    same-name identity group); ambiguous names are counted and skipped.
    """
    from .identity import url_slug

    report = TottReport()
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    report.tott_rows = len(df)

    conn = connect(db_path)
    try:
        init_schema(conn)
        by_slug = {
            slug: fid for slug, fid in conn.execute(
                "SELECT source_fighter_id, fighter_id FROM fighter_source_ids "
                "WHERE fighter_id IS NOT NULL"
            )
        }
        name_counts: dict[str, int] = {}
        identity_name_counts: dict[str, int] = {}
        db_fighters = {}
        for fid, key in conn.execute(
            "SELECT fighter_id, normalized_name FROM fighters"
        ):
            name_counts[key] = name_counts.get(key, 0) + 1
            db_fighters[key] = fid
        for (key,) in conn.execute(
            "SELECT normalized_name FROM fighter_source_ids"
        ):
            identity_name_counts[key] = identity_name_counts.get(key, 0) + 1
        report.fighters_in_db = len(db_fighters)

        for _, row in df.iterrows():
            slug = url_slug(row.get("url"))
            key = name_key(clean_display_name(row.get("fighter")))
            fid = by_slug.get(slug)
            if fid is not None:
                report.matched_by_url += 1
            else:
                # Name fallback only when provably unambiguous.
                if name_counts.get(key, 0) != 1 or identity_name_counts.get(key, 0) > 1:
                    if name_counts.get(key, 0) > 1 or identity_name_counts.get(key, 0) > 1:
                        report.skipped_ambiguous_name += 1
                    continue
                fid = db_fighters[key]
                report.matched_by_name += 1
            report.matched += 1
            values = {
                "height": parse_height_cm(row.get("height")),
                "reach": parse_reach_cm(row.get("reach")),
                "stance": (str(row["stance"]).strip() or None)
                if pd.notna(row.get("stance")) else None,
                "date_of_birth": parse_dob_iso(row.get("dob")),
            }
            for col, val in values.items():
                if val is None:
                    continue
                # FILL-ONLY: update succeeds only where the field is NULL.
                cur = conn.execute(
                    f"UPDATE fighters SET {col} = ?, updated_at = datetime('now') "
                    f"WHERE fighter_id = ? AND {col} IS NULL",
                    (val, fid),
                )
                if cur.rowcount:
                    report.fields_filled += 1
                else:
                    report.skipped_existing += 1
        conn.commit()
    finally:
        conn.close()
    return report
