"""Fighter identity layer (Part A): durable source-specific fighter IDs.

Why this exists: the pipeline used to key fighters purely on a normalized
name, which silently MERGED different fighters who share a name. In the real
Greco data there are 7 such names (14 people) — e.g. two Bruno Silvas, a
flyweight and a middleweight, whose 21 fights shared one Elo rating and one
rolling-stats history. Greco's `ufc_fighter_tott.csv` carries UFCStats
fighter URLs (`/fighter-details/<hex>`), which are durable per-person IDs;
this module registers them in the fighter_source_ids table and uses them,
plus supporting evidence, to keep same-name fighters apart.

Matching priority (deterministic, never guessed):
  1. Known source URL already linked -> use the existing fighter_id.
  2. New URL whose normalized name is UNIQUE among source identities and
     matches exactly one canonical fighter -> link.
  3. Same-name groups (2+ URLs sharing a normalized name) are resolved
     per bout during ingestion using supporting evidence, in order:
       a. exact division-weight match (bout weight class vs the identity's
          listed weight: Flyweight bout -> the 125 lbs identity),
       b. career continuity (the bout is within 540 days of a bout already
          assigned to exactly one of the candidates — handles one-off
          division moves, e.g. Jean Silva's 2024 lightweight bout),
       c. DOB plausibility (age 18-45 at fight date filters candidates).
  4. Still ambiguous -> UNRESOLVED: reported, and the bout falls back to a
     shared name-keyed fighter row that is clearly flagged. No guessing,
     no silent merging.

The raw source name is always stored alongside the normalized key for
debugging. Existing fighter_id values are never remapped by registration;
same-name splits only materialize on (re-)ingestion of fights.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date as date_cls

import pandas as pd

from .columns import clean_display_name, name_key
from .dates import to_iso_date_or_none

GRECO_SOURCE = "greco"

# Evidence rules for same-name disambiguation.
CONTINUITY_WINDOW_DAYS = 540
DOB_PLAUSIBLE_AGE = (18.0, 45.0)

# UFC division -> contracted weight limit in lbs. Women's divisions share the
# number. Catch/Open weight and unknown divisions give no weight evidence.
DIVISION_WEIGHT_LBS = {
    "atomweight": 105.0,
    "strawweight": 115.0,
    "flyweight": 125.0,
    "bantamweight": 135.0,
    "featherweight": 145.0,
    "lightweight": 155.0,
    "welterweight": 170.0,
    "middleweight": 185.0,
    "light heavyweight": 205.0,
    "heavyweight": 265.0,
    "super heavyweight": 300.0,
}


def division_weight_lbs(weight_class) -> float | None:
    """'Women's Flyweight' -> 125.0; Catch Weight/None -> None."""
    if weight_class is None or (isinstance(weight_class, float) and pd.isna(weight_class)):
        return None
    s = str(weight_class).strip().lower()
    s = re.sub(r"^women'?s\s+", "", s)
    return DIVISION_WEIGHT_LBS.get(s)


def url_slug(url) -> str | None:
    """Stable source fighter id: the final path segment of the URL.

    'http://ufcstats.com/fighter-details/93fe7332d16c6ad9' -> '93fe7332d16c6ad9'
    """
    if url is None or (isinstance(url, float) and pd.isna(url)):
        return None
    s = str(url).strip().rstrip("/")
    if not s:
        return None
    return s.rsplit("/", 1)[-1] or None


def _parse_weight_lbs(value) -> float | None:
    """'155 lbs.' -> 155.0; '--'/missing -> None."""
    m = re.match(r"\s*(\d+(?:\.\d+)?)", str(value or ""))
    return float(m.group(1)) if m else None


def _age_at(fight_date: str, dob: str) -> float | None:
    try:
        f = date_cls.fromisoformat(fight_date)
        b = date_cls.fromisoformat(dob)
    except (TypeError, ValueError):
        return None
    return (f - b).days / 365.25


@dataclass(frozen=True)
class SourceIdentity:
    slug: str
    url: str | None
    raw_name: str
    key: str                      # normalized name
    weight_lbs: float | None
    dob: str | None
    height_cm: float | None
    reach_cm: float | None
    fighter_id: int | None = None  # linked canonical fighter, if any


# ---------------------------------------------------------------------------
# Registration: tott CSV -> fighter_source_ids rows
# ---------------------------------------------------------------------------

def load_greco_identities(tott_csv: str) -> list[SourceIdentity]:
    """Parse Greco `ufc_fighter_tott.csv` into SourceIdentity records."""
    from .fighter_details import parse_height_cm, parse_reach_cm

    df = pd.read_csv(tott_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    out: list[SourceIdentity] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        slug = url_slug(row.get("url"))
        raw = clean_display_name(row.get("fighter"))
        if slug is None or not raw or slug in seen:
            continue
        seen.add(slug)
        out.append(
            SourceIdentity(
                slug=slug,
                url=str(row.get("url")).strip(),
                raw_name=raw,
                key=name_key(raw),
                weight_lbs=_parse_weight_lbs(row.get("weight")),
                dob=to_iso_date_or_none(row.get("dob"))
                if str(row.get("dob") or "").strip() not in ("", "--") else None,
                height_cm=parse_height_cm(row.get("height")),
                reach_cm=parse_reach_cm(row.get("reach")),
            )
        )
    return out


@dataclass
class IdentityReport:
    total_identities: int = 0
    already_linked: int = 0
    newly_linked_by_name: int = 0
    unlinked_not_in_db: int = 0
    ambiguous_same_name: int = 0
    duplicate_normalized_names: list[str] = field(default_factory=list)
    same_name_not_merged: list[dict] = field(default_factory=list)
    linked_examples: list[dict] = field(default_factory=list)
    unresolved_examples: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"greco identities in source file : {self.total_identities}",
            f"already linked to a fighter     : {self.already_linked}",
            f"newly linked by unique name     : {self.newly_linked_by_name}",
            f"unlinked (fighter not in DB)    : {self.unlinked_not_in_db}",
            f"ambiguous same-name identities  : {self.ambiguous_same_name}",
            f"duplicate normalized names      : {len(self.duplicate_normalized_names)}"
            f" {self.duplicate_normalized_names}",
        ]
        for case in self.same_name_not_merged:
            lines.append(f"  same-name, NOT merged: {case}")
        for ex in self.unresolved_examples[:10]:
            lines.append(f"  unresolved: {ex}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_identities": self.total_identities,
            "already_linked": self.already_linked,
            "newly_linked_by_name": self.newly_linked_by_name,
            "unlinked_not_in_db": self.unlinked_not_in_db,
            "ambiguous_same_name": self.ambiguous_same_name,
            "duplicate_normalized_names": self.duplicate_normalized_names,
            "same_name_not_merged": self.same_name_not_merged,
            "linked_examples": self.linked_examples[:20],
            "unresolved_examples": self.unresolved_examples,
        }


def register_greco_identities(
    conn: sqlite3.Connection, identities: list[SourceIdentity]
) -> IdentityReport:
    """Upsert identities into fighter_source_ids and link the safe ones.

    Linking here handles priorities 1-2 only (known URL, unique-name match).
    Same-name groups are deliberately left for evidence-based resolution at
    ingestion time; on an already-merged legacy DB they are marked ambiguous
    and reported instead of guessed.
    """
    report = IdentityReport(total_identities=len(identities))

    by_key: dict[str, list[SourceIdentity]] = {}
    for ident in identities:
        by_key.setdefault(ident.key, []).append(ident)
    report.duplicate_normalized_names = sorted(
        k for k, v in by_key.items() if len(v) > 1
    )

    for ident in identities:
        # Upsert WITHOUT touching an existing fighter_id link (priority 1:
        # a known URL keeps its canonical fighter forever).
        conn.execute(
            """INSERT INTO fighter_source_ids
                 (source, source_fighter_id, source_fighter_url, raw_name,
                  normalized_name, listed_weight_lbs, date_of_birth,
                  height_cm, reach_cm)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(source, source_fighter_id) DO UPDATE SET
                 source_fighter_url = excluded.source_fighter_url,
                 raw_name = excluded.raw_name,
                 normalized_name = excluded.normalized_name,
                 listed_weight_lbs = excluded.listed_weight_lbs,
                 date_of_birth = excluded.date_of_birth,
                 height_cm = excluded.height_cm,
                 reach_cm = excluded.reach_cm,
                 updated_at = datetime('now')""",
            (GRECO_SOURCE, ident.slug, ident.url, ident.raw_name, ident.key,
             ident.weight_lbs, ident.dob, ident.height_cm, ident.reach_cm),
        )

    for key, group in by_key.items():
        slugs = [g.slug for g in group]
        linked = {
            slug: fid
            for slug, fid in conn.execute(
                f"""SELECT source_fighter_id, fighter_id FROM fighter_source_ids
                    WHERE source = ? AND source_fighter_id IN
                    ({','.join('?' * len(slugs))}) AND fighter_id IS NOT NULL""",
                [GRECO_SOURCE, *slugs],
            )
        }
        report.already_linked += len(linked)
        unlinked = [g for g in group if g.slug not in linked]
        if not unlinked:
            continue

        if len(group) > 1:
            # Same-name group: never link by name alone. Report and defer to
            # evidence-based per-bout resolution during ingestion.
            for g in unlinked:
                conn.execute(
                    """UPDATE fighter_source_ids
                       SET resolution = 'ambiguous',
                           resolution_note = ?, updated_at = datetime('now')
                       WHERE source = ? AND source_fighter_id = ?""",
                    (f"same-name group ({len(group)} identities share "
                     f"'{key}'); resolved per bout at ingestion",
                     GRECO_SOURCE, g.slug),
                )
                report.ambiguous_same_name += 1
            report.same_name_not_merged.append(
                {
                    "normalized_name": key,
                    "identities": [
                        {"slug": g.slug, "weight_lbs": g.weight_lbs, "dob": g.dob}
                        for g in group
                    ],
                }
            )
            continue

        ident = unlinked[0]
        # Priority 2: unique normalized-name match against canonical fighters
        # that no OTHER greco identity already claims.
        rows = conn.execute(
            """SELECT f.fighter_id FROM fighters f
               WHERE f.normalized_name = ?
                 AND NOT EXISTS (SELECT 1 FROM fighter_source_ids s
                                 WHERE s.source = ? AND s.fighter_id = f.fighter_id)
               ORDER BY f.fighter_id""",
            (key, GRECO_SOURCE),
        ).fetchall()
        if len(rows) == 1:
            conn.execute(
                """UPDATE fighter_source_ids
                   SET fighter_id = ?, resolution = 'linked',
                       resolution_note = 'unique normalized-name match',
                       updated_at = datetime('now')
                   WHERE source = ? AND source_fighter_id = ?""",
                (rows[0][0], GRECO_SOURCE, ident.slug),
            )
            report.newly_linked_by_name += 1
            if len(report.linked_examples) < 20:
                report.linked_examples.append(
                    {"slug": ident.slug, "name": ident.raw_name,
                     "fighter_id": rows[0][0], "how": "unique name match"}
                )
        elif len(rows) == 0:
            report.unlinked_not_in_db += 1
        else:
            # Multiple same-name canonical fighters but only one identity:
            # cannot pick safely by name -> unresolved, reported.
            conn.execute(
                """UPDATE fighter_source_ids
                   SET resolution = 'unresolved',
                       resolution_note = ?, updated_at = datetime('now')
                   WHERE source = ? AND source_fighter_id = ?""",
                (f"{len(rows)} canonical fighters share name '{key}'",
                 GRECO_SOURCE, ident.slug),
            )
            report.unresolved_examples.append(
                {"slug": ident.slug, "name": ident.raw_name,
                 "reason": f"{len(rows)} canonical fighters share this name"}
            )
    conn.commit()
    return report


def write_identity_report(report: IdentityReport, path: str) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report.to_dict(), fh, indent=2)


# ---------------------------------------------------------------------------
# Ingestion-time resolution (priorities 3-4: evidence-based, per bout)
# ---------------------------------------------------------------------------

def load_identities_by_name(
    conn: sqlite3.Connection, source: str = GRECO_SOURCE
) -> dict[str, list[SourceIdentity]]:
    """All registered identities for a source, grouped by normalized name."""
    out: dict[str, list[SourceIdentity]] = {}
    for slug, url, raw, key, w, dob, h, r, fid in conn.execute(
        """SELECT source_fighter_id, source_fighter_url, raw_name,
                  normalized_name, listed_weight_lbs, date_of_birth,
                  height_cm, reach_cm, fighter_id
           FROM fighter_source_ids WHERE source = ?
           ORDER BY source_fighter_id""",
        (source,),
    ):
        out.setdefault(key, []).append(
            SourceIdentity(slug=slug, url=url, raw_name=raw, key=key,
                           weight_lbs=w, dob=dob, height_cm=h, reach_cm=r,
                           fighter_id=fid)
        )
    return out


def assign_bouts_to_identities(
    bouts: list[tuple], candidates: list[SourceIdentity]
) -> dict:
    """Partition one name's bouts among its same-name candidate identities.

    bouts: list of (bout_ref, fight_date_iso, weight_class).
    Returns {bout_ref: slug or None}; None = unresolved (never guessed).

    Evidence order (documented in the module docstring): exact division
    weight -> career continuity -> DOB plausibility.
    """
    assign: dict = {}
    dates_by_slug: dict[str, list[date_cls]] = {c.slug: [] for c in candidates}

    def _date(iso):
        try:
            return date_cls.fromisoformat(str(iso))
        except (TypeError, ValueError):
            return None

    # a. exact division-weight match, unique among candidates
    for ref, fdate, wc in bouts:
        lbs = division_weight_lbs(wc)
        if lbs is None:
            continue
        hits = [c for c in candidates if c.weight_lbs == lbs]
        if len(hits) == 1:
            assign[ref] = hits[0].slug
            d = _date(fdate)
            if d:
                dates_by_slug[hits[0].slug].append(d)

    # b. career continuity: repeat until stable so assignments can chain
    changed = True
    while changed:
        changed = False
        for ref, fdate, wc in bouts:
            if ref in assign:
                continue
            d = _date(fdate)
            if d is None:
                continue
            near = [
                c for c in candidates
                if any(abs((d - ad).days) <= CONTINUITY_WINDOW_DAYS
                       for ad in dates_by_slug[c.slug])
            ]
            if len(near) == 1:
                assign[ref] = near[0].slug
                dates_by_slug[near[0].slug].append(d)
                changed = True

    # c. DOB plausibility: age 18-45 at fight date filters to one candidate
    lo, hi = DOB_PLAUSIBLE_AGE
    for ref, fdate, wc in bouts:
        if ref in assign:
            continue
        plausible = []
        for c in candidates:
            if c.dob is None:
                continue
            age = _age_at(str(fdate), c.dob)
            if age is not None and lo <= age <= hi:
                plausible.append(c)
        if len(plausible) == 1:
            assign[ref] = plausible[0].slug

    # d. anything left is honestly unresolved
    for ref, _, _ in bouts:
        assign.setdefault(ref, None)
    return assign


def plan_identity_assignments(
    clean: pd.DataFrame, identities_by_name: dict[str, list[SourceIdentity]]
) -> tuple[dict, list[dict]]:
    """Pre-compute the identity slug for every bout-fighter slot.

    Returns (plan, unresolved):
      plan: {(row_position, 'a'|'b'): slug or None}. A key is present only
        when the name has registered identities; value None = unresolved
        same-name case (caller falls back to a shared name-keyed fighter
        and reports it).
      unresolved: list of {name, date, weight_class} report entries.
    """
    plan: dict = {}
    unresolved: list[dict] = []
    groups: dict[str, list[tuple]] = {}

    for pos, row in enumerate(clean.itertuples(index=False)):
        for side, name in (("a", row.fighter_a_name), ("b", row.fighter_b_name)):
            key = name_key(name)
            cands = identities_by_name.get(key)
            if not cands:
                continue  # unknown to the source: legacy name matching
            if len(cands) == 1:
                plan[(pos, side)] = cands[0].slug
            else:
                groups.setdefault(key, []).append(
                    ((pos, side), row.date, getattr(row, "weight_class", None))
                )

    for key, bouts in groups.items():
        result = assign_bouts_to_identities(bouts, identities_by_name[key])
        for (ref, fdate, wc) in bouts:
            plan[ref] = result[ref]
            if result[ref] is None:
                unresolved.append(
                    {"name": key, "date": str(fdate), "weight_class": wc}
                )
    return plan, unresolved
