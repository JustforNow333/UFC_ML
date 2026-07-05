"""Ingestion: raw CSV -> source adapter -> validation -> SQLite.

The ingestion layer intentionally keeps ONLY identity + result fields.
Odds, rankings, and precomputed averages are dropped at the adapter stage
so they can never leak into pre-fight features downstream.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from .adapters import get_adapter
from .columns import name_key
from .dates import to_datetime_mixed
from .db import connect, init_schema
from .identity import GRECO_SOURCE, load_identities_by_name, plan_identity_assignments


@dataclass
class IngestReport:
    total_rows: int = 0
    ingested: int = 0
    dropped_no_winner: int = 0        # draws / no-contests / unparseable winner
    dropped_bad_names: int = 0        # null names or fighter_a == fighter_b
    dropped_bad_date: int = 0
    duplicate_rows_reported: int = 0
    # Identity layer: bout-fighter slots resolved via source identities.
    identity_resolved_slots: int = 0
    identity_new_links: int = 0
    identity_unresolved: list[dict] = field(default_factory=list)
    same_name_splits: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"rows in source file        : {self.total_rows}",
            f"fights ingested            : {self.ingested}",
            f"dropped (no decisive winner): {self.dropped_no_winner}",
            f"dropped (bad fighter names) : {self.dropped_bad_names}",
            f"dropped (unparseable date)  : {self.dropped_bad_date}",
            f"duplicate rows reported     : {self.duplicate_rows_reported}",
        ]
        if self.identity_resolved_slots or self.identity_unresolved:
            lines += [
                f"identity-resolved slots     : {self.identity_resolved_slots}",
                f"new identity links          : {self.identity_new_links}",
                f"unresolved identity slots   : {len(self.identity_unresolved)}",
            ]
            for case in self.same_name_splits:
                lines.append(f"  same-name SPLIT (not merged): {case}")
            for case in self.identity_unresolved[:10]:
                lines.append(f"  UNRESOLVED identity: {case}")
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def validate_and_clean(df: pd.DataFrame, report: IngestReport) -> pd.DataFrame:
    """Validate normalized fights; drop rows that cannot be used for Elo.

    Elo needs a decisive result, so draws/no-contests are dropped (reported).
    """
    out = df.copy()

    # --- names present and distinct ---
    bad_names = (
        out["fighter_a_name"].isna()
        | out["fighter_b_name"].isna()
        | (out["fighter_a_name"].astype(str).str.strip() == "")
        | (out["fighter_b_name"].astype(str).str.strip() == "")
        | (
            out["fighter_a_name"].map(name_key)
            == out["fighter_b_name"].map(name_key)
        )
    )
    report.dropped_bad_names = int(bad_names.sum())
    out = out[~bad_names]

    # --- date parseable ---
    parsed = to_datetime_mixed(out["date"], errors="coerce")
    report.dropped_bad_date = int(parsed.isna().sum())
    out = out[parsed.notna()].copy()
    out["date"] = parsed[parsed.notna()].dt.strftime("%Y-%m-%d")

    # --- winner must be one of the two fighters (else: draw/NC -> drop) ---
    wk = out["winner_name"].map(lambda v: name_key(v) if pd.notna(v) else "")
    ak = out["fighter_a_name"].map(name_key)
    bk = out["fighter_b_name"].map(name_key)
    a_won = wk == ak
    b_won = wk == bk
    decisive = a_won | b_won
    report.dropped_no_winner = int((~decisive).sum())
    out = out[decisive].copy()
    out["fighter_a_won"] = a_won[decisive].astype(int)

    # --- duplicates: report, keep first exact duplicate only ---
    dup_key = out[["date", "fighter_a_name", "fighter_b_name"]].apply(
        lambda r: (r["date"], name_key(r["fighter_a_name"]), name_key(r["fighter_b_name"])),
        axis=1,
    )
    dups = dup_key.duplicated(keep="first")
    report.duplicate_rows_reported = int(dups.sum())
    if report.duplicate_rows_reported:
        report.warnings.append(
            f"{report.duplicate_rows_reported} duplicate (date, fighter_a, fighter_b) "
            "rows found; kept the first occurrence of each."
        )
    out = out[~dups]

    return out


def _get_or_create_fighter(
    conn: sqlite3.Connection,
    cache: dict,
    display_name: str,
    slug: str | None = None,
    report: IngestReport | None = None,
    identity_source: str = GRECO_SOURCE,
) -> int:
    """Resolve a bout-fighter slot to a canonical fighter_id.

    slug given (identity path, used whenever the source registered an
    identity for this name):
      1. identity already linked -> that fighter_id, always.
      2. else claim the first same-name fighter no OTHER identity of this
         source owns, or create a new fighter row — this is what allows two
         canonical fighters to share a normalized name without merging.
    slug None (legacy path: name unknown to the source, or unresolved
    same-name slot): reuse/create a fighter row not owned by any source
    identity, so unresolved bouts never silently land on a split fighter.
    """
    if slug is not None:
        slug_key = ("slug", identity_source, slug)
        if slug_key in cache:
            return cache[slug_key]
        row = conn.execute(
            "SELECT fighter_id FROM fighter_source_ids "
            "WHERE source = ? AND source_fighter_id = ? AND fighter_id IS NOT NULL",
            (identity_source, slug),
        ).fetchone()
        if row is not None:
            cache[slug_key] = row[0]
            return row[0]
        key = name_key(display_name)
        unclaimed = conn.execute(
            """SELECT f.fighter_id FROM fighters f
               WHERE f.normalized_name = ?
                 AND NOT EXISTS (SELECT 1 FROM fighter_source_ids s
                                 WHERE s.fighter_id = f.fighter_id)
               ORDER BY f.fighter_id LIMIT 1""",
            (key,),
        ).fetchone()
        if unclaimed is not None:
            fid = unclaimed[0]
        else:
            fid = conn.execute(
                "INSERT INTO fighters (name, normalized_name) VALUES (?, ?)",
                (display_name, key),
            ).lastrowid
        conn.execute(
            """UPDATE fighter_source_ids
               SET fighter_id = ?, resolution = 'linked',
                   resolution_note = COALESCE(resolution_note, '') ||
                                     ' [linked at ingestion]',
                   updated_at = datetime('now')
               WHERE source = ? AND source_fighter_id = ?""",
            (fid, identity_source, slug),
        )
        if report is not None:
            report.identity_new_links += 1
        cache[slug_key] = fid
        return fid

    key = name_key(display_name)
    if ("name", key) in cache:
        return cache[("name", key)]
    row = conn.execute(
        """SELECT f.fighter_id FROM fighters f
           WHERE f.normalized_name = ?
             AND NOT EXISTS (SELECT 1 FROM fighter_source_ids s
                             WHERE s.fighter_id = f.fighter_id)
           ORDER BY f.fighter_id LIMIT 1""",
        (key,),
    ).fetchone()
    if row is not None:
        cache[("name", key)] = row[0]
        return row[0]
    cur = conn.execute(
        "INSERT INTO fighters (name, normalized_name) VALUES (?, ?)",
        (display_name, key),
    )
    cache[("name", key)] = cur.lastrowid
    return cur.lastrowid


def ingest_csv(csv_path: str, db_path: str, source: str) -> IngestReport:
    """Full ingestion: read CSV, adapt, validate, write fighters + fights."""
    report = IngestReport()

    raw = pd.read_csv(csv_path)
    report.total_rows = len(raw)

    adapter = get_adapter(source)
    normalized = adapter(raw)
    clean = validate_and_clean(normalized, report).reset_index(drop=True)

    conn = connect(db_path)
    try:
        init_schema(conn)
        # Identity layer: if source identities are registered (run
        # scripts/build_fighter_identities.py first), resolve every
        # bout-fighter slot through them BEFORE inserting, so same-name
        # fighters split correctly regardless of the fights source.
        identities_by_name = load_identities_by_name(conn)
        plan: dict = {}
        if identities_by_name:
            plan, unresolved = plan_identity_assignments(clean, identities_by_name)
            report.identity_unresolved = unresolved
            # Report every same-name group that was split, with bout counts.
            split_counts: dict[str, dict] = {}
            for (pos, side), slug in plan.items():
                key = name_key(
                    clean.iloc[pos]["fighter_a_name" if side == "a" else "fighter_b_name"]
                )
                if len(identities_by_name.get(key, [])) > 1:
                    split_counts.setdefault(key, {})
                    label = slug or "UNRESOLVED"
                    split_counts[key][label] = split_counts[key].get(label, 0) + 1
            report.same_name_splits = [
                {"name": k, "bout_slots_per_identity": v}
                for k, v in sorted(split_counts.items())
            ]

        cache: dict = {}
        seen_dates: dict[str, list[str]] = {}
        # Insert in source order so fight_id preserves dataset order,
        # which the Elo builder uses to break same-date ties.
        for pos, row in clean.iterrows():
            slug_a = plan.get((pos, "a"))
            slug_b = plan.get((pos, "b"))
            a_id = _get_or_create_fighter(
                conn, cache, row["fighter_a_name"], slug_a, report
            )
            b_id = _get_or_create_fighter(
                conn, cache, row["fighter_b_name"], slug_b, report
            )
            for slug in (slug_a, slug_b):
                if slug is not None:
                    report.identity_resolved_slots += 1
                    seen_dates.setdefault(slug, []).append(row["date"])
            winner_id = a_id if row["fighter_a_won"] == 1 else b_id
            def _num(field: str):
                """Optional numeric field -> float or None (never guessed)."""
                val = row.get(field)
                return float(val) if val is not None and pd.notna(val) else None

            conn.execute(
                """INSERT INTO fights
                   (date, event, fighter_a_id, fighter_b_id, winner_id,
                    fighter_a_won, weight_class, method, round, source, source_row_id,
                    fighter_a_age, fighter_b_age,
                    fighter_a_height, fighter_b_height,
                    fighter_a_reach, fighter_b_reach)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["date"],
                    row.get("event"),
                    a_id,
                    b_id,
                    winner_id,
                    int(row["fighter_a_won"]),
                    row.get("weight_class"),
                    row.get("method"),
                    str(row["round"]) if pd.notna(row.get("round")) else None,
                    source,
                    int(row["source_row_id"]),
                    _num("fighter_a_age"), _num("fighter_b_age"),
                    _num("fighter_a_height"), _num("fighter_b_height"),
                    _num("fighter_a_reach"), _num("fighter_b_reach"),
                ),
            )
            report.ingested += 1

        # Identity bookkeeping: when was each source identity seen in fights.
        for slug, dates in seen_dates.items():
            conn.execute(
                """UPDATE fighter_source_ids
                   SET first_seen_date = MIN(COALESCE(first_seen_date, ?), ?),
                       last_seen_date = MAX(COALESCE(last_seen_date, ?), ?),
                       updated_at = datetime('now')
                   WHERE source_fighter_id = ?""",
                (min(dates), min(dates), max(dates), max(dates), slug),
            )
        conn.commit()
    finally:
        conn.close()

    return report
