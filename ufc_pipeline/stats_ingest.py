"""Per-fight stats ingestion (Step 3B).

Two pieces:

1. A "wide" stats CSV ingester: one row per fight with date, red/blue
   fighter names, total fight minutes, and per-corner offensive totals
   (KD, sig strikes landed/attempted, TD landed/attempted, sub attempts,
   control seconds). Rows are matched to already-ingested fights by
   (date, unordered fighter-name pair) and written to the fight_stats
   table — two rows per fight, one per fighter.

2. `convert_greco(...)`: converts the three Greco1899/scrape_ufc_stats
   files (ufc_fight_stats.csv per-round rows, ufc_fight_results.csv,
   ufc_event_details.csv) into that wide format. So getting real stats is:
       download the 3 CSVs from the repo
       python scripts/convert_greco_stats.py ... --output data/raw/fight_stats_wide.csv
       python scripts/ingest_fight_stats.py --input data/raw/fight_stats_wide.csv --db data/ufc.db

Time-safety note: these are the CURRENT fight's totals — they are outcomes
of that fight and are NEVER features for it. They are stored so the rolling
builder can use them for the fighter's *later* fights only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .columns import _canon, clean_display_name, find_column, name_key
from .dates import to_iso_date_or_none
from .db import connect, init_schema

# ---------------------------------------------------------------------------
# Wide-format column mapping (separate from the fight-identity mapping)
# ---------------------------------------------------------------------------

_STAT_FIELDS = {
    "kd": ["kd", "knockdowns"],
    "sig_str_landed": ["sig_str_landed", "sig_strikes_landed", "sigstrlanded"],
    "sig_str_attempted": ["sig_str_attempted", "sig_strikes_attempted", "sigstratt"],
    "td_landed": ["td_landed", "takedowns_landed"],
    "td_attempted": ["td_attempted", "takedowns_attempted"],
    "sub_att": ["sub_att", "submission_attempts", "sub_attempts"],
    "ctrl_seconds": ["ctrl_seconds", "control_time_seconds", "ctrl_time_seconds"],
}
_MINUTES_CANDIDATES = ["total_minutes", "minutes", "fight_minutes", "total_fight_minutes"]
_CORNERS = {"a": ["red", "r", "fighter_a"], "b": ["blue", "b", "fighter_b"]}


def _text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _find_stat_column(df: pd.DataFrame, corner: str, stat: str) -> str | None:
    canon_map = {_canon(c): c for c in df.columns}
    for prefix in _CORNERS[corner]:
        for cand in _STAT_FIELDS[stat]:
            hit = canon_map.get(_canon(f"{prefix}_{cand}"))
            if hit is not None:
                return hit
    return None


@dataclass
class StatsIngestReport:
    total_rows: int = 0
    matched: int = 0
    unmatched: int = 0
    # Why rows were unmatched (identity layer, Part A: never lump reasons):
    unmatched_bad_date: int = 0          # date failed to parse
    unmatched_unknown_fighter: int = 0   # a fighter name not in the DB at all
    unmatched_unresolved_identity: int = 0  # name is an unresolved same-name slot
    unmatched_no_fight_on_date: int = 0  # fighters known, but no fight that day
    missing_minutes: int = 0
    unmatched_examples: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"stats rows in source file : {self.total_rows}",
            f"matched to known fights   : {self.matched}",
            f"unmatched (no fight found): {self.unmatched}",
            f"  bad/unparseable date    : {self.unmatched_bad_date}",
            f"  fighter not in DB       : {self.unmatched_unknown_fighter}",
            f"  unresolved identity     : {self.unmatched_unresolved_identity}",
            f"  no fight on that date   : {self.unmatched_no_fight_on_date} "
            "(usually draws/NCs dropped at fight ingestion)",
            f"skipped (missing minutes) : {self.missing_minutes}",
        ]
        for ex in self.unmatched_examples[:10]:
            lines.append(f"  unmatched example: {ex}")
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def ingest_stats_csv(csv_path: str, db_path: str, source: str = "greco") -> StatsIngestReport:
    """Match wide stats rows to fights and populate fight_stats."""
    report = StatsIngestReport()
    df = pd.read_csv(csv_path)
    report.total_rows = len(df)

    date_col = find_column(df, "date")
    a_col = find_column(df, "fighter_a_name")
    b_col = find_column(df, "fighter_b_name")
    minutes_col = None
    canon_map = {_canon(c): c for c in df.columns}
    for cand in _MINUTES_CANDIDATES:
        minutes_col = canon_map.get(_canon(cand))
        if minutes_col:
            break
    if not all([date_col, a_col, b_col, minutes_col]):
        raise ValueError(
            "Stats CSV is missing required columns.\n"
            f"  Needed: date, red/blue fighter names, total_minutes\n"
            f"  Columns found: {list(df.columns)}"
        )

    stat_cols = {
        (corner, stat): _find_stat_column(df, corner, stat)
        for corner in ("a", "b")
        for stat in _STAT_FIELDS
    }
    missing_stats = sorted({s for (c, s), col in stat_cols.items() if col is None})
    if missing_stats:
        report.warnings.append(
            f"stat columns not found (stored as null): {missing_stats}"
        )

    conn = connect(db_path)
    try:
        init_schema(conn)
        # (date, unordered name pair) -> (fight_id, a_key, a_id, b_id).
        # Identity note: each fight row already references CANONICAL fighter
        # ids (same-name fighters were split at fight ingestion via
        # fighter_source_ids), so resolving a stats row to a fight resolves
        # both fighters' identities transitively. Corner alignment below is
        # by name key, which is safe because a fight cannot pair two
        # fighters with the same key (rejected at ingestion).
        fights = {}
        for fid, date, a_id, b_id, a_key, b_key in conn.execute(
            """SELECT f.fight_id, f.date, f.fighter_a_id, f.fighter_b_id,
                      fa.normalized_name, fb.normalized_name
               FROM fights f
               JOIN fighters fa ON fa.fighter_id = f.fighter_a_id
               JOIN fighters fb ON fb.fighter_id = f.fighter_b_id"""
        ):
            fights[(date, frozenset((a_key, b_key)))] = (fid, a_key, a_id, b_id)

        known_keys = {
            key for (key,) in conn.execute("SELECT normalized_name FROM fighters")
        }
        unresolved_keys = {
            key for (key,) in conn.execute(
                """SELECT DISTINCT normalized_name FROM fighter_source_ids
                   WHERE resolution IN ('ambiguous', 'unresolved')
                     AND fighter_id IS NULL"""
            )
        }

        def _record_unmatched(reason: str, **detail) -> None:
            report.unmatched += 1
            setattr(report, reason, getattr(report, reason) + 1)
            if len(report.unmatched_examples) < 50:
                report.unmatched_examples.append({"reason": reason, **detail})

        for _, row in df.iterrows():
            date = to_iso_date_or_none(row[date_col])
            if date is None:
                _record_unmatched(
                    "unmatched_bad_date", raw_date=str(row[date_col]),
                    fighters=f"{row[a_col]} vs {row[b_col]}",
                )
                continue
            key_a = name_key(clean_display_name(row[a_col]))
            key_b = name_key(clean_display_name(row[b_col]))
            hit = fights.get((date, frozenset((key_a, key_b))))
            if hit is None:
                unknown = [k for k in (key_a, key_b) if k not in known_keys]
                unresolved = [k for k in (key_a, key_b) if k in unresolved_keys]
                if unknown:
                    reason = "unmatched_unknown_fighter"
                elif unresolved:
                    reason = "unmatched_unresolved_identity"
                else:
                    reason = "unmatched_no_fight_on_date"
                _record_unmatched(
                    reason, date=date, fighters=f"{row[a_col]} vs {row[b_col]}"
                )
                continue
            fid, fight_a_key, a_id, b_id = hit

            minutes = pd.to_numeric(row[minutes_col], errors="coerce")
            if pd.isna(minutes) or minutes <= 0:
                report.missing_minutes += 1
                continue

            def stat(corner: str, s: str):
                col = stat_cols[(corner, s)]
                if col is None:
                    return None
                v = pd.to_numeric(row[col], errors="coerce")
                return float(v) if pd.notna(v) else None

            # The CSV's "a" corner may be the fight's fighter_b (sources can
            # disagree on red/blue) — align by name key, not by position.
            csv_a_is_fight_a = key_a == fight_a_key
            side_for = {
                a_id: "a" if csv_a_is_fight_a else "b",
                b_id: "b" if csv_a_is_fight_a else "a",
            }
            for fighter_id, corner in side_for.items():
                conn.execute(
                    """INSERT OR REPLACE INTO fight_stats
                       (fight_id, fighter_id, minutes, knockdowns,
                        sig_str_landed, sig_str_attempted,
                        td_landed, td_attempted, sub_attempts, ctrl_seconds,
                        source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        fid, fighter_id, float(minutes),
                        stat(corner, "kd"),
                        stat(corner, "sig_str_landed"),
                        stat(corner, "sig_str_attempted"),
                        stat(corner, "td_landed"),
                        stat(corner, "td_attempted"),
                        stat(corner, "sub_att"),
                        stat(corner, "ctrl_seconds"),
                        source,
                    ),
                )
            report.matched += 1
        conn.commit()
    finally:
        conn.close()
    return report


# ---------------------------------------------------------------------------
# Greco1899/scrape_ufc_stats -> wide format
# ---------------------------------------------------------------------------

def _parse_of(value) -> tuple[float, float]:
    """'37 of 102' -> (37, 102); missing/junk -> (0, 0)."""
    m = re.match(r"\s*(\d+)\s+of\s+(\d+)", _text(value))
    return (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)


def _parse_mmss_seconds(value) -> float:
    """'2:03' -> 123 seconds; '--'/missing -> 0."""
    m = re.match(r"\s*(\d+):(\d{2})", _text(value))
    return float(int(m.group(1)) * 60 + int(m.group(2))) if m else 0.0


def _round_lengths(time_format) -> list[float]:
    """'3 Rnd (5-5-5)' -> [5,5,5]; unknown -> five 5-minute rounds."""
    m = re.search(r"\(([\d\-]+)\)", _text(time_format))
    if m:
        return [float(x) for x in m.group(1).split("-")]
    return [5.0] * 5


def _fight_minutes(final_round, final_time, time_format) -> float | None:
    try:
        rnd = int(final_round)
    except (TypeError, ValueError):
        return None
    lengths = _round_lengths(time_format)
    full = sum(lengths[: max(rnd - 1, 0)])
    return full + _parse_mmss_seconds(final_time) / 60.0


def convert_greco(
    stats: pd.DataFrame, results: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """Join Greco's three files into one wide row per fight.

    stats  : per-round rows (EVENT, BOUT, FIGHTER, KD, 'SIG.STR.' 'x of y',
             TD 'x of y', SUB.ATT, CTRL 'm:ss', ...)
    results: per-fight rows (EVENT, BOUT, ROUND, TIME, TIME FORMAT, ...)
    events : EVENT -> DATE
    """
    def canon_cols(df):
        return df.rename(columns={c: _canon(c) for c in df.columns})

    stats, results, events = map(canon_cols, (stats, results, events))

    event_date = {}
    for _, r in events.iterrows():
        date = to_iso_date_or_none(r["date"])
        if date is not None:
            event_date[str(r["event"]).strip()] = date

    # Pre-group per-round stat rows by (event, bout) once — the per-fight
    # filtering alternative is quadratic and unusable at real data volume
    # (8,700 fights x 41,000 rows).
    stats = stats.copy()
    stats["_event"] = stats["event"].astype(str).str.strip()
    stats["_bout"] = stats["bout"].astype(str).str.strip()
    stats_by_bout = {k: g for k, g in stats.groupby(["_event", "_bout"], sort=False)}

    rows = []
    for _, res in results.iterrows():
        event = str(res["event"]).strip()
        bout = str(res["bout"]).strip()
        fighters = re.split(r"\s+vs\.?\s+", bout, flags=re.IGNORECASE)
        if len(fighters) != 2 or event not in event_date:
            continue
        red_name, blue_name = (clean_display_name(f) for f in fighters)
        minutes = _fight_minutes(
            res.get("round"), res.get("time"), res.get("timeformat")
        )
        if minutes is None or minutes <= 0:
            continue

        bout_stats = stats_by_bout.get((event, bout))
        if bout_stats is None or bout_stats.empty:
            continue

        totals = {"red": None, "blue": None}
        for corner, cname in (("red", red_name), ("blue", blue_name)):
            frows = bout_stats[
                bout_stats["fighter"].map(lambda v: name_key(str(v))) == name_key(cname)
            ]
            if frows.empty:
                continue
            sig = frows["sig.str."].map(_parse_of)
            td = frows["td"].map(_parse_of)
            totals[corner] = {
                "kd": pd.to_numeric(frows["kd"], errors="coerce").fillna(0).sum(),
                "sl": sum(x[0] for x in sig),
                "sa": sum(x[1] for x in sig),
                "tdl": sum(x[0] for x in td),
                "tda": sum(x[1] for x in td),
                "sub": pd.to_numeric(frows["sub.att"], errors="coerce").fillna(0).sum(),
                "ctrl": frows["ctrl"].map(_parse_mmss_seconds).sum(),
            }
        if totals["red"] is None or totals["blue"] is None:
            continue

        row = {
            "date": event_date[event],
            "red_fighter": red_name,
            "blue_fighter": blue_name,
            "total_minutes": round(minutes, 3),
        }
        for corner in ("red", "blue"):
            t = totals[corner]
            row.update(
                {
                    f"{corner}_kd": t["kd"],
                    f"{corner}_sig_str_landed": t["sl"],
                    f"{corner}_sig_str_attempted": t["sa"],
                    f"{corner}_td_landed": t["tdl"],
                    f"{corner}_td_attempted": t["tda"],
                    f"{corner}_sub_att": t["sub"],
                    f"{corner}_ctrl_seconds": t["ctrl"],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


_DIVISIONS = [  # most-specific first so "Light Heavyweight" wins over "Heavyweight"
    "Super Heavyweight", "Light Heavyweight", "Heavyweight", "Middleweight",
    "Welterweight", "Lightweight", "Featherweight", "Bantamweight",
    "Flyweight", "Strawweight", "Atomweight", "Open Weight", "Catch Weight",
    "Catchweight",
]


def normalize_weight_class(value) -> str | None:
    """Greco WEIGHTCLASS strings -> clean division names.

    'UFC Women's Flyweight Title Bout'   -> "Women's Flyweight"
    'Ultimate Fighter ... Welterweight Tournament Title Bout' -> 'Welterweight'
    'UFC Superfight Championship Bout'   -> None (no division named)
    mdabbert's already-clean labels pass through unchanged.
    """
    s = _text(value)
    if not s:
        return None
    womens = "women" in s.lower()
    for div in _DIVISIONS:
        if div.lower() in s.lower():
            div = "Catch Weight" if div == "Catchweight" else div
            return f"Women's {div}" if womens else div
    return None


def convert_greco_fights(results: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Greco fight results + events -> bout-level fights CSV (winner as name).

    OUTCOME encodes the result from the first-listed (red) fighter's side:
    'W/L' red won, 'L/W' blue won, 'D/D' draw, 'NC/NC' no contest.
    Draws/NCs get winner='' and are dropped (and reported) at ingestion,
    matching the Elo requirement for decisive results.
    """
    def canon_cols(df):
        return df.rename(columns={c: _canon(c) for c in df.columns})

    results, events = map(canon_cols, (results, events))
    event_date = {}
    for _, r in events.iterrows():
        date = to_iso_date_or_none(r["date"])
        if date is not None:
            event_date[str(r["event"]).strip()] = date

    rows = []
    for _, res in results.iterrows():
        event = str(res["event"]).strip()
        bout = str(res["bout"]).strip()
        fighters = re.split(r"\s+vs\.?\s+", bout, flags=re.IGNORECASE)
        if len(fighters) != 2 or event not in event_date:
            continue
        red, blue = (clean_display_name(f) for f in fighters)
        outcome = str(res.get("outcome") or "").strip().upper()
        winner = red if outcome == "W/L" else blue if outcome == "L/W" else ""
        rows.append(
            {
                "date": event_date[event],
                "event": event,
                "red_fighter": red,
                "blue_fighter": blue,
                "winner": winner,
                "weight_class": normalize_weight_class(res.get("weightclass")),
                "method": str(res.get("method") or "").strip() or None,
                "round": res.get("round"),
            }
        )
    return pd.DataFrame(rows)
