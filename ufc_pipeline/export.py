"""Export the Elo baseline dataset + current fighter ratings to CSV.

Includes a leakage guard: the exported baseline is checked against a list of
forbidden column patterns (odds, rankings, precomputed averages, current-fight
stats, winner_*/loser_* features). Result/metadata fields `winner`,
`fighter_a_won`, `method`, `round` are explicitly allowed — they describe the
outcome and must never be used to compute pre-fight features, but they are
needed as labels/metadata.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .db import connect

# Substrings that must NOT appear in exported model-ready columns...
FORBIDDEN_COLUMN_PATTERNS = [
    "odds", "ranking", "rank", "winner_", "loser_",
    "sig_str", "takedown", "control", "avg",
]
# ...except these exact allowed result/metadata columns.
ALLOWED_RESULT_COLUMNS = {"winner", "fighter_a_won", "method", "round"}


def find_forbidden_columns(columns, allowed: set | None = None) -> list[str]:
    """Columns matching forbidden patterns, minus explicitly allowed names.

    `allowed` defaults to the result/metadata columns; Step 3B passes a
    larger set because its historical rolling stats legitimately contain
    words like sig_str/takedown/control in their names."""
    if allowed is None:
        allowed = ALLOWED_RESULT_COLUMNS
    bad = []
    for col in columns:
        if col in allowed:
            continue
        low = str(col).lower()
        if any(p in low for p in FORBIDDEN_COLUMN_PATTERNS):
            bad.append(col)
    return bad


BASELINE_QUERY = """
SELECT
    f.fight_id,
    f.date,
    f.event,
    fa.name  AS fighter_a,
    fb.name  AS fighter_b,
    fw.name  AS winner,
    s.fighter_a_won,
    s.fighter_a_pre_elo,
    s.fighter_b_pre_elo,
    s.elo_diff,
    s.fighter_a_expected_win_prob,
    s.fighter_b_expected_win_prob,
    s.fighter_a_post_elo,
    s.fighter_b_post_elo,
    f.weight_class,
    f.method,
    f.round
FROM fight_elo_snapshots s
JOIN fights   f  ON f.fight_id   = s.fight_id
JOIN fighters fa ON fa.fighter_id = f.fighter_a_id
JOIN fighters fb ON fb.fighter_id = f.fighter_b_id
LEFT JOIN fighters fw ON fw.fighter_id = f.winner_id
ORDER BY f.date, f.fight_id
"""

RATINGS_QUERY = """
WITH per_fighter AS (
    SELECT fighter_a_id AS fighter_id, date, fight_id,
           fighter_a_post_elo AS post_elo, fighter_a_won AS won
    FROM fight_elo_snapshots
    UNION ALL
    SELECT fighter_b_id AS fighter_id, date, fight_id,
           fighter_b_post_elo AS post_elo, 1 - fighter_a_won AS won
    FROM fight_elo_snapshots
),
latest AS (
    SELECT fighter_id, post_elo AS current_elo,
           ROW_NUMBER() OVER (PARTITION BY fighter_id
                              ORDER BY date DESC, fight_id DESC) AS rn
    FROM per_fighter
),
agg AS (
    SELECT fighter_id,
           COUNT(*)  AS fights_processed,
           SUM(won)  AS wins,
           COUNT(*) - SUM(won) AS losses,
           MAX(date) AS last_fight_date
    FROM per_fighter
    GROUP BY fighter_id
)
SELECT fi.name AS fighter,
       l.current_elo,
       a.fights_processed,
       a.wins,
       a.losses,
       a.last_fight_date
FROM agg a
JOIN latest l  ON l.fighter_id = a.fighter_id AND l.rn = 1
JOIN fighters fi ON fi.fighter_id = a.fighter_id
ORDER BY l.current_elo DESC
"""


def export_baseline(db_path: str, output_csv: str, ratings_csv: str) -> tuple[int, int]:
    conn = connect(db_path)
    try:
        baseline = pd.read_sql_query(BASELINE_QUERY, conn)
        ratings = pd.read_sql_query(RATINGS_QUERY, conn)
    finally:
        conn.close()

    bad = find_forbidden_columns(baseline.columns)
    if bad:
        raise RuntimeError(
            f"Leakage guard tripped: forbidden columns in export: {bad}. "
            "Only pre-fight Elo features plus result/metadata "
            f"{sorted(ALLOWED_RESULT_COLUMNS)} are allowed."
        )

    for path, df in ((output_csv, baseline), (ratings_csv, ratings)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    return len(baseline), len(ratings)
