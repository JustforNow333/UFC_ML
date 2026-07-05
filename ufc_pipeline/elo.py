"""Elo baseline engine.

TIME SAFETY is the entire point of this module:

  1. Fights are processed strictly in chronological order (date ascending).
     Ties on the same date are broken by fight_id, which preserves the
     original dataset order from ingestion.
  2. For every fight, PRE-fight Elo and the Elo-implied win probability are
     recorded FIRST, and only then is the result applied to update ratings.
     A fight's outcome can therefore never influence its own (or any
     earlier) pre-fight numbers.

Update rule (classic Elo):
  expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
  new_elo_a  = old_elo_a + K * (actual_a - expected_a)
  new_elo_b  = old_elo_b + K * (actual_b - expected_b)
"""

from __future__ import annotations

from dataclasses import dataclass

from .db import connect, init_schema

DEFAULT_K = 32.0
DEFAULT_STARTING_ELO = 1500.0


def expected_score(elo_a: float, elo_b: float) -> float:
    """Elo-implied probability that fighter A beats fighter B."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


@dataclass
class EloSnapshot:
    fight_id: int
    date: str
    fighter_a_id: int
    fighter_b_id: int
    fighter_a_pre_elo: float
    fighter_b_pre_elo: float
    elo_diff: float
    fighter_a_expected_win_prob: float
    fighter_b_expected_win_prob: float
    fighter_a_post_elo: float
    fighter_b_post_elo: float
    fighter_a_won: int


def run_elo(
    fights: list[dict],
    k: float = DEFAULT_K,
    starting_elo: float = DEFAULT_STARTING_ELO,
) -> tuple[list[EloSnapshot], dict[int, float]]:
    """Run Elo over fights (each a dict with fight_id, date, fighter ids, result).

    Returns (snapshots, final_ratings). Pure function -> easy to test.
    Fights are sorted here by (date, fight_id) even if the caller passed them
    out of order, so input ordering mistakes cannot corrupt the timeline.
    """
    ordered = sorted(fights, key=lambda f: (f["date"], f["fight_id"]))

    ratings: dict[int, float] = {}
    snapshots: list[EloSnapshot] = []

    for fight in ordered:
        a, b = fight["fighter_a_id"], fight["fighter_b_id"]

        # Any fighter not seen before starts at starting_elo (default 1500).
        pre_a = ratings.get(a, starting_elo)
        pre_b = ratings.get(b, starting_elo)

        # --- record pre-fight state BEFORE the result touches anything ---
        exp_a = expected_score(pre_a, pre_b)
        exp_b = 1.0 - exp_a

        # --- only now apply the result ---
        actual_a = 1.0 if fight["fighter_a_won"] == 1 else 0.0
        actual_b = 1.0 - actual_a
        post_a = pre_a + k * (actual_a - exp_a)
        post_b = pre_b + k * (actual_b - exp_b)
        ratings[a] = post_a
        ratings[b] = post_b

        snapshots.append(
            EloSnapshot(
                fight_id=fight["fight_id"],
                date=fight["date"],
                fighter_a_id=a,
                fighter_b_id=b,
                fighter_a_pre_elo=pre_a,
                fighter_b_pre_elo=pre_b,
                elo_diff=pre_a - pre_b,
                fighter_a_expected_win_prob=exp_a,
                fighter_b_expected_win_prob=exp_b,
                fighter_a_post_elo=post_a,
                fighter_b_post_elo=post_b,
                fighter_a_won=int(fight["fighter_a_won"]),
            )
        )

    return snapshots, ratings


def build_elo_for_db(db_path: str, k: float, starting_elo: float) -> int:
    """Load fights from SQLite, run Elo, and (re)write fight_elo_snapshots."""
    conn = connect(db_path)
    try:
        init_schema(conn)
        rows = conn.execute(
            """SELECT fight_id, date, fighter_a_id, fighter_b_id, fighter_a_won
               FROM fights ORDER BY date, fight_id"""
        ).fetchall()
        fights = [
            dict(zip(("fight_id", "date", "fighter_a_id", "fighter_b_id", "fighter_a_won"), r))
            for r in rows
        ]

        snapshots, _ = run_elo(fights, k=k, starting_elo=starting_elo)

        # Rebuild snapshots from scratch each run: deterministic + idempotent.
        conn.execute("DELETE FROM fight_elo_snapshots")
        conn.executemany(
            """INSERT INTO fight_elo_snapshots
               (fight_id, date, fighter_a_id, fighter_b_id,
                fighter_a_pre_elo, fighter_b_pre_elo, elo_diff,
                fighter_a_expected_win_prob, fighter_b_expected_win_prob,
                fighter_a_post_elo, fighter_b_post_elo, fighter_a_won)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    s.fight_id, s.date, s.fighter_a_id, s.fighter_b_id,
                    s.fighter_a_pre_elo, s.fighter_b_pre_elo, s.elo_diff,
                    s.fighter_a_expected_win_prob, s.fighter_b_expected_win_prob,
                    s.fighter_a_post_elo, s.fighter_b_post_elo, s.fighter_a_won,
                )
                for s in snapshots
            ],
        )
        conn.commit()
        return len(snapshots)
    finally:
        conn.close()
