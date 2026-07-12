"""Odds validation, conversion, and proportional de-vigging."""

from __future__ import annotations

import math
from dataclasses import dataclass


class InvalidOddsError(ValueError):
    pass


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise InvalidOddsError(f"{label} must be numeric")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise InvalidOddsError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise InvalidOddsError(f"{label} must be finite")
    return number


def american_to_decimal(value: object) -> float:
    american = _finite_number(value, "American odds")
    if american == 0:
        raise InvalidOddsError("American odds cannot be zero")
    decimal = 1.0 + (american / 100.0 if american > 0 else 100.0 / abs(american))
    return validate_decimal_odds(decimal)


def validate_decimal_odds(value: object) -> float:
    decimal = _finite_number(value, "decimal odds")
    if decimal <= 1.0:
        raise InvalidOddsError("decimal odds must be greater than 1")
    probability = 1.0 / decimal
    if not 0.0 < probability < 1.0:
        raise InvalidOddsError("implied probability must be strictly between 0 and 1")
    return decimal


def to_decimal_odds(value: object, odds_format: str) -> float:
    fmt = str(odds_format).strip().casefold()
    if fmt in {"decimal", "decimal_odds"}:
        return validate_decimal_odds(value)
    if fmt in {"american", "american_odds"}:
        return american_to_decimal(value)
    raise InvalidOddsError(f"unsupported odds format: {odds_format!r}")


def implied_probability(decimal_odds: object) -> float:
    return 1.0 / validate_decimal_odds(decimal_odds)


@dataclass(frozen=True)
class DevigResult:
    raw_probability_a: float
    raw_probability_b: float
    overround: float
    fair_probability_a: float
    fair_probability_b: float
    method: str = "proportional"
    version: str = "proportional_v1"


def proportional_devig(decimal_odds_a: object, decimal_odds_b: object, *, tolerance: float = 1e-12) -> DevigResult:
    raw_a = implied_probability(decimal_odds_a)
    raw_b = implied_probability(decimal_odds_b)
    overround = raw_a + raw_b
    if not math.isfinite(overround) or overround <= 0.0:
        raise InvalidOddsError("overround must be finite and positive")
    fair_a = raw_a / overround
    fair_b = raw_b / overround
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (fair_a, fair_b)):
        raise InvalidOddsError("de-vigged probabilities must be finite and within [0, 1]")
    if abs((fair_a + fair_b) - 1.0) > tolerance:
        raise InvalidOddsError("de-vigged probabilities do not sum to one")
    return DevigResult(raw_a, raw_b, overround, fair_a, fair_b)
