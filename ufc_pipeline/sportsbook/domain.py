"""Provider-neutral sportsbook domain objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def parse_utc(value: str | datetime) -> datetime:
    """Parse an aware timestamp and normalize it to UTC."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    else:
        raise ValueError("timestamp must be a non-empty ISO-8601 string or datetime")
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def utc_iso(value: str | datetime) -> str:
    return parse_utc(value).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class NormalizedOutcome:
    raw_outcome_name: str
    raw_opponent_name: str | None
    decimal_odds: float | None
    american_odds: float | None
    raw_implied_probability: float | None
    original_odds: Any
    original_odds_format: str
    canonical_fighter_id: int | None = None
    canonical_opponent_id: int | None = None
    orientation_status: str = "unmatched"
    validation_status: str = "valid"
    validation_reason: str | None = None


@dataclass(frozen=True)
class NormalizedSnapshot:
    provider_name: str
    provider_event_id: str
    provider_market_id: str
    sportsbook_key: str | None
    provider_sportsbook_key: str
    sportsbook_display_name: str
    venue_type: str
    market_type: str
    observed_at_utc: str
    observed_at_original: str
    scheduled_event_time_utc: str
    scheduled_event_time_original: str
    pre_fight_or_live_status: str
    raw_event_name: str
    sport_key: str
    league_key: str | None
    outcomes: tuple[NormalizedOutcome, ...]
    normalization_status: str = "normalized"
    rejection_reason: str | None = None
    canonical_event_id: str | None = None
    canonical_bout_id: str | None = None
    matching_method: str | None = None
    matching_diagnostics: dict[str, Any] = field(default_factory=dict)
    source_payload_hash: str | None = None


@dataclass(frozen=True)
class NormalizationResult:
    provider_name: str
    snapshots: tuple[NormalizedSnapshot, ...]
    errors: tuple[dict[str, Any], ...] = ()
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    quota_metadata: dict[str, Any] = field(default_factory=dict)
