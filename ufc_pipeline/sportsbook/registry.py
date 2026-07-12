"""Canonical source classification and traditional-sportsbook allowlist."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VenueType(str, Enum):
    SPORTSBOOK = "sportsbook"
    PREDICTION_MARKET = "prediction_market"
    EXCHANGE = "exchange"
    DAILY_FANTASY = "daily_fantasy"
    SWEEPSTAKES = "sweepstakes"
    PEER_TO_PEER = "peer_to_peer"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SportsbookDefinition:
    canonical_key: str
    display_name: str
    venue_type: VenueType
    approved: bool
    consensus_eligible: bool
    sharp_reference: bool = False
    secondary_candidate: bool = False
    provider_confirmed: bool = True
    parent_operator: str | None = None
    provider_specific_keys: tuple[str, ...] = ()
    historical_aliases: tuple[str, ...] = ()
    active_from: str | None = None
    active_to: str | None = None
    notes: str | None = None


SPORTSBOOKS: tuple[SportsbookDefinition, ...] = (
    SportsbookDefinition("draftkings", "DraftKings", VenueType.SPORTSBOOK, True, True, provider_specific_keys=("draftkings",)),
    SportsbookDefinition("fanduel", "FanDuel", VenueType.SPORTSBOOK, True, True, provider_specific_keys=("fanduel", "fanduel_us")),
    SportsbookDefinition("betmgm", "BetMGM", VenueType.SPORTSBOOK, True, True, provider_specific_keys=("betmgm",)),
    SportsbookDefinition("caesars", "Caesars Sportsbook", VenueType.SPORTSBOOK, True, True, provider_specific_keys=("caesars", "williamhill_us"), historical_aliases=("William Hill US",)),
    SportsbookDefinition("fanatics", "Fanatics Sportsbook", VenueType.SPORTSBOOK, True, True, parent_operator="Fanatics", provider_specific_keys=("fanatics", "fanatics_us", "pointsbetus"), historical_aliases=("PointsBet US",)),
    SportsbookDefinition("betrivers", "BetRivers", VenueType.SPORTSBOOK, True, True, parent_operator="Rush Street Interactive", provider_specific_keys=("betrivers", "betrivers_us", "sugarhouse"), historical_aliases=("SugarHouse",)),
    SportsbookDefinition("thescore_bet", "theScore Bet", VenueType.SPORTSBOOK, True, True, parent_operator="PENN Entertainment", provider_specific_keys=("thescorebet", "espnbet"), historical_aliases=("ESPN BET",), notes="One logical book across the historical brand transition."),
    SportsbookDefinition("pinnacle", "Pinnacle", VenueType.SPORTSBOOK, True, False, sharp_reference=True, provider_specific_keys=("pinnacle",)),
    SportsbookDefinition("hard_rock_bet", "Hard Rock Bet", VenueType.SPORTSBOOK, True, False, secondary_candidate=True, provider_specific_keys=("hardrockbet", "hard_rock_bet")),
    SportsbookDefinition("bet365", "bet365", VenueType.SPORTSBOOK, False, False, provider_confirmed=False, provider_specific_keys=("bet365", "bet365_us"), notes="Pending provider and historical coverage confirmation."),
    SportsbookDefinition("circa", "Circa Sports", VenueType.SPORTSBOOK, False, False, provider_confirmed=False, provider_specific_keys=("circa", "circasports"), notes="Pending provider and historical coverage confirmation."),
)


@dataclass(frozen=True)
class SourceResolution:
    raw_key: str
    canonical_key: str | None
    venue_type: VenueType
    approved: bool
    consensus_eligible: bool
    sharp_reference: bool
    reason: str | None = None


_BY_PROVIDER_KEY = {
    key.casefold(): item
    for item in SPORTSBOOKS
    for key in (item.canonical_key, *item.provider_specific_keys)
}

EXCLUDED_SOURCES: dict[str, VenueType] = {
    "kalshi": VenueType.PREDICTION_MARKET,
    "polymarket": VenueType.PREDICTION_MARKET,
    "betfair_ex": VenueType.EXCHANGE,
    "betfair_exchange": VenueType.EXCHANGE,
    "matchbook": VenueType.EXCHANGE,
    "smarkets": VenueType.EXCHANGE,
    "sporttrade": VenueType.PEER_TO_PEER,
    "prophet_exchange": VenueType.PEER_TO_PEER,
    "prizepicks": VenueType.DAILY_FANTASY,
    "underdog": VenueType.DAILY_FANTASY,
    "underdog_fantasy": VenueType.DAILY_FANTASY,
    "sleeper": VenueType.DAILY_FANTASY,
    "sleeper_picks": VenueType.DAILY_FANTASY,
    "fliff": VenueType.SWEEPSTAKES,
}


def resolve_provider_source(provider_key: str, *, approved_overrides: tuple[str, ...] = ()) -> SourceResolution:
    """Classify an exact provider key; display names are deliberately ignored."""
    raw = str(provider_key or "").strip()
    key = raw.casefold()
    item = _BY_PROVIDER_KEY.get(key)
    if item is not None:
        approved = item.approved or item.canonical_key in approved_overrides
        reason = None if approved else "sportsbook is pending provider confirmation and is not approved"
        return SourceResolution(raw, item.canonical_key, item.venue_type, approved, item.consensus_eligible, item.sharp_reference, reason)
    excluded = EXCLUDED_SOURCES.get(key)
    if excluded is not None:
        return SourceResolution(raw, None, excluded, False, False, False, f"venue type {excluded.value} is prohibited")
    return SourceResolution(raw, None, VenueType.UNKNOWN, False, False, False, "unknown provider source")


def sportsbook_by_key(canonical_key: str) -> SportsbookDefinition:
    for item in SPORTSBOOKS:
        if item.canonical_key == canonical_key:
            return item
    raise KeyError(canonical_key)
