import math

import pytest

from ufc_pipeline.sportsbook.odds import (
    InvalidOddsError,
    implied_probability,
    proportional_devig,
    to_decimal_odds,
)
from ufc_pipeline.sportsbook.registry import VenueType, resolve_provider_source


@pytest.mark.parametrize(
    ("provider_key", "canonical_key"),
    [
        ("draftkings", "draftkings"),
        ("fanduel", "fanduel"),
        ("betmgm", "betmgm"),
        ("caesars", "caesars"),
        ("fanatics", "fanatics"),
        ("betrivers", "betrivers"),
        ("espnbet", "thescore_bet"),
    ],
)
def test_broad_consensus_sportsbooks_are_approved(provider_key, canonical_key):
    source = resolve_provider_source(provider_key)
    assert source.canonical_key == canonical_key
    assert source.venue_type == VenueType.SPORTSBOOK
    assert source.approved
    assert source.consensus_eligible


def test_historical_aliases_resolve_to_one_logical_sportsbook():
    assert resolve_provider_source("pointsbetus").canonical_key == "fanatics"
    assert resolve_provider_source("fanatics_us").canonical_key == "fanatics"
    assert resolve_provider_source("williamhill_us").canonical_key == "caesars"
    assert resolve_provider_source("sugarhouse").canonical_key == "betrivers"
    assert resolve_provider_source("thescorebet").canonical_key == resolve_provider_source("espnbet").canonical_key


def test_pinnacle_and_secondary_candidates_are_not_broad_by_default():
    pinnacle = resolve_provider_source("pinnacle")
    assert pinnacle.approved and pinnacle.sharp_reference and not pinnacle.consensus_eligible
    hard_rock = resolve_provider_source("hardrockbet")
    assert hard_rock.approved and not hard_rock.sharp_reference and not hard_rock.consensus_eligible


@pytest.mark.parametrize("provider_key", ["bet365", "circa"])
def test_pending_books_are_sportsbooks_but_not_approved(provider_key):
    source = resolve_provider_source(provider_key)
    assert source.venue_type == VenueType.SPORTSBOOK
    assert not source.approved


def test_pending_book_can_only_be_enabled_by_explicit_configuration():
    assert resolve_provider_source("bet365").approved is False
    assert resolve_provider_source("bet365", approved_overrides=("bet365",)).approved is True


@pytest.mark.parametrize(
    ("provider_key", "venue_type"),
    [
        ("kalshi", VenueType.PREDICTION_MARKET),
        ("polymarket", VenueType.PREDICTION_MARKET),
        ("betfair_exchange", VenueType.EXCHANGE),
        ("matchbook", VenueType.EXCHANGE),
        ("smarkets", VenueType.EXCHANGE),
        ("sporttrade", VenueType.PEER_TO_PEER),
        ("prophet_exchange", VenueType.PEER_TO_PEER),
        ("prizepicks", VenueType.DAILY_FANTASY),
        ("underdog_fantasy", VenueType.DAILY_FANTASY),
        ("sleeper_picks", VenueType.DAILY_FANTASY),
        ("fliff", VenueType.SWEEPSTAKES),
    ],
)
def test_prohibited_sources_are_classified_and_rejected(provider_key, venue_type):
    source = resolve_provider_source(provider_key)
    assert source.venue_type == venue_type
    assert not source.approved
    assert source.canonical_key is None


def test_unknown_source_is_rejected_without_display_name_guessing():
    source = resolve_provider_source("Definitely Draft Kings Maybe")
    assert source.venue_type == VenueType.UNKNOWN
    assert not source.approved


@pytest.mark.parametrize(
    ("value", "odds_format", "expected"),
    [(150, "american", 2.5), (-200, "american", 1.5), (2.25, "decimal", 2.25)],
)
def test_odds_conversion(value, odds_format, expected):
    assert to_decimal_odds(value, odds_format) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("value", "odds_format"),
    [
        (0, "american"),
        (1.0, "decimal"),
        (0.9, "decimal"),
        ("abc", "decimal"),
        (math.inf, "decimal"),
        (math.nan, "decimal"),
        (2.0, "fractional"),
    ],
)
def test_invalid_odds_fail_clearly(value, odds_format):
    with pytest.raises(InvalidOddsError):
        to_decimal_odds(value, odds_format)


def test_implied_probability_is_strictly_between_zero_and_one():
    probability = implied_probability(2.0)
    assert probability == pytest.approx(0.5)
    assert 0.0 < probability < 1.0


def test_proportional_devig_is_complementary_and_order_sensitive():
    forward = proportional_devig(1.8, 2.1)
    reverse = proportional_devig(2.1, 1.8)
    assert forward.overround == pytest.approx((1 / 1.8) + (1 / 2.1))
    assert forward.fair_probability_a + forward.fair_probability_b == pytest.approx(1.0)
    assert forward.fair_probability_a == pytest.approx(reverse.fair_probability_b)
    assert forward.fair_probability_b == pytest.approx(reverse.fair_probability_a)
