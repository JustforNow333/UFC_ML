"""Isolated traditional-sportsbook data infrastructure.

This package intentionally contains no model training or prediction-market code.
"""

from .config import SportsbookConfig, TheOddsApiConfig
from .consensus import ConsensusBuilder, ConsensusResult
from .odds import proportional_devig, to_decimal_odds
from .registry import SPORTSBOOKS, VenueType, resolve_provider_source
from .storage import SportsbookStore

__all__ = [
    "ConsensusBuilder",
    "ConsensusResult",
    "SPORTSBOOKS",
    "SportsbookConfig",
    "SportsbookStore",
    "TheOddsApiConfig",
    "VenueType",
    "proportional_devig",
    "resolve_provider_source",
    "to_decimal_odds",
]
