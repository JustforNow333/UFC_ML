"""Configuration for the isolated sportsbook data layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SportsbookConfig:
    database_path: Path = Path("data/sportsbook/sportsbook_odds.db")
    raw_archive_root: Path = Path("data/sportsbook/raw")
    canonical_ufc_db: Path = Path("data/ufc.db")
    reports_dir: Path = Path("reports/sportsbook")
    schema_version: int = 1
    devig_method: str = "proportional"
    devig_version: str = "proportional_v1"
    consensus_policy_version: str = "sportsbook_consensus_median_v1"
    selection_policy_version: str = "newest_complete_prefight_at_or_before_cutoff_v1"
    minimum_normal_coverage: int = 3
    include_pinnacle_in_broad_consensus: bool = False
    max_snapshot_age_hours: float | None = None
    approved_sportsbook_overrides: tuple[str, ...] = ()
    additional_consensus_sportsbook_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("database_path", "raw_archive_root", "canonical_ufc_db", "reports_dir"):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))


@dataclass(frozen=True)
class TheOddsApiConfig:
    api_key_env: str = "THE_ODDS_API_KEY"
    base_url: str = "https://api.the-odds-api.com/v4"
    sport_key: str = "mma_mixed_martial_arts"
    regions: tuple[str, ...] = ("us", "us2", "eu")
    markets: tuple[str, ...] = ("h2h",)
    odds_format: str = "decimal"
    date_format: str = "iso"
    timeout_seconds: float = 30.0
    retry_count: int = 2
    backoff_seconds: float = 0.5
    requested_bookmakers: tuple[str, ...] = field(default_factory=tuple)
    approved_sportsbook_overrides: tuple[str, ...] = field(default_factory=tuple)
