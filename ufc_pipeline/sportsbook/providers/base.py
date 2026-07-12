"""Provider-neutral adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..domain import NormalizationResult


class ProviderError(RuntimeError):
    pass


class ProviderAuthenticationError(ProviderError):
    pass


class ProviderRateLimitError(ProviderError):
    pass


class ProviderTransientError(ProviderError):
    pass


class ProviderSchemaError(ProviderError):
    pass


@dataclass(frozen=True)
class ProviderMetadata:
    provider_name: str
    supports_current: bool
    supports_historical: bool
    historical_start: str | None = None
    documentation_url: str | None = None
    supported_sports: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderResponse:
    payload: Any
    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    request_type: str = "current"
    query_timestamp_utc: str | None = None


class ProviderAdapter(ABC):
    @property
    @abstractmethod
    def metadata(self) -> ProviderMetadata:
        raise NotImplementedError

    @abstractmethod
    def list_supported_sports(self, **kwargs: Any) -> ProviderResponse:
        raise NotImplementedError

    @abstractmethod
    def fetch_current_odds(self, **kwargs: Any) -> ProviderResponse:
        raise NotImplementedError

    @abstractmethod
    def fetch_historical_odds(self, as_of: str | datetime, **kwargs: Any) -> ProviderResponse:
        raise NotImplementedError

    @abstractmethod
    def normalize(self, payload: Any, *, request_type: str, source_payload_hash: str | None = None) -> NormalizationResult:
        raise NotImplementedError

    @abstractmethod
    def validate_required_fields(self, payload: Any, *, request_type: str) -> tuple[str, ...]:
        raise NotImplementedError

    @staticmethod
    def quota_metadata(headers: dict[str, str]) -> dict[str, str]:
        return {}
