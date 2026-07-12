"""The Odds API adapter using documented current and historical shapes.

No request is made unless network access is explicitly enabled. Historical
requests additionally require explicit acknowledgement that the call may be
billable.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Callable

from ..config import TheOddsApiConfig
from ..domain import NormalizationResult, NormalizedOutcome, NormalizedSnapshot, parse_utc, utc_iso
from ..odds import InvalidOddsError, american_to_decimal, implied_probability, to_decimal_odds
from ..registry import VenueType, resolve_provider_source
from .base import (
    ProviderAdapter,
    ProviderAuthenticationError,
    ProviderError,
    ProviderMetadata,
    ProviderRateLimitError,
    ProviderResponse,
    ProviderSchemaError,
    ProviderTransientError,
)


Transport = Callable[[urllib.request.Request, float], tuple[int, dict[str, str], bytes]]


def _stdlib_transport(request: urllib.request.Request, timeout: float) -> tuple[int, dict[str, str], bytes]:
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310 - explicit gated provider URL
        return int(response.status), dict(response.headers.items()), response.read()


class TheOddsApiAdapter(ProviderAdapter):
    provider_name = "the_odds_api"

    def __init__(
        self,
        config: TheOddsApiConfig | None = None,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = config or TheOddsApiConfig()
        self._transport = transport or _stdlib_transport
        self._sleep = sleep

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider_name=self.provider_name,
            supports_current=True,
            supports_historical=True,
            historical_start="2020-06",
            documentation_url="https://the-odds-api.com/liveapi/guides/v4/",
            supported_sports=(self.config.sport_key,),
        )

    @staticmethod
    def quota_metadata(headers: dict[str, str]) -> dict[str, str]:
        wanted = {"x-requests-remaining", "x-requests-used", "x-requests-last"}
        return {key.casefold(): value for key, value in headers.items() if key.casefold() in wanted}

    def _api_key(self) -> str:
        key = os.environ.get(self.config.api_key_env)
        if not key:
            raise ProviderAuthenticationError(f"missing API key environment variable {self.config.api_key_env}")
        return key

    def _request(
        self,
        path: str,
        params: dict[str, str],
        *,
        request_type: str,
        query_timestamp_utc: str | None = None,
        allow_network: bool = False,
        allow_billable_request: bool = False,
    ) -> ProviderResponse:
        if not allow_network:
            raise ProviderError("network access is disabled; pass --allow-network explicitly")
        if request_type == "historical" and not allow_billable_request:
            raise ProviderError("historical requests may be billable; pass --allow-billable-request explicitly")
        params = dict(params)
        params["apiKey"] = self._api_key()
        url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "ufc-elo-sportsbook-stage2/1"})
        last_error: BaseException | None = None
        for attempt in range(self.config.retry_count + 1):
            try:
                status, headers, body = self._transport(request, self.config.timeout_seconds)
                if status in {401, 403}:
                    raise ProviderAuthenticationError("provider authentication failed")
                if status == 429:
                    raise ProviderRateLimitError("provider rate limit exceeded")
                if status >= 500:
                    raise ProviderTransientError(f"provider returned transient HTTP {status}")
                if status >= 400:
                    raise ProviderError(f"provider returned HTTP {status}")
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ProviderSchemaError("provider returned invalid JSON") from exc
                return ProviderResponse(payload, status, headers, request_type, query_timestamp_utc)
            except urllib.error.HTTPError as exc:
                if exc.code in {401, 403}:
                    raise ProviderAuthenticationError("provider authentication failed") from exc
                if exc.code == 429:
                    raise ProviderRateLimitError("provider rate limit exceeded") from exc
                if exc.code < 500:
                    raise ProviderError(f"provider returned HTTP {exc.code}") from exc
                last_error = ProviderTransientError(f"provider returned transient HTTP {exc.code}")
            except (urllib.error.URLError, TimeoutError, ProviderTransientError) as exc:
                last_error = exc
            if attempt < self.config.retry_count:
                self._sleep(self.config.backoff_seconds * (2**attempt))
        raise ProviderTransientError("provider request failed after bounded retries") from last_error

    def list_supported_sports(self, *, allow_network: bool = False, **_: Any) -> ProviderResponse:
        return self._request("sports", {}, request_type="sports", allow_network=allow_network)

    def fetch_current_odds(self, *, allow_network: bool = False, **_: Any) -> ProviderResponse:
        params = self._common_params()
        return self._request(
            f"sports/{self.config.sport_key}/odds",
            params,
            request_type="current",
            allow_network=allow_network,
        )

    def fetch_historical_odds(
        self,
        as_of: str | datetime,
        *,
        allow_network: bool = False,
        allow_billable_request: bool = False,
        **_: Any,
    ) -> ProviderResponse:
        query = utc_iso(as_of)
        params = self._common_params()
        params["date"] = query
        return self._request(
            f"historical/sports/{self.config.sport_key}/odds",
            params,
            request_type="historical",
            query_timestamp_utc=query,
            allow_network=allow_network,
            allow_billable_request=allow_billable_request,
        )

    def _common_params(self) -> dict[str, str]:
        params = {
            "regions": ",".join(self.config.regions),
            "markets": ",".join(self.config.markets),
            "oddsFormat": self.config.odds_format,
            "dateFormat": self.config.date_format,
        }
        if self.config.requested_bookmakers:
            params["bookmakers"] = ",".join(self.config.requested_bookmakers)
        return params

    def validate_required_fields(self, payload: Any, *, request_type: str) -> tuple[str, ...]:
        errors: list[str] = []
        events: Any
        if request_type == "historical":
            if not isinstance(payload, dict):
                return ("historical response must be an object",)
            if "timestamp" not in payload:
                errors.append("historical response is missing timestamp")
            events = payload.get("data")
            if not isinstance(events, list):
                errors.append("historical response data must be a list")
                return tuple(errors)
        elif request_type == "current":
            events = payload
            if not isinstance(events, list):
                return ("current response must be a list",)
        else:
            return (f"unsupported normalization request type: {request_type}",)
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                errors.append(f"event[{index}] must be an object")
                continue
            for field in ("id", "sport_key", "commence_time", "home_team", "away_team", "bookmakers"):
                if field not in event:
                    errors.append(f"event[{index}] is missing {field}")
        return tuple(errors)

    def normalize(self, payload: Any, *, request_type: str, source_payload_hash: str | None = None) -> NormalizationResult:
        structural_errors = self.validate_required_fields(payload, request_type=request_type)
        if structural_errors:
            raise ProviderSchemaError("; ".join(structural_errors))
        wrapper_timestamp = payload.get("timestamp") if request_type == "historical" else None
        events = payload["data"] if request_type == "historical" else payload
        snapshots: list[NormalizedSnapshot] = []
        errors: list[dict[str, Any]] = []
        for event_index, event in enumerate(events):
            scheduled_original = str(event["commence_time"])
            try:
                scheduled = utc_iso(scheduled_original)
            except ValueError as exc:
                errors.append({"event_index": event_index, "status": "rejected_invalid_timestamp", "reason": str(exc)})
                continue
            event_id = str(event["id"])
            raw_event_name = f"{event['home_team']} vs {event['away_team']}"
            provider_event_status = str(event.get("status") or "").strip().casefold()
            for book_index, bookmaker in enumerate(event.get("bookmakers", [])):
                if not isinstance(bookmaker, dict):
                    errors.append({"event_id": event_id, "bookmaker_index": book_index, "status": "rejected_schema", "reason": "bookmaker must be an object"})
                    continue
                provider_book_key = str(bookmaker.get("key", "")).strip()
                resolution = resolve_provider_source(
                    provider_book_key,
                    approved_overrides=self.config.approved_sportsbook_overrides,
                )
                observed_original = bookmaker.get("last_update") or wrapper_timestamp
                if not observed_original:
                    errors.append({"event_id": event_id, "sportsbook": provider_book_key, "status": "rejected_missing_observation_timestamp", "reason": "bookmaker last_update and historical timestamp are absent"})
                    continue
                try:
                    observed = utc_iso(str(observed_original))
                except ValueError as exc:
                    errors.append({"event_id": event_id, "sportsbook": provider_book_key, "status": "rejected_invalid_timestamp", "reason": str(exc)})
                    continue
                prefight_status = "pre_fight" if parse_utc(observed) < parse_utc(scheduled) else "live_or_post_start"
                markets = bookmaker.get("markets")
                if not isinstance(markets, list):
                    errors.append({"event_id": event_id, "sportsbook": provider_book_key, "status": "rejected_schema", "reason": "markets must be a list"})
                    continue
                for market_index, market in enumerate(markets):
                    if not isinstance(market, dict):
                        errors.append({"event_id": event_id, "sportsbook": provider_book_key, "market_index": market_index, "status": "rejected_schema", "reason": "market must be an object"})
                        continue
                    market_type = str(market.get("key", ""))
                    market_observed_original = market.get("last_update") or observed_original
                    try:
                        market_observed = utc_iso(str(market_observed_original))
                    except ValueError as exc:
                        errors.append({"event_id": event_id, "sportsbook": provider_book_key, "market_index": market_index, "status": "rejected_invalid_timestamp", "reason": str(exc)})
                        continue
                    market_prefight = "pre_fight" if parse_utc(market_observed) < parse_utc(scheduled) else "live_or_post_start"
                    outcomes_data = market.get("outcomes")
                    if not isinstance(outcomes_data, list):
                        errors.append({"event_id": event_id, "sportsbook": provider_book_key, "market_index": market_index, "status": "rejected_schema", "reason": "outcomes must be a list"})
                        continue
                    outcomes: list[NormalizedOutcome] = []
                    for outcome_index, raw_outcome in enumerate(outcomes_data):
                        if not isinstance(raw_outcome, dict) or "name" not in raw_outcome or "price" not in raw_outcome:
                            errors.append({"event_id": event_id, "sportsbook": provider_book_key, "market_index": market_index, "outcome_index": outcome_index, "status": "rejected_schema", "reason": "outcome requires name and price"})
                            continue
                        raw_price = raw_outcome["price"]
                        try:
                            decimal = to_decimal_odds(raw_price, self.config.odds_format)
                            american = float(raw_price) if self.config.odds_format == "american" else None
                            probability = implied_probability(decimal)
                            validation_status = "valid"
                            validation_reason = None
                        except InvalidOddsError as exc:
                            decimal = None
                            american = None
                            probability = None
                            validation_status = "rejected_invalid_odds"
                            validation_reason = str(exc)
                        opponents = [str(item.get("name")) for item in outcomes_data if isinstance(item, dict) and item.get("name") != raw_outcome.get("name")]
                        outcomes.append(NormalizedOutcome(
                            raw_outcome_name=str(raw_outcome["name"]),
                            raw_opponent_name=opponents[0] if len(opponents) == 1 else None,
                            decimal_odds=decimal,
                            american_odds=american,
                            raw_implied_probability=probability,
                            original_odds=raw_price,
                            original_odds_format=self.config.odds_format,
                            validation_status=validation_status,
                            validation_reason=validation_reason,
                        ))
                    provider_market_id = str(market.get("id") or f"{event_id}:{provider_book_key}:{market_type}")
                    rejection_reason: str | None = None
                    normalization_status = "normalized"
                    if market_type != "h2h":
                        normalization_status, rejection_reason = "rejected_unsupported_market", "only head-to-head moneyline markets are supported"
                    elif provider_event_status in {"cancelled", "canceled"}:
                        normalization_status, rejection_reason = "rejected_cancelled_bout", "provider marks the bout cancelled"
                    elif resolution.venue_type != VenueType.SPORTSBOOK:
                        normalization_status, rejection_reason = f"rejected_{resolution.venue_type.value}", resolution.reason
                    elif not resolution.approved:
                        normalization_status, rejection_reason = "rejected_unapproved_sportsbook", resolution.reason
                    elif market_prefight != "pre_fight":
                        normalization_status, rejection_reason = "rejected_live_market", "observation is at or after scheduled start"
                    elif len(outcomes) != 2:
                        normalization_status, rejection_reason = "rejected_incomplete_market", "market must contain exactly two outcomes"
                    elif any(item.validation_status != "valid" for item in outcomes):
                        normalization_status, rejection_reason = "rejected_invalid_odds", "one or more outcomes has invalid odds"
                    snapshots.append(NormalizedSnapshot(
                        provider_name=self.provider_name,
                        provider_event_id=event_id,
                        provider_market_id=provider_market_id,
                        sportsbook_key=resolution.canonical_key,
                        provider_sportsbook_key=provider_book_key,
                        sportsbook_display_name=str(bookmaker.get("title") or provider_book_key),
                        venue_type=resolution.venue_type.value,
                        market_type=market_type,
                        observed_at_utc=market_observed,
                        observed_at_original=str(market_observed_original),
                        scheduled_event_time_utc=scheduled,
                        scheduled_event_time_original=scheduled_original,
                        pre_fight_or_live_status=market_prefight,
                        raw_event_name=raw_event_name,
                        sport_key=str(event["sport_key"]),
                        league_key=str(event.get("sport_title")) if event.get("sport_title") else None,
                        outcomes=tuple(outcomes),
                        normalization_status=normalization_status,
                        rejection_reason=rejection_reason,
                        matching_diagnostics={"provider_event_status": provider_event_status} if provider_event_status else {},
                        source_payload_hash=source_payload_hash,
                    ))
        metadata = {
            "request_type": request_type,
            "historical_query_timestamp": utc_iso(wrapper_timestamp) if wrapper_timestamp else None,
            "event_count": len(events),
        }
        return NormalizationResult(self.provider_name, tuple(snapshots), tuple(errors), metadata, {})
