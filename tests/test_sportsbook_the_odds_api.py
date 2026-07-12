import json
import os
import urllib.error
import urllib.parse
from pathlib import Path

import pytest

from ufc_pipeline.sportsbook.config import TheOddsApiConfig
from ufc_pipeline.sportsbook.providers.base import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderSchemaError,
    ProviderTransientError,
)
from ufc_pipeline.sportsbook.providers.the_odds_api import TheOddsApiAdapter


FIXTURES = Path(__file__).parent / "fixtures" / "sportsbook"


def load_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_historical_fixture_normalization_preserves_timestamps_and_rejections():
    result = TheOddsApiAdapter().normalize(
        load_fixture("the_odds_api_historical.json"),
        request_type="historical",
        source_payload_hash="abc",
    )
    assert result.provider_metadata["historical_query_timestamp"] == "2024-04-12T02:00:00Z"
    assert len(result.snapshots) == 5
    draftkings = next(item for item in result.snapshots if item.provider_sportsbook_key == "draftkings")
    assert draftkings.observed_at_utc == "2024-04-12T01:58:00Z"
    assert draftkings.scheduled_event_time_utc == "2024-04-14T02:00:00Z"
    assert draftkings.pre_fight_or_live_status == "pre_fight"
    assert draftkings.normalization_status == "normalized"
    assert len(draftkings.outcomes) == 2
    assert all(outcome.validation_status == "valid" for outcome in draftkings.outcomes)
    prediction_market = next(item for item in result.snapshots if item.provider_sportsbook_key == "polymarket")
    assert prediction_market.normalization_status == "rejected_prediction_market"
    assert prediction_market.sportsbook_key is None


def test_current_fixture_normalization():
    result = TheOddsApiAdapter().normalize(load_fixture("the_odds_api_current.json"), request_type="current")
    assert len(result.snapshots) == 1
    assert result.snapshots[0].sportsbook_key == "caesars"
    assert result.snapshots[0].observed_at_utc == "2026-07-31T23:00:00Z"


@pytest.mark.parametrize(
    ("payload", "request_type"),
    [
        ({"data": []}, "historical"),
        ({"timestamp": "2024-01-01T00:00:00Z"}, "historical"),
        ({}, "current"),
        ([{"id": "x"}], "current"),
    ],
)
def test_missing_required_fields_and_unexpected_schema_fail(payload, request_type):
    with pytest.raises(ProviderSchemaError):
        TheOddsApiAdapter().normalize(payload, request_type=request_type)


def test_empty_provider_responses_are_valid_empty_collections():
    assert TheOddsApiAdapter().normalize([], request_type="current").snapshots == ()
    result = TheOddsApiAdapter().normalize({"timestamp": "2024-01-01T00:00:00Z", "data": []}, request_type="historical")
    assert result.snapshots == ()


def test_network_is_disabled_by_default_and_checked_before_api_key(monkeypatch):
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    adapter = TheOddsApiAdapter()
    with pytest.raises(ProviderError, match="network access is disabled"):
        adapter.fetch_current_odds()
    with pytest.raises(ProviderError, match="network access is disabled"):
        adapter.fetch_historical_odds("2024-01-01T00:00:00Z")


def test_billable_historical_gate_is_required_before_api_key(monkeypatch):
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="billable"):
        TheOddsApiAdapter().fetch_historical_odds("2024-01-01T00:00:00Z", allow_network=True)


def test_missing_api_key_is_clear_but_never_printed(monkeypatch):
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    with pytest.raises(ProviderAuthenticationError, match="THE_ODDS_API_KEY"):
        TheOddsApiAdapter().fetch_current_odds(allow_network=True)


def test_successful_mock_transport_receives_expected_historical_query_and_hides_key(monkeypatch):
    secret = "super-secret-key"
    monkeypatch.setenv("THE_ODDS_API_KEY", secret)
    calls = []

    def transport(request, timeout):
        calls.append((request.full_url, timeout))
        body = json.dumps({"timestamp": "2024-04-12T02:00:00Z", "data": []}).encode()
        return 200, {"x-requests-remaining": "9", "x-requests-used": "1"}, body

    adapter = TheOddsApiAdapter(transport=transport)
    response = adapter.fetch_historical_odds(
        "2024-04-12T02:00:00-00:00",
        allow_network=True,
        allow_billable_request=True,
    )
    parsed = urllib.parse.urlparse(calls[0][0])
    query = urllib.parse.parse_qs(parsed.query)
    assert query["date"] == ["2024-04-12T02:00:00Z"]
    assert query["apiKey"] == [secret]
    assert response.query_timestamp_utc == "2024-04-12T02:00:00Z"
    assert secret not in repr(response)
    assert adapter.quota_metadata(response.headers) == {"x-requests-remaining": "9", "x-requests-used": "1"}


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(401, ProviderAuthenticationError), (429, ProviderRateLimitError), (400, ProviderError)],
)
def test_provider_http_errors_are_typed(monkeypatch, status, error_type):
    monkeypatch.setenv("THE_ODDS_API_KEY", "secret")

    def transport(request, timeout):
        return status, {}, b"{}"

    with pytest.raises(error_type):
        TheOddsApiAdapter(transport=transport).fetch_current_odds(allow_network=True)


def test_transient_provider_errors_retry_only_to_configured_bound(monkeypatch):
    monkeypatch.setenv("THE_ODDS_API_KEY", "secret")
    calls = []
    sleeps = []

    def transport(request, timeout):
        calls.append(1)
        return 503, {}, b"{}"

    adapter = TheOddsApiAdapter(
        TheOddsApiConfig(retry_count=2, backoff_seconds=0.01),
        transport=transport,
        sleep=sleeps.append,
    )
    with pytest.raises(ProviderTransientError, match="bounded retries"):
        adapter.fetch_current_odds(allow_network=True)
    assert len(calls) == 3
    assert sleeps == [0.01, 0.02]


def test_invalid_json_is_schema_error(monkeypatch):
    monkeypatch.setenv("THE_ODDS_API_KEY", "secret")

    def transport(request, timeout):
        return 200, {}, b"not-json"

    with pytest.raises(ProviderSchemaError):
        TheOddsApiAdapter(transport=transport).fetch_current_odds(allow_network=True)


def test_market_level_edge_cases_are_stored_with_explicit_rejections():
    payload = load_fixture("the_odds_api_current.json")
    market = payload[0]["bookmakers"][0]["markets"][0]
    market["key"] = "totals"
    result = TheOddsApiAdapter().normalize(payload, request_type="current")
    assert result.snapshots[0].normalization_status == "rejected_unsupported_market"

    payload = load_fixture("the_odds_api_current.json")
    payload[0]["bookmakers"][0]["markets"][0]["outcomes"] = [{"name": "Only One", "price": 2.0}]
    result = TheOddsApiAdapter().normalize(payload, request_type="current")
    assert result.snapshots[0].normalization_status == "rejected_incomplete_market"

    payload = load_fixture("the_odds_api_current.json")
    payload[0]["bookmakers"][0]["last_update"] = payload[0]["commence_time"]
    result = TheOddsApiAdapter().normalize(payload, request_type="current")
    assert result.snapshots[0].normalization_status == "rejected_live_market"
