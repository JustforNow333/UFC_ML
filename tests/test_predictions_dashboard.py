"""Read-only API and static frontend tests for the predictions dashboard."""

from __future__ import annotations

import csv
import hashlib
import json
import threading
from contextlib import contextmanager
from datetime import date
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from ufc_pipeline.predictions_dashboard import (
    DEFAULT_LEDGER_PATH,
    DEFAULT_STATIC_DIR,
    DashboardConfig,
    LedgerUnavailableError,
    confidence_label,
    create_dashboard_handler,
    load_upcoming_predictions,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_COLUMNS = [
    "prediction_id", "prediction_batch_id", "prediction_mode", "event_date", "event_name",
    "fighter_a", "fighter_b", "model_version", "calibration_version", "feature_schema_version",
    "input_source", "predicted_probability_a", "predicted_probability_b", "prediction_timestamp_utc",
    "status", "target_a_win", "winner", "result_source", "resolved_timestamp_utc", "log_loss",
    "brier", "correct_prediction", "notes", "predicted_winner",
]


def prediction_row(
    prediction_id="p1",
    event_date="2026-08-01",
    event_name="UFC Test One",
    fighter_a="Alpha Fighter",
    fighter_b="Beta Fighter",
    probability_a=0.63,
    probability_b=0.37,
    status="pending",
    batch="ufc_test_official_frozen_batch",
    **overrides,
):
    row = {column: "" for column in LEDGER_COLUMNS}
    row.update({
        "prediction_id": prediction_id,
        "prediction_batch_id": batch,
        "prediction_mode": "live_forward",
        "event_date": event_date,
        "event_name": event_name,
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "model_version": "official-model-v1",
        "calibration_version": "platt-official-v1",
        "feature_schema_version": "step3c-test",
        "input_source": "",
        "predicted_probability_a": probability_a,
        "predicted_probability_b": probability_b,
        "prediction_timestamp_utc": "2026-07-01T12:00:00+00:00",
        "status": status,
    })
    row.update(overrides)
    return row


def write_ledger(path: Path, rows: list[dict], columns=LEDGER_COLUMNS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def dashboard_config(tmp_path: Path, ledger: Path, today=date(2026, 7, 1)) -> DashboardConfig:
    return DashboardConfig(
        ledger_path=ledger,
        feature_root=tmp_path / "data" / "live" / "features",
        static_dir=DEFAULT_STATIC_DIR,
        repo_root=tmp_path,
        today=today,
    )


@contextmanager
def running_server(config: DashboardConfig):
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_dashboard_handler(config))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def get_json(url: str) -> tuple[int, dict]:
    with urlopen(url, timeout=3) as response:
        return response.status, json.loads(response.read())


def test_upcoming_prediction_maps_probabilities_winner_and_metadata(tmp_path):
    ledger = tmp_path / "ledger.csv"
    write_ledger(ledger, [prediction_row()])
    payload = load_upcoming_predictions(dashboard_config(tmp_path, ledger))

    assert len(payload["events"]) == 1
    event = payload["events"][0]
    assert event["event_name"] == "UFC Test One"
    assert event["event_date"] == "2026-08-01"
    assert event["batch_id"] == "ufc_test_official_frozen_batch"
    assert event["prediction_status"] == "frozen"
    fight = event["fights"][0]
    assert fight["fighter_a_probability"] == pytest.approx(0.63)
    assert fight["fighter_b_probability"] == pytest.approx(0.37)
    assert fight["predicted_winner"] == "Alpha Fighter"
    assert fight["predicted_winner_side"] == "a"
    assert fight["confidence_label"] == "Moderate confidence"


def test_resolved_scored_and_past_fights_are_excluded(tmp_path):
    ledger = tmp_path / "ledger.csv"
    rows = [
        prediction_row("resolved", status="resolved", winner="Alpha Fighter", target_a_win="1"),
        prediction_row("scored", log_loss="0.4"),
        prediction_row("past", event_date="2026-06-01"),
        prediction_row("pending"),
    ]
    write_ledger(ledger, rows)
    payload = load_upcoming_predictions(dashboard_config(tmp_path, ledger))
    assert [fight["fight_id"] for fight in payload["events"][0]["fights"]] == ["pending"]
    assert payload["diagnostics"]["excluded_row_count"] == 3


def test_multiple_events_group_and_sort_chronologically(tmp_path):
    ledger = tmp_path / "ledger.csv"
    rows = [
        prediction_row("late", event_date="2026-09-01", event_name="UFC Later"),
        prediction_row("early-b", event_date="2026-08-01", event_name="UFC Early", fighter_a="Zulu", fighter_b="Yankee"),
        prediction_row("early-a", event_date="2026-08-01", event_name="UFC Early", fighter_a="Alpha", fighter_b="Bravo"),
    ]
    write_ledger(ledger, rows)
    events = load_upcoming_predictions(dashboard_config(tmp_path, ledger))["events"]
    assert [event["event_name"] for event in events] == ["UFC Early", "UFC Later"]
    assert [fight["fight_id"] for fight in events[0]["fights"]] == ["early-a", "early-b"]


def test_explicit_bout_order_is_preserved_when_available(tmp_path):
    ledger = tmp_path / "ledger.csv"
    columns = LEDGER_COLUMNS + ["bout_order"]
    write_ledger(ledger, [
        prediction_row("second", fighter_a="Alpha", fighter_b="Bravo", bout_order="2"),
        prediction_row("first", fighter_a="Zulu", fighter_b="Yankee", bout_order="1"),
    ], columns=columns)
    fights = load_upcoming_predictions(dashboard_config(tmp_path, ledger))["events"][0]["fights"]
    assert [fight["fight_id"] for fight in fights] == ["first", "second"]


@pytest.mark.parametrize(("probability", "expected"), [
    (0.500, "Toss-up"),
    (0.549, "Toss-up"),
    (0.550, "Slight lean"),
    (0.619, "Slight lean"),
    (0.620, "Moderate confidence"),
    (0.699, "Moderate confidence"),
    (0.700, "High confidence"),
    (0.799, "High confidence"),
    (0.800, "Very high confidence"),
])
def test_confidence_boundaries(probability, expected):
    assert confidence_label(probability) == expected


@pytest.mark.parametrize(("probability_a", "probability_b"), [
    ("nan", 0.4),
    ("inf", 0.4),
    (-0.1, 1.1),
    (1.1, -0.1),
    (0.7, 0.4),
])
def test_invalid_probabilities_are_not_displayed(tmp_path, probability_a, probability_b, caplog):
    ledger = tmp_path / "ledger.csv"
    write_ledger(ledger, [prediction_row(probability_a=probability_a, probability_b=probability_b)])
    payload = load_upcoming_predictions(dashboard_config(tmp_path, ledger))
    assert payload["events"] == []
    assert payload["diagnostics"]["invalid_row_count"] == 1
    assert "Skipping invalid prediction row" in caplog.text


def test_missing_names_and_inconsistent_stored_winner_are_rejected(tmp_path):
    ledger = tmp_path / "ledger.csv"
    write_ledger(ledger, [
        prediction_row("missing", fighter_a=""),
        prediction_row("wrong-winner", predicted_winner="Beta Fighter"),
    ])
    payload = load_upcoming_predictions(dashboard_config(tmp_path, ledger))
    assert payload["events"] == []
    assert payload["diagnostics"]["invalid_row_count"] == 2


def test_exact_tie_is_returned_without_inventing_a_winner(tmp_path):
    ledger = tmp_path / "ledger.csv"
    write_ledger(ledger, [prediction_row(probability_a=0.5, probability_b=0.5)])
    fight = load_upcoming_predictions(dashboard_config(tmp_path, ledger))["events"][0]["fights"][0]
    assert fight["predicted_winner"] is None
    assert fight["predicted_winner_side"] == "tie"
    assert fight["confidence_label"] == "Toss-up"


def test_only_official_frozen_batches_are_displayed(tmp_path):
    ledger = tmp_path / "ledger.csv"
    write_ledger(ledger, [
        prediction_row("official"),
        prediction_row("not-frozen", batch="ufc_test_official_batch"),
        prediction_row("not-official", batch="ufc_test_frozen_batch"),
        prediction_row("wrong-mode", prediction_mode="historical_replay"),
    ])
    payload = load_upcoming_predictions(dashboard_config(tmp_path, ledger))
    assert [fight["fight_id"] for fight in payload["events"][0]["fights"]] == ["official"]


def test_weight_class_is_safely_enriched_from_feature_source(tmp_path):
    feature_root = tmp_path / "data" / "live" / "features"
    features = feature_root / "card.csv"
    features.parent.mkdir(parents=True)
    with features.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_date", "event_name", "fighter_a", "fighter_b", "weight_class"])
        writer.writeheader()
        writer.writerow({
            "event_date": "2026-08-01", "event_name": "UFC Test One",
            "fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter", "weight_class": "Lightweight",
        })
    ledger = tmp_path / "ledger.csv"
    write_ledger(ledger, [prediction_row(input_source="data/live/features/card.csv")])
    fight = load_upcoming_predictions(dashboard_config(tmp_path, ledger))["events"][0]["fights"][0]
    assert fight["weight_class"] == "Lightweight"
    assert fight["bout_label"] == "Lightweight Bout"


def test_feature_source_outside_allowed_root_is_not_read(tmp_path):
    outside = tmp_path / "private.csv"
    outside.write_text("event_date,event_name,fighter_a,fighter_b,weight_class\n2026-08-01,UFC Test One,Alpha Fighter,Beta Fighter,Secret\n")
    ledger = tmp_path / "ledger.csv"
    write_ledger(ledger, [prediction_row(input_source=str(outside))])
    fight = load_upcoming_predictions(dashboard_config(tmp_path, ledger))["events"][0]["fights"][0]
    assert fight["weight_class"] is None
    assert fight["bout_label"] == "UFC Bout"


def test_missing_and_empty_ledgers_have_controlled_behavior(tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(LedgerUnavailableError):
        load_upcoming_predictions(dashboard_config(tmp_path, missing))

    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert load_upcoming_predictions(dashboard_config(tmp_path, empty)) == {
        "events": [], "diagnostics": {"invalid_row_count": 0},
    }

    headers_only = tmp_path / "headers.csv"
    write_ledger(headers_only, [])
    assert load_upcoming_predictions(dashboard_config(tmp_path, headers_only))["events"] == []


def test_api_is_read_only_and_returns_valid_json(tmp_path):
    ledger = tmp_path / "ledger.csv"
    write_ledger(ledger, [prediction_row()])
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()
    with running_server(dashboard_config(tmp_path, ledger)) as base_url:
        status, payload = get_json(f"{base_url}/api/predictions/upcoming")
        assert status == 200
        assert payload["events"][0]["fight_count"] == 1
        request = Request(f"{base_url}/api/predictions/upcoming", method="POST")
        with pytest.raises(HTTPError) as exc:
            urlopen(request, timeout=3)
        assert exc.value.code == 405
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before


def test_missing_ledger_api_response_hides_internal_details(tmp_path):
    with running_server(dashboard_config(tmp_path, tmp_path / "secret" / "missing.csv")) as base_url:
        with pytest.raises(HTTPError) as exc:
            urlopen(f"{base_url}/api/predictions/upcoming", timeout=3)
        assert exc.value.code == 503
        payload = json.loads(exc.value.read())
        assert payload["events"] == []
        assert payload["error"]["message"] == "Upcoming predictions could not be loaded."
        assert "secret" not in json.dumps(payload)


def test_static_frontend_is_served_with_required_states_and_components(tmp_path):
    ledger = tmp_path / "ledger.csv"
    write_ledger(ledger, [prediction_row()])
    with running_server(dashboard_config(tmp_path, ledger)) as base_url:
        with urlopen(f"{base_url}/", timeout=3) as response:
            html = response.read().decode()
        with urlopen(f"{base_url}/assets/app.js", timeout=3) as response:
            javascript = response.read().decode()
        with urlopen(f"{base_url}/assets/styles.css", timeout=3) as response:
            css = response.read().decode()
    assert "Loading upcoming predictions…" in html
    assert "No upcoming official predictions are currently available." in javascript
    assert "Upcoming predictions could not be loaded." in javascript
    assert "Official frozen prediction" in javascript
    assert 'toFixed(1)' in javascript
    assert "predicted_winner_side" in javascript
    assert "barA.style.width" in javascript and "barB.style.width" in javascript
    assert 'payload.events.forEach' in javascript
    assert ".fighter-name" in css and "overflow-wrap" in css
    assert "@media (max-width: 440px)" in css


def test_long_fighter_names_are_preserved(tmp_path):
    long_name = "A Very Long Multi-Part Fighter Name That Must Wrap Cleanly"
    ledger = tmp_path / "ledger.csv"
    write_ledger(ledger, [prediction_row(fighter_a=long_name)])
    fight = load_upcoming_predictions(dashboard_config(tmp_path, ledger))["events"][0]["fights"][0]
    assert fight["fighter_a"] == long_name


@pytest.mark.skipif(not DEFAULT_LEDGER_PATH.exists(), reason="production ledger is not present")
def test_production_ufc329_rows_parse_without_mutation():
    before = hashlib.sha256(DEFAULT_LEDGER_PATH.read_bytes()).hexdigest()
    payload = load_upcoming_predictions(DashboardConfig(today=date(2026, 7, 11)))
    assert len(payload["events"]) == 1
    assert payload["events"][0]["fight_count"] == 8
    assert payload["diagnostics"]["invalid_row_count"] == 0
    assert all(fight["weight_class"] for fight in payload["events"][0]["fights"])
    assert hashlib.sha256(DEFAULT_LEDGER_PATH.read_bytes()).hexdigest() == before
