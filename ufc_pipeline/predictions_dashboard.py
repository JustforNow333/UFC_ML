"""Read-only upcoming-predictions dashboard service.

This module deliberately has no dependency on the modeling or prediction
pipeline.  It reads probabilities that Step 6B already froze in the production
ledger, validates them, and exposes a small JSON/static HTTP surface.  It never
writes the ledger and never invokes inference, training, or calibration code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEDGER_PATH = REPO_ROOT / "data" / "live" / "live_predictions.csv"
DEFAULT_FEATURE_ROOT = REPO_ROOT / "data" / "live" / "features"
DEFAULT_STATIC_DIR = REPO_ROOT / "web" / "predictions"
PROBABILITY_TOLERANCE = 1e-6

REQUIRED_LEDGER_COLUMNS = frozenset({
    "prediction_id",
    "prediction_batch_id",
    "prediction_mode",
    "event_date",
    "event_name",
    "fighter_a",
    "fighter_b",
    "model_version",
    "calibration_version",
    "input_source",
    "predicted_probability_a",
    "predicted_probability_b",
    "prediction_timestamp_utc",
    "status",
})
RESOLUTION_COLUMNS = (
    "target_a_win",
    "winner",
    "result_source",
    "resolved_timestamp_utc",
    "log_loss",
    "brier",
    "correct_prediction",
)
STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/assets/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class LedgerUnavailableError(RuntimeError):
    """The configured production ledger cannot be read."""


class LedgerSchemaError(RuntimeError):
    """The configured ledger does not satisfy the display contract."""


@dataclass(frozen=True)
class DashboardConfig:
    ledger_path: Path = DEFAULT_LEDGER_PATH
    feature_root: Path = DEFAULT_FEATURE_ROOT
    static_dir: Path = DEFAULT_STATIC_DIR
    repo_root: Path = REPO_ROOT
    today: date | None = None


def confidence_label(probability: float) -> str:
    """Return the presentation label for the higher stored probability."""
    if probability >= 0.80:
        return "Very high confidence"
    if probability >= 0.70:
        return "High confidence"
    if probability >= 0.62:
        return "Moderate confidence"
    if probability >= 0.55:
        return "Slight lean"
    return "Toss-up"


def _blank(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().casefold() in {"", "nan", "none", "null"}


def _text(value: Any) -> str:
    return "" if _blank(value) else re.sub(r"\s+", " ", str(value).strip())


def _name_key(value: Any) -> str:
    return _text(value).casefold()


def _fight_key(row: dict[str, Any]) -> tuple[str, str, frozenset[str]]:
    return (
        _text(row.get("event_date")),
        _name_key(row.get("event_name")),
        frozenset({_name_key(row.get("fighter_a")), _name_key(row.get("fighter_b"))}),
    )


def _safe_event_id(event_date: str, event_name: str, batch_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", event_name.casefold()).strip("-") or "ufc-event"
    digest = hashlib.sha256(f"{event_date}|{event_name}|{batch_id}".encode()).hexdigest()[:8]
    return f"{slug}-{event_date}-{digest}"


def _parse_probability(value: Any) -> float:
    probability = float(value)
    if not math.isfinite(probability):
        raise ValueError("probability is not finite")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability is outside [0, 1]")
    return probability


def _fight_sort_key(row: dict[str, Any], fighter_a: str, fighter_b: str) -> tuple[Any, ...]:
    """Prefer an explicit numeric bout order, otherwise use stable text/ID order."""
    for column in ("bout_order", "fight_order"):
        value = _text(row.get(column))
        if value:
            try:
                return (0, float(value), _name_key(fighter_a), _name_key(fighter_b))
            except ValueError:
                break
    return (1, _name_key(fighter_a), _name_key(fighter_b), _text(row.get("prediction_id")))


def _is_official_frozen(row: dict[str, Any]) -> bool:
    batch_id = _text(row.get("prediction_batch_id")).casefold()
    return (
        _text(row.get("prediction_mode")).casefold() == "live_forward"
        and "official" in batch_id
        and "frozen" in batch_id
        and bool(_text(row.get("model_version")))
        and bool(_text(row.get("calibration_version")))
    )


def _is_unresolved(row: dict[str, Any]) -> bool:
    return (
        _text(row.get("status")).casefold() == "pending"
        and all(_blank(row.get(column)) for column in RESOLUTION_COLUMNS)
    )


class _WeightClassLookup:
    """Safely read optional weight classes from ledger-referenced feature CSVs."""

    def __init__(self, repo_root: Path, feature_root: Path):
        self.repo_root = repo_root.resolve()
        self.feature_root = feature_root.resolve()
        self._cache: dict[Path, dict[tuple[str, str, frozenset[str]], str]] = {}

    def get(self, row: dict[str, Any]) -> str | None:
        direct = _text(row.get("weight_class"))
        if direct:
            return direct
        source_text = _text(row.get("input_source"))
        if not source_text:
            return None
        source = Path(source_text)
        if not source.is_absolute():
            source = self.repo_root / source
        try:
            source = source.resolve(strict=True)
            source.relative_to(self.feature_root)
        except (FileNotFoundError, OSError, ValueError):
            return None
        if source not in self._cache:
            self._cache[source] = self._load(source)
        return self._cache[source].get(_fight_key(row))

    @staticmethod
    def _load(path: Path) -> dict[tuple[str, str, frozenset[str]], str]:
        result: dict[tuple[str, str, frozenset[str]], str] = {}
        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                required = {"event_date", "event_name", "fighter_a", "fighter_b", "weight_class"}
                if not reader.fieldnames or not required.issubset(reader.fieldnames):
                    return result
                for row in reader:
                    weight_class = _text(row.get("weight_class"))
                    if weight_class:
                        result.setdefault(_fight_key(row), weight_class)
        except (OSError, csv.Error):
            return {}
        return result


def _read_ledger(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists() or not path.is_file():
        raise LedgerUnavailableError("Upcoming predictions ledger is unavailable.")
    try:
        if path.stat().st_size == 0:
            return [], []
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            if not columns:
                return [], []
            missing = sorted(REQUIRED_LEDGER_COLUMNS - set(columns))
            if missing:
                raise LedgerSchemaError(f"Prediction ledger is missing required columns: {missing}")
            return list(reader), columns
    except LedgerSchemaError:
        raise
    except (OSError, csv.Error) as exc:
        raise LedgerUnavailableError("Upcoming predictions ledger is unavailable.") from exc


def load_upcoming_predictions(config: DashboardConfig = DashboardConfig()) -> dict[str, Any]:
    """Validate and group displayable unresolved predictions without mutations."""
    rows, columns = _read_ledger(config.ledger_path)
    if not columns:
        return {"events": [], "diagnostics": {"invalid_row_count": 0}}

    lookup = _WeightClassLookup(config.repo_root, config.feature_root)
    current_date = config.today or date.today()
    invalid_count = 0
    excluded_count = 0
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}

    for row_number, row in enumerate(rows, start=2):
        if not _is_unresolved(row):
            excluded_count += 1
            continue
        if not _is_official_frozen(row):
            excluded_count += 1
            continue
        try:
            event_date = date.fromisoformat(_text(row.get("event_date")))
            event_name = _text(row.get("event_name"))
            fighter_a = _text(row.get("fighter_a"))
            fighter_b = _text(row.get("fighter_b"))
            if not event_name or not fighter_a or not fighter_b:
                raise ValueError("event and both fighter names are required")
            if _name_key(fighter_a) == _name_key(fighter_b):
                raise ValueError("fighter names must be distinct")
            if event_date < current_date:
                excluded_count += 1
                continue
            probability_a = _parse_probability(row.get("predicted_probability_a"))
            probability_b = _parse_probability(row.get("predicted_probability_b"))
            if not math.isclose(probability_a + probability_b, 1.0, abs_tol=PROBABILITY_TOLERANCE):
                raise ValueError("probabilities are not complementary")

            if probability_a == probability_b:
                winner = None
                winner_side = "tie"
            elif probability_a > probability_b:
                winner = fighter_a
                winner_side = "a"
            else:
                winner = fighter_b
                winner_side = "b"

            stored_winner = _text(row.get("predicted_winner"))
            if stored_winner and (winner is None or _name_key(stored_winner) != _name_key(winner)):
                raise ValueError("stored predicted winner disagrees with probabilities")

            batch_id = _text(row.get("prediction_batch_id"))
            model_version = _text(row.get("model_version"))
            calibration_version = _text(row.get("calibration_version"))
            group_key = (
                event_date.isoformat(), event_name, batch_id, model_version, calibration_version,
            )
            weight_class = lookup.get(row)
            fight = {
                "fight_id": _text(row.get("prediction_id")),
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "fighter_a_probability": probability_a,
                "fighter_b_probability": probability_b,
                "predicted_winner": winner,
                "predicted_winner_side": winner_side,
                "predicted_winner_probability": max(probability_a, probability_b),
                "confidence_label": confidence_label(max(probability_a, probability_b)),
                "weight_class": weight_class,
                "bout_label": f"{weight_class} Bout" if weight_class else "UFC Bout",
                "prediction_status": "frozen",
                "prediction_created_at": _text(row.get("prediction_timestamp_utc")) or None,
                "_sort_key": _fight_sort_key(row, fighter_a, fighter_b),
            }
            grouped.setdefault(group_key, []).append(fight)
        except (TypeError, ValueError) as exc:
            invalid_count += 1
            LOGGER.warning(
                "Skipping invalid prediction row %s (%s): %s",
                row_number,
                _text(row.get("prediction_id")) or "unknown prediction",
                exc,
            )

    events = []
    for group_key, fights in grouped.items():
        event_date, event_name, batch_id, model_version, calibration_version = group_key
        fights.sort(key=lambda fight: fight.pop("_sort_key"))
        timestamps = [fight["prediction_created_at"] for fight in fights if fight["prediction_created_at"]]
        events.append({
            "event_id": _safe_event_id(event_date, event_name, batch_id),
            "event_name": event_name,
            "event_date": event_date,
            "batch_id": batch_id,
            "prediction_status": "frozen",
            "model_version": model_version,
            "calibration_version": calibration_version,
            "prediction_created_at": min(timestamps) if timestamps else None,
            "fight_count": len(fights),
            "fights": fights,
        })
    events.sort(key=lambda event: (event["event_date"], event["event_name"].casefold(), event["batch_id"]))
    return {
        "events": events,
        "diagnostics": {
            "invalid_row_count": invalid_count,
            "excluded_row_count": excluded_count,
        },
    }


def create_dashboard_handler(config: DashboardConfig):
    """Create an explicit-route HTTP handler bound to a dashboard config."""

    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "UFC-Predictions-Dashboard/1.0"

        def _security_headers(self, cache_control: str) -> None:
            self.send_header("Cache-Control", cache_control)
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._security_headers("no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_static(self, relative_path: str, content_type: str) -> None:
            path = config.static_dir / relative_path
            try:
                body = path.read_bytes()
            except OSError:
                self._send_json(404, {"error": {"code": "not_found", "message": "Resource not found."}})
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self._security_headers("no-cache" if relative_path == "index.html" else "public, max-age=300")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _handle_get(self) -> None:
            path = urlsplit(self.path).path
            if path == "/api/predictions/upcoming":
                try:
                    self._send_json(200, load_upcoming_predictions(config))
                except LedgerUnavailableError:
                    self._send_json(503, {
                        "events": [],
                        "error": {"code": "ledger_unavailable", "message": "Upcoming predictions could not be loaded."},
                    })
                except LedgerSchemaError:
                    LOGGER.exception("Prediction ledger schema validation failed")
                    self._send_json(500, {
                        "events": [],
                        "error": {"code": "ledger_invalid", "message": "Upcoming predictions could not be loaded."},
                    })
                return
            static = STATIC_ROUTES.get(path)
            if static:
                self._send_static(*static)
                return
            self._send_json(404, {"error": {"code": "not_found", "message": "Resource not found."}})

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            self._handle_get()

        def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
            self._handle_get()

        def do_POST(self) -> None:  # noqa: N802 - explicit read-only response
            self._send_json(405, {"error": {"code": "method_not_allowed", "message": "This dashboard is read-only."}})

        def do_PUT(self) -> None:  # noqa: N802 - explicit read-only response
            self.do_POST()

        def do_PATCH(self) -> None:  # noqa: N802 - explicit read-only response
            self.do_POST()

        def do_DELETE(self) -> None:  # noqa: N802 - explicit read-only response
            self.do_POST()

        def log_message(self, format: str, *args: Any) -> None:
            LOGGER.info("dashboard request: " + format, *args)

    return DashboardHandler


def serve_dashboard(config: DashboardConfig, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the dashboard until interrupted; defaults to loopback only."""
    server = ThreadingHTTPServer((host, port), create_dashboard_handler(config))
    LOGGER.info("Predictions dashboard listening on http://%s:%s", host, port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
