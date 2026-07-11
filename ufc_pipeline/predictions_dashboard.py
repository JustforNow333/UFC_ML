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
import unicodedata
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
DEFAULT_MANIFEST_ROOT = REPO_ROOT / "data" / "live" / "event_manifests"
DEFAULT_STATIC_DIR = REPO_ROOT / "web" / "predictions"
PROBABILITY_TOLERANCE = 1e-6

CARD_SECTION_ORDER = {
    "main_event": 0,
    "main_card": 1,
    "prelims": 2,
    "early_prelims": 3,
    "fight_card": 4,
}
CARD_SECTION_LABELS = {
    "main_event": "Main Event",
    "main_card": "Main Card",
    "prelims": "Prelims",
    "early_prelims": "Early Prelims",
    "fight_card": "Fight Card",
}
ACTIVE_FIGHT_STATUSES = frozenset({"confirmed"})

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
    manifest_root: Path = DEFAULT_MANIFEST_ROOT
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
    text = unicodedata.normalize("NFKD", _text(value))
    text = "".join(char for char in text if not unicodedata.combining(char)).casefold()
    text = re.sub(r"[.'`’]", "", text)
    text = re.sub(r"[-_/]", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _fight_key(row: dict[str, Any]) -> tuple[str, str, frozenset[str]]:
    return (
        _text(row.get("event_date")),
        _name_key(row.get("event_name")),
        frozenset({_name_key(row.get("fighter_a")), _name_key(row.get("fighter_b"))}),
    )


def _safe_event_id(event_date: str, event_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", event_name.casefold()).strip("-") or "ufc-event"
    digest = hashlib.sha256(f"{event_date}|{_name_key(event_name)}".encode()).hexdigest()[:8]
    return f"{slug}-{event_date}-{digest}"


def _parse_probability(value: Any) -> float:
    probability = float(value)
    if not math.isfinite(probability):
        raise ValueError("probability is not finite")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability is outside [0, 1]")
    return probability


def _fight_sort_key(
    row: dict[str, Any], fighter_a: str, fighter_b: str, manifest: dict[str, Any] | None = None,
) -> tuple[Any, ...]:
    """Prefer an explicit numeric bout order, otherwise use stable text/ID order."""
    if manifest:
        section = _text(manifest.get("card_section")).casefold()
        section_order = CARD_SECTION_ORDER.get(section, CARD_SECTION_ORDER["fight_card"])
        try:
            bout_order = int(_text(manifest.get("bout_order")))
        except ValueError:
            bout_order = 1_000_000
        return (0, section_order, bout_order, _name_key(fighter_a), _name_key(fighter_b))
    for column in ("bout_order", "fight_order"):
        value = _text(row.get(column))
        if value:
            try:
                return (1, float(value), _name_key(fighter_a), _name_key(fighter_b))
            except ValueError:
                break
    return (2, _name_key(fighter_a), _name_key(fighter_b), _text(row.get("prediction_id")))


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


class _EventManifestLookup:
    """Read optional card metadata without making the ledger mutable.

    Manifests are sidecars under one configured root. A matched inactive row is
    authoritative for display filtering, while a missing manifest or unmatched
    fight safely falls back to the legacy single-card presentation.
    """

    REQUIRED_COLUMNS = frozenset({
        "event_name", "event_date", "fighter_a", "fighter_b", "weight_class",
        "card_section", "bout_order", "fight_status",
    })

    def __init__(self, manifest_root: Path):
        self.manifest_root = manifest_root.resolve()
        self.rows: dict[tuple[str, str, frozenset[str]], dict[str, str]] = {}
        self._load()

    def get(self, row: dict[str, Any]) -> dict[str, str] | None:
        return self.rows.get(_fight_key(row))

    def _load(self) -> None:
        if not self.manifest_root.is_dir():
            return
        for path in sorted(self.manifest_root.glob("*.csv")):
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(self.manifest_root)
                with resolved.open(newline="", encoding="utf-8-sig") as handle:
                    reader = csv.DictReader(handle)
                    if not reader.fieldnames or not self.REQUIRED_COLUMNS.issubset(reader.fieldnames):
                        LOGGER.warning("Skipping card manifest with an invalid schema: %s", path.name)
                        continue
                    for row_number, row in enumerate(reader, start=2):
                        key = _fight_key(row)
                        section = _text(row.get("card_section")).casefold()
                        status = _text(row.get("fight_status")).casefold()
                        try:
                            date.fromisoformat(key[0])
                            bout_order = int(_text(row.get("bout_order")))
                        except (TypeError, ValueError):
                            LOGGER.warning("Skipping invalid manifest row %s:%s", path.name, row_number)
                            continue
                        if (
                            not key[1] or len(key[2]) != 2 or not status
                            or section not in CARD_SECTION_ORDER or section == "fight_card"
                            or bout_order < 1
                        ):
                            LOGGER.warning("Skipping invalid manifest row %s:%s", path.name, row_number)
                            continue
                        normalized = dict(row)
                        normalized["card_section"] = section
                        normalized["fight_status"] = status
                        normalized["bout_order"] = str(bout_order)
                        if key in self.rows:
                            LOGGER.warning("Ignoring duplicate manifest matchup in %s:%s", path.name, row_number)
                            continue
                        self.rows[key] = normalized
            except (OSError, csv.Error, ValueError):
                LOGGER.warning("Skipping unreadable card manifest: %s", path.name)


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
    manifests = _EventManifestLookup(config.manifest_root)
    current_date = config.today or date.today()
    invalid_count = 0
    excluded_count = 0
    grouped: dict[tuple[str, str], dict[str, Any]] = {}

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
            manifest = manifests.get(row)
            if manifest and _text(manifest.get("fight_status")).casefold() not in ACTIVE_FIGHT_STATUSES:
                excluded_count += 1
                continue
            weight_class = _text(manifest.get("weight_class")) if manifest else ""
            weight_class = weight_class or lookup.get(row)
            card_section = _text(manifest.get("card_section")).casefold() if manifest else "fight_card"
            bout_order = int(manifest["bout_order"]) if manifest else None
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
                "card_section": card_section,
                "card_section_label": CARD_SECTION_LABELS[card_section],
                "bout_order": bout_order,
                "fight_status": _text(manifest.get("fight_status")).casefold() if manifest else "confirmed",
                "prediction_status": "frozen",
                "batch_id": batch_id,
                "model_version": model_version,
                "calibration_version": calibration_version,
                "prediction_created_at": _text(row.get("prediction_timestamp_utc")) or None,
                "_sort_key": _fight_sort_key(row, fighter_a, fighter_b, manifest),
            }
            group_key = (event_date.isoformat(), _name_key(event_name))
            event = grouped.setdefault(group_key, {
                "event_name": event_name,
                "fights": [],
                "batch_ids": [],
                "model_versions": [],
                "calibration_versions": [],
            })
            event["fights"].append(fight)
            for field, value in (
                ("batch_ids", batch_id),
                ("model_versions", model_version),
                ("calibration_versions", calibration_version),
            ):
                if value not in event[field]:
                    event[field].append(value)
        except (TypeError, ValueError) as exc:
            invalid_count += 1
            LOGGER.warning(
                "Skipping invalid prediction row %s (%s): %s",
                row_number,
                _text(row.get("prediction_id")) or "unknown prediction",
                exc,
            )

    events = []
    for group_key, event_group in grouped.items():
        event_date, _event_name_key = group_key
        event_name = event_group["event_name"]
        fights = event_group["fights"]
        fights.sort(key=lambda fight: fight.pop("_sort_key"))
        timestamps = [fight["prediction_created_at"] for fight in fights if fight["prediction_created_at"]]
        batch_ids = event_group["batch_ids"]
        model_versions = event_group["model_versions"]
        calibration_versions = event_group["calibration_versions"]
        events.append({
            "event_id": _safe_event_id(event_date, event_name),
            "event_name": event_name,
            "event_date": event_date,
            "batch_id": batch_ids[0],
            "batch_ids": batch_ids,
            "prediction_status": "frozen",
            "model_version": model_versions[0],
            "model_versions": model_versions,
            "calibration_version": calibration_versions[0],
            "calibration_versions": calibration_versions,
            "prediction_created_at": min(timestamps) if timestamps else None,
            "fight_count": len(fights),
            "fights": fights,
        })
    events.sort(key=lambda event: (event["event_date"], event["event_name"].casefold()))
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
