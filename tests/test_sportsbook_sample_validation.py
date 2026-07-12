import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from ufc_pipeline.sportsbook.config import SportsbookConfig
from ufc_pipeline.sportsbook.sample_validation import SampleValidator


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sportsbook" / "the_odds_api_historical.json"
UFC_DB = ROOT / "data" / "ufc.db"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def validator(tmp_path):
    config = SportsbookConfig(
        database_path=tmp_path / "sportsbook_odds.db",
        raw_archive_root=tmp_path / "raw",
        canonical_ufc_db=UFC_DB,
        reports_dir=tmp_path / "reports",
    )
    return SampleValidator(config)


def test_valid_synthetic_sample_passes_schema_but_never_claims_coverage_or_stage3(validator):
    result = validator.validate(
        provider_name="the_odds_api", input_path=FIXTURE,
        request_type="historical", sample_provenance="synthetic_fixture",
        raw_retention_permitted=True,
    )
    assert result.schema_accepted
    assert not result.historical_coverage_validated
    assert not result.stage3_coverage_audit_ready
    assert result.checks["historical_24h_cutoff_reconstructable"]
    assert result.checks["preliminary_bouts_demonstrated"]
    assert result.counts["matched_snapshots"] >= 1
    assert Path(result.report_json_path).exists()
    assert Path(result.report_markdown_path).exists()
    report = json.loads(Path(result.report_json_path).read_text(encoding="utf-8"))
    assert report["historical_coverage_validated"] is False


def test_report_paths_are_deterministic_for_same_sample(validator):
    first = validator.validate(provider_name="the_odds_api", input_path=FIXTURE, sample_provenance="synthetic_fixture", raw_retention_permitted=True, dry_run=True)
    second = validator.validate(provider_name="the_odds_api", input_path=FIXTURE, sample_provenance="synthetic_fixture", raw_retention_permitted=True, dry_run=True)
    assert first.report_json_path == second.report_json_path
    assert first.report_markdown_path == second.report_markdown_path


def test_missing_observation_timestamp_fails(validator, tmp_path):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload.pop("timestamp")
    for event in payload["data"]:
        for book in event["bookmakers"]:
            book.pop("last_update", None)
            for market in book["markets"]:
                market.pop("last_update", None)
    result = validator.validate(provider_name="the_odds_api", input_path=_write_json(tmp_path / "missing_observed.json", payload), raw_retention_permitted=True, dry_run=True)
    assert not result.schema_accepted
    assert not result.checks["observation_timestamps_present"]


def test_missing_scheduled_timestamp_fails(validator, tmp_path):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["data"][0].pop("commence_time")
    result = validator.validate(provider_name="the_odds_api", input_path=_write_json(tmp_path / "missing_scheduled.json", payload), raw_retention_permitted=True, dry_run=True)
    assert not result.schema_accepted
    assert not result.checks["valid_json_structure"]


def test_missing_two_sided_markets_fails(validator, tmp_path):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for event in payload["data"]:
        for book in event["bookmakers"]:
            book["markets"][0]["outcomes"] = book["markets"][0]["outcomes"][:1]
    result = validator.validate(provider_name="the_odds_api", input_path=_write_json(tmp_path / "one_sided.json", payload), raw_retention_permitted=True, dry_run=True)
    assert not result.schema_accepted
    assert not result.checks["complete_two_sided_moneyline_present"]


def test_no_target_books_and_no_prelims_are_warnings_not_fabricated_coverage(validator, tmp_path):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["data"] = payload["data"][:1]
    payload["data"][0].pop("card_section", None)
    for book in payload["data"][0]["bookmakers"]:
        book["key"] = "unknown_book"
    result = validator.validate(provider_name="the_odds_api", input_path=_write_json(tmp_path / "unknown_books.json", payload), raw_retention_permitted=True, dry_run=True)
    assert not result.checks["target_sportsbook_keys_identifiable"]
    assert not result.checks["preliminary_bouts_demonstrated"]
    assert any("No approved target sportsbook" in warning for warning in result.warnings)
    assert any("preliminary" in warning.lower() for warning in result.warnings)
    assert result.preliminary_representation == "not_demonstrated"


def test_unmatched_and_ambiguous_fights_are_reported(validator, tmp_path):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    event = payload["data"][0]
    event["home_team"] = "Michael Aswell Jr."
    event["away_team"] = "Not A Real Fighter"
    for book in event["bookmakers"]:
        for market in book["markets"]:
            market["outcomes"] = [
                {"name": "Michael Aswell Jr.", "price": 1.9},
                {"name": "Not A Real Fighter", "price": 1.9},
            ]
    payload["data"] = [event]
    result = validator.validate(provider_name="the_odds_api", input_path=_write_json(tmp_path / "unresolved.json", payload), raw_retention_permitted=True, dry_run=True)
    assert result.ambiguous_count > 0
    assert not result.checks["canonical_ufc_identity_sufficient"]


def test_raw_retention_confirmation_is_required(validator):
    result = validator.validate(provider_name="the_odds_api", input_path=FIXTURE, sample_provenance="synthetic_fixture", raw_retention_permitted=False, dry_run=True)
    assert not result.schema_accepted
    assert not result.checks["raw_retention_permitted_confirmed"]


def test_real_sample_flag_only_unlocks_coverage_audit_after_schema_acceptance(validator):
    result = validator.validate(provider_name="the_odds_api", input_path=FIXTURE, sample_provenance="real_provider_sample", raw_retention_permitted=True, dry_run=True)
    assert result.schema_accepted
    assert result.stage3_coverage_audit_ready
    assert not result.historical_coverage_validated


def test_validation_never_alters_protected_production_files(validator):
    protected = [
        UFC_DB,
        ROOT / "data/live/live_predictions.csv",
        ROOT / "benchmarks/official_baseline.json",
        ROOT / "data/models/snapshots/pre_layoff_20260711/official_step5c_lr_platt_snapshot.joblib",
        ROOT / "data/processed/ufc_prefight_features_step3c.csv",
    ]
    before = {str(path): _hash(path) for path in protected}
    result = validator.validate(provider_name="the_odds_api", input_path=FIXTURE, sample_provenance="synthetic_fixture", raw_retention_permitted=True)
    after = {str(path): _hash(path) for path in protected}
    assert result.protected_state_unchanged
    assert before == after


def test_cli_dry_run_validates_without_creating_sidecar_database(tmp_path):
    db = tmp_path / "odds.db"
    command = [
        sys.executable, str(ROOT / "scripts/validate_sportsbook_provider_sample.py"),
        "--provider", "the_odds_api", "--input", str(FIXTURE),
        "--sample-provenance", "synthetic_fixture", "--raw-retention-permitted",
        "--db", str(db), "--raw-dir", str(tmp_path / "raw"),
        "--ufc-db", str(UFC_DB), "--output-dir", str(tmp_path / "reports"), "--dry-run",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    output = json.loads(completed.stdout)
    assert output["schema_accepted"]
    assert not db.exists()


def test_network_cli_refuses_without_explicit_flag_and_makes_no_call():
    command = [sys.executable, str(ROOT / "scripts/fetch_the_odds_api.py"), "--request-type", "current"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "network access is disabled" in completed.stderr


def test_historical_network_cli_requires_billable_acknowledgement(monkeypatch):
    command = [
        sys.executable, str(ROOT / "scripts/fetch_the_odds_api.py"),
        "--request-type", "historical", "--as-of-utc", "2024-01-01T00:00:00Z", "--allow-network",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "billable" in completed.stderr
