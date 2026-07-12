"""Schema/workflow acceptance gate for user-supplied provider samples."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from .config import SportsbookConfig
from .domain import parse_utc
from .matching import CanonicalMatcher
from .providers.base import ProviderSchemaError
from .providers.the_odds_api import TheOddsApiAdapter
from .service import SportsbookIngestionService
from .storage import canonical_json_bytes


def sha256_file(path: str | Path) -> str | None:
    source = Path(path)
    if not source.exists():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class SampleValidationResult:
    provider_name: str
    sample_hash: str
    sample_provenance: str
    schema_accepted: bool
    historical_coverage_validated: bool
    stage3_coverage_audit_ready: bool
    request_type: str
    checks: dict[str, bool]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    counts: dict[str, int]
    bookmakers_found: tuple[str, ...]
    markets_found: tuple[str, ...]
    target_books_found: tuple[str, ...]
    matched_bout_ids: tuple[str, ...]
    unmatched_count: int
    ambiguous_count: int
    preliminary_representation: str
    replacement_or_cancellation_indicators: str
    cutoff_policy: str
    protected_hashes_before: dict[str, str | None]
    protected_hashes_after: dict[str, str | None]
    protected_state_unchanged: bool
    ingestion_batch_id: str | None
    report_json_path: str
    report_markdown_path: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SampleValidator:
    def __init__(self, config: SportsbookConfig | None = None):
        self.config = config or SportsbookConfig()

    def validate(
        self,
        *,
        provider_name: str,
        input_path: str | Path,
        request_type: str = "historical",
        sample_provenance: str = "user_supplied_unverified",
        raw_retention_permitted: bool = False,
        dry_run: bool = False,
    ) -> SampleValidationResult:
        if provider_name != "the_odds_api":
            raise ValueError("Stage 2 sample validation currently supports the_odds_api")
        protected = {
            "canonical_ufc_db": self.config.canonical_ufc_db,
            "live_prediction_ledger": Path("data/live/live_predictions.csv"),
            "official_benchmark": Path("benchmarks/official_baseline.json"),
        }
        before = {key: sha256_file(path) for key, path in protected.items()}
        payload = SportsbookIngestionService.load_json(input_path)
        sample_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        adapter = TheOddsApiAdapter()
        structural_errors = adapter.validate_required_fields(payload, request_type=request_type)
        warnings: list[str] = []
        errors: list[str] = list(structural_errors)
        normalized = None
        matched = ()
        reviews = ()
        ingestion_batch_id = None
        if not structural_errors:
            normalized = adapter.normalize(payload, request_type=request_type, source_payload_hash=sample_hash)
            matcher = CanonicalMatcher(self.config.canonical_ufc_db, sportsbook_db=self.config.database_path)
            matched, reviews = matcher.match_many(normalized.snapshots)
            try:
                ingestion = SportsbookIngestionService(self.config).ingest_local_payload(
                    provider_name=provider_name,
                    payload=payload,
                    request_type=request_type,
                    source_file=str(input_path),
                    query_timestamp_utc=normalized.provider_metadata.get("historical_query_timestamp"),
                    dry_run=dry_run,
                )
                ingestion_batch_id = ingestion.ingestion_batch_id
            except ProviderSchemaError as exc:
                errors.append(str(exc))
        snapshots = normalized.snapshots if normalized else ()
        events = payload.get("data", []) if isinstance(payload, dict) else payload if isinstance(payload, list) else []
        bookmakers = sorted({item.provider_sportsbook_key for item in snapshots if item.provider_sportsbook_key})
        markets = sorted({item.market_type for item in snapshots if item.market_type})
        target_books = sorted({item.sportsbook_key for item in snapshots if item.sportsbook_key and item.normalization_status == "normalized"})
        matched_ids = sorted({item.canonical_bout_id for item in matched if item.canonical_bout_id})
        unmatched_count = sum(item["status"] == "unmatched" for item in reviews)
        ambiguous_count = sum(item["status"] == "ambiguous" for item in reviews)
        sections = {
            str(event.get("card_section", "")).casefold()
            for event in events if isinstance(event, dict) and event.get("card_section")
        }
        has_prelims = bool({"prelims", "preliminary", "early_prelims", "early prelims"} & sections)
        if not has_prelims:
            warnings.append("No preliminary-bout metadata was present; preliminary coverage is not proven.")
        if not target_books:
            warnings.append("No approved target sportsbook was present in the sample.")
        event_statuses = {
            str(event.get("status", "")).casefold()
            for event in events if isinstance(event, dict) and event.get("status")
        }
        replacement_fields = any(
            isinstance(event, dict) and any(key in event for key in ("replacement", "replaced_by", "cancelled", "canceled"))
            for event in events
        )
        has_cancel_replacement = bool(event_statuses & {"cancelled", "canceled", "postponed"}) or replacement_fields
        if not has_cancel_replacement:
            warnings.append("Replacement/cancellation semantics were not demonstrated by this sample.")
        historical_query_timestamp = normalized.provider_metadata.get("historical_query_timestamp") if normalized else None
        observations_present = bool(snapshots) and all(item.observed_at_utc for item in snapshots)
        scheduled_present = bool(snapshots) and all(item.scheduled_event_time_utc for item in snapshots)
        complete_two_sided = any(
            item.market_type == "h2h" and len(item.outcomes) == 2 and all(outcome.validation_status == "valid" for outcome in item.outcomes)
            for item in snapshots
        )
        timestamps_ordered = bool(snapshots) and all(parse_utc(item.observed_at_utc) < parse_utc(item.scheduled_event_time_utc) for item in snapshots if item.normalization_status == "normalized")
        cutoff_reconstructable = bool(historical_query_timestamp and scheduled_present and observations_present and timestamps_ordered)
        identity_sufficient = bool(matched_ids)
        ufc_classifiable = bool(matched_ids) and all(item.sport_key == "mma_mixed_martial_arts" for item in matched if item.canonical_bout_id)
        if not raw_retention_permitted:
            warnings.append("Raw-payload retention permission has not been confirmed under the user's provider terms.")
        checks = {
            "valid_json_structure": not structural_errors,
            "observation_timestamps_present": observations_present,
            "scheduled_timestamps_present": scheduled_present,
            "historical_query_timestamp_present": bool(historical_query_timestamp),
            "complete_two_sided_moneyline_present": complete_two_sided,
            "canonical_ufc_identity_sufficient": identity_sufficient,
            "ufc_classification_supported": ufc_classifiable,
            "target_sportsbook_keys_identifiable": bool(target_books),
            "preliminary_bouts_demonstrated": has_prelims,
            "replacement_or_cancellation_behavior_demonstrated": has_cancel_replacement,
            "raw_retention_permitted_confirmed": raw_retention_permitted,
            "historical_24h_cutoff_reconstructable": cutoff_reconstructable,
        }
        required = (
            "valid_json_structure",
            "observation_timestamps_present",
            "scheduled_timestamps_present",
            "historical_query_timestamp_present",
            "complete_two_sided_moneyline_present",
            "canonical_ufc_identity_sufficient",
            "ufc_classification_supported",
            "raw_retention_permitted_confirmed",
            "historical_24h_cutoff_reconstructable",
        )
        schema_accepted = all(checks[key] for key in required) and not errors
        if not schema_accepted:
            errors.extend(f"required sample check failed: {key}" for key in required if not checks[key])
        after = {key: sha256_file(path) for key, path in protected.items()}
        protected_unchanged = before == after
        short_hash = sample_hash[:12]
        base = self.config.reports_dir / f"sample_validation_{provider_name}_{short_hash}"
        json_path = base.with_suffix(".json")
        markdown_path = base.with_suffix(".md")
        result = SampleValidationResult(
            provider_name=provider_name,
            sample_hash=sample_hash,
            sample_provenance=sample_provenance,
            schema_accepted=schema_accepted,
            historical_coverage_validated=False,
            stage3_coverage_audit_ready=schema_accepted and sample_provenance == "real_provider_sample",
            request_type=request_type,
            checks=checks,
            warnings=tuple(dict.fromkeys(warnings)),
            errors=tuple(dict.fromkeys(errors)),
            counts={
                "events": len(events),
                "bookmakers": len(bookmakers),
                "markets": len(snapshots),
                "outcomes": sum(len(item.outcomes) for item in snapshots),
                "complete_two_sided_markets": sum(item.market_type == "h2h" and len(item.outcomes) == 2 and all(outcome.validation_status == "valid" for outcome in item.outcomes) for item in snapshots),
                "matched_snapshots": sum(item.canonical_bout_id is not None for item in matched),
                "unmatched_snapshots": unmatched_count,
                "ambiguous_snapshots": ambiguous_count,
            },
            bookmakers_found=tuple(bookmakers),
            markets_found=tuple(markets),
            target_books_found=tuple(target_books),
            matched_bout_ids=tuple(matched_ids),
            unmatched_count=unmatched_count,
            ambiguous_count=ambiguous_count,
            preliminary_representation="demonstrated_by_sample_metadata" if has_prelims else "not_demonstrated",
            replacement_or_cancellation_indicators="demonstrated" if has_cancel_replacement else "not_demonstrated",
            cutoff_policy="historical: scheduled bout time minus 24 hours; prospective: immutable prediction freeze timestamp",
            protected_hashes_before=before,
            protected_hashes_after=after,
            protected_state_unchanged=protected_unchanged,
            ingestion_batch_id=ingestion_batch_id,
            report_json_path=str(json_path),
            report_markdown_path=str(markdown_path),
        )
        _atomic_write(json_path, json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n")
        _atomic_write(markdown_path, self._markdown(result))
        return result

    @staticmethod
    def _markdown(result: SampleValidationResult) -> str:
        lines = [
            "# Sportsbook provider sample validation",
            "",
            f"- Provider: `{result.provider_name}`",
            f"- Sample hash: `{result.sample_hash}`",
            f"- Provenance: `{result.sample_provenance}`",
            f"- Schema accepted: `{str(result.schema_accepted).lower()}`",
            "- Historical coverage validated: `false`",
            f"- Stage 3 coverage audit ready: `{str(result.stage3_coverage_audit_ready).lower()}`",
            f"- Protected state unchanged: `{str(result.protected_state_unchanged).lower()}`",
            "",
            "## Acceptance checks",
            "",
        ]
        lines.extend(f"- {key}: `{str(value).lower()}`" for key, value in result.checks.items())
        lines.extend(["", "## Counts", ""])
        lines.extend(f"- {key}: {value}" for key, value in result.counts.items())
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings or ("None",))
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in result.errors or ("None",))
        lines.extend([
            "",
            "This gate validates schema and timestamp workflow compatibility only. It does not establish historical coverage.",
            "",
        ])
        return "\n".join(lines)
