import json
from dataclasses import replace
from pathlib import Path

import pytest

from ufc_pipeline.sportsbook.config import SportsbookConfig
from ufc_pipeline.sportsbook.consensus import ConsensusBuilder
from ufc_pipeline.sportsbook.domain import NormalizedOutcome
from ufc_pipeline.sportsbook.matching import CanonicalMatcher, import_reviewed_mapping
from ufc_pipeline.sportsbook.providers.the_odds_api import TheOddsApiAdapter
from ufc_pipeline.sportsbook.storage import SportsbookStore


ROOT = Path(__file__).resolve().parents[1]
UFC_DB = ROOT / "data" / "ufc.db"
FIXTURE = ROOT / "tests" / "fixtures" / "sportsbook" / "the_odds_api_historical.json"


@pytest.fixture
def sportsbook_env(tmp_path):
    config = SportsbookConfig(
        database_path=tmp_path / "sportsbook_odds.db",
        raw_archive_root=tmp_path / "raw",
        canonical_ufc_db=UFC_DB,
        reports_dir=tmp_path / "reports",
    )
    store = SportsbookStore(config)
    store.initialize()
    return config, store


def _normalized_fixture():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return payload, TheOddsApiAdapter().normalize(payload, request_type="historical", source_payload_hash="placeholder")


def _matched_main_snapshots(sportsbook_env):
    config, store = sportsbook_env
    payload, normalized = _normalized_fixture()
    archive = store.archive_payload("the_odds_api", payload)
    snapshots = tuple(replace(item, source_payload_hash=archive.payload_hash) for item in normalized.snapshots)
    matcher = CanonicalMatcher(UFC_DB, sportsbook_db=config.database_path)
    matched, reviews = matcher.match_many(snapshots)
    return archive, matcher, matched, reviews


def test_fighter_matching_exact_alias_accents_apostrophes_hyphens_and_suffixes(sportsbook_env):
    config, _ = sportsbook_env
    matcher = CanonicalMatcher(UFC_DB, sportsbook_db=config.database_path)
    assert matcher.match_fighter("the_odds_api", "Max Holloway").status == "matched"
    alias = matcher.match_fighter("the_odds_api", "Bobby Green")
    assert alias.status == "matched" and alias.canonical_name == "King Green"
    assert alias.method == "reviewed_repository_alias"
    assert matcher.match_fighter("the_odds_api", "Benoît Saint-Denis").canonical_name == "Benoit Saint Denis"
    assert matcher.match_fighter("the_odds_api", "Sean OMalley").canonical_name == "Sean O'Malley"
    assert matcher.match_fighter("the_odds_api", "Raul Rosas-Jr.").canonical_name == "Raul Rosas Jr."
    suffix = matcher.match_fighter("the_odds_api", "Khalil Rountree Jr")
    assert suffix.status == "matched" and suffix.canonical_name == "Khalil Rountree Jr."


def test_ambiguous_and_fuzzy_names_remain_unresolved_without_creating_alias(sportsbook_env):
    config, store = sportsbook_env
    matcher = CanonicalMatcher(UFC_DB, sportsbook_db=config.database_path)
    assert matcher.match_fighter("the_odds_api", "Michael Aswell Jr.").status == "ambiguous"
    fuzzy = matcher.match_fighter("the_odds_api", "Gable Stevenson")
    assert fuzzy.status == "unmatched"
    connection = store.connect(read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM provider_mappings").fetchone()[0] == 0
    finally:
        connection.close()


def test_reviewed_fighter_mapping_resolves_deterministically(sportsbook_env):
    config, _ = sportsbook_env
    result = import_reviewed_mapping(
        config.database_path,
        UFC_DB,
        provider_name="the_odds_api",
        entity_type="fighter",
        raw_identifier="The Highlight",
        canonical_identifier="362",
        created_by="pytest",
        review_note="fixture mapping",
    )
    assert result["canonical_name"] == "Justin Gaethje"
    matched = CanonicalMatcher(UFC_DB, sportsbook_db=config.database_path).match_fighter("the_odds_api", "The Highlight")
    assert matched.status == "matched"
    assert matched.fighter_id == 362
    assert matched.method == "reviewed_provider_mapping"


def test_reversed_order_alias_and_event_date_match_canonical_bouts(sportsbook_env):
    _, _, matched, reviews = _matched_main_snapshots(sportsbook_env)
    gaethje = [item for item in matched if item.provider_event_id == "toa-ufc300-gaethje-holloway"]
    assert {item.canonical_bout_id for item in gaethje if item.normalization_status == "normalized"} == {"1073"}
    green = next(item for item in matched if item.provider_event_id == "toa-ufc300-green-miller")
    assert green.canonical_bout_id == "1082"
    assert {outcome.canonical_fighter_id for outcome in green.outcomes} == {35, 43}
    assert any(item["status"] == "rejected_prediction_market" for item in reviews)


def test_one_day_utc_event_shift_is_allowed_but_repeat_matchups_still_require_unique_date(sportsbook_env):
    config, _ = sportsbook_env
    _, normalized = _normalized_fixture()
    draftkings = next(item for item in normalized.snapshots if item.provider_sportsbook_key == "draftkings")
    shifted = replace(draftkings, scheduled_event_time_utc="2024-04-14T00:30:00Z")
    result = CanonicalMatcher(UFC_DB, sportsbook_db=config.database_path).match_snapshot(shifted)
    assert result.status == "matched"
    assert result.snapshot.canonical_bout_id == "1073"


def test_cancelled_bout_and_replacement_mismatch_are_explicit(sportsbook_env):
    config, _ = sportsbook_env
    payload, normalized = _normalized_fixture()
    base = next(item for item in normalized.snapshots if item.provider_sportsbook_key == "draftkings")
    cancelled = replace(base, normalization_status="rejected_cancelled_bout", rejection_reason="provider marks cancelled")
    matcher = CanonicalMatcher(UFC_DB, sportsbook_db=config.database_path)
    assert matcher.match_snapshot(cancelled).status == "rejected_cancelled_bout"

    import_reviewed_mapping(
        config.database_path,
        UFC_DB,
        provider_name="the_odds_api",
        entity_type="bout",
        raw_identifier="replacement-event",
        canonical_identifier="1073",
        created_by="pytest",
    )
    alex = matcher.match_fighter("the_odds_api", "Alex Pereira")
    jamahal = matcher.match_fighter("the_odds_api", "Jamahal Hill")
    replacement = replace(
        base,
        provider_event_id="replacement-event",
        outcomes=(
            replace(base.outcomes[0], raw_outcome_name="Alex Pereira", raw_opponent_name="Jamahal Hill"),
            replace(base.outcomes[1], raw_outcome_name="Jamahal Hill", raw_opponent_name="Alex Pereira"),
        ),
    )
    result = CanonicalMatcher(UFC_DB, sportsbook_db=config.database_path).match_snapshot(replacement)
    assert result.status == "rejected_replacement_mismatch"


def test_explicit_non_ufc_league_is_rejected(sportsbook_env):
    config, _ = sportsbook_env
    _, normalized = _normalized_fixture()
    base = next(item for item in normalized.snapshots if item.provider_sportsbook_key == "draftkings")
    non_ufc = replace(base, league_key="Bellator MMA")
    result = CanonicalMatcher(UFC_DB, sportsbook_db=config.database_path).match_snapshot(non_ufc)
    assert result.status == "rejected_non_ufc"


def test_duplicate_and_duplicate_fighter_sides_are_rejected(sportsbook_env):
    config, _ = sportsbook_env
    _, normalized = _normalized_fixture()
    base = next(item for item in normalized.snapshots if item.provider_sportsbook_key == "draftkings")
    matcher = CanonicalMatcher(UFC_DB, sportsbook_db=config.database_path)
    _, reviews = matcher.match_many((base, base))
    assert any(item["status"] == "rejected_duplicate" for item in reviews)
    duplicate_sides = replace(
        base,
        outcomes=(base.outcomes[0], replace(base.outcomes[1], raw_outcome_name=base.outcomes[0].raw_outcome_name)),
    )
    assert matcher.match_snapshot(duplicate_sides).status == "rejected_incomplete_market"


def _seed_consensus(sportsbook_env):
    config, store = sportsbook_env
    archive, _, matched, _ = _matched_main_snapshots(sportsbook_env)
    main = [item for item in matched if item.provider_event_id == "toa-ufc300-gaethje-holloway"]
    draftkings = next(item for item in main if item.provider_sportsbook_key == "draftkings")
    # Exact-cutoff DraftKings update; the later one must be excluded.
    exact = replace(
        draftkings,
        observed_at_utc="2024-04-12T02:00:00Z",
        observed_at_original="2024-04-12T02:00:00Z",
        outcomes=(
            replace(draftkings.outcomes[0], decimal_odds=1.70, raw_implied_probability=1 / 1.70, original_odds=1.70),
            replace(draftkings.outcomes[1], decimal_odds=2.20, raw_implied_probability=1 / 2.20, original_odds=2.20),
        ),
    )
    later = replace(exact, observed_at_utc="2024-04-12T03:00:00Z", observed_at_original="2024-04-12T03:00:00Z")
    betmgm = replace(
        draftkings,
        provider_market_id="toa-ufc300-gaethje-holloway:betmgm:h2h",
        sportsbook_key="betmgm",
        provider_sportsbook_key="betmgm",
        sportsbook_display_name="BetMGM",
        observed_at_utc="2024-04-12T01:55:00Z",
        observed_at_original="2024-04-12T01:55:00Z",
        outcomes=(
            replace(draftkings.outcomes[0], decimal_odds=1.66, raw_implied_probability=1 / 1.66, original_odds=1.66),
            replace(draftkings.outcomes[1], decimal_odds=2.28, raw_implied_probability=1 / 2.28, original_odds=2.28),
        ),
    )
    batch = store.create_ingestion_batch(
        provider_name="the_odds_api",
        request_type="historical",
        archive=archive,
        query_timestamp_utc="2024-04-12T02:00:00Z",
    )
    store.store_snapshots(batch, tuple(main) + (exact, later, betmgm))
    return config


def test_consensus_selects_newest_at_or_before_cutoff_and_computes_statistics(sportsbook_env):
    config = _seed_consensus(sportsbook_env)
    result = ConsensusBuilder(config.database_path, config=config).build(
        canonical_bout_id="1073",
        canonical_fighter_id=250,
        canonical_opponent_id=362,
        prediction_as_of_utc="2024-04-12T02:00:00Z",
    )
    assert result.data_available
    assert result.sportsbook_count == 3
    assert not result.coverage_low
    assert result.included_sportsbook_keys == ("betmgm", "draftkings", "fanduel")
    assert result.consensus_opponent_probability == pytest.approx(1 - result.consensus_fighter_probability)
    assert result.minimum_fair_probability <= result.mean_fair_probability <= result.maximum_fair_probability
    assert result.probability_range == pytest.approx(result.maximum_fair_probability - result.minimum_fair_probability)
    assert result.probability_std >= 0
    selected_dk = next(item for item in result.selected_books if item.sportsbook_key == "draftkings")
    assert selected_dk.observed_at_utc == "2024-04-12T02:00:00Z"
    assert "observation_after_prediction_cutoff" in result.excluded_sportsbooks["draftkings"]
    assert result.pinnacle_available
    assert result.pinnacle_fair_probability is not None
    assert result.pinnacle_minus_broad_consensus == pytest.approx(result.pinnacle_fair_probability - result.consensus_fighter_probability)
    assert not result.pinnacle_included_in_broad_consensus


def test_consensus_side_swap_reverses_probabilities(sportsbook_env):
    config = _seed_consensus(sportsbook_env)
    builder = ConsensusBuilder(config.database_path, config=config)
    forward = builder.build(canonical_bout_id="1073", canonical_fighter_id=250, canonical_opponent_id=362, prediction_as_of_utc="2024-04-12T02:00:00Z")
    reverse = builder.build(canonical_bout_id="1073", canonical_fighter_id=362, canonical_opponent_id=250, prediction_as_of_utc="2024-04-12T02:00:00Z")
    assert forward.consensus_fighter_probability == pytest.approx(reverse.consensus_opponent_probability)
    assert forward.consensus_opponent_probability == pytest.approx(reverse.consensus_fighter_probability)


@pytest.mark.parametrize(("keys", "count", "low"), [(set(), 0, True), ({"draftkings"}, 1, True), ({"draftkings", "fanduel"}, 2, True), ({"draftkings", "fanduel", "betmgm"}, 3, False)])
def test_zero_one_two_and_three_book_coverage_flags(sportsbook_env, keys, count, low):
    config = _seed_consensus(sportsbook_env)
    result = ConsensusBuilder(config.database_path, config=config).build(
        canonical_bout_id="1073",
        canonical_fighter_id=250,
        canonical_opponent_id=362,
        prediction_as_of_utc="2024-04-12T02:00:00Z",
        eligible_sportsbook_keys=keys,
    )
    assert result.sportsbook_count == count
    assert result.coverage_low is low
    assert result.data_available is (count > 0)
    if count == 0:
        assert result.consensus_fighter_probability is None


def test_staleness_and_later_closing_snapshot_are_excluded(sportsbook_env):
    config = _seed_consensus(sportsbook_env)
    result = ConsensusBuilder(config.database_path, config=config).build(
        canonical_bout_id="1073",
        canonical_fighter_id=250,
        canonical_opponent_id=362,
        prediction_as_of_utc="2024-04-12T02:00:00+00:00",
        max_snapshot_age_hours=0.01,
    )
    assert result.sportsbook_count == 1  # DraftKings observation exactly at cutoff.
    assert "snapshot_stale" in result.excluded_sportsbooks["fanduel"]
    assert "observation_after_prediction_cutoff" in result.excluded_sportsbooks["draftkings"]


def test_optional_pinnacle_inclusion_and_persisted_selection_are_deterministic(sportsbook_env):
    config = _seed_consensus(sportsbook_env)
    builder = ConsensusBuilder(config.database_path, config=config)
    first = builder.build(
        canonical_bout_id="1073",
        canonical_fighter_id=250,
        canonical_opponent_id=362,
        prediction_as_of_utc="2024-04-12T02:00:00Z",
        include_pinnacle_in_broad_consensus=True,
        persist_selection=True,
    )
    second = builder.build(
        canonical_bout_id="1073",
        canonical_fighter_id=250,
        canonical_opponent_id=362,
        prediction_as_of_utc="2024-04-12T02:00:00Z",
        include_pinnacle_in_broad_consensus=True,
        persist_selection=True,
    )
    assert first == second
    assert first.sportsbook_count == 4
    assert first.pinnacle_included_in_broad_consensus
    connection = config.database_path
    import sqlite3
    db = sqlite3.connect(connection)
    try:
        assert db.execute("SELECT COUNT(*) FROM selected_consensus_inputs").fetchone()[0] == 4
    finally:
        db.close()


def test_prediction_markets_never_influence_consensus(sportsbook_env):
    config = _seed_consensus(sportsbook_env)
    result = ConsensusBuilder(config.database_path, config=config).build(
        canonical_bout_id="1073",
        canonical_fighter_id=250,
        canonical_opponent_id=362,
        prediction_as_of_utc="2024-04-12T02:00:00Z",
    )
    assert "polymarket" not in result.included_sportsbook_keys
    assert all(item.sportsbook_key != "polymarket" for item in result.selected_books)
    # Prohibited venues are rejected before canonical bout assignment, so they
    # never even enter the bout-specific candidate pool.
    assert "polymarket" not in result.excluded_sportsbooks
