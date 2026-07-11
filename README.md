# UFC Fight Database + Elo Baseline + Pre-Fight Features + First Models

**Stages 1–4 of a UFC fight prediction project:** data ingestion, a
time-safe Elo baseline, a time-safe pre-fight feature builder, and a first
model training/evaluation stage. There is deliberately no betting strategy,
no UI, and no live predictions yet. The point of the early stages is a
foundation you can trust: a clean fight database and feature datasets where
every pre-fight number provably uses only information available *before*
that fight. Stage 4 then asks one honest question: does a first ML model
beat the Elo-only baseline on held-out *future* fights?

## Project goal

Given historical UFC fights, produce a dataset where each row is one fight,
processed chronologically, containing each fighter's **pre-fight** Elo, the
Elo difference, the Elo-implied win probability, and the actual result. That
dataset is both (a) a sane baseline to beat and (b) a leak-free feature for a
future ML model.

## The one rule that matters: time safety

For every fight, pre-fight Elo and expected win probability are recorded
**before** the result of that fight updates any rating. Fights are processed
strictly in date order; fights on the same date keep their original dataset
order (encoded in `fight_id`). A later fight can never influence an earlier
fight's pre-fight numbers. See `ufc_pipeline/elo.py` — the critical section
is commented.

## Data source (v1): mdabbert Ultimate UFC Dataset

Download the CSV from Kaggle (`mdabbert/ultimate-ufc-dataset`) and place it at:

```
data/raw/mdabbert_ufc.csv
```

(any path works — you pass it on the command line).

**Assumptions made about this dataset:**

- Bout-level rows with red/blue corner structure (`RedFighter`/`BlueFighter`
  or similar). Red maps to `fighter_a`, blue to `fighter_b`.
- The `Winner` column may contain either a corner label (`"Red"`/`"Blue"`) or
  a fighter name — both are handled.
- Column spellings vary between versions (`Date`/`date`,
  `R_fighter`/`RedFighter`, ...). Ingestion uses fuzzy, case/underscore-
  insensitive column matching (`ufc_pipeline/columns.py`), so it is not
  brittle about exact spellings. If required columns genuinely can't be
  found, you get an error listing what's missing, what columns *were* found,
  and examples of expected names.
- Draws and no-contests are dropped (Elo needs a decisive result) and
  reported in the ingestion summary.

**Why odds / rankings / precomputed stats are ignored for now:** the mdabbert
file ships betting odds, rankings, and many precomputed fighter averages.
Those are dropped at the adapter stage. Odds and rankings are market/editorial
signals we want to compare *against* later, not bake into the baseline; the
precomputed averages have unclear as-of semantics (some may include the fight
being predicted), so trusting them blindly risks leakage. Stage 2 will rebuild
rolling stats ourselves, time-safely. Current-fight stats (strikes, takedowns,
control time) are outcomes, never features.

## Install & run

```bash
pip install pandas pytest

# 1. Ingest the raw CSV into SQLite
python scripts/ingest_fights.py --input data/raw/mdabbert_ufc.csv --db data/ufc.db --source mdabbert

# 2. Build time-safe Elo snapshots (K and starting Elo are configurable)
python scripts/build_elo.py --db data/ufc.db --k 32 --starting-elo 1500

# 3. Export the model-ready baseline + current ratings
python scripts/export_elo_dataset.py --db data/ufc.db \
    --output data/processed/ufc_elo_baseline.csv \
    --ratings-output data/processed/current_fighter_elos.csv

# 4. Build the time-safe pre-fight feature dataset (requires steps 1-2)
python scripts/build_prefight_features.py --db data/ufc.db \
    --output data/processed/ufc_prefight_features.csv

# 5. Train + evaluate models (Elo baseline vs logistic regression [vs RF])
python scripts/train_model.py --input data/processed/ufc_prefight_features.csv
#    optional: --include-random-forest --split-date 2022-01-01 --test-size 0.2
#              --no-categorical --features elo_diff age_diff reach_diff

# --- Step 3B: rolling striking/grappling stats (needs a per-fight stats source) ---

# 6. Convert Greco1899/scrape_ufc_stats files (download the 3 CSVs from
#    https://github.com/Greco1899/scrape_ufc_stats) into the wide format:
python scripts/convert_greco_stats.py --stats data/raw/ufc_fight_stats.csv \
    --results data/raw/ufc_fight_results.csv --events data/raw/ufc_event_details.csv \
    --output data/raw/fight_stats_wide.csv

# 7. Ingest per-fight stats (matched to fights by date + fighter names)
python scripts/ingest_fight_stats.py --input data/raw/fight_stats_wide.csv --db data/ufc.db

# 8. Build the expanded Step 3 + 3B feature dataset
python scripts/build_prefight_features_step3b.py --db data/ufc.db \
    --output data/processed/ufc_prefight_features_step3b.csv \
    --debug-fighters "Max Holloway"        # optional: print rolling timelines

# 9. Compare: Elo baseline vs basic Step-3 model vs Step-3+3B model
python scripts/compare_models.py --input data/processed/ufc_prefight_features_step3b.csv \
    --include-random-forest

# 10. Calibrate the best Step 3B model (Platt + isotonic) on a 3-way
#     chronological train/calibration/test split
python scripts/calibrate_model.py \
    --input data/processed/ufc_prefight_features_step3b.csv \
    --output-dir data/processed --model-dir data/models
#    optional: --train-frac 0.70 --calibration-frac 0.15
#              or date mode: --train-end-date 2021-01-01 --calibration-end-date 2023-01-01

# Tests
python -m pytest tests/ -v
```

## Database schema (`data/ufc.db`)

- **fighters** — `fighter_id`, `name` (display, original casing),
  `normalized_name` (lowercase matching key — deliberately NOT unique:
  two different fighters can share a name; true identity lives in
  `fighter_source_ids`), plus optional height/reach/stance/date_of_birth
  slots for stage-2 metadata. Name handling trims/collapses whitespace but
  deliberately does *not* strip accents or punctuation, to avoid merging
  different fighters.
- **fighter_source_ids** — durable source-specific fighter identities
  (see "Fighter identity layer"): `fighter_id` (nullable until linked),
  `source`, `source_fighter_id` (URL slug; UNIQUE per source),
  `source_fighter_url`, `raw_name`, `normalized_name`, evidence fields
  (`listed_weight_lbs`, `date_of_birth`, `height_cm`, `reach_cm`),
  `resolution` + `resolution_note`, `first_seen_date`/`last_seen_date`,
  timestamps.
- **fights** — `fight_id`, `date`, `event`, `fighter_a_id`, `fighter_b_id`,
  `winner_id`, `fighter_a_won`, `weight_class`, `method`, `round`, `source`,
  `source_row_id`. `fight_id` order preserves dataset order for same-date
  tie-breaking.
- **fight_elo_snapshots** — per fight: pre-Elo for both fighters, `elo_diff`,
  expected win probabilities, post-Elo, and `fighter_a_won`. Rebuilt from
  scratch on every `build_elo.py` run (idempotent, deterministic).

## Output columns

`data/processed/ufc_elo_baseline.csv` (one row per fight, chronological):

| column | meaning |
|---|---|
| `fight_id`, `date`, `event` | identity |
| `fighter_a`, `fighter_b` | red corner is A, blue is B |
| `winner`, `fighter_a_won` | **result labels** (never features for pre-fight Elo) |
| `fighter_a_pre_elo`, `fighter_b_pre_elo` | ratings *before* the fight |
| `elo_diff` | `fighter_a_pre_elo - fighter_b_pre_elo` |
| `fighter_a_expected_win_prob`, `fighter_b_expected_win_prob` | Elo-implied probabilities (sum to 1) |
| `fighter_a_post_elo`, `fighter_b_post_elo` | ratings after applying the result |
| `weight_class`, `method`, `round` | result/metadata only |

`data/processed/current_fighter_elos.csv`: `fighter`, `current_elo`,
`fights_processed`, `wins`, `losses`, `last_fight_date`.

A **leakage guard** in the exporter refuses to write the baseline if any
forbidden column pattern (odds, rank, `winner_*`, `loser_*`, sig_str,
takedown, control, avg) appears, except the explicitly allowed result fields
`winner`, `fighter_a_won`, `method`, `round`.

## Elo details

- New fighters start at **1500** (`--starting-elo`).
- **K = 32** by default (`--k`).
- `expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))`
- `new_elo = old_elo + K * (actual - expected)`

## Source-adapter design (adding Greco1899 later)

All source-specific logic lives in `ufc_pipeline/adapters/`. An adapter is one
function, `adapt(raw_df) -> normalized_df`, returning the internal columns
(`date`, `fighter_a_name`, `fighter_b_name`, `winner_name`, ...). Everything
downstream (validation, DB, Elo, export) only sees normalized names.

To add **Greco1899/scrape_ufc_stats** later: write `adapters/greco.py` that
joins its `ufc_fight_results.csv` + `ufc_events.csv` into bout-level rows with
the same normalized columns, register it in `adapters/__init__.py`, and run
`ingest_fights.py --source greco`. Its daily-refreshed data then serves for
validation, backfilling, or predicting upcoming cards — no other code changes.

For winner/loser-shaped sources (no corners), conversion alternates
deterministically by row index (even row → winner in the A slot, odd row →
winner in the B slot), so `fighter_a` is not systematically the winner and
output is reproducible without randomness.

## Tests (`tests/test_pipeline.py`)

1. New fighters start at 1500.
2. Pre-fight Elo is recorded before the update (a fight's own result never
   touches its pre-fight numbers).
3. Winner gains Elo, loser loses Elo.
4. Chronological ordering respected; same-date ties broken by dataset order.
5. Out-of-order input CSVs are sorted by date before processing.
6. Expected win probabilities sum to ~1.
7. Same input → byte-identical output (determinism).
8. No forbidden (leaky) columns in the exported baseline; allowed result
   fields still present.
9. Red → `fighter_a`, blue → `fighter_b`; corner-label winners resolved to names.
10. Winner/loser-only sources: `fighter_a` is not always the winner.

Plus: clear error on missing columns; draws/bad dates/self-fights dropped and
reported.

## Pre-fight feature builder (Stage 3)

`scripts/build_prefight_features.py` produces the model-ready dataset
`data/processed/ufc_prefight_features.csv`: one row per fight, where **every
feature is computed only from information available before that fight** and
`fighter_a_won` is the target.

**Why time safety matters:** a model trained on features that secretly
contain the answer (career totals computed after the fact, averages that
include the fight being predicted) will look brilliant in backtests and be
useless on real upcoming fights. The feature builder therefore uses the same
discipline as the Elo engine: for each fight in chronological order it (1)
reads fighter A's historical state, (2) reads fighter B's, (3) writes the
feature row and target, and (4) only then updates both histories with the
result. Same-date fights keep dataset order via `fight_id`; out-of-order
input is re-sorted internally.

**Columns created** (51 total; every `*_diff` is fighter_a − fighter_b):

- *Identity/result:* `fight_id`, `date`, `event`, `fighter_a`, `fighter_b`,
  `winner`, `fighter_a_won` (target), `weight_class`, `method`, `round` —
  result/metadata only, never inputs to features.
- *Elo (pre-fight, from `fight_elo_snapshots`):* `fighter_a_pre_elo`,
  `fighter_b_pre_elo`, `elo_diff`, `fighter_a_expected_win_prob`,
  `fighter_b_expected_win_prob`.
- *Physical:* age, height, reach for each fighter plus `age_diff`,
  `height_diff`, `reach_diff`. These come from the source's per-fight
  tale-of-the-tape values (mdabbert `RedAge`, `RedHeightCms`, `RedReachCms`,
  ...), which are as-of-fight-date and therefore time-safe. If a
  `date_of_birth` is ever stored in the fighters table, age falls back to
  DOB-at-fight-date. A single "current age" value is never used, because it
  would inject future information into old fights.
- *Prior record:* `prior_fights`, `prior_wins`, `prior_losses`,
  `prior_win_pct` per fighter + diffs. "Prior" always means *before* the
  current fight.
- *Recent form:* `wins_last_3`, `win_pct_last_3`, `wins_last_5`,
  `win_pct_last_5` per fighter + diffs, over the previous ≤3/≤5 fights.
- *Activity:* `days_since_last_fight` per fighter + diff.

**Missing values** are written as real nulls (empty CSV cells) — never
silently filled:

- Debut fighters: `prior_win_pct`, `win_pct_last_3/5`, and
  `days_since_last_fight` are null. (We chose null over 0.5 for debut
  win-pct: "no information" should look like no information; impute at model
  time.) `prior_fights`/`prior_wins`/`prior_losses`/`wins_last_*` are 0,
  which is exact, not imputed.
- Missing age/height/reach in the source: null.
- Any diff with a null side: null.

**Intentionally excluded for now:** betting odds, rankings, style-based
rolling stats (strikes/takedowns/control), post-fight totals, and any
precomputed career averages. Style stats need per-round data with the same
record-before-update discipline; that's a later stage, not a quick add. The
same leakage guard as the Elo export runs on the final columns.

## Model training + evaluation (Stage 4)

`scripts/train_model.py` trains a first model on
`ufc_prefight_features.csv` and evaluates it against the Elo-only baseline
on held-out newer fights.

**Why the split is chronological, never random:** a random split mixes past
and future — the model could train on a fighter's 2023 form and be tested on
their 2021 fight, which inflates every metric and tells you nothing about
predicting real upcoming cards. Here fights are sorted by date; the model
trains on the oldest 80% and is tested on the newest 20% (or use
`--split-date YYYY-MM-DD`).

**Models:**
1. *Elo baseline* — `fighter_a_expected_win_prob` used directly as the
   predicted probability. No training; this is the bar to clear.
2. *Logistic regression* — sklearn Pipeline: median imputation + scaling for
   numerics, most-frequent imputation + one-hot for `weight_class`
   (`handle_unknown="ignore"`), then `LogisticRegression(max_iter=5000,
   random_state=42)`. Default regularization; no tuning yet.
3. *Random forest* (optional, `--include-random-forest`) — untuned
   `RandomForestClassifier(n_estimators=300, random_state=42)`, comparison
   only.

**Features used** (defaults; any missing ones are skipped and reported):
`elo_diff`, `age_diff`, `height_diff`, `reach_diff`, `prior_fights_diff`,
`prior_wins_diff`, `prior_losses_diff`, `prior_win_pct_diff`,
`wins_last_3_diff`, `win_pct_last_3_diff`, `wins_last_5_diff`,
`win_pct_last_5_diff`, `days_since_last_fight_diff`, plus one-hot
`weight_class` (drop with `--no-categorical`). Missing numerics are imputed
*inside* the pipeline from training data only — rows are not silently
dropped, and any drops are printed.

**Forbidden as inputs** (enforced by a leakage guard that hard-stops
training): `fighter_a_won`/`winner` (target/result), `method`, `round`
(known only after the fight), post-fight Elo, betting odds, rankings,
current-fight strikes/takedowns/control, `winner_*`/`loser_*` columns, and
any future-derived averages. The guard checks selected feature names against
forbidden patterns (`winner, loser, post, method, round, odds, rank,
sig_str, takedown, control, result, outcome`) before any training happens.

**Outputs:**
- `data/models/logistic_regression_prefight.joblib` (and
  `random_forest_prefight.joblib` if enabled) — full pipelines, so
  imputation/scaling/encoding travel with the model.
- `data/processed/model_predictions.csv` — one row per *test* fight with
  Elo, logistic (and RF) probabilities and 0.5-threshold classes.
- `data/processed/model_evaluation.json` — all metrics per model.
- `data/processed/logistic_regression_coefficients.csv` — coefficients
  sorted by absolute value. These show which features the model leans on
  given everything else in the model; they are **not** causal explanations
  (features are scaled and correlated — e.g. `prior_wins_diff` and
  `prior_fights_diff` overlap, so individual signs can look odd while the
  combination is fine).
- `data/processed/random_forest_feature_importances.csv` if RF is enabled.

**How to read the metrics:**
- *Accuracy alone is not enough.* Picking the Elo favourite every time
  already gets decent accuracy; a probability model must do more.
- *Log loss* scores the predicted probabilities themselves — being confident
  and wrong is punished hard. It is the primary comparison number here.
- *Brier score* is a gentler probability error measure (lower is better).
- *ROC AUC* measures ranking: does the model give real winners higher
  probabilities than losers, regardless of threshold.
- *Calibration table* (10 buckets, 0.0–0.1 … 0.9–1.0): among fights where
  the model said ~60%, did fighter A actually win ~60% of the time? A model
  can have decent accuracy and still be badly calibrated — and calibration
  is what matters if probabilities will ever be compared to odds.
- *Comparing to Elo:* the logistic model earns its keep only if its log loss
  (and ideally Brier/AUC) beats the Elo baseline on the same test fights.

**Interpretation warnings:** beating Elo is the *first* goal; beating the
betting market is a much harder, later goal (odds already encode most public
information). Nothing here should be treated as a profitable betting system.

## Rolling striking & grappling features (Step 3B)

Step 3B adds historical stat rates per fighter, built with the same
record-before-update discipline as Elo and Step 3: for each fight in
(date, fight_id) order, both fighters' accumulated stats are snapshotted
*first*, the feature row is emitted, and only then does the current fight's
stat line enter either accumulator. The current fight's strikes, takedowns,
control time, knockdowns, and submission attempts can therefore never appear
in its own row, and future fights never affect earlier rows (both are
covered by tests).

**Data requirement:** per-fight stat totals, which the mdabbert dataset does
not carry. The intended source is Greco1899/scrape_ufc_stats: download its
`ufc_fight_stats.csv`, `ufc_fight_results.csv`, and `ufc_event_details.csv`,
run `convert_greco_stats.py` (sums per-round rows into per-fight totals,
parses "x of y" and "m:ss" formats, derives fight minutes from
round/time/format), then `ingest_fight_stats.py` (matches rows to known
fights by date + fighter-name pair, corner-order-insensitive; unmatched rows
are reported, not guessed). Stats live in the `fight_stats` table: one row
per (fight, fighter) with offensive totals; absorbed/defense numbers come
from the opponent's row.

**Features** (per fighter, then `fighter_a − fighter_b` diffs):
`sig_strikes_landed_per_min` (hist landed / hist minutes),
`sig_strikes_absorbed_per_min` (hist opponent landed / hist minutes),
`sig_strike_differential` (landed − absorbed per min), `striking_accuracy`
(hist landed / hist attempted), `striking_defense` (1 − opp landed / opp
attempted), `knockdown_rate` (hist KD / min × 15), `takedowns_per_15`,
`takedown_accuracy`, `takedown_defense` (1 − TDs allowed / opp TD attempts),
`control_time_per_15`, `submission_attempts_per_15`.

**Missing-history policy:** a fighter with no prior fights-with-stats gets
*null* for every rolling metric plus `fighter_x_no_prior_stats = 1` (and
`no_prior_stats_diff`). Nulls are deliberately NOT zeros — zero would claim
a debutant "lands nothing per minute", which is false information, not
missing information. Ratios with zero denominators (no TD attempts yet) are
also null. Imputation happens later inside the model pipeline from
training-split medians, and the flags let the model treat missing history
explicitly. A past fight with no stats row counts toward Step-3 record
features but contributes nothing to stat accumulators.

**Output:** `data/processed/ufc_prefight_features_step3b.csv` — all Step 3
columns unchanged, plus per-fighter values, the 11 diffs, and the 3 flags
(row count identical to Step 3; asserted at build time). The leakage guard
allowlists these exact historical names; any *other* column containing
sig_str/takedown/control still trips it — raw current-fight stat names
remain forbidden as model inputs.

**Comparison (`compare_models.py`):** evaluates, on one shared chronological
test split: (1) the untrained Elo baseline, (2) logistic regression on basic
Step-3 features, (3) logistic regression on Step 3 + 3B (plus RF variants
with `--include-random-forest`). Writes `model_comparison.json`,
`model_comparison_predictions.csv`, and per-set
`coefficients_<name>.csv` / `rf_importances_<name>.csv`. The point is not
more features — it is whether historical striking/grappling stats add
predictive value beyond Elo and the basic features, measured primarily by
log loss on future fights. With many correlated features on few training
rows, the expanded logistic model can genuinely do *worse* (overfitting);
that is a finding, not a bug.

## Greco central-database expansion (real-data workflow)

The central database combines **mdabbert** (preferred fights base, per-fight
tale-of-the-tape) with **Greco1899/scrape_ufc_stats** (per-fight stat
totals, fighter details, and a fallback fights base). Field-by-field
classification, overlap audit, and canonical-source rules live in
`docs/greco_field_audit.md`. The short version:

- **From Greco (new):** per-fight offensive totals into `fight_stats`
  (with a `source` provenance column); fighter height/reach/stance/DOB into
  `fighters` from `ufc_fighter_tott.csv` (**fill-only** — existing values
  are never overwritten); fights themselves via
  `convert_greco_fights.py` + `--source greco` *only when no mdabbert base
  is available* (one fights base per database build).
- **Overlaps (deterministic, documented, never averaged):** per-fight
  mdabbert age/height/reach always win; Greco static values only fill gaps
  (SQL `COALESCE` in the feature query, DOB-derived age fallback).
  mdabbert's precomputed career averages stay ignored — Greco's raw totals
  feed our own provably time-safe rolling stats instead.
- **Greco quirks handled:** `OUTCOME` ('W/L'/'L/W'/'D/D'/'NC/NC') → winner
  name; messy `WEIGHTCLASS` strings normalized to clean divisions
  ("UFC Women's Flyweight Title Bout" → "Women's Flyweight"); "x of y" and
  "m:ss" parsing; fight minutes derived from round/time/format.

Greco file location: `data/raw/greco/` (`ufc_fight_stats.csv`,
`ufc_fight_results.csv`, `ufc_event_details.csv`, `ufc_fighter_tott.csv`
from https://github.com/Greco1899/scrape_ufc_stats).

Full Greco-based sequence (when no mdabbert CSV is present). Register
fighter identities BEFORE ingesting fights so same-name fighters split
correctly:

```bash
python scripts/convert_greco_fights.py --results data/raw/greco/ufc_fight_results.csv \
    --events data/raw/greco/ufc_event_details.csv --output data/raw/greco_fights.csv
python scripts/build_fighter_identities.py --tott data/raw/greco/ufc_fighter_tott.csv \
    --db data/ufc.db --report data/processed/fighter_identity_report.json
python scripts/ingest_fights.py --input data/raw/greco_fights.csv --db data/ufc.db --source greco
python scripts/build_elo.py --db data/ufc.db
python scripts/ingest_fighter_details.py --input data/raw/greco/ufc_fighter_tott.csv --db data/ufc.db
python scripts/convert_greco_stats.py --stats data/raw/greco/ufc_fight_stats.csv \
    --results data/raw/greco/ufc_fight_results.csv --events data/raw/greco/ufc_event_details.csv \
    --output data/processed/greco_fight_stats_converted.csv
python scripts/ingest_fight_stats.py --input data/processed/greco_fight_stats_converted.csv --db data/ufc.db
python scripts/build_prefight_features.py --db data/ufc.db
python scripts/build_prefight_features_step3b.py --db data/ufc.db
python scripts/compare_models.py --input data/processed/ufc_prefight_features_step3b.csv
```

With a real mdabbert CSV, ingest it as the fights base first
(`--source mdabbert`) and skip `convert_greco_fights`; the stats/tott layers
match by date + normalized name pair (corner-order-insensitive) and report
unmatched rows rather than guessing.

## Fighter identity layer (Greco URLs)

**The problem:** the pipeline used to key fighters purely on a normalized
name, which silently MERGED different fighters who share one. The real
Greco data has 7 such names (14 people) — e.g. two Bruno Silvas, a
flyweight and a middleweight, whose 21 fights shared one Elo rating, one
record, and one rolling-stats history; the merged "Jean Silva" row even
carried the wrong fighter's DOB. Since calibration corrects the model's
probabilities, correcting them on top of poisoned histories fixes the
wrong thing — so identity was cleaned before touching calibration again.

**The fix:** Greco's `ufc_fighter_tott.csv` carries UFCStats fighter URLs
(`/fighter-details/<hex>`), which are durable per-person identifiers. The
`fighter_source_ids` table stores them: one row per (source,
source_fighter_id) with the URL, the raw and normalized source name,
supporting evidence (listed weight, DOB, height, reach), a resolution
status, first/last seen fight dates, and the linked canonical
`fighter_id`. One source URL maps to exactly one canonical fighter
(enforced by a UNIQUE constraint); a canonical fighter may hold IDs from
several sources. `fighters.normalized_name` is therefore no longer UNIQUE
— it is a matching key, not an identity (existing databases are migrated
in place, preserving all fighter_id values).

**Matching priority (deterministic, never guessed):**

1. A known source URL already linked → its existing `fighter_id`, always.
2. A new URL whose normalized name is unique among source identities and
   matches exactly one canonical fighter → link.
3. Same-name groups (2+ URLs sharing a name) are resolved **per bout**
   during fight ingestion using supporting evidence, in order:
   (a) exact division-weight match — a Flyweight bout belongs to the
   125 lbs identity; (b) career continuity — a bout within 540 days of a
   bout already assigned to exactly one candidate (this is what routes
   Jean Silva's one-off 2024 lightweight bout to the active featherweight,
   not the retired 1977 lightweight whose listed weight is numerically
   closer); (c) DOB plausibility — age 18–45 at fight date.
4. Still ambiguous → **unresolved**: reported in the ingestion summary and
   identity report, and the bout falls back to a shared name-keyed fighter
   row that no identity owns. No guessing, no silent merging either way.

Tott ingestion (`ingest_fighter_details.py`) is now URL-first: a linked
URL updates its own fighter row directly; the name fallback fires only for
provably unambiguous names, so the 1965 kickboxer Michael McDonald's DOB
can never land on the 1991 bantamweight again. Stats ingestion resolves
identity transitively — each stats row matches a fight, and the fight
already references canonical (split) fighter ids — and its unmatched
report now distinguishes bad dates, unknown fighters, unresolved
identities, and no-fight-on-that-date (draws/NCs dropped at ingestion).

**Command** (run before ingesting fights; safe to re-run any time — known
URLs keep their links; re-running after ingestion refreshes the report):

```bash
python scripts/build_fighter_identities.py \
    --tott data/raw/greco/ufc_fighter_tott.csv --db data/ufc.db \
    --report data/processed/fighter_identity_report.json
```

**Outputs:** `data/processed/fighter_identity_report.json` (totals,
already-linked/newly-linked/unlinked counts, duplicate normalized names,
same-name-not-merged cases with per-identity evidence, linked and
unresolved examples) plus identity lines in the fight-ingestion summary
(resolved slot counts, new links, per-identity bout splits, unresolved
cases).

**Real-data results:** all 4,496 Greco URLs registered; 2,677 linked to
canonical fighters (100% of tott matches now via URL); 1,812 belong to
fighters not in the modelable dataset; 14 same-name identities across 7
names handled with 0 unresolved bout slots. Bruno Silva's 21 fights split
10 flyweight / 11 middleweight; all 7 Jean Silva bouts (including the
lightweight one) went to the 1996-born featherweight.

**Real-data results (8,547 fights, 1994–2026; chronological 80/20 split,
test = 2023-01 onward; after the fighter identity layer):** the Step 3B
logistic regression is the best model on every probability metric — log
loss 0.6567 vs 0.6617 (basic) vs 0.6839 (Elo); accuracy 0.610; AUC 0.662.
(Pre-identity numbers were 0.6593/0.6641 — cleaning the same-name merges
improved every trained model.) Two honest caveats: (1) the first-listed
(red) corner encodes the promotional favorite and its win rate drifted from
66% (train era) to 56% (2023+), so all trained models are somewhat
overconfident toward fighter A in recent years — fixed by the Step 4B
calibration layer below; (2) unmatched Greco stats rows were exactly the
draws/no-contests deliberately excluded from the fights table plus a
handful of fighters absent from the DB (0% effective unmatched rate on
modelable fights; the ingestion report now itemizes every reason).

## Style-matchup features (Step 3C)

Step 3B gave each fighter individual rolling rates; Step 3C adds what those
rates say about the **pairing**: does A's striking output exceed what B
historically absorbs, does A's wrestling meet a B who historically cannot
stop takedowns, does A's reach advantage coincide with the striking volume
to use it. Built by `ufc_pipeline/matchup_features.py` with the exact
record-before-update loop of Step 3B (snapshot both fighters' past stats →
emit the row → only then fold in the current fight), so current-fight stats
can never appear in their own row and future fights never affect earlier
rows (both covered by tests).

**Direction convention (every matchup feature): positive = fighter A
advantage.**

**New rolling "against" stats** (per fighter + A−B diffs; derived from
opponents' offensive rows already stored in `fight_stats` — nothing was
invented, so no proposed feature had to be skipped for missing data):
`takedowns_allowed_per_15`, `opp_takedown_attempts_per_15`,
`opp_sig_str_attempted_per_min`, `control_time_absorbed_per_15`,
`knockdowns_absorbed_per_15`, `submission_attempts_absorbed_per_15`.

**Matchup features (exact formulas; `a_`/`b_` are pre-fight rolling values):**

| feature | formula |
|---|---|
| `striking_matchup_net_advantage` | `(a_sig_landed_pm − b_sig_absorbed_pm) − (b_sig_landed_pm − a_sig_absorbed_pm)` |
| `striking_accuracy_matchup_net_advantage` | `(a_strike_acc − b_strike_def) − (b_strike_acc − a_strike_def)` |
| `takedown_matchup_net_advantage` | `(a_td_per15 − b_td_allowed_per15) − (b_td_per15 − a_td_allowed_per15)` |
| `takedown_accuracy_matchup_net_advantage` | `(a_td_acc − b_td_def) − (b_td_acc − a_td_def)` |
| `control_matchup_net_advantage` | `(a_ctrl_per15 − b_ctrl_absorbed_per15) − (b_ctrl_per15 − a_ctrl_absorbed_per15)` |
| `knockdown_matchup_net_advantage` | `(a_kd_rate − b_kd_absorbed_per15) − (b_kd_rate − a_kd_absorbed_per15)` |
| `submission_matchup_net_advantage` | `(a_sub_per15 − b_sub_absorbed_per15) − (b_sub_per15 − a_sub_absorbed_per15)` |
| `reach_volume_interaction` | `reach_diff × (a_sig_landed_pm + b_sig_landed_pm) / 2` |
| `pace_pressure_advantage` | `(a_sig_landed_pm + a_td_per15) − (b_sig_landed_pm + b_td_per15)` |
| `opponent_pressure_absorption_advantage` | `(b_sig_absorbed_pm + b_td_allowed_per15) − (a_sig_absorbed_pm + a_td_allowed_per15)` |

Two documented honesty notes: (1) each `*_matchup_net_advantage` expands
algebraically to `(aX + aY) − (bX + bY)`; where both underlying diffs
already exist in 3B (striking) the net adds interpretability but no new
information to a *linear* model — where one side is a new against-stat
(takedown/control/knockdown/submission) it carries genuinely new signal;
`reach_volume_interaction` is a true product interaction. (2) The naive
`reach_diff × volume_diff` product was rejected because it breaks the
direction rule (a shorter AND less active fighter A would score positive).
**Style archetype scores were deliberately skipped** in this version:
honest era-relative "striker-ness" needs time-safe expanding-window
normalization — real complexity for unproven payoff.

**Missing history:** identical policy to 3B — nulls, never silent zeros;
any matchup feature with a null input is null; the new
`matchup_history_missing` flag is 1 when *either* fighter lacks stat
history (the per-side flags can't express "either"). Coverage on real data:
against-stats ~74–90% per side, matchup nets ~74%
(`takedown_accuracy_matchup_net` 55% — needs TD attempts on both sides),
flag 100%. Nulls are median-imputed inside the model pipeline from
training-split data.

**Commands** (versioned outputs; Step 3B files are never overwritten):

```bash
python scripts/build_prefight_features_step3c.py --db data/ufc.db \
    --output data/processed/ufc_prefight_features_step3c.csv
#   optional: --debug-fighters "Max Holloway"
python scripts/compare_models.py --input data/processed/ufc_prefight_features_step3c.csv \
    --metrics-output data/processed/model_comparison_step3c.json \
    --predictions-output data/processed/model_comparison_step3c_predictions.csv
python scripts/calibrate_model.py --input data/processed/ufc_prefight_features_step3c.csv \
    --output-dir data/processed --model-dir data/models --feature-set step3c
```

**Results (same official 70/15/15 benchmark split; test = 1,282 fights,
2023-11 → 2026-05):**

| model | log loss | Brier | AUC | acc |
|---|---|---|---|---|
| Step 3B LR + Platt (old official) | 0.6449 | 0.2271 | 0.6712 | 0.6186 |
| **Step 3C LR + Platt (new official)** | **0.6442** | **0.2267** | **0.6760** | 0.6193 |

Step 3C + Platt improves every headline metric: calibrated log loss
−0.0007 (marginal), Brier −0.0004, AUC +0.0048 (the most substantive gain),
and the above-0.7 calibration gap halves (−0.053 → −0.029). Honest caveats:
the log-loss edge is small, and on the separate 80/20 comparison split the
*uncalibrated* 3C model was a wash vs 3B (0.6569 vs 0.6567) — the gains are
real but modest, not a breakthrough. The leakage guard was also
strengthened in this step: the raw `fight_stats` column names
(`td_landed`, `ctrl_seconds`, `knockdowns`, `sub_attempts`) are now
explicitly forbidden as model inputs alongside the existing patterns.

## Probability calibration (Step 4B)

The Step 3B comparison exposed one clear weakness: the best model was
**overconfident** — test-window predictions above 0.7 won about 7 points
less often than predicted, driven by the first-listed/red-corner win rate
drifting from ~66% in the training era to ~56% in 2023+. Calibration fixes
the *probabilities* without touching features or feature generation:
`scripts/calibrate_model.py` adds a recalibration layer on top of the best
Step 3B logistic regression. It is a separate evaluation layer — the
`compare_models.py` outputs are preserved unchanged.

**What calibration is:** a learned correction `raw_probability →
calibrated_probability`, fit on fights the base model has never trained on.
Two methods are implemented:

- **Platt scaling** — a two-parameter logistic curve fit on the base
  model's *logits* (`logit(p) = log(p / (1-p))`; probabilities are clipped
  to `[1e-6, 1-1e-6]` first). Smooth, monotonic, and nearly impossible to
  overfit. Logits were chosen over raw probabilities because we control the
  base model and they're one line to compute — this is the textbook Platt
  formulation and behaves better near 0/1.
- **Isotonic regression** — a monotonic step function fit directly on the
  base model's probabilities (`out_of_bounds="clip"`). More flexible but
  can overfit small calibration windows; its outputs can be exactly 0/1,
  so calibrated probabilities are clipped before log loss.

**The three-way chronological split (never random):**

```
oldest fights ──────────────► newest fights
[   train 70%   ][ calibration 15% ][ test 15% ]
  base model fits    calibrators        ALL metrics;
  HERE ONLY          fit HERE ONLY      NOTHING fits here
```

Defaults are 70/15/15 by row count (`--train-frac`, `--calibration-frac`);
date mode uses `--train-end-date` + `--calibration-end-date` (train is
before the first date, calibration is between them, test is on/after the
second). The split function hard-fails on empty windows, misordered dates,
or a lone date argument. **The calibrator must never be fit on the final
test set** — that would turn the calibration metrics into in-sample
fiction. Tests enforce all of this.

**Command:**

```bash
python scripts/calibrate_model.py \
    --input data/processed/ufc_prefight_features_step3b.csv \
    --output-dir data/processed --model-dir data/models
```

**Outputs:**
- `data/processed/calibration_comparison.json` — config, per-model metrics
  (uncalibrated / Platt / isotonic + Elo and basic-Step-3 context),
  high-confidence diagnostics, and a plain-English verdict.
- `data/processed/calibration_predictions.csv` — one row per **test** fight
  with all three probabilities and 0.5-threshold predictions.
- `data/processed/calibration_tables.csv` — 10-bucket calibration tables
  (long format: model, bucket, n, avg predicted, actual, gap).
- `data/models/step3b_logistic_regression_uncalibrated.joblib`,
  `..._platt_calibrated.joblib`, `..._isotonic_calibrated.joblib` — the last
  two are `CalibratedPipeline` objects whose `predict_proba` returns
  calibrated probabilities.

**How to read the numbers:** log loss and Brier score the probabilities
themselves (lower is better) and are the success metrics here; the
calibration gap (avg predicted − actual win rate, per bucket and for
predictions above 0.7/0.8) shows overconfidence directly (positive =
overconfident). AUC only ranks predictions, so a monotonic calibration
barely moves it — a big AUC change would be a red flag, not a win. Accuracy
may not move at all; calibration reshapes probabilities, not necessarily
picks.

**Real-data results (8,547 fights; train to 2021-05, calibration to
2023-11, test 2023-11 → 2026-05, 1,282 test fights; after the fighter
identity layer):** Platt is the best method — log loss 0.6562 → **0.6449**,
Brier 0.2323 → **0.2271**, and the above-0.7 overconfidence gap went from
**+6.4 points to −5.3 points** (slightly *under*confident now). Isotonic
also improved (0.6458) but Platt wins on both log loss and Brier, and it
stays the project default: isotonic only replaces it when the data clearly
proves it better (a log-loss win beyond the near-tie margin) and never on
an overfit-prone small calibration window. AUC was unchanged (0.6712) and
accuracy moved only +0.2 points, exactly as expected. The calibrated Platt
model is the one to use for probability estimates.

### Rolling/live calibration (`--mode live`)

The backtest mode above freezes one historical calibration window forever,
which is right for honest evaluation but wrong for production: the
red-corner drift that motivated calibration keeps drifting, so the
calibrator should be refit on recent fights as new results arrive. Live
mode does exactly that:

- **backtest mode** (default) = honest final-test evaluation. Three
  windows; nothing is ever fit on the test window; test metrics are real.
- **live mode** = current model-building workflow for future predictions.
  The base Step 3B logistic regression trains on all fights **before** a
  rolling recent window (default 730 days, ending at `--as-of-date` or the
  latest completed fight); the calibrator (Platt by default) is refit on
  that window; the base model + calibrator pair is saved for future use.
  Live mode reports **calibration-window diagnostics only** — the
  calibrated numbers are in-sample for the calibrator (they look good by
  construction) and are a fit sanity check, **not** a final-test metric.

```bash
python scripts/calibrate_model.py \
    --input data/processed/ufc_prefight_features_step3b.csv \
    --output-dir data/processed --model-dir data/models \
    --mode live --calibration-window-days 730 --method platt
#   optional: --as-of-date 2026-01-01 --min-calibration-rows 500
```

730 days is the recommended default: on current data it holds ~1,000
fights — enough for a stable two-parameter Platt fit while staying recent.
A warning is printed (and saved in the report) whenever the window holds
fewer than `--min-calibration-rows` (default 500) fights; isotonic in
particular should not be trusted on small windows. Fights after
`--as-of-date` are excluded from *everything* (training, calibration,
reporting), so backfilled "as of last January" runs stay time-safe.

**Live outputs** (`live_*` names so backtest outputs are never
overwritten): `data/processed/live_calibration_report.json` (mode, method,
window config, train/calibration date ranges and row counts, raw vs
calibrated log loss/Brier/high-confidence gaps on the window, warnings),
`live_calibration_table.csv` (10-bucket tables, raw vs calibrated),
`live_calibration_predictions.csv` (window rows), and
`data/models/step3b_logistic_regression_live_uncalibrated.joblib` +
`step3b_logistic_regression_live_platt_calibrated.joblib` (a
`CalibratedPipeline`; its `predict_proba` returns calibrated
probabilities for future fights).

## Read-only upcoming predictions dashboard

The repository includes a dependency-free dashboard for official predictions
already frozen by Step 6B. It does **not** run the model, generate predictions,
or modify the production ledger. The server uses Python's standard library and
binds to loopback by default.

```bash
.venv/bin/python scripts/serve_predictions_dashboard.py
# Open http://127.0.0.1:8000/
# Optional: --host 127.0.0.1 --port 8000
```

The same process serves the stable read-only endpoint:

```text
GET /api/predictions/upcoming
```

The endpoint rereads `data/live/live_predictions.csv` for every request, so a
future event appears without frontend changes when valid unresolved rows are
added to an official frozen Step 6B batch. A fight disappears when its ledger
status is no longer `pending`, any resolution/scoring field is populated, or
its event date is in the past. An empty upcoming ledger is a normal response:
`{"events": []}`.

### Ledger-to-API schema mapping

| Dashboard field | Production source |
|---|---|
| fight ID | `prediction_id` |
| batch ID | `prediction_batch_id` |
| event/date | `event_name`, `event_date` |
| fighters | `fighter_a`, `fighter_b` |
| probabilities | `predicted_probability_a`, `predicted_probability_b` |
| unresolved state | `status` plus blank target/result/resolution/scoring fields |
| model metadata | `model_version`, `calibration_version` |
| creation time | `prediction_timestamp_utc` |
| weight class | `weight_class` when present; otherwise the safely confined feature CSV named by `input_source` |

The current ledger has no `weight_class` column. The service may enrich that
display field from an `input_source` CSV only when the resolved file remains
under `data/live/features/`; arbitrary paths are never read. If no safe weight
class is available, the card displays `UFC Bout`.

Rows are displayable only when they are `live_forward`, unresolved, and their
batch identifier marks them as both `official` and `frozen`. Fighter/event
names must be present; probabilities must be numeric, finite, within `[0, 1]`,
and complementary within `1e-6`. Malformed rows are logged and excluded, and
the API exposes only a non-sensitive invalid-row count. The predicted winner
and confidence label are derived on the backend from the stored probabilities:

| Higher stored probability | Label |
|---|---|
| 50.0% to under 55.0% | Toss-up |
| 55.0% to under 62.0% | Slight lean |
| 62.0% to under 70.0% | Moderate confidence |
| 70.0% to under 80.0% | High confidence |
| 80.0% and above | Very high confidence |

The frontend is static HTML/CSS/JavaScript served from the same origin. It
formats the API's stored probabilities for display and never performs model
inference or replacement probability calculations. For network deployment,
keep the app behind a production reverse proxy and explicitly choose a bind
address; the built-in server intentionally defaults to local access.

## Not a betting system (and not finished)

This is an Elo baseline, a leak-free feature foundation, and a first,
deliberately simple modeling pass. The logistic model is only as good as its
margin over the Elo baseline on future fights.

The dashboard is a display surface only. It has no accounts, administrative
controls, odds, betting links, or ability to edit predictions.
