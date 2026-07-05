# CLAUDE.md

Guidance for Claude Code when working in this repository. `agents.md` holds
the same core rules; keep the two files in sync when either changes.

## What this project is

A UFC fight prediction ML pipeline built in deliberate stages, all feeding a
central SQLite database (`data/ufc.db`):

1. **Ingestion** — mdabbert Kaggle CSV (preferred fights base) and/or
   Greco1899/scrape_ufc_stats (per-fight stat totals, fighter details,
   fallback fights base). Source-specific parsing lives in
   `ufc_pipeline/adapters/`.
2. **Fighter identity layer** — durable source-specific fighter IDs
   (UFCStats fighter URLs from Greco) in the `fighter_source_ids` table
   (`ufc_pipeline/identity.py`). Same-name fighters (e.g. the two Bruno
   Silvas) are split, never merged; ambiguous cases are reported, never
   guessed. Register identities BEFORE ingesting fights.
3. **Elo baseline** — time-safe pre-fight Elo snapshots (`ufc_pipeline/elo.py`).
4. **Step 3 features** — time-safe pre-fight features: record, recent form,
   physical diffs (`ufc_pipeline/features.py`).
5. **Step 3B features** — rolling striking/grappling rates over *previous*
   fights only (`ufc_pipeline/stats_features.py`).
6. **Step 3C features** — style-matchup interactions + rolling "against"
   stats (`ufc_pipeline/matchup_features.py`). Positive matchup value =
   fighter A advantage, always. Same record-before-update discipline.
7. **Modeling** — chronological train/test evaluation of Elo vs logistic
   regression vs optional random forest (`ufc_pipeline/modeling.py`).
8. **Calibration (Step 4B)** — `ufc_pipeline/calibration.py`, two modes:
   *backtest* (3-way chronological split, honest held-out test metrics)
   and *live* (rolling recent window, default 730 days, refits Platt for
   future predictions; diagnostics only, no held-out test).

Official model: **Step 3C LR + Platt** (backtest log loss 0.6442, Brier
0.2267, AUC 0.6760). Step 3B + Platt (0.6449) is the preserved fallback
benchmark; Step 3C calibration outputs use `step3c_`-prefixed file names.

There is deliberately **no betting logic, no odds as features, no UI**.

## Setup

- Use `python3`; plain `python` may not exist. Prefer the project venv at
  `.venv/bin/python` when present.
- Dependencies (no pyproject/requirements file exists):
  `pandas numpy scikit-learn joblib pytest`
- If the system Python lacks `ensurepip`, bootstrap the venv with
  `python3 -m venv --without-pip .venv` then run get-pip.py.

## Common commands

```bash
# Tests (always run before finishing a change)
.venv/bin/python -m pytest tests/ -q

# Full pipeline (Greco-based; see README for the mdabbert variant).
# Identity registration MUST run before fight ingestion.
.venv/bin/python scripts/build_fighter_identities.py --tott data/raw/greco/ufc_fighter_tott.csv \
    --db data/ufc.db --report data/processed/fighter_identity_report.json
.venv/bin/python scripts/ingest_fights.py --input data/raw/greco_fights.csv --db data/ufc.db --source greco
.venv/bin/python scripts/build_elo.py --db data/ufc.db
.venv/bin/python scripts/ingest_fighter_details.py --input data/raw/greco/ufc_fighter_tott.csv --db data/ufc.db
.venv/bin/python scripts/ingest_fight_stats.py --input data/processed/greco_fight_stats_converted.csv --db data/ufc.db
.venv/bin/python scripts/build_prefight_features.py --db data/ufc.db
.venv/bin/python scripts/build_prefight_features_step3b.py --db data/ufc.db
.venv/bin/python scripts/build_prefight_features_step3c.py --db data/ufc.db
.venv/bin/python scripts/compare_models.py --input data/processed/ufc_prefight_features_step3c.csv \
    --metrics-output data/processed/model_comparison_step3c.json \
    --predictions-output data/processed/model_comparison_step3c_predictions.csv
# calibration: backtest (honest held-out eval) and live (rolling refit for future)
.venv/bin/python scripts/calibrate_model.py --input data/processed/ufc_prefight_features_step3b.csv \
    --output-dir data/processed --model-dir data/models
.venv/bin/python scripts/calibrate_model.py --input data/processed/ufc_prefight_features_step3b.csv \
    --output-dir data/processed --model-dir data/models --mode live --calibration-window-days 730
```

## Non-negotiable invariants

- **Time safety.** Every pre-fight number must be computed from information
  available *before* that fight. The pattern everywhere: read both fighters'
  historical state, emit the feature row, and only then apply the current
  fight's result/stats. Sort by `(date, fight_id)`; `fight_id` breaks
  same-date ties deterministically.
- **Chronological splits only, never random.** Two-way in `modeling.py`,
  three-way (train → calibration → test) in `calibration.py`. The base model
  fits only on train rows, calibrators fit only on calibration rows, and all
  reported metrics come from the final test window. **Never fit a calibrator
  on the final test set.** Live mode's calibration-window diagnostics are
  in-sample for the calibrator — never present them as test metrics.
- **Fighter identity.** `fighters.normalized_name` is a matching key, NOT
  unique; durable identity is `fighter_source_ids` (UFCStats URLs). Never
  silently merge same-name fighters; never guess ambiguous identities —
  resolution order is URL → unique name → per-bout evidence (exact division
  weight → career continuity → DOB plausibility) → reported unresolved.
  Never remap existing fighter_id values in place; splits materialize on a
  fresh rebuild. Platt stays the default calibration method unless data
  clearly proves otherwise.
- **Forbidden model inputs** (leakage guard in `modeling.py` hard-stops):
  odds, rankings, `winner`/`loser`-derived columns, `method`, `round`,
  post-fight Elo, current-fight stats, precomputed source averages. The
  Step 3B rolling-stat names are exact-name allowlisted because they are
  historical aggregates; anything else matching sig_str/takedown/control is
  rejected. Result columns stay in outputs as labels/metadata only.
- **Missing history is null, not zero** — unless zero is an exact count
  (debut prior wins). Imputation happens inside the sklearn Pipeline from
  training-split statistics only.
- **Metrics:** log loss is the primary comparison number, then Brier and
  calibration gaps. Accuracy alone is never a success metric.
- **Preserve existing outputs.** Calibration is a separate layer; do not
  replace `compare_models.py` outputs. Avoid rewriting files under
  `data/processed/` unless the task requires regenerating them.

## Fable 5 Orchestrator Workflow

When using Claude Fable 5, act as the orchestrator.

Your job:
- Understand the goal.
- Inspect enough context to form a plan.
- Break the work into clear subtasks.
- Delegate implementation to the `implementer` subagent when edits are needed.
- Delegate review to the `reviewer` subagent after meaningful changes.
- Integrate results into a final answer.
- Keep the main context clean.

Do not personally perform large mechanical edits if a subagent can do them.
Use Fable for planning, architecture, risk assessment, debugging strategy, and
final synthesis.

Before delegating, give the subagent:
- The exact goal.
- Relevant files or search targets.
- Constraints.
- Definition of done.
- Tests/checks to run.

After subagents return, verify their result before claiming the task is
complete.

## Code style

- Small modules in `ufc_pipeline/`, thin argparse wrappers in `scripts/`
  (they import `_path` to fix sys.path). Source-specific parsing stays in
  adapters/ingestion modules.
- Docstrings explain *design decisions*, not just mechanics — follow the
  existing voice (e.g. the split rationale at the top of `modeling.py`).
- Prefer pandas/SQLite structured operations over ad hoc CSV string work.
- Tests focus on time-safety, leakage guards, deterministic ordering, and
  edge cases; synthetic fixtures live in the test files themselves
  (see `synthetic_features` in `tests/test_modeling.py`).
- Determinism matters: `RANDOM_STATE = 42`, same input → same output.

## Key files

| Path | Role |
|---|---|
| `ufc_pipeline/modeling.py` | chronological split, leakage guard, metrics, LR/RF pipelines, model comparison |
| `ufc_pipeline/calibration.py` | backtest 3-way split + rolling live mode, Platt/isotonic calibrators, diagnostics, verdict |
| `ufc_pipeline/identity.py` | fighter identity layer: URL registration, same-name evidence resolution |
| `ufc_pipeline/matchup_features.py` | Step 3C style-matchup features + rolling against-stats |
| `ufc_pipeline/elo.py` | time-safe Elo engine |
| `ufc_pipeline/features.py`, `stats_features.py` | Step 3 / Step 3B feature builders |
| `scripts/` | one CLI wrapper per pipeline stage |
| `docs/greco_field_audit.md` | source-overlap rules (mdabbert vs Greco) |
