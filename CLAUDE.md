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
9. **Feature diagnostics (Step 5A)** — `ufc_pipeline/feature_diagnostics.py`:
   feature/scaling audit, feature-group ablations on the official 3-way
   split, standardized coefficients, permutation importance, leakage
   checks. Diagnostics only — it never changes the official model. Reports
   go to `reports/` (`step5a_feature_diagnostics.{json,md}`,
   `step5a_coefficients.csv`).
10. **Regularization + pruning search (Step 5B)** —
   `ufc_pipeline/step5b_regularization_search.py`: LR-only L1/L2/elastic-net
   sweep plus `weight_class` cleanup configs. Selection is **validation-only**
   (3 expanding-window rolling-origin folds *inside* the pre-test period,
   scored by mean Platt log loss); a small pre-declared finalist set is then
   scored **once** on the held-out test. Reports go to `reports/`
   (`step5b_regularization_search.{json,md}`, `step5b_candidate_results.csv`,
   `step5b_coefficients.csv`). It never changes the official model.
11. **Regularization confirmation (Step 5B.1)** —
   `ufc_pipeline/step5b1_regularization_confirmation.py`: confirmation-only
   audit of four pre-declared LR candidates from Step 5B. It runs five
   pre-test rolling-origin folds, re-reports the already-touched final test
   for those candidates only, and writes
   `reports/step5b1_regularization_confirmation.{json,md}`. It never runs a
   broad grid search and never changes the official model.
12. **Division interactions (Step 5C / Run 3)** —
   `ufc_pipeline/step5c_division_interactions.py`: LR-only controlled test
   of `weight_class × feature` interactions for common/current divisions
   selected from pre-test counts only. Uses five Step 5B.1 folds, fixed
   candidate configs, and final-test re-reports for validation-selected
   finalists only. Reports go to `reports/`
   (`step5c_division_interactions.{json,md}`,
   `step5c_candidate_results.csv`, `step5c_interaction_coefficients.csv`).
   It never changes the official model.

Official model: **Step 3C LR + Platt** (backtest log loss 0.6442, Brier
0.2267, AUC 0.6760). Step 3B + Platt (0.6449) is the preserved fallback
benchmark; Step 3C calibration outputs use `step3c_`-prefixed file names.
`benchmarks/official_baseline.json` is the fixed benchmark reference: a
candidate must beat it on held-out log loss (then Brier, then calibration
gaps) on the same split before the official model changes; never
regenerate that file without explicit sign-off. Step 5A (Run 1) found the
age and experience groups most valuable, most groups within the ±0.002
log-loss noise band, and one borderline win (dropping `weight_class`,
−0.0026 on the *test* set only).

Step 5B (Run 2) tested that hypothesis validation-first. Findings: the
official anchor reproduced exactly (0.6442244532062779); the best finalist
(elastic-net C=0.003, l1_ratio=0.1, all `weight_class` dropped) reached test
log loss 0.6419, +0.0023 over official — *just* past the 0.002 threshold.
But dropping `weight_class` is **not** confirmed on pre-test validation
(sub-noise, ~0.0003); the robust, validation-confirmed lever is **stronger
regularization** than the default C=1.0 (~0.011 validation improvement), and
L1/elastic-net zero out the `weight_class` one-hots on their own. saga
convergence was clean (no fits hit max_iter=5000). **The official model was
left unchanged**: the margin is thin and the weight_class win did not
reproduce on held-out validation.

Step 5B.1 (confirmation audit) used five pre-test folds and four
pre-declared candidates only. It confirmed stronger regularization as
validation-stable: official mean validation Platt log loss 0.66733 vs
elastic-net/drop_all_weight_class 0.65847 (gain 0.00886), with the
regularized candidates beating official in all 5 folds. The already-touched
final-test re-report remains 0.641919 for elastic-net/drop_all_weight_class
(gain 0.002305), so the edge is real but still borderline and not a fresh
independent test. Elastic-net all-features (0.65873 validation mean) and the
best simple L1 candidate (0.65863) were essentially tied with the drop-all
elastic-net candidate; elastic-net zeroed all `weight_class` one-hots when
included, and explicit `weight_class` dropping was only +0.00026 on
validation (far below 0.002). **The official model remains unchanged**:
retain the official feature set for now; if carrying a candidate forward,
prefer keeping `weight_class` available and letting elastic-net zero it.

Step 5C / Run 3 tested division-specific effects with `weight_class ×
feature` interactions. Eligible interaction divisions were the 11 current
divisions with at least 100 pre-test fights; `Catch Weight`, `Open Weight`,
`Super Heavyweight`, and `Women's Featherweight` were excluded from
interaction generation. Result: interactions beat the official C=1.0
baseline but **did not beat the stronger-regularized no-interaction LR
challenger**. Validation mean Platt log loss: stronger EN/drop raw
`weight_class` 0.65847; best interaction (`interactions_top_features_only`)
0.66166; full priority interactions 0.66348/0.66351 with and without raw
`weight_class`; official 0.66733. Best interaction lost to the challenger by
0.00319 and won only 1/5 folds vs the challenger. Final-test re-report:
stronger EN/drop raw `weight_class` 0.641920; best interaction 0.642407;
official 0.644224. Some stable interactions survived (especially
division-specific `age_diff`, plus a few striking interactions), but the
aggregate interaction models were worse and more complex. Raw `weight_class`
remained suspect: the raw-included interaction finalist retained 2/15 raw
one-hots and performed worse than official on the already-touched final
test. **Conclusion:** reject/keep division interactions experimental; the
best promotion candidate is still the stronger-regularized no-interaction LR
(`elasticnet`, C=0.003, l1_ratio=0.1, raw `weight_class` dropped), subject to
explicit human sign-off and a separate benchmark update. Run 4 can compare
nonlinear models/ensembles against both official and this locked LR
challenger, but odds/markets remain out unless explicitly approved later.

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
# Step 5A feature diagnostics (writes reports/step5a_*; --skip-permutation for a faster run)
.venv/bin/python scripts/run_step5a_feature_diagnostics.py
# Step 5B regularization + weight_class pruning search (writes reports/step5b_*; ~19 min full run,
# --quick for a fast smoke run, --max-candidates N to cap the grid)
.venv/bin/python scripts/run_step5b_regularization_search.py --output-dir reports
# Step 5B.1 confirmation audit (writes reports/step5b1_regularization_confirmation.{json,md})
.venv/bin/python scripts/run_step5b1_regularization_confirmation.py --output-dir reports
# Step 5C division interactions (writes reports/step5c_*; LR-only)
.venv/bin/python scripts/run_step5c_division_interactions.py --output-dir reports
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
| `ufc_pipeline/feature_diagnostics.py` | Step 5A: feature groups, ablations, coefficients, permutation importance, leakage checks |
| `ufc_pipeline/step5b_regularization_search.py` | Step 5B: validation-gated LR L1/L2/elastic-net + weight_class-pruning search, finalists scored once on test |
| `ufc_pipeline/step5b1_regularization_confirmation.py` | Step 5B.1: fixed-candidate regularization confirmation, extra pre-test folds, final-test re-report only |
| `ufc_pipeline/step5c_division_interactions.py` | Step 5C: LR-only common-division interaction candidates, validation-only selection, finalist test re-report |
| `benchmarks/official_baseline.json` | fixed official-model benchmark reference (never regenerate casually) |
| `scripts/` | one CLI wrapper per pipeline stage |
| `docs/greco_field_audit.md` | source-overlap rules (mdabbert vs Greco) |
