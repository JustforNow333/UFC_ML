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
   The report generator does not mutate the benchmark file; promotion is a
   separate explicit decision.
13. **Nonlinear models + blend (Step 5D / Run 4)** —
   `ufc_pipeline/step5d_nonlinear_models.py`: controlled nonlinear-model
   comparison against the current official LR. Re-fits the official LR anchor
   and the historical Step 3C LR from locked configs, runs a bounded sklearn
   `HistGradientBoostingClassifier` grid (24 points; early stopping carved
   from the fit window only; NaN handled natively, no scaling), optionally an
   already-installed XGBoost/LightGBM (never installed here) and a
   non-promotable random-forest sanity check, plus a validation-tuned convex
   LR+HGB blend. Uses the five Step 5B.1 pre-test folds, Platt/isotonic
   calibration, validation-only selection, and final-test re-reports for
   finalists only. Reports go to `reports/` (`step5d_nonlinear_models.{json,md}`,
   `step5d_candidate_results.csv`, `step5d_feature_importance.csv`,
   `step5d_blend_results.csv`). It never changes the official model or the
   benchmark file. No sportsbook odds, prediction markets, neural networks, or
   new data sources.
14. **Nonlinear sanity audit (Step 5D.1 / Run 4.1)** —
   `ufc_pipeline/step5d1_nonlinear_sanity_audit.py`: diagnostics-only audit of
   the surprising Run 4 result. Checks anchor + Run 4 HGB reproduction,
   probability orientation (log loss/AUC of `p` vs `1-p`), LR/HGB row-feature
   parity across every split/fold, calibration protocol (uncal/Platt/isotonic
   with AUC-invariance), HGB native NaN handling, an expanded *staged*
   HGB search (coarse spanning grid → local refinement + a max_iter /
   early-stopping sub-study, validation-only), an early-stopping isolation
   audit, a blend stability audit (fold-by-fold optimal weight), and a
   feature-importance / probability-distribution audit. Reuses the Step 5D
   fit/calibration/blend machinery verbatim. Reports go to `reports/`
   (`step5d1_nonlinear_sanity_audit.{json,md}`,
   `step5d1_hgb_sensitivity_results.csv`, `step5d1_calibration_audit.csv`). It
   promotes nothing and never edits the benchmark file.
15. **Historical pseudo-live replay (Step 6A)** —
   `ufc_pipeline/step6a_pseudo_live_replay.py`: event-by-event replay of the
   official model. For each past event it re-fits the official LR on the older
   portion of strictly-earlier fights, fits Platt on the most recent earlier
   window, predicts the event, then reveals results and scores them. It writes
   a replay ledger (`reports/replay/step6a_pseudo_live_predictions.csv`) plus
   `reports/step6a_pseudo_live_replay.{json,md}` and
   `reports/step6a_rolling_metrics.csv` — overall/rolling/by-year/by-event/
   by-confidence metrics, calibration buckets, drift/data-quality, and leakage
   checks. It is a **pipeline-validation and calibration/drift tool, not a new
   model search**: it reuses the official config as-is, never retunes or
   replaces the model, never writes `data/processed/` or the benchmark file,
   and never touches sportsbook odds / prediction markets / new model families.
   *Pseudo*-live ≠ the static backtest (which is a single fit on the fixed
   2023-11..2026-05 held-out window) and ≠ true future live tracking (it reuses
   already-built Step 3C features, so its time-safety rests on the upstream
   feature builders being pre-fight only). The ledger is not overwritten
   without `--overwrite`.
16. **Forward live prediction ledger (Step 6B)** —
   `ufc_pipeline/step6b_live_predictions.py`: the forward live-tracking layer.
   The user supplies a CSV of *genuinely upcoming* pre-fight feature rows; the
   system reproduces the official model (base LR on the official train split,
   Platt on the official calibration split), validates the input, generates
   calibrated `P(fighter_a wins)`, and appends predictions to a durable
   append-only ledger (`data/live/live_predictions.csv`) as `pending`. Later a
   results CSV resolves the pending rows and scores them; a live model report
   aggregates the resolved rows. Three commands / entry points:
   `run_live_predictions`, `resolve_predictions`, `build_live_model_report`.
   Reports go to `reports/live/` (`live_prediction_batch_<id>.{json,md}`,
   `live_resolution_<ts>.json`, `live_model_report.{json,md}`). Unlike Step 6A
   (which replays historical events that already have outcomes), Step 6B
   predicts fights BEFORE they happen and scores them only after results
   arrive. It is append-only (no silent overwrite; duplicates rejected unless
   `--allow-duplicate-predictions`, existing pending replaced only with
   `--overwrite-existing-pending`), reuses the official config as-is, never
   retunes/replaces the model, and writes only `data/live/` and `reports/live/`.
   It is NOT statistical validation until enough live predictions resolve: the
   report loudly warns at < 50 resolved (too noisy) and < 100 (preliminary).
   No scraper is included — the user prepares the feature CSV manually; a future
   Step 6C could add a data-source updater / scraper (still no odds).
17. **Upcoming-card feature builder (Step 6C)** —
   `ufc_pipeline/step6c_upcoming_feature_builder.py`: turns a simple matchup CSV
   into the model-ready Step 3C feature CSV Step 6B expects, closing the manual
   gap. Input needs only `event_date, event_name, fighter_a, fighter_b` (+
   optional `weight_class`). It is a *direct recomputation*, not a lossy bridge:
   each upcoming bout is appended to the historical fight list (from `data/ufc.db`)
   as a synthetic, result-less fight and the exact Step 3/3B/3C builders
   (`features.build_feature_rows`, `stats_features.build_step3b_rows`,
   `matchup_features.build_against_rows` + `add_matchup_features`) are run, with
   Elo-going-in obtained by running the deterministic Elo engine over the
   strictly-earlier fights. Because those builders read pre-fight state from
   prior fights only, the synthetic row reproduces exactly what a real bout's
   features would be — verified to match the processed file to `|Δ|=0.0`.
   Fighter names are matched deterministically (accent/punct-normalized, exact
   only); unmatched or ambiguous names become reported failed rows, never
   guessed; debut/low-history fighters get training's null policy +
   `no_prior_stats` flags. Output goes to `data/live/` (or an explicit `--output`)
   and a build report to `reports/live/step6c_feature_build_<ts>.{json,md}`; it
   validates the output against Step 6B and refuses to write an invalid file. It
   does NOT scrape and adds no odds/markets; it reads the existing history DB
   only and never writes `data/processed/` or the benchmark file.

Official LR baseline as of 2026-07-07:
**step5c_stronger_regularized_lr_drop_weight_class_platt**. This is Step 3C
numeric features with raw `weight_class` dropped, no division interactions,
logistic regression with elastic-net (`C=0.003`, `l1_ratio=0.1`, `solver=saga`)
and Platt calibration. New official benchmark metrics: held-out test
re-report log loss **0.641920**, Brier **0.225595**, AUC **0.679135**,
accuracy **0.628705**. The old official **Step 3C LR + Platt** remains a
historical baseline (log loss 0.6442244532062779, Brier 0.226717, AUC
0.675968, accuracy 0.619345), and Step 3B + Platt (0.6449) remains the
fallback benchmark. `benchmarks/official_baseline.json` is the fixed
benchmark reference: a candidate must beat the current official model on log
loss first (then Brier, then calibration gaps) on the same chronological
split before the official model changes; never regenerate that file without
explicit sign-off. Because the final held-out test was touched in prior
diagnostics, future claims should emphasize validation stability plus
consistency with the held-out re-report, not a fresh untouched test.

Step 5A (Run 1) found the age and experience groups most valuable, most
groups within the +/-0.002 log-loss noise band, and one borderline win
(dropping `weight_class`, -0.0026 on the *test* set only).

Step 5B (Run 2) tested that hypothesis validation-first. Findings: the
official anchor reproduced exactly (0.6442244532062779); the best finalist
(elastic-net C=0.003, l1_ratio=0.1, all `weight_class` dropped) reached test
log loss 0.6419, +0.0023 over official — *just* past the 0.002 threshold.
But dropping `weight_class` is **not** confirmed on pre-test validation
(sub-noise, ~0.0003); the robust, validation-confirmed lever is **stronger
regularization** than the default C=1.0 (~0.011 validation improvement), and
L1/elastic-net zero out the `weight_class` one-hots on their own. saga
convergence was clean (no fits hit max_iter=5000). At the end of Run 2 the
official model was left unchanged: the margin was thin and the weight_class
win had not reproduced on held-out validation.

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
validation (far below 0.002). At the end of Run 2.1 the official model
remained unchanged pending Run 3, but the stronger-regularized candidate was
locked for follow-up.

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
test. **Conclusion:** reject/keep division interactions experimental; promote
the stronger-regularized no-interaction LR after explicit sign-off and record
the old Step 3C LR + Platt as a historical baseline. Run 4 should compare
nonlinear models and ensembles against both the old Step 3C baseline and the
new stronger-regularized LR baseline; odds/markets remain out unless
explicitly approved later.

Step 5D / Run 4 tested nonlinear models and a blend against the official LR.
Both LR references reproduced the benchmark **exactly** (official anchor test
log loss 0.6419199384228512, historical Step 3C 0.6442244532062779, |Δ|=0.0).
The best `HistGradientBoosting` candidate (drop `weight_class`, lr=0.03,
max_leaf_nodes=15, min_samples_leaf=100, l2=0.0) **lost** to the official LR
on validation (mean Platt log loss 0.66164 vs official 0.65847, −0.00317, only
2/5 folds) and lost badly on the already-touched final test (0.665388 vs
0.641920). HGB did not even improve AUC (val 0.6230 vs official 0.6320; test
0.6606 vs 0.6791), so there was no "better ranking, worse calibration"
trade-off — it was simply worse. Isotonic hurt HGB (val 0.6855 vs Platt
0.6616); **Platt is the right calibrator** for it. Including raw `weight_class`
in HGB was sub-noise, consistent with prior runs. The validation-tuned convex
blend (w=0.6 on LR, 0.4 on HGB) *did* beat both on **validation** (mean log
loss 0.65548, +0.00292 vs official), but the edge **did not survive** the
final-test re-report (blend test log loss 0.643676 > official 0.641920), so it
is **not** a promotion candidate. Permutation importance (calibration window
only; HGB has no native `feature_importances_`) put `age_diff` and `elo_diff`
on top, overlapping the LR-important set. **Conclusion:** no nonlinear model or
blend cleared the bar (validation edge >0.002 + fold consistency + consistent
final-test re-report). **Keep the stronger-regularized LR as official;** the
recommended next phase is live tracking/monitoring of that model (rolling Platt
recalibration, drift/calibration monitoring), not a bigger model zoo. The
official model and `benchmarks/official_baseline.json` are unchanged.

Step 5D.1 / Run 4.1 audited that surprising Run 4 result before accepting it,
and found **no bug**. All anchors reproduced to |Δ|=0.0 (official 0.641920,
historical 0.6442244532062779, Run 4 best HGB 0.665388). Probability
orientation is correct for both LR and HGB (`classes_=[0,1]`, predicted
probability = P(fighter A wins); `LL(p) ≪ LL(1−p)`, `AUC(p) ≈ 0.66–0.68` vs
`AUC(1−p) ≈ 0.32–0.34`). LR and HGB consume identical rows/targets/features
across every split and fold (5983/1282/1282; 43 features; identical NaN
counts). Calibration is correct — AUC is invariant under Platt, and although
HGB's *uncalibrated* validation LL (0.681) beats the heavily-regularized LR's
raw LL (0.706), Platt fixes the LR (0.658) so it edges HGB (0.662); isotonic
is worse for both. HGB handles the 22.5% missing training cells natively (no
imputer), which is intended. An **expanded staged HGB search (83 configs, ~3.5×
Run 4)** — coarse spanning grid then local refinement, plus a max_iter /
early-stopping sub-study — found its best in the very shallow (leaf=3),
strongly-regularized region at validation LL 0.658343 vs official 0.658466: a
**+0.00012 tie, far inside the 0.002 noise band**, i.e. HGB never beats the LR
on validation. Early stopping is isolated (internal validation from the fit
window; effective n_iter 153 < cap). The Run 4 blend reproduced (validation
0.655548, test 0.643676) but is **not robust**: per-fold optimal weights swing
0.1→0.8 (range 0.70), so the validation edge is fold-noise, consistent with its
final-test failure. **Conclusion: the Run 4 result is real — HGB genuinely
loses because the signal in these 43 features is essentially linear and a
well-regularized LR is already at the frontier. Keep the stronger-regularized
LR official and close Run 4; next phase is live tracking/monitoring.** The
official model and `benchmarks/official_baseline.json` remain unchanged.

Step 6A (pseudo-live replay) is the first step of that live-tracking phase — a
workflow-validation and calibration/drift tool, not more model-zoo work. On the
real data it replayed **360 events / 4,260 fights** (first qualifying event UFC
217, 2017-11-04, through 2026-05-16; 414 earlier events skipped for
insufficient history) with `min_train_fights=3000`, `calibration_fights=1282`.
Overall pseudo-live log loss **0.6516** (Brier 0.2299, AUC 0.6456, acc 0.6185)
is +0.0097 above the static backtest 0.6419 — expected, because the replay era
is broader and early years (2017-2019, ~0.667) drag the aggregate while recent
years converge toward the backtest (2022 0.632, 2025 0.627). Calibration is
strong (all bucket gaps within ±0.021; high-confidence two-sided gap −0.0068;
confidence-bucket accuracy monotonic 0.55→0.86). All 11 leakage/protocol checks
pass. Worst events are upset-heavy cards, best are chalk cards (single-event
noise). The official model is **unchanged**; these numbers must never be used to
select or retune a model (the final test was already touched). **Recommended
next phase: a forward live-prediction ledger and/or a data-source updater — NOT
another model-zoo search, and no odds/markets unless explicitly approved.**

Step 6B (forward live prediction ledger) implements that live-tracking layer.
Expected upcoming-card input CSV: identifier columns `event_date, event_name,
fighter_a, fighter_b` **plus every official model feature column** (the 43 Step
3C numeric features, e.g. `elo_diff, age_diff, reach_diff,
fighter_a_expected_win_prob, …`) and **no** result/target/odds columns
(`fighter_a_won`, `winner`, `method`, `round`, odds/market names are rejected;
raw `weight_class` is ignored if present). The ledger lives at
`data/live/live_predictions.csv` (created on first run) with `prediction_mode =
live_forward` and `status ∈ {pending, resolved, error}`; a deterministic
`prediction_id` (hash of date + fighter set + model_version + batch_id) makes
duplicate/versioned rows explicit. Resolution matches a results CSV
(`event_date, event_name, fighter_a, fighter_b, winner`) by exact normalized
`(date, fighter set)` — fighter order may differ — fills `target_a_win`, and
computes per-fight log loss / Brier / correctness. The live model report
compares to the static backtest (0.641920) and the Step 6A pseudo-live
reference (0.651582) but treats the live sample as too noisy below 50 resolved
and preliminary below 100. The official model and
`benchmarks/official_baseline.json` are unchanged, and nothing is written under
`data/processed/`.

Step 6C (upcoming-card feature builder) removes the manual-CSV gap. Matchup
input schema: `event_date, event_name, fighter_a, fighter_b` (+ optional
`weight_class, scheduled_rounds, fight_order, notes`, the last two report-only).
It emits `[event_date, event_name, fighter_a, fighter_b, weight_class]` + the 43
Step 3C model feature columns and no result/target/odds columns. On real data it
reproduced a historical event's features to `|Δ|=0.0` against the processed file,
built 12/12 rows for UFC 302 with Step 6B validation PASS, and the full
build → predict → resolve chain runs through the CLIs. Time-safety is asserted
per matchup (max historical fight date used strictly `< event_date`); the 2024
Jones fight is correctly excluded when the event date is earlier. Next likely
step is **Step 6D: a data-source updater / UFCStats scraper** (only after Step 6C
is verified in production use) — still no sportsbook odds or prediction markets
unless explicitly approved.

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
# Step 5D nonlinear models + blend (writes reports/step5d_*; ~1 min full run,
# --quick for a fast smoke grid, --skip-slow to skip permutation importance,
# --skip-blend to skip the blend, --max-candidates N to cap the grid)
.venv/bin/python scripts/run_step5d_nonlinear_models.py --output-dir reports
# Step 5D.1 nonlinear sanity audit (writes reports/step5d1_*; ~2.5 min full run,
# --quick for a fast staged grid, --skip-slow to skip permutation importance)
.venv/bin/python scripts/run_step5d1_nonlinear_sanity_audit.py --output-dir reports
# Step 6A pseudo-live replay (event-by-event; writes reports/step6a_* + reports/replay/ ledger;
# ~40s full run; --max-events N for a quick smoke; ledger needs --overwrite to replace)
.venv/bin/python scripts/run_step6a_pseudo_live_replay.py --input data/processed/ufc_prefight_features_step3c.csv \
    --output-dir reports --min-train-fights 3000 --calibration-fights 1282 --overwrite
# Step 6B forward live prediction ledger (upcoming-card CSV -> predictions -> resolve -> live report;
# writes data/live/ + reports/live/; append-only, needs flags to duplicate/overwrite/re-resolve)
.venv/bin/python scripts/run_step6b_live_predictions.py --input upcoming_card_features.csv \
    --ledger data/live/live_predictions.csv --output-dir reports/live
.venv/bin/python scripts/resolve_step6b_live_predictions.py --results completed_results.csv \
    --ledger data/live/live_predictions.csv --output-dir reports/live
.venv/bin/python scripts/run_step6b_live_model_report.py --ledger data/live/live_predictions.csv --output-dir reports/live
# Step 6C upcoming-card feature builder (matchup CSV -> Step 3C feature CSV for Step 6B; reads data/ufc.db,
# no scraping/odds; writes data/live/ + reports/live/; needs --overwrite to replace output)
.venv/bin/python scripts/build_step6c_upcoming_features.py --matchups data/live/upcoming_card_matchups.csv \
    --output data/live/upcoming_card_features.csv --validate-for-step6b --overwrite
# then feed the built features into Step 6B (or use --run-predictions to chain automatically):
.venv/bin/python scripts/run_step6b_live_predictions.py --input data/live/upcoming_card_features.csv \
    --ledger data/live/live_predictions.csv --output-dir reports/live
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
| `ufc_pipeline/step5d_nonlinear_models.py` | Step 5D: HistGradientBoosting grid + LR/HGB blend vs official LR, Platt/isotonic calibration, validation-only selection, finalist test re-report, permutation importance |
| `ufc_pipeline/step5d1_nonlinear_sanity_audit.py` | Step 5D.1: diagnostics-only audit of the Run 4 result — anchor/HGB reproduction, probability orientation, row-feature parity, calibration protocol, NaN handling, staged HGB search, early-stopping + blend stability audits |
| `ufc_pipeline/step6a_pseudo_live_replay.py` | Step 6A: event-by-event pseudo-live replay of the official model — per-event refit+Platt on strictly-earlier fights, replay ledger, rolling/calibration/drift/leakage reporting (pipeline-validation tool, never changes the model) |
| `ufc_pipeline/step6b_live_predictions.py` | Step 6B: forward live prediction ledger — validate upcoming-card features, predict with the official model, append-only `data/live/` ledger, result resolution, live model report with small-sample warnings (never changes the model) |
| `ufc_pipeline/step6c_upcoming_feature_builder.py` | Step 6C: matchup CSV → Step 3C feature CSV for Step 6B — synthetic-fight direct recomputation via the real builders (time-safe < event_date), deterministic name matching, debut/unmatched/ambiguous handling, Step 6B output validation, build report (no scraping, no odds) |
| `benchmarks/official_baseline.json` | fixed official-model benchmark reference (never regenerate casually) |
| `scripts/` | one CLI wrapper per pipeline stage |
| `docs/greco_field_audit.md` | source-overlap rules (mdabbert vs Greco) |
