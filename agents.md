# Agent Instructions

This repository is a Python data pipeline for UFC fight ingestion, time-safe Elo snapshots, pre-fight feature generation, rolling fight-stat features, model evaluation, and probability calibration.

## Setup

- Use `python3`; `python` may not exist on all machines.
- A project virtualenv lives at `.venv/` (created with `python3 -m venv --without-pip .venv` plus get-pip.py on machines whose system Python lacks `ensurepip`). Prefer `.venv/bin/python` when it exists.
- Install runtime/test dependencies in a virtual environment when needed:
  `python3 -m pip install pandas numpy scikit-learn joblib pytest`
- There is currently no `pyproject.toml` or requirements file, so do not assume packaging metadata exists.

## Common Commands

- Run tests: `python3 -m pytest tests/ -q`
- Register fighter identities (run BEFORE ingesting fights so same-name fighters split): `python3 scripts/build_fighter_identities.py --tott data/raw/greco/ufc_fighter_tott.csv --db data/ufc.db --report data/processed/fighter_identity_report.json`
- Ingest fights: `python3 scripts/ingest_fights.py --input data/raw/mdabbert_ufc.csv --db data/ufc.db --source mdabbert`
- Build Elo: `python3 scripts/build_elo.py --db data/ufc.db`
- Export Elo baseline: `python3 scripts/export_elo_dataset.py --db data/ufc.db --output data/processed/ufc_elo_baseline.csv --ratings-output data/processed/current_fighter_elos.csv`
- Build Step 3 features: `python3 scripts/build_prefight_features.py --db data/ufc.db --output data/processed/ufc_prefight_features.csv`
- Build Step 3B features: `python3 scripts/build_prefight_features_step3b.py --db data/ufc.db --output data/processed/ufc_prefight_features_step3b.csv`
- Train/evaluate models: `python3 scripts/train_model.py --input data/processed/ufc_prefight_features.csv`
- Compare Step 3 vs Step 3B: `python3 scripts/compare_models.py --input data/processed/ufc_prefight_features_step3b.csv`
- Calibrate the best Step 3B model, backtest mode (Platt + isotonic, honest held-out test): `python3 scripts/calibrate_model.py --input data/processed/ufc_prefight_features_step3b.csv --output-dir data/processed --model-dir data/models` (optional `--train-frac/--calibration-frac` or `--train-end-date/--calibration-end-date`)
- Rolling/live calibration for future predictions (refits Platt on the most recent window; diagnostics only, no held-out test): `python3 scripts/calibrate_model.py --input data/processed/ufc_prefight_features_step3b.csv --output-dir data/processed --model-dir data/models --mode live --calibration-window-days 730 --method platt` (optional `--as-of-date`, `--min-calibration-rows`)

## Project Invariants

- Preserve time safety. Elo, record features, recent-form features, and rolling stat features must read historical state before applying the current fight result/stats.
- Sort fights by `(date, fight_id)` when building chronological outputs. `fight_id` preserves source order for same-date tie-breaking.
- Do not let current-fight outcome stats, odds, rankings, post-fight Elo, winner/loser-derived feature names, or precomputed averages become model inputs.
- Keep result/metadata columns such as `winner`, `fighter_a_won`, `method`, and `round` available as labels/metadata, but never use them as pre-fight model features.
- Treat missing historical information as missing/null unless a zero is an exact count, such as prior wins for a debut.
- Model evaluation splits are chronological, never random. Backtest calibration uses a 3-way chronological split (train → calibration → test): the base model fits only on the train window, calibrators (Platt/isotonic) fit only on the calibration window, and reported metrics come only from the final test window. Never fit a calibrator on the final test set.
- Live/rolling calibration (`--mode live`) refits the calibrator on the most recent completed fights (default 730-day window) for FUTURE predictions. Its diagnostics are calibration-window numbers, in-sample for the calibrator — never present them as final-test metrics. Fights after `--as-of-date` must be excluded from training and calibration.
- Platt is the default calibration method; switch to isotonic only when the data clearly proves it better on held-out log loss, and never on a small calibration window.
- Calibration is a separate evaluation layer on top of the Step 3B logistic regression (`ufc_pipeline/calibration.py`); do not replace or overwrite the `compare_models.py` comparison outputs when adding calibrated variants. Live-mode outputs use `live_*` names so backtest outputs are never overwritten.
- Prioritize log loss, Brier score, and calibration gaps when judging probability models; accuracy alone is not a success metric.
- Fighter identity: `fighters.normalized_name` is a matching key, NOT unique — different fighters can share a name. Durable identity lives in `fighter_source_ids` (UFCStats fighter URLs from Greco). Register identities (`build_fighter_identities.py`) before ingesting fights. Never silently merge same-name fighters and never guess ambiguous identities: resolution uses URL first, then unique-name match, then per-bout evidence (exact division weight → career continuity → DOB plausibility); anything still ambiguous is reported as unresolved.
- Never remap existing fighter_id values in place; same-name splits materialize only on a fresh rebuild/re-ingestion.

## Code Style

- Follow the existing small-module style in `ufc_pipeline/`; keep source-specific parsing in adapters or ingestion modules.
- Prefer pandas/SQLite structured operations over ad hoc CSV string manipulation.
- Keep tests focused on time-safety, leakage guards, deterministic ordering, and source conversion edge cases.
- Avoid rewriting generated data files under `data/processed/` unless the task explicitly requires regenerating outputs.
