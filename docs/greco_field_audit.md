# Greco1899 expansion: field inventory & source audit

Date: 2026-07-04. Sources compared: **mdabbert Ultimate UFC Dataset**
(Kaggle; the project's preferred fights base) and
**Greco1899/scrape_ufc_stats** (GitHub; scraped from ufcstats.com; files:
`ufc_fight_results.csv`, `ufc_event_details.csv`, `ufc_fight_stats.csv`,
`ufc_fighter_tott.csv`).

> **Environment note (important):** the real mdabbert Kaggle CSV is not
> present in this environment (Kaggle is not reachable from the sandbox; the
> only mdabbert-format file here is the synthetic smoke-test sample). The
> real Greco files WERE downloaded and used. Therefore this audit is
> *structural* (definitions, units, timing, coverage), and the central
> database in this run is built from Greco fights. Row-level disagreement
> examples for overlapping fields require the real mdabbert file — the
> canonical-source rules below are implemented so that when it is added,
> mdabbert wins where specified with no code changes.

## What the central database currently stores (and from where)

| table.column | populated by | notes |
|---|---|---|
| fights.date/event/fighters/winner/weight_class/method/round | fights base source (mdabbert preferred; greco adapter now available) | identity + result |
| fights.fighter_x_age/height/reach | mdabbert per-fight tale-of-the-tape (RedAge, RedHeightCms, RedReachCms...) | as-of-fight values |
| fighters.height/reach/stance/date_of_birth | **NEW: Greco `ufc_fighter_tott.csv`** | static per-fighter values, cm / ISO dates |
| fight_stats.* (+ NEW `source` column) | **Greco `ufc_fight_stats.csv` (+ results/events for minutes/dates)** | per-fight offensive totals; post-fight facts, used only for later fights' rolling features |
| fight_elo_snapshots | derived (Elo engine) | |

## Category A — useful, non-overlapping Greco fields (ADDED)

Per-fight totals mdabbert does not provide, stored in `fight_stats` with
`source='greco'`: significant strikes landed & attempted, takedowns landed &
attempted, knockdowns, submission attempts, control time (seconds), fight
minutes (derived from finish round + time + round format). Absorbed strikes,
opponent attempts, and takedowns allowed are represented by the *opponent's
row* of the same fight (normalized storage; the rolling builder joins the
two rows), so no separate absorbed columns are stored — same information,
no duplication.

Also added: `fighters.date_of_birth/height/reach/stance` from
`ufc_fighter_tott.csv` (fill-only; see Category B).

Ignored from the stats file for now: per-target/position splits (HEAD, BODY,
LEG, DISTANCE, CLINCH, GROUND), TOTAL STR., REV., SIG.STR. % / TD % (the
percentages are derivable; the splits are a future feature idea, not needed
for the Step 3B metrics).

## Category B — overlapping fields: audit + deterministic canonical choice

| field | mdabbert | Greco | canonical rule (implemented) | why |
|---|---|---|---|---|
| age | per-fight `RedAge`/`BlueAge` (age at fight; known pre-fight) | static `DOB` | **per-fight age if present, else DOB-derived age at fight date** (`features.py` fallback) | per-fight value is exactly the as-of quantity; DOB derivation is time-safe and fills gaps |
| height / reach | per-fight `...Cms` columns (cm) | static tott (feet+inches / inches → converted to cm) | **per-fight value if present, else static** (SQL `COALESCE` in the feature query) | per-fight listing can reflect updates; near-static anyway; units unified to cm |
| stance | sometimes present | tott stance | **fill-only into `fighters.stance`** (not yet a model feature) | no conflict handling needed until it becomes a feature |
| sig strikes / takedowns / control / KD / sub attempts | only as **precomputed career averages** (`RedAvgSigStrLanded`, ...) — as-of semantics unverifiable | **raw per-fight totals** | **Greco raw totals only; mdabbert averages remain ignored** (dropped at the adapter since Part 1) | we rebuild rolling averages ourselves with provable time safety; unverifiable precomputed averages are a leakage risk |
| fight result fields (winner/method/round/weight class) | present | present (`OUTCOME`, `METHOD`, `ROUND`, `WEIGHTCLASS`) | **whichever source is the fights base for a given DB**; result fields are metadata/labels, never features | identical concepts; no merging across sources to avoid double rows |

**No overwriting, no averaging:** tott ingestion only fills NULL fighter
fields (`WHERE column IS NULL`), per-fight mdabbert values always take
precedence via COALESCE, and `fight_stats.source` records provenance.
Re-ingesting stats replaces rows *from the same pipeline deterministically*
(INSERT OR REPLACE keyed by fight+fighter), never silently mixing sources.

## Category C — dangerous/forbidden (unchanged policy)

Betting odds (mdabbert `RedOdds`...), rankings (`RMatchWCRank`...),
winner/loser-derived features, method/round as inputs, current-fight stats
in the same fight's feature row, and precomputed career averages remain
forbidden as model inputs, enforced by the leakage guards in
`ufc_pipeline/modeling.py` and the feature builders.

## Known matching limitations (documented, not hacked around)

* Fighters are keyed by case-insensitive exact name. Distinct fighters who
  share a full name (a handful in UFC history) merge into one record; fixing
  this properly needs URL-based fighter IDs from Greco — a future
  reconciliation layer, noted in the README.
* When fights come from mdabbert and stats from Greco, cross-source name
  spelling differences (accents, "Jr.") appear as unmatched stats rows and
  are *reported*, never guessed.
* The fights base must be ONE source per database build (mdabbert preferred
  when available; Greco otherwise). Ingesting both as fights would create
  duplicate bouts, because duplicate detection is per-ingest-run.
