# Unified RehaPiano pipeline

One command turns the raw recordings into one feature table per person (EXACT_FORCE and RHYTHM games), and then into the week-6 statistics, decade norms, classifier and ablation. It replaces the separate loading code of the weekly notebooks; the weekly notebooks stay as they are.

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/run_pipeline.py                      # everything (~2 min first time, ~1 min afterwards)
.venv/bin/python scripts/run_pipeline.py --stages analysis report
.venv/bin/python -m unittest discover tests
```

## New data from the teacher

1. Copy the new `results_*` folders anywhere under `data/raw/RehaPiano/`.
2. Copy the new metadata CSV(s) into `data/metadata/` (all `*.csv` there are read; exact duplicate rows are merged). If the new patients come in a new file, name it on the command line: `--patient-files dataset_patients.csv new_patients.csv`.
3. Update `data/sources/questionnaire.csv` if new controls answered the questionnaire.
4. Run `scripts/run_pipeline.py`. Only files that are new or changed are read and processed; everything else comes from the caches.
5. Check `outputs/pipeline/recordings.csv` (status and reason of every file), `steps_not_analysed.csv` (game steps that the analysis ignores; a new step of a known game must be added to `GAMES` in `src/rehapiano/pipeline/settings.py`) and `run_summary.json`.

## Stages

| Stage | Code | Output (`outputs/pipeline/`) |
|---|---|---|
| index | `pipeline/index.py` | `recordings.csv`: every file, its game, `used` / `copy` / `excluded` and why; `steps_not_analysed.csv` (merged metadata goes to the uncommitted `cache/`) |
| participants | `pipeline/participants.py` | `participants.csv`: label (Healthy / Sick / Other control), questionnaire status, age, decade, recording period |
| features | `pipeline/extract.py`, `features/exact_force_game.py`, `features/rhythm_game.py` | `ef_attempt_finger.csv`, `ef_attempt_hand.csv`, `rhythm_attempt.csv`, `extraction_errors.csv` |
| aggregate | `pipeline/aggregate.py` | `person_features.csv` (one row per person), `person_finger.csv` |
| analysis | `pipeline/analysis.py` | `analysis/*.csv`: hypotheses, age trends, decade norms, classifier, ablation |
| report | `pipeline/report.py` | `analysis/figures/01–04*.png` |

All choices (game steps, crop windows, bands, thresholds, CV settings) are in `src/rehapiano/pipeline/settings.py`; the feature list and its factors are in `pipeline/catalog.py`.

## Preprocessing (shared with week 3)

Decoding, channel-9 polarity, timestamp checks, averaging of duplicate timestamps, interpolation of gaps ≤ 50 ms, splitting at longer gaps, anti-aliased resampling to 100 Hz, cropping (EXACT_FORCE 3–15 s, RHYTHM 2–30 s): `rehapiano.preprocessing.pipeline.load_measurement_data`. Copies of the same recording in several folders are collapsed when byte-identical and excluded when they differ.

## Features (one value per person)

Aggregation: attempt → session (median over attempts) → person (mean over sessions). Missing evidence stays missing.

| Feature | Factor | Definition |
|---|---|---|
| `ef_tremor_amp`, `ef_tremor_amp_max` | EF fast oscillation | mean spectrogram amplitude 4–15 Hz of active fingers; mean of hands / larger hand |
| `ef_slow_amp` | EF slow force control | same, 0.2–3 Hz |
| `ef_tremor_share` | EF fast-to-slow share | 4–15 / (0.2–3 + 4–15) |
| `ef_balance`, `ef_tremor_balance` | EF hand balance | weaker ÷ stronger hand: mean finger force / 4–15 Hz amplitude |
| `rh_press_rate_hz` | Rhythm tempo | presses per second (attempts 2–4; attempt 1 is practice) |
| `rh_interval_cv` | Rhythm regularity | SD / mean of intervals between presses |
| `rh_amplitude_strong`, `rh_amplitude_weak` | Rhythm press force | median press amplitude of the stronger / weaker hand |
| `rh_balance_range`, `rh_balance_press` | Rhythm hand balance | weaker ÷ stronger hand: force range (95th−5th pct) / press amplitude |
| `rh_amplitude_cv` | Rhythm force stability | SD / mean of press amplitudes |
| `rh_sync_lag_ms` | Rhythm hand synchrony | median lag between the left and right maxima within a press |
| `rh_release_ratio` | Rhythm release | force at the minimum after a press ÷ peak force |
| `rh_fatigue_amplitude`, `rh_fatigue_rate` | Rhythm fatigue | 2nd half ÷ 1st half of the attempt |

The EXACT_FORCE finger values are identical to `outputs/features_by_attempt_finger.csv` (week 3). Week-3 person values differ slightly because week 3 averages attempts after removing outliers, while the pipeline takes the median.

## Analysis (task of week 6)

* **Hypotheses**: H0 equal distributions in healthy and sick people, H1 different (two-sided). Mann–Whitney U for all ages and for age 50+, plus a permutation test with labels shuffled only within 10-year groups (age cannot create the difference). Holm correction over the 17 features.
* **Healthy people and age**: Spearman ρ with age (Holm), change per 10 years with a 95 % CI.
* **Norms by decade**: median and 2.5–97.5 % range. A decade gets its own empirical range only with ≥ 20 healthy people; otherwise the pooled healthy range is used (no age trend) or a linear quantile regression on age (age trend).
* **Classifier**: logistic regression (median imputation, standardisation, balanced classes; amplitudes log-transformed) and random forest; 5-fold stratified CV repeated 20×, split by person; permutation test of the AUC. Feature sets: EXACT_FORCE only, rhythm only, all, all + age, age only; also for age 50+.
* **Ablation**: AUC without each factor and with each factor alone, with and without age.

## Limitations to keep in every report

Group membership coincides with age, recording period and game step (healthy and sick people played different step versions of both games); "healthy" is questionnaire-screened; diagnoses, severity and medication are unknown; force is in uncalibrated device units; the finger mapping is inferred, not verified; the healthy reference aged 50+ has only 17 people. A classifier result is cohort-membership prediction, not a diagnosis.
