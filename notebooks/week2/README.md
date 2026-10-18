# RehaPiano — week 2 analysis

All newly authored material is in English. Original notebooks and source recordings are unchanged.

## Open these first

- `REPORT.html`: executed notebook as a readable report with embedded plots; opens without Jupyter.
- `week2_tremor_analysis.ipynb`: executable notebook with the method, tables and interpretation.
- `SUMMARY.md`: concise findings and presentation notes.
- `GUIDE.md`: step-by-step explanation and questions for your presentation.
- `outputs/participant_features.csv`: one row for every one of the 138 participants, with separate hands.

## Reproduce

Install the repository's `requirements.txt` (tested with numpy 2.0.2, pandas 2.3.3, scipy 1.13.1, matplotlib 3.9.4). Open the notebook from this folder. It reads existing derived results by default. Set `RECOMPUTE = True` to rebuild them from the raw recordings. Inputs are read from the repository's gitignored `data/` folder: recordings and indices under `data/RehaPiano/`, the questionnaire snapshot at `data/sources/questionnaire.csv` (see `data/README.md`).

Command-line alternative, from this folder:

```sh
python analysis.py            # --project defaults to ../../data
python -c 'from pathlib import Path; import extensions; extensions.enrich(Path("../../data").resolve())'
```

After recomputing, rerun the notebook to refresh its embedded results. Export it to HTML with Jupyter's HTML export, or `jupyter nbconvert --to html --no-input week2_tremor_analysis.ipynb --output REPORT.html`.

`analysis.py` handles loading, validation, signal features, participant aggregation, baseline evaluation and primary plots. `extensions.py` adds the compact participant table, transient/cutoff sensitivity, uncertainty summaries and the illustrative case.

## Inputs and scope

The source index is built from `RehaPiano/datasets/dataset_patients.csv`, `dataset_tuke.csv` and `dataset_99.csv`. Exact duplicate index rows are removed. The chosen Balanced Force scenario IDs match the earlier notebooks: `d2e6b7bd-…` for the patient cohort and `5867ebbc-…` for reference cohorts. The other reference EXACT_FORCE step (`d475bc4a-…`) is not mixed into this comparison because its equivalence is unverified.

All 173 selected sessions have local raw files. Repeated raw copies are checked by SHA-256; divergent copies would be excluded for review. `outputs/recording_manifest.csv` records their relative paths and hashes. The provided 9.94 GB archive was checked against the extracted folder: zero missing paths or file-size differences. No additional remote recording download was performed, and the latest available remote dataset could not be established from the supplied links.

`data/sources/questionnaire.csv`, `protocol.txt` and `document.txt` (not committed: the questionnaire holds self-reported health answers) are read-only snapshots retrieved from the user-provided Google links on 10 October 2026. They retain their source language. The questionnaire supplies self-reported reference status, not diagnoses. Raw recordings and questionnaire records remain local.

The introductory document mentions 70 Hz for an earlier hardware description. These files contain approximately 1,000 timestamped packets per second. The pipeline uses timestamps to create its time axis, not the older document's nominal rate; confirmation of the acquisition/firmware chain remains necessary.

## Output tables

| File | Unit of observation |
|---|---|
| `participant_features.csv` | One person; separate left/right feature columns |
| `participant_hand_features_primary.csv` | One person and one hand, primary 5 s windows |
| `participant_hand_features.csv` | One person, hand and window length (3/5/10 s) |
| `session_features.csv`, `iteration_features.csv` | Intermediate aggregation levels |
| `finger_windows.csv`, `hand_windows.csv` | Auditable individual feature calculations |
| `recording_quality.csv` | One recording repetition; timestamps, invalid values and gaps |
| `hand_usage.csv` | Primary window-level activity flags and reasons |
| `participant_labels.csv` | Cohort provenance and questionnaire screening status |
| `processing_errors.csv` | Processing failures; an empty file means no failures in this run |
| `baseline_metrics.csv`, `baseline_predictions.csv` | Participant-held-out exploratory classification |
| `late_baseline_metrics.csv` | The same baseline using only late 8–13 s windows |
| `activity_sensitivity_summary.csv` | Hand-use threshold sensitivity |
| `band_window_sensitivity.csv`, `transient_sensitivity.csv` | Frequency/window/timing sensitivity |
| `run_summary.json` | Counts, configuration, package versions and method checks |

All force amplitudes are in relative device units. No Newton/gram calibration is asserted. IDs are pseudonyms. This is an exploratory analysis of the supplied cohort, not a diagnostic system.
