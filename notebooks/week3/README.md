# Week 3: age, weakly used hands and per-finger tremor

Task (sheet column F): basic statistics by age (10-year histogram), healthy vs sick; count the sick participants who barely use one hand; propose a per-finger tremor analysis.

| File | What it is |
|---|---|
| `week3_age_hands_fingers.ipynb` | **English version (main).** Answers, figures, checks and limitations. |
| `week3_analysis.py` | All calculations and figures for the English version (`python week3_analysis.py`, ~5 s). |
| `figures/` | Eight presentation figures (`01_age_histogram` … `08_peak_frequency_check`). |
| `outputs/` | One CSV per table plus `summary.json` with the headline numbers. |
| `analyza_prstov.ipynb` | Original Slovak version by D. Sholtys, kept unchanged. |

## Inputs

- Per-finger pipeline tables in `<repo>/outputs/` (`person_summary.csv`, `features_by_person_finger.csv`, `features_by_attempt_finger.csv`, `sensor_noise.csv`), built by `scripts/build_dataset.py` from `src/rehapiano`.
- Week-2 questionnaire labels and window flags: `notebooks/week2/outputs/participant_labels.csv`, `hand_usage.csv`.
- Raw recordings (only for the two example figures 03 and 08): `data/raw/RehaPiano/` (see `data/README.md`).

## Headline answers

- **Age:** the healthy median is 23 years and the sick median 66.5; only 17 healthy people are aged 50+. Age, recording period and game step all coincide with the group.
- **Weak hand:** 7 of 48 sick participants are below the healthy lower limit of hand symmetry (weaker ÷ stronger hand force < 0.62); resampling gives a range of 6–10. No hand is completely silent, but 0IKB-27Y1 effectively plays with one hand.
- **Fingers:** thumb and little finger oscillate most and the ring finger least, in both groups and also at equal force. In people aged 50+, every finger is 1.2–2.0× higher in the sick group (9/10 significant after Holm correction), with almost the same finger pattern. Healthy amplitude does not rise with age, so age alone does not explain the difference.
- **Peak frequency:** the reported ~5 Hz "tremor peak" mostly reflects the falling spectrum, not a tremor rhythm.

## Differences from the Slovak version

The healthy group here is the 76 questionnaire-screened controls, not all 90. The age-matched reference (healthy, 50+) is scored leave-one-out. Fingers are also compared at equal force, and the age, ratio-measure and peak-frequency checks are new. See the last section of the notebook for the full comparison table.
