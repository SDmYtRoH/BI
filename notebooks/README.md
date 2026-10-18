# Notebooks by week

Each folder answers one column of the course task sheet ("Zadania", RehaPiano row).
Run every notebook from its own folder; data paths are relative to the repository's `data/` folder (see `data/README.md`).

| Week | Task (sheet column) | Contents |
|---|---|---|
| 1 | **D** – take the updated Balanced Force (`EXACT_FORCE`) data; demonstrate Fourier analysis on one sick and one healthy person with 3, 5 and 10 s windows; make spectrograms for both. | `week1/task_1.ipynb` – Fourier and spectrogram comparison of healthy `992601` vs sick `UQIS-DXIH`. `week1/dataset_full_analysis.ipynb` – what the metadata index `dataset_full.csv` contains. |
| 2 | **E** – separate force variation from controlling the game from involuntary tremor (up to 15 Hz); compute per-person values (component amplitudes, mean frequency, SD, …); try to decide sick vs healthy from them; watch for people who use one hand little or not at all. | `week2/week2_tremor_analysis.ipynb` (+ `REPORT.html` export), code in `analysis.py` / `extensions.py`, presentation notes in `SUMMARY.md` and `GUIDE.md`, figures in `figures/`, per-person exports in `outputs/` (`DATA_DICTIONARY.md`). |
| 3 | **F** – basic statistics by age (10-year histogram), healthy vs patients; count patients who barely use one hand; design a per-finger tremor analysis. | `week3/analyza_prstov.ipynb` (Slovak, by D. Sholtys) – uses the package in `src/rehapiano`, the pipeline in `scripts/` and the tables in `outputs/`. |
| 4 | – | next task |

## Known inconsistencies between weeks

- **Healthy group.** Week 2 uses the 76 controls whose questionnaire reports no relevant condition; week 3 uses all 90 controls (7 of its 24 controls aged ≥ 50 report a condition).
- **Bands.** Week 2: 0.2–3 Hz slow, 3–4 Hz boundary, 4–15 Hz fast. Week 3: 0–4 Hz movement, 4–15 Hz tremor.
- **Low-use hand.** Week 2 flags individual 5 s hand-windows; week 3 compares mean force of the weaker and stronger hand per person.
- **Finger names** (week 3) assume sensor 1 = little finger … sensor 5 = thumb; this mapping is not verified.
