# Week 5, short version (Mykyta): hand balance by 10-year age group

Task (sheet column G): evaluate hand-force balance also in the rhythm test, compare with EXACT_FORCE, analyse by age categories and separate the effect of age from the illness.

| File | What it is |
|---|---|
| `week5_hand_balance_by_decade.ipynb` | Short English notebook: method, figures, answers, limitations |
| `week5_analysis.py` | All calculations and figures (`python week5_analysis.py`; add `--recompute` to rebuild the rhythm table from raw data, ~30 s) |
| `figures/` | `01_rhythm_examples`, `02_balance_by_decade`, `03_weak_hand_share_by_decade` |
| `outputs/` | `people.csv` (one row per person, both games), `rhythm_attempts.csv`, decade summary, tests, `summary.json` |

Differences from Maksym's `../rovnovaha_ruk.ipynb`:
- 10-year age groups instead of < 50 / 50–64 / ≥ 65;
- 76 questionnaire-screened healthy people as the reference (Maksym: all 90 controls);
- rhythm balance from each hand's force range (95th − 5th percentile), with no press detection; it works for people who alternate hands, and it agrees with Maksym's press-amplitude measure (ρ = 0.90);
- age vs illness tested by shuffling labels within each 10-year group, instead of a linear model.

Inputs: raw recordings via `outputs/recordings_inventory.csv` (rhythm steps `1e177cdd…` healthy, `65a5a4b1…` patients), EXACT_FORCE balance from `outputs/stats/hand_activity.csv`, labels from `notebooks/week2/outputs/participant_labels.csv`.
