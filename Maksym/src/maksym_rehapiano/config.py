"""Settings of the rhythm-test analysis (weeks 4-5). Shared settings stay in ``rehapiano.config``."""

from __future__ import annotations

from pathlib import Path

MAKSYM_ROOT = Path(__file__).resolve().parents[2]
WEEK4_OUTPUT_DIR = MAKSYM_ROOT / "week4" / "outputs"
WEEK5_OUTPUT_DIR = MAKSYM_ROOT / "week5" / "outputs"
# Questionnaire screening of the healthy group (copy of notebooks/week2/outputs/participant_labels.csv
# from M. Holovcyk's branch `mykyta`); eligible_comparison = healthy without a reported condition.
PARTICIPANT_LABELS_CSV = MAKSYM_ROOT / "data" / "participant_labels.csv"

# --------------------------------------------------------------------------- rhythm test
# RHYTHM: 4 attempts of 31 s; both hands squeeze and release as fast as possible.
# As in EXACT_FORCE, controls and patients played different step_ids of the game.
RHYTHM_STEP_TYPE = "RHYTHM"
RHYTHM_CONTROL_STEP_ID = "1e177cdd-0baf-42fe-b297-fbeba73ea828"
RHYTHM_PATIENT_STEP_ID = "65a5a4b1-907c-4717-b47b-cd8f31eb3e4b"
RHYTHM_STEP_IDS = (RHYTHM_CONTROL_STEP_ID, RHYTHM_PATIENT_STEP_ID)
RHYTHM_CROP_START_S = 2.0  # the first presses start after ~1 s
RHYTHM_CROP_END_S = 30.0
# press_offset / release_offset in additional_data are not aligned with the force signal
# (they fall on peaks in some recordings and on troughs in others), so presses are
# detected from the signal itself: peaks of the summed force of both hands.
RHYTHM_MIN_CYCLE_S = 0.25  # presses closer than this are one press
RHYTHM_MIN_PROMINENCE = 0.02  # absolute minimum peak height above the neighbouring troughs
RHYTHM_RELATIVE_PROMINENCE = 0.25  # ... and at least this share of the attempt's 5-95 % force range
RHYTHM_MIN_CYCLES = 5  # fewer detected presses: the attempt is not rhythmic, not analysed

# Age categories: 10-year bins as in the week 3 histogram. Healthy people aged 50+ are few
# (24), so many bins are small; a bin is tested only when both groups have at least
# RHYTHM_MIN_GROUP_SIZE people.
AGE_CATEGORY_EDGES = tuple(range(10, 100, 10))
AGE_CATEGORY_LABELS = tuple(f"{low}–{low + 9}" for low in AGE_CATEGORY_EDGES[:-1])
RHYTHM_MIN_GROUP_SIZE = 3
