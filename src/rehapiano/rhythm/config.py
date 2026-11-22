"""Settings of the RHYTHM exercise (rhythmic squeezing with both hands).

The person squeezes and releases with both hands as fast as possible: 4 attempts of 31 s.
Shared settings (paths, device, preprocessing, statistics) are in ``rehapiano.common.config``.
"""

from __future__ import annotations

from rehapiano.common.config import OUTPUT_DIR, PRACTICE_ATTEMPTS  # noqa: F401 - re-exported

# --------------------------------------------------------------------------- outputs
RHYTHM_OUTPUT_DIR = OUTPUT_DIR / "rhythm"  # attempt-level tables
HAND_BALANCE_OUTPUT_DIR = RHYTHM_OUTPUT_DIR / "hand_balance"  # week 4: balance vs EXACT_FORCE, age vs disease
ANALYSIS_OUTPUT_DIR = RHYTHM_OUTPUT_DIR / "analysis"  # week 5: hypotheses, decade norms, classifier, ablation

# --------------------------------------------------------------------------- exercise
# As in EXACT_FORCE, controls and patients played different step_ids of the game.
RHYTHM_STEP_TYPE = "RHYTHM"
RHYTHM_CONTROL_STEP_ID = "1e177cdd-0baf-42fe-b297-fbeba73ea828"
RHYTHM_PATIENT_STEP_ID = "65a5a4b1-907c-4717-b47b-cd8f31eb3e4b"
RHYTHM_STEP_IDS = (RHYTHM_CONTROL_STEP_ID, RHYTHM_PATIENT_STEP_ID)
RHYTHM_CROP_START_S = 2.0  # the first presses start after ~1 s
RHYTHM_CROP_END_S = 30.0
# PRACTICE_ATTEMPTS (common.config): the first attempt is a practice attempt, people press slower and harder in it.

# --------------------------------------------------------------------------- press detection
# press_offset / release_offset in additional_data are not aligned with the force signal
# (they fall on peaks in some recordings and on troughs in others), so presses are
# detected from the signal itself: peaks of the summed force of both hands.
RHYTHM_MIN_CYCLE_S = 0.25  # presses closer than this are one press
RHYTHM_MIN_PROMINENCE = 0.02  # absolute minimum peak height above the neighbouring troughs
RHYTHM_RELATIVE_PROMINENCE = 0.25  # ... and at least this share of the attempt's 5-95 % force range
RHYTHM_MIN_CYCLES = 5  # fewer detected presses: the attempt is not rhythmic, not analysed

# --------------------------------------------------------------------------- hand balance (week 4)
# A 10-year age category is tested only when both groups have at least this many people
# (healthy people aged 50+ are few).
RHYTHM_MIN_GROUP_SIZE = 3
