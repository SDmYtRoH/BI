"""Settings of the EXACT_FORCE exercise (per-finger tremor).

The person presses all fingers and holds the force at a target level: 4 attempts of 16 s.
Shared settings (paths, device, preprocessing, statistics) are in ``rehapiano.common.config``.
"""

from __future__ import annotations

from rehapiano.common.config import ANALYSIS_SAMPLE_RATE_HZ, MEASUREMENTS_DIR, OLDER_AGE, OUTPUT_DIR

# --------------------------------------------------------------------------- outputs
EF_OUTPUT_DIR = OUTPUT_DIR / "exact_force"  # attempt / measurement / person feature tables
STATS_OUTPUT_DIR = EF_OUTPUT_DIR / "stats"  # week 3: age, weak hand, finger tremor
ANALYSIS_OUTPUT_DIR = EF_OUTPUT_DIR / "analysis"  # hypotheses, decade norms, classifier, ablation

# --------------------------------------------------------------------------- exercise
STEP_TYPE = "EXACT_FORCE"
# EXACT_FORCE has a different step_id for controls and for patients; no step_id
# contains both groups, so step_id is fully confounded with the group.
# The second control ID was truncated in the task description ("...cab77413ac2").
CONTROL_STEP_IDS = (
    "5867ebbc-a463-4d20-8b20-50df7c64ace7",
    "d475bc4a-2c13-4297-98f9-cab77413ac2b",  # only one attempt per measurement
)
PATIENT_STEP_IDS = ("d2e6b7bd-a952-4652-afe3-9888e29ac205",)
ANALYSED_STEP_IDS = CONTROL_STEP_IDS + PATIENT_STEP_IDS

CROP_START_S = 3.0  # the first seconds contain the force build-up
CROP_END_S = 15.0

# --------------------------------------------------------------------------- spectral features
SPECTROGRAM_WINDOW_S = 5.0
SPECTROGRAM_NPERSEG = int(SPECTROGRAM_WINDOW_S * ANALYSIS_SAMPLE_RATE_HZ)  # 0.2 Hz resolution
SPECTROGRAM_NOVERLAP = SPECTROGRAM_NPERSEG - 1  # hop of one sample (0.01 s)
SPECTROGRAM_WINDOW = "hann"
FREQ_MAX_HZ = 15.0
# Bands as (low, high, include_low); high is inclusive. DC is excluded from movement.
# The boundary is 4 Hz: pathological tremor is >= 4 Hz, and with 3 Hz the
# decaying movement spectrum put 63 % of "tremor peaks" at <= 3.6 Hz.
MOVEMENT_BAND = (0.0, 4.0, False)  # voluntary force corrections
TREMOR_BAND = (4.0, 15.0, True)
EPSILON = 1e-12

# --------------------------------------------------------------------------- quality
# Unloaded sensors in the calibration recordings read 0 +- 0.002 with a noise
# std of ~1e-4 (relative device units).
FLAT_SIGNAL_MAX_STD = 1e-5  # far below sensor noise -> sensor dead / disconnected
FINGER_INACTIVE_MAX_FORCE = 0.002  # mean force within the unloaded range -> finger not pressing
INACTIVE_HAND_MAX_ACTIVITY_RATIO = 0.10  # weaker hand force < 10 % of the stronger one

# Outliers: a unit (attempt or measurement) whose value differs from the median
# of its group by more than this factor (in either direction) is flagged.
OUTLIER_MAX_FACTOR = 3.0
OUTLIER_MIN_GROUP_SIZE = 3  # a median of fewer units is not a reliable reference

# --------------------------------------------------------------------------- statistics (week 3)
# Controls did two steps; the first one has 4 attempts like the patient step,
# so it is the control step used for group comparisons.
PRIMARY_CONTROL_STEP_ID = CONTROL_STEP_IDS[0]
WEAK_HAND_RATIO_THRESHOLDS = (0.10, 0.25, 0.50)  # weaker / stronger hand
AGE_MATCHED_MIN_AGE = OLDER_AGE
MIN_ACTIVE_FINGERS_FOR_SPREAD = 3  # between-finger spread needs at least 3 active fingers
# Features compared between fingers (dimensionless or frequency features are
# preferred: raw amplitudes also depend on how hard each finger presses).
FINGER_COMPARISON_FEATURES = (
    "tremor_to_movement_ratio",
    "tremor_mean_amplitude",
    "tremor_peak_frequency_hz",
    "tremor_peak_prominence",
)

# --------------------------------------------------------------------------- reference values
CALIBRATION_DIR = MEASUREMENTS_DIR / "25_03_2025_calibration"
# A calibration channel counts as unloaded (pure sensor noise) below these limits.
UNLOADED_MAX_ABS_MEAN = 0.003
UNLOADED_MAX_STD = 0.0005
WEAK_HAND_METRIC = "force_ratio"
# Finger tremor is judged against healthy controls of comparable age; the
# younger controls would make every older person look abnormal.
REFERENCE_MIN_AGE = AGE_MATCHED_MIN_AGE
ELEVATED_TREMOR_Z = 1.96  # above ~97.5 % of the reference controls
# With 10 fingers and a 2.5 % chance per finger, a healthy person has ~22 % chance
# of one elevated finger but only ~2.5 % of two, so one finger is not a pattern.
LOCALIZED_MIN_FINGERS = 2
GENERALIZED_MIN_FINGERS = 5  # elevated on at least half of the 10 fingers
BOOTSTRAP_SAMPLES = 2000
