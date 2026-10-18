"""Project-wide paths, identifiers and analysis parameters.

Every number that influences the results lives here, so the pipeline can be
re-tuned without touching the processing code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MEASUREMENTS_DIR = DATA_DIR / "raw" / "RehaPiano"
METADATA_DIR = DATA_DIR / "metadata"
FULL_METADATA_CSV = METADATA_DIR / "dataset_full.csv"
PATIENTS_METADATA_CSV = METADATA_DIR / "dataset_patients.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

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
MATCH_KEY = ["measurement_id", "step_id"]

AGE_GROUP_WIDTH_YEARS = 10

# --------------------------------------------------------------------------- device
RAW_SAMPLE_RATE_HZ = 1000.0  # observed packet rate: 1000-1010 packets/s
CHANNELS_PER_PACKET = 16
HANDS = ("left", "right")
FINGERS_PER_HAND = 5
SENSOR_COLUMNS = [f"{hand}_sensor_{k}" for hand in HANDS for k in range(1, FINGERS_PER_HAND + 1)]
# Physical packet channel of every sensor column. Channels 5-7 and 13-15 are always 0.
SOURCE_CHANNELS = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12])
# Sign applied to each sensor column. Channel 9 (right_sensor_2) reacts with the
# opposite sign to all other sensors - both in the game data and in the
# calibration recordings (data/raw/RehaPiano/25_03_2025_calibration).
SENSOR_POLARITY = {column: 1.0 for column in SENSOR_COLUMNS} | {"right_sensor_2": -1.0}

# Finger names of sensor_1..sensor_5. NOT verified: channels 0/8 are believed to
# be the little fingers and 4/12 the thumbs, possibly the other way round.
# Change only this mapping if the order turns out to be different.
FINGER_NAMES = {1: "little", 2: "ring", 3: "middle", 4: "index", 5: "thumb"}

# --------------------------------------------------------------------------- preprocessing
ANALYSIS_SAMPLE_RATE_HZ = 100.0
DOWNSAMPLE_FACTOR = int(RAW_SAMPLE_RATE_HZ / ANALYSIS_SAMPLE_RATE_HZ)
MAX_INTERPOLATED_GAP_S = 0.05  # longer gaps split the attempt into segments
MIN_SEGMENT_DURATION_S = 3.0
MAX_MISSING_FRACTION_PER_CHANNEL = 0.01
MAX_MISSING_AT_EDGE_S = 0.01
PLAUSIBLE_PACKET_RATE_HZ = (50.0, 2000.0)
CROP_START_S = 3.0  # the first seconds contain the force build-up
CROP_END_S = 15.0
ANALYSED_SEGMENT = 1  # only the first continuous segment is analysed

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

# --------------------------------------------------------------------------- statistics
STATS_OUTPUT_DIR = OUTPUT_DIR / "stats"
# Controls did two steps; the first one has 4 attempts like the patient step,
# so it is the control step used for group comparisons.
PRIMARY_CONTROL_STEP_ID = CONTROL_STEP_IDS[0]
WEAK_HAND_RATIO_THRESHOLDS = (0.10, 0.25, 0.50)  # weaker / stronger hand
AGE_MATCHED_MIN_AGE = 50  # controls are much younger; compare also within age >= 50
MIN_ACTIVE_FINGERS_FOR_SPREAD = 3  # between-finger spread needs at least 3 active fingers
ALPHA = 0.05
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
# Weak hand: hand symmetry below this quantile of the healthy controls.
WEAK_HAND_METRIC = "force_ratio"
WEAK_HAND_REFERENCE_QUANTILE = 0.025
# Finger tremor is judged against healthy controls of comparable age; the
# younger controls would make every older person look abnormal.
REFERENCE_MIN_AGE = AGE_MATCHED_MIN_AGE
ELEVATED_TREMOR_Z = 1.96  # above ~97.5 % of the reference controls
# With 10 fingers and a 2.5 % chance per finger, a healthy person has ~22 % chance
# of one elevated finger but only ~2.5 % of two, so one finger is not a pattern.
LOCALIZED_MIN_FINGERS = 2
GENERALIZED_MIN_FINGERS = 5  # elevated on at least half of the 10 fingers
BOOTSTRAP_SAMPLES = 2000
RANDOM_SEED = 0
