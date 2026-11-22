"""Settings shared by all exercises: paths, device, preprocessing and statistics.

Exercise-specific settings (game steps, crop window, features, thresholds) live in
``rehapiano.exact_force.config`` and ``rehapiano.rhythm.config``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
MEASUREMENTS_DIR = DATA_DIR / "raw" / "RehaPiano"
METADATA_DIR = DATA_DIR / "metadata"
FULL_METADATA_CSV = METADATA_DIR / "dataset_full.csv"
PATIENTS_METADATA_CSV = METADATA_DIR / "dataset_patients.csv"
# Questionnaire screening of the healthy group (week 2, M. Holovcyk): eligible_comparison =
# healthy without a reported condition affecting the hands, or a patient.
PARTICIPANT_LABELS_CSV = DATA_DIR / "participant_labels.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
COMBINED_OUTPUT_DIR = OUTPUT_DIR / "combined"
COMPARISON_OUTPUT_DIR = OUTPUT_DIR / "comparison"  # EXACT_FORCE vs RHYTHM: conclusions, agreement, repeatability

MATCH_KEY = ["measurement_id", "step_id"]

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
ANALYSED_SEGMENT = 1  # only the first continuous segment is analysed

# --------------------------------------------------------------------------- age
AGE_GROUP_WIDTH_YEARS = 10
# Decades as ordered labels (10-19 ... 80-89), used for norms and age categories.
AGE_CATEGORY_EDGES = tuple(range(10, 100, AGE_GROUP_WIDTH_YEARS))
AGE_CATEGORY_LABELS = tuple(f"{low}–{low + AGE_GROUP_WIDTH_YEARS - 1}" for low in AGE_CATEGORY_EDGES[:-1])
OLDER_AGE = 50  # controls are much younger; groups are also compared within age >= 50

# --------------------------------------------------------------------------- statistics
ALPHA = 0.05
RANDOM_SEED = 0
# Weak hand: hand balance below this quantile of the healthy controls.
WEAK_HAND_REFERENCE_QUANTILE = 0.025
# An empirical 2.5-97.5 % range needs at least this many healthy people in a decade.
MIN_NORM_SIZE = 20
NORM_QUANTILES = (0.025, 0.5, 0.975)
# Classifier evaluation: repeated stratified k-fold cross-validation.
CV_FOLDS = 5
CV_REPEATS = 20
CLASSIFIER_PERMUTATIONS = 200
