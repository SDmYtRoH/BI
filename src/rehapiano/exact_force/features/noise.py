"""Sensor noise floor from the calibration recordings.

In the calibration files single sensors are loaded with a weight while the
others stay unloaded. The tremor features of unloaded channels describe pure
sensor noise and give the lower limit of a measurable finger tremor.
"""

from __future__ import annotations

import pandas as pd

from rehapiano.common.config import SENSOR_COLUMNS
from rehapiano.exact_force.config import (
    CALIBRATION_DIR,
    CROP_END_S,
    CROP_START_S,
    UNLOADED_MAX_ABS_MEAN,
    UNLOADED_MAX_STD,
)
from rehapiano.exact_force.features.finger import compute_finger_features
from rehapiano.common.io.metadata import find_measurement_files, to_project_relative
from rehapiano.common.preprocessing.pipeline import load_measurement_data


def estimate_sensor_noise(calibration_dir=CALIBRATION_DIR) -> pd.DataFrame:
    """Tremor features of every unloaded sensor in every calibration attempt."""

    rows = []
    for path in find_measurement_files(calibration_dir):
        data = load_measurement_data(path, CROP_START_S, CROP_END_S)
        for attempt, frame in data.groupby("attempt"):
            for column in SENSOR_COLUMNS:
                values = frame[column].to_numpy()
                if abs(values.mean()) < UNLOADED_MAX_ABS_MEAN and values.std() < UNLOADED_MAX_STD:
                    rows.append({"file_path": to_project_relative(path), "attempt": int(attempt),
                                 "sensor": column, **compute_finger_features(values)})
    return pd.DataFrame(rows)
