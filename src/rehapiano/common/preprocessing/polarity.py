"""Bring all sensors to the same sign convention."""

from __future__ import annotations

import numpy as np

from rehapiano.common.config import SENSOR_COLUMNS, SENSOR_POLARITY

POLARITY_VECTOR = np.array([SENSOR_POLARITY[column] for column in SENSOR_COLUMNS])


def correct_polarity(sensor_values: np.ndarray) -> np.ndarray:
    """Multiply every column by its sign from ``config.SENSOR_POLARITY``.

    Only the mean force depends on this; spectral magnitudes are sign-invariant.
    """

    return np.asarray(sensor_values, dtype=np.float64) * POLARITY_VECTOR
