"""Read one measurement file and run all preprocessing steps.

read -> polarity -> fill missing (1 ms grid) -> downsample to 100 Hz -> crop

Example
-------
from rehapiano.preprocessing.pipeline import load_measurement_data
data = load_measurement_data("data/raw/RehaPiano/.../results_<uuid>_<date>")
"""

from __future__ import annotations

from typing import List

import pandas as pd

from rehapiano.config import CROP_END_S, CROP_START_S, SENSOR_COLUMNS
from rehapiano.io.reader import PathLike, read_raw_attempts
from rehapiano.preprocessing.crop import crop_mask
from rehapiano.preprocessing.gaps import fill_missing
from rehapiano.preprocessing.polarity import correct_polarity
from rehapiano.preprocessing.resample import downsample


def load_measurement_data(
    path: PathLike,
    start_s: float = CROP_START_S,
    end_s: float = CROP_END_S,
) -> pd.DataFrame:
    """Return one preprocessed measurement.

    Columns: ``attempt`` (from 1), ``segment`` (continuous part of the attempt,
    from 1), ``time_s`` (from attempt start) and the ten ``config.SENSOR_COLUMNS``
    in relative device units, polarity-corrected, at 100 Hz. A channel that could
    not be filled stays NaN. Pass ``start_s=0, end_s=16`` for whole attempts.
    """

    frames: List[pd.DataFrame] = []
    for attempt_number, (timestamps_ms, raw_values) in enumerate(read_raw_attempts(path), start=1):
        segments = fill_missing(timestamps_ms, correct_polarity(raw_values))
        for segment_number, (grid_time_s, grid_values) in enumerate(segments, start=1):
            time_s, values = downsample(grid_time_s, grid_values)
            keep = crop_mask(time_s, start_s, end_s)
            if not keep.any():
                continue
            frame = pd.DataFrame(values[keep], columns=SENSOR_COLUMNS)
            frame.insert(0, "time_s", time_s[keep])
            frame.insert(0, "segment", segment_number)
            frame.insert(0, "attempt", attempt_number)
            frames.append(frame)

    if not frames:
        raise ValueError("No usable data remain after preprocessing")
    return pd.concat(frames, ignore_index=True)
