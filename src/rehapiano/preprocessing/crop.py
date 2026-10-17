"""Keep the steady force-holding part of an attempt."""

from __future__ import annotations

import numpy as np

from rehapiano.config import CROP_END_S, CROP_START_S


def crop_mask(time_s: np.ndarray, start_s: float = CROP_START_S, end_s: float = CROP_END_S) -> np.ndarray:
    """Boolean mask of samples with ``start_s <= t < end_s`` (time from attempt start).

    Every attempt lasts 16 s; the default 3-15 s gives 1200 samples at 100 Hz.
    """

    if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        raise ValueError("end_s must be greater than start_s")
    return (time_s >= start_s) & (time_s < end_s)
