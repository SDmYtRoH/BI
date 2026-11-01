"""Keep the analysed part of an attempt (the window is set per exercise)."""

from __future__ import annotations

import numpy as np


def crop_mask(time_s: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    """Boolean mask of samples with ``start_s <= t < end_s`` (time from attempt start).

    EXACT_FORCE uses 3-15 s (1200 samples at 100 Hz), RHYTHM 2-30 s.
    """

    if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        raise ValueError("end_s must be greater than start_s")
    return (time_s >= start_s) & (time_s < end_s)
