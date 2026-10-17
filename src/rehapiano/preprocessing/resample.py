"""Reduce the regular 1000 Hz grid to the 100 Hz analysis rate.

The analysis stops at 15 Hz, so 100 Hz (Nyquist 50 Hz) loses nothing.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from rehapiano.config import ANALYSIS_SAMPLE_RATE_HZ, DOWNSAMPLE_FACTOR
from rehapiano.preprocessing.gaps import Segment


def downsample(time_s: np.ndarray, values: np.ndarray) -> Segment:
    """Polyphase resampling; it applies an anti-alias low-pass filter first."""

    downsampled = signal.resample_poly(values, up=1, down=DOWNSAMPLE_FACTOR, axis=0, padtype="line")
    downsampled_time_s = time_s[0] + np.arange(len(downsampled)) / ANALYSIS_SAMPLE_RATE_HZ
    return downsampled_time_s, downsampled
