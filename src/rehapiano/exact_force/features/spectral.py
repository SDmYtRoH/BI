"""Spectrogram and frequency-band helpers shared by all feature extractors."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal as sp_signal

from rehapiano.common.config import ANALYSIS_SAMPLE_RATE_HZ
from rehapiano.exact_force.config import FREQ_MAX_HZ, SPECTROGRAM_NOVERLAP, SPECTROGRAM_NPERSEG, SPECTROGRAM_WINDOW

Band = Tuple[float, float, bool]  # (low, high, include_low); high is inclusive


def compute_spectrogram(
    values: np.ndarray,
    fs: float = ANALYSIS_SAMPLE_RATE_HZ,
    nperseg: int = SPECTROGRAM_NPERSEG,
    noverlap: int = SPECTROGRAM_NOVERLAP,
    freq_max: float = FREQ_MAX_HZ,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``frequencies, times, magnitude`` limited to ``<= freq_max`` Hz.

    Each 5 s Hann window is mean-detrended, so the force level (DC) is removed
    and only fluctuations remain.
    """

    frequencies, times, magnitude = sp_signal.spectrogram(
        np.asarray(values, dtype=float),
        fs=fs,
        window=SPECTROGRAM_WINDOW,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="spectrum",
        mode="magnitude",
    )
    keep = frequencies <= freq_max
    return frequencies[keep], times, magnitude[keep]


def band_mask(frequencies: np.ndarray, band: Band) -> np.ndarray:
    """Boolean mask of the frequencies inside ``band``."""

    low, high, include_low = band
    above_low = frequencies >= low if include_low else frequencies > low
    return above_low & (frequencies <= high)
