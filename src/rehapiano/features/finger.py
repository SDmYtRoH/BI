"""Per-finger features of the EXACT_FORCE exercise.

The person squeezes all fingers and tries to hold the force at a target level.
Slow fluctuations (0-4 Hz) are voluntary force corrections present in everyone;
tremor shows up as faster oscillations (4-15 Hz). Band limits: ``config``. Every attempt, hand and
finger is analysed separately on the cropped 3-15 s part (1200 samples).

Features (``FINGER_FEATURE_COLUMNS``):

* ``tremor_mean_amplitude``    - mean spectrogram magnitude in 4-15 Hz over all
                                 windows: overall strength of the tremor-band
                                 oscillations;
* ``tremor_mean_frequency_hz`` - spectral centroid of the time-averaged 4-15 Hz
                                 spectrum: where the tremor energy is centred;
* ``tremor_peak_frequency_hz`` - frequency of the highest local maximum of the
                                 time-averaged 4-15 Hz spectrum. A plain argmax
                                 is not used: the spectrum decays with frequency,
                                 so the argmax often sits on the lower band edge
                                 without any real peak there. NaN if the band
                                 has no local maximum;
* ``tremor_peak_amplitude``    - magnitude of that component;
* ``mean_force``               - mean polarity-corrected signal: how strongly
                                 the finger presses. Needed for inactive
                                 fingers / hands and because tremor amplitude
                                 grows with force. Unloaded sensors read ~0
                                 (calibration recordings), so it is comparable
                                 between fingers;
* ``tremor_to_movement_ratio`` - tremor mean amplitude / movement (0-3 Hz) mean
                                 amplitude (4-15 Hz / 0-4 Hz): share of fast
                                 oscillation relative
                                 to normal slow corrections. Dimensionless, so
                                 independent of force level and sensor gain;
* ``tremor_peak_prominence``   - peak amplitude / median of the 4-15 Hz
                                 spectrum: ~1-2 means broadband noise (the peak
                                 frequency is then random), clearly larger means
                                 a rhythmic tremor at ``tremor_peak_frequency_hz``.

Possible additional features (not computed, add here when needed):

* ``tremor_amplitude_std``     - std over windows of the band magnitude
                                 (stability of the tremor in time);
* movement-band (0-4 Hz) peak / mean frequency and amplitude;
* ``force_std`` / RMS of the force signal, coefficient of variation of force;
* sub-band power, e.g. 4-7 Hz (pathological) vs 8-12 Hz (physiological tremor);
* spectral entropy of the 3-15 Hz band (rhythmic vs irregular oscillation);
* coherence of the tremor between fingers of one hand (common central source
  vs local finger tremor);
* tremor_to_force ratio (tremor amplitude / mean force).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from rehapiano.config import (
    ANALYSED_SEGMENT,
    EPSILON,
    FINGER_NAMES,
    FINGERS_PER_HAND,
    HANDS,
    MOVEMENT_BAND,
    TREMOR_BAND,
)
from rehapiano.features.spectral import band_mask, compute_spectrogram
from rehapiano.quality import NO_SEGMENT, VALID, assess_finger_signal, is_finger_inactive

FINGER_FEATURE_COLUMNS = [
    "tremor_mean_amplitude",
    "tremor_mean_frequency_hz",
    "tremor_peak_frequency_hz",
    "tremor_peak_amplitude",
    "mean_force",
    "tremor_to_movement_ratio",
    "tremor_peak_prominence",
]


def sensor_column(hand: str, finger: int) -> str:
    """Column name of one finger in the preprocessed measurement."""

    return f"{hand}_sensor_{finger}"


def compute_finger_features(values: np.ndarray) -> Dict[str, float]:
    """All ``FINGER_FEATURE_COLUMNS`` for one valid finger signal."""

    values = np.asarray(values, dtype=float)
    frequencies, _, magnitude = compute_spectrogram(values)

    tremor = band_mask(frequencies, TREMOR_BAND)
    tremor_frequencies = frequencies[tremor]
    tremor_spectrum = magnitude[tremor].mean(axis=1)  # time-averaged spectrum
    local_maxima, _ = find_peaks(tremor_spectrum)  # never the first / last bin
    if local_maxima.size:
        peak_index = int(local_maxima[np.argmax(tremor_spectrum[local_maxima])])
        peak_frequency, peak_amplitude = float(tremor_frequencies[peak_index]), float(tremor_spectrum[peak_index])
    else:
        peak_frequency = peak_amplitude = np.nan
    tremor_mean_amplitude = float(magnitude[tremor].mean())
    movement_mean_amplitude = float(magnitude[band_mask(frequencies, MOVEMENT_BAND)].mean())

    return {
        "tremor_mean_amplitude": tremor_mean_amplitude,
        "tremor_mean_frequency_hz": float(
            np.sum(tremor_frequencies * tremor_spectrum) / (np.sum(tremor_spectrum) + EPSILON)
        ),
        "tremor_peak_frequency_hz": peak_frequency,
        "tremor_peak_amplitude": peak_amplitude,
        "mean_force": float(values.mean()),
        "tremor_to_movement_ratio": tremor_mean_amplitude / (movement_mean_amplitude + EPSILON),
        "tremor_peak_prominence": peak_amplitude / (float(np.median(tremor_spectrum)) + EPSILON),
    }


def analyze_finger(values: np.ndarray) -> Dict[str, object]:
    """Validity, inactivity flag and features of one finger in one attempt."""

    status, reason = assess_finger_signal(values)
    row: Dict[str, object] = {
        "n_samples": int(np.asarray(values).size),
        "status": status,
        "invalid_reason": reason,
        "finger_inactive": pd.NA,
        **{column: np.nan for column in FINGER_FEATURE_COLUMNS},
    }
    if status == VALID:
        row.update(compute_finger_features(values))
        row["finger_inactive"] = is_finger_inactive(row["mean_force"])
    return row


def extract_measurement_finger_features(measurement: pd.DataFrame) -> pd.DataFrame:
    """One row per ``attempt x hand x finger`` of a preprocessed measurement."""

    analysed = measurement.loc[measurement["segment"] == ANALYSED_SEGMENT]
    rows = []
    for attempt in sorted(measurement["attempt"].unique()):
        attempt_data = analysed.loc[analysed["attempt"] == attempt]
        for hand in HANDS:
            for finger in range(1, FINGERS_PER_HAND + 1):
                column = sensor_column(hand, finger)
                row = {"attempt": int(attempt), "hand": hand, "finger": finger,
                       "finger_name": FINGER_NAMES[finger], "sensor": column}
                row.update(analyze_finger(attempt_data[column].to_numpy()))
                if attempt_data.empty:
                    row["invalid_reason"] = NO_SEGMENT
                rows.append(row)
    return pd.DataFrame(rows)
