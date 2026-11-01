"""EXACT_FORCE features of one preprocessed measurement, per attempt x finger and per attempt x hand.

Finger level: the week-3 finger features (``features.finger.analyze_finger``) plus the mean spectrogram
amplitude in the slow (0.2-3 Hz) and boundary (3-4 Hz) bands, so the week-2 decomposition and the week-3
finger analysis share one definition. ``tremor_mean_amplitude`` is the 4-15 Hz band.

Hand level: spectra are computed per finger and only then averaged, so opposite-phase finger oscillations
cannot cancel. Inactive fingers (force within the unloaded-sensor range) contribute force but no oscillation
values; a hand without an active finger has missing oscillation values, never zero.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from rehapiano.config import ANALYSED_SEGMENT, FINGER_NAMES, FINGERS_PER_HAND, HANDS
from rehapiano.features.finger import analyze_finger, sensor_column
from rehapiano.features.spectral import compute_spectrogram
from rehapiano.quality import VALID


def band_amplitudes(values: np.ndarray, bands: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """Mean spectrogram magnitude per band; low <= f < high, the last band includes its upper edge."""

    frequencies, _, magnitude = compute_spectrogram(values)
    result, names = {}, list(bands)
    for name, (low, high) in bands.items():
        inside = (frequencies >= low) & ((frequencies < high) | ((frequencies <= high) & (name == names[-1])))
        result[f"{name}_amplitude"] = float(magnitude[inside].mean()) if inside.any() else np.nan
    return result


def finger_features(measurement: pd.DataFrame, bands: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """One row per attempt x hand x finger."""

    analysed = measurement.loc[measurement["segment"] == ANALYSED_SEGMENT]
    rows = []
    for attempt in sorted(measurement["attempt"].unique()):
        attempt_data = analysed.loc[analysed["attempt"] == attempt]
        for hand in HANDS:
            for finger in range(1, FINGERS_PER_HAND + 1):
                values = attempt_data[sensor_column(hand, finger)].to_numpy()
                row = {"attempt": int(attempt), "hand": hand, "finger": finger, "finger_name": FINGER_NAMES[finger],
                       **analyze_finger(values)}
                row.update(band_amplitudes(values, bands) if row["status"] == VALID
                           else {f"{name}_amplitude": np.nan for name in bands})
                rows.append(row)
    return pd.DataFrame(rows)


def hand_features(fingers: pd.DataFrame) -> pd.DataFrame:
    """One row per attempt x hand from the finger rows."""

    valid = fingers.loc[fingers["status"] == VALID].copy()
    valid["active"] = ~valid["finger_inactive"].astype(bool)
    rows = []
    for (attempt, hand), frame in valid.groupby(["attempt", "hand"]):
        active = frame.loc[frame["active"]]
        slow, tremor = active["slow_amplitude"].mean(), active["tremor_mean_amplitude"].mean()
        rows.append({"attempt": attempt, "hand": hand, "valid_fingers": len(frame), "active_fingers": len(active),
                     "mean_force": frame["mean_force"].mean(),
                     "slow_amplitude": slow, "boundary_amplitude": active["boundary_amplitude"].mean(),
                     "tremor_amplitude": tremor,
                     "tremor_share": tremor / (slow + tremor) if len(active) else np.nan,
                     "tremor_peak_frequency_hz": active["tremor_peak_frequency_hz"].median()})
    return pd.DataFrame(rows)
