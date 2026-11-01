"""Hand force balance in the RHYTHM test.

The person squeezes and releases with both hands about once per second. Unlike
EXACT_FORCE the force is not constant, so the mean force depends on how long the
person rests between presses. The balance of the hands is therefore measured on
the presses themselves:

1. hand force = mean of its five polarity-corrected sensors (as ``mean_force`` in
   EXACT_FORCE, averaged over fingers);
2. presses = peaks of the summed force of both hands (both hands press together);
   each press spans the two troughs around its peak;
3. ``press_amplitude`` of a hand = median over presses of (maximum - minimum) of
   that hand inside the press;
4. ``mean_force`` of a hand = mean over the attempt (the EXACT_FORCE measure, for
   comparison);
5. ``force_range`` of a hand = 95th - 5th percentile of its force over the attempt: a
   check that needs no press detection and also works when the hands alternate
   (seen in ~1-2 % of attempts).

Balance = weaker / stronger hand, computed later on person-level values.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from rehapiano.rhythm.config import (
    RHYTHM_MIN_CYCLE_S,
    RHYTHM_MIN_CYCLES,
    RHYTHM_MIN_PROMINENCE,
    RHYTHM_RELATIVE_PROMINENCE,
)
from rehapiano.common.config import ANALYSED_SEGMENT, ANALYSIS_SAMPLE_RATE_HZ, FINGERS_PER_HAND, HANDS

Press = Tuple[int, int, int]  # (trough before, peak, trough after) as sample indices

RHYTHM_HAND_COLUMNS = ["press_amplitude", "press_peak_force", "mean_force", "force_range"]


def hand_forces(attempt: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Mean force of the five sensors of each hand."""

    return {hand: attempt[[f"{hand}_sensor_{k}" for k in range(1, FINGERS_PER_HAND + 1)]].to_numpy().mean(axis=1)
            for hand in HANDS}


def detect_presses(total: np.ndarray, fs: float = ANALYSIS_SAMPLE_RATE_HZ) -> List[Press]:
    """Presses as peaks of ``total`` with the troughs between neighbouring peaks.

    A peak must rise above its surroundings by at least ``RHYTHM_MIN_PROMINENCE`` and by
    ``RHYTHM_RELATIVE_PROMINENCE`` of the 5-95 % range of the attempt, so that small
    corrections of force are not counted as presses.
    """

    total = np.asarray(total, dtype=float)
    if not np.isfinite(total).all() or total.size < 3:
        return []
    low, high = np.percentile(total, [5, 95])
    prominence = max(RHYTHM_MIN_PROMINENCE, RHYTHM_RELATIVE_PROMINENCE * (high - low))
    peaks, _ = find_peaks(total, prominence=prominence, distance=max(1, int(RHYTHM_MIN_CYCLE_S * fs)))
    presses = []
    bounds = np.r_[0, peaks, total.size - 1]
    for i, peak in enumerate(peaks, start=1):
        before = bounds[i - 1] + int(np.argmin(total[bounds[i - 1]:peak + 1]))
        after = peak + int(np.argmin(total[peak:bounds[i + 1] + 1]))
        presses.append((before, int(peak), after))
    return presses


def attempt_hand_balance(attempt: pd.DataFrame, fs: float = ANALYSIS_SAMPLE_RATE_HZ) -> List[Dict[str, object]]:
    """One row per hand: validity, number of presses, press rate and ``RHYTHM_HAND_COLUMNS``."""

    forces = hand_forces(attempt)
    duration_s = len(attempt) / fs
    finite = all(np.isfinite(values).all() for values in forces.values())
    presses = detect_presses(forces["left"] + forces["right"], fs) if finite and len(attempt) else []
    status = "valid" if len(presses) >= RHYTHM_MIN_CYCLES else "invalid"
    reason = "" if status == "valid" else ("non_finite_values" if not finite else "too_few_presses")
    rows = []
    for hand, values in forces.items():
        row: Dict[str, object] = {"hand": hand, "status": status, "invalid_reason": reason,
                                  "n_presses": len(presses), "duration_s": duration_s,
                                  "press_rate_hz": len(presses) / duration_s if duration_s else np.nan,
                                  **{column: np.nan for column in RHYTHM_HAND_COLUMNS}}
        if status == "valid":
            maxima = np.array([values[a:b + 1].max() for a, _, b in presses])
            minima = np.array([values[a:b + 1].min() for a, _, b in presses])
            low, high = np.percentile(values, [5, 95])
            row.update(press_amplitude=float(np.median(maxima - minima)), press_peak_force=float(np.median(maxima)),
                       mean_force=float(values.mean()), force_range=float(high - low))
        rows.append(row)
    return rows


RHYTHM_PARAMETER_COLUMNS = [
    "press_rate_hz", "interval_cv", "amplitude_strong", "amplitude_weak", "balance", "amplitude_cv",
    "sync_lag_ms", "release_ratio", "fatigue_amplitude", "fatigue_rate",
]


def attempt_rhythm_parameters(attempt: pd.DataFrame, fs: float = ANALYSIS_SAMPLE_RATE_HZ) -> Dict[str, object]:
    """Parameters of one attempt for the week 5 analysis (one row per attempt).

    * ``press_rate_hz``      presses per second;
    * ``interval_cv``        SD / mean of the intervals between presses (0 = perfectly regular rhythm);
    * ``amplitude_strong`` / ``amplitude_weak``  median press amplitude of the stronger / weaker hand;
    * ``balance``            weak / strong (as in week 4, but per attempt);
    * ``amplitude_cv``       SD / mean of the press amplitudes of both hands together (0 = every press equal);
    * ``sync_lag_ms``        median |time of the left peak - time of the right peak| inside a press;
    * ``release_ratio``      median force left at the trough after a press / force at its peak
                             (0 = the hands release completely);
    * ``fatigue_amplitude``  median press amplitude in the second half / first half of the attempt (< 1 = weaker);
    * ``fatigue_rate``       presses in the second half / first half (< 1 = slower).
    """

    forces = hand_forces(attempt)
    finite = all(np.isfinite(values).all() for values in forces.values())
    total = forces["left"] + forces["right"]
    presses = detect_presses(total, fs) if finite and len(attempt) else []
    row: Dict[str, object] = {"n_presses": len(presses), "duration_s": len(attempt) / fs,
                              "status": "valid" if len(presses) >= RHYTHM_MIN_CYCLES else "invalid",
                              **{column: np.nan for column in RHYTHM_PARAMETER_COLUMNS}}
    if row["status"] != "valid":
        return row

    peaks = np.array([p for _, p, _ in presses])
    amplitude = {hand: np.array([values[a:b + 1].max() - values[a:b + 1].min() for a, _, b in presses])
                 for hand, values in forces.items()}
    hand_median = {hand: float(np.median(v)) for hand, v in amplitude.items()}
    strong, weak = max(hand_median.values()), min(hand_median.values())
    both = amplitude["left"] + amplitude["right"]
    lags = [abs(int(np.argmax(forces["left"][a:b + 1])) - int(np.argmax(forces["right"][a:b + 1]))) / fs * 1000
            for a, _, b in presses]
    release = [total[b] / total[p] for _, p, b in presses if total[p] > 0]
    intervals = np.diff(peaks) / fs
    half = len(total) / 2
    first, second = peaks < half, peaks >= half
    row.update(
        press_rate_hz=len(presses) / row["duration_s"],
        interval_cv=float(intervals.std() / intervals.mean()),
        amplitude_strong=strong, amplitude_weak=weak, balance=weak / strong if strong > 0 else np.nan,
        amplitude_cv=float(both.std() / both.mean()) if both.mean() > 0 else np.nan,
        sync_lag_ms=float(np.median(lags)),
        release_ratio=float(np.median(release)) if release else np.nan,
        fatigue_amplitude=(float(np.median(both[second]) / np.median(both[first]))
                           if first.any() and second.any() and np.median(both[first]) > 0 else np.nan),
        fatigue_rate=float(second.sum() / first.sum()) if first.any() else np.nan,
    )
    return row


def extract_measurement_rhythm_parameters(measurement: pd.DataFrame) -> pd.DataFrame:
    """One row per attempt of a preprocessed RHYTHM measurement with ``RHYTHM_PARAMETER_COLUMNS``."""

    analysed = measurement.loc[measurement["segment"] == ANALYSED_SEGMENT]
    return pd.DataFrame([{"attempt": int(attempt), **attempt_rhythm_parameters(analysed.loc[analysed["attempt"] == attempt])}
                         for attempt in sorted(measurement["attempt"].unique())])


def extract_measurement_rhythm_features(measurement: pd.DataFrame) -> pd.DataFrame:
    """One row per ``attempt x hand`` of a preprocessed RHYTHM measurement."""

    analysed = measurement.loc[measurement["segment"] == ANALYSED_SEGMENT]
    rows = []
    for attempt in sorted(measurement["attempt"].unique()):
        for row in attempt_hand_balance(analysed.loc[analysed["attempt"] == attempt]):
            rows.append({"attempt": int(attempt), **row})
    return pd.DataFrame(rows)
