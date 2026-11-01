"""RHYTHM-game features of one attempt (both hands press and release as fast as possible).

Hand force = mean of the hand's five sensors. Presses are the peaks of the summed force of both hands; a press
lasts from the minimum before its peak to the minimum after it. The press times written by the game are not
used: they sit on force peaks in some recordings and on minima in others (week 5).

Parameters (the week-6 set, plus the week-5 force-range balance):

* ``press_rate_hz``      presses per second;
* ``interval_cv``        SD / mean of the intervals between presses (0 = perfectly regular);
* ``amplitude_strong``,  median press amplitude (max - min within a press) of the hand with the larger /
  ``amplitude_weak``     smaller median amplitude;
* ``balance_press``      amplitude_weak / amplitude_strong;
* ``balance_range``      weaker / stronger hand by force range (95th - 5th percentile over the attempt); needs no
                         press detection and also works when the hands alternate;
* ``amplitude_cv``       SD / mean of the press amplitudes of the summed force;
* ``sync_lag_ms``        median time between the left- and right-hand maxima within a press;
* ``release_ratio``      median of (force at the minimum after a press / force at its peak), summed force;
* ``fatigue_amplitude``, ``fatigue_rate``  second half / first half of the attempt.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from rehapiano.pipeline import settings

PARAMETERS = ["press_rate_hz", "interval_cv", "amplitude_strong", "amplitude_weak", "balance_press", "balance_range",
              "amplitude_cv", "sync_lag_ms", "release_ratio", "fatigue_amplitude", "fatigue_rate"]


def hand_forces(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {hand: frame[[f"{hand}_sensor_{k}" for k in range(1, 6)]].mean(axis=1).to_numpy()
            for hand in ("left", "right")}


def find_presses(total: np.ndarray, fs: float) -> np.ndarray:
    spread = np.percentile(total, 95) - np.percentile(total, 5)
    peaks, _ = find_peaks(total, distance=max(1, int(settings.RHYTHM_PRESS_MIN_DISTANCE_S * fs)),
                          prominence=settings.RHYTHM_PRESS_MIN_PROMINENCE * spread)
    return peaks


def _ratio(a: float, b: float) -> float:
    high = max(a, b)
    return float(min(a, b) / high) if high > 0 else np.nan


def attempt_features(frame: pd.DataFrame, fs: float = 100.0) -> Dict[str, float]:
    force = hand_forces(frame)
    left, right = force["left"], force["right"]
    total = left + right
    time = frame["time_s"].to_numpy()
    duration = float(time[-1] - time[0])
    span = {hand: float(np.percentile(v, 95) - np.percentile(v, 5)) for hand, v in force.items()}
    result: Dict[str, float] = {name: np.nan for name in PARAMETERS}
    result.update(duration_s=duration, balance_range=_ratio(span["left"], span["right"]),
                  weaker_hand="left" if span["left"] < span["right"] else "right")

    peaks = find_presses(total, fs)
    result.update(presses=len(peaks), press_rate_hz=len(peaks) / duration if duration > 0 else np.nan)
    if len(peaks) < settings.RHYTHM_MIN_PRESSES:
        return result

    troughs = [int(np.argmin(total[a:b])) + a for a, b in zip(peaks[:-1], peaks[1:])]
    bounds = [int(np.argmin(total[:peaks[0] + 1])), *troughs, int(np.argmin(total[peaks[-1]:])) + int(peaks[-1])]
    windows = [slice(bounds[i], bounds[i + 1] + 1) for i in range(len(peaks))]
    amp_left = np.array([np.ptp(left[w]) for w in windows])
    amp_right = np.array([np.ptp(right[w]) for w in windows])
    amp_total = np.array([np.ptp(total[w]) for w in windows])
    strong, weak = sorted([np.median(amp_left), np.median(amp_right)], reverse=True)
    intervals = np.diff(time[peaks])
    lags = [abs(int(np.argmax(left[w])) - int(np.argmax(right[w]))) / fs * 1000 for w in windows]
    release = [total[bounds[i + 1]] / total[p] for i, p in enumerate(peaks) if total[p] > 0]
    half = time[0] + duration / 2
    first, second = time[peaks] < half, time[peaks] >= half
    result.update(
        interval_cv=float(intervals.std() / intervals.mean()),
        amplitude_strong=float(strong), amplitude_weak=float(weak), balance_press=_ratio(weak, strong),
        amplitude_cv=float(amp_total.std() / amp_total.mean()) if amp_total.mean() > 0 else np.nan,
        sync_lag_ms=float(np.median(lags)), release_ratio=float(np.median(release)) if release else np.nan,
        fatigue_amplitude=(float(np.median(amp_total[second]) / np.median(amp_total[first]))
                           if first.sum() >= 2 and second.sum() >= 2 and np.median(amp_total[first]) > 0 else np.nan),
        fatigue_rate=float(second.sum() / first.sum()) if first.sum() else np.nan,
    )
    return result
