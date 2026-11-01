"""Timestamp checks and filling of missing values on a regular 1 ms grid.

Observed in the data: ~25 % of packets share a timestamp with a neighbour,
0.1-0.2 % of rows contain NaN (longest run 12 ms) and no gap exceeds 50 ms.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from rehapiano.common.config import (
    MAX_INTERPOLATED_GAP_S,
    MAX_MISSING_AT_EDGE_S,
    MAX_MISSING_FRACTION_PER_CHANNEL,
    MIN_SEGMENT_DURATION_S,
    PLAUSIBLE_PACKET_RATE_HZ,
    RAW_SAMPLE_RATE_HZ,
)

Segment = Tuple[np.ndarray, np.ndarray]  # (time_s, values[n, n_channels])


def validate_timestamps(timestamps_ms: np.ndarray) -> None:
    """Reject an attempt with backward time or an implausible packet rate."""

    if len(timestamps_ms) < 2:
        raise ValueError("An attempt must contain at least two packets")
    if np.any(np.diff(timestamps_ms) < 0):
        raise ValueError("Backward timestamp detected; the attempt was rejected")
    duration_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
    rate_hz = (len(timestamps_ms) - 1) / duration_s if duration_s > 0 else np.inf
    low, high = PLAUSIBLE_PACKET_RATE_HZ
    if duration_s <= 0 or not low <= rate_hz <= high:
        raise ValueError("Implausible timestamp scale or packet rate")


def merge_duplicate_timestamps(timestamps_ms: np.ndarray, values: np.ndarray) -> Segment:
    """Seconds from attempt start; packets with the same timestamp are averaged."""

    relative_time_s = (timestamps_ms - timestamps_ms[0]) / 1000.0
    frame = pd.DataFrame(values, index=relative_time_s)
    frame = frame.replace([np.inf, -np.inf], np.nan).groupby(level=0).mean()
    return frame.index.to_numpy(dtype=np.float64), frame.to_numpy(dtype=np.float64)


def split_at_gaps(time_s: np.ndarray, values: np.ndarray) -> List[Segment]:
    """Split where packets are missing for longer than ``MAX_INTERPOLATED_GAP_S``.

    Segments shorter than ``MIN_SEGMENT_DURATION_S`` are dropped.
    """

    split_indices = np.where(np.diff(time_s) > MAX_INTERPOLATED_GAP_S)[0] + 1
    bounds = np.r_[0, split_indices, len(time_s)]
    segments = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        segment_time = time_s[start:end]
        if len(segment_time) >= 2 and segment_time[-1] - segment_time[0] >= MIN_SEGMENT_DURATION_S:
            segments.append((segment_time, values[start:end]))
    return segments


def interpolate_to_regular_grid(time_s: np.ndarray, values: np.ndarray) -> Segment:
    """Linear interpolation of every channel onto a regular 1 ms grid.

    A channel that cannot be filled safely - more than 1 % missing, an internal
    gap longer than ``MAX_INTERPOLATED_GAP_S`` or missing data at either edge -
    is left entirely NaN instead of being invented.
    """

    step_ms = 1000.0 / RAW_SAMPLE_RATE_HZ
    regular_time_s = np.arange(
        np.ceil(time_s[0] * 1000.0 / step_ms) * step_ms,
        np.floor(time_s[-1] * 1000.0 / step_ms) * step_ms + step_ms,
        step_ms,
    ) / 1000.0
    regular_values = np.full((len(regular_time_s), values.shape[1]), np.nan)

    for channel_index in range(values.shape[1]):
        channel = values[:, channel_index]
        finite = np.isfinite(channel)
        if finite.sum() < 2 or 1.0 - finite.mean() > MAX_MISSING_FRACTION_PER_CHANNEL:
            continue
        valid_time = time_s[finite]
        if (
            np.max(np.diff(valid_time)) > MAX_INTERPOLATED_GAP_S
            or valid_time[0] - time_s[0] > MAX_MISSING_AT_EDGE_S
            or time_s[-1] - valid_time[-1] > MAX_MISSING_AT_EDGE_S
        ):
            continue
        regular_values[:, channel_index] = np.interp(regular_time_s, valid_time, channel[finite])
    return regular_time_s, regular_values


def fill_missing(timestamps_ms: np.ndarray, values: np.ndarray) -> List[Segment]:
    """Full gap handling for one attempt: checks, merge, split, interpolate."""

    validate_timestamps(timestamps_ms)
    time_s, merged = merge_duplicate_timestamps(timestamps_ms, values)
    return [interpolate_to_regular_grid(t, v) for t, v in split_at_gaps(time_s, merged)]
