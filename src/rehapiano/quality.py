"""Validity criteria for finger signals and outlier flags.

Levels of checks (earlier levels are applied during preprocessing):

1. file      - decodable, payload UUID equals the file name, matched to metadata
               (``io.metadata.build_manifest``);
2. attempt   - monotonic timestamps, plausible packet rate, a continuous segment
               of >= 3 s (``preprocessing.gaps``);
3. finger    - :func:`assess_finger_signal` -> ``status`` = valid / invalid;
               :func:`is_finger_inactive` only flags a valid finger that hardly
               presses, its row is kept;
4. outliers  - :func:`flag_outliers` marks attempts / measurements that differ
               strongly from the rest of their group; they are kept in the
               tables but excluded from the averages.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from rehapiano.config import (
    FINGER_INACTIVE_MAX_FORCE,
    FLAT_SIGNAL_MAX_STD,
    OUTLIER_MAX_FACTOR,
    OUTLIER_MIN_GROUP_SIZE,
    SPECTROGRAM_NPERSEG,
)

VALID = "valid"
INVALID = "invalid"

# invalid_reason values
NO_SEGMENT = "no_analysed_segment"  # attempt has no data in the analysed segment
TOO_SHORT = "too_short_for_spectrogram"  # fewer samples than one spectrogram window
NON_FINITE = "non_finite_values"  # channel could not be filled during preprocessing
FLAT_SIGNAL = "flat_signal"  # constant output: sensor dead or disconnected

# Not implemented yet (no such cases seen, device range unknown):
#   saturation      - long runs at the sensor maximum;
#   artifact_spike  - isolated jumps far outside the robust range of the signal.


def assess_finger_signal(values: np.ndarray) -> tuple[str, str]:
    """Return ``(status, invalid_reason)`` for one finger in one attempt."""

    values = np.asarray(values, dtype=float)
    if values.size < SPECTROGRAM_NPERSEG:
        return INVALID, TOO_SHORT
    if not np.isfinite(values).all():
        return INVALID, NON_FINITE
    if np.std(values) < FLAT_SIGNAL_MAX_STD:
        return INVALID, FLAT_SIGNAL
    return VALID, ""


def is_finger_inactive(mean_force: float) -> bool:
    """True when the mean force is within the unloaded-sensor range."""

    return bool(np.isfinite(mean_force) and mean_force < FINGER_INACTIVE_MAX_FORCE)


def flag_outliers(
    table: pd.DataFrame,
    group_columns: Sequence[str],
    value_columns: Sequence[str],
    max_factor: float = OUTLIER_MAX_FACTOR,
    min_group_size: int = OUTLIER_MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """Flag rows whose value differs from the group median by more than ``max_factor``.

    The comparison is a ratio (x / median), so it does not depend on the units
    or the overall level of a person. Only positive values are compared; groups
    with fewer than ``min_group_size`` finite values are never flagged.

    Returns ``is_outlier`` (bool) and ``outlier_reason`` aligned with ``table``.
    """

    reasons = pd.Series("", index=table.index, dtype=object)
    for column in value_columns:
        values = table[column].where(table[column] > 0)
        grouped = values.groupby([table[c] for c in group_columns])
        median = grouped.transform("median")
        count = grouped.transform("count")
        ratio = values / median
        flagged = (count >= min_group_size) & ((ratio > max_factor) | (ratio < 1.0 / max_factor))
        text = column + " x" + ratio.round(2).astype(str) + " vs group median"
        reasons[flagged] = (reasons[flagged] + "; " + text[flagged]).str.lstrip("; ")
    return pd.DataFrame({"is_outlier": reasons != "", "outlier_reason": reasons})
