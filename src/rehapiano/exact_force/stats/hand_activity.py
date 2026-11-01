"""How strongly each person uses the weaker hand.

Two complementary ratios (weaker hand / stronger hand, 1 = symmetric):

* ``force_ratio``       - mean finger force; low when one hand hardly presses;
* ``fluctuation_ratio`` - mean finger ``tremor_mean_amplitude``; low when one
                          hand produces almost no force changes, e.g. it only
                          rests on the keys (force present, no active control).

There is no agreed threshold. The main answer uses the healthy controls as the
reference: a hand is weakly used when its symmetry is lower than in 97.5 % of
controls (:func:`weak_hand_patients`); fixed thresholds are reported as well.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rehapiano.common.config import WEAK_HAND_REFERENCE_QUANTILE
from rehapiano.exact_force.config import WEAK_HAND_METRIC, WEAK_HAND_RATIO_THRESHOLDS
from rehapiano.common.stats.demographics import GROUP_ORDER
from rehapiano.exact_force.stats.selection import primary_step

RATIO_COLUMNS = ["force_ratio", "fluctuation_ratio"]


def _ratio(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    left, right = left.clip(lower=0), right.clip(lower=0)
    stronger = pd.concat([left, right], axis=1).max(axis=1)
    ratio = pd.concat([left, right], axis=1).min(axis=1) / stronger.where(stronger > 0)
    weaker = pd.Series(np.where(left < right, "left", "right"), index=left.index).where(ratio.notna())
    return ratio, weaker


def hand_activity_table(person_summary: pd.DataFrame, features_by_person: pd.DataFrame) -> pd.DataFrame:
    """One row per person (primary step) with both ratios and the weaker hand."""

    summary = primary_step(person_summary)
    fluctuation = (primary_step(features_by_person)
                   .groupby(["patient_id", "step_id", "hand"])["tremor_mean_amplitude"].mean().unstack("hand"))
    table = summary.merge(fluctuation.add_suffix("_fluctuation").reset_index(), on=["patient_id", "step_id"])
    table["force_ratio"], table["force_weaker_hand"] = _ratio(table["left_mean_force"], table["right_mean_force"])
    table["fluctuation_ratio"], table["fluctuation_weaker_hand"] = _ratio(
        table["left_fluctuation"], table["right_fluctuation"])
    columns = ["patient_id", "group", "age", "age_group", "gender", "step_id",
               "left_mean_force", "right_mean_force", "force_ratio", "force_weaker_hand",
               "left_fluctuation", "right_fluctuation", "fluctuation_ratio", "fluctuation_weaker_hand",
               "left_n_inactive_fingers", "right_n_inactive_fingers"]
    return table[columns].sort_values(["group", "force_ratio"]).reset_index(drop=True)


def weak_hand_counts(activity: pd.DataFrame, thresholds=WEAK_HAND_RATIO_THRESHOLDS) -> pd.DataFrame:
    """People with ratio below each threshold, per group and ratio type."""

    rows = []
    for metric in RATIO_COLUMNS:
        for threshold in thresholds:
            for group in GROUP_ORDER:
                frame = activity.loc[activity["group"] == group]
                below = frame[metric] < threshold
                rows.append({"metric": metric, "threshold": threshold, "group": group, "n_people": len(frame),
                             "n_below": int(below.sum()), "percent_below": 100 * below.mean(),
                             "patient_ids": ", ".join(frame.loc[below, "patient_id"].astype(str))})
    return pd.DataFrame(rows)


def weak_hand_threshold(activity: pd.DataFrame, metric: str = WEAK_HAND_METRIC,
                        quantile: float = WEAK_HAND_REFERENCE_QUANTILE) -> float:
    """Lower ``quantile`` of ``metric`` among the healthy controls."""

    return float(activity.loc[activity["group"] == "control", metric].quantile(quantile))


def weak_hand_patients(activity: pd.DataFrame, metric: str = WEAK_HAND_METRIC,
                       quantile: float = WEAK_HAND_REFERENCE_QUANTILE) -> pd.DataFrame:
    """Everyone below the control-based threshold, with the weaker hand and both ratios."""

    threshold = weak_hand_threshold(activity, metric, quantile)
    weaker = "force_weaker_hand" if metric == "force_ratio" else "fluctuation_weaker_hand"
    below = activity.loc[activity[metric] < threshold].copy()
    below["weaker_hand"] = below[weaker]
    below["threshold"] = threshold
    columns = ["patient_id", "group", "age", "gender", "weaker_hand", metric, "threshold",
               "force_ratio", "fluctuation_ratio", "left_mean_force", "right_mean_force"]
    columns = list(dict.fromkeys(columns))
    return below[columns].sort_values(["group", metric]).reset_index(drop=True)
