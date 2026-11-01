"""Aggregation of finger features: attempt -> measurement -> person.

* attempt level: outlier attempts are flagged (compared with the other
  attempts of the same measurement and hand);
* measurement level: mean of valid, non-outlier attempts; outlier measurements
  are flagged (compared with the other measurements of the same person, step
  and hand - only patients have repeated sessions);
* person level: mean of non-outlier measurements, kept separately per
  ``step_id`` (controls did both control steps, which differ in attempts).

Outliers are compared on hand-level summaries (mean over valid fingers) of
``OUTLIER_FEATURES``, so a whole attempt-hand is flagged, never single fingers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rehapiano.common.config import HANDS
from rehapiano.exact_force.config import FINGER_INACTIVE_MAX_FORCE, INACTIVE_HAND_MAX_ACTIVITY_RATIO
from rehapiano.exact_force.features.finger import FINGER_FEATURE_COLUMNS
from rehapiano.exact_force.quality import VALID, flag_outliers

PERSON_KEYS = ["patient_id", "step_id"]
MEASUREMENT_KEYS = ["patient_id", "measurement_id", "step_id"]
FINGER_KEYS = ["hand", "finger", "finger_name", "sensor"]
CONTEXT_COLUMNS = ["is_patient", "group", "age", "age_group", "gender", "metadata_hand"]
OUTLIER_FEATURES = ["mean_force", "tremor_mean_amplitude"]


def _hand_level(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return table.groupby([*keys, "hand"], dropna=False)[OUTLIER_FEATURES].mean().reset_index()


def flag_attempt_outliers(features_by_attempt: pd.DataFrame) -> pd.DataFrame:
    """Add ``attempt_outlier`` and ``attempt_outlier_reason``."""

    valid = features_by_attempt.loc[features_by_attempt["status"] == VALID]
    hands = _hand_level(valid, [*MEASUREMENT_KEYS, "attempt"])
    flags = flag_outliers(hands, [*MEASUREMENT_KEYS, "hand"], OUTLIER_FEATURES)
    hands = hands.assign(attempt_outlier=flags["is_outlier"], attempt_outlier_reason=flags["outlier_reason"])
    result = features_by_attempt.merge(
        hands[[*MEASUREMENT_KEYS, "attempt", "hand", "attempt_outlier", "attempt_outlier_reason"]],
        on=[*MEASUREMENT_KEYS, "attempt", "hand"], how="left",
    )
    result["attempt_outlier"] = result["attempt_outlier"].fillna(False).astype(bool)
    result["attempt_outlier_reason"] = result["attempt_outlier_reason"].fillna("")
    return result


def _mean_features(used: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    means = used.groupby(keys, dropna=False)[FINGER_FEATURE_COLUMNS].mean()
    means["finger_inactive"] = means["mean_force"] < FINGER_INACTIVE_MAX_FORCE
    return means


def aggregate_by_measurement(features_by_attempt: pd.DataFrame) -> pd.DataFrame:
    """One row per ``measurement x hand x finger``; adds measurement outlier flags."""

    keys = [*MEASUREMENT_KEYS, *FINGER_KEYS]
    used = features_by_attempt.loc[
        (features_by_attempt["status"] == VALID) & ~features_by_attempt["attempt_outlier"]
    ]
    grouped = features_by_attempt.groupby(keys, dropna=False)
    table = grouped[[*CONTEXT_COLUMNS, "timestamp"]].first()
    table["n_attempts"] = grouped.size()
    table["n_valid_attempts"] = grouped["status"].apply(lambda status: int((status == VALID).sum()))
    table["n_used_attempts"] = used.groupby(keys, dropna=False).size().reindex(table.index).fillna(0).astype(int)
    table = table.join(_mean_features(used, keys)).reset_index()

    hands = _hand_level(table, MEASUREMENT_KEYS)
    flags = flag_outliers(hands, [*PERSON_KEYS, "hand"], OUTLIER_FEATURES)
    hands = hands.assign(measurement_outlier=flags["is_outlier"],
                         measurement_outlier_reason=flags["outlier_reason"])
    table = table.merge(hands[[*MEASUREMENT_KEYS, "hand", "measurement_outlier", "measurement_outlier_reason"]],
                        on=[*MEASUREMENT_KEYS, "hand"], how="left")
    return table


def aggregate_by_person(features_by_measurement: pd.DataFrame) -> pd.DataFrame:
    """One row per ``person x step x hand x finger`` (mean of non-outlier measurements)."""

    keys = [*PERSON_KEYS, *FINGER_KEYS]
    used = features_by_measurement.loc[
        ~features_by_measurement["measurement_outlier"] & (features_by_measurement["n_used_attempts"] > 0)
    ]
    grouped = features_by_measurement.groupby(keys, dropna=False)
    table = grouped[CONTEXT_COLUMNS].first()
    table["n_measurements"] = grouped.size()
    table["n_used_measurements"] = used.groupby(keys, dropna=False).size().reindex(table.index).fillna(0).astype(int)
    table["n_used_attempts"] = used.groupby(keys, dropna=False)["n_used_attempts"].sum().reindex(table.index) \
        .fillna(0).astype(int)
    return table.join(_mean_features(used, keys)).reset_index()


def classify_hand_activity(left_force: float, right_force: float) -> tuple[float, str]:
    """Return ``(activity_ratio, inactive_hand)`` from the mean finger force of each hand.

    ``activity_ratio`` = weaker / stronger hand (negative forces count as 0).
    ``inactive_hand``: "left" / "right" when the ratio is below
    ``INACTIVE_HAND_MAX_ACTIVITY_RATIO``, "none" otherwise, "unknown" when a
    hand has no valid data or neither hand presses.
    """

    if not (np.isfinite(left_force) and np.isfinite(right_force)):
        return np.nan, "unknown"
    left, right = max(left_force, 0.0), max(right_force, 0.0)
    if max(left, right) < FINGER_INACTIVE_MAX_FORCE:
        return np.nan, "unknown"
    ratio = min(left, right) / max(left, right)
    if ratio >= INACTIVE_HAND_MAX_ACTIVITY_RATIO:
        return ratio, "none"
    return ratio, "left" if left < right else "right"


def summarize_persons(
    features_by_attempt: pd.DataFrame,
    features_by_measurement: pd.DataFrame,
    features_by_person: pd.DataFrame,
) -> pd.DataFrame:
    """One row per ``person x step``: demographics, data counts and hand activity."""

    attempts = features_by_attempt.groupby(PERSON_KEYS, dropna=False)
    summary = attempts[CONTEXT_COLUMNS].first()
    summary["n_measurements"] = attempts["measurement_id"].nunique()
    summary["n_attempts"] = attempts.apply(lambda g: len(g.drop_duplicates(["measurement_id", "attempt"])))
    summary["n_finger_rows"] = attempts.size()
    summary["n_finger_rows_valid"] = attempts["status"].apply(lambda s: int((s == VALID).sum()))
    summary["n_outlier_attempt_hands"] = (
        features_by_attempt.loc[features_by_attempt["attempt_outlier"]]
        .drop_duplicates([*MEASUREMENT_KEYS, "attempt", "hand"]).groupby(PERSON_KEYS).size()
    )
    summary["n_outlier_measurement_hands"] = (
        features_by_measurement.loc[features_by_measurement["measurement_outlier"]]
        .drop_duplicates([*MEASUREMENT_KEYS, "hand"]).groupby(PERSON_KEYS).size()
    )
    summary[["n_outlier_attempt_hands", "n_outlier_measurement_hands"]] = (
        summary[["n_outlier_attempt_hands", "n_outlier_measurement_hands"]].fillna(0).astype(int)
    )

    person = features_by_person.groupby([*PERSON_KEYS, "hand"])
    hand_force = person["mean_force"].mean().unstack("hand")
    inactive_fingers = person["finger_inactive"].sum().unstack("hand")
    for hand in HANDS:
        summary[f"{hand}_mean_force"] = hand_force.get(hand)
        summary[f"{hand}_n_inactive_fingers"] = inactive_fingers.get(hand)
    activity = [classify_hand_activity(left, right)
                for left, right in zip(summary["left_mean_force"], summary["right_mean_force"])]
    summary["activity_ratio"] = [item[0] for item in activity]
    summary["inactive_hand"] = [item[1] for item in activity]
    return summary.reset_index().sort_values(["is_patient", "patient_id", "step_id"]).reset_index(drop=True)


def build_wide_table(features_by_person: pd.DataFrame, person_summary: pd.DataFrame) -> pd.DataFrame:
    """One row per person with every finger feature as a column.

    Controls enter with the primary control step only (four attempts, like the
    patient step), so every person appears exactly once. Columns are named
    ``<hand>_<finger_name>_<feature>`` plus ``<hand>_<finger_name>_inactive``.
    """

    from rehapiano.exact_force.stats.selection import primary_step  # local import: stats builds on this module

    fingers = primary_step(features_by_person).sort_values(["hand", "finger"])
    fingers["column_prefix"] = fingers["hand"] + "_" + fingers["finger_name"]
    fingers = fingers.rename(columns={"finger_inactive": "inactive"})
    wide = fingers.pivot(index="patient_id", columns="column_prefix",
                         values=[*FINGER_FEATURE_COLUMNS, "inactive"])
    prefixes = fingers["column_prefix"].drop_duplicates()  # hand, then finger order
    ordered = [(value, prefix) for prefix in prefixes for value in [*FINGER_FEATURE_COLUMNS, "inactive"]]
    wide = wide[ordered]
    wide.columns = [f"{prefix}_{value}" for value, prefix in ordered]

    summary_columns = ["patient_id", "group", "is_patient", "age", "age_group", "gender", "metadata_hand",
                       "step_id", "n_measurements", "n_attempts", "left_mean_force", "right_mean_force",
                       "activity_ratio", "inactive_hand"]
    summary = primary_step(person_summary)[summary_columns]
    return (summary.merge(wide.reset_index(), on="patient_id", how="left")
            .sort_values(["is_patient", "patient_id"]).reset_index(drop=True))
