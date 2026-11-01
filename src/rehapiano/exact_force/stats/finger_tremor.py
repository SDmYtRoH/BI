"""Per-finger tremor analysis on person-level finger features.

Questions and methods:

1. Do some fingers tremble more than others?  -> within each group and hand,
   Friedman test over the five fingers (persons are repeated measures) with
   Kendall's W, then Holm-adjusted pairwise Wilcoxon tests; plus how often each
   finger is the most tremulous one.
2. How different is the tremor between the fingers of one person?  ->
   between-finger spread per person and hand (coefficient of variation and
   max / min), compared between groups.
3. Do patients and controls differ for a given finger?  -> Mann-Whitney per
   hand, finger and feature, for everyone and for age >= 50 only (controls are
   much younger), Holm-adjusted within each feature.
4. Left vs right hand for the same finger  -> paired Wilcoxon per group.

Tremor features of inactive fingers (``finger_inactive``) describe sensor
noise, not tremor, so they are set to NaN before any comparison.
Persons enter with their primary step only (see ``stats.selection``).
Caveat for every group comparison: step_id is fully confounded with the group.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from rehapiano.exact_force.config import (
    AGE_MATCHED_MIN_AGE,
    FINGER_COMPARISON_FEATURES,
    MIN_ACTIVE_FINGERS_FOR_SPREAD,
)
from rehapiano.exact_force.features.finger import FINGER_FEATURE_COLUMNS
from rehapiano.common.stats.demographics import GROUP_ORDER
from rehapiano.exact_force.stats.selection import primary_step
from rehapiano.common.stats.tests import friedman, holm_adjust, mann_whitney, pairwise_wilcoxon, wilcoxon_paired

TREMOR_FEATURES = [column for column in FINGER_FEATURE_COLUMNS if column != "mean_force"]
PERSON_HAND = ["patient_id", "group", "age", "hand"]


def active_finger_table(features_by_person: pd.DataFrame) -> pd.DataFrame:
    """Primary-step person x hand x finger rows with tremor features of inactive fingers set to NaN."""

    table = primary_step(features_by_person)
    inactive = table["finger_inactive"].astype(bool)
    table.loc[inactive, TREMOR_FEATURES] = np.nan
    return table


def _subsets(table: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "all", table
    yield f"age>={AGE_MATCHED_MIN_AGE}", table.loc[table["age"] >= AGE_MATCHED_MIN_AGE]


def finger_descriptives(table: pd.DataFrame, features=FINGER_FEATURE_COLUMNS) -> pd.DataFrame:
    """n, median, quartiles and mean per group, hand, finger and feature."""

    long = table.melt(id_vars=["group", "hand", "finger", "finger_name"], value_vars=list(features),
                      var_name="feature").dropna(subset=["value"])
    grouped = long.groupby(["feature", "group", "hand", "finger", "finger_name"])["value"]
    result = grouped.agg(n="count", median="median", mean="mean",
                         q1=lambda v: v.quantile(0.25), q3=lambda v: v.quantile(0.75))
    return result.reset_index()


def _wide_by_finger(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    wide = frame.pivot_table(index="patient_id", columns="finger_name", values=feature)
    order = frame.drop_duplicates("finger").sort_values("finger")["finger_name"]
    return wide.reindex(columns=[name for name in order if name in wide.columns])


def finger_differences(table: pd.DataFrame, features=FINGER_COMPARISON_FEATURES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Friedman test over fingers and pairwise post-hoc tests, per group, hand and feature."""

    overall, pairwise = [], []
    for (group, hand), frame in table.groupby(["group", "hand"]):
        for feature in features:
            wide = _wide_by_finger(frame, feature)
            keys = {"group": group, "hand": hand, "feature": feature}
            overall.append({**keys, **friedman(wide), **wide.median().add_prefix("median_").to_dict()})
            pairs = pairwise_wilcoxon(wide)
            if not pairs.empty:
                pairwise.append(pairs.assign(**keys))
    pairwise_table = pd.concat(pairwise, ignore_index=True) if pairwise else pd.DataFrame()
    if not pairwise_table.empty:
        pairwise_table = pairwise_table[["group", "hand", "feature", *[c for c in pairwise_table
                                                                        if c not in ("group", "hand", "feature")]]]
    return pd.DataFrame(overall), pairwise_table


def most_tremulous_finger(table: pd.DataFrame, feature: str = "tremor_to_movement_ratio") -> pd.DataFrame:
    """How often each finger has the largest ``feature`` within a person's hand."""

    complete = table.dropna(subset=[feature])
    winners = complete.loc[complete.groupby(["patient_id", "hand"])[feature].idxmax()]
    counts = winners.groupby(["group", "hand", "finger", "finger_name"]).size().rename("n_people").reset_index()
    counts["percent"] = 100 * counts["n_people"] / counts.groupby(["group", "hand"])["n_people"].transform("sum")
    return counts.assign(feature=feature)


def finger_spread(table: pd.DataFrame, features=FINGER_COMPARISON_FEATURES) -> pd.DataFrame:
    """Between-finger spread per person and hand: CV, max / min and the extreme fingers."""

    rows = []
    for keys, frame in table.groupby(PERSON_HAND):
        row = dict(zip(PERSON_HAND, keys))
        for feature in features:
            values = frame.set_index("finger_name")[feature].dropna()
            enough = len(values) >= MIN_ACTIVE_FINGERS_FOR_SPREAD and values.mean() > 0
            row[f"{feature}_n_fingers"] = len(values)
            row[f"{feature}_cv"] = values.std() / values.mean() if enough else np.nan
            row[f"{feature}_max_min_ratio"] = values.max() / values.min() if enough and values.min() > 0 else np.nan
            row[f"{feature}_max_finger"] = values.idxmax() if enough else None
            row[f"{feature}_min_finger"] = values.idxmin() if enough else None
        rows.append(row)
    return pd.DataFrame(rows)


def spread_group_comparison(spread: pd.DataFrame, features=FINGER_COMPARISON_FEATURES) -> pd.DataFrame:
    """Mann-Whitney of the between-finger CV, controls (a) vs patients (b)."""

    rows = []
    for subset, frame in _subsets(spread):
        for hand, hand_frame in frame.groupby("hand"):
            for feature in features:
                column = f"{feature}_cv"
                groups = [hand_frame.loc[hand_frame["group"] == g, column] for g in GROUP_ORDER]
                rows.append({"subset": subset, "hand": hand, "feature": feature, "measure": "cv",
                             **mann_whitney(*groups)})
    result = pd.DataFrame(rows)
    result["p_holm"] = result.groupby(["subset", "feature"])["p"].transform(lambda p: holm_adjust(p))
    return result


def group_comparison(table: pd.DataFrame, features=FINGER_FEATURE_COLUMNS) -> pd.DataFrame:
    """Mann-Whitney per hand, finger and feature: controls (a) vs patients (b)."""

    rows = []
    for subset, frame in _subsets(table):
        for (hand, finger, finger_name), finger_frame in frame.groupby(["hand", "finger", "finger_name"]):
            for feature in features:
                groups = [finger_frame.loc[finger_frame["group"] == g, feature] for g in GROUP_ORDER]
                rows.append({"subset": subset, "hand": hand, "finger": finger, "finger_name": finger_name,
                             "feature": feature, **mann_whitney(*groups)})
    result = pd.DataFrame(rows)
    result["p_holm"] = result.groupby(["subset", "feature"])["p"].transform(lambda p: holm_adjust(p))
    return result


def left_right_comparison(table: pd.DataFrame, features=FINGER_COMPARISON_FEATURES) -> pd.DataFrame:
    """Paired Wilcoxon left (a) vs right (b) for the same finger, per group."""

    rows = []
    for (group, finger, finger_name), frame in table.groupby(["group", "finger", "finger_name"]):
        for feature in features:
            wide = frame.pivot_table(index="patient_id", columns="hand", values=feature)
            if {"left", "right"} <= set(wide.columns):
                rows.append({"group": group, "finger": finger, "finger_name": finger_name, "feature": feature,
                             **wilcoxon_paired(wide["left"], wide["right"])})
    result = pd.DataFrame(rows)
    result["p_holm"] = result.groupby(["group", "feature"])["p"].transform(lambda p: holm_adjust(p))
    return result


# --------------------------------------------------------------------------- reference-based analysis
# Raw tremor amplitude differs between fingers mostly because the fingers press
# with very different force (within a person the ranks of force and tremor
# amplitude correlate with rho ~0.6). Fingers are therefore never compared with
# each other directly: every finger is compared with the SAME finger of healthy
# reference controls, which cancels finger force and sensor differences.

REFERENCE_FEATURE = "tremor_mean_amplitude"


def reference_mask(table: pd.DataFrame, min_age: float = None) -> pd.Series:
    """Healthy controls of comparable age (``config.REFERENCE_MIN_AGE``)."""

    from rehapiano.exact_force.config import REFERENCE_MIN_AGE

    min_age = REFERENCE_MIN_AGE if min_age is None else min_age
    return (table["group"] == "control") & (table["age"] >= min_age)


def noise_comparison(table: pd.DataFrame, sensor_noise: pd.DataFrame, feature: str = REFERENCE_FEATURE) -> pd.DataFrame:
    """Signal-to-noise per group, hand and finger: median signal / median unloaded-sensor noise."""

    noise_median = sensor_noise[feature].median()
    noise_p95 = sensor_noise[feature].quantile(0.95)
    grouped = table.dropna(subset=[feature]).groupby(["group", "hand", "finger", "finger_name"])[feature]
    result = grouped.agg(n="count", median="median").reset_index()
    result["noise_median"] = noise_median
    result["signal_to_noise"] = result["median"] / noise_median
    result["percent_below_noise_p95"] = grouped.apply(lambda v: 100 * (v < noise_p95).mean()).to_numpy()
    return result


def _bootstrap_ratio_ci(a: np.ndarray, b: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    ratios = [np.median(rng.choice(b, len(b))) / np.median(rng.choice(a, len(a))) for _ in range(samples)]
    return tuple(np.percentile(ratios, [2.5, 97.5]))


def ratio_to_reference(table: pd.DataFrame, feature: str = REFERENCE_FEATURE) -> pd.DataFrame:
    """Patients vs reference controls per finger: ratio of medians, bootstrap 95 % CI, Mann-Whitney."""

    from rehapiano.common.config import RANDOM_SEED
    from rehapiano.exact_force.config import BOOTSTRAP_SAMPLES

    rng = np.random.default_rng(RANDOM_SEED)
    reference = reference_mask(table)
    rows = []
    for (hand, finger, finger_name), frame in table.groupby(["hand", "finger", "finger_name"]):
        controls = frame.loc[reference.loc[frame.index], feature].dropna().to_numpy()
        patients = frame.loc[frame["group"] == "patient", feature].dropna().to_numpy()
        low, high = _bootstrap_ratio_ci(controls, patients, BOOTSTRAP_SAMPLES, rng)
        rows.append({"hand": hand, "finger": finger, "finger_name": finger_name, "feature": feature,
                     "ratio": np.median(patients) / np.median(controls), "ci_low": low, "ci_high": high,
                     **mann_whitney(controls, patients)})
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p"])
    return result


def z_scores(table: pd.DataFrame, feature: str = REFERENCE_FEATURE) -> pd.DataFrame:
    """Robust z-score of log(feature) against the reference controls of the same hand and finger.

    z = (log x - median_ref) / (1.4826 * MAD_ref); z > 1.96 means higher than ~97.5 %
    of comparable healthy people for that very finger.
    """

    result = table[["patient_id", "group", "age", "hand", "finger", "finger_name", feature]].copy()
    result["log_value"] = np.log(result[feature].where(result[feature] > 0))
    reference = result.loc[reference_mask(result)].dropna(subset=["log_value"])
    stats_ref = reference.groupby(["hand", "finger"])["log_value"].agg(
        ref_n="count", ref_median="median", ref_mad=lambda v: 1.4826 * np.median(np.abs(v - np.median(v))))
    result = result.merge(stats_ref.reset_index(), on=["hand", "finger"], how="left")
    result["z"] = (result["log_value"] - result["ref_median"]) / result["ref_mad"]
    return result.drop(columns="log_value")


def tremor_pattern(z_table: pd.DataFrame) -> pd.DataFrame:
    """Per person: fingers with elevated tremor and the pattern.

    none: fewer than ``LOCALIZED_MIN_FINGERS`` elevated fingers; localized: up to
    ``GENERALIZED_MIN_FINGERS - 1``; generalized: at least ``GENERALIZED_MIN_FINGERS``.
    """

    from rehapiano.exact_force.config import ELEVATED_TREMOR_Z, GENERALIZED_MIN_FINGERS, LOCALIZED_MIN_FINGERS

    scored = z_table.dropna(subset=["z"]).assign(elevated=lambda t: t["z"] > ELEVATED_TREMOR_Z)
    scored["finger_label"] = scored["hand"] + "_" + scored["finger_name"]
    grouped = scored.groupby(["patient_id", "group", "age"])
    result = grouped.agg(n_fingers=("z", "size"), n_elevated=("elevated", "sum"), median_z=("z", "median"),
                         max_z=("z", "max")).reset_index()
    result["elevated_fingers"] = grouped.apply(
        lambda g: ", ".join(g.loc[g["elevated"], "finger_label"])).to_numpy()
    result["pattern"] = np.select(
        [result["n_elevated"] < LOCALIZED_MIN_FINGERS, result["n_elevated"] < GENERALIZED_MIN_FINGERS],
        ["none", "localized"], default="generalized")
    result["is_reference"] = reference_mask(result)
    return result.sort_values(["group", "n_elevated"], ascending=[True, False]).reset_index(drop=True)
