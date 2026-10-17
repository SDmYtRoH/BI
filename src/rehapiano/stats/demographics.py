"""Age and gender structure of the two groups (one row per person)."""

from __future__ import annotations

import pandas as pd

from rehapiano.config import AGE_GROUP_WIDTH_YEARS
from rehapiano.stats.tests import mann_whitney

GROUP_ORDER = ["control", "patient"]


def persons(person_summary: pd.DataFrame) -> pd.DataFrame:
    """One row per person (controls appear once per step in ``person_summary``)."""

    return (person_summary.groupby("patient_id", as_index=False)
            .agg(group=("group", "first"), age=("age", "first"), age_group=("age_group", "first"),
                 gender=("gender", "first"), metadata_hand=("metadata_hand", "first"),
                 n_measurements=("n_measurements", "sum")))


def describe_groups(people: pd.DataFrame) -> pd.DataFrame:
    """Count, age statistics, gender split and measurements per group."""

    rows = []
    for group in GROUP_ORDER:
        frame = people.loc[people["group"] == group]
        age = frame["age"]
        rows.append({
            "group": group, "n_people": len(frame), "n_measurements": int(frame["n_measurements"].sum()),
            "age_mean": age.mean(), "age_sd": age.std(), "age_median": age.median(),
            "age_q1": age.quantile(0.25), "age_q3": age.quantile(0.75), "age_min": age.min(), "age_max": age.max(),
            "n_female": int((frame["gender"] == "female").sum()), "n_male": int((frame["gender"] == "male").sum()),
            "n_hand_right": int((frame["metadata_hand"] == "right").sum()),
            "n_hand_left": int((frame["metadata_hand"] == "left").sum()),
            "n_hand_unknown": int(frame["metadata_hand"].isna().sum()),
        })
    return pd.DataFrame(rows)


def age_histogram(people: pd.DataFrame, width: int = AGE_GROUP_WIDTH_YEARS) -> pd.DataFrame:
    """People per ``width``-year age group and group; all groups between min and max age are listed."""

    low = int(people["age"].min() // width * width)
    high = int(people["age"].max() // width * width)
    labels = [f"{start}-{start + width - 1}" for start in range(low, high + 1, width)]
    counts = pd.crosstab(people["age_group"], people["group"]).reindex(index=labels, columns=GROUP_ORDER,
                                                                       fill_value=0)
    counts.index.name = "age_group"
    counts["total"] = counts.sum(axis=1)
    for group in GROUP_ORDER:
        counts[f"{group}_percent"] = 100 * counts[group] / counts[group].sum()
    return counts.reset_index()


def age_difference_test(people: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney test of age between controls (a) and patients (b)."""

    control = people.loc[people["group"] == "control", "age"]
    patient = people.loc[people["group"] == "patient", "age"]
    return pd.DataFrame([{"variable": "age", **mann_whitney(control, patient)}])


def gender_by_age_group(people: pd.DataFrame) -> pd.DataFrame:
    """People per age group, group and gender (long format)."""

    counts = people.groupby(["age_group", "group", "gender"]).size().rename("n_people").reset_index()
    return counts.sort_values(["age_group", "group", "gender"]).reset_index(drop=True)

