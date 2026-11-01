"""Person-level parameters of the EXACT_FORCE test and their factors (teacher's week 5 task).

Built from the per-finger person table (``features_by_person_finger.csv``) and the person summary of
``scripts/build_exact_force.py``. Only the primary step of every person is used (the control step with
4 attempts, like the patient step), and tremor values of fingers that do not press are left out
(they describe sensor noise). Every parameter is one number per person:

* finger features are first summarised per hand (median over active fingers), then over both hands
  (mean of the two hands), so a person with one tremulous hand is not averaged away by finger counts;
* hand balance = weaker / stronger hand (as in week 3);
* finger spread = how different the fingers of one hand are (coefficient of variation, mean of hands).

Amplitudes are strongly right-skewed, so the classifier uses their log10 (``MODEL_FEATURES``); the
hypothesis tests are rank-based and the norms are quantiles, so they use the raw values.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from rehapiano.common.stats.group_analysis import add_decade, add_screening
from rehapiano.exact_force.stats.finger_tremor import active_finger_table, finger_spread
from rehapiano.exact_force.stats.hand_activity import hand_activity_table

# Per-finger features summarised per person: person parameter -> finger feature.
FINGER_PARAMETERS = {
    "tremor_amplitude": "tremor_mean_amplitude",
    "movement_amplitude": "movement_mean_amplitude",
    "tremor_to_movement": "tremor_to_movement_ratio",
    "tremor_frequency_hz": "tremor_peak_frequency_hz",
    "tremor_prominence": "tremor_peak_prominence",
}

# Factors for the ablation; names are Slovak because they go straight to the notebook.
PARAMETER_GROUPS: Dict[str, List[str]] = {
    "amplitúda trasu": ["tremor_amplitude", "tremor_amplitude_worse_hand"],
    "pomalé korekcie sily": ["movement_amplitude"],
    "podiel trasu": ["tremor_to_movement"],
    "frekvencia trasu": ["tremor_frequency_hz"],
    "rytmickosť trasu": ["tremor_prominence"],
    "sila": ["mean_force"],
    "rovnováha rúk": ["force_balance", "tremor_balance"],
    "rozdiely medzi prstami": ["finger_spread"],
}
PARAMETERS = [p for group in PARAMETER_GROUPS.values() for p in group]
LOG_PARAMETERS = ["tremor_amplitude", "tremor_amplitude_worse_hand", "movement_amplitude", "mean_force"]
# The classifier works on log10 amplitudes; the factors keep their names.
MODEL_FEATURES = [f"log_{p}" if p in LOG_PARAMETERS else p for p in PARAMETERS]
MODEL_GROUPS: Dict[str, List[str]] = {name: [f"log_{p}" if p in LOG_PARAMETERS else p for p in features]
                                      for name, features in PARAMETER_GROUPS.items()}

# Expected direction in patients (H1): +1 higher, -1 lower.
EXPECTED_DIRECTION = {
    "tremor_amplitude": +1, "tremor_amplitude_worse_hand": +1, "movement_amplitude": +1, "tremor_to_movement": +1,
    "tremor_frequency_hz": -1, "tremor_prominence": +1, "mean_force": -1, "force_balance": -1,
    "tremor_balance": -1, "finger_spread": +1,
}
# Tremor amplitude grows with the force a finger holds: the disease effect is also estimated at equal force.
COVARIATE = "mean_force"


def person_parameters(features_by_person: pd.DataFrame, person_summary: pd.DataFrame,
                      labels: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per person with ``PARAMETERS``, their log10 versions and ``screened`` (see ``add_screening``)."""

    fingers = active_finger_table(features_by_person)  # primary step, inactive fingers -> NaN
    by_hand = fingers.groupby(["patient_id", "hand"])[list(FINGER_PARAMETERS.values())].median()
    by_hand.columns = list(FINGER_PARAMETERS)
    table = by_hand.groupby("patient_id").mean()
    table["tremor_amplitude_worse_hand"] = by_hand["tremor_amplitude"].groupby("patient_id").max()

    activity = hand_activity_table(person_summary, features_by_person).set_index("patient_id")
    table["mean_force"] = activity[["left_mean_force", "right_mean_force"]].mean(axis=1)
    table["force_balance"] = activity["force_ratio"]
    table["tremor_balance"] = activity["fluctuation_ratio"]

    spread = finger_spread(fingers, features=("tremor_to_movement_ratio",))
    table["finger_spread"] = spread.groupby("patient_id")["tremor_to_movement_ratio_cv"].mean()

    info = activity[["group", "age", "gender"]]
    counts = person_summary.loc[person_summary["patient_id"].isin(info.index)]
    counts = counts.loc[counts["step_id"].isin(fingers["step_id"].unique())].set_index("patient_id")
    info = info.assign(n_recordings=counts["n_measurements"], n_attempts=counts["n_attempts"])
    table = info.join(table[PARAMETERS]).reset_index()
    for parameter in LOG_PARAMETERS:
        values = table[parameter]
        table[f"log_{parameter}"] = np.log10(values.where(values > 0))
    table["patient"] = (table["group"] == "patient").astype(int)
    table = add_decade(table)
    if labels is not None:
        table = add_screening(table, labels)
    return table.sort_values("patient_id").reset_index(drop=True)
