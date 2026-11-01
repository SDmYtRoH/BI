"""Person-level parameters of the RHYTHM test and their factors (week 5, M. Kuznietsov).

Input is ``rhythm_parameters_by_attempt.csv`` (``scripts/build_rhythm.py``). The first attempt of every
recording is a practice attempt and is left out by default. The statistics (hypotheses, norms, classifier,
ablation) are in :mod:`rehapiano.common.stats.group_analysis`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from rehapiano.common.stats.group_analysis import add_decade, add_screening
from rehapiano.rhythm.config import PRACTICE_ATTEMPTS
from rehapiano.rhythm.presses import RHYTHM_PARAMETER_COLUMNS

# Groups of parameters (factors) for the ablation; names are Slovak because they go straight to the notebook.
PARAMETER_GROUPS: Dict[str, List[str]] = {
    "tempo": ["press_rate_hz"],
    "pravidelnosť rytmu": ["interval_cv"],
    "sila stlačenia": ["amplitude_strong", "amplitude_weak"],
    "rovnováha rúk": ["balance"],
    "stálosť sily": ["amplitude_cv"],
    "súčasnosť rúk": ["sync_lag_ms"],
    "uvoľnenie": ["release_ratio"],
    "únava": ["fatigue_amplitude", "fatigue_rate"],
}
PARAMETERS = [p for group in PARAMETER_GROUPS.values() for p in group]
assert sorted(PARAMETERS) == sorted(RHYTHM_PARAMETER_COLUMNS)

# Expected direction in patients (H1, one-sided reading of the two-sided test): +1 higher, -1 lower.
EXPECTED_DIRECTION = {
    "press_rate_hz": -1, "interval_cv": +1, "amplitude_strong": -1, "amplitude_weak": -1, "balance": -1,
    "amplitude_cv": +1, "sync_lag_ms": +1, "release_ratio": +1, "fatigue_amplitude": -1, "fatigue_rate": -1,
}
# Patients press slower; differences that disappear at the same press rate are only a consequence of the tempo.
COVARIATE = "press_rate_hz"


def person_parameters(by_attempt: pd.DataFrame, labels: Optional[pd.DataFrame] = None,
                      skip_first_attempt: bool = True) -> pd.DataFrame:
    """One row per person: median over attempts of a recording, mean over the person's recordings.

    ``labels`` = ``data/participant_labels.csv``; adds ``screened`` (healthy without a reported condition
    in the questionnaire, or a patient).
    """

    valid = by_attempt.loc[by_attempt["status"] == "valid"]
    if skip_first_attempt:
        valid = valid.loc[~valid["attempt"].isin(PRACTICE_ATTEMPTS)]
    per_recording = valid.groupby(["patient_id", "measurement_id"])[PARAMETERS].median()
    table = per_recording.groupby("patient_id").mean()
    info = by_attempt.groupby("patient_id").agg(group=("group", "first"), age=("age", "first"),
                                                gender=("gender", "first"),
                                                n_recordings=("measurement_id", "nunique"))
    info["n_attempts"] = valid.groupby("patient_id").size()
    table = info.join(table).reset_index()
    table["patient"] = (table["group"] == "patient").astype(int)
    table = add_decade(table)
    if labels is not None:
        table = add_screening(table, labels)
    return table
