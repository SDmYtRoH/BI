"""Both exercises together: one person table with the EXACT_FORCE and RHYTHM parameters.

Used for the classifier healthy vs patients on both tests and for the ablation of the exercises and
their factors. Column prefixes: ``ef_`` (EXACT_FORCE, classifier features incl. log10 amplitudes) and
``rh_`` (RHYTHM). Only people with both tests enter.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from rehapiano.exact_force import parameters as ep
from rehapiano.rhythm import parameters as rp

EF_GROUPS: Dict[str, List[str]] = {f"EF: {name}": [f"ef_{f}" for f in features] for name, features in ep.MODEL_GROUPS.items()}
RH_GROUPS: Dict[str, List[str]] = {f"rytmus: {name}": [f"rh_{f}" for f in features]
                                   for name, features in rp.PARAMETER_GROUPS.items()}
EF_FEATURES = [f for features in EF_GROUPS.values() for f in features]
RH_FEATURES = [f for features in RH_GROUPS.values() for f in features]
FACTOR_GROUPS = {**EF_GROUPS, **RH_GROUPS}
EXERCISE_GROUPS = {"EXACT_FORCE": EF_FEATURES, "RHYTHM": RH_FEATURES}

INFO = ["patient_id", "group", "patient", "age", "gender", "decade", "screened"]


def combined_people(exact_force: pd.DataFrame, rhythm: pd.DataFrame) -> pd.DataFrame:
    """Join the person tables of both exercises (``parameters.person_parameters`` of each)."""

    ef = exact_force[INFO + ep.MODEL_FEATURES].rename(columns={f: f"ef_{f}" for f in ep.MODEL_FEATURES})
    rh = rhythm[["patient_id", *rp.PARAMETERS]].rename(columns={f: f"rh_{f}" for f in rp.PARAMETERS})
    return ef.merge(rh, on="patient_id", how="inner").sort_values("patient_id").reset_index(drop=True)
