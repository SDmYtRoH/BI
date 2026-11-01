"""Choice of the rows that enter group comparisons."""

from __future__ import annotations

import pandas as pd

from rehapiano.exact_force.config import PATIENT_STEP_IDS, PRIMARY_CONTROL_STEP_ID


def primary_step(table: pd.DataFrame) -> pd.DataFrame:
    """Keep one step per person: the primary control step and the patient step.

    Both have four attempts per measurement, so the person means are built in
    the same way in both groups.
    """

    keep = table["step_id"].isin([PRIMARY_CONTROL_STEP_ID, *PATIENT_STEP_IDS])
    return table.loc[keep].copy()
