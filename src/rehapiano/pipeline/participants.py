"""Stage 2: one row per person with cohort, questionnaire screening and plain-language label.

Labels used in all reports:

* ``Sick``          - recruited into the patient cohort (no diagnosis available);
* ``Healthy``       - control whose questionnaire (exactly one matching answer) reports no condition affecting
                      hand strength or coordination;
* ``Other control`` - control with a reported condition, no usable questionnaire or a duplicated questionnaire ID.
                      Kept in descriptive tables, left out of healthy-vs-sick comparisons.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rehapiano.pipeline import settings

logger = logging.getLogger(__name__)

HEALTHY, SICK, OTHER = "Healthy", "Sick", "Other control"


def _column(frame: pd.DataFrame, fragment: str) -> str:
    matches = [c for c in frame.columns if fragment.lower() in c.lower()]
    if len(matches) != 1:
        raise ValueError(f"questionnaire: expected one column containing '{fragment}', found {matches}")
    return matches[0]


def questionnaire_status(patient_ids: pd.Series, questionnaire_csv: Path) -> pd.DataFrame:
    """``label_status`` of every control from the questionnaire (week-2 rule)."""

    if not questionnaire_csv.exists():
        logger.warning("questionnaire not found (%s): every control becomes 'Other control'", questionnaire_csv)
        return pd.DataFrame({"patient_id": patient_ids, "label_status": "no_questionnaire", "questionnaire_matches": 0})
    answers = pd.read_csv(questionnaire_csv, dtype=str)
    ids = answers[_column(answers, settings.QUESTIONNAIRE_ID_COLUMN)].str.strip()
    health = answers[_column(answers, settings.QUESTIONNAIRE_HEALTH_COLUMN)].str.strip()
    rows = []
    for pid in patient_ids:
        matched = health[ids == str(pid)].dropna().unique().tolist()
        n = int((ids == str(pid)).sum())
        if n > 1:
            status = "duplicate_questionnaire_id"
        elif matched == ["Nie"]:
            status = "no_reported_condition"
        elif matched == ["Áno"]:
            status = "reported_condition"
        else:
            status = "unknown_or_conflicting"
        rows.append({"patient_id": pid, "label_status": status, "questionnaire_matches": n})
    return pd.DataFrame(rows)


def build_participants(index: pd.DataFrame, metadata: pd.DataFrame,
                       questionnaire_csv: Path = settings.QUESTIONNAIRE_CSV) -> pd.DataFrame:
    used = index.loc[index["status"] == "used"]
    people = (metadata.loc[metadata["patient_id"].isin(used["patient_id"])]
              .assign(timestamp=lambda t: pd.to_datetime(t["timestamp"], format="mixed", errors="coerce"))
              .sort_values("timestamp")
              .groupby("patient_id")
              .agg(cohort=("cohort", "first"), age=("age", "first"), gender=("gender", "first"),
                   metadata_hand=("hand", "first"), first_recording=("timestamp", "min"),
                   source_files=("source_files", "first"), n_cohorts=("cohort", "nunique"),
                   n_ages=("age", "nunique"))
              .reset_index())
    for column, what in [("n_cohorts", "cohort"), ("n_ages", "age")]:
        if (people[column] > 1).any():
            logger.warning("%d people have more than one %s in the metadata; the first is used",
                           int((people[column] > 1).sum()), what)
    people["age"] = pd.to_numeric(people["age"], errors="coerce")
    controls = people.loc[people["cohort"] == "control", "patient_id"]
    people = people.merge(questionnaire_status(controls, questionnaire_csv), on="patient_id", how="left")
    people["label_status"] = people["label_status"].fillna("patient_cohort")
    people["questionnaire_matches"] = people["questionnaire_matches"].fillna(0).astype(int)
    people["label"] = np.select([people["cohort"].eq("patient"), people["label_status"].eq("no_reported_condition")],
                                [SICK, HEALTHY], OTHER)
    width = settings.DECADE_YEARS
    people["decade"] = people["age"].map(
        lambda a: f"{int(a // width * width)}–{int(a // width * width) + width - 1}" if pd.notna(a) else None)
    people["recording_period"] = people["first_recording"].dt.strftime("%Y-%m")
    games = used.pivot_table(index="patient_id", columns="game", values="measurement_id", aggfunc="nunique")
    people = people.merge(games.add_prefix("sessions_").reset_index(), on="patient_id", how="left")
    columns = ["patient_id", "label", "cohort", "label_status", "questionnaire_matches", "age", "decade", "gender",
               "metadata_hand", "recording_period", "first_recording", "source_files",
               *[c for c in people.columns if c.startswith("sessions_")]]
    return people[columns].sort_values(["label", "patient_id"]).reset_index(drop=True)


def run(output_dir: Path = settings.OUTPUT_DIR, questionnaire_csv: Path = settings.QUESTIONNAIRE_CSV) -> pd.DataFrame:
    index = pd.read_csv(output_dir / "recordings.csv", dtype={"patient_id": str})
    metadata = pd.read_csv(output_dir / "cache" / "metadata.csv", dtype=str)
    people = build_participants(index, metadata, questionnaire_csv)
    people.to_csv(output_dir / "participants.csv", index=False)
    logger.info("participants: %s", people["label"].value_counts().to_dict())
    return people
