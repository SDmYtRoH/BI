"""Stage 4: attempt -> session (median over attempts) -> person (mean over sessions).

One person is always one statistical unit. Writes ``person_features.csv`` (one row per person: labels, features
of both games, data-quality counts) and ``person_finger.csv`` (person x hand x finger, EXACT_FORCE).
Missing evidence stays missing: a hand without an active finger has no oscillation value, a person without a
game has no values for that game.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rehapiano.config import FINGER_INACTIVE_MAX_FORCE
from rehapiano.features.rhythm_game import PARAMETERS as RHYTHM_PARAMETERS
from rehapiano.pipeline import settings
from rehapiano.quality import VALID

logger = logging.getLogger(__name__)

SESSION = ["patient_id", "measurement_id"]
HAND_VALUES = ["mean_force", "slow_amplitude", "boundary_amplitude", "tremor_amplitude", "tremor_share",
               "active_fingers"]


def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    high = pd.concat([a, b], axis=1).max(axis=1)
    return pd.concat([a, b], axis=1).min(axis=1) / high.where(high > 0)


def exact_force_person(hands: pd.DataFrame) -> pd.DataFrame:
    per_hand = (hands.groupby([*SESSION, "hand"])[HAND_VALUES].median()
                .groupby(["patient_id", "hand"]).mean().unstack("hand"))
    per_hand.columns = [f"{value}_{hand}" for value, hand in per_hand.columns]
    p = per_hand
    out = pd.DataFrame(index=p.index)
    out["ef_force_left"], out["ef_force_right"] = p["mean_force_left"], p["mean_force_right"]
    out["ef_balance"] = _ratio(p["mean_force_left"], p["mean_force_right"])
    out["ef_weaker_hand"] = np.where(p["mean_force_left"] < p["mean_force_right"], "left", "right")
    tremor = p[["tremor_amplitude_left", "tremor_amplitude_right"]]
    out["ef_tremor_amp"] = tremor.mean(axis=1)
    out["ef_tremor_amp_max"] = tremor.max(axis=1)
    out["ef_slow_amp"] = p[["slow_amplitude_left", "slow_amplitude_right"]].mean(axis=1)
    out["ef_tremor_share"] = p[["tremor_share_left", "tremor_share_right"]].mean(axis=1)
    out["ef_tremor_balance"] = _ratio(p["tremor_amplitude_left"], p["tremor_amplitude_right"])
    out["ef_hands_without_active_finger"] = tremor.isna().sum(axis=1)
    out["ef_sessions"] = hands.groupby("patient_id")["measurement_id"].nunique()
    out["ef_attempts"] = hands.drop_duplicates([*SESSION, "attempt"]).groupby("patient_id").size()
    return out.reset_index()


def exact_force_fingers(fingers: pd.DataFrame) -> pd.DataFrame:
    valid = fingers.loc[fingers["status"] == VALID].copy()
    valid.loc[valid["finger_inactive"].astype(bool), ["slow_amplitude", "boundary_amplitude",
                                                      "tremor_mean_amplitude"]] = np.nan
    values = ["mean_force", "slow_amplitude", "boundary_amplitude", "tremor_mean_amplitude"]
    keys = ["hand", "finger", "finger_name"]
    person = (valid.groupby([*SESSION, *keys])[values].median().groupby(["patient_id", *keys]).mean().reset_index())
    person["finger_inactive"] = person["mean_force"] < FINGER_INACTIVE_MAX_FORCE
    person.loc[person["finger_inactive"], ["slow_amplitude", "boundary_amplitude", "tremor_mean_amplitude"]] = np.nan
    return person.rename(columns={"tremor_mean_amplitude": "tremor_amplitude"})


def rhythm_person(attempts: pd.DataFrame) -> pd.DataFrame:
    practice = settings.GAMES["rhythm"].practice_attempts
    used = attempts.loc[attempts["valid"].astype(bool) & ~attempts["attempt"].isin(practice)]
    person = used.groupby(SESSION)[RHYTHM_PARAMETERS].median().groupby("patient_id").mean()
    person.columns = [f"rh_{c}" for c in person.columns]
    person["rh_weaker_hand"] = used.groupby("patient_id")["weaker_hand"].agg(lambda v: v.mode().iloc[0])
    person["rh_sessions"] = used.groupby("patient_id")["measurement_id"].nunique()
    person["rh_attempts"] = used.groupby("patient_id").size()
    return person.reset_index()


def run(output_dir: Path = settings.OUTPUT_DIR) -> pd.DataFrame:
    read = lambda name: pd.read_csv(output_dir / f"{name}.csv", dtype={"patient_id": str})  # noqa: E731
    people = read("participants")
    people = (people.merge(exact_force_person(read("ef_attempt_hand")), on="patient_id", how="left")
              .merge(rhythm_person(read("rhythm_attempt")), on="patient_id", how="left"))
    people.to_csv(output_dir / "person_features.csv", index=False)
    fingers = exact_force_fingers(read("ef_attempt_finger")).merge(
        people[["patient_id", "label", "age", "decade"]], on="patient_id")
    fingers.to_csv(output_dir / "person_finger.csv", index=False)
    logger.info("person_features: %d people, %d columns", len(people), people.shape[1])
    return people
