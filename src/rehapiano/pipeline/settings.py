"""Paths, games and analysis choices of the unified pipeline.

Kept apart from ``rehapiano.config`` (the per-finger EXACT_FORCE pipeline) so both can evolve independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from rehapiano import config

# --------------------------------------------------------------------------- paths
MEASUREMENTS_DIR = config.MEASUREMENTS_DIR  # searched recursively for results_* files
METADATA_DIR = config.METADATA_DIR  # every *.csv with the metadata columns is read
PATIENT_METADATA_FILES = ("dataset_patients.csv",)  # rows in these files are the patient cohort
QUESTIONNAIRE_CSV = config.DATA_DIR / "sources" / "questionnaire.csv"
OUTPUT_DIR = config.OUTPUT_DIR / "pipeline"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"

# Questionnaire question about a condition affecting hand strength or coordination (answers "Áno" / "Nie").
QUESTIONNAIRE_ID_COLUMN = "identifikačné číslo"
QUESTIONNAIRE_HEALTH_COLUMN = "Trpíte ochorením"


# --------------------------------------------------------------------------- games
@dataclass(frozen=True)
class Game:
    """One game (``step_type``) and the game steps that enter the analysis."""

    step_type: str
    steps: Dict[str, str]  # step_id -> cohort that played it ("control" / "patient")
    crop_s: Tuple[float, float]  # analysed part of every attempt
    practice_attempts: Tuple[int, ...] = field(default_factory=tuple)  # attempts left out of the person values


GAMES = {
    "exact_force": Game(
        step_type="EXACT_FORCE",
        # d475bc4a-... (controls, 1 attempt) is not used: its equivalence to the 4-attempt steps is unverified.
        steps={"5867ebbc-a463-4d20-8b20-50df7c64ace7": "control", "d2e6b7bd-a952-4652-afe3-9888e29ac205": "patient"},
        crop_s=(config.CROP_START_S, config.CROP_END_S),
    ),
    "rhythm": Game(
        step_type="RHYTHM",
        steps={"1e177cdd-0baf-42fe-b297-fbeba73ea828": "control", "65a5a4b1-907c-4717-b47b-cd8f31eb3e4b": "patient"},
        crop_s=(2.0, 30.0),
        practice_attempts=(1,),  # people press slower and harder in the first attempt (week 6)
    ),
}

# --------------------------------------------------------------------------- feature definitions
# EXACT_FORCE bands (Hz); low <= f < high, the last band includes its upper edge.
EF_BANDS = {"slow": (0.2, 3.0), "boundary": (3.0, 4.0), "tremor": (4.0, 15.0)}

RHYTHM_MIN_PRESSES = 5
RHYTHM_PRESS_MIN_DISTANCE_S = 0.15
RHYTHM_PRESS_MIN_PROMINENCE = 0.5  # share of the attempt's 5-95 % range of the summed force
RHYTHM_MIN_ATTEMPT_S = 20.0

# --------------------------------------------------------------------------- analysis
DECADE_YEARS = 10
REFERENCE_QUANTILES = (0.025, 0.975)
MIN_PEOPLE_FOR_DECADE_RANGE = 20  # fewer people cannot give a 2.5-97.5 % range
OLDER_AGE = 50
CV_FOLDS = 5
CV_REPEATS = 20
PERMUTATIONS = 2000  # decade-stratified group test
CLASSIFIER_PERMUTATIONS = 200
RANDOM_SEED = 0
