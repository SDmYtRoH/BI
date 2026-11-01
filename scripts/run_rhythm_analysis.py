"""RHYTHM step 3 (week 5, M. Kuznietsov): hypotheses, decade norms, classifier and ablation.

Run from the repository root (after build_rhythm.py):
    python scripts/run_rhythm_analysis.py
    python scripts/run_rhythm_analysis.py --cv-repeats 3 --permutations 20   # quick check
Writes CSV tables to ``outputs/rhythm/analysis/`` (full run: ~10-40 minutes).
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rehapiano.common import config  # noqa: E402
from rehapiano.common.stats import group_analysis as ga  # noqa: E402
from rehapiano.rhythm import config as rhythm_config, parameters as rp  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rhythm-dir", type=Path, default=rhythm_config.RHYTHM_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=rhythm_config.ANALYSIS_OUTPUT_DIR)
    parser.add_argument("--labels", type=Path, default=config.PARTICIPANT_LABELS_CSV)
    parser.add_argument("--cv-repeats", type=int, default=config.CV_REPEATS)
    parser.add_argument("--permutations", type=int, default=config.CLASSIFIER_PERMUTATIONS)
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arguments = parse_arguments()
    by_attempt = pd.read_csv(arguments.rhythm_dir / "rhythm_parameters_by_attempt.csv")
    labels = pd.read_csv(arguments.labels)

    people = rp.person_parameters(by_attempt, labels)                       # practice attempt 1 left out
    people_all_attempts = rp.person_parameters(by_attempt, labels, skip_first_attempt=False)
    main_set = people.loc[people["screened"]].reset_index(drop=True)        # healthy without complaints + patients
    n_patients = int((people["group"] == "patient").sum())
    checks = {
        "all_healthy": (f"{int((people['group'] == 'control').sum())} zdravých + {n_patients} pacientov",
                        people.reset_index(drop=True)),
        "with_first_attempt": ("s 1. pokusom", people_all_attempts.loc[people_all_attempts["screened"]]),
    }
    tables = {"people": people}
    tables.update(ga.analyse(main_set, rp.PARAMETERS, rp.EXPECTED_DIRECTION, rp.PARAMETER_GROUPS, "parametre rytmu",
                             covariate=rp.COVARIATE, checks=checks, n_repeats=arguments.cv_repeats,
                             n_permutations=arguments.permutations))
    ga.write_tables(tables, arguments.output_dir)


if __name__ == "__main__":
    main()
