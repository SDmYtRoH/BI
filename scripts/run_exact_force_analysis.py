"""EXACT_FORCE step 3: hypotheses, decade norms, classifier and ablation (teacher's week 5 task).

Run from the repository root (after build_exact_force.py):
    python scripts/run_exact_force_analysis.py
    python scripts/run_exact_force_analysis.py --cv-repeats 3 --permutations 20   # quick check
Writes CSV tables to ``outputs/exact_force/analysis/``.
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
from rehapiano.exact_force import config as ef_config, parameters as ep  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input-dir", type=Path, default=ef_config.EF_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=ef_config.ANALYSIS_OUTPUT_DIR)
    parser.add_argument("--labels", type=Path, default=config.PARTICIPANT_LABELS_CSV)
    parser.add_argument("--cv-repeats", type=int, default=config.CV_REPEATS)
    parser.add_argument("--permutations", type=int, default=config.CLASSIFIER_PERMUTATIONS)
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arguments = parse_arguments()
    people = ep.person_parameters(pd.read_csv(arguments.input_dir / "features_by_person_finger.csv"),
                                  pd.read_csv(arguments.input_dir / "person_summary.csv"),
                                  pd.read_csv(arguments.labels))
    main_set = people.loc[people["screened"]].reset_index(drop=True)
    n_patients = int((people["group"] == "patient").sum())
    checks = {"all_healthy": (f"{int((people['group'] == 'control').sum())} zdravých + {n_patients} pacientov",
                              people)}
    tables = {"people": people}
    tables.update(ga.analyse(main_set, ep.PARAMETERS, ep.EXPECTED_DIRECTION, ep.MODEL_GROUPS, "parametre EF",
                             covariate=ep.COVARIATE, checks=checks, n_repeats=arguments.cv_repeats,
                             n_permutations=arguments.permutations))
    ga.write_tables(tables, arguments.output_dir)


if __name__ == "__main__":
    main()
