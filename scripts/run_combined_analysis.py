"""Both exercises: classifier healthy vs patients on EXACT_FORCE + RHYTHM and ablation of exercises / factors.

Run from the repository root (after run_exact_force_analysis.py and run_rhythm_analysis.py, which write the
person tables ``outputs/*/analysis/people.csv``):
    python scripts/run_combined_analysis.py
Writes CSV tables to ``outputs/combined/``.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rehapiano import combined  # noqa: E402
from rehapiano.common import config  # noqa: E402
from rehapiano.common.stats import group_analysis as ga  # noqa: E402
from rehapiano.exact_force import config as ef_config  # noqa: E402
from rehapiano.rhythm import config as rhythm_config  # noqa: E402

EF, RH = "EXACT_FORCE", "rytmus"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--exact-force", type=Path, default=ef_config.ANALYSIS_OUTPUT_DIR / "people.csv")
    parser.add_argument("--rhythm", type=Path, default=rhythm_config.ANALYSIS_OUTPUT_DIR / "people.csv")
    parser.add_argument("--output-dir", type=Path, default=config.COMBINED_OUTPUT_DIR)
    parser.add_argument("--cv-repeats", type=int, default=config.CV_REPEATS)
    parser.add_argument("--permutations", type=int, default=config.CLASSIFIER_PERMUTATIONS)
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    arguments = parse_arguments()
    people = combined.combined_people(pd.read_csv(arguments.exact_force), pd.read_csv(arguments.rhythm))
    main_set = people.loc[people["screened"]].reset_index(drop=True)
    n_healthy, n_patients = int((main_set["group"] == "control").sum()), int((main_set["group"] == "patient").sum())
    both = combined.EF_FEATURES + combined.RH_FEATURES
    feature_sets = {EF: combined.EF_FEATURES, RH: combined.RH_FEATURES, f"{EF} + {RH}": both,
                    f"{EF} + {RH} + vek": both + ["age"], "len vek": ["age"]}

    tables = {"people": people}
    classifier, oof = ga.classifier_table(main_set, f"{n_healthy} zdravých + {n_patients} pacientov", feature_sets,
                                          older_feature_sets=[EF, RH, f"{EF} + {RH}"], n_repeats=arguments.cv_repeats)
    tables["classifier"] = classifier
    tables["classifier_oof"] = oof
    tables["classifier_coefficients"] = ga.logistic_coefficients(main_set, both)
    print("classifier done")
    tables["ablation_exercises"] = ga.ablation(main_set, {**combined.EXERCISE_GROUPS, "vek": ["age"]},
                                               n_repeats=arguments.cv_repeats)
    tables["ablation_factors"] = ga.ablation(main_set, combined.FACTOR_GROUPS, n_repeats=arguments.cv_repeats)
    tables["ablation_factors_with_age"] = ga.ablation(main_set, {**combined.FACTOR_GROUPS, "vek": ["age"]},
                                                      n_repeats=arguments.cv_repeats)
    tables["permutation_test"] = pd.DataFrame([ga.permutation_auc(main_set, both, n_permutations=arguments.permutations)])
    print("ablation done")
    ga.write_tables(tables, arguments.output_dir)


if __name__ == "__main__":
    main()
