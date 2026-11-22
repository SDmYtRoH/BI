"""EXACT_FORCE vs RHYTHM (week 6, M. Kuznietsov): group differences with one method, agreement of the tests,
agreement of their classifiers and repeatability in patients measured twice.

Run from the repository root (after run_exact_force_analysis.py and run_rhythm_analysis.py):
    python scripts/run_comparison.py
Writes CSV tables to ``outputs/comparison/`` (a few seconds, no cross-validation).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rehapiano import comparison  # noqa: E402
from rehapiano.common import config  # noqa: E402
from rehapiano.common.stats.group_analysis import write_tables  # noqa: E402
from rehapiano.exact_force import config as ef_config  # noqa: E402
from rehapiano.rhythm import config as rhythm_config  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--exact-force-dir", type=Path, default=ef_config.ANALYSIS_OUTPUT_DIR)
    parser.add_argument("--rhythm-dir", type=Path, default=rhythm_config.ANALYSIS_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.COMPARISON_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    arguments = parse_arguments()
    ef, rh = arguments.exact_force_dir, arguments.rhythm_dir
    tables = comparison.run(
        ef_people=pd.read_csv(ef / "people.csv"), rh_people=pd.read_csv(rh / "people.csv"),
        ef_oof=pd.read_csv(ef / "classifier_oof.csv"), rh_oof=pd.read_csv(rh / "classifier_oof.csv"),
        ef_features_by_measurement=pd.read_csv(ef_config.EF_OUTPUT_DIR / "features_by_measurement_finger.csv"),
        rh_by_attempt=pd.read_csv(rhythm_config.RHYTHM_OUTPUT_DIR / "rhythm_parameters_by_attempt.csv"))
    write_tables(tables, arguments.output_dir)


if __name__ == "__main__":
    main()
