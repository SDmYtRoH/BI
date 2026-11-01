"""Hand-balance statistics: RHYTHM vs EXACT_FORCE, age categories, age vs disease (week 4).

Run from the repository root (after scripts/build_dataset.py, scripts/run_analysis.py and Maksym/scripts/build_rhythm.py):
    python Maksym/scripts/run_rhythm_analysis.py
Writes CSV tables to ``Maksym/week4/outputs/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repository root (shared rehapiano package)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Maksym" / "src"))

from rehapiano import config  # noqa: E402
from maksym_rehapiano import config as rhythm_config, hand_balance  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rhythm-dir", type=Path, default=rhythm_config.WEEK4_OUTPUT_DIR)
    parser.add_argument("--exact-force-activity", type=Path, default=config.STATS_OUTPUT_DIR / "hand_activity.csv")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    out: Path = arguments.rhythm_dir
    rhythm = hand_balance.rhythm_person_table(pd.read_csv(out / "rhythm_by_attempt.csv"))
    people = hand_balance.combine_tests(rhythm, pd.read_csv(arguments.exact_force_activity))
    thresholds = {test: hand_balance.weak_threshold(people, f"{test}_ratio") for test in hand_balance.TESTS}
    people = hand_balance.add_weak_flags(people, thresholds)

    tables = {
        "balance_by_person": people,
        "weak_hand_thresholds": pd.DataFrame([{"test": hand_balance.TESTS[k], "threshold": v}
                                              for k, v in thresholds.items()]),
        "test_agreement": pd.DataFrame([hand_balance.test_agreement(people)]),
        "balance_by_age_category": hand_balance.by_age_category(people),
        "age_effect_controls": hand_balance.age_effect_in_controls(people),
        "disease_effect_by_age": hand_balance.disease_effect_by_age(people),
        "age_disease_model": hand_balance.age_disease_model(people),
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
        print(f"{name + '.csv':<32} {len(table):>5} rows")
    print(f"\nTables written to {out.resolve()}")


if __name__ == "__main__":
    main()
