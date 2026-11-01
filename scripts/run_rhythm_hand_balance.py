"""RHYTHM step 2 (week 4, M. Kuznietsov): hand balance vs EXACT_FORCE, age categories, age vs disease.

Run from the repository root (after build_exact_force.py, run_exact_force_stats.py and build_rhythm.py):
    python scripts/run_rhythm_hand_balance.py
Writes CSV tables to ``outputs/rhythm/hand_balance/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rehapiano.exact_force import config as ef_config  # noqa: E402
from rehapiano.rhythm import config as rhythm_config, hand_balance  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rhythm-dir", type=Path, default=rhythm_config.RHYTHM_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=rhythm_config.HAND_BALANCE_OUTPUT_DIR)
    parser.add_argument("--exact-force-activity", type=Path, default=ef_config.STATS_OUTPUT_DIR / "hand_activity.csv")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    out: Path = arguments.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rhythm = hand_balance.rhythm_person_table(pd.read_csv(arguments.rhythm_dir / "rhythm_by_attempt.csv"))
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
