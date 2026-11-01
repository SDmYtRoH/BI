"""EXACT_FORCE step 2 (week 3): age, weakly used hand and finger tremor statistics.

Run from the project root (after build_exact_force.py):
    python scripts/run_exact_force_stats.py
Writes CSV tables to ``outputs/exact_force/stats/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rehapiano.common.stats import demographics  # noqa: E402
from rehapiano.exact_force import config  # noqa: E402
from rehapiano.exact_force.stats import finger_tremor, hand_activity  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input-dir", type=Path, default=config.EF_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.STATS_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output_dir: Path = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    person_summary = pd.read_csv(arguments.input_dir / "person_summary.csv")
    features_by_person = pd.read_csv(arguments.input_dir / "features_by_person_finger.csv")

    people = demographics.persons(person_summary)
    activity = hand_activity.hand_activity_table(person_summary, features_by_person)
    fingers = finger_tremor.active_finger_table(features_by_person)
    friedman_table, posthoc_table = finger_tremor.finger_differences(fingers)
    spread = finger_tremor.finger_spread(fingers)

    z_table = finger_tremor.z_scores(fingers)
    sensor_noise_path = arguments.input_dir / "sensor_noise.csv"

    tables = {
        "demographics_by_group": demographics.describe_groups(people),
        "age_histogram": demographics.age_histogram(people),
        "age_test": demographics.age_difference_test(people),
        "gender_by_age_group": demographics.gender_by_age_group(people),
        "hand_activity": activity,
        "weak_hand_counts": hand_activity.weak_hand_counts(activity),
        "weak_hand_patients": hand_activity.weak_hand_patients(activity),
        "finger_ratio_to_reference": finger_tremor.ratio_to_reference(fingers),
        "finger_z_scores": z_table,
        "tremor_pattern": finger_tremor.tremor_pattern(z_table),
        "finger_descriptives": finger_tremor.finger_descriptives(fingers),
        "finger_friedman": friedman_table,
        "finger_posthoc": posthoc_table,
        "most_tremulous_finger": finger_tremor.most_tremulous_finger(fingers),
        "finger_spread_by_person": spread,
        "finger_spread_group_test": finger_tremor.spread_group_comparison(spread),
        "group_comparison_by_finger": finger_tremor.group_comparison(fingers),
        "left_right_by_finger": finger_tremor.left_right_comparison(fingers),
    }
    if sensor_noise_path.exists():
        tables["finger_signal_to_noise"] = finger_tremor.noise_comparison(fingers, pd.read_csv(sensor_noise_path))
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
        print(f"{name + '.csv':<36} {len(table):>5} rows")
    print(f"\nTables written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
