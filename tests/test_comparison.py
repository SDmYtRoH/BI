"""Unit tests for the EXACT_FORCE vs RHYTHM comparison. Run from the project root:

    python -m unittest discover tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rehapiano import comparison  # noqa: E402
from rehapiano.common.stats.tests import icc_agreement, van_elteren  # noqa: E402
from rehapiano.exact_force import parameters as ep  # noqa: E402


class VanElterenTests(unittest.TestCase):
    def test_age_confounding_is_not_a_difference(self):
        # patients are older and the value grows with age, but inside every decade both groups are equal
        rng = np.random.default_rng(0)
        decade = np.repeat(np.arange(6), 40)
        patient = np.r_[[(rng.random(40) < share).astype(int) for share in np.linspace(0.05, 0.9, 6)]].ravel()
        value = decade + rng.normal(0, 1, len(decade))
        self.assertGreater(van_elteren(value, patient, decade)["p"], 0.05)

    def test_difference_inside_decades_is_found(self):
        rng = np.random.default_rng(1)
        decade = np.repeat(np.arange(4), 30)
        patient = np.tile(np.r_[np.zeros(15, int), np.ones(15, int)], 4)
        value = decade + 1.5 * patient + rng.normal(0, 1, len(decade))
        result = van_elteren(value, patient, decade)
        self.assertLess(result["p"], 0.001)
        self.assertGreater(result["z"], 0)

    def test_decades_with_one_group_are_left_out(self):
        result = van_elteren([1, 2, 3, 4, 5, 6], [0, 0, 0, 1, 1, 1], ["a", "a", "a", "b", "b", "b"])
        self.assertEqual(result["n"], 0)
        self.assertTrue(np.isnan(result["p"]))


class RepeatabilityTests(unittest.TestCase):
    def test_icc(self):
        rng = np.random.default_rng(2)
        a = rng.normal(0, 1, 200)
        self.assertAlmostEqual(icc_agreement(a, a), 1.0)
        self.assertLess(abs(icc_agreement(a, rng.normal(0, 1, 200))), 0.2)
        # a constant shift between visits lowers absolute agreement
        self.assertLess(icc_agreement(a, a + 2), 0.5)

    def test_two_visits_are_paired_in_time_order(self):
        people = [f"p{i}" for i in range(10)]
        rows = []
        for i, person in enumerate(people):
            rows.append({"person_id": person, "measurement_id": f"{person}-b", "timestamp": "2026-01-08", "x": i + 0.5})
            rows.append({"person_id": person, "measurement_id": f"{person}-a", "timestamp": "2026-01-01", "x": float(i)})
        rows.append({"person_id": "single", "measurement_id": "s", "timestamp": "2026-01-01", "x": 100.0})
        result = comparison.repeatability(pd.DataFrame(rows), ["x"]).iloc[0]
        self.assertEqual(result["n_people"], 10)
        self.assertAlmostEqual(result["rho"], 1.0)
        self.assertAlmostEqual(result["median_change"], 0.5)


class AgreementTests(unittest.TestCase):
    def test_flags(self):
        ef = pd.DataFrame({"patient_id": list("abcd"), "group": "patient", "age": 60, "patient": 1,
                           comparison.OOF_COLUMNS[comparison.EF]: [0.9, 0.9, 0.1, 0.1]})
        rh = pd.DataFrame({"patient_id": list("abcd"), comparison.OOF_COLUMNS[comparison.RH]: [0.8, 0.2, 0.7, 0.3]})
        flags = comparison.classifier_agreement(ef, rh)["flagged"].tolist()
        self.assertEqual(flags, ["oba", "len EXACT_FORCE", "len RYTMUS", "žiadny"])


@unittest.skipUnless((ROOT / "outputs/exact_force/features_by_measurement_finger.csv").exists(), "no EF tables")
class ExactForceByRecordingTests(unittest.TestCase):
    def test_single_recording_equals_person_table(self):
        recordings = comparison.exact_force_by_recording(
            pd.read_csv(ROOT / "outputs/exact_force/features_by_measurement_finger.csv"))
        people = pd.read_csv(ROOT / "outputs/exact_force/analysis/people.csv").set_index("patient_id")
        single = recordings.groupby("person_id").filter(lambda g: len(g) == 1).set_index("person_id")
        difference = (single[ep.PARAMETERS] - people.loc[single.index, ep.PARAMETERS]).abs().max().max()
        self.assertGreater(len(single), 50)
        self.assertLess(difference, 1e-9)


if __name__ == "__main__":
    unittest.main()
