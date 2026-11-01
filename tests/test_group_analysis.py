"""Unit tests for the shared healthy-vs-patient analysis. Run from the project root:

    python -m unittest discover tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rehapiano.common.stats import group_analysis as ga  # noqa: E402


def people(n_controls: int = 40, n_patients: int = 30, shift: float = 2.0, seed: int = 0) -> pd.DataFrame:
    """``signal`` is higher in patients by ``shift`` SD, ``noise`` does not differ."""

    rng = np.random.default_rng(seed)
    n = n_controls + n_patients
    patient = np.r_[np.zeros(n_controls, int), np.ones(n_patients, int)]
    table = pd.DataFrame({"patient_id": [f"p{i}" for i in range(n)], "patient": patient,
                          "group": np.where(patient == 1, "patient", "control"),
                          "age": rng.integers(20, 80, n).astype(float),
                          "signal": rng.normal(0, 1, n) + shift * patient, "noise": rng.normal(0, 1, n)})
    return ga.add_decade(table)


class GroupAnalysisTests(unittest.TestCase):
    def test_hypotheses_find_only_the_real_difference(self):
        result = ga.hypothesis_tests(people(), ["signal", "noise"], {"signal": +1, "noise": +1}).set_index("parameter")
        self.assertLess(result.loc["signal", "p_holm"], 0.001)
        self.assertGreater(result.loc["signal", "auc"], 0.85)
        self.assertGreater(result.loc["noise", "p_holm"], 0.05)
        self.assertTrue(result.loc["signal", "as_expected"])

    def test_covariate_model_is_skipped_for_the_covariate_itself(self):
        result = ga.hypothesis_tests(people(), ["signal", "noise"], {"signal": 1, "noise": 1}, covariate="noise")
        result = result.set_index("parameter")
        self.assertTrue(np.isnan(result.loc["noise", "disease_adj_p"]))
        self.assertFalse(np.isnan(result.loc["signal", "disease_adj_p"]))

    def test_pooled_norms_are_equal_in_every_decade(self):
        healthy = people().query("group == 'control'")
        trends = ga.age_trends(healthy, ["noise"])
        norms = ga.norms_by_age_model(healthy, trends.assign(age_dependent=False), ["noise"])
        self.assertEqual(norms["q50"].nunique(), 1)
        self.assertAlmostEqual(norms["q50"].iloc[0], healthy["noise"].median())

    def test_classifier_separates_only_with_the_informative_feature(self):
        table = people()
        good = ga.cross_validate(table, ["signal"], n_repeats=2)
        bad = ga.cross_validate(table, ["noise"], n_repeats=2)
        self.assertGreater(good["auc"], 0.85)
        self.assertLess(bad["auc"], 0.7)
        self.assertEqual(len(good["oof_probability"]), len(table))

    def test_ablation_drop_is_largest_for_the_informative_group(self):
        result = ga.ablation(people(), {"a": ["signal"], "b": ["noise"]}, n_repeats=2)
        self.assertEqual(result.iloc[0]["group"], "a")

    def test_screening_keeps_patients_missing_from_labels(self):
        table = people(n_controls=2, n_patients=2)
        labels = pd.DataFrame({"patient_id": ["p0"], "eligible_comparison": [True]})
        with self.assertLogs(ga.logger, level="WARNING"):
            screened = ga.add_screening(table, labels).set_index("patient_id")["screened"]
        self.assertEqual(screened.to_dict(), {"p0": True, "p1": False, "p2": True, "p3": True})


if __name__ == "__main__":
    unittest.main()
