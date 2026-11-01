"""Tests of the unified pipeline (``rehapiano.pipeline``) on synthetic data with known answers.

Run from the project root:  python -m unittest discover tests
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rehapiano.config import SENSOR_COLUMNS  # noqa: E402
from rehapiano.features import exact_force_game, rhythm_game  # noqa: E402
from rehapiano.pipeline import aggregate, index, participants, settings  # noqa: E402

FS = 100.0


def rhythm_frame(left: np.ndarray, right: np.ndarray) -> pd.DataFrame:
    """One attempt: every sensor of a hand carries the same signal."""

    frame = pd.DataFrame({column: (left if column.startswith("left") else right) for column in SENSOR_COLUMNS})
    frame.insert(0, "time_s", np.arange(len(left)) / FS)
    return frame


def presses(rate_hz: float, seconds: float = 28.0, amplitude: float = 0.1, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(seconds * FS)) / FS
    return amplitude * (0.5 + 0.5 * np.sin(2 * np.pi * rate_hz * t + phase)) + 0.002


class RhythmFeatures(unittest.TestCase):
    def test_both_hands_together(self):
        result = rhythm_game.attempt_features(rhythm_frame(presses(2.0), presses(2.0)))
        self.assertAlmostEqual(result["press_rate_hz"], 2.0, delta=0.1)
        self.assertAlmostEqual(result["balance_range"], 1.0, places=6)
        self.assertAlmostEqual(result["balance_press"], 1.0, places=6)
        self.assertLess(result["interval_cv"], 0.05)
        self.assertEqual(result["sync_lag_ms"], 0.0)

    def test_one_silent_hand_gives_balance_near_zero(self):
        silent = np.full(int(28 * FS), 0.002)
        result = rhythm_game.attempt_features(rhythm_frame(presses(2.0), silent))
        self.assertLess(result["balance_range"], 0.01)
        self.assertLess(result["balance_press"], 0.01)
        self.assertEqual(result["weaker_hand"], "right")

    def test_alternating_hands_keep_full_range_balance(self):
        result = rhythm_game.attempt_features(rhythm_frame(presses(1.0), presses(1.0, phase=np.pi)))
        self.assertGreater(result["balance_range"], 0.95)

    def test_too_few_presses_leave_press_features_missing(self):
        result = rhythm_game.attempt_features(rhythm_frame(presses(0.1), presses(0.1)))
        self.assertLess(result["presses"], settings.RHYTHM_MIN_PRESSES)
        self.assertTrue(np.isnan(result["balance_press"]))
        self.assertFalse(np.isnan(result["balance_range"]))


class ExactForceBands(unittest.TestCase):
    def test_sine_lands_in_its_band(self):
        t = np.arange(1200) / FS
        fast = 0.02 + 1e-3 * np.sin(2 * np.pi * 6.0 * t)
        slow = 0.02 + 1e-3 * np.sin(2 * np.pi * 1.0 * t)
        fast_bands = exact_force_game.band_amplitudes(fast, settings.EF_BANDS)
        slow_bands = exact_force_game.band_amplitudes(slow, settings.EF_BANDS)
        self.assertGreater(fast_bands["tremor_amplitude"], 10 * fast_bands["slow_amplitude"])
        self.assertGreater(slow_bands["slow_amplitude"], 10 * slow_bands["tremor_amplitude"])

    def test_hand_without_active_finger_has_missing_oscillation(self):
        rows = []
        for hand, force in [("left", 0.02), ("right", 0.001)]:  # right hand within the unloaded-sensor range
            for finger in range(1, 6):
                rows.append({"attempt": 1, "hand": hand, "finger": finger, "status": "valid", "mean_force": force,
                             "finger_inactive": force < 0.002, "slow_amplitude": 1e-4, "boundary_amplitude": 1e-5,
                             "tremor_mean_amplitude": 1e-5, "tremor_peak_frequency_hz": 5.0})
        hands = exact_force_game.hand_features(pd.DataFrame(rows)).set_index("hand")
        self.assertEqual(hands.loc["right", "active_fingers"], 0)
        self.assertTrue(np.isnan(hands.loc["right", "tremor_amplitude"]))
        self.assertAlmostEqual(hands.loc["left", "tremor_amplitude"], 1e-5)


class Aggregation(unittest.TestCase):
    def test_median_over_attempts_ignores_one_odd_attempt(self):
        rows = [{"patient_id": "p", "measurement_id": "m", "attempt": a, "hand": hand,
                 "mean_force": (0.05 if (a == 1 and hand == "left") else 0.02), "slow_amplitude": 1e-4,
                 "boundary_amplitude": 1e-5, "tremor_amplitude": 1e-5, "tremor_share": 0.1, "active_fingers": 5}
                for a in range(1, 5) for hand in ("left", "right")]
        person = aggregate.exact_force_person(pd.DataFrame(rows)).iloc[0]
        self.assertAlmostEqual(person["ef_balance"], 1.0)
        self.assertEqual(person["ef_attempts"], 4)

    def test_practice_attempt_is_left_out(self):
        rows = [{"patient_id": "p", "measurement_id": "m", "attempt": a, "valid": True, "weaker_hand": "left",
                 **{name: (9.0 if a == 1 else 1.0) for name in rhythm_game.PARAMETERS}} for a in range(1, 5)]
        person = aggregate.rhythm_person(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(person["rh_press_rate_hz"], 1.0)
        self.assertEqual(person["rh_attempts"], 3)


class IndexAndLabels(unittest.TestCase):
    def test_copies_are_collapsed_and_conflicts_excluded(self):
        step = next(iter(settings.GAMES["rhythm"].steps))
        identity = pd.DataFrame({
            "file_path": ["a/results_1", "b/results_1", "a/results_2", "b/results_2"],
            "size": 1, "mtime_ns": 1, "file_uuid": ["u1", "u1", "u2", "u2"], "examination_id": ["u1", "u1", "u2", "u2"],
            "step_id": step, "md5": ["x", "x", "y", "z"], "error": ""})
        metadata = pd.DataFrame({"patient_id": ["P1", "P2"], "timestamp": "2025-01-01 10:00:00", "age": "50",
                                 "height": "", "weight": "", "gender": "male", "hand": "right",
                                 "measurement_id": ["u1", "u2"], "step_id": step, "step_type": "RHYTHM",
                                 "cohort": "control", "source_files": "test.csv"})
        result = index.build_index(identity, metadata).set_index("file_path")
        self.assertEqual(result.loc["a/results_1", "status"], "used")
        self.assertEqual(result.loc["b/results_1", "reason"], "identical_copy")
        self.assertEqual(set(result.loc[["a/results_2", "b/results_2"], "reason"]), {"conflicting_copies"})

    def test_questionnaire_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.csv"
            pd.DataFrame({"Vaše identifikačné číslo": ["A ", "B", "C", "C"],
                          "Trpíte ochorením, ktoré postihuje ...": ["Nie", "Áno", "Nie", "Nie"]}).to_csv(path)
            status = participants.questionnaire_status(pd.Series(["A", "B", "C", "D"]), path)
        self.assertEqual(status.set_index("patient_id")["label_status"].to_dict(),
                         {"A": "no_reported_condition", "B": "reported_condition", "C": "duplicate_questionnaire_id",
                          "D": "unknown_or_conflicting"})


if __name__ == "__main__":
    unittest.main()
