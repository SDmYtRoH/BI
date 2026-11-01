"""Unit tests for the RHYTHM code (weeks 4-5, M. Kuznietsov). Run from the repository root:

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

from rehapiano.common.config import ANALYSIS_SAMPLE_RATE_HZ as FS, SENSOR_COLUMNS  # noqa: E402
from rehapiano.rhythm.presses import (  # noqa: E402
    detect_presses,
    extract_measurement_rhythm_features,
    extract_measurement_rhythm_parameters,
)
from rehapiano.rhythm.hand_balance import _balance, ols  # noqa: E402


def presses(n_samples: int, rate_hz: float = 2.0) -> np.ndarray:
    """Half-wave pulses: force 0 between presses, 1 at the top of each press."""

    return np.clip(np.sin(2 * np.pi * rate_hz * np.arange(n_samples) / FS), 0, None)


def make_measurement(left_amplitude: float, right_amplitude: float, n_samples: int = 2800) -> pd.DataFrame:
    frame = pd.DataFrame({"attempt": 1, "segment": 1, "time_s": np.arange(n_samples) / FS})
    pulse = presses(n_samples)
    for column in SENSOR_COLUMNS:
        frame[column] = 0.002 + (left_amplitude if column.startswith("left") else right_amplitude) * pulse
    return frame


class RhythmTests(unittest.TestCase):
    def test_detects_every_press(self) -> None:
        # 28 s at 2 presses per second = 56 presses; small noise must not add presses
        signal = presses(2800) + np.random.default_rng(0).normal(0, 0.005, 2800)
        self.assertEqual(len(detect_presses(signal)), 56)

    def test_balance_of_hands(self) -> None:
        rows = extract_measurement_rhythm_features(make_measurement(0.10, 0.05)).set_index("hand")
        self.assertTrue((rows["status"] == "valid").all())
        self.assertAlmostEqual(rows.loc["right", "press_amplitude"] / rows.loc["left", "press_amplitude"], 0.5, places=2)
        self.assertAlmostEqual(rows.loc["left", "press_rate_hz"], 2.0, places=1)

    def test_silent_hand_still_valid(self) -> None:
        # presses are found on the sum of both hands, so a hand that does not press gives balance ~0
        rows = extract_measurement_rhythm_features(make_measurement(0.10, 0.0)).set_index("hand")
        self.assertEqual(rows.loc["right", "status"], "valid")
        self.assertLess(rows.loc["right", "press_amplitude"], 1e-9)

    def test_flat_attempt_is_invalid(self) -> None:
        rows = extract_measurement_rhythm_features(make_measurement(0.0, 0.0))
        self.assertTrue((rows["invalid_reason"] == "too_few_presses").all())

    def test_week5_parameters(self) -> None:
        row = extract_measurement_rhythm_parameters(make_measurement(0.10, 0.05)).iloc[0]
        self.assertAlmostEqual(row["press_rate_hz"], 2.0, places=1)
        self.assertLess(row["interval_cv"], 0.03)          # regular rhythm; peaks fall between samples (±10 ms)
        self.assertAlmostEqual(row["balance"], 0.5, places=2)
        self.assertAlmostEqual(row["amplitude_strong"], 0.10, places=2)
        self.assertLess(row["amplitude_cv"], 0.01)         # every press equal
        self.assertEqual(row["sync_lag_ms"], 0.0)          # both hands peak together
        self.assertLess(row["release_ratio"], 0.05)        # hands release completely
        self.assertAlmostEqual(row["fatigue_amplitude"], 1.0, places=2)
        self.assertAlmostEqual(row["fatigue_rate"], 1.0, delta=0.05)

    def test_balance_ratio(self) -> None:
        ratio, weaker = _balance(pd.Series([0.2, 0.1, -0.01]), pd.Series([0.1, 0.1, 0.05]))
        np.testing.assert_allclose(ratio, [0.5, 1.0, 0.0])
        self.assertEqual(list(weaker), ["right", "right", "left"])

    def test_ols_recovers_coefficients(self) -> None:
        rng = np.random.default_rng(1)
        X = np.column_stack([np.ones(200), rng.normal(size=200), rng.integers(0, 2, 200)])
        y = X @ np.array([0.9, -0.01, -0.2]) + rng.normal(0, 0.01, 200)
        result = ols(y, X, ["b0", "x", "group"]).set_index("term")
        np.testing.assert_allclose(result["coefficient"], [0.9, -0.01, -0.2], atol=0.01)
        self.assertLess(result.loc["group", "p"], 1e-6)


if __name__ == "__main__":
    unittest.main()
