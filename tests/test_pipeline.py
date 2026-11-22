"""Unit tests for the finger pipeline. Run from the project root:

    python -m unittest discover tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rehapiano.exact_force.aggregation import build_wide_table, classify_hand_activity, flag_attempt_outliers  # noqa: E402
from rehapiano.common.config import ANALYSIS_SAMPLE_RATE_HZ as FS, SENSOR_COLUMNS  # noqa: E402
from rehapiano.exact_force.config import (  # noqa: E402
    ANALYSED_STEP_IDS,
    CONTROL_STEP_IDS,
    CROP_END_S,
    CROP_START_S,
    PATIENT_STEP_IDS,
    STEP_TYPE,
    MOVEMENT_BAND,
    SPECTROGRAM_NPERSEG,
    TREMOR_BAND,
)
from rehapiano.exact_force.features.finger import (  # noqa: E402
    analyze_finger,
    compute_finger_features,
    extract_measurement_finger_features,
)
from rehapiano.exact_force.features.spectral import band_mask, compute_spectrogram  # noqa: E402
from rehapiano.common.io.metadata import age_group_label, build_manifest, extract_file_uuid, label_patients  # noqa: E402
from rehapiano.common.preprocessing.crop import crop_mask  # noqa: E402
from rehapiano.common.preprocessing.gaps import fill_missing  # noqa: E402
from rehapiano.common.preprocessing.polarity import correct_polarity  # noqa: E402
from rehapiano.exact_force.quality import FLAT_SIGNAL, NO_SEGMENT, NON_FINITE, TOO_SHORT, flag_outliers  # noqa: E402
from rehapiano.exact_force.stats.finger_tremor import tremor_pattern, z_scores  # noqa: E402
from rehapiano.exact_force.stats.hand_activity import weak_hand_patients  # noqa: E402
from rehapiano.common.stats.tests import holm_adjust  # noqa: E402

UUID = "7d55e12b-66df-4ca6-9ef3-27a880edec7a"
STEP = "5867ebbc-a463-4d20-8b20-50df7c64ace7"


def tone(n_samples: int, level: float = 0.05, tremor_hz: float = 6.0, tremor_amp: float = 0.002) -> np.ndarray:
    time_s = np.arange(n_samples) / FS
    return level + 0.005 * np.sin(2 * np.pi * 1.0 * time_s) + tremor_amp * np.sin(2 * np.pi * tremor_hz * time_s)


def make_measurement(n_samples: int, segments: tuple[int, ...] = (1,), attempt: int = 1) -> pd.DataFrame:
    """Synthetic preprocessed measurement: every sensor holds a force with a 6 Hz tremor."""

    frames = []
    for segment in segments:
        frame = pd.DataFrame({"attempt": attempt, "segment": segment, "time_s": np.arange(n_samples) / FS})
        for column in SENSOR_COLUMNS:
            frame[column] = tone(n_samples)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class FilenameTests(unittest.TestCase):
    def test_extracts_uuid(self) -> None:
        self.assertEqual(extract_file_uuid(f"results_{UUID}_20260909-154349"), UUID)
        self.assertEqual(extract_file_uuid(f"some/dir/results_{UUID.upper()}_20260909-154349"), UUID)

    def test_rejects_invalid_names(self) -> None:
        for name in (f"results_{UUID}", f"result_{UUID}_20260909-154349", "results_not-a-uuid_20260909-154349",
                     f"results_{UUID}_20260909-154349.csv"):
            self.assertIsNone(extract_file_uuid(name), name)


class DemographicsTests(unittest.TestCase):
    def test_age_groups_are_ten_years(self) -> None:
        self.assertEqual(age_group_label(20), "20-29")
        self.assertEqual(age_group_label(29), "20-29")
        self.assertEqual(age_group_label(85), "80-89")
        self.assertIs(age_group_label(np.nan), pd.NA)


class PreprocessingTests(unittest.TestCase):
    def test_polarity_flips_only_right_sensor_2(self) -> None:
        corrected = correct_polarity(np.ones((3, len(SENSOR_COLUMNS))))
        expected = np.ones(len(SENSOR_COLUMNS))
        expected[SENSOR_COLUMNS.index("right_sensor_2")] = -1
        np.testing.assert_array_equal(corrected[0], expected)

    def test_short_nan_gap_is_interpolated(self) -> None:
        timestamps = np.arange(4000, dtype=np.int64)
        values = np.tile(np.arange(4000, dtype=float)[:, None], (1, 2))
        values[100:105, 0] = np.nan
        [(time_s, filled)] = fill_missing(timestamps, values)
        self.assertEqual(len(time_s), 4000)
        np.testing.assert_allclose(filled[:, 0], np.arange(4000))

    def test_channel_with_too_many_nans_stays_nan(self) -> None:
        timestamps = np.arange(4000, dtype=np.int64)
        values = np.ones((4000, 2))
        values[::50, 1] = np.nan  # 2 % missing
        [(_, filled)] = fill_missing(timestamps, values)
        self.assertTrue(np.isnan(filled[:, 1]).all())
        self.assertFalse(np.isnan(filled[:, 0]).any())

    def test_large_gap_splits_segments(self) -> None:
        timestamps = np.r_[np.arange(4000), np.arange(4200, 8200)].astype(np.int64)
        self.assertEqual(len(fill_missing(timestamps, np.ones((8000, 1)))), 2)

    def test_crop_interval(self) -> None:
        time_s = np.arange(1600) / FS
        self.assertEqual(crop_mask(time_s, CROP_START_S, CROP_END_S).sum(), 1200)


class SpectralTests(unittest.TestCase):
    def test_band_boundaries(self) -> None:
        frequencies = np.array([0.0, 0.2, 3.8, 4.0, 4.2, 14.8, 15.0])
        np.testing.assert_array_equal(frequencies[band_mask(frequencies, MOVEMENT_BAND)], [0.2, 3.8, 4.0])
        np.testing.assert_array_equal(frequencies[band_mask(frequencies, TREMOR_BAND)], [4.0, 4.2, 14.8, 15.0])

    def test_spectrogram_limited_to_15_hz(self) -> None:
        frequencies, times, magnitude = compute_spectrogram(np.random.default_rng(0).normal(size=800))
        self.assertLessEqual(frequencies.max(), 15.0)
        self.assertAlmostEqual(frequencies[1] - frequencies[0], 0.2)
        self.assertEqual(len(times), 800 - SPECTROGRAM_NPERSEG + 1)
        self.assertEqual(magnitude.shape, (len(frequencies), len(times)))


class FingerFeatureTests(unittest.TestCase):
    def test_tremor_peak_and_force(self) -> None:
        features = compute_finger_features(tone(1200))
        self.assertAlmostEqual(features["tremor_peak_frequency_hz"], 6.0)
        self.assertAlmostEqual(features["mean_force"], 0.05, places=3)
        self.assertGreater(features["tremor_peak_prominence"], 10)
        self.assertGreater(features["tremor_mean_frequency_hz"], 3.0)
        self.assertLess(features["tremor_to_movement_ratio"], 1.0)

    def test_decaying_spectrum_has_no_edge_peak(self) -> None:
        # Slow drift only: the tremor band decays monotonically -> no local maximum.
        drift = 0.05 + 0.01 * np.sin(2 * np.pi * 0.5 * np.arange(1200) / FS)
        features = compute_finger_features(drift)
        self.assertNotEqual(features["tremor_peak_frequency_hz"], 4.0)

    def test_noise_has_low_prominence(self) -> None:
        noise = 0.05 + 0.001 * np.random.default_rng(1).normal(size=1200)
        self.assertLess(compute_finger_features(noise)["tremor_peak_prominence"], 3)

    def test_invalid_reasons(self) -> None:
        self.assertEqual(analyze_finger(np.ones(SPECTROGRAM_NPERSEG - 1))["invalid_reason"], TOO_SHORT)
        with_nan = tone(1200)
        with_nan[5] = np.nan
        self.assertEqual(analyze_finger(with_nan)["invalid_reason"], NON_FINITE)
        flat = analyze_finger(np.full(1200, 0.03))
        self.assertEqual((flat["status"], flat["invalid_reason"]), ("invalid", FLAT_SIGNAL))
        self.assertTrue(np.isnan(flat["tremor_mean_amplitude"]))

    def test_inactive_finger_is_flagged_not_invalid(self) -> None:
        row = analyze_finger(tone(1200, level=0.0005, tremor_amp=0.0002))
        self.assertEqual(row["status"], "valid")
        self.assertTrue(row["finger_inactive"])

    def test_one_row_per_attempt_hand_finger(self) -> None:
        data = pd.concat([make_measurement(300, attempt=1), make_measurement(1200, segments=(1, 2), attempt=2)],
                         ignore_index=True)
        rows = extract_measurement_finger_features(data)
        self.assertEqual(len(rows), 2 * 2 * 5)
        self.assertTrue((rows.loc[rows["attempt"] == 1, "invalid_reason"] == TOO_SHORT).all())
        self.assertTrue((rows.loc[rows["attempt"] == 2, "status"] == "valid").all())
        self.assertTrue((rows.loc[rows["attempt"] == 2, "n_samples"] == 1200).all())  # segment 2 ignored

    def test_attempt_without_analysed_segment(self) -> None:
        rows = extract_measurement_finger_features(make_measurement(1200, segments=(2,)))
        self.assertTrue((rows["invalid_reason"] == NO_SEGMENT).all())


class MetadataUpdateTests(unittest.TestCase):
    def test_update_rows_replace_rows_with_the_same_key(self) -> None:
        from rehapiano.common.io.metadata import METADATA_COLUMNS, read_all_metadata

        def row(patient, measurement, step):
            return dict(zip(METADATA_COLUMNS, [patient, "t", "70", "1", "1", "female", "right", measurement, step,
                                               "EXACT_FORCE"]))
        with tempfile.TemporaryDirectory() as folder:
            full, update = Path(folder) / "full.csv", Path(folder) / "update.csv"
            pd.DataFrame([row("A", "m1", "s"), row("WRONG", "m2", "s")]).to_csv(full, sep=";", index=False)
            pd.DataFrame([row("B", "m2", "s"), row("C", "m3", "s")]).to_csv(update, sep=";", index=False)
            metadata = read_all_metadata(full, [update]).set_index("measurement_id")["patient_id"]
        self.assertEqual(metadata.sort_index().to_dict(), {"m1": "A", "m2": "B", "m3": "C"})


class QualityTests(unittest.TestCase):
    def test_outlier_by_factor(self) -> None:
        table = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 2, "x": [1.0, 1.1, 0.9, 5.0, 1.0, 10.0]})
        flags = flag_outliers(table, ["g"], ["x"])
        self.assertEqual(flags["is_outlier"].tolist(), [False, False, False, True, False, False])
        self.assertIn("x x", flags.loc[3, "outlier_reason"])

    def test_attempt_outlier_flags_whole_hand(self) -> None:
        rows = []
        for attempt, force in zip(range(1, 5), [0.05, 0.05, 0.05, 0.5]):
            for finger in range(1, 6):
                rows.append(dict(patient_id="P", measurement_id="M", step_id=STEP, attempt=attempt, hand="left",
                                 finger=finger, status="valid", mean_force=force, tremor_mean_amplitude=0.001))
        flagged = flag_attempt_outliers(pd.DataFrame(rows))
        self.assertEqual(flagged.groupby("attempt")["attempt_outlier"].all().tolist(), [False, False, False, True])

    def test_first_attempt_is_practice_unless_single(self) -> None:
        rows = [dict(patient_id="P", measurement_id=measurement, step_id=STEP, attempt=attempt, hand="left",
                     finger=1, status="valid", mean_force=0.05, tremor_mean_amplitude=0.001)
                for measurement, n_attempts in (("M4", 4), ("M1", 1)) for attempt in range(1, n_attempts + 1)]
        flagged = flag_attempt_outliers(pd.DataFrame(rows))
        practice = flagged.set_index(["measurement_id", "attempt"])["practice_attempt"]
        self.assertTrue(practice[("M4", 1)])
        self.assertFalse(practice[("M4", 2)] or practice[("M1", 1)])


class WideTableTests(unittest.TestCase):
    def test_one_row_per_person_primary_step_only(self) -> None:
        people = [("C1", CONTROL_STEP_IDS[0], 0), ("C1", CONTROL_STEP_IDS[1], 0), ("P1", PATIENT_STEP_IDS[0], 1)]
        fingers, summary = [], []
        for patient_id, step_id, is_patient in people:
            summary.append(dict(patient_id=patient_id, step_id=step_id, is_patient=is_patient,
                                group="patient" if is_patient else "control", age=30, age_group="30-39",
                                gender="f", metadata_hand="right", n_measurements=1, n_attempts=4,
                                left_mean_force=0.01, right_mean_force=0.01, activity_ratio=1.0,
                                inactive_hand="none"))
            for hand in ("left", "right"):
                for finger, name in enumerate(("little", "ring", "middle", "index", "thumb"), start=1):
                    row = dict(patient_id=patient_id, step_id=step_id, hand=hand, finger=finger,
                               finger_name=name, finger_inactive=False)
                    row.update({column: float(finger) for column in compute_finger_features(tone(1200))})
                    fingers.append(row)
        wide = build_wide_table(pd.DataFrame(fingers), pd.DataFrame(summary))
        self.assertEqual(wide["patient_id"].tolist(), ["C1", "P1"])
        self.assertTrue((wide["step_id"] != CONTROL_STEP_IDS[1]).all())
        self.assertEqual(wide.loc[0, "right_thumb_mean_force"], 5.0)
        self.assertIn("left_little_inactive", wide.columns)


class ActivityTests(unittest.TestCase):
    def test_inactive_hand_rules(self) -> None:
        self.assertEqual(classify_hand_activity(0.04, 0.02)[1], "none")
        self.assertEqual(classify_hand_activity(0.003, 0.05)[1], "left")
        self.assertEqual(classify_hand_activity(0.05, -0.001)[1], "right")
        self.assertEqual(classify_hand_activity(np.nan, 0.05)[1], "unknown")
        self.assertEqual(classify_hand_activity(0.001, 0.0005)[1], "unknown")
        self.assertAlmostEqual(classify_hand_activity(0.01, 0.04)[0], 0.25)


class ReferenceAnalysisTests(unittest.TestCase):
    def test_weak_hand_uses_control_quantile(self) -> None:
        ratios = list(np.linspace(0.6, 1.0, 40))
        activity = pd.DataFrame({
            "patient_id": [f"C{i}" for i in range(40)] + ["P1", "P2"],
            "group": ["control"] * 40 + ["patient"] * 2, "age": 60, "gender": "f",
            "force_ratio": ratios + [0.3, 0.9], "force_weaker_hand": "left", "fluctuation_ratio": 1.0,
            "fluctuation_weaker_hand": "left", "left_mean_force": 0.01, "right_mean_force": 0.02})
        weak = weak_hand_patients(activity)
        self.assertEqual(weak.loc[weak["group"] == "patient", "patient_id"].tolist(), ["P1"])

    def test_z_scores_and_pattern(self) -> None:
        rng = np.random.default_rng(0)
        rows = []
        for person in range(30):  # reference controls aged >= 50
            for finger, name in enumerate(("little", "ring", "middle", "index", "thumb"), start=1):
                for hand in ("left", "right"):
                    rows.append(dict(patient_id=f"C{person}", group="control", age=60, hand=hand, finger=finger,
                                     finger_name=name, tremor_mean_amplitude=float(np.exp(rng.normal(0, 0.3)))))
        for hand in ("left", "right"):  # a patient with strongly elevated tremor on every finger
            for finger, name in enumerate(("little", "ring", "middle", "index", "thumb"), start=1):
                rows.append(dict(patient_id="P", group="patient", age=70, hand=hand, finger=finger, finger_name=name,
                                 tremor_mean_amplitude=float(np.exp(3.0))))
        z = z_scores(pd.DataFrame(rows))
        self.assertTrue((z.loc[z["group"] == "patient", "z"] > 5).all())
        self.assertAlmostEqual(z.loc[z["group"] == "control", "z"].median(), 0, delta=0.3)
        pattern = tremor_pattern(z).set_index("patient_id")
        self.assertEqual(pattern.loc["P", "pattern"], "generalized")
        self.assertEqual(pattern.loc["P", "n_elevated"], 10)

    def test_holm_adjust(self) -> None:
        np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


class ManifestTests(unittest.TestCase):
    def test_matching_reasons(self) -> None:
        row = dict(patient_id="P1", timestamp="t", age="1", height="1", weight="1", gender="f", hand="",
                   measurement_id=UUID, step_id=STEP, step_type="EXACT_FORCE")
        full = pd.DataFrame([row])
        metadata = label_patients(full, full.iloc[:0])
        other_uuid = "11111111-2222-3333-4444-555555555555"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(relative: str, examination_id: str, step_id: str = STEP) -> None:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"payload": {"examination_id": examination_id, "step_id": step_id}}))

            write(f"a/results_{UUID}_20250101-000000", UUID)
            write(f"b/results_{UUID}_20250101-000000", UUID)  # identical copy
            write(f"results_{other_uuid}_20250101-000000", other_uuid)  # no metadata
            write(f"results_{other_uuid}_20250101-000001", UUID)  # payload disagrees with name
            write("results_bad_name", UUID)
            write(f"results_{UUID}_20250101-000002", UUID, step_id="other-step")
            manifest, unmatched = build_manifest(metadata, root, ANALYSED_STEP_IDS, STEP_TYPE)

        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest.loc[0, "patient_id"], "P1")
        self.assertEqual(manifest.loc[0, "is_patient"], 0)
        self.assertEqual(sorted(unmatched["reason"]), sorted(
            ["duplicate_file", "no_match", "uuid_mismatch", "invalid_filename", "not_target_step"]))


if __name__ == "__main__":
    unittest.main()
