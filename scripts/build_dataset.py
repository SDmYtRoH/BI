"""Build the per-finger EXACT_FORCE feature tables (no statistics, no plots).

Run from the project root:
    python scripts/build_dataset.py            # all defaults from rehapiano.config
    python scripts/build_dataset.py --limit 5  # quick check on the first 5 files
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rehapiano import config  # noqa: E402
from rehapiano.aggregation import (  # noqa: E402
    aggregate_by_measurement,
    aggregate_by_person,
    build_wide_table,
    flag_attempt_outliers,
    summarize_persons,
)
from rehapiano.features.finger import extract_measurement_finger_features  # noqa: E402
from rehapiano.io.metadata import (  # noqa: E402
    add_demographics,
    build_inventory,
    build_manifest,
    filter_target_steps,
    label_patients,
    read_metadata,
    resolve_project_path,
)
from rehapiano.preprocessing.pipeline import load_measurement_data  # noqa: E402
from rehapiano.quality import VALID  # noqa: E402

logger = logging.getLogger("build_dataset")

CONTEXT_COLUMNS = [
    "patient_id", "is_patient", "group", "age", "age_group", "gender", "metadata_hand",
    "step_id", "measurement_id", "timestamp", "file_path",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--full-csv", type=Path, default=config.FULL_METADATA_CSV)
    parser.add_argument("--patients-csv", type=Path, default=config.PATIENTS_METADATA_CSV)
    parser.add_argument("--measurements-dir", type=Path, default=config.MEASUREMENTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--step-ids", nargs="+", default=list(config.ANALYSED_STEP_IDS))
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N matched files")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def process_manifest(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess every file and extract finger features.

    Returns ``(features_by_attempt, failed_files)``; a broken file is logged and
    reported, it does not stop the run.
    """

    frames, failures = [], []
    for index, record in enumerate(manifest.to_dict("records"), start=1):
        path = record["file_path"]
        try:
            features = extract_measurement_finger_features(load_measurement_data(resolve_project_path(path)))
        except Exception as error:  # noqa: BLE001 - one bad file must not stop the run
            logger.error("Failed to process %s: %r", path, error)
            failures.append({"file_path": path, "reason": "processing_error", "detail": repr(error)[:300]})
            continue
        for column in reversed(CONTEXT_COLUMNS):
            features.insert(0, column, record.get(column))
        frames.append(features)
        if index % 25 == 0:
            logger.info("Processed %d / %d files", index, len(manifest))
    features_by_attempt = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return features_by_attempt, pd.DataFrame(failures, columns=["file_path", "reason", "detail"])


def build_summary(
    inventory: pd.DataFrame,
    n_metadata_without_file: int,
    features_by_attempt: pd.DataFrame,
    features_by_measurement: pd.DataFrame,
    person_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Counts of recordings at every level, as ``level, item, value`` rows."""

    rows = [("files", "results_* files found", len(inventory)),
            ("files", "matched to EXACT_FORCE metadata", int((inventory["status"] == "matched").sum())),
            ("files", "metadata rows without a file", n_metadata_without_file)]
    excluded = inventory.loc[inventory["status"] == "excluded", "reason"].value_counts()
    rows += [("files", f"excluded: {reason}", int(count)) for reason, count in excluded.items()]

    if not features_by_attempt.empty:
        valid = features_by_attempt["status"] == VALID
        attempts = features_by_attempt.drop_duplicates(["measurement_id", "step_id", "attempt"])
        attempt_hands = features_by_attempt.drop_duplicates(["measurement_id", "step_id", "attempt", "hand"])
        rows += [("attempts", "attempts", len(attempts)),
                 ("attempts", "attempt x hand units", len(attempt_hands)),
                 ("attempts", "attempt x hand flagged as outlier", int(attempt_hands["attempt_outlier"].sum())),
                 ("fingers", "attempt x finger rows", len(features_by_attempt)),
                 ("fingers", "valid", int(valid.sum())),
                 ("fingers", "invalid", int((~valid).sum()))]
        reasons = features_by_attempt.loc[~valid, "invalid_reason"].value_counts()
        rows += [("fingers", f"invalid: {reason}", int(count)) for reason, count in reasons.items()]
        rows += [("fingers", "valid but finger_inactive",
                  int(features_by_attempt.loc[valid, "finger_inactive"].astype(bool).sum())),
                 ("fingers", "used in averages (valid, not outlier attempt)",
                  int((valid & ~features_by_attempt["attempt_outlier"]).sum()))]
        measurement_hands = features_by_measurement.drop_duplicates(["measurement_id", "step_id", "hand"])
        rows += [("measurements", "measurement x hand units", len(measurement_hands)),
                 ("measurements", "measurement x hand flagged as outlier",
                  int(measurement_hands["measurement_outlier"].sum()))]
    for group, frame in person_summary.groupby("group"):
        rows += [("people", f"{group}: people", frame["patient_id"].nunique()),
                 ("people", f"{group}: measurements", int(frame["n_measurements"].sum()))]
        for value, count in frame["inactive_hand"].value_counts().items():
            rows.append(("people", f"{group}: inactive_hand={value} (person x step)", int(count)))
    return pd.DataFrame(rows, columns=["level", "item", "value"])


def main() -> None:
    arguments = parse_arguments()
    logging.basicConfig(level=arguments.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    output_dir: Path = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = label_patients(read_metadata(arguments.full_csv), read_metadata(arguments.patients_csv))
    manifest, unmatched = build_manifest(metadata, arguments.measurements_dir, arguments.step_ids)
    manifest = add_demographics(manifest)
    target_keys = filter_target_steps(metadata, arguments.step_ids)[config.MATCH_KEY].drop_duplicates()
    n_metadata_without_file = int(
        (target_keys.merge(manifest[config.MATCH_KEY], how="left", indicator=True)["_merge"] == "left_only").sum()
    )

    to_process = manifest.head(arguments.limit) if arguments.limit else manifest
    features_by_attempt, failed = process_manifest(to_process)
    inventory = build_inventory(manifest[~manifest["file_path"].isin(failed["file_path"])],
                                pd.concat([unmatched, failed], ignore_index=True))
    inventory.to_csv(output_dir / "recordings_inventory.csv", index=False)
    if features_by_attempt.empty:
        logger.error("No features were extracted")
        return

    features_by_attempt = flag_attempt_outliers(features_by_attempt)
    features_by_measurement = aggregate_by_measurement(features_by_attempt)
    features_by_person = aggregate_by_person(features_by_measurement)
    person_summary = summarize_persons(features_by_attempt, features_by_measurement, features_by_person)
    summary = build_summary(inventory, n_metadata_without_file, features_by_attempt,
                            features_by_measurement, person_summary)

    features_by_attempt.to_csv(output_dir / "features_by_attempt_finger.csv", index=False)
    features_by_measurement.to_csv(output_dir / "features_by_measurement_finger.csv", index=False)
    features_by_person.to_csv(output_dir / "features_by_person_finger.csv", index=False)
    person_summary.to_csv(output_dir / "person_summary.csv", index=False)
    build_wide_table(features_by_person, person_summary).to_csv(output_dir / "dataset_wide.csv", index=False)
    summary.to_csv(output_dir / "processing_summary.csv", index=False)

    print("\n===== Summary =====")
    for level, frame in summary.groupby("level", sort=False):
        print(f"[{level}]")
        for item, value in zip(frame["item"], frame["value"]):
            print(f"    {item:<52} {value}")
    print(f"\nOutputs written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
