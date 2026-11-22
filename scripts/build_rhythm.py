"""RHYTHM step 1: press detection and attempt-level tables from the raw recordings (M. Kuznietsov).

Run from the repository root:
    python scripts/build_rhythm.py            # all files
    python scripts/build_rhythm.py --limit 5  # quick check
Writes to ``outputs/rhythm/``: ``rhythm_by_attempt.csv`` (attempt x hand, hand balance),
``rhythm_parameters_by_attempt.csv`` (attempt, 10 rhythm parameters) and ``rhythm_inventory.csv``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rehapiano.common import config  # noqa: E402
from rehapiano.rhythm import config as rhythm_config  # noqa: E402
from rehapiano.rhythm.presses import (  # noqa: E402
    extract_measurement_rhythm_features,
    extract_measurement_rhythm_parameters,
)
from rehapiano.common.io.metadata import (  # noqa: E402
    add_demographics,
    build_inventory,
    build_manifest,
    label_patients,
    read_all_metadata,
    read_metadata,
    resolve_project_path,
)
from rehapiano.common.preprocessing.pipeline import load_measurement_data  # noqa: E402

logger = logging.getLogger("build_rhythm")

CONTEXT_COLUMNS = ["patient_id", "is_patient", "group", "age", "age_group", "gender", "metadata_hand",
                   "step_id", "measurement_id", "timestamp", "file_path"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--full-csv", type=Path, default=config.FULL_METADATA_CSV)
    parser.add_argument("--patients-csv", type=Path, default=config.PATIENTS_METADATA_CSV)
    parser.add_argument("--update-csvs", type=Path, nargs="*", default=list(config.METADATA_UPDATE_CSVS),
                        help="later metadata exports; their rows replace rows of --full-csv")
    parser.add_argument("--measurements-dir", type=Path, default=config.MEASUREMENTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=rhythm_config.RHYTHM_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N matched files")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    logging.basicConfig(level=arguments.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    output_dir: Path = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = label_patients(read_all_metadata(arguments.full_csv, arguments.update_csvs),
                              read_metadata(arguments.patients_csv))
    manifest, unmatched = build_manifest(metadata, arguments.measurements_dir, rhythm_config.RHYTHM_STEP_IDS,
                                         step_type=rhythm_config.RHYTHM_STEP_TYPE)
    manifest = add_demographics(manifest)
    to_process = manifest.head(arguments.limit) if arguments.limit else manifest

    frames, parameter_frames, failures = [], [], []
    for index, record in enumerate(to_process.to_dict("records"), start=1):
        path = record["file_path"]
        try:
            data = load_measurement_data(resolve_project_path(path), rhythm_config.RHYTHM_CROP_START_S,
                                         rhythm_config.RHYTHM_CROP_END_S)
            features = extract_measurement_rhythm_features(data)
            parameters = extract_measurement_rhythm_parameters(data)
        except Exception as error:  # noqa: BLE001 - one bad file must not stop the run
            logger.error("Failed to process %s: %r", path, error)
            failures.append({"file_path": path, "reason": "processing_error", "detail": repr(error)[:300]})
            continue
        for column in reversed(CONTEXT_COLUMNS):
            features.insert(0, column, record.get(column))
            parameters.insert(0, column, record.get(column))
        frames.append(features)
        parameter_frames.append(parameters)
        if index % 25 == 0:
            logger.info("Processed %d / %d files", index, len(to_process))

    failed = pd.DataFrame(failures, columns=["file_path", "reason", "detail"])
    inventory = build_inventory(manifest[~manifest["file_path"].isin(failed["file_path"])],
                                pd.concat([unmatched, failed], ignore_index=True))
    inventory.to_csv(output_dir / "rhythm_inventory.csv", index=False)
    by_attempt = pd.concat(frames, ignore_index=True)
    by_attempt.to_csv(output_dir / "rhythm_by_attempt.csv", index=False)
    pd.concat(parameter_frames, ignore_index=True).to_csv(output_dir / "rhythm_parameters_by_attempt.csv", index=False)

    matched = inventory["status"] == "matched"
    print(f"files: {len(inventory)} scanned, {int(matched.sum())} matched RHYTHM recordings, "
          f"{failed.shape[0]} failed")
    print(inventory.loc[~matched, "reason"].value_counts().to_string())
    attempts = by_attempt.drop_duplicates(["measurement_id", "attempt"])
    print(f"attempts: {len(attempts)}, valid: {int((attempts['status'] == 'valid').sum())}")
    print(f"people: {by_attempt.groupby('group')['patient_id'].nunique().to_dict()}")
    print(f"\nOutputs written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
