"""Metadata loading and safe matching of RehaPiano measurement files.

Matching rule (verified on the real data, see README.md):

* the UUID in the file name ``results_<uuid>_<YYYYMMDD-HHMMSS>`` equals
  ``payload.examination_id`` inside the file and ``measurement_id`` in the CSVs;
* one measurement contains several games, so the UUID alone is not unique.
  The game is identified by ``payload.step_id`` inside the file;
* the matching key is therefore ``(measurement_id, step_id)``.

The CSV column ``filename`` is outdated and is never used. ``additional_data``
does not contain file UUIDs and is not used for matching either.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd

from rehapiano.config import AGE_GROUP_WIDTH_YEARS, ANALYSED_STEP_IDS, MATCH_KEY, PROJECT_ROOT, STEP_TYPE

PathLike = Union[str, Path]
TARGET_STEP_IDS = frozenset(ANALYSED_STEP_IDS)

_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
FILENAME_PATTERN = re.compile(rf"^results_({_UUID})_(\d{{8}}-\d{{6}})$")

METADATA_COLUMNS = [
    "patient_id",
    "timestamp",
    "age",
    "height",
    "weight",
    "gender",
    "hand",
    "measurement_id",
    "step_id",
    "step_type",
]
UNMATCHED_COLUMNS = ["file_path", "file_uuid", "payload_step_id", "reason", "detail"]


def read_metadata(path: PathLike) -> pd.DataFrame:
    """Read a semicolon-separated metadata CSV with all columns as strings."""

    frame = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, na_values=[""])
    missing = set(METADATA_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return frame


def filter_target_steps(
    metadata: pd.DataFrame,
    step_ids: Iterable[str] = TARGET_STEP_IDS,
) -> pd.DataFrame:
    """Keep only EXACT_FORCE rows of the selected game steps."""

    selected = metadata["step_type"].eq(STEP_TYPE) & metadata["step_id"].isin(set(step_ids))
    return metadata.loc[selected].copy()


def label_patients(full: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """Return deduplicated metadata with ``is_patient`` and ``n_metadata_rows``.

    Exact duplicate rows (present in both CSVs) are collapsed. A key that still
    has several different rows is kept with ``n_metadata_rows > 1`` so that the
    matcher can report it as ``multiple_matches`` instead of guessing.
    """

    columns = METADATA_COLUMNS
    labelled = full[columns].drop_duplicates().copy()
    patient_keys = pd.MultiIndex.from_frame(patients[MATCH_KEY].drop_duplicates())
    labelled["is_patient"] = (
        pd.MultiIndex.from_frame(labelled[MATCH_KEY]).isin(patient_keys).astype(int)
    )
    labelled["n_metadata_rows"] = labelled.groupby(MATCH_KEY)["patient_id"].transform("size")
    return labelled.reset_index(drop=True)


def extract_file_uuid(file_name: str) -> Optional[str]:
    """Extract the lower-case UUID from ``results_<uuid>_<date>-<time>``."""

    match = FILENAME_PATTERN.match(Path(file_name).name)
    return match.group(1).lower() if match else None


def read_file_identity(path: PathLike) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(examination_id, step_id)`` stored in the file payload."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = document.get("payload") or {}
    return payload.get("examination_id"), payload.get("step_id")


def to_project_relative(path: PathLike) -> str:
    """Path relative to the project root when possible (portable CSV outputs)."""

    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_project_path(path: PathLike) -> Path:
    """Inverse of :func:`to_project_relative`."""

    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _file_digest(path: PathLike) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def find_measurement_files(measurements_dir: PathLike) -> list[Path]:
    """Recursively list ``results_*`` files in a stable order."""

    return sorted(path for path in Path(measurements_dir).rglob("results_*") if path.is_file())


def build_manifest(
    metadata: pd.DataFrame,
    measurements_dir: PathLike,
    step_ids: Iterable[str] = TARGET_STEP_IDS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Match measurement files to metadata rows.

    Parameters
    ----------
    metadata:
        Output of :func:`label_patients` (all step types; filtering happens here).
    measurements_dir:
        Root directory searched recursively for ``results_*`` files.

    Returns
    -------
    manifest, unmatched
        ``manifest`` has one row per matched (measurement_id, step_id).
        ``unmatched`` lists every other file with a ``reason``:
        ``invalid_filename``, ``unreadable_file``, ``uuid_mismatch``,
        ``missing_step_id``, ``not_target_step``, ``no_match``,
        ``multiple_matches``, ``duplicate_file`` (byte-identical copy of a
        processed file) or ``conflicting_files`` (different files, same key).
    """

    step_ids = set(step_ids)
    target = filter_target_steps(metadata, step_ids)
    target_by_key = {key: rows for key, rows in target.groupby(MATCH_KEY)}

    unmatched: list[dict] = []
    candidates: list[dict] = []

    def reject(path: Path, uuid: Optional[str], step: Optional[str], reason: str, detail: str = "") -> None:
        unmatched.append(
            {"file_path": str(path), "file_uuid": uuid, "payload_step_id": step, "reason": reason, "detail": detail}
        )

    for path in find_measurement_files(measurements_dir):
        file_uuid = extract_file_uuid(path.name)
        if file_uuid is None:
            reject(path, None, None, "invalid_filename")
            continue
        try:
            examination_id, step_id = read_file_identity(path)
        except (OSError, ValueError, AttributeError) as error:
            reject(path, file_uuid, None, "unreadable_file", repr(error)[:200])
            continue
        if not examination_id or examination_id.lower() != file_uuid:
            reject(path, file_uuid, step_id, "uuid_mismatch", f"payload.examination_id={examination_id}")
            continue
        if not step_id:
            reject(path, file_uuid, None, "missing_step_id")
            continue
        if step_id not in step_ids:
            reject(path, file_uuid, step_id, "not_target_step")
            continue

        rows = target_by_key.get((file_uuid, step_id))
        if rows is None:
            reject(path, file_uuid, step_id, "no_match", "no EXACT_FORCE metadata row for (measurement_id, step_id)")
            continue
        if len(rows) > 1:
            reject(path, file_uuid, step_id, "multiple_matches", f"{len(rows)} different metadata rows")
            continue
        candidates.append({"file_path": str(path), "file_uuid": file_uuid, **rows.iloc[0].to_dict()})

    # The same recording can be stored in several directories.
    manifest_rows: list[dict] = []
    for _, group in pd.DataFrame(candidates, columns=["file_path", "file_uuid", *metadata.columns]).groupby(
        MATCH_KEY, sort=False
    ):
        records = group.sort_values("file_path").to_dict("records")
        if len(records) == 1:
            manifest_rows.append(records[0])
            continue
        digests = {_file_digest(record["file_path"]) for record in records}
        if len(digests) == 1:
            manifest_rows.append(records[0])
            for record in records[1:]:
                reject(Path(record["file_path"]), record["file_uuid"], record["step_id"],
                       "duplicate_file", f"identical to {to_project_relative(records[0]['file_path'])}")
        else:
            for record in records:
                reject(Path(record["file_path"]), record["file_uuid"], record["step_id"],
                       "conflicting_files", f"{len(records)} different files for the same key")

    manifest = pd.DataFrame(manifest_rows).rename(columns={"hand": "metadata_hand"})
    if not manifest.empty:
        manifest = manifest.drop(columns="n_metadata_rows").sort_values(["patient_id", "timestamp", "file_path"])
        leading = ["file_path", "file_uuid", "patient_id", "measurement_id", "step_id", "step_type", "is_patient"]
        manifest = manifest[leading + [c for c in manifest.columns if c not in leading]]
    unmatched_frame = pd.DataFrame(unmatched, columns=UNMATCHED_COLUMNS)
    for frame in (manifest, unmatched_frame):
        if not frame.empty:
            frame["file_path"] = frame["file_path"].map(to_project_relative)
    return manifest.reset_index(drop=True), unmatched_frame


def build_inventory(manifest: pd.DataFrame, unmatched: pd.DataFrame) -> pd.DataFrame:
    """One row per ``results_*`` file: ``status`` (matched / excluded) and ``reason``."""

    matched = manifest[["file_path", "file_uuid", "step_id", "patient_id", "is_patient"]].assign(
        status="matched", reason="", detail=""
    )
    excluded = unmatched.rename(columns={"payload_step_id": "step_id"}).assign(status="excluded")
    inventory = pd.concat([matched, excluded], ignore_index=True)
    columns = ["file_path", "file_uuid", "step_id", "patient_id", "is_patient", "status", "reason", "detail"]
    return inventory[columns].sort_values("file_path").reset_index(drop=True)


def age_group_label(age: float, width: int = AGE_GROUP_WIDTH_YEARS) -> object:
    """``"20-29"`` style label of a ``width``-year age interval."""

    if not np.isfinite(age):
        return pd.NA
    low = int(age // width * width)
    return f"{low}-{low + width - 1}"


def add_demographics(manifest: pd.DataFrame) -> pd.DataFrame:
    """Add numeric ``age``, ``age_group`` and ``group`` (patient / control)."""

    result = manifest.copy()
    result["age"] = pd.to_numeric(result["age"], errors="coerce")
    result["age_group"] = result["age"].map(age_group_label)
    result["group"] = np.where(result["is_patient"] == 1, "patient", "control")
    return result
