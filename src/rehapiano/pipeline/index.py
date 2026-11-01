"""Stage 1: find every recording, identify it and match it to the metadata of the analysed games."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from rehapiano.io.metadata import (
    METADATA_COLUMNS,
    extract_file_uuid,
    find_measurement_files,
    read_metadata,
    to_project_relative,
)
from rehapiano.pipeline import settings

logger = logging.getLogger(__name__)

IDENTITY_COLUMNS = ["file_path", "size", "mtime_ns", "file_uuid", "examination_id", "step_id", "md5", "error"]


def _identify(path: Path) -> dict:
    stat = path.stat()
    row = {"file_path": to_project_relative(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
           "file_uuid": extract_file_uuid(path.name), "examination_id": None, "step_id": None, "md5": None, "error": ""}
    try:
        content = path.read_bytes()
        row["md5"] = hashlib.md5(content).hexdigest()
        payload = json.loads(content).get("payload") or {}
        row["examination_id"] = (payload.get("examination_id") or "").lower() or None
        row["step_id"] = payload.get("step_id")
    except (OSError, ValueError, AttributeError) as error:
        row["error"] = repr(error)[:200]
    return row


def identify_files(measurements_dir: Path, cache_path: Path) -> pd.DataFrame:
    """Payload identity and MD5 of every ``results_*`` file; unchanged files are taken from the cache."""

    cached = (pd.read_csv(cache_path, dtype={"file_path": str, "md5": str, "error": str}).fillna({"error": ""})
              if cache_path.exists() else pd.DataFrame(columns=IDENTITY_COLUMNS))
    known = {(r.file_path, int(r.size), int(r.mtime_ns)): r._asdict() for r in cached.itertuples(index=False)}
    rows, new = [], 0
    for path in find_measurement_files(measurements_dir):
        stat = path.stat()
        key = (to_project_relative(path), stat.st_size, stat.st_mtime_ns)
        if key in known:
            rows.append(known[key])
            continue
        rows.append(_identify(path))
        new += 1
        if new % 200 == 0:
            logger.info("identified %d new files", new)
    identity = pd.DataFrame(rows, columns=IDENTITY_COLUMNS)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    identity.to_csv(cache_path, index=False)
    logger.info("%d files, %d newly read", len(identity), new)
    return identity


def read_all_metadata(metadata_dir: Path, patient_files: Iterable[str]) -> pd.DataFrame:
    """Union of all metadata CSVs (exact duplicates collapsed) with ``cohort`` and ``source_file``."""

    frames = []
    for path in sorted(metadata_dir.glob("*.csv")):
        try:
            frame = read_metadata(path)
        except ValueError as error:
            logger.warning("skipping %s: %s", path.name, error)
            continue
        frames.append(frame[METADATA_COLUMNS].assign(source_file=path.name))
    metadata = pd.concat(frames, ignore_index=True)
    patient_keys = set(map(tuple, metadata.loc[metadata["source_file"].isin(set(patient_files)),
                                               ["measurement_id", "step_id"]].to_numpy()))
    metadata["cohort"] = [("patient" if key in patient_keys else "control")
                          for key in map(tuple, metadata[["measurement_id", "step_id"]].to_numpy())]
    sources = metadata.groupby(["measurement_id", "step_id"])["source_file"].agg(lambda v: ";".join(sorted(set(v))))
    metadata = metadata.drop(columns="source_file").drop_duplicates()
    return metadata.merge(sources.rename("source_files").reset_index(), on=["measurement_id", "step_id"])


def build_index(identity: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """One row per file with ``game``, ``status`` (used / copy / excluded) and ``reason``."""

    step_game = {step: name for name, game in settings.GAMES.items() for step in game.steps}
    step_cohort = {step: cohort for game in settings.GAMES.values() for step, cohort in game.steps.items()}
    rows_per_key = metadata.groupby(["measurement_id", "step_id"]).size()
    first_row = metadata.drop_duplicates(["measurement_id", "step_id"])  # conflicts are flagged below
    index = identity.merge(first_row.rename(columns={"measurement_id": "examination_id"}),
                           on=["examination_id", "step_id"], how="left")
    index["game"] = index["step_id"].map(step_game)

    def reason(row) -> str:
        if row["error"]:
            return "unreadable_file"
        if row["file_uuid"] is None or row["file_uuid"] != row["examination_id"]:
            return "uuid_mismatch"
        if pd.isna(row["game"]):
            return "step_not_analysed"
        if pd.isna(row["patient_id"]):
            return "no_metadata"
        if rows_per_key.get((row["examination_id"], row["step_id"]), 0) > 1:
            return "conflicting_metadata"
        if row["cohort"] != step_cohort[row["step_id"]]:
            return "cohort_step_mismatch"
        return ""

    index["reason"] = index.apply(reason, axis=1)
    index["status"] = index["reason"].map(lambda r: "excluded" if r else "used")

    # The same recording stored in several folders: keep one byte-identical copy, exclude conflicting ones.
    used = index["status"] == "used"
    for _, group in index.loc[used].sort_values("file_path").groupby(["examination_id", "step_id"]):
        if len(group) == 1:
            continue
        if group["md5"].nunique() == 1:
            index.loc[group.index[1:], ["status", "reason"]] = ["copy", "identical_copy"]
        else:
            index.loc[group.index, ["status", "reason"]] = ["excluded", "conflicting_copies"]
    index = index.rename(columns={"examination_id": "measurement_id"})
    index["timestamp"] = pd.to_datetime(index["timestamp"], format="mixed", errors="coerce")
    columns = ["file_path", "status", "reason", "game", "patient_id", "cohort", "measurement_id", "step_id", "step_type",
               "timestamp", "md5", "size", "source_files"]
    return index[columns].sort_values(["status", "game", "patient_id", "timestamp"]).reset_index(drop=True)


def run(output_dir: Path = settings.OUTPUT_DIR, measurements_dir: Path = settings.MEASUREMENTS_DIR,
        metadata_dir: Path = settings.METADATA_DIR,
        patient_files: Iterable[str] = settings.PATIENT_METADATA_FILES) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = identify_files(measurements_dir, output_dir / "cache" / "file_identity.csv")
    metadata = read_all_metadata(metadata_dir, patient_files)
    index = build_index(identity, metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(output_dir / "recordings.csv", index=False)
    metadata.to_csv(output_dir / "cache" / "metadata.csv", index=False)  # raw metadata rows: never committed

    analysed = metadata.loc[metadata["step_id"].isin(index["step_id"].dropna().unique())
                            & metadata["step_type"].isin([g.step_type for g in settings.GAMES.values()])]
    expected = analysed.loc[analysed["step_id"].isin({s for g in settings.GAMES.values() for s in g.steps}),
                            ["measurement_id", "step_id"]].drop_duplicates()
    found = index.loc[index["status"] == "used", ["measurement_id", "step_id"]]
    missing = expected.merge(found, how="left", indicator=True)["_merge"].eq("left_only").sum()
    logger.info("used recordings per game: %s; metadata rows without a usable file: %d",
                index.loc[index["status"] == "used", "game"].value_counts().to_dict(), missing)
    other_steps = (metadata.loc[~metadata["step_id"].isin({s for g in settings.GAMES.values() for s in g.steps})]
                   .groupby(["step_type", "step_id"]).size().rename("metadata_rows").reset_index())
    other_steps.to_csv(output_dir / "steps_not_analysed.csv", index=False)
    return index, metadata
