"""Stage 3: per-attempt features of every used recording, incremental by file content (MD5).

Writes ``ef_attempt_finger.csv``, ``ef_attempt_hand.csv``, ``rhythm_attempt.csv`` and ``extraction_errors.csv``.
A recording whose MD5 is already in a table is not processed again; rows of recordings that are no longer used
are dropped. ``recompute=True`` rebuilds everything.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from rehapiano.features import exact_force_game, rhythm_game
from rehapiano.io.metadata import resolve_project_path
from rehapiano.pipeline import settings
from rehapiano.preprocessing.pipeline import load_measurement_data

logger = logging.getLogger(__name__)

KEYS = ["md5", "patient_id", "measurement_id", "step_id"]
TABLES = {"exact_force": ["ef_attempt_finger", "ef_attempt_hand"], "rhythm": ["rhythm_attempt"]}


def _exact_force(record: dict, game: settings.Game) -> dict:
    data = load_measurement_data(resolve_project_path(record["file_path"]), *game.crop_s)
    fingers = exact_force_game.finger_features(data, settings.EF_BANDS)
    return {"ef_attempt_finger": fingers, "ef_attempt_hand": exact_force_game.hand_features(fingers)}


def _rhythm(record: dict, game: settings.Game) -> dict:
    data = load_measurement_data(resolve_project_path(record["file_path"]), *game.crop_s)
    rows = []
    for attempt, frame in data.loc[data["segment"] == 1].groupby("attempt"):
        row = {"attempt": int(attempt), **rhythm_game.attempt_features(frame)}
        row["valid"] = row["duration_s"] >= settings.RHYTHM_MIN_ATTEMPT_S
        rows.append(row)
    return {"rhythm_attempt": pd.DataFrame(rows)}


EXTRACTORS = {"exact_force": _exact_force, "rhythm": _rhythm}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"patient_id": str, "md5": str}) if path.exists() else pd.DataFrame()


def run(output_dir: Path = settings.OUTPUT_DIR, recompute: bool = False) -> dict:
    index = pd.read_csv(output_dir / "recordings.csv", dtype={"patient_id": str, "md5": str})
    used = index.loc[index["status"] == "used"]
    errors, results = [], {}
    for name, game in settings.GAMES.items():
        records = used.loc[used["game"] == name]
        existing = {t: (pd.DataFrame() if recompute else _read(output_dir / f"{t}.csv")) for t in TABLES[name]}
        done = set.intersection(*[set(f["md5"]) if not f.empty else set() for f in existing.values()])
        todo = records.loc[~records["md5"].isin(done)]
        logger.info("%s: %d recordings, %d already processed, %d to process", name, len(records),
                    len(records) - len(todo), len(todo))
        new = {t: [] for t in TABLES[name]}
        for i, record in enumerate(todo.to_dict("records"), start=1):
            try:
                tables = EXTRACTORS[name](record, game)
            except Exception as error:  # noqa: BLE001 - one broken file must not stop the run
                logger.error("%s failed: %r", record["file_path"], error)
                errors.append({"game": name, "file_path": record["file_path"], "error": repr(error)[:300]})
                continue
            for table, frame in tables.items():
                for key in reversed(KEYS):
                    frame.insert(0, key, record[key])
                new[table].append(frame)
            if i % 25 == 0:
                logger.info("%s: %d / %d", name, i, len(todo))
        for table in TABLES[name]:
            frame = pd.concat([existing[table], *new[table]], ignore_index=True)
            frame = frame.loc[frame["md5"].isin(records["md5"])] if not frame.empty else frame
            frame.to_csv(output_dir / f"{table}.csv", index=False)
            results[table] = frame
    pd.DataFrame(errors, columns=["game", "file_path", "error"]).to_csv(output_dir / "extraction_errors.csv",
                                                                       index=False)
    return results
