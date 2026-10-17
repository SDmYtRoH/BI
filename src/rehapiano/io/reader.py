"""Decode raw RehaPiano ``results_*`` files.

File layers: JSON -> ``payload.raw`` (Base64) -> zlib -> pickle (a list of
attempts, each a list of 72-byte packets). A packet is a little-endian
``uint64`` millisecond timestamp followed by 16 ``float32`` channels; only the
ten channels in ``config.SOURCE_CHANNELS`` carry finger forces.

No cleaning happens here: the output is the raw ~1000 Hz stream, NaNs included.
"""

from __future__ import annotations

import base64
import io
import json
import pickle
import zlib
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from rehapiano.config import CHANNELS_PER_PACKET, SOURCE_CHANNELS

PathLike = Union[str, Path]
RawAttempt = Tuple[np.ndarray, np.ndarray]  # (timestamps_ms, sensor_values[n, 10])

PACKET_DTYPE = np.dtype([("timestamp_ms", "<u8"), ("channels", "<f4", (CHANNELS_PER_PACKET,))])


class _DataOnlyUnpickler(pickle.Unpickler):
    """Unpickle built-in data containers without importing arbitrary classes."""

    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError("Class loading is disabled")

    def persistent_load(self, persistent_id):
        raise pickle.UnpicklingError("Persistent objects are disabled")


def read_raw_attempts(path: PathLike) -> List[RawAttempt]:
    """Return ``(timestamps_ms, sensor_values)`` for every attempt in the file.

    ``sensor_values`` has one column per entry of ``config.SENSOR_COLUMNS``.
    Raises ``ValueError`` for a structurally broken file.
    """

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        encoded_raw = document["payload"]["raw"]
    except (KeyError, TypeError) as error:
        raise ValueError("The file does not contain payload.raw") from error

    packed = zlib.decompress(base64.b64decode(encoded_raw, validate=True))
    attempts = _DataOnlyUnpickler(io.BytesIO(packed)).load()
    if not isinstance(attempts, (list, tuple)):
        raise ValueError("Expected a sequence of measurement attempts")

    decoded: List[RawAttempt] = []
    for attempt_number, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, (list, tuple)):
            raise ValueError(f"Attempt {attempt_number} is not a packet sequence")
        packets = [packet for packet in attempt if packet is not None]
        if not packets:
            raise ValueError(f"Attempt {attempt_number} contains no data packets")
        if any(not isinstance(packet, (bytes, bytearray)) or len(packet) != PACKET_DTYPE.itemsize
               for packet in packets):
            raise ValueError(f"Attempt {attempt_number} contains an invalid packet")

        packet_array = np.frombuffer(b"".join(packets), dtype=PACKET_DTYPE)
        timestamps_ms = packet_array["timestamp_ms"].astype(np.int64)
        sensor_values = packet_array["channels"][:, SOURCE_CHANNELS].astype(np.float64)
        decoded.append((timestamps_ms, sensor_values))
    return decoded
