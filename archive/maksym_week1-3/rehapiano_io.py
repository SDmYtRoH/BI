"""Načítanie nameraných súborov Rehapiano (formát z roku 2025/2026).

Súbor `results_<examination_id>_<YYYYMMDD-HHMMSS>` je JSON:
    {"action": "STEP_RESULT", "payload": {"examination_id", "step_id", "additional_data", "raw"}}
Pole `raw` = base64 -> zlib -> pickle -> zoznam iterácií -> zoznam 72-bajtových paketov.

Paket (little-endian):
    8 B  int64   unixový čas v milisekundách   (POZOR: nie double, ako v rehapiano_data_converter.py)
    16x  float32 kanály 0..15                   (0-4 ľavá ruka, 8-12 pravá ruka, ostatné nevyužité)
"""
import base64
import json
import pickle
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PACKET = struct.Struct("<q16f")
LEFT_CHANNELS = [0, 1, 2, 3, 4]
RIGHT_CHANNELS = [8, 9, 10, 11, 12]
FINGER_CHANNELS = LEFT_CHANNELS + RIGHT_CHANNELS


@dataclass
class Iteration:
    """Jedna iterácia (opakovanie) hry prevzorkovaná na rovnomernú mriežku."""
    t: np.ndarray        # čas od začiatku iterácie [s], tvar (N,)
    x: np.ndarray        # sila prstov, tvar (N, 10) v poradí FINGER_CHANNELS
    fs: float            # vzorkovacia frekvencia rovnomernej mriežky [Hz]
    n_raw: int           # počet pôvodných paketov
    n_nan: int           # počet hodnôt NaN, ktoré boli doplnené interpoláciou


@dataclass
class Measurement:
    path: Path
    examination_id: str
    step_id: str
    additional_data: str
    iterations: list


def read_raw(path):
    """Vráti (payload, iterácie), kde iterácia je pole tvaru (n, 17): [čas_ms, kanál0..15]."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)["payload"]
    # pickle je bezpečný len pre dôveryhodné súbory z merania Rehapiano
    raw = pickle.loads(zlib.decompress(base64.b64decode(payload["raw"])))
    iterations = []
    for packets in raw:
        rows = [PACKET.unpack(p) for p in packets if p is not None]
        iterations.append(np.array(rows, dtype=float))
    return payload, iterations


def to_uniform_grid(packets, fs=1000.0):
    """Prevzorkuje jednu iteráciu na rovnomernú mriežku `fs` Hz.

    Časové značky prichádzajú nerovnomerne (rozdiely 0-5 ms, pakety chodia v dávkach).
    Pakety zaradíme do košov s dĺžkou 1/fs podľa časovej značky, v koši spriemerujeme,
    prázdne koše a hodnoty NaN doplníme lineárnou interpoláciou.
    """
    t_ms = packets[:, 0] - packets[0, 0]
    values = packets[:, 1:][:, FINGER_CHANNELS]
    n_nan = int(np.isnan(values).sum())

    step_ms = 1000.0 / fs
    bins = np.floor(t_ms / step_ms).astype(int)
    n_bins = bins[-1] + 1
    x = np.full((n_bins, values.shape[1]), np.nan)
    for ch in range(values.shape[1]):
        ok = np.isfinite(values[:, ch])
        sums = np.bincount(bins[ok], weights=values[ok, ch], minlength=n_bins)
        counts = np.bincount(bins[ok], minlength=n_bins)
        filled = counts > 0
        x[filled, ch] = sums[filled] / counts[filled]
        idx = np.arange(n_bins)
        x[~filled, ch] = np.interp(idx[~filled], idx[filled], x[filled, ch])
    t = np.arange(n_bins) / fs
    return Iteration(t=t, x=x, fs=fs, n_raw=len(packets), n_nan=n_nan)


def load_measurement(path, fs=1000.0):
    """Načíta súbor a vráti Measurement so všetkými iteráciami na rovnomernej mriežke."""
    payload, raw_iterations = read_raw(path)
    return Measurement(
        path=Path(path),
        examination_id=payload["examination_id"],
        step_id=payload["step_id"],
        additional_data=payload["additional_data"],
        iterations=[to_uniform_grid(p, fs) for p in raw_iterations],
    )


def channel_label(index):
    """Popisok stĺpca `index` v poli Iteration.x (0..9)."""
    ch = FINGER_CHANNELS[index]
    hand = "ľavá" if ch in LEFT_CHANNELS else "pravá"
    return f"kanál {ch} ({hand} ruka)"
