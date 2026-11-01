"""Person-level features that enter the statistics and the classifier, grouped into factors for the ablation."""

from __future__ import annotations

from typing import Dict, List, NamedTuple


class Feature(NamedTuple):
    factor: str
    description: str
    log: bool = False  # strongly right-skewed amplitude: log-transformed for the classifier


FEATURES: Dict[str, Feature] = {
    # EXACT_FORCE (hold a steady force), 3-15 s; oscillation = mean spectrogram amplitude of active fingers.
    "ef_tremor_amp": Feature("EF fast oscillation", "4-15 Hz amplitude, mean of both hands", True),
    "ef_tremor_amp_max": Feature("EF fast oscillation", "4-15 Hz amplitude of the more oscillating hand", True),
    "ef_slow_amp": Feature("EF slow force control", "0.2-3 Hz amplitude, mean of both hands", True),
    "ef_tremor_share": Feature("EF fast-to-slow share", "4-15 Hz / (0.2-3 Hz + 4-15 Hz) amplitude"),
    "ef_balance": Feature("EF hand balance", "weaker / stronger hand mean finger force"),
    "ef_tremor_balance": Feature("EF hand balance", "weaker / stronger hand 4-15 Hz amplitude"),
    # RHYTHM (press and release repeatedly), attempts 2-4, 2-30 s.
    "rh_press_rate_hz": Feature("Rhythm tempo", "presses per second"),
    "rh_interval_cv": Feature("Rhythm regularity", "SD / mean of intervals between presses"),
    "rh_amplitude_strong": Feature("Rhythm press force", "median press amplitude, stronger hand", True),
    "rh_amplitude_weak": Feature("Rhythm press force", "median press amplitude, weaker hand", True),
    "rh_balance_range": Feature("Rhythm hand balance", "weaker / stronger hand force range"),
    "rh_balance_press": Feature("Rhythm hand balance", "weaker / stronger hand press amplitude"),
    "rh_amplitude_cv": Feature("Rhythm force stability", "SD / mean of press amplitudes"),
    "rh_sync_lag_ms": Feature("Rhythm hand synchrony", "median lag between left and right maxima (ms)", True),
    "rh_release_ratio": Feature("Rhythm release", "force left after a press / peak force"),
    "rh_fatigue_amplitude": Feature("Rhythm fatigue", "press amplitude, 2nd / 1st half"),
    "rh_fatigue_rate": Feature("Rhythm fatigue", "press count, 2nd / 1st half"),
}

GAME_OF = {name: ("EXACT_FORCE" if name.startswith("ef_") else "RHYTHM") for name in FEATURES}


def factors() -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for name, feature in FEATURES.items():
        result.setdefault(feature.factor, []).append(name)
    return result
