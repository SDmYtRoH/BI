"""Run the whole project after the data were updated: raw recordings -> tables -> statistics -> notebooks.

Run from the repository root:
    python scripts/run_all.py                    # everything (full cross-validation, ~15-20 min)
    python scripts/run_all.py --quick            # fewer CV repeats / permutations (a few minutes, for checks)
    python scripts/run_all.py --only rhythm      # one exercise (exact_force / rhythm / combined)
    python scripts/run_all.py --skip-build       # statistics only, raw recordings are not read again
    python scripts/run_all.py --notebooks        # also re-execute the notebooks (figures and texts)

Order: EXACT_FORCE first (the RHYTHM hand balance compares with its hand_activity.csv), then RHYTHM,
then both exercises together (needs the person tables of both).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# (exercise, is a raw-data build step, script)
STEPS = [
    ("exact_force", True, "build_exact_force.py"),
    ("exact_force", False, "run_exact_force_stats.py"),
    ("exact_force", False, "run_exact_force_analysis.py"),
    ("rhythm", True, "build_rhythm.py"),
    ("rhythm", False, "run_rhythm_hand_balance.py"),
    ("rhythm", False, "run_rhythm_analysis.py"),
    ("combined", False, "run_combined_analysis.py"),
]
CV_SCRIPTS = {"run_exact_force_analysis.py", "run_rhythm_analysis.py", "run_combined_analysis.py"}
NOTEBOOKS = {
    "exact_force": ["notebooks/exact_force/analyza_prstov.ipynb",
                    "notebooks/exact_force/statisticka_analyza_exact_force.ipynb"],
    "rhythm": ["notebooks/rhythm/rovnovaha_ruk.ipynb", "notebooks/rhythm/statisticka_analyza_rytmus.ipynb"],
    "combined": ["notebooks/combined/klasifikator_oba_testy.ipynb"],
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", choices=["exact_force", "rhythm", "combined"], action="append",
                        help="run only these parts (can be repeated)")
    parser.add_argument("--skip-build", action="store_true", help="do not read the raw recordings again")
    parser.add_argument("--quick", action="store_true", help="3 CV repeats and 20 permutations")
    parser.add_argument("--notebooks", action="store_true", help="re-execute the notebooks in place")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    start = time.time()
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"  ({time.time() - start:.0f} s)", flush=True)


def main() -> None:
    arguments = parse_arguments()
    parts = arguments.only or ["exact_force", "rhythm", "combined"]
    for part, is_build, script in STEPS:
        if part not in parts or (is_build and arguments.skip_build):
            continue
        command = [sys.executable, str(SCRIPTS / script)]
        if arguments.quick and script in CV_SCRIPTS:
            command += ["--cv-repeats", "3", "--permutations", "20"]
        run(command)
    if arguments.notebooks:
        for part in parts:
            for notebook in NOTEBOOKS[part]:
                path = ROOT / notebook
                # nbconvert runs every notebook in its own folder (figures/ are saved next to it)
                run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                     "--ExecutePreprocessor.timeout=1800", str(path)])
    print("\nDone.")


if __name__ == "__main__":
    main()
