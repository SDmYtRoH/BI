"""Run the unified RehaPiano pipeline (all games) end to end, or selected stages.

Run from the project root:
    python scripts/run_pipeline.py                         # all stages, reuses cached files and features
    python scripts/run_pipeline.py --stages analysis report
    python scripts/run_pipeline.py --recompute             # re-extract features of every recording
New data: copy the recordings anywhere under the measurements directory and the metadata CSV(s) into the
metadata directory (add a new patient CSV with --patient-files), then run again; only new files are processed.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402

from rehapiano.pipeline import aggregate, analysis, extract, index, participants, report, settings  # noqa: E402

STAGES = ["index", "participants", "features", "aggregate", "analysis", "report"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--measurements-dir", type=Path, default=settings.MEASUREMENTS_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=settings.METADATA_DIR)
    parser.add_argument("--patient-files", nargs="+", default=list(settings.PATIENT_METADATA_FILES))
    parser.add_argument("--questionnaire", type=Path, default=settings.QUESTIONNAIRE_CSV)
    parser.add_argument("--output-dir", type=Path, default=settings.OUTPUT_DIR)
    parser.add_argument("--recompute", action="store_true", help="re-extract features of every recording")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def summary(output_dir: Path) -> dict:
    recordings = pd.read_csv(output_dir / "recordings.csv")
    people = pd.read_csv(output_dir / "person_features.csv")
    counts = recordings.fillna({"reason": ""}).groupby(["status", "reason"]).size()
    return {
        "recordings": {f"{status}/{reason}".rstrip("/"): int(n) for (status, reason), n in counts.items()},
        "used_recordings_per_game": recordings.loc[recordings["status"] == "used", "game"].value_counts().to_dict(),
        "people_per_label": people["label"].value_counts().to_dict(),
        "extraction_errors": len(pd.read_csv(output_dir / "extraction_errors.csv")),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
    }


def main() -> None:
    args = parse_arguments()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    out = args.output_dir
    analysis_dir = out / "analysis"
    steps = {
        "index": lambda: index.run(out, args.measurements_dir, args.metadata_dir, args.patient_files),
        "participants": lambda: participants.run(out, args.questionnaire),
        "features": lambda: extract.run(out, args.recompute),
        "aggregate": lambda: aggregate.run(out),
        "analysis": lambda: analysis.run(out, analysis_dir),
        "report": lambda: report.run(out, analysis_dir),
    }
    for stage in STAGES:
        if stage in args.stages:
            start = time.time()
            logging.info("stage %s", stage)
            steps[stage]()
            logging.info("stage %s done in %.0f s", stage, time.time() - start)
    if (out / "person_features.csv").exists():
        info = summary(out)
        (out / "run_summary.json").write_text(json.dumps(info, indent=2, default=str))
        print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
