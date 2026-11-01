"""Week 5: hypotheses, age norms, classifier and ablation for the RHYTHM parameters.

Run from the repository root (after Maksym/scripts/build_rhythm.py):
    python Maksym/scripts/run_rhythm_week5.py
Writes CSV tables to ``Maksym/week5/outputs/`` (the cross-validation takes ~40 minutes).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repository root (shared rehapiano package)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Maksym" / "src"))

from maksym_rehapiano import config as rhythm_config, rhythm_parameters as rp  # noqa: E402

LABELS_CSV = rhythm_config.PARTICIPANT_LABELS_CSV


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rhythm-dir", type=Path, default=rhythm_config.WEEK5_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=rhythm_config.WEEK5_OUTPUT_DIR)
    parser.add_argument("--permutations", type=int, default=200)
    return parser.parse_args()


def classifier_table(people: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CV results for both models and several feature sets; also the out-of-fold probabilities."""

    feature_sets = {"parametre rytmu": rp.PARAMETERS, "parametre rytmu + vek": rp.PARAMETERS + ["age"],
                    "len vek": ["age"]}
    rows, probabilities = [], {}
    for model in ("logistická regresia", "náhodný les"):
        for name, features in feature_sets.items():
            result = rp.cross_validate(people, features, model)
            probabilities[(model, name)] = result.pop("oof_probability")
            rows.append({"set": label, "features": name, **result})
    older = people.loc[people["age"] >= 50]
    for model in ("logistická regresia", "náhodný les"):
        result = rp.cross_validate(older, rp.PARAMETERS, model)
        result.pop("oof_probability")
        rows.append({"set": f"{label}, len ≥ 50 rokov", "features": "parametre rytmu", **result})
    oof = people[["patient_id", "group", "age", "patient"]].copy()
    for (model, name), values in probabilities.items():
        oof[f"{model} | {name}"] = values
    return pd.DataFrame(rows), oof


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    arguments = parse_arguments()
    out: Path = arguments.output_dir
    out.mkdir(parents=True, exist_ok=True)
    by_attempt = pd.read_csv(arguments.rhythm_dir / "rhythm_parameters_by_attempt.csv")
    labels = pd.read_csv(LABELS_CSV)

    people = rp.person_parameters(by_attempt, labels)                       # attempts 2-4
    people_all_attempts = rp.person_parameters(by_attempt, labels, skip_first_attempt=False)
    main_set = people.loc[people["screened"]].reset_index(drop=True)        # 76 healthy + 48 patients
    healthy = main_set.loc[main_set["group"] == "control"]

    tables = {"people": people, "hypotheses": rp.hypothesis_tests(main_set),
              "hypotheses_all_90_healthy": rp.hypothesis_tests(people),
              "hypotheses_with_first_attempt": rp.hypothesis_tests(
                  people_all_attempts.loc[people_all_attempts["screened"]])}
    trends = rp.age_trends(healthy)
    tables["age_trends"] = trends
    tables["decade_norms"] = rp.decade_norms(healthy)
    tables["norms_by_age_model"] = rp.norms_by_age_model(healthy, trends)
    print("tests and norms done")

    classifier, oof = classifier_table(main_set, "76 zdravých + 48 pacientov")
    check_90, _ = classifier_table(people.reset_index(drop=True), "90 zdravých + 48 pacientov")
    check_first, _ = classifier_table(people_all_attempts.loc[people_all_attempts["screened"]].reset_index(drop=True),
                                      "s 1. pokusom")
    tables["classifier"] = pd.concat([classifier, check_90, check_first], ignore_index=True)
    tables["classifier_oof"] = oof
    print("classifier done")

    tables["ablation_parameters"] = rp.ablation(main_set, rp.PARAMETER_GROUPS)
    tables["ablation_with_age"] = rp.ablation(main_set, {**rp.PARAMETER_GROUPS, "vek": ["age"]})
    permutation = rp.permutation_auc(main_set, rp.PARAMETERS, n_permutations=arguments.permutations)
    tables["permutation_test"] = pd.DataFrame([permutation])
    print("ablation done")

    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
        print(f"{name + '.csv':<40} {len(table):>5} rows")
    print(f"\nTables written to {out.resolve()}")


if __name__ == "__main__":
    main()
