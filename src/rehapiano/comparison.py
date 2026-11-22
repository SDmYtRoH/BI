"""EXACT_FORCE vs RHYTHM: do the two tests lead to the same conclusions? (teacher's week 6 task, M. Kuznietsov)

Input are the person tables of both exercises (``outputs/*/analysis/people.csv``), their classifier
out-of-fold probabilities and the recording-level tables. Four questions:

1. group differences with one method for both tests: patients vs healthy people of the same decade
   (van Elteren test, Holm within each test);
2. do the tests measure the same thing? Spearman correlation of every EXACT_FORCE parameter with every
   RHYTHM parameter, separately in healthy people and in patients;
3. do the classifiers of the two tests flag the same patients? (out-of-fold probabilities);
4. repeatability: patients measured twice (3-14 days apart) - which test gives more stable values?
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from rehapiano.common.stats.tests import holm_adjust, icc_agreement, van_elteren, wilcoxon_paired
from rehapiano.exact_force import parameters as ep
from rehapiano.rhythm import parameters as rp
from rehapiano.rhythm.config import PRACTICE_ATTEMPTS

EF, RH = "EXACT_FORCE", "RYTMUS"
TESTS = {EF: (ep.PARAMETERS, ep.EXPECTED_DIRECTION), RH: (rp.PARAMETERS, rp.EXPECTED_DIRECTION)}
# Probability columns of the logistic regression on the parameters of each test (``classifier_oof.csv``).
OOF_COLUMNS = {EF: "logistická regresia | parametre EF", RH: "logistická regresia | parametre rytmu"}


# ------------------------------------------------------------------ 1. group differences, same method
def same_age_tests(people: pd.DataFrame, parameters: Sequence[str],
                   expected_direction: Mapping[str, int]) -> pd.DataFrame:
    """Patients vs healthy people of the same decade for every parameter (van Elteren, Holm).

    ``auc`` (all people, no age matching) is added so that the size of the difference can be compared
    between parameters of both tests: 0.5 = no difference, distance from 0.5 = how well it separates.
    """

    rows = []
    for parameter in parameters:
        frame = people.dropna(subset=[parameter])
        test = van_elteren(frame[parameter], frame["patient"], frame["decade"])
        controls = frame.loc[frame["patient"] == 0, parameter]
        patients = frame.loc[frame["patient"] == 1, parameter]
        u = stats.mannwhitneyu(patients, controls).statistic
        rows.append({"parameter": parameter, "n_controls": len(controls), "n_patients": len(patients),
                     "median_controls": controls.median(), "median_patients": patients.median(),
                     "auc": u / (len(controls) * len(patients)), "z_same_age": test["z"], "p_same_age": test["p"]})
    result = pd.DataFrame(rows)
    result["p_same_age_holm"] = holm_adjust(result["p_same_age"])
    direction = result["parameter"].map(expected_direction)
    result["as_expected"] = np.sign(result["z_same_age"]) == direction
    result["separation"] = (result["auc"] - 0.5).abs() + 0.5  # 0.5-1, direction removed
    return result


# ------------------------------------------------------------------ 2. do the tests measure the same thing?
def cross_correlation(people: pd.DataFrame, ef_parameters: Sequence[str] = ep.PARAMETERS,
                      rh_parameters: Sequence[str] = rp.PARAMETERS) -> pd.DataFrame:
    """Long table: Spearman rho and p of every EXACT_FORCE x RHYTHM parameter pair, within each group.

    ``people`` has the parameters of both tests in one row per person (see ``joined_people``).
    """

    rows = []
    for group, frame in people.groupby("group"):
        for a in ef_parameters:
            for b in rh_parameters:
                pair = frame[[a, b]].dropna()
                rho, p = stats.spearmanr(pair[a], pair[b])
                rows.append({"group": group, "exact_force": a, "rhythm": b, "n": len(pair), "rho": rho, "p": p})
    table = pd.DataFrame(rows)
    table["p_holm"] = table.groupby("group")["p"].transform(lambda p: holm_adjust(p.to_numpy()))
    return table


def joined_people(exact_force: pd.DataFrame, rhythm: pd.DataFrame) -> pd.DataFrame:
    """One row per screened person with the raw parameters of both tests (people with both tests only)."""

    info = ["patient_id", "group", "patient", "age", "decade"]
    ef = exact_force.loc[exact_force["screened"], info + ep.PARAMETERS]
    return ef.merge(rhythm[["patient_id", *rp.PARAMETERS]], on="patient_id", how="inner").reset_index(drop=True)


# ------------------------------------------------------------------ 3. same patients flagged?
def classifier_agreement(ef_oof: pd.DataFrame, rh_oof: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """One row per person: out-of-fold patient probability from each test and who flags the person.

    ``flagged`` = "oba" / "len EXACT_FORCE" / "len RYTMUS" / "žiadny" (probability >= ``threshold``).
    ``mean_probability`` = average of both probabilities (the simplest combination of the two tests).
    """

    table = (ef_oof[["patient_id", "group", "age", "patient", OOF_COLUMNS[EF]]]
             .merge(rh_oof[["patient_id", OOF_COLUMNS[RH]]], on="patient_id", how="inner")
             .rename(columns={OOF_COLUMNS[EF]: "p_exact_force", OOF_COLUMNS[RH]: "p_rhythm"}))
    ef_flag, rh_flag = table["p_exact_force"] >= threshold, table["p_rhythm"] >= threshold
    table["flagged"] = np.select([ef_flag & rh_flag, ef_flag, rh_flag], ["oba", f"len {EF}", f"len {RH}"], "žiadny")
    table["mean_probability"] = (table["p_exact_force"] + table["p_rhythm"]) / 2
    return table


def agreement_summary(agreement: pd.DataFrame) -> pd.DataFrame:
    """Per group: correlation of the two probabilities and the AUC of each test and of their average."""

    from sklearn.metrics import roc_auc_score

    rows = []
    for name, frame in [("všetci", agreement), *agreement.groupby("group")]:
        rho, p = stats.spearmanr(frame["p_exact_force"], frame["p_rhythm"])
        row = {"group": name, "n": len(frame), "rho": rho, "p": p}
        if name == "všetci":  # AUC needs both groups
            row.update({f"auc_{column}": roc_auc_score(frame["patient"], frame[column])
                        for column in ["p_exact_force", "p_rhythm", "mean_probability"]})
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 4. repeatability
def exact_force_by_recording(features_by_measurement: pd.DataFrame) -> pd.DataFrame:
    """EXACT_FORCE parameters of every recording (not averaged over a person's recordings).

    Every recording is treated as a separate "person" and goes through ``exact_force.parameters`` - the
    same code as the person table, so a person with one recording gets exactly the person values.
    """

    used = features_by_measurement.loc[~features_by_measurement["measurement_outlier"].astype(bool)
                                       & (features_by_measurement["n_used_attempts"] > 0)].copy()
    used["person_id"] = used["patient_id"]
    used["patient_id"] = used["patient_id"] + "|" + used["measurement_id"]
    keys = ["patient_id", "step_id"]
    summary = used.groupby(keys).agg(group=("group", "first"), age=("age", "first"), age_group=("age_group", "first"),
                                     gender=("gender", "first"), n_attempts=("n_used_attempts", "max"),
                                     timestamp=("timestamp", "first")).reset_index()
    summary["n_measurements"] = 1
    by_hand = used.groupby([*keys, "hand"])
    force, inactive = by_hand["mean_force"].mean().unstack("hand"), by_hand["finger_inactive"].sum().unstack("hand")
    index = pd.MultiIndex.from_frame(summary[keys])
    for hand in ("left", "right"):
        summary[f"{hand}_mean_force"] = force[hand].reindex(index).to_numpy()
        summary[f"{hand}_n_inactive_fingers"] = inactive[hand].reindex(index).to_numpy()
    table = ep.person_parameters(used, summary)
    table = table.merge(summary.loc[summary["step_id"].isin(used["step_id"]), ["patient_id", "timestamp"]]
                        .drop_duplicates("patient_id"), on="patient_id", how="left")
    table[["person_id", "measurement_id"]] = table["patient_id"].str.split("|", n=1, expand=True)
    return table


def rhythm_by_recording(by_attempt: pd.DataFrame, skip_first_attempt: bool = True) -> pd.DataFrame:
    """RHYTHM parameters of every recording: median of the valid attempts (without the practice attempt)."""

    valid = by_attempt.loc[by_attempt["status"] == "valid"]
    if skip_first_attempt:
        valid = valid.loc[~valid["attempt"].isin(PRACTICE_ATTEMPTS)]
    grouped = valid.groupby(["patient_id", "measurement_id"])
    table = grouped[rp.PARAMETERS].median()
    table["group"], table["timestamp"] = grouped["group"].first(), grouped["timestamp"].first()
    return table.reset_index().rename(columns={"patient_id": "person_id"})


def repeatability(by_recording: pd.DataFrame, parameters: Sequence[str], log_parameters: Sequence[str] = (),
                  people: Sequence[str] | None = None) -> pd.DataFrame:
    """People with exactly two recordings: agreement of the 1st and the 2nd visit for every parameter.

    ``icc`` = ICC(2,1) absolute agreement (log10 for ``log_parameters``, whose values span decades);
    ``rho`` = Spearman correlation (rank-based, not sensitive to a few extreme values);
    ``median_change`` and ``p_change`` = 2nd - 1st visit, Wilcoxon (is there a learning / therapy effect?).
    """

    table = by_recording if people is None else by_recording.loc[by_recording["person_id"].isin(people)]
    table = table.sort_values(["person_id", "timestamp"])
    table = table.loc[table.groupby("person_id")["measurement_id"].transform("size") == 2].copy()
    table["visit"] = table.groupby("person_id").cumcount() + 1
    wide = table.pivot(index="person_id", columns="visit", values=list(parameters))
    rows = []
    for parameter in parameters:
        first, second = wide[(parameter, 1)].to_numpy(float), wide[(parameter, 2)].to_numpy(float)
        a, b = (np.log10(first), np.log10(second)) if parameter in log_parameters else (first, second)
        keep = np.isfinite(a) & np.isfinite(b)
        change = wilcoxon_paired(first, second)
        rows.append({"parameter": parameter, "n_people": int(keep.sum()), "icc": icc_agreement(a[keep], b[keep]),
                     "rho": stats.spearmanr(a[keep], b[keep])[0],
                     "median_visit1": np.nanmedian(first), "median_visit2": np.nanmedian(second),
                     "median_change": change["median_difference"], "p_change": change["p"]})
    result = pd.DataFrame(rows)
    result["p_change_holm"] = holm_adjust(result["p_change"])
    return result


def visit_gap_days(by_recording: pd.DataFrame, people: Sequence[str]) -> pd.Series:
    """Days between the 1st and the 2nd recording of people with two recordings."""

    table = by_recording.loc[by_recording["person_id"].isin(people)].copy()
    table["time"] = pd.to_datetime(table["timestamp"], format="mixed")
    span = table.groupby("person_id")["time"].agg(["min", "max", "size"])
    span = span.loc[span["size"] == 2]
    return (span["max"] - span["min"]).dt.total_seconds() / 86400


def run(ef_people: pd.DataFrame, rh_people: pd.DataFrame, ef_oof: pd.DataFrame, rh_oof: pd.DataFrame,
        ef_features_by_measurement: pd.DataFrame, rh_by_attempt: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """All comparison tables by name."""

    tables: Dict[str, pd.DataFrame] = {}
    frames = []
    for test, people in [(EF, ef_people), (RH, rh_people)]:
        parameters, direction = TESTS[test]
        frames.append(same_age_tests(people.loc[people["screened"]], parameters, direction).assign(test=test))
    tables["group_differences"] = pd.concat(frames, ignore_index=True)

    joined = joined_people(ef_people, rh_people)
    tables["cross_correlation"] = cross_correlation(joined)

    agreement = classifier_agreement(ef_oof, rh_oof)
    tables["classifier_agreement"] = agreement
    tables["classifier_agreement_summary"] = agreement_summary(agreement)

    patients = ef_people.loc[ef_people["group"] == "patient", "patient_id"]
    ef_recordings = exact_force_by_recording(ef_features_by_measurement)
    rh_recordings = rhythm_by_recording(rh_by_attempt)
    tables["repeatability"] = pd.concat([
        repeatability(ef_recordings, ep.PARAMETERS, ep.LOG_PARAMETERS, patients).assign(test=EF),
        repeatability(rh_recordings, rp.PARAMETERS, people=patients).assign(test=RH)], ignore_index=True)
    gaps = pd.concat([visit_gap_days(ef_recordings, patients).rename(EF),
                      visit_gap_days(rh_recordings, patients).rename(RH)], axis=1)
    tables["visit_gaps"] = gaps.rename_axis("patient_id").reset_index()
    return tables
