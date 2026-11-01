"""Week 5: statistics of the RHYTHM parameters.

1. hypothesis tests healthy vs patients (Mann-Whitney, Holm; age-matched subset; model parameter ~ age + disease);
2. age trends in healthy people and "normal" values by decade;
3. classifier healthy vs patients with repeated cross-validation;
4. ablation: how much each group of parameters contributes to the classification.

Input is ``rhythm_parameters_by_attempt.csv`` (``scripts/build_rhythm.py``). The first attempt of every
recording is a practice attempt and is left out by default.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from maksym_rehapiano.config import AGE_CATEGORY_EDGES, AGE_CATEGORY_LABELS
from maksym_rehapiano.hand_balance import ols
from maksym_rehapiano.rhythm import RHYTHM_PARAMETER_COLUMNS
from rehapiano.config import RANDOM_SEED
from rehapiano.stats.tests import holm_adjust, mann_whitney

# Groups of parameters (factors) for the ablation; names are Slovak because they go straight to the notebook.
PARAMETER_GROUPS: Dict[str, List[str]] = {
    "tempo": ["press_rate_hz"],
    "pravidelnosť rytmu": ["interval_cv"],
    "sila stlačenia": ["amplitude_strong", "amplitude_weak"],
    "rovnováha rúk": ["balance"],
    "stálosť sily": ["amplitude_cv"],
    "súčasnosť rúk": ["sync_lag_ms"],
    "uvoľnenie": ["release_ratio"],
    "únava": ["fatigue_amplitude", "fatigue_rate"],
}
PARAMETERS = [p for group in PARAMETER_GROUPS.values() for p in group]
assert sorted(PARAMETERS) == sorted(RHYTHM_PARAMETER_COLUMNS)

# Expected direction in patients (H1, one-sided reading of the two-sided test): +1 higher, -1 lower.
EXPECTED_DIRECTION = {
    "press_rate_hz": -1, "interval_cv": +1, "amplitude_strong": -1, "amplitude_weak": -1, "balance": -1,
    "amplitude_cv": +1, "sync_lag_ms": +1, "release_ratio": +1, "fatigue_amplitude": -1, "fatigue_rate": -1,
}
MIN_NORM_SIZE = 20  # an empirical 2.5-97.5 % range needs at least this many healthy people in the decade


def person_parameters(by_attempt: pd.DataFrame, labels: pd.DataFrame | None = None,
                      skip_first_attempt: bool = True) -> pd.DataFrame:
    """One row per person: median over attempts of a recording, mean over the person's recordings.

    ``labels`` = week 2 ``participant_labels.csv``; adds ``screened`` (healthy without a reported condition
    in the questionnaire, or a patient).
    """

    valid = by_attempt.loc[by_attempt["status"] == "valid"]
    if skip_first_attempt:
        valid = valid.loc[valid["attempt"] > 1]
    per_recording = valid.groupby(["patient_id", "measurement_id"])[PARAMETERS].median()
    table = per_recording.groupby("patient_id").mean()
    info = by_attempt.groupby("patient_id").agg(group=("group", "first"), age=("age", "first"),
                                                gender=("gender", "first"),
                                                n_recordings=("measurement_id", "nunique"))
    info["n_attempts"] = valid.groupby("patient_id").size()
    table = info.join(table).reset_index()
    table["patient"] = (table["group"] == "patient").astype(int)
    table["decade"] = pd.cut(table["age"], AGE_CATEGORY_EDGES, right=False, labels=AGE_CATEGORY_LABELS)
    if labels is not None:
        eligible = labels.set_index("patient_id")["eligible_comparison"]
        table["screened"] = table["patient_id"].map(eligible).fillna(False).astype(bool)
    return table


# ------------------------------------------------------------------ 1. hypotheses
def hypothesis_tests(people: pd.DataFrame, parameters: Sequence[str] = PARAMETERS) -> pd.DataFrame:
    """H0: same distribution in healthy and patients. Mann-Whitney on all people and on people >= 50,
    AUC (probability that a random patient has a higher value), and the disease term of
    parameter = b0 + b_age * age / 10 + b_disease * patient (HC3). Holm within each family of tests."""

    rows = []
    older = people["age"] >= 50
    for parameter in parameters:
        frame = people.dropna(subset=[parameter])
        controls = frame.loc[frame["group"] == "control", parameter]
        patients = frame.loc[frame["group"] == "patient", parameter]
        test = mann_whitney(controls, patients)
        old = frame.loc[older.loc[frame.index]]
        test_old = mann_whitney(old.loc[old["group"] == "control", parameter], old.loc[old["group"] == "patient", parameter])
        X = np.column_stack([np.ones(len(frame)), frame["age"] / 10, frame["patient"]])
        model = ols(frame[parameter], X, ["b0", "age", "disease"]).set_index("term")
        sd = controls.std()
        # the same model with the press rate as a covariate: is the difference only a consequence of slower pressing?
        if parameter != "press_rate_hz":
            Xt = np.column_stack([X, frame["press_rate_hz"]])
            tempo_model = ols(frame[parameter], Xt, ["b0", "age", "disease", "tempo"]).set_index("term")
            tempo_effect, tempo_p = tempo_model.loc["disease", "coefficient"], tempo_model.loc["disease", "p"]
        else:
            tempo_effect = tempo_p = np.nan
        rows.append({
            "parameter": parameter, "expected_in_patients": "vyššie" if EXPECTED_DIRECTION[parameter] > 0 else "nižšie",
            "n_controls": test["n_a"], "n_patients": test["n_b"],
            "median_controls": test["median_a"], "median_patients": test["median_b"],
            "auc": (test["effect_r"] + 1) / 2, "effect_r": test["effect_r"], "p": test["p"],
            "median_controls_50": test_old["median_a"], "median_patients_50": test_old["median_b"],
            "n_controls_50": test_old["n_a"], "n_patients_50": test_old["n_b"], "p_50": test_old["p"],
            "disease_effect": model.loc["disease", "coefficient"],
            "disease_effect_sd": model.loc["disease", "coefficient"] / sd if sd > 0 else np.nan,
            "disease_p": model.loc["disease", "p"], "age_effect_per_10y": model.loc["age", "coefficient"],
            "disease_effect_tempo_sd": tempo_effect / sd if sd > 0 else np.nan, "disease_tempo_p": tempo_p,
            "age_p": model.loc["age", "p"],
        })
    result = pd.DataFrame(rows)
    for column in ("p", "p_50", "disease_p", "disease_tempo_p"):
        result[f"{column}_holm"] = holm_adjust(result[column])
    result["as_expected"] = np.sign(result["median_patients"] - result["median_controls"]) == \
        result["parameter"].map(EXPECTED_DIRECTION)
    return result


# ------------------------------------------------------------------ 2. age and norms
def age_trends(healthy: pd.DataFrame, parameters: Sequence[str] = PARAMETERS) -> pd.DataFrame:
    """Healthy only: Spearman correlation with age and the linear change per 10 years (HC3), Holm."""

    rows = []
    for parameter in parameters:
        frame = healthy.dropna(subset=[parameter])
        rho, p = stats.spearmanr(frame["age"], frame[parameter])
        model = ols(frame[parameter], np.column_stack([np.ones(len(frame)), frame["age"] / 10]), ["b0", "age"])
        slope = model.set_index("term").loc["age"]
        rows.append({"parameter": parameter, "n": len(frame), "spearman_rho": rho, "spearman_p": p,
                     "change_per_10y": slope["coefficient"], "change_ci_low": slope["ci_low"],
                     "change_ci_high": slope["ci_high"], "median": frame[parameter].median()})
    result = pd.DataFrame(rows)
    result["spearman_p_holm"] = holm_adjust(result["spearman_p"])
    result["age_dependent"] = result["spearman_p_holm"] < 0.05
    return result


def decade_norms(healthy: pd.DataFrame, parameters: Sequence[str] = PARAMETERS,
                 min_size: int = MIN_NORM_SIZE) -> pd.DataFrame:
    """Healthy people by decade: n, median, 2.5 and 97.5 % quantiles (only for n >= ``min_size``)
    and min - max. ``reliable`` marks decades with enough people for an empirical range."""

    rows = []
    for parameter in parameters:
        for decade, frame in healthy.groupby("decade", observed=False):
            values = frame[parameter].dropna()
            reliable = len(values) >= min_size
            rows.append({"parameter": parameter, "decade": decade, "n": len(values),
                         "median": values.median() if len(values) else np.nan,
                         "p2_5": values.quantile(0.025) if reliable else np.nan,
                         "p97_5": values.quantile(0.975) if reliable else np.nan,
                         "min": values.min() if len(values) else np.nan,
                         "max": values.max() if len(values) else np.nan, "reliable": reliable})
    return pd.DataFrame(rows)


def norms_by_age_model(healthy: pd.DataFrame, trends: pd.DataFrame, parameters: Sequence[str] = PARAMETERS,
                       quantiles: Sequence[float] = (0.025, 0.5, 0.975)) -> pd.DataFrame:
    """Normal range for every decade from all healthy people at once.

    Parameters without an age trend (``trends.age_dependent`` False): the same pooled quantiles for every decade.
    Age-dependent parameters: linear quantile regression on age, evaluated at the middle of each decade.
    """

    from sklearn.linear_model import QuantileRegressor

    centres = np.array([low + 5 for low in AGE_CATEGORY_EDGES[:-1]], dtype=float)
    dependent = trends.set_index("parameter")["age_dependent"]
    rows = []
    for parameter in parameters:
        frame = healthy.dropna(subset=[parameter])
        if dependent.get(parameter, False):
            values = {}
            for q in quantiles:
                model = QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(frame[["age"]], frame[parameter])
                values[q] = model.predict(pd.DataFrame({"age": centres}))
            method = "kvantilová regresia na vek"
        else:
            pooled = frame[parameter].quantile(list(quantiles))
            values = {q: np.full(len(centres), pooled[q]) for q in quantiles}
            method = "spoločná norma (bez vplyvu veku)"
        for i, decade in enumerate(AGE_CATEGORY_LABELS):
            rows.append({"parameter": parameter, "decade": decade, "method": method,
                         **{f"q{q * 100:g}": values[q][i] for q in quantiles}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 3. classifier
def _models(seed: int = RANDOM_SEED):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logistická regresia": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
        "náhodný les": make_pipeline(SimpleImputer(strategy="median"),
                                     RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                            class_weight="balanced", random_state=seed, n_jobs=1)),
    }


def cross_validate(people: pd.DataFrame, features: Sequence[str], model_name: str = "logistická regresia",
                   n_splits: int = 5, n_repeats: int = 20, seed: int = RANDOM_SEED) -> Dict[str, object]:
    """Repeated stratified k-fold CV. Every repeat gives out-of-fold probabilities for all people; AUC,
    sensitivity, specificity and balanced accuracy (threshold 0.5) are computed per repeat.

    Returns the mean and the 2.5-97.5 % range over repeats and the mean out-of-fold probability per person.
    """

    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X = people[list(features)].to_numpy(dtype=float)
    y = people["patient"].to_numpy()
    template = _models(seed)[model_name]
    scores, probabilities = [], np.zeros((n_repeats, len(y)))
    for repeat in range(n_repeats):
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for train, test in folds.split(X, y):
            model = clone(template).fit(X[train], y[train])
            probabilities[repeat, test] = model.predict_proba(X[test])[:, 1]
        predicted = probabilities[repeat] >= 0.5
        sensitivity = predicted[y == 1].mean()
        specificity = (~predicted[y == 0]).mean()
        scores.append({"auc": roc_auc_score(y, probabilities[repeat]), "sensitivity": sensitivity,
                       "specificity": specificity, "balanced_accuracy": (sensitivity + specificity) / 2})
    scores = pd.DataFrame(scores)
    summary: Dict[str, object] = {"model": model_name, "n_people": len(y), "n_patients": int(y.sum()),
                                  "n_features": len(features)}
    for metric in scores.columns:
        summary[metric] = scores[metric].mean()
        summary[f"{metric}_low"], summary[f"{metric}_high"] = scores[metric].quantile([0.025, 0.975])
    summary["oof_probability"] = probabilities.mean(axis=0)
    return summary


def ablation(people: pd.DataFrame, groups: Dict[str, List[str]], model_name: str = "logistická regresia",
             **cv_options) -> pd.DataFrame:
    """For every group of features: AUC without it (drop-one) and AUC with it alone (only-one)."""

    all_features = [f for features in groups.values() for f in features]
    full = cross_validate(people, all_features, model_name, **cv_options)
    rows = []
    for name, features in groups.items():
        rest = [f for f in all_features if f not in features]
        without = cross_validate(people, rest, model_name, **cv_options)
        alone = cross_validate(people, features, model_name, **cv_options)
        rows.append({"group": name, "features": ", ".join(features), "auc_all": full["auc"],
                     "auc_without": without["auc"], "auc_drop": full["auc"] - without["auc"],
                     "auc_alone": alone["auc"], "auc_alone_low": alone["auc_low"],
                     "auc_alone_high": alone["auc_high"]})
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True)


def permutation_auc(people: pd.DataFrame, features: Sequence[str], n_permutations: int = 200,
                    model_name: str = "logistická regresia", seed: int = RANDOM_SEED) -> Dict[str, float]:
    """Is the cross-validated AUC better than chance? AUC with labels shuffled (5-fold CV, 1 repeat each)."""

    observed = cross_validate(people, features, model_name, n_repeats=5, seed=seed)["auc"]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_permutations):
        shuffled = people.assign(patient=rng.permutation(people["patient"].to_numpy()))
        null.append(cross_validate(shuffled, features, model_name, n_repeats=1, seed=int(rng.integers(1e6)))["auc"])
    null = np.array(null)
    return {"auc": observed, "null_mean": null.mean(), "null_p97_5": float(np.quantile(null, 0.975)),
            "p": float((np.sum(null >= observed) + 1) / (len(null) + 1))}
