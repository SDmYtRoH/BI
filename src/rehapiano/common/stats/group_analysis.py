"""Healthy vs patients: hypotheses, age norms, classifier and ablation (any exercise).

Input is a person table (one row per person) with ``group`` ("control" / "patient"), ``patient``
(0 / 1), ``age`` and the numeric parameters. The same functions are used for EXACT_FORCE,
RHYTHM and for both exercises together, so every analysis follows one method:

1. hypothesis tests healthy vs patients (Mann-Whitney, Holm; people >= 50; model parameter ~ age + disease,
   optionally with a covariate such as the press rate);
2. age trends in healthy people and "normal" values by decade;
3. classifier healthy vs patients with repeated cross-validation;
4. ablation: how much each group of parameters (factor) contributes to the classification.

Written for the RHYTHM test by M. Kuznietsov (week 5), generalised to all exercises.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from rehapiano.common.config import (
    AGE_CATEGORY_EDGES,
    AGE_CATEGORY_LABELS,
    CV_FOLDS,
    CV_REPEATS,
    MIN_NORM_SIZE,
    NORM_QUANTILES,
    OLDER_AGE,
    RANDOM_SEED,
)
from rehapiano.common.stats.models import ols
from rehapiano.common.stats.tests import holm_adjust, mann_whitney

logger = logging.getLogger(__name__)

LOGISTIC = "logistická regresia"
FOREST = "náhodný les"


# ------------------------------------------------------------------ person table helpers
def add_decade(table: pd.DataFrame, column: str = "decade") -> pd.DataFrame:
    """Ordered 10-year category of ``age`` (10–19 ... 80–89)."""

    table = table.copy()
    table[column] = pd.cut(table["age"], AGE_CATEGORY_EDGES, right=False, labels=AGE_CATEGORY_LABELS)
    return table


def add_screening(table: pd.DataFrame, labels: Optional[pd.DataFrame]) -> pd.DataFrame:
    """``screened`` = healthy without a reported condition in the questionnaire, or a patient.

    ``labels`` is ``data/participant_labels.csv``. People missing from it are not screened (a new healthy
    person needs a questionnaire answer first); they are reported in the log so that the labels can be updated.
    """

    table = table.copy()
    if labels is None:
        table["screened"] = True
        return table
    eligible = labels.drop_duplicates("patient_id").set_index("patient_id")["eligible_comparison"]
    known = table["patient_id"].isin(eligible.index)
    table["screened"] = table["patient_id"].map(eligible).fillna(False).astype(bool)
    missing = table.loc[~known & (table["group"] == "control"), "patient_id"].tolist()
    if missing:
        logger.warning("%d healthy people are not in participant_labels.csv and are left out of the main "
                       "analysis: %s", len(missing), ", ".join(map(str, missing)))
    # patients are always part of the comparison, even when the labels file is older than the data
    table.loc[~known & (table["group"] == "patient"), "screened"] = True
    return table


# ------------------------------------------------------------------ 1. hypotheses
def hypothesis_tests(people: pd.DataFrame, parameters: Sequence[str], expected_direction: Mapping[str, int],
                     covariate: Optional[str] = None, older_age: float = OLDER_AGE) -> pd.DataFrame:
    """H0: same distribution in healthy and patients. Mann-Whitney on all people and on people >= ``older_age``,
    AUC (probability that a random patient has a higher value), and the disease term of
    parameter = b0 + b_age * age / 10 + b_disease * patient (HC3). With ``covariate`` the same model is fitted
    with the covariate added (e.g. the press rate: is the difference only a consequence of slower pressing?).
    Holm within each family of tests."""

    rows = []
    older = people["age"] >= older_age
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
        adjusted_effect = adjusted_p = np.nan
        if covariate is not None and parameter != covariate:
            adj = frame.dropna(subset=[covariate])
            Xc = np.column_stack([np.ones(len(adj)), adj["age"] / 10, adj["patient"], adj[covariate]])
            adjusted = ols(adj[parameter], Xc, ["b0", "age", "disease", "covariate"]).set_index("term")
            adjusted_effect, adjusted_p = adjusted.loc["disease", "coefficient"], adjusted.loc["disease", "p"]
        rows.append({
            "parameter": parameter, "expected_in_patients": "vyššie" if expected_direction[parameter] > 0 else "nižšie",
            "n_controls": test["n_a"], "n_patients": test["n_b"],
            "median_controls": test["median_a"], "median_patients": test["median_b"],
            "auc": (test["effect_r"] + 1) / 2, "effect_r": test["effect_r"], "p": test["p"],
            "median_controls_50": test_old["median_a"], "median_patients_50": test_old["median_b"],
            "n_controls_50": test_old["n_a"], "n_patients_50": test_old["n_b"], "p_50": test_old["p"],
            "disease_effect": model.loc["disease", "coefficient"],
            "disease_effect_sd": model.loc["disease", "coefficient"] / sd if sd > 0 else np.nan,
            "disease_p": model.loc["disease", "p"], "age_effect_per_10y": model.loc["age", "coefficient"],
            "covariate": covariate or "",
            "disease_effect_adj_sd": adjusted_effect / sd if sd > 0 else np.nan, "disease_adj_p": adjusted_p,
            "age_p": model.loc["age", "p"],
        })
    result = pd.DataFrame(rows)
    for column in ("p", "p_50", "disease_p", "disease_adj_p"):
        result[f"{column}_holm"] = holm_adjust(result[column])
    result["as_expected"] = np.sign(result["median_patients"] - result["median_controls"]) == \
        result["parameter"].map(expected_direction)
    return result


# ------------------------------------------------------------------ 2. age and norms
def age_trends(healthy: pd.DataFrame, parameters: Sequence[str]) -> pd.DataFrame:
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


def decade_norms(healthy: pd.DataFrame, parameters: Sequence[str], min_size: int = MIN_NORM_SIZE) -> pd.DataFrame:
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


def norms_by_age_model(healthy: pd.DataFrame, trends: pd.DataFrame, parameters: Sequence[str],
                       quantiles: Sequence[float] = NORM_QUANTILES) -> pd.DataFrame:
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


def outside_norms(patients: pd.DataFrame, norms: pd.DataFrame, parameters: Sequence[str]) -> pd.DataFrame:
    """Number of patients below / above the normal range (2.5-97.5 %) of their decade."""

    rows = []
    for parameter in parameters:
        m = norms.loc[norms["parameter"] == parameter].set_index("decade")
        pts = patients.dropna(subset=[parameter])
        low = pts["decade"].astype(str).map(m["q2.5"]).astype(float)
        high = pts["decade"].astype(str).map(m["q97.5"]).astype(float)
        rows.append({"parameter": parameter, "below": int((pts[parameter] < low).sum()),
                     "above": int((pts[parameter] > high).sum()), "n_patients": len(pts)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 3. classifier
def _models(seed: int = RANDOM_SEED):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # n_jobs does not change the forest: the tree seeds are drawn from random_state before fitting
    return {
        LOGISTIC: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
        FOREST: make_pipeline(SimpleImputer(strategy="median"),
                              RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                     class_weight="balanced", random_state=seed, n_jobs=-1)),
    }


def cross_validate(people: pd.DataFrame, features: Sequence[str], model_name: str = LOGISTIC,
                   n_splits: int = CV_FOLDS, n_repeats: int = CV_REPEATS, seed: int = RANDOM_SEED) -> Dict[str, object]:
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


def classifier_table(people: pd.DataFrame, label: str, feature_sets: Mapping[str, Sequence[str]],
                     older_feature_sets: Sequence[str] = (), models: Sequence[str] = (LOGISTIC, FOREST),
                     older_age: float = OLDER_AGE, **cv_options) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CV results for every model and feature set, plus the sets in ``older_feature_sets`` on people
    >= ``older_age`` only (where age hardly separates the groups). Also the out-of-fold probabilities."""

    rows, probabilities = [], {}
    for model in models:
        for name, features in feature_sets.items():
            result = cross_validate(people, features, model, **cv_options)
            probabilities[(model, name)] = result.pop("oof_probability")
            rows.append({"set": label, "features": name, **result})
    older = people.loc[people["age"] >= older_age]
    for model in models:
        for name in older_feature_sets:
            result = cross_validate(older, feature_sets[name], model, **cv_options)
            result.pop("oof_probability")
            rows.append({"set": f"{label}, len ≥ {older_age:g} rokov", "features": name, **result})
    oof = people[["patient_id", "group", "age", "patient"]].copy()
    for (model, name), values in probabilities.items():
        oof[f"{model} | {name}"] = values
    return pd.DataFrame(rows), oof


def logistic_coefficients(people: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Standardised coefficients of the logistic regression fitted on all people (direction and weight)."""

    model = _models()[LOGISTIC].fit(people[list(features)].to_numpy(dtype=float), people["patient"].to_numpy())
    coefficients = model[-1].coef_[0]
    return (pd.DataFrame({"feature": list(features), "coefficient": coefficients,
                          "odds_ratio_per_sd": np.exp(coefficients)})
            .sort_values("coefficient", key=np.abs, ascending=False).reset_index(drop=True))


# ------------------------------------------------------------------ 4. ablation
def ablation(people: pd.DataFrame, groups: Mapping[str, List[str]], model_name: str = LOGISTIC,
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
                    model_name: str = LOGISTIC, seed: int = RANDOM_SEED) -> Dict[str, float]:
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


# ------------------------------------------------------------------ all steps together
def analyse(main_set: pd.DataFrame, parameters: Sequence[str], expected_direction: Mapping[str, int],
            model_groups: Mapping[str, List[str]], feature_name: str, covariate: Optional[str] = None,
            checks: Optional[Mapping[str, tuple[str, pd.DataFrame]]] = None, n_repeats: int = CV_REPEATS,
            n_permutations: int = 200, log=print) -> Dict[str, pd.DataFrame]:
    """The whole teacher's task for one set of parameters; returns the tables by name.

    ``main_set``      person table of the main comparison (healthy without complaints + patients);
    ``parameters``    raw parameters for the tests and norms; ``model_groups`` factor -> classifier features
                      (e.g. log10 amplitudes) for the classifier and the ablation;
    ``feature_name``  Slovak name of the feature set in the classifier tables (e.g. "parametre rytmu");
    ``checks``        robustness checks ``{suffix: (label, person table)}``: hypotheses and classifier are
                      repeated on each of them (``hypotheses_<suffix>``, rows of ``classifier`` with main=False).
    """

    checks = checks or {}
    features = [f for group in model_groups.values() for f in group]
    healthy = main_set.loc[main_set["group"] == "control"]
    patients = main_set.loc[main_set["group"] == "patient"]
    label = f"{len(healthy)} zdravých + {len(patients)} pacientov"

    tables: Dict[str, pd.DataFrame] = {"hypotheses": hypothesis_tests(main_set, parameters, expected_direction, covariate)}
    for suffix, (_, frame) in checks.items():
        tables[f"hypotheses_{suffix}"] = hypothesis_tests(frame, parameters, expected_direction, covariate)
    trends = age_trends(healthy, parameters)
    tables["age_trends"] = trends
    tables["decade_norms"] = decade_norms(healthy, parameters)
    tables["norms_by_age_model"] = norms_by_age_model(healthy, trends, parameters)
    tables["outside_norms"] = outside_norms(patients, tables["norms_by_age_model"], parameters)
    log("tests and norms done")

    feature_sets = {feature_name: features, f"{feature_name} + vek": features + ["age"], "len vek": ["age"]}
    classifier, oof = classifier_table(main_set, label, feature_sets, older_feature_sets=[feature_name],
                                       n_repeats=n_repeats)
    frames = [classifier.assign(main=True)]
    for check_label, frame in checks.values():
        check, _ = classifier_table(frame.reset_index(drop=True), check_label, feature_sets,
                                    older_feature_sets=[feature_name], n_repeats=n_repeats)
        frames.append(check.assign(main=False))
    tables["classifier"] = pd.concat(frames, ignore_index=True)
    tables["classifier_oof"] = oof
    tables["classifier_coefficients"] = logistic_coefficients(main_set, features)
    log("classifier done")

    tables["ablation_parameters"] = ablation(main_set, model_groups, n_repeats=n_repeats)
    tables["ablation_with_age"] = ablation(main_set, {**model_groups, "vek": ["age"]}, n_repeats=n_repeats)
    tables["permutation_test"] = pd.DataFrame([permutation_auc(main_set, features, n_permutations=n_permutations)])
    log("ablation done")
    return tables


def write_tables(tables: Mapping[str, pd.DataFrame], out_dir) -> None:
    """Every table as ``<name>.csv`` in ``out_dir``."""

    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"{name + '.csv':<40} {len(table):>5} rows")
    print(f"\nTables written to {out_dir.resolve()}")
