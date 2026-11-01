"""Stage 5 (task of week 6): hypotheses, age trends and decade norms of healthy people, classifier, ablation.

Groups: ``Healthy`` (questionnaire-screened) vs ``Sick``; ``Other control`` is excluded. One person = one unit.

Hypotheses, for every feature of ``catalog.FEATURES``:
    H0: healthy and sick people have the same distribution of the feature;
    H1: the distributions differ (two-sided).
Tests: Mann-Whitney U (all ages; people aged 50+) and a decade-stratified permutation test (labels shuffled only
within 10-year groups, so age cannot create the difference). Holm correction over the features of each test.

Norms: healthy people only. An age trend is tested with Spearman's rho (Holm). Decade ranges (2.5-97.5 %) are
reported only for decades with >= ``MIN_PEOPLE_FOR_DECADE_RANGE`` people; otherwise the pooled healthy range is the
norm for features without an age trend, and a linear quantile regression on age for features with one.

Classifier: logistic regression (median imputation, standardisation, balanced classes) and random forest,
repeated stratified 5-fold cross-validation; people never appear in both training and test folds.
Ablation: logistic regression without each factor and with each factor alone.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rehapiano.pipeline import catalog, settings
from rehapiano.pipeline.participants import HEALTHY, SICK
from rehapiano.stats.tests import holm_adjust

logger = logging.getLogger(__name__)

FEATURES = list(catalog.FEATURES)


def _decade_start(decade: pd.Series) -> pd.Series:
    return decade.str.split("–").str[0].astype(int)


# --------------------------------------------------------------------------- 1. hypotheses
def _auc(healthy: np.ndarray, sick: np.ndarray) -> float:
    """P(sick > healthy) + 0.5 P(tie); 0.5 = no difference."""

    return float(stats.mannwhitneyu(sick, healthy).statistic / (len(sick) * len(healthy)))


def _stratified_effect(values: np.ndarray, sick: np.ndarray, strata: np.ndarray, rng: np.random.Generator):
    """Weighted within-decade difference of means (sick - healthy) and its permutation p-value."""

    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    groups = [g for g in groups if 0 < sick[g].sum() < len(g)]
    weights = np.array([sick[g].sum() * (~sick[g]).sum() / len(g) for g in groups])

    def effect(labels):
        return float(np.sum(weights * [values[g][labels[g]].mean() - values[g][~labels[g]].mean() for g in groups])
                     / weights.sum())

    observed = effect(sick)
    null = np.empty(settings.PERMUTATIONS)
    for i in range(settings.PERMUTATIONS):
        shuffled = sick.copy()
        for g in groups:
            shuffled[g] = rng.permutation(sick[g])
        null[i] = effect(shuffled)
    return observed, float((np.sum(np.abs(null) >= abs(observed)) + 1) / (settings.PERMUTATIONS + 1))


def hypotheses(people: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(settings.RANDOM_SEED)
    data = people.loc[people["label"].isin([HEALTHY, SICK])]
    rows = []
    for feature in FEATURES:
        frame = data.dropna(subset=[feature])
        h = frame.loc[frame["label"] == HEALTHY, feature].to_numpy()
        s = frame.loc[frame["label"] == SICK, feature].to_numpy()
        older = frame.loc[frame["age"] >= settings.OLDER_AGE]
        h50 = older.loc[older["label"] == HEALTHY, feature].to_numpy()
        s50 = older.loc[older["label"] == SICK, feature].to_numpy()
        sd = h.std(ddof=1)
        effect, p_strat = _stratified_effect(frame[feature].to_numpy() / sd, frame["label"].eq(SICK).to_numpy(),
                                             frame["decade"].to_numpy(), rng)
        rows.append({"feature": feature, "factor": catalog.FEATURES[feature].factor,
                     "game": catalog.GAME_OF[feature], "description": catalog.FEATURES[feature].description,
                     "n_healthy": len(h), "n_sick": len(s), "median_healthy": np.median(h), "median_sick": np.median(s),
                     "auc": _auc(h, s), "p": stats.mannwhitneyu(h, s).pvalue,
                     "n_healthy_50": len(h50), "n_sick_50": len(s50), "median_healthy_50": np.median(h50),
                     "median_sick_50": np.median(s50), "auc_50": _auc(h50, s50),
                     "p_50": stats.mannwhitneyu(h50, s50).pvalue,
                     "effect_same_decade_sd": effect, "p_same_decade": p_strat})
    table = pd.DataFrame(rows)
    for column in ["p", "p_50", "p_same_decade"]:
        table[f"{column}_holm"] = holm_adjust(table[column])
    table["reject_h0"] = table["p_holm"] < 0.05
    table["reject_h0_same_decade"] = table["p_same_decade_holm"] < 0.05
    return table


# --------------------------------------------------------------------------- 2. healthy people: age and norms
def age_trends(people: pd.DataFrame) -> pd.DataFrame:
    healthy = people.loc[people["label"] == HEALTHY]
    rows = []
    for feature in FEATURES:
        frame = healthy.dropna(subset=[feature])
        rho, p = stats.spearmanr(frame["age"], frame[feature])
        fit = stats.linregress(frame["age"] / 10, frame[feature])
        t = stats.t.ppf(0.975, len(frame) - 2)
        rows.append({"feature": feature, "n": len(frame), "spearman_rho": rho, "p": p,
                     "change_per_10_years": fit.slope, "ci_low": fit.slope - t * fit.stderr,
                     "ci_high": fit.slope + t * fit.stderr})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table["p"])
    table["age_dependent"] = table["p_holm"] < 0.05
    return table


def decade_norms(people: pd.DataFrame, trends: pd.DataFrame) -> pd.DataFrame:
    """Normal value (median) and range (2.5-97.5 %) of healthy people for every decade present in the data."""

    low_q, high_q = settings.REFERENCE_QUANTILES
    healthy = people.loc[people["label"] == HEALTHY]
    decades = sorted(people["decade"].dropna().unique(), key=lambda d: int(d.split("–")[0]))
    dependent = trends.set_index("feature")["age_dependent"]
    rows = []
    for feature in FEATURES:
        frame = healthy.dropna(subset=[feature])
        if dependent[feature]:
            x = frame[["age"]].to_numpy()
            models = {q: QuantileRegressor(quantile=q, alpha=0, solver="highs").fit(x, frame[feature])
                      for q in (low_q, 0.5, high_q)}
        for decade in decades:
            part = frame.loc[frame["decade"] == decade, feature]
            row = {"feature": feature, "decade": decade, "n_healthy": len(part),
                   "empirical_median": part.median() if len(part) else np.nan,
                   "empirical_low": part.quantile(low_q) if len(part) >= settings.MIN_PEOPLE_FOR_DECADE_RANGE else np.nan,
                   "empirical_high": part.quantile(high_q) if len(part) >= settings.MIN_PEOPLE_FOR_DECADE_RANGE
                   else np.nan}
            if dependent[feature]:
                middle = np.array([[int(decade.split("–")[0]) + settings.DECADE_YEARS / 2]])
                row.update(method="age regression (quantile)", norm_median=models[0.5].predict(middle)[0],
                           norm_low=models[low_q].predict(middle)[0], norm_high=models[high_q].predict(middle)[0])
            else:
                row.update(method="pooled healthy (no age trend)", norm_median=frame[feature].median(),
                           norm_low=frame[feature].quantile(low_q), norm_high=frame[feature].quantile(high_q))
            rows.append(row)
    return pd.DataFrame(rows)


def outside_norm(people: pd.DataFrame, norms: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = norms.set_index(["feature", "decade"])
    for label in [HEALTHY, SICK]:
        group = people.loc[people["label"] == label]
        for feature in FEATURES:
            frame = group.dropna(subset=[feature, "decade"])
            bounds = indexed.loc[[(feature, d) for d in frame["decade"]], ["norm_low", "norm_high"]].to_numpy()
            rows.append({"feature": feature, "label": label, "people": len(frame),
                         "below_range": int((frame[feature].to_numpy() < bounds[:, 0]).sum()),
                         "above_range": int((frame[feature].to_numpy() > bounds[:, 1]).sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 3. classifier and ablation
def design_matrix(people: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Feature columns; skewed amplitudes are log-transformed (fixed, label-free transform)."""

    x = people[features].astype(float).copy()
    for feature in features:
        if feature in catalog.FEATURES and catalog.FEATURES[feature].log:
            positive = x[feature][x[feature] > 0]
            x[feature] = np.log(x[feature] + 0.5 * positive.min())
    return x


MODELS = {
    "logistic regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                         LogisticRegression(class_weight="balanced", max_iter=5000)),
    "random forest": make_pipeline(SimpleImputer(strategy="median"),
                                   RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                                          random_state=settings.RANDOM_SEED, n_jobs=-1)),
}


def cross_validate(model, x: pd.DataFrame, y: np.ndarray, repeats: int = settings.CV_REPEATS) -> dict:
    metrics, probabilities = [], []
    for repeat in range(repeats):
        folds = StratifiedKFold(settings.CV_FOLDS, shuffle=True, random_state=settings.RANDOM_SEED + repeat)
        p = cross_val_predict(clone(model), x, y, cv=folds, method="predict_proba")[:, 1]
        predicted = p >= 0.5
        metrics.append({"auc": roc_auc_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, predicted),
                        "sensitivity": predicted[y == 1].mean(), "specificity": (~predicted[y == 0]).mean()})
        probabilities.append(p)
    metrics = pd.DataFrame(metrics)
    summary = {f"{m}": metrics[m].mean() for m in metrics}
    summary.update({f"{m}_low": metrics[m].quantile(0.025) for m in metrics})
    summary.update({f"{m}_high": metrics[m].quantile(0.975) for m in metrics})
    return {"summary": summary, "oof": np.mean(probabilities, axis=0)}


def _dataset(people: pd.DataFrame, min_age: float = None):
    data = people.loc[people["label"].isin([HEALTHY, SICK])]
    if min_age is not None:
        data = data.loc[data["age"] >= min_age]
    return data.reset_index(drop=True), data["label"].eq(SICK).astype(int).to_numpy()


def classifier(people: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    ef = [f for f in FEATURES if catalog.GAME_OF[f] == "EXACT_FORCE"]
    rh = [f for f in FEATURES if catalog.GAME_OF[f] == "RHYTHM"]
    sets = {"EXACT_FORCE features": ef, "Rhythm features": rh, "All game features": FEATURES,
            "All game features + age": FEATURES + ["age"], "Age only": ["age"]}
    rows, oof = [], {}
    for subset, min_age in [("all ages", None), (f"age {settings.OLDER_AGE}+", settings.OLDER_AGE)]:
        data, y = _dataset(people, min_age)
        oof.setdefault(subset, data[["patient_id", "label", "age"]].copy())
        for set_name, features in sets.items():
            for model_name, model in MODELS.items():
                if model_name == "random forest" and (set_name != "All game features" or min_age is not None):
                    continue
                result = cross_validate(model, design_matrix(data, features), y)
                rows.append({"people": subset, "n_healthy": int((y == 0).sum()), "n_sick": int((y == 1).sum()),
                             "features": set_name, "model": model_name, **result["summary"]})
                oof[subset][f"{model_name} | {set_name}"] = result["oof"]
    data, y = _dataset(people)
    model = clone(MODELS["logistic regression"]).fit(design_matrix(data, FEATURES), y)
    coefficients = pd.DataFrame({"feature": FEATURES, "factor": [catalog.FEATURES[f].factor for f in FEATURES],
                                 "standardised_coefficient": model[-1].coef_[0]})
    return {"classifier": pd.DataFrame(rows), "classifier_coefficients": coefficients,
            **{f"classifier_oof_{k.replace(' ', '_').replace('+', 'plus')}": v for k, v in oof.items()}}


def permutation_test(people: pd.DataFrame) -> pd.DataFrame:
    data, y = _dataset(people)
    x = design_matrix(data, FEATURES)
    model = MODELS["logistic regression"]
    observed = cross_validate(model, x, y, repeats=1)["summary"]["auc"]
    rng = np.random.default_rng(settings.RANDOM_SEED)
    null = [cross_validate(model, x, rng.permutation(y), repeats=1)["summary"]["auc"]
            for _ in range(settings.CLASSIFIER_PERMUTATIONS)]
    return pd.DataFrame([{"features": "All game features", "model": "logistic regression", "auc": observed,
                          "null_mean": np.mean(null), "null_p97_5": np.quantile(null, 0.975),
                          "p": (np.sum(np.array(null) >= observed) + 1) / (len(null) + 1)}])


def ablation(people: pd.DataFrame) -> pd.DataFrame:
    data, y = _dataset(people)
    model = MODELS["logistic regression"]
    rows = []
    for with_age in (False, True):
        groups = dict(catalog.factors())
        if with_age:
            groups["Age"] = ["age"]
        everything = [f for fs in groups.values() for f in fs]
        full = cross_validate(model, design_matrix(data, everything), y)["summary"]
        for factor, features in groups.items():
            rest = [f for f in everything if f not in features]
            without = cross_validate(model, design_matrix(data, rest), y)["summary"]
            alone = cross_validate(model, design_matrix(data, features), y)["summary"]
            rows.append({"model_includes_age": with_age, "factor": factor, "features": ", ".join(features),
                         "auc_all": full["auc"], "auc_without": without["auc"], "auc_drop": full["auc"] - without["auc"],
                         "auc_alone": alone["auc"], "auc_alone_low": alone["auc_low"],
                         "auc_alone_high": alone["auc_high"]})
    return pd.DataFrame(rows).sort_values(["model_includes_age", "auc_drop"], ascending=[True, False])


# --------------------------------------------------------------------------- run
def run(output_dir: Path = settings.OUTPUT_DIR, analysis_dir: Path = settings.ANALYSIS_DIR) -> Dict[str, pd.DataFrame]:
    people = pd.read_csv(output_dir / "person_features.csv", dtype={"patient_id": str})
    analysis_dir.mkdir(parents=True, exist_ok=True)
    results = {"hypotheses": hypotheses(people), "age_trends": age_trends(people)}
    results["decade_norms"] = decade_norms(people, results["age_trends"])
    results["outside_norm"] = outside_norm(people, results["decade_norms"])
    logger.info("statistics done; cross-validating classifiers")
    results.update(classifier(people))
    results["classifier_permutation_test"] = permutation_test(people)
    logger.info("ablation")
    results["ablation"] = ablation(people)
    for name, table in results.items():
        table.to_csv(analysis_dir / f"{name}.csv", index=False)
    return results
