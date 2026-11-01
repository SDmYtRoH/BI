"""Hand force balance: RHYTHM vs EXACT_FORCE, age categories, age vs disease (week 4).

Balance = weaker hand / stronger hand (1 = both hands equal, 0 = one hand does not
press). A hand is weakly used when the balance is below the 2.5 % quantile of the
healthy controls (``config.WEAK_HAND_REFERENCE_QUANTILE``), as in week 3.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from maksym_rehapiano.config import AGE_CATEGORY_EDGES, AGE_CATEGORY_LABELS, RHYTHM_MIN_GROUP_SIZE
from rehapiano.config import WEAK_HAND_REFERENCE_QUANTILE
from rehapiano.stats.tests import holm_adjust, mann_whitney

TESTS = {"exact_force": "EXACT_FORCE", "rhythm": "RHYTHM"}


def _balance(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    left, right = left.clip(lower=0), right.clip(lower=0)
    stronger = np.maximum(left, right)
    ratio = np.minimum(left, right) / stronger.where(stronger > 0)
    weaker = pd.Series(np.where(left < right, "left", "right"), index=left.index).where(ratio.notna())
    return ratio, weaker


def rhythm_person_table(by_attempt: pd.DataFrame) -> pd.DataFrame:
    """One row per person: hand values (median over attempts, mean over measurements) and balances."""

    valid = by_attempt.loc[by_attempt["status"] == "valid"]
    values = ["press_amplitude", "mean_force", "force_range", "press_rate_hz"]
    measurement = valid.groupby(["patient_id", "measurement_id", "hand"])[values].median().reset_index()
    person = measurement.groupby(["patient_id", "hand"])[values].mean()
    wide = person[["press_amplitude", "mean_force", "force_range"]].unstack("hand")
    wide.columns = [f"{hand}_{value}" for value, hand in wide.columns]
    info = by_attempt.groupby("patient_id").agg(
        group=("group", "first"), age=("age", "first"), gender=("gender", "first"),
        n_measurements=("measurement_id", "nunique"))
    info["n_attempts"] = valid.drop_duplicates(["measurement_id", "attempt"]).groupby("patient_id").size()
    info["press_rate_hz"] = person["press_rate_hz"].groupby("patient_id").mean()
    table = info.join(wide)
    table["rhythm_ratio"], table["rhythm_weaker_hand"] = _balance(table["left_press_amplitude"],
                                                                  table["right_press_amplitude"])
    table["rhythm_mean_force_ratio"], _ = _balance(table["left_mean_force"], table["right_mean_force"])
    table["rhythm_range_ratio"], _ = _balance(table["left_force_range"], table["right_force_range"])
    return table.reset_index()


def combine_tests(rhythm: pd.DataFrame, exact_force_activity: pd.DataFrame) -> pd.DataFrame:
    """Person table with both balances; ``exact_force_activity`` is week 3 ``hand_activity.csv``."""

    ef = exact_force_activity.rename(columns={"force_ratio": "exact_force_ratio",
                                              "force_weaker_hand": "exact_force_weaker_hand"})
    table = rhythm.merge(ef[["patient_id", "exact_force_ratio", "exact_force_weaker_hand"]], on="patient_id",
                         how="outer")
    table["age_category"] = pd.cut(table["age"], AGE_CATEGORY_EDGES, right=False, labels=AGE_CATEGORY_LABELS)
    return table


def weak_threshold(table: pd.DataFrame, column: str, reference: pd.Series | None = None,
                   quantile: float = WEAK_HAND_REFERENCE_QUANTILE) -> float:
    """``quantile`` of ``column`` among the healthy controls (or the rows selected by ``reference``)."""

    mask = table["group"].eq("control") if reference is None else reference
    return float(table.loc[mask, column].quantile(quantile))


def add_weak_flags(table: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    """``weak_<test>`` = balance below the threshold of that test (NaN when the test is missing)."""

    result = table.copy()
    for test, threshold in thresholds.items():
        values = result[f"{test}_ratio"]
        result[f"weak_{test}"] = (values < threshold).where(values.notna())
    return result


def test_agreement(table: pd.DataFrame) -> Dict[str, float]:
    """How well RHYTHM and EXACT_FORCE agree: correlation, weak-hand agreement and side agreement."""

    both = table.dropna(subset=["exact_force_ratio", "rhythm_ratio"])
    rho, p = stats.spearmanr(both["exact_force_ratio"], both["rhythm_ratio"])
    a, b = both["weak_exact_force"].astype(bool), both["weak_rhythm"].astype(bool)
    observed = (a == b).mean()
    expected = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    kappa = (observed - expected) / (1 - expected) if expected < 1 else np.nan
    flagged = both.loc[a | b]
    side = (flagged["exact_force_weaker_hand"] == flagged["rhythm_weaker_hand"]).mean() if len(flagged) else np.nan
    return {"n_people": len(both), "spearman_rho": float(rho), "spearman_p": float(p),
            "weak_both": int((a & b).sum()), "weak_only_exact_force": int((a & ~b).sum()),
            "weak_only_rhythm": int((~a & b).sum()), "weak_none": int((~a & ~b).sum()),
            "cohen_kappa": float(kappa), "same_weaker_hand_if_flagged": float(side)}


def by_age_category(table: pd.DataFrame, tests: Sequence[str] = tuple(TESTS)) -> pd.DataFrame:
    """Group x age category: number of people, median balance (IQR) and weakly used hands per test."""

    rows = []
    for (group, category), frame in table.groupby(["group", "age_category"], observed=False):
        for test in tests:
            values = frame[f"{test}_ratio"].dropna()
            weak = frame[f"weak_{test}"].dropna().astype(bool)
            rows.append({"group": group, "age_category": category, "test": TESTS[test], "n_people": len(values),
                         "median_ratio": values.median(), "q1": values.quantile(0.25), "q3": values.quantile(0.75),
                         "n_weak": int(weak.sum()), "percent_weak": 100 * weak.mean() if len(weak) else np.nan})
    return pd.DataFrame(rows)


def age_effect_in_controls(table: pd.DataFrame, tests: Sequence[str] = tuple(TESTS)) -> pd.DataFrame:
    """Age alone: Spearman age ~ balance and Mann-Whitney (< 50 vs >= 50) in healthy controls only."""

    controls = table.loc[table["group"] == "control"]
    rows = []
    for test in tests:
        frame = controls.dropna(subset=[f"{test}_ratio"])
        rho, p = stats.spearmanr(frame["age"], frame[f"{test}_ratio"])
        young, old = frame.loc[frame["age"] < 50, f"{test}_ratio"], frame.loc[frame["age"] >= 50, f"{test}_ratio"]
        rows.append({"test": TESTS[test], "n_controls": len(frame), "spearman_rho_age": rho, "spearman_p": p,
                     **{f"young_vs_old_{k}": v for k, v in mann_whitney(young, old).items()}})
    return pd.DataFrame(rows)


def disease_effect_by_age(table: pd.DataFrame, tests: Sequence[str] = tuple(TESTS),
                          min_group_size: int = RHYTHM_MIN_GROUP_SIZE) -> pd.DataFrame:
    """Disease alone: patients vs controls of the same age category (Mann-Whitney, Holm over all rows).

    Categories where either group has fewer than ``min_group_size`` people are skipped.

    ``effect_r`` > 0 means a higher balance in patients, < 0 a lower one (more asymmetric hands).
    """

    rows = []
    categories = [*AGE_CATEGORY_LABELS, "≥ 50 spolu"]
    for test in tests:
        for category in categories:
            mask = table["age"] >= 50 if category == "≥ 50 spolu" else table["age_category"] == category
            frame = table.loc[mask].dropna(subset=[f"{test}_ratio"])
            controls = frame.loc[frame["group"] == "control", f"{test}_ratio"]
            patients = frame.loc[frame["group"] == "patient", f"{test}_ratio"]
            if min(len(controls), len(patients)) < min_group_size:
                continue
            rows.append({"test": TESTS[test], "age_category": category, **mann_whitney(controls, patients)})
    result = pd.DataFrame(rows).rename(columns={"n_a": "n_controls", "n_b": "n_patients",
                                                "median_a": "median_controls", "median_b": "median_patients"})
    result["p_holm"] = holm_adjust(result["p"])
    return result


def ols(y: np.ndarray, X: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """Ordinary least squares with HC3 robust standard errors (no normality of residuals assumed)."""

    y, X = np.asarray(y, float), np.asarray(X, float)
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    residuals = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    meat = X.T @ (X * (residuals / (1 - leverage))[:, None] ** 2)
    se = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), n - k)
    q = stats.t.ppf(0.975, n - k)
    return pd.DataFrame({"term": names, "coefficient": beta, "ci_low": beta - q * se, "ci_high": beta + q * se,
                         "p": p, "n": n})


def age_disease_model(table: pd.DataFrame, tests: Sequence[str] = tuple(TESTS)) -> pd.DataFrame:
    """balance = b0 + b_age * (age / 10) + b_disease * patient, fitted for each test.

    ``b_age`` is the change of the balance per 10 years at the same health status and
    ``b_disease`` the difference patient - control at the same age.
    """

    frames = []
    for test in tests:
        frame = table.dropna(subset=[f"{test}_ratio", "age"])
        X = np.column_stack([np.ones(len(frame)), frame["age"] / 10, (frame["group"] == "patient").astype(float)])
        model = ols(frame[f"{test}_ratio"], X, ["intercept", "age_per_10_years", "patient"])
        frames.append(model.assign(test=TESTS[test]))
    return pd.concat(frames, ignore_index=True)[["test", "term", "coefficient", "ci_low", "ci_high", "p", "n"]]
