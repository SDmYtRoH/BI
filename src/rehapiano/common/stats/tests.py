"""Thin wrappers around non-parametric tests that return plain dicts.

Non-parametric tests are used throughout: the features are skewed amplitudes
and ratios, and the groups are small.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaNs are left untouched)."""

    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    m = len(order)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (m - rank) * p[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Two independent groups; effect size = rank-biserial correlation (b > a is positive)."""

    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    result = {"n_a": len(a), "n_b": len(b), "median_a": np.median(a) if len(a) else np.nan,
              "median_b": np.median(b) if len(b) else np.nan, "u": np.nan, "p": np.nan, "effect_r": np.nan}
    if len(a) >= 3 and len(b) >= 3:
        u, p = stats.mannwhitneyu(b, a, alternative="two-sided")
        result.update(u=float(u), p=float(p), effect_r=float(2 * u / (len(a) * len(b)) - 1))
    return result


def van_elteren(values: Sequence[float], patient: Sequence[int], strata: Sequence) -> Dict[str, float]:
    """Mann-Whitney test inside every stratum (e.g. age decade), combined over strata (van Elteren).

    Only strata with both groups contribute, so patients are compared with healthy people of the same
    stratum. ``z`` > 0 = patients (``patient`` == 1) have higher values.
    """

    frame = pd.DataFrame({"value": values, "patient": patient, "stratum": strata}).dropna(subset=["value"])
    numerator = variance = 0.0
    used = 0
    for _, stratum in frame.groupby("stratum", observed=True):
        m, n = int(stratum["patient"].sum()), int((stratum["patient"] == 0).sum())
        if m == 0 or n == 0:
            continue
        total = m + n
        ranks = stats.rankdata(stratum["value"])
        _, ties = np.unique(stratum["value"], return_counts=True)
        weight = 1 / (total + 1)
        numerator += weight * (ranks[stratum["patient"].to_numpy() == 1].sum() - m * (total + 1) / 2)
        variance += weight ** 2 * m * n / 12 * ((total + 1) - (ties ** 3 - ties).sum() / (total * (total - 1)))
        used += total
    if variance <= 0:
        return {"n": used, "z": np.nan, "p": np.nan}
    z = numerator / np.sqrt(variance)
    return {"n": used, "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


def icc_agreement(a: Sequence[float], b: Sequence[float]) -> float:
    """ICC(2,1): absolute agreement of two measurements of the same people (1 = identical, 0 = no agreement)."""

    x = np.column_stack([np.asarray(a, float), np.asarray(b, float)])
    x = x[np.isfinite(x).all(axis=1)]
    n, k = x.shape
    if n < 3:
        return np.nan
    grand = x.mean()
    ms_rows = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_columns = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    residual = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand
    ms_error = (residual ** 2).sum() / ((n - 1) * (k - 1))
    return float((ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n))


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired samples; effect size = matched-pairs rank-biserial correlation (b > a is positive)."""

    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    difference = b[keep] - a[keep]
    result = {"n_pairs": int(keep.sum()), "median_difference": np.median(difference) if keep.any() else np.nan,
              "p": np.nan, "effect_r": np.nan}
    nonzero = difference[difference != 0]
    if len(nonzero) >= 5:
        _, p = stats.wilcoxon(nonzero)
        ranks = stats.rankdata(np.abs(nonzero))
        result.update(p=float(p), effect_r=float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()))
    return result


def friedman(wide: pd.DataFrame) -> Dict[str, float]:
    """Friedman test over the columns of ``wide`` (rows = subjects, complete cases only).

    Effect size = Kendall's W (0 no agreement between subjects, 1 same ranking for all).
    """

    complete = wide.dropna()
    n, k = complete.shape
    result = {"n_subjects": n, "chi2": np.nan, "p": np.nan, "kendall_w": np.nan}
    if n >= 5 and k >= 3:
        chi2, p = stats.friedmanchisquare(*[complete[column] for column in complete.columns])
        result.update(chi2=float(chi2), p=float(p), kendall_w=float(chi2 / (n * (k - 1))))
    return result


def pairwise_wilcoxon(wide: pd.DataFrame) -> pd.DataFrame:
    """Post-hoc paired Wilcoxon tests between all column pairs, Holm-adjusted."""

    complete = wide.dropna()
    rows = [{"a": a, "b": b, **wilcoxon_paired(complete[a], complete[b])}
            for a, b in combinations(complete.columns, 2)]
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_holm"] = holm_adjust(table["p"])
    return table
