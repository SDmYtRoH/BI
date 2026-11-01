"""Linear model with robust standard errors (used for age / disease effects)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


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
