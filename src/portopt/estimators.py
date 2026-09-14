"""
Return and covariance estimators.

All estimators take a (T x N) DataFrame of *daily* simple returns and return
annualised quantities, so that everything downstream works in annual units
and Sharpe ratios are directly interpretable.

Available covariance estimators
-------------------------------
sample       : plain sample covariance (the thing that breaks Markowitz)
ledoit_wolf  : Ledoit-Wolf (2004) shrinkage toward a scaled identity target
const_corr   : Ledoit-Wolf (2003) shrinkage toward constant correlation
ewma         : exponentially weighted covariance (RiskMetrics style)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


@dataclass
class Estimates:
    """Container for one set of inputs to an optimiser (annualised)."""
    mu: pd.Series
    cov: pd.DataFrame
    label: str = ""

    @property
    def assets(self) -> list:
        return list(self.mu.index)

    def as_arrays(self):
        return self.mu.values.astype(float), self.cov.values.astype(float)


# --------------------------------------------------------------------------- #
# Expected returns
# --------------------------------------------------------------------------- #
def mean_historical(returns: pd.DataFrame, periods: int = TRADING_DAYS) -> pd.Series:
    """Arithmetic mean daily return, annualised. Extremely noisy; that is the point."""
    return returns.mean() * periods


def mean_shrunk(returns: pd.DataFrame, shrink: float = 0.5,
                periods: int = TRADING_DAYS) -> pd.Series:
    """
    James-Stein style shrinkage of the mean vector toward the cross-sectional
    grand mean. `shrink=1` collapses all expected returns to a single value,
    which makes mean-variance identical to minimum variance.
    """
    mu = mean_historical(returns, periods)
    grand = mu.mean()
    return (1 - shrink) * mu + shrink * grand


# --------------------------------------------------------------------------- #
# Covariance
# --------------------------------------------------------------------------- #
def cov_sample(returns: pd.DataFrame, periods: int = TRADING_DAYS) -> pd.DataFrame:
    return returns.cov() * periods


def cov_ledoit_wolf(returns: pd.DataFrame, periods: int = TRADING_DAYS) -> pd.DataFrame:
    """
    Ledoit-Wolf shrinkage toward mu*I where mu is the average variance.
    Optimal shrinkage intensity is estimated analytically from the data.
    """
    lw = LedoitWolf(assume_centered=False).fit(returns.values)
    cov = pd.DataFrame(lw.covariance_ * periods, index=returns.columns, columns=returns.columns)
    cov.attrs["shrinkage"] = float(lw.shrinkage_)
    return cov


def cov_constant_correlation(returns: pd.DataFrame,
                             periods: int = TRADING_DAYS) -> pd.DataFrame:
    """
    Ledoit & Wolf (2003) "Honey, I shrunk the sample covariance matrix".
    Target: all pairwise correlations equal to their average. The shrinkage
    intensity formula follows the paper's Appendix B.
    """
    # Port of Ledoit & Wolf's reference implementation (covCor.m).
    x = returns.values - returns.values.mean(axis=0)
    t, n = x.shape
    sample = (x.T @ x) / t
    var = np.diag(sample)
    sd = np.sqrt(var)
    r_bar = (np.sum(sample / np.outer(sd, sd)) - n) / (n * (n - 1))
    target = r_bar * np.outer(sd, sd)
    np.fill_diagonal(target, var)

    # pi-hat: asymptotic variance of the sample covariance entries
    y = x ** 2
    phi_mat = (y.T @ y) / t - 2 * (x.T @ x) * sample / t + sample ** 2
    phi = phi_mat.sum()

    # rho-hat: asymptotic covariance between sample and target errors
    term1 = ((x ** 3).T @ x) / t
    help_ = (x.T @ x) / t
    help_diag = np.diag(help_)[:, None]
    term2 = help_diag * sample
    term3 = help_ * var[:, None]
    term4 = var[:, None] * sample
    theta = term1 - term2 - term3 + term4
    np.fill_diagonal(theta, 0.0)
    rho = np.diag(phi_mat).sum() + r_bar * (np.outer(1.0 / sd, sd) * theta).sum()

    gamma = np.linalg.norm(sample - target, "fro") ** 2
    kappa = (phi - rho) / gamma if gamma > 0 else 0.0
    shrink = float(np.clip(kappa / t, 0.0, 1.0))
    cov = shrink * target + (1 - shrink) * sample
    out = pd.DataFrame(cov * periods, index=returns.columns, columns=returns.columns)
    out.attrs["shrinkage"] = shrink
    return out


def cov_ewma(returns: pd.DataFrame, halflife: int = 63,
             periods: int = TRADING_DAYS) -> pd.DataFrame:
    """Exponentially weighted covariance with the given half-life in days."""
    cov = returns.ewm(halflife=halflife, min_periods=halflife).cov().iloc[-len(returns.columns):]
    cov.index = cov.index.get_level_values(1)
    return cov * periods


def nearest_psd(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Clip negative eigenvalues; used defensively before Cholesky."""
    sym = (cov + cov.T) / 2
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, eps, None)
    return (vecs * vals) @ vecs.T


def cholesky_factor(cov: np.ndarray) -> np.ndarray:
    """Lower-triangular L with cov = L L'. Adds jitter if needed."""
    cov = np.asarray(cov, dtype=float)
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(nearest_psd(cov) + 1e-10 * np.eye(len(cov)))


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
def estimate(returns: pd.DataFrame, cov_method: str = "ledoit_wolf",
             mean_method: str = "historical", label: Optional[str] = None) -> Estimates:
    """Build an `Estimates` object from a returns window."""
    cov_fn = {
        "sample": cov_sample,
        "ledoit_wolf": cov_ledoit_wolf,
        "const_corr": cov_constant_correlation,
        "ewma": cov_ewma,
    }[cov_method]
    mean_fn = {"historical": mean_historical, "shrunk": mean_shrunk}[mean_method]
    return Estimates(mu=mean_fn(returns), cov=cov_fn(returns),
                     label=label or f"{mean_method}/{cov_method}")


def condition_number(cov: pd.DataFrame) -> float:
    vals = np.linalg.eigvalsh(cov.values)
    return float(vals.max() / max(vals.min(), 1e-16))
