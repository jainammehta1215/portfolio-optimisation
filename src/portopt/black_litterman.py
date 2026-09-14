"""
Black-Litterman model (Black & Litterman 1992; He & Litterman 1999).

Pipeline
--------
1. Reverse optimisation: implied equilibrium excess returns
       Pi = delta * Sigma * w_mkt
   where delta is the market's risk aversion, estimated as
       delta = (E[r_mkt] - rf) / Var(r_mkt)   (clipped to a sane range).

2. Views: k linear views  P mu = Q + eps,  eps ~ N(0, Omega).
   Omega defaults to He-Litterman: diag(P (tau Sigma) P').

3. Posterior (the "master formula"):
       mu_BL    = [ (tau Sigma)^-1 + P' Omega^-1 P ]^-1 [ (tau Sigma)^-1 Pi + P' Omega^-1 Q ]
       Sigma_BL = Sigma + [ (tau Sigma)^-1 + P' Omega^-1 P ]^-1

   With no views the posterior collapses to the prior, and the equilibrium
   portfolio is recovered exactly by unconstrained mean-variance at delta.

The backtest uses a *systematic* view (12-1 momentum spread) so that no
human judgement, and therefore no lookahead, enters the historical test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import MODEL, ModelConfig


@dataclass
class BLResult:
    prior: pd.Series            # equilibrium excess returns Pi
    posterior_mu: pd.Series     # mu_BL (excess)
    posterior_cov: pd.DataFrame
    delta: float
    P: np.ndarray
    Q: np.ndarray
    omega: np.ndarray
    view_labels: List[str]

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({"prior": self.prior, "posterior": self.posterior_mu,
                             "shift": self.posterior_mu - self.prior})


# --------------------------------------------------------------------------- #
# Equilibrium
# --------------------------------------------------------------------------- #
def implied_risk_aversion(market_returns: pd.Series, rf_daily: pd.Series,
                          cfg: ModelConfig = MODEL, periods: int = 252) -> float:
    """
    delta = (mean excess return) / variance, both annualised. Clipped to
    cfg.bl_delta_bounds because short windows can give absurd values
    (negative after a crash, huge after a calm rally).
    """
    excess = market_returns - rf_daily.reindex(market_returns.index).fillna(0.0)
    mean_ann = excess.mean() * periods
    var_ann = market_returns.var() * periods
    if var_ann <= 0 or not np.isfinite(mean_ann):
        return cfg.bl_default_delta
    delta = mean_ann / var_ann
    lo, hi = cfg.bl_delta_bounds
    return float(np.clip(delta, lo, hi))


def equilibrium_returns(cov: pd.DataFrame, w_mkt: pd.Series, delta: float) -> pd.Series:
    """Pi = delta * Sigma * w_mkt (annualised excess returns)."""
    w = w_mkt.reindex(cov.index).fillna(0.0)
    w = w / w.sum()
    return pd.Series(delta * cov.values @ w.values, index=cov.index, name="Pi")


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def absolute_view(assets: List[str], asset: str, ret: float) -> Tuple[np.ndarray, float, str]:
    p = np.zeros(len(assets)); p[assets.index(asset)] = 1.0
    return p, ret, f"{asset} returns {ret:.1%}"


def relative_view(assets: List[str], long: List[str], short: List[str],
                  spread: float) -> Tuple[np.ndarray, float, str]:
    """Equal-weighted `long` basket outperforms `short` basket by `spread`."""
    p = np.zeros(len(assets))
    for a in long:
        p[assets.index(a)] = 1.0 / len(long)
    for a in short:
        p[assets.index(a)] = -1.0 / len(short)
    return p, spread, f"{'+'.join(long)} beats {'+'.join(short)} by {spread:.1%}"


def momentum_view(returns_window: pd.DataFrame, cfg: ModelConfig = MODEL
                  ) -> Optional[Tuple[np.ndarray, float, str]]:
    """
    Systematic relative view from 12-1 month momentum: the top-k names by
    trailing return (skipping the most recent month) are expected to beat
    the bottom-k by cfg.bl_momentum_view_ann per year.
    """
    lb, skip, k = cfg.bl_momentum_lookback, cfg.bl_momentum_skip, cfg.bl_top_bottom_k
    if len(returns_window) < lb + skip:
        return None
    window = returns_window.iloc[-(lb + skip):-skip] if skip > 0 else returns_window.iloc[-lb:]
    mom = (1 + window).prod() - 1
    ranked = mom.sort_values()
    assets = list(returns_window.columns)
    return relative_view(assets, long=list(ranked.index[-k:]),
                         short=list(ranked.index[:k]), spread=cfg.bl_momentum_view_ann)


# --------------------------------------------------------------------------- #
# Posterior
# --------------------------------------------------------------------------- #
def black_litterman(cov: pd.DataFrame, w_mkt: pd.Series, delta: float,
                    views: Optional[List[Tuple[np.ndarray, float, str]]] = None,
                    tau: float = MODEL.bl_tau,
                    omega: Optional[np.ndarray] = None,
                    view_confidence: Optional[np.ndarray] = None) -> BLResult:
    """
    Compute the Black-Litterman posterior.

    views           : list of (P_row, Q, label). None or [] -> prior only.
    omega           : explicit (k x k) view covariance, overrides default.
    view_confidence : optional per-view multipliers in (0, inf). Values < 1
                      tighten Omega (more confident), > 1 loosen it. This is
                      a simple, transparent alternative to Idzorek's method.
    """
    assets = list(cov.index)
    n = len(assets)
    pi = equilibrium_returns(cov, w_mkt, delta)
    S = cov.values
    tS = tau * S

    if not views:
        return BLResult(prior=pi, posterior_mu=pi.copy(), posterior_cov=cov.copy(),
                        delta=delta, P=np.zeros((0, n)), Q=np.zeros(0),
                        omega=np.zeros((0, 0)), view_labels=[])

    P = np.vstack([v[0] for v in views])
    Q = np.array([v[1] for v in views], dtype=float)
    labels = [v[2] for v in views]
    k = P.shape[0]
    if P.shape[1] != n:
        raise ValueError("View matrix width does not match number of assets")

    if omega is None:
        omega = np.diag(np.diag(P @ tS @ P.T))
        if view_confidence is not None:
            omega = omega * np.asarray(view_confidence, dtype=float)[:, None] * np.eye(k)
    if np.any(np.diag(omega) <= 0):
        raise ValueError("Omega must have strictly positive diagonal")

    tS_inv = np.linalg.inv(tS)
    om_inv = np.linalg.inv(omega)
    M = np.linalg.inv(tS_inv + P.T @ om_inv @ P)
    mu_bl = M @ (tS_inv @ pi.values + P.T @ om_inv @ Q)
    cov_bl = S + M

    return BLResult(prior=pi,
                    posterior_mu=pd.Series(mu_bl, index=assets, name="mu_BL"),
                    posterior_cov=pd.DataFrame(cov_bl, index=assets, columns=assets),
                    delta=delta, P=P, Q=Q, omega=omega, view_labels=labels)


def unconstrained_bl_weights(res: BLResult) -> pd.Series:
    """
    w* = (delta Sigma_BL)^-1 mu_BL, the textbook unconstrained solution.
    Shown for reference; the project uses the constrained optimiser.
    """
    w = np.linalg.solve(res.delta * res.posterior_cov.values, res.posterior_mu.values)
    return pd.Series(w, index=res.posterior_mu.index, name="w_BL_unconstrained")
