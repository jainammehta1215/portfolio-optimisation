"""
Portfolio optimisers.

Design
------
All convex problems are compiled once per (universe, constraint set) via
cvxpy's Disciplined Parametrized Programming (DPP). Subsequent solves only
update parameter values, which makes the walk-forward backtest and the
Michaud resampling loop (thousands of solves) fast.

The covariance enters through its Cholesky factor L (Sigma = L L') so that
the risk term `sum_squares(L.T @ w)` is DPP-compliant; `quad_form(w, Sigma)`
with a parametrised Sigma is not.

Objectives
----------
mean_variance : max  mu'w - (lambda/2) w'Sigma w
min_variance  : min  w'Sigma w
max_sharpe    : max  (mu'w - rf) / sqrt(w'Sigma w)   (Charnes-Cooper transform)
target_return : min  w'Sigma w  s.t. mu'w >= r       (efficient frontier)
risk_parity   : equal (or budgeted) risk contributions
resampled     : Michaud (1998) averaging of optimal weights across bootstraps

Constraint set (all optional)
-----------------------------
fully invested, long-only, per-asset cap, per-sector cap, L1 turnover cap
relative to the previous weights.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import CONSTRAINTS, ConstraintConfig
from .estimators import Estimates, cholesky_factor

logger = logging.getLogger(__name__)

SOLVER_ORDER = [cp.CLARABEL, cp.OSQP, cp.SCS]


# --------------------------------------------------------------------------- #
# Constraint specification
# --------------------------------------------------------------------------- #
@dataclass
class ConstraintSet:
    """
    Numeric constraint set for a fixed universe of size n.

    sector_matrix : (k x n) binary matrix, row k sums the weights in sector k.
    """
    n: int
    long_only: bool = True
    max_weight: Optional[float] = 0.20
    sector_matrix: Optional[np.ndarray] = None
    max_sector_weight: Optional[float] = 0.35
    max_turnover: Optional[float] = 0.50

    @classmethod
    def from_config(cls, n: int, sector_matrix: Optional[np.ndarray] = None,
                    cfg: ConstraintConfig = CONSTRAINTS) -> "ConstraintSet":
        return cls(n=n, long_only=cfg.long_only, max_weight=cfg.max_weight,
                   sector_matrix=sector_matrix, max_sector_weight=cfg.max_sector_weight,
                   max_turnover=cfg.max_turnover)

    def unconstrained(self) -> "ConstraintSet":
        """Long-only, fully invested, nothing else. Useful for illustrations."""
        return ConstraintSet(n=self.n, long_only=True, max_weight=None,
                             sector_matrix=None, max_sector_weight=None, max_turnover=None)

    def validate(self) -> None:
        if self.max_weight is not None and self.max_weight * self.n < 1.0 - 1e-9:
            raise ValueError(f"max_weight={self.max_weight} infeasible for n={self.n}")
        if self.sector_matrix is not None and self.max_sector_weight is not None:
            k = self.sector_matrix.shape[0]
            if self.max_sector_weight * k < 1.0 - 1e-9:
                raise ValueError("Sector caps cannot sum to less than 100%")


def _cvx_constraints(w, cs: ConstraintSet, w_prev, tau, scale=None) -> list:
    """
    Build cvxpy constraints for weight vector `w`.
    `scale` is the homogenising variable kappa in the max-Sharpe transform;
    when None the portfolio is fully invested (sum to 1).
    """
    one = 1.0 if scale is None else scale
    cons = [cp.sum(w) == one]
    if cs.long_only:
        cons.append(w >= 0)
    if cs.max_weight is not None:
        cons.append(w <= cs.max_weight * one)
    if cs.sector_matrix is not None and cs.max_sector_weight is not None:
        cons.append(cs.sector_matrix @ w <= cs.max_sector_weight * one)
    if cs.max_turnover is not None and w_prev is not None:
        cons.append(cp.norm1(w - one * w_prev) <= tau * one)
    return cons


# --------------------------------------------------------------------------- #
# Compiled convex engine
# --------------------------------------------------------------------------- #
class ConvexEngine:
    """
    Holds compiled cvxpy problems for one universe and constraint set.

    Parameters updated per solve: Cholesky factor L, expected returns mu,
    previous weights w_prev, turnover cap tau, target return r_target.
    """

    def __init__(self, cs: ConstraintSet):
        cs.validate()
        self.cs = cs
        n = cs.n
        self.n = n

        # Shared parameters
        self.L = cp.Parameter((n, n), name="L")
        self.mu = cp.Parameter(n, name="mu")
        self.w_prev = cp.Parameter(n, name="w_prev", value=np.full(n, 1.0 / n))
        self.tau = cp.Parameter(nonneg=True, name="tau", value=2.0)
        self.r_target = cp.Parameter(name="r_target", value=0.0)
        self.mu_excess = cp.Parameter(n, name="mu_excess")

        # ---- Problem A: mean-variance utility  (mu pre-scaled by 1/lambda) --
        self.w = cp.Variable(n, name="w")
        risk = cp.sum_squares(self.L.T @ self.w)
        self._mv = cp.Problem(cp.Minimize(0.5 * risk - self.mu @ self.w),
                              _cvx_constraints(self.w, cs, self.w_prev, self.tau))

        # ---- Problem B: target-return (frontier) ----------------------------
        self._tr = cp.Problem(cp.Minimize(risk),
                              _cvx_constraints(self.w, cs, self.w_prev, self.tau)
                              + [self.mu @ self.w >= self.r_target])

        # ---- Problem C: max Sharpe via Charnes-Cooper -----------------------
        self.y = cp.Variable(n, name="y")
        self.kappa = cp.Variable(nonneg=True, name="kappa")
        self._ms = cp.Problem(cp.Minimize(cp.sum_squares(self.L.T @ self.y)),
                              _cvx_constraints(self.y, cs, self.w_prev, self.tau, scale=self.kappa)
                              + [self.mu_excess @ self.y == 1])

        # ---- Problem D: max expected return (LP, frontier upper bound) ------
        self._mr = cp.Problem(cp.Maximize(self.mu @ self.w),
                              _cvx_constraints(self.w, cs, self.w_prev, self.tau))

        # ---- Problem E: risk parity via Spinu's convex formulation ----------
        # min 0.5 y'Sigma y - sum b_i log(y_i);  w = y / sum(y). Exact ERC for
        # long-only; caps are checked afterwards and a SLSQP fallback is used.
        self.b = cp.Parameter(n, nonneg=True, name="budgets", value=np.full(n, 1.0 / n))
        self.y_rp = cp.Variable(n, name="y_rp")
        self._rp = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(self.L.T @ self.y_rp)
                                          - self.b @ cp.log(self.y_rp)))

        for p in (self._mv, self._tr, self._ms, self._mr, self._rp):
            if not p.is_dcp(dpp=True):
                logger.warning("Problem is not DPP-compliant; solves will be slower")

    # -- utilities -----------------------------------------------------------
    def _set_inputs(self, mu, cov, w_prev=None, tau=None):
        self.L.value = cholesky_factor(cov)
        self.mu.value = np.asarray(mu, dtype=float)
        if w_prev is not None:
            self.w_prev.value = np.asarray(w_prev, dtype=float)
            self.tau.value = float(tau if tau is not None else (self.cs.max_turnover or 2.0))
        else:
            self.w_prev.value = np.full(self.n, 1.0 / self.n)
            self.tau.value = 2.0   # L1 distance can never exceed 2 -> inactive

    @staticmethod
    def _solve(problem: cp.Problem) -> bool:
        for solver in SOLVER_ORDER:
            try:
                problem.solve(solver=solver, warm_start=True)
                if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    return True
            except (cp.SolverError, ValueError) as exc:
                logger.debug("solver %s failed: %s", solver, exc)
        return False

    @staticmethod
    def _clean(w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float).copy()
        w[np.abs(w) < 1e-8] = 0.0
        w = np.clip(w, 0.0, None) if (w >= -1e-6).all() else w
        return w / w.sum()

    # -- public solves -------------------------------------------------------
    def mean_variance(self, mu, cov, risk_aversion: float = 2.5,
                      w_prev=None, tau=None) -> np.ndarray:
        """max mu'w - (lambda/2) w'Sigma w subject to the constraint set."""
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        self._set_inputs(np.asarray(mu) / risk_aversion, cov, w_prev, tau)
        if not self._solve(self._mv):
            raise RuntimeError("mean_variance: no solver returned an optimal solution")
        return self._clean(self.w.value)

    def min_variance(self, cov, w_prev=None, tau=None) -> np.ndarray:
        return self.mean_variance(np.zeros(self.n), cov, 1.0, w_prev, tau)

    def max_return(self, mu, cov, w_prev=None, tau=None) -> float:
        self._set_inputs(mu, cov, w_prev, tau)
        if not self._solve(self._mr):
            raise RuntimeError("max_return LP failed")
        return float(self._mr.value)

    def max_sharpe(self, mu, cov, rf: float = 0.0, w_prev=None, tau=None,
                   fallback_min_var: bool = True) -> np.ndarray:
        """
        Tangency portfolio. If no feasible portfolio has expected excess
        return > 0 the transform is infeasible; we then fall back to the
        minimum-variance portfolio (documented behaviour, logged).
        """
        self._set_inputs(mu, cov, w_prev, tau)
        self.mu_excess.value = np.asarray(mu, dtype=float) - rf
        ok = self._solve(self._ms)
        if ok and self.kappa.value is not None and self.kappa.value > 1e-10:
            return self._clean(self.y.value / self.kappa.value)
        if fallback_min_var:
            logger.info("max_sharpe infeasible (all excess returns <= 0?); using min variance")
            return self.min_variance(cov, w_prev, tau)
        raise RuntimeError("max_sharpe infeasible")

    def target_return(self, mu, cov, r: float, w_prev=None, tau=None) -> Optional[np.ndarray]:
        self._set_inputs(mu, cov, w_prev, tau)
        self.r_target.value = float(r)
        if not self._solve(self._tr):
            return None
        return self._clean(self.w.value)

    def efficient_frontier(self, mu, cov, n_points: int = 30,
                           w_prev=None, tau=None) -> pd.DataFrame:
        """
        Sweep target returns from the min-variance return to the maximum
        feasible return. Returns a frame with columns ret, vol, and weights.
        """
        mu = np.asarray(mu, dtype=float)
        cov = np.asarray(cov, dtype=float)
        w_mv = self.min_variance(cov, w_prev, tau)
        r_lo = float(mu @ w_mv)
        r_hi = self.max_return(mu, cov, w_prev, tau)
        rows = []
        for r in np.linspace(r_lo, r_hi - 1e-9, n_points):
            w = self.target_return(mu, cov, r, w_prev, tau)
            if w is None:
                continue
            rows.append(dict(ret=float(mu @ w), vol=float(np.sqrt(w @ cov @ w)), weights=w))
        return pd.DataFrame(rows)

    def risk_parity(self, cov, budgets=None, w_prev=None, tau=None) -> np.ndarray:
        """
        Equal (or budgeted) risk contribution portfolio.

        Uses the convex log-barrier formulation, which is exact for the
        long-only case. If the result violates position or sector caps the
        problem is re-solved with SLSQP under bounds; contributions are then
        equalised as far as the caps allow.
        """
        cov = np.asarray(cov, dtype=float)
        b = np.full(self.n, 1.0 / self.n) if budgets is None else np.asarray(budgets, float)
        b = b / b.sum()
        self.L.value = cholesky_factor(cov)
        self.b.value = b
        if self._solve(self._rp) and self.y_rp.value is not None:
            w = self._clean(self.y_rp.value)
        else:
            w = 1.0 / np.sqrt(np.diag(cov)); w /= w.sum()

        if self._violates(w, w_prev):
            w = self._risk_parity_slsqp(cov, b, w0=w, w_prev=w_prev, tau=tau)
        return w

    def _violates(self, w, w_prev=None) -> bool:
        cs = self.cs
        if cs.max_weight is not None and (w > cs.max_weight + 1e-6).any():
            return True
        if cs.sector_matrix is not None and cs.max_sector_weight is not None:
            if (cs.sector_matrix @ w > cs.max_sector_weight + 1e-6).any():
                return True
        if cs.max_turnover is not None and w_prev is not None:
            if np.abs(w - w_prev).sum() > cs.max_turnover + 1e-6:
                return True
        return False

    def _risk_parity_slsqp(self, cov, b, w0, w_prev=None, tau=None) -> np.ndarray:
        cs = self.cs
        n = self.n

        def objective(w):
            port_var = w @ cov @ w
            rc = w * (cov @ w) / port_var
            return float(np.sum((rc - b) ** 2)) * 1e4

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        if cs.sector_matrix is not None and cs.max_sector_weight is not None:
            cons.append({"type": "ineq",
                         "fun": lambda w: cs.max_sector_weight - cs.sector_matrix @ w})
        if cs.max_turnover is not None and w_prev is not None:
            t = tau if tau is not None else cs.max_turnover
            cons.append({"type": "ineq", "fun": lambda w: t - np.abs(w - w_prev).sum()})
        ub = cs.max_weight if cs.max_weight is not None else 1.0
        lb = 0.0 if cs.long_only else -1.0
        res = minimize(objective, np.clip(w0, lb, ub), method="SLSQP",
                       bounds=[(lb, ub)] * n, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-12})
        if not res.success:
            logger.warning("risk parity SLSQP did not converge: %s", res.message)
        return self._clean(res.x)


# --------------------------------------------------------------------------- #
# Michaud resampled optimisation
# --------------------------------------------------------------------------- #
def resampled_weights(engine: ConvexEngine, est: Estimates, n_obs: int,
                      objective: str = "max_sharpe", n_sims: int = 100,
                      rf: float = 0.0, risk_aversion: float = 2.5,
                      w_prev=None, tau=None, seed: int = 0,
                      periods: int = 252) -> np.ndarray:
    """
    Michaud (1998) resampled portfolio for a single objective.

    Treat (mu_hat, Sigma_hat) as the truth, simulate `n_sims` histories of
    `n_obs` daily returns, re-estimate on each, optimise, and average the
    weights. Averaging is the regularisation: it smooths the corner
    solutions that individual noisy estimates produce.
    """
    rng = np.random.default_rng(seed)
    mu, cov = est.as_arrays()
    mu_d, cov_d = mu / periods, cov / periods
    L = cholesky_factor(cov_d)
    acc = np.zeros(engine.n)
    used = 0
    for _ in range(n_sims):
        z = rng.standard_normal((n_obs, engine.n))
        sim = z @ L.T + mu_d
        mu_s = sim.mean(axis=0) * periods
        cov_s = np.cov(sim, rowvar=False) * periods
        try:
            if objective == "max_sharpe":
                w = engine.max_sharpe(mu_s, cov_s, rf, w_prev, tau)
            elif objective == "min_variance":
                w = engine.min_variance(cov_s, w_prev, tau)
            elif objective == "mean_variance":
                w = engine.mean_variance(mu_s, cov_s, risk_aversion, w_prev, tau)
            else:
                raise ValueError(objective)
        except RuntimeError as exc:
            logger.debug("resample draw skipped: %s", exc)
            continue
        acc += w
        used += 1
    if used == 0:
        raise RuntimeError("All resampling draws failed")
    return acc / used


def resampled_frontier(engine: ConvexEngine, est: Estimates, n_obs: int,
                       n_points: int = 25, n_sims: int = 200, seed: int = 0,
                       periods: int = 252) -> pd.DataFrame:
    """
    Michaud resampled efficient frontier: average weights *by rank* across
    simulated frontiers, then evaluate the averaged weights on the original
    estimates. Always lies inside the classical frontier (it is not
    mean-variance efficient by construction; it is more stable).
    """
    rng = np.random.default_rng(seed)
    mu, cov = est.as_arrays()
    mu_d, cov_d = mu / periods, cov / periods
    L = cholesky_factor(cov_d)
    weight_sum = np.zeros((n_points, engine.n))
    counts = np.zeros(n_points)
    for _ in range(n_sims):
        sim = rng.standard_normal((n_obs, engine.n)) @ L.T + mu_d
        mu_s, cov_s = sim.mean(axis=0) * periods, np.cov(sim, rowvar=False) * periods
        try:
            fr = engine.efficient_frontier(mu_s, cov_s, n_points)
        except RuntimeError:
            continue
        if len(fr) < n_points:
            # interpolate weights onto a fixed rank grid
            idx = np.linspace(0, len(fr) - 1, n_points).round().astype(int)
            fr = fr.iloc[idx].reset_index(drop=True)
        W = np.vstack(fr["weights"].values)
        weight_sum += W
        counts += 1
    W_bar = weight_sum / np.maximum(counts, 1)[:, None]
    rows = [dict(ret=float(mu @ w), vol=float(np.sqrt(w @ cov @ w)), weights=w) for w in W_bar]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Simple allocations and diagnostics
# --------------------------------------------------------------------------- #
def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def inverse_volatility(cov) -> np.ndarray:
    iv = 1.0 / np.sqrt(np.diag(np.asarray(cov)))
    return iv / iv.sum()


def risk_contributions(w, cov) -> np.ndarray:
    """Fraction of portfolio variance attributable to each asset (sums to 1)."""
    w = np.asarray(w); cov = np.asarray(cov)
    port_var = w @ cov @ w
    return w * (cov @ w) / port_var


def portfolio_stats(w, mu, cov, rf: float = 0.0) -> Dict[str, float]:
    w = np.asarray(w); mu = np.asarray(mu); cov = np.asarray(cov)
    ret = float(mu @ w)
    vol = float(np.sqrt(w @ cov @ w))
    return dict(ret=ret, vol=vol, sharpe=(ret - rf) / vol if vol > 0 else np.nan,
                hhi=float(np.sum(w ** 2)), effective_n=float(1.0 / np.sum(w ** 2)),
                max_weight=float(w.max()))
