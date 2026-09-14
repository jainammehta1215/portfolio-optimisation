"""
Static experiments that sit alongside the walk-forward backtest.

These are the "why" sections of the project: they show the mechanism behind
Markowitz instability and how each fix changes the picture, on a single
estimation window where everything can be inspected.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import black_litterman as bl
from . import estimators as est
from . import metrics as met
from . import optimisers as opt
from .backtest import BacktestResult
from .config import MODEL, ModelConfig


# --------------------------------------------------------------------------- #
# 1. Efficient frontiers: sample vs shrinkage vs resampled
# --------------------------------------------------------------------------- #
def frontier_comparison(window: pd.DataFrame, engine: opt.ConvexEngine,
                        n_points: int = 30, n_resample: int = MODEL.n_resample_static,
                        seed: int = 0) -> Dict[str, object]:
    e_s = est.estimate(window, "sample", label="Sample")
    e_lw = est.estimate(window, "ledoit_wolf", label="Ledoit-Wolf")
    fr_sample = engine.efficient_frontier(*e_s.as_arrays(), n_points)
    fr_lw = engine.efficient_frontier(*e_lw.as_arrays(), n_points)
    fr_res = opt.resampled_frontier(engine, e_s, n_obs=len(window),
                                    n_points=n_points, n_sims=n_resample, seed=seed)
    # Evaluate the LW frontier weights on the *sample* estimates so that all
    # three curves are drawn in the same (mu_hat, Sigma_hat) coordinates.
    mu_s, cov_s = e_s.as_arrays()
    fr_lw_on_sample = pd.DataFrame([
        dict(ret=float(mu_s @ w), vol=float(np.sqrt(w @ cov_s @ w)), weights=w)
        for w in fr_lw["weights"]])
    assets = pd.DataFrame({"ret": e_s.mu, "vol": np.sqrt(np.diag(e_s.cov))})
    return dict(sample=fr_sample, ledoit_wolf=fr_lw_on_sample, resampled=fr_res,
                assets=assets, est_sample=e_s, est_lw=e_lw)


# --------------------------------------------------------------------------- #
# 2. Weight instability under estimation-error-sized perturbations
# --------------------------------------------------------------------------- #
def instability_experiment(window: pd.DataFrame, engine: opt.ConvexEngine,
                           caps_weights: pd.Series, rf_ann: float,
                           n_trials: int = 200, seed: int = 1,
                           n_resample: int = 50,
                           cfg: ModelConfig = MODEL) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Perturb the expected-return vector by noise with the *standard error of
    the mean* (sigma_i / sqrt(T)) - exactly the estimation error a
    practitioner faces - and re-optimise. Repeat n_trials times per method.

    Returns a summary table and the raw weight draws per method.
    """
    rng = np.random.default_rng(seed)
    T = len(window)
    e_s = est.estimate(window, "sample")
    e_lw = est.estimate(window, "ledoit_wolf")
    mu, cov_s = e_s.as_arrays()
    cov_lw = e_lw.cov.values
    se = np.sqrt(np.diag(cov_s) / T)                     # annualised SE of mean
    assets = list(window.columns)
    n = len(assets)

    # Black-Litterman: the perturbed historical means enter as one absolute
    # view per asset with He-Litterman uncertainty (Omega = tau * sigma_i^2),
    # so the posterior is a precision-weighted blend of equilibrium and the
    # noisy estimate. Same max-Sharpe objective as the other MVO variants so
    # that only the *inputs* differ across panels.
    delta = cfg.bl_default_delta
    def bl_weights(mu_pert):
        views = [bl.absolute_view(assets, a, float(m)) for a, m in zip(assets, mu_pert - rf_ann)]
        res = bl.black_litterman(e_lw.cov, caps_weights, delta, views, tau=cfg.bl_tau)
        return engine.max_sharpe(res.posterior_mu.values, res.posterior_cov.values, 0.0)

    methods = {
        "MVO (Sample cov)": lambda m: engine.max_sharpe(m, cov_s, rf_ann),
        "MVO (Ledoit-Wolf cov)": lambda m: engine.max_sharpe(m, cov_lw, rf_ann),
        "MVO (LW cov + shrunk means)": lambda m: engine.max_sharpe(0.5 * m + 0.5 * m.mean(), cov_lw, rf_ann),
        "Michaud Resampled": lambda m: opt.resampled_weights(
            engine, est.Estimates(pd.Series(m, index=assets), e_s.cov), T,
            "max_sharpe", n_sims=n_resample, rf=rf_ann, seed=seed),   # fixed seed: only mu varies
        "Black-Litterman (views = estimates)": bl_weights,
        "Minimum Variance (LW)": lambda m: engine.min_variance(cov_lw),
        "Risk Parity (LW)": lambda m: engine.risk_parity(cov_lw),
    }

    draws: Dict[str, np.ndarray] = {k: np.zeros((n_trials, n)) for k in methods}
    base: Dict[str, np.ndarray] = {k: f(mu) for k, f in methods.items()}
    for i in range(n_trials):
        mu_p = mu + rng.standard_normal(n) * se
        for k, f in methods.items():
            draws[k][i] = f(mu_p)

    rows = []
    for k in methods:
        W = draws[k]
        rows.append({
            "Method": k,
            "Mean L1 distance from base": float(np.abs(W - base[k]).sum(axis=1).mean()),
            "Mean per-asset weight std": float(W.std(axis=0).mean()),
            "Max per-asset weight std": float(W.std(axis=0).max()),
            "Avg effective N": float((1 / (W ** 2).sum(axis=1)).mean()),
            "Base Sharpe (in-sample)": opt.portfolio_stats(base[k], mu, cov_s, rf_ann)["sharpe"],
        })
    return pd.DataFrame(rows).set_index("Method"), draws


# --------------------------------------------------------------------------- #
# 3. Shrinkage intensity and conditioning versus window length
# --------------------------------------------------------------------------- #
def shrinkage_vs_window(returns: pd.DataFrame,
                        windows: List[int] = (63, 126, 252, 504, 756, 1260)) -> pd.DataFrame:
    rows = []
    for T in windows:
        w = returns.iloc[-T:]
        cs_ = est.cov_sample(w)
        lw = est.cov_ledoit_wolf(w)
        rows.append(dict(window_days=T, lw_shrinkage=lw.attrs["shrinkage"],
                         cond_sample=est.condition_number(cs_),
                         cond_ledoit_wolf=est.condition_number(lw),
                         n_over_t=returns.shape[1] / T))
    return pd.DataFrame(rows).set_index("window_days")


# --------------------------------------------------------------------------- #
# 4. Black-Litterman walkthrough on one window
# --------------------------------------------------------------------------- #
def bl_walkthrough(window: pd.DataFrame, engine: opt.ConvexEngine,
                   caps_weights: pd.Series, rf: pd.Series,
                   views: Optional[List[Tuple[np.ndarray, float, str]]] = None,
                   cfg: ModelConfig = MODEL) -> Dict[str, object]:
    e_lw = est.estimate(window, "ledoit_wolf")
    assets = list(window.columns)
    mkt_ret = (window * caps_weights.reindex(assets)).sum(axis=1)
    delta = bl.implied_risk_aversion(mkt_ret, rf, cfg)
    if views is None:
        v = bl.momentum_view(window, cfg)
        views = [v] if v is not None else []
    prior_only = bl.black_litterman(e_lw.cov, caps_weights, delta, [], tau=cfg.bl_tau)
    posterior = bl.black_litterman(e_lw.cov, caps_weights, delta, views, tau=cfg.bl_tau)

    weights = pd.DataFrame({
        "Market cap": caps_weights.reindex(assets),
        "Prior (constrained MVO on Pi)": engine.mean_variance(
            prior_only.posterior_mu.values, prior_only.posterior_cov.values, delta),
        "Posterior (constrained MVO on mu_BL)": engine.mean_variance(
            posterior.posterior_mu.values, posterior.posterior_cov.values, delta),
        "Historical-mean MVO (for contrast)": engine.mean_variance(
            e_lw.mu.values, e_lw.cov.values, delta),
    }, index=assets)
    return dict(delta=delta, prior=prior_only, posterior=posterior, weights=weights,
                unconstrained=bl.unconstrained_bl_weights(posterior), views=views)


# --------------------------------------------------------------------------- #
# 5. Cost sensitivity and significance tests on backtest output
# --------------------------------------------------------------------------- #
def cost_sensitivity(results: Dict[str, BacktestResult], cost_grid: List[float],
                     rf: pd.Series) -> Dict[str, pd.DataFrame]:
    sharpe_tab, cagr_tab = {}, {}
    for c in cost_grid:
        sharpe_tab[c] = {k: met.sharpe(r.net_returns(c), rf) for k, r in results.items()}
        cagr_tab[c] = {k: met.cagr(r.net_returns(c)) for k, r in results.items()}
    return dict(sharpe=pd.DataFrame(sharpe_tab), cagr=pd.DataFrame(cagr_tab))


def breakeven_cost_bps(results: Dict[str, BacktestResult], benchmark: str,
                       rf: pd.Series, grid: np.ndarray = np.arange(0, 201, 1)) -> pd.Series:
    """
    Lowest cost level (bps per unit traded) at which a strategy's net Sharpe
    drops below the benchmark's net Sharpe *at the same cost*. Both sides pay
    costs; higher-turnover strategies lose ground faster. 0 means the strategy
    never led the benchmark; inf means it led across the whole grid.
    """
    bench = np.array([met.sharpe(results[benchmark].net_returns(c), rf) for c in grid])
    out = {}
    for k, r in results.items():
        if k == benchmark:
            out[k] = np.nan
            continue
        srs = np.array([met.sharpe(r.net_returns(c), rf) for c in grid])
        below = np.where(srs < bench)[0]
        out[k] = float(grid[below[0]]) if len(below) else np.inf
    return pd.Series(out, name="breakeven_bps_vs_" + benchmark)


def significance_vs_benchmark(net: pd.DataFrame, rf: pd.Series, benchmark: str) -> pd.DataFrame:
    rows = {}
    for col in net.columns:
        if col == benchmark:
            continue
        rows[col] = met.sharpe_difference_test(net[col], net[benchmark], rf)
    df = pd.DataFrame(rows).T
    return df[["diff_ann", "z", "p_value", "corr", "n_periods"]].astype(float)


def risk_contribution_table(results: Dict[str, BacktestResult], cov: pd.DataFrame) -> pd.DataFrame:
    """Risk contribution of each asset under each strategy's latest target weights."""
    out = {}
    for k, r in results.items():
        w = r.target_weights.iloc[-1].reindex(cov.index).values
        out[k] = opt.risk_contributions(w, cov.values)
    return pd.DataFrame(out, index=cov.index)
