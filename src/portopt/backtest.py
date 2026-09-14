"""
Walk-forward backtest engine.

Timing convention (no lookahead)
--------------------------------
On each rebalance date t:
  1. The day's return is earned by the holdings carried into t.
  2. After the close, estimates are formed from returns up to and including t.
  3. New target weights are set; turnover is measured against the *drifted*
     holdings, not the previous target.
  4. From t+1 the new weights earn returns and drift with prices until the
     next rebalance.

Costs
-----
Costs are applied as `traded_notional x cost_bps`, where traded notional is
sum|w_new - w_drifted|. Gross returns and traded notional are stored
separately so net performance at any cost level can be computed without
re-running the optimiser (the weights do not depend on the cost level).

At inception every strategy buys 100% of its book, so traded = 1.0 on the
first rebalance for all of them; this keeps the comparison fair.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from . import black_litterman as bl
from . import estimators as est
from . import optimisers as opt
from .config import BACKTEST, CONSTRAINTS, MODEL, BacktestConfig, ConstraintConfig, ModelConfig
from .data import market_weights_at

logger = logging.getLogger(__name__)

try:                                    # progress bar is optional
    from tqdm.auto import tqdm
except ImportError:                     # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# --------------------------------------------------------------------------- #
# Strategy specification
# --------------------------------------------------------------------------- #
@dataclass
class StrategySpec:
    name: str
    kind: str                              # equal | min_variance | max_sharpe | resampled | black_litterman | risk_parity | mean_variance
    cov_method: str = "ledoit_wolf"        # sample | ledoit_wolf | const_corr | ewma
    mean_method: str = "historical"
    params: Dict = field(default_factory=dict)


DEFAULT_STRATEGIES: List[StrategySpec] = [
    StrategySpec("1/N Equal Weight", "equal"),
    StrategySpec("MVO Max-Sharpe (Sample)", "max_sharpe", cov_method="sample"),
    StrategySpec("MVO Max-Sharpe (Ledoit-Wolf)", "max_sharpe", cov_method="ledoit_wolf"),
    StrategySpec("MVO Max-Sharpe (LW + Shrunk Means)", "max_sharpe", cov_method="ledoit_wolf",
                 mean_method="shrunk"),
    StrategySpec("Michaud Resampled (Sample)", "resampled", cov_method="sample"),
    StrategySpec("Black-Litterman (Momentum views)", "black_litterman", cov_method="ledoit_wolf"),
    StrategySpec("Minimum Variance (Ledoit-Wolf)", "min_variance", cov_method="ledoit_wolf"),
    StrategySpec("Risk Parity (Ledoit-Wolf)", "risk_parity", cov_method="ledoit_wolf"),
]


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class BacktestResult:
    name: str
    gross_returns: pd.Series           # daily portfolio return before costs
    traded: pd.Series                  # sum|dw| on rebalance days, 0 elsewhere
    target_weights: pd.DataFrame       # weights set on each rebalance date
    holdings: pd.DataFrame             # daily drifted weights
    diagnostics: pd.DataFrame          # per-rebalance info (delta, views, ...)

    def net_returns(self, cost_bps: float) -> pd.Series:
        net = self.gross_returns - self.traded * cost_bps / 1e4
        net.name = self.name
        return net

    @property
    def rebalance_dates(self) -> pd.DatetimeIndex:
        return self.target_weights.index


# --------------------------------------------------------------------------- #
# Weight computation per strategy
# --------------------------------------------------------------------------- #
class WeightSolver:
    """Translates a StrategySpec into target weights at a rebalance date."""

    def __init__(self, engine: opt.ConvexEngine, assets: List[str],
                 caps: pd.DataFrame, rf: pd.Series,
                 model_cfg: ModelConfig = MODEL, periods: int = 252):
        self.engine = engine
        self.assets = assets
        self.caps = caps
        self.rf = rf
        self.cfg = model_cfg
        self.periods = periods
        self.n = len(assets)

    def _rf_ann(self, window: pd.DataFrame) -> float:
        return float(self.rf.reindex(window.index).fillna(0.0).mean() * self.periods)

    def solve(self, spec: StrategySpec, window: pd.DataFrame, w_prev: Optional[np.ndarray],
              date: pd.Timestamp, rebalance_idx: int) -> tuple[np.ndarray, Dict]:
        diag: Dict = {}
        tau = self.engine.cs.max_turnover
        kind = spec.kind

        if kind == "equal":
            return opt.equal_weight(self.n), diag

        e = est.estimate(window, spec.cov_method, spec.mean_method)
        mu, cov = e.as_arrays()
        rf_ann = self._rf_ann(window)
        diag["rf_ann"] = rf_ann
        if "shrinkage" in e.cov.attrs:
            diag["shrinkage"] = e.cov.attrs["shrinkage"]

        if kind == "min_variance":
            return self.engine.min_variance(cov, w_prev, tau), diag

        if kind == "max_sharpe":
            return self.engine.max_sharpe(mu, cov, rf_ann, w_prev, tau), diag

        if kind == "mean_variance":
            lam = spec.params.get("risk_aversion", self.cfg.bl_default_delta)
            return self.engine.mean_variance(mu, cov, lam, w_prev, tau), diag

        if kind == "risk_parity":
            return self.engine.risk_parity(cov, w_prev=w_prev, tau=tau), diag

        if kind == "resampled":
            n_sims = spec.params.get("n_sims", self.cfg.n_resample)
            w = opt.resampled_weights(self.engine, e, n_obs=len(window),
                                      objective=spec.params.get("objective", "max_sharpe"),
                                      n_sims=n_sims, rf=rf_ann, w_prev=w_prev, tau=tau,
                                      seed=rebalance_idx, periods=self.periods)
            return w, diag

        if kind == "black_litterman":
            w_mkt = market_weights_at(self.caps, date).reindex(self.assets)
            mkt_ret = (window * w_mkt).sum(axis=1)
            delta = bl.implied_risk_aversion(mkt_ret, self.rf, self.cfg, self.periods)
            views = []
            if spec.params.get("use_momentum_view", True):
                v = bl.momentum_view(window, self.cfg)
                if v is not None:
                    views.append(v)
            res = bl.black_litterman(e.cov, w_mkt, delta, views, tau=self.cfg.bl_tau)
            diag.update(delta=delta, n_views=len(views),
                        view=res.view_labels[0] if res.view_labels else "")
            # Canonical BL: mean-variance at the market's risk aversion on
            # posterior *excess* returns (rf cancels under full investment).
            w = self.engine.mean_variance(res.posterior_mu.values, res.posterior_cov.values,
                                          delta, w_prev, tau)
            return w, diag

        raise ValueError(f"Unknown strategy kind: {kind}")


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def rebalance_schedule(index: pd.DatetimeIndex, freq: str, first_valid: int) -> pd.DatetimeIndex:
    """Last trading day of each period, restricted to dates with enough history."""
    s = pd.Series(index, index=index)
    period_ends = s.resample(freq).last().dropna()
    dates = pd.DatetimeIndex(period_ends.values)
    return dates[dates >= index[first_valid]]


def _simulate_segment(w0: np.ndarray, seg: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Buy-and-hold simulation of weights w0 over a block of daily returns.
    Returns the daily portfolio return series and end-of-day drifted weights.
    """
    growth = (1.0 + seg.values).cumprod(axis=0)               # per-asset value path
    value = growth * w0                                       # dollar value per asset
    total = value.sum(axis=1)
    port_ret = np.empty(len(seg))
    port_ret[0] = total[0] - 1.0
    port_ret[1:] = total[1:] / total[:-1] - 1.0
    holdings = value / total[:, None]
    return (pd.Series(port_ret, index=seg.index),
            pd.DataFrame(holdings, index=seg.index, columns=seg.columns))


def run_backtest(returns: pd.DataFrame, rf: pd.Series, caps: pd.DataFrame,
                 sector_matrix: pd.DataFrame,
                 strategies: List[StrategySpec] = DEFAULT_STRATEGIES,
                 model_cfg: ModelConfig = MODEL,
                 bt_cfg: BacktestConfig = BACKTEST,
                 constraint_cfg: ConstraintConfig = CONSTRAINTS,
                 start: Optional[str] = None,
                 verbose: bool = True) -> Dict[str, BacktestResult]:
    """
    Run every strategy through the same walk-forward loop.

    `start` (optional) restricts the out-of-sample period; estimation still
    uses `lookback_days` of history before each rebalance date.
    """
    assets = list(returns.columns)
    n = len(assets)
    cs = opt.ConstraintSet.from_config(n, sector_matrix.reindex(columns=assets).values, constraint_cfg)
    engine = opt.ConvexEngine(cs)
    solver = WeightSolver(engine, assets, caps, rf, model_cfg)

    first_valid = model_cfg.lookback_days
    if start is not None:
        first_valid = max(first_valid, returns.index.searchsorted(pd.Timestamp(start)))
    rebal = rebalance_schedule(returns.index, bt_cfg.rebalance_freq, first_valid)
    if len(rebal) < 3:
        raise ValueError("Not enough data for a walk-forward backtest")
    logger.info("%d rebalances from %s to %s", len(rebal), rebal[0].date(), rebal[-1].date())

    results: Dict[str, BacktestResult] = {}
    for spec in strategies:
        gross_parts, hold_parts, traded_rows, tw_rows, diag_rows = [], [], [], [], []
        w_hold: Optional[np.ndarray] = None

        iterator = tqdm(range(len(rebal)), desc=spec.name, leave=False) if verbose else range(len(rebal))
        for i in iterator:
            t = rebal[i]
            loc = returns.index.get_loc(t)
            window = returns.iloc[max(0, loc - model_cfg.lookback_days + 1): loc + 1]

            try:
                w_new, diag = solver.solve(spec, window, w_hold, t, i)
            except Exception as exc:  # noqa: BLE001 - keep the backtest alive, log loudly
                logger.error("%s failed at %s (%s); holding previous weights", spec.name, t.date(), exc)
                w_new = w_hold if w_hold is not None else opt.equal_weight(n)
                diag = {"error": str(exc)}

            traded = 1.0 if w_hold is None else float(np.abs(w_new - w_hold).sum())
            traded_rows.append((t, traded))
            tw_rows.append(pd.Series(w_new, index=assets, name=t))
            diag_rows.append(pd.Series(diag, name=t))

            # Simulate from t+1 up to and including the next rebalance date.
            end_loc = returns.index.get_loc(rebal[i + 1]) if i + 1 < len(rebal) else len(returns) - 1
            seg = returns.iloc[loc + 1: end_loc + 1]
            if len(seg) == 0:
                w_hold = w_new
                continue
            seg_ret, seg_hold = _simulate_segment(w_new, seg)
            gross_parts.append(seg_ret)
            hold_parts.append(seg_hold)
            w_hold = seg_hold.iloc[-1].values

        gross = pd.concat(gross_parts)
        gross.name = spec.name
        traded_s = pd.Series(0.0, index=gross.index)
        for t, v in traded_rows:
            # cost is charged on the first trading day after the rebalance decision
            nxt = gross.index[gross.index.searchsorted(t, side="right")] if t < gross.index[-1] else None
            if nxt is not None:
                traded_s.loc[nxt] += v
        results[spec.name] = BacktestResult(
            name=spec.name, gross_returns=gross, traded=traded_s,
            target_weights=pd.DataFrame(tw_rows), holdings=pd.concat(hold_parts),
            diagnostics=pd.DataFrame(diag_rows))
    return results


def net_returns_frame(results: Dict[str, BacktestResult], cost_bps: float) -> pd.DataFrame:
    return pd.concat([r.net_returns(cost_bps) for r in results.values()], axis=1)
