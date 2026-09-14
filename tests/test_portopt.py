"""
Offline test-suite (synthetic data, no network).

Run with:  PYTHONPATH=src pytest -q tests/
"""
import numpy as np
import pandas as pd
import pytest

from portopt import backtest as bt
from portopt import black_litterman as bl
from portopt import data
from portopt import estimators as est
from portopt import metrics as met
from portopt import optimisers as opt
from portopt.config import ConstraintConfig, ModelConfig, BacktestConfig

TICKERS = ["AAPL", "MSFT", "JPM", "GS", "JNJ", "XOM", "PG", "CAT", "NEE", "LIN"]


@pytest.fixture(scope="module")
def synthetic():
    px = data.synthetic_prices(TICKERS, n_days=1500, seed=3)
    rets = data.compute_returns(px)
    sectors = data.sector_matrix(TICKERS)
    shares = pd.Series({t: 1e9 * (i + 1) for i, t in enumerate(TICKERS)})
    caps = data.market_cap_proxy(px, shares)
    rf = pd.Series(0.02 / 252, index=rets.index)
    return dict(prices=px, returns=rets, sectors=sectors, caps=caps, rf=rf)


@pytest.fixture(scope="module")
def engine(synthetic):
    cs = opt.ConstraintSet(n=len(TICKERS), long_only=True, max_weight=0.25,
                           sector_matrix=synthetic["sectors"].values,
                           max_sector_weight=0.40, max_turnover=0.30)
    return opt.ConvexEngine(cs)


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def test_covariances_are_psd_and_shrinkage_in_unit_interval(synthetic):
    w = synthetic["returns"].iloc[-252:]
    for fn in (est.cov_sample, est.cov_ledoit_wolf, est.cov_constant_correlation):
        cov = fn(w)
        assert np.all(np.linalg.eigvalsh(cov.values) > 0)
        if "shrinkage" in cov.attrs:
            assert 0.0 <= cov.attrs["shrinkage"] <= 1.0


def test_ledoit_wolf_reduces_condition_number(synthetic):
    w = synthetic["returns"].iloc[-126:]
    assert est.condition_number(est.cov_ledoit_wolf(w)) < est.condition_number(est.cov_sample(w))


def test_mean_shrinkage_full_collapses_to_grand_mean(synthetic):
    w = synthetic["returns"].iloc[-252:]
    mu = est.mean_shrunk(w, shrink=1.0)
    assert np.allclose(mu.values, mu.values[0])


# --------------------------------------------------------------------------- #
# Optimisers
# --------------------------------------------------------------------------- #
def _check_constraints(w, engine, w_prev=None):
    cs = engine.cs
    assert abs(w.sum() - 1) < 1e-6
    assert (w >= -1e-7).all()
    assert (w <= cs.max_weight + 1e-6).all()
    assert (cs.sector_matrix @ w <= cs.max_sector_weight + 1e-6).all()
    if w_prev is not None:
        assert np.abs(w - w_prev).sum() <= cs.max_turnover + 1e-6


def test_min_variance_has_lowest_variance_among_feasible(synthetic, engine):
    w = synthetic["returns"].iloc[-504:]
    e = est.estimate(w, "ledoit_wolf")
    mu, cov = e.as_arrays()
    w_mv = engine.min_variance(cov)
    _check_constraints(w_mv, engine)
    for other in (engine.max_sharpe(mu, cov, 0.02), engine.risk_parity(cov), opt.equal_weight(len(mu))):
        assert w_mv @ cov @ w_mv <= other @ cov @ other + 1e-9


def test_max_sharpe_respects_constraints_and_beats_equal_weight_in_sample(synthetic, engine):
    w = synthetic["returns"].iloc[-504:]
    mu, cov = est.estimate(w, "sample").as_arrays()
    w_ms = engine.max_sharpe(mu, cov, 0.02)
    _check_constraints(w_ms, engine)
    sr = lambda x: (mu @ x - 0.02) / np.sqrt(x @ cov @ x)
    assert sr(w_ms) >= sr(opt.equal_weight(len(mu))) - 1e-9


def test_turnover_constraint_binds(synthetic, engine):
    w = synthetic["returns"].iloc[-504:]
    mu, cov = est.estimate(w, "sample").as_arrays()
    w_prev = opt.equal_weight(len(mu))
    w_new = engine.max_sharpe(mu, cov, 0.02, w_prev=w_prev, tau=0.10)
    assert np.abs(w_new - w_prev).sum() <= 0.10 + 1e-6
    _check_constraints(w_new, engine)


def test_risk_parity_equalises_contributions(synthetic):
    w = synthetic["returns"].iloc[-504:]
    cov = est.cov_ledoit_wolf(w).values
    eng = opt.ConvexEngine(opt.ConstraintSet(n=len(TICKERS)).unconstrained())
    w_rp = eng.risk_parity(cov)
    rc = opt.risk_contributions(w_rp, cov)
    assert np.allclose(rc, 1 / len(TICKERS), atol=1e-4)


def test_efficient_frontier_is_monotone(synthetic, engine):
    w = synthetic["returns"].iloc[-504:]
    mu, cov = est.estimate(w, "ledoit_wolf").as_arrays()
    fr = engine.efficient_frontier(mu, cov, n_points=12)
    assert len(fr) >= 10
    assert (np.diff(fr["ret"]) >= -1e-9).all()
    assert (np.diff(fr["vol"]) >= -1e-7).all()


def test_resampled_weights_are_feasible_and_smoother(synthetic, engine):
    w = synthetic["returns"].iloc[-504:]
    e = est.estimate(w, "sample")
    mu, cov = e.as_arrays()
    w_rs = opt.resampled_weights(engine, e, n_obs=504, n_sims=30, rf=0.02, seed=1)
    w_ms = engine.max_sharpe(mu, cov, 0.02)
    _check_constraints(w_rs, engine)
    assert np.sum(w_rs ** 2) <= np.sum(w_ms ** 2) + 1e-9      # lower HHI (more diversified)


# --------------------------------------------------------------------------- #
# Black-Litterman
# --------------------------------------------------------------------------- #
def test_bl_without_views_recovers_market_weights(synthetic):
    w = synthetic["returns"].iloc[-504:]
    cov = est.cov_ledoit_wolf(w)
    w_mkt = data.market_weights_at(synthetic["caps"], w.index[-1])
    res = bl.black_litterman(cov, w_mkt, delta=3.0, views=[])
    w_unc = bl.unconstrained_bl_weights(res)
    assert np.allclose(w_unc.values, w_mkt.reindex(cov.index).values, atol=1e-8)


def test_bl_view_moves_posterior_in_view_direction(synthetic):
    w = synthetic["returns"].iloc[-504:]
    cov = est.cov_ledoit_wolf(w)
    w_mkt = data.market_weights_at(synthetic["caps"], w.index[-1])
    prior = bl.black_litterman(cov, w_mkt, 3.0, []).prior
    view = bl.relative_view(TICKERS, ["AAPL"], ["XOM"], 0.10)
    post = bl.black_litterman(cov, w_mkt, 3.0, [view]).posterior_mu
    assert (post["AAPL"] - post["XOM"]) > (prior["AAPL"] - prior["XOM"])


def test_bl_confident_view_is_pulled_closer_to_q(synthetic):
    w = synthetic["returns"].iloc[-504:]
    cov = est.cov_ledoit_wolf(w)
    w_mkt = data.market_weights_at(synthetic["caps"], w.index[-1])
    view = bl.absolute_view(TICKERS, "JPM", 0.20)
    loose = bl.black_litterman(cov, w_mkt, 3.0, [view], view_confidence=np.array([10.0]))
    tight = bl.black_litterman(cov, w_mkt, 3.0, [view], view_confidence=np.array([0.01]))
    assert abs(tight.posterior_mu["JPM"] - 0.20) < abs(loose.posterior_mu["JPM"] - 0.20)


def test_momentum_view_has_zero_net_exposure(synthetic):
    v = bl.momentum_view(synthetic["returns"].iloc[-400:], ModelConfig(bl_top_bottom_k=3))
    assert v is not None
    assert abs(v[0].sum()) < 1e-12


# --------------------------------------------------------------------------- #
# Backtest accounting
# --------------------------------------------------------------------------- #
def test_equal_weight_backtest_matches_manual_rebalance(synthetic):
    rets = synthetic["returns"]
    cfg_m = ModelConfig(lookback_days=252, n_resample=5)
    res = bt.run_backtest(rets, synthetic["rf"], synthetic["caps"], synthetic["sectors"],
                          strategies=[bt.StrategySpec("EW", "equal")],
                          model_cfg=cfg_m, bt_cfg=BacktestConfig(rebalance_freq="ME"),
                          constraint_cfg=ConstraintConfig(max_weight=0.25, max_sector_weight=0.4),
                          verbose=False)["EW"]
    # holdings must always sum to one and drift between rebalances
    assert np.allclose(res.holdings.sum(axis=1), 1.0)
    # traded notional after inception must be > 0 (drift) and < 2
    tr = res.traded[res.traded > 0].iloc[1:]
    assert (tr > 0).all() and (tr < 2).all()
    # first day return equals equal-weight return of the segment start
    first = res.gross_returns.index[0]
    assert abs(res.gross_returns.loc[first] - rets.loc[first].mean()) < 1e-12


def test_net_returns_decrease_with_cost(synthetic):
    rets = synthetic["returns"]
    res = bt.run_backtest(rets, synthetic["rf"], synthetic["caps"], synthetic["sectors"],
                          strategies=[bt.StrategySpec("MV", "min_variance")],
                          model_cfg=ModelConfig(lookback_days=252),
                          constraint_cfg=ConstraintConfig(max_weight=0.25, max_sector_weight=0.4),
                          verbose=False)["MV"]
    assert met.cagr(res.net_returns(0)) > met.cagr(res.net_returns(50)) > met.cagr(res.net_returns(200))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_drawdown_and_cagr_basics():
    r = pd.Series([0.10, -0.50, 0.20], index=pd.bdate_range("2020-01-01", periods=3))
    assert abs(met.max_drawdown(r) - (-0.5)) < 1e-12
    wealth = float((1 + r).prod())
    assert abs(met.cagr(r) - (wealth ** (252 / 3) - 1)) < 1e-12


def test_sharpe_difference_test_symmetric():
    idx = pd.bdate_range("2015-01-01", periods=1500)
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0.0005, 0.01, 1500), index=idx)
    b = pd.Series(rng.normal(0.0003, 0.01, 1500), index=idx)
    t1 = met.sharpe_difference_test(a, b)
    t2 = met.sharpe_difference_test(b, a)
    assert abs(t1["z"] + t2["z"]) < 1e-10
    assert 0 <= t1["p_value"] <= 1
