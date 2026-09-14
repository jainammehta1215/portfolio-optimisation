"""Assemble notebooks/portfolio_optimisation.ipynb from the src/ modules."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src" / "portopt"
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
               "language_info": {"name": "python"},
               "colab": {"provenance": [], "toc_visible": True}}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# --------------------------------------------------------------------------- #
md(r"""
# Portfolio Optimisation: Mean-Variance, Black-Litterman and Robust Methods

**Markowitz is an error maximiser.** Feed it noisy expected returns and it concentrates the
portfolio into whatever it most overestimated. This notebook builds the classical optimiser,
demonstrates the failure quantitatively, then applies the fixes that are actually used at scale:

| Fix | What it addresses |
|---|---|
| Ledoit-Wolf covariance shrinkage | Noise and ill-conditioning in Σ̂ |
| Mean shrinkage | Noise in μ̂ (the larger problem, per Chopra & Ziemba 1993) |
| Michaud resampled frontier | Corner solutions; averages over estimation error |
| Black-Litterman | Replaces μ̂ with market-implied equilibrium + views |
| Realistic constraints | Position caps, sector caps, turnover limits |

Every method is then run through a **walk-forward backtest with transaction costs** and
benchmarked against equal weight (1/N), minimum variance and risk parity.

> **Reading guide.** Sections 1–3 write the package to disk and load data. Sections 4–7 are
> static experiments on one estimation window that show *why* things break. Sections 8–12
> are the out-of-sample horse race. Section 13 is the interpretation.

**Data:** Yahoo Finance (prices), FRED (3-month T-bill), yfinance shares outstanding (market-cap proxy).
**Universe:** 16 US large caps across 10 sectors, 2010 → present, monthly rebalancing, 2-year estimation window.
""")

md("## 1. Setup")
code(r"""
# Install what Colab does not ship with. cvxpy is usually present; yfinance often is not.
import importlib, subprocess, sys

def ensure(pkg, mod):
    try:
        importlib.import_module(mod); return
    except ImportError:
        pass
    for extra in ([], ["--break-system-packages"]):          # second form for PEP-668 systems
        if subprocess.call([sys.executable, "-m", "pip", "install", "-q", pkg] + extra) == 0:
            return
    print(f"WARNING: could not install {pkg}; some features may be unavailable")

for pkg, mod in [("yfinance", "yfinance"), ("cvxpy", "cvxpy"), ("scikit-learn", "sklearn"), ("tqdm", "tqdm")]:
    ensure(pkg, mod)
print("dependencies ready")
""")
code(r"""
import os, sys, warnings, logging, time
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Optional: persist data cache and outputs to Google Drive across Colab sessions.
USE_DRIVE = False
if USE_DRIVE:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
    os.chdir("/content/drive/MyDrive/portfolio-optimisation")

for d in ["src/portopt", "data/cache", "outputs/figures", "outputs/tables", "tests"]:
    os.makedirs(d, exist_ok=True)
if os.path.abspath("src") not in sys.path:
    sys.path.insert(0, os.path.abspath("src"))
print("working directory:", os.getcwd())
""")

md(r"""
## 2. Package source

The project is written as an importable package (`src/portopt/`) rather than one long script.
The cells below write each module to disk; later cells import from it. In a cloned repository
you would skip this section and `pip install -e .` instead.

```
portfolio-optimisation/
├── README.md
├── requirements.txt
├── notebooks/portfolio_optimisation.ipynb     ← this file
├── src/portopt/
│   ├── config.py           universe, sectors, constraints, hyper-parameters
│   ├── data.py             Yahoo/FRED loaders, market-cap proxy, quality gate
│   ├── estimators.py       sample / Ledoit-Wolf / constant-corr / EWMA covariance, mean shrinkage
│   ├── optimisers.py       compiled cvxpy engine, risk parity, Michaud resampling
│   ├── black_litterman.py  equilibrium, views, posterior
│   ├── backtest.py         walk-forward engine with drift, turnover, costs
│   ├── metrics.py          performance statistics, Sharpe difference test
│   ├── analysis.py         static experiments
│   └── plots.py            figures
├── tests/test_portopt.py   17 offline unit tests
├── data/cache/             downloaded prices and rates (git-ignored)
└── outputs/{figures,tables}/
```
""")

MODULE_ORDER = ["__init__", "config", "data", "estimators", "optimisers",
                "black_litterman", "backtest", "metrics", "analysis", "plots"]
MODULE_BLURB = {
    "__init__": "Package docstring and version.",
    "config": "Everything a researcher might change: universe, sector map, constraint levels, lookback, resampling draws, Black-Litterman τ and view parameters, rebalance frequency, cost grid.",
    "data": "Downloads with retry and caching, a data-quality gate that fails loudly, the FRED risk-free series, and a **point-in-time market-cap proxy** (current shares × historical price) so equilibrium weights do not embed knowledge of which companies later grew.",
    "estimators": "Annualised mean and covariance estimators. Ledoit-Wolf via scikit-learn plus a faithful port of the constant-correlation variant; James-Stein style mean shrinkage.",
    "optimisers": "All convex problems are compiled **once** with cvxpy's DPP and re-solved by updating parameters (2 ms per solve). The covariance enters through its Cholesky factor so the problems stay DPP-compliant. Max-Sharpe uses the Charnes-Cooper transform; risk parity uses Spinu's convex log-barrier formulation with a SLSQP fallback when caps bind; Michaud resampling averages optimal weights across simulated histories.",
    "black_litterman": "Reverse optimisation for Π, He-Litterman Ω, the posterior mean and covariance, and a **systematic 12-1 momentum view** so the backtest involves no discretionary (lookahead-prone) views.",
    "backtest": "Walk-forward loop with a strict timing convention: estimates use data up to the rebalance close, weights apply from the next day, holdings drift between rebalances, turnover is measured against drifted holdings. Gross returns and traded notional are stored separately so any cost level can be evaluated without re-optimising.",
    "metrics": "CAGR, volatility, Sharpe/Sortino against the daily T-bill series, drawdowns, VaR/CVaR, turnover, concentration, and a Jobson-Korkie/Memmel test for Sharpe differences.",
    "analysis": "The static experiments: frontier comparison, the estimation-error perturbation experiment, shrinkage vs window length, the Black-Litterman walkthrough, cost sensitivity and break-even costs.",
    "plots": "Consistent styling for every figure; functions return Figures and never call `plt.show()`.",
}
for name in MODULE_ORDER:
    src_text = (SRC / f"{name}.py").read_text()
    md(f"### `portopt/{name}.py`\n\n{MODULE_BLURB[name]}")
    code(f"%%writefile src/portopt/{name}.py\n{src_text}")

code(r"""
# Reload after writing so edits made above are picked up in the same session.
import importlib
import portopt
for m in ["config", "data", "estimators", "optimisers", "black_litterman", "backtest", "metrics", "analysis", "plots"]:
    importlib.reload(importlib.import_module(f"portopt.{m}"))
from portopt import data, estimators as est, optimisers as opt, black_litterman as bl
from portopt import backtest as bt, metrics as met, analysis as an, plots
from portopt.config import DATA, CONSTRAINTS, MODEL, BACKTEST, UNIVERSE, SECTOR_MAP

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plots.set_style()
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40); pd.set_option("display.precision", 4)
print("portopt", portopt.__version__, "| universe:", len(UNIVERSE), "assets |", CONSTRAINTS, "|", MODEL.lookback_days, "day lookback")
""")

md(r"""
> **Using your own companies:** edit `UNIVERSE` (and optionally `SECTOR_MAP`) in the `config.py`
> cell above and re-run from there. Non-US symbols need Yahoo's exchange suffix (`RELIANCE.NS`,
> `HSBA.L`, `7203.T`). The loader drops cross-exchange holidays, converts every price to USD
> through Yahoo FX crosses, and fetches shares outstanding and sectors automatically. Tickers
> listed after `DataConfig.start` are dropped with a message; move `start` later to keep them.
> With fewer than 5 names or 3 sectors, relax `max_weight` / `max_sector_weight`.
""")

# --------------------------------------------------------------------------- #
md(r"""
## 3. Data collection and quality checks

Adjusted closes already include dividends and splits, so simple returns on them are total
returns. The quality gate drops thin tickers, forward-fills isolated gaps (max 5 days) and
reports everything it touched. Set `USE_SYNTHETIC = True` to run the whole notebook offline
on simulated prices (clearly labelled as such).
""")
code(r"""
USE_SYNTHETIC = False   # True -> deterministic GBM data, no network needed

t0 = time.time()
D = data.load_all(use_synthetic=USE_SYNTHETIC)
prices, returns, rf, caps, sectors = D["prices"], D["returns"], D["rf"], D["caps"], D["sectors"]
assets = list(returns.columns)
print(f"loaded in {time.time()-t0:.1f}s | {prices.shape[0]} days x {prices.shape[1]} assets | "
      f"{prices.index[0].date()} -> {prices.index[-1].date()}")
print("data issues:", D["issues"] or "none")
print(f"risk-free (3m T-bill) mean since start: {rf.mean()*252:.2%} p.a.; latest: {rf.iloc[-1]*252:.2%}")
display(prices.tail(3).round(2))
""")
code(r"""
# Per-asset descriptive statistics over the full sample
desc = pd.DataFrame({
    "sector": pd.Series(SECTOR_MAP).reindex(assets),
    "ann_return": met.TRADING_DAYS * returns.mean(),
    "ann_vol": returns.std() * np.sqrt(met.TRADING_DAYS),
    "skew": returns.skew(), "kurtosis": returns.kurt(),
    "max_drawdown": returns.apply(met.max_drawdown),
    "cap_weight_today": data.market_weights_at(caps, prices.index[-1]).reindex(assets),
})
display(desc.style.format({"ann_return": "{:.1%}", "ann_vol": "{:.1%}", "skew": "{:.2f}",
                           "kurtosis": "{:.1f}", "max_drawdown": "{:.1%}", "cap_weight_today": "{:.1%}"}))
""")
code(r"""
fig, ax = plt.subplots(figsize=(11, 5))
(prices / prices.iloc[0]).plot(ax=ax, lw=1, logy=True, cmap="tab20")
ax.set_title("Normalised prices (log scale)"); ax.set_ylabel("Growth of 1"); ax.legend(ncol=4, fontsize=7)
plots.save(fig, "01_prices"); plt.show()

window = returns.iloc[-MODEL.lookback_days:]
fig = plots.plot_correlation(window.corr()); plots.save(fig, "02_correlation"); plt.show()
""")

# --------------------------------------------------------------------------- #
md(r"""
## 4. Inputs: sample vs shrinkage estimates

The optimiser engine is compiled once for the universe and constraint set. Then we compare
the sample and Ledoit-Wolf covariance on the most recent estimation window, and show how the
shrinkage intensity depends on the window length relative to the number of assets.
""")
code(r"""
cs = opt.ConstraintSet.from_config(len(assets), sectors.reindex(columns=assets).values)
engine = opt.ConvexEngine(cs)
rf_ann = float(rf.reindex(window.index).mean() * met.TRADING_DAYS)
w_mkt = data.market_weights_at(caps, window.index[-1]).reindex(assets)

e_sample = est.estimate(window, "sample", label="Sample")
e_lw = est.estimate(window, "ledoit_wolf", label="Ledoit-Wolf")
e_cc = est.estimate(window, "const_corr", label="Constant-correlation")
print(f"Estimation window: {window.index[0].date()} -> {window.index[-1].date()} ({len(window)} days), rf = {rf_ann:.2%}")
print(f"Ledoit-Wolf shrinkage intensity : {e_lw.cov.attrs['shrinkage']:.3f}")
print(f"Constant-corr shrinkage intensity: {e_cc.cov.attrs['shrinkage']:.3f}")
print(f"Condition number  sample: {est.condition_number(e_sample.cov):.1f}   LW: {est.condition_number(e_lw.cov):.1f}   CC: {est.condition_number(e_cc.cov):.1f}")

# Standard error of the mean vs the mean itself: this is the core problem.
se = np.sqrt(np.diag(e_sample.cov.values) / len(window))
inputs = pd.DataFrame({"mean (ann)": e_sample.mu, "std error": se, "t-stat": e_sample.mu / se,
                       "vol (ann)": np.sqrt(np.diag(e_sample.cov.values))}, index=assets)
display(inputs.style.format({"mean (ann)": "{:.1%}", "std error": "{:.1%}", "t-stat": "{:.2f}", "vol (ann)": "{:.1%}"}))
print(f"{(inputs['t-stat'].abs() < 2).sum()} of {len(assets)} expected returns are statistically indistinguishable from zero at 2 years of data.")
""")
code(r"""
shrink_tbl = an.shrinkage_vs_window(returns)
display(shrink_tbl.round(3))
fig = plots.plot_shrinkage_vs_window(shrink_tbl); plots.save(fig, "03_shrinkage_vs_window"); plt.show()
""")
md(r"""
**Read this before the horse race.** With 16 assets and 504 days, N/T ≈ 0.03 and the optimal
shrinkage intensity is small. Covariance shrinkage helps most when N/T is large (hundreds of
assets, short windows, or the instant after a regime change). In this configuration the
instability in mean-variance weights is dominated by the *means*, whose standard errors are
comparable to the means themselves. Keep this in mind when the sample-covariance and
Ledoit-Wolf MVO results look almost identical later: that is the correct outcome, not a bug.
""")

# --------------------------------------------------------------------------- #
md(r"""
## 5. Efficient frontiers: classical, shrinkage, resampled

All three curves are drawn in the coordinates of the *sample* estimates so they are comparable.
The Michaud frontier averages optimal weights by rank across simulated histories drawn from
N(μ̂, Σ̂); it lies inside the classical frontier by construction (it is not mean-variance
efficient; it is more stable).
""")
code(r"""
t0 = time.time()
fc = an.frontier_comparison(window, engine, n_points=30, n_resample=MODEL.n_resample_static)
print(f"frontiers computed in {time.time()-t0:.1f}s ({MODEL.n_resample_static} resampled draws)")
fig = plots.plot_frontiers(fc, rf_ann); plots.save(fig, "04_frontiers"); plt.show()
""")
code(r"""
# Max-Sharpe portfolio under each input set, same constraints
mu_s, cov_s = e_sample.as_arrays()
w_tbl = pd.DataFrame({
    "MVO (sample)": engine.max_sharpe(mu_s, cov_s, rf_ann),
    "MVO (Ledoit-Wolf)": engine.max_sharpe(mu_s, e_lw.cov.values, rf_ann),
    "MVO (LW + shrunk means)": engine.max_sharpe(est.mean_shrunk(window).values, e_lw.cov.values, rf_ann),
    "Michaud resampled": opt.resampled_weights(engine, e_sample, len(window), n_sims=MODEL.n_resample_static, rf=rf_ann),
    "Min variance (LW)": engine.min_variance(e_lw.cov.values),
    "Risk parity (LW)": engine.risk_parity(e_lw.cov.values),
    "1/N": opt.equal_weight(len(assets)),
}, index=assets)
stats = pd.DataFrame({c: opt.portfolio_stats(w_tbl[c].values, mu_s, cov_s, rf_ann) for c in w_tbl}).T
display(stats.style.format({"ret": "{:.1%}", "vol": "{:.1%}", "sharpe": "{:.2f}", "hhi": "{:.3f}", "effective_n": "{:.1f}", "max_weight": "{:.1%}"}))
fig = plots.plot_weight_comparison(w_tbl, "Max-Sharpe / benchmark weights on the latest window"); plots.save(fig, "05_weights_static"); plt.show()
""")

# --------------------------------------------------------------------------- #
md(r"""
## 6. The instability experiment

Perturb each expected return by noise of one **standard error** (σᵢ/√T) — exactly the
estimation error a practitioner faces — and re-optimise. Repeat 200 times per method.
Methods that ignore μ (min variance, risk parity) are constant by construction; the
interesting comparison is among the methods that do use it.
""")
code(r"""
t0 = time.time()
instab_tbl, draws = an.instability_experiment(window, engine, w_mkt, rf_ann, n_trials=200, n_resample=MODEL.n_resample)
print(f"instability experiment: {time.time()-t0:.1f}s")
display(instab_tbl.style.format({"Mean L1 distance from base": "{:.3f}", "Mean per-asset weight std": "{:.2%}",
                                 "Max per-asset weight std": "{:.2%}", "Avg effective N": "{:.1f}", "Base Sharpe (in-sample)": "{:.2f}"})
        .background_gradient(subset=["Mean L1 distance from base"], cmap="Reds"))
instab_tbl.to_csv("outputs/tables/instability.csv")
fig = plots.plot_instability(draws, assets); plots.save(fig, "06_instability"); plt.show()
""")
md(r"""
The classical optimiser moves roughly 10–12% of the portfolio in response to noise that is
statistically indistinguishable from nothing. Each fix reduces that: covariance shrinkage
slightly, mean shrinkage substantially, and Black-Litterman and Michaud the most among the
methods that still use return information. The cost is a lower *in-sample* Sharpe — which is
exactly the in-sample Sharpe you should not believe.
""")

# --------------------------------------------------------------------------- #
md(r"""
## 7. Black-Litterman walkthrough

1. **Reverse optimisation.** Π = δ Σ w_mkt gives the excess returns at which the market-cap
   portfolio is optimal. δ is estimated from the market's own Sharpe/variance and clipped to
   a sane range.
2. **Views.** Here: the systematic 12-1 momentum spread (top 4 beat bottom 4 by 5% p.a.),
   with He-Litterman uncertainty Ω = diag(P τΣ Pᵀ).
3. **Posterior.** Precision-weighted blend of Π and the views; then constrained mean-variance
   at δ.

The unconstrained textbook solution w* = (δΣ_BL)⁻¹ μ_BL is shown for reference — note it
contains short positions and does not sum to one, which is why practice uses the constrained
version.
""")
code(r"""
blw = an.bl_walkthrough(window, engine, w_mkt, rf)
post = blw["posterior"]
print(f"market-implied risk aversion delta = {blw['delta']:.2f}   (bounds {MODEL.bl_delta_bounds}, tau = {MODEL.bl_tau})")
for v in post.view_labels: print("view:", v)
display(post.summary().style.format("{:.2%}").background_gradient(subset=["shift"], cmap="RdBu_r", vmin=-0.05, vmax=0.05))
fig = plots.plot_bl_returns(post); plots.save(fig, "07_bl_prior_posterior"); plt.show()
""")
code(r"""
print("Unconstrained BL weights  (sum = %.2f, min = %.2f, max = %.2f)" % (blw["unconstrained"].sum(), blw["unconstrained"].min(), blw["unconstrained"].max()))
display(blw["weights"].style.format("{:.1%}").background_gradient(cmap="Blues", axis=None))
fig = plots.plot_weight_comparison(blw["weights"], "Black-Litterman: market cap vs prior vs posterior vs historical-mean MVO"); plots.save(fig, "08_bl_weights"); plt.show()
""")
code(r"""
# Expressing a discretionary view: "NVDA outperforms MSFT by 8% p.a." at two confidence levels.
if all(t in assets for t in ["NVDA", "MSFT"]):
    view = bl.relative_view(assets, ["NVDA"], ["MSFT"], 0.08)
    for conf, label in [(np.array([4.0]), "low confidence (Omega x4)"), (np.array([0.25]), "high confidence (Omega /4)")]:
        r = bl.black_litterman(e_lw.cov, w_mkt, blw["delta"], [view], tau=MODEL.bl_tau, view_confidence=conf)
        w = engine.mean_variance(r.posterior_mu.values, r.posterior_cov.values, blw["delta"])
        print(f"{label:28s} posterior NVDA-MSFT spread = {r.posterior_mu['NVDA']-r.posterior_mu['MSFT']:+.2%}   "
              f"weights NVDA {w[assets.index('NVDA')]:.1%}  MSFT {w[assets.index('MSFT')]:.1%}")
""")

# --------------------------------------------------------------------------- #
md(r"""
## 8. Walk-forward backtest

Eight strategies, identical constraints, identical calendar. Monthly rebalancing on the last
trading day; estimates use the trailing 504 days up to and including that close; weights apply
from the next day and drift until the next rebalance. Turnover is measured against the
*drifted* holdings and capped at 50% (sum of absolute weight changes).

The Michaud strategy re-solves 100 resampled problems at every rebalance, which is most of the
runtime (about a minute in total thanks to the compiled solver).
""")
code(r"""
for s in bt.DEFAULT_STRATEGIES:
    print(f"{s.name:38s} kind={s.kind:16s} cov={s.cov_method:12s} mean={s.mean_method}")
""")
code(r"""
t0 = time.time()
results = bt.run_backtest(returns, rf, caps, sectors, strategies=bt.DEFAULT_STRATEGIES,
                          model_cfg=MODEL, bt_cfg=BACKTEST, constraint_cfg=CONSTRAINTS, verbose=True)
print(f"backtest complete in {time.time()-t0:.0f}s")
first = next(iter(results.values()))
print(f"out-of-sample: {first.gross_returns.index[0].date()} -> {first.gross_returns.index[-1].date()}, "
      f"{len(first.target_weights)} rebalances")

COST = BACKTEST.cost_bps
net = bt.net_returns_frame(results, COST)
gross = pd.concat([r.gross_returns for r in results.values()], axis=1)
traded = pd.concat([r.traded.rename(k) for k, r in results.items()], axis=1)
weights = {k: r.target_weights for k, r in results.items()}
""")

md("## 9. Performance metrics")
code(r"""
summary = met.summary_table(net, rf, gross, traded, weights)
summary.to_csv("outputs/tables/summary.csv")
display(met.format_summary(summary))
fig = plots.plot_summary_heatmap(summary, ["CAGR", "Volatility", "Sharpe", "Sortino", "Max Drawdown", "Calmar",
                                           "Annual Turnover", "Cost Drag (CAGR)", "Avg Effective N", "Weight Instability"])
plots.save(fig, "09_scorecard"); plt.show()
""")
code(r"""
fig = plots.plot_cumulative(net); plots.save(fig, "10_cumulative"); plt.show()
fig = plots.plot_drawdowns(net); plots.save(fig, "11_drawdowns"); plt.show()
fig = plots.plot_rolling_sharpe(net, rf); plots.save(fig, "12_rolling_sharpe"); plt.show()
""")
code(r"""
annual = met.annual_returns(net)
display(annual.style.format("{:.1%}").background_gradient(cmap="RdYlGn", axis=1))
fig = plots.plot_annual_returns(annual); plots.save(fig, "13_annual_returns"); plt.show()
""")
code(r"""
show = [k for k in results if k != "1/N Equal Weight"]
fig = plots.plot_allocation_grid(results, show); plots.save(fig, "14_allocations"); plt.show()
fig = plots.plot_turnover(traded); plots.save(fig, "15_turnover"); plt.show()
""")

# --------------------------------------------------------------------------- #
md(r"""
## 10. Transaction-cost sensitivity

Weights do not depend on the cost level, so net returns at any cost are
`gross − traded × cost` on the stored series. The break-even column is the cost per unit
traded at which each strategy's net Sharpe drops below 1/N's net Sharpe at the *same* cost
(0 = never ahead of 1/N; ∞ = ahead across the whole 0–200 bps grid).
""")
code(r"""
sens = an.cost_sensitivity(results, BACKTEST.cost_grid_bps, rf)
be = an.breakeven_cost_bps(results, "1/N Equal Weight", rf)
tbl = sens["sharpe"].copy(); tbl.columns = [f"Sharpe @ {c:g} bps" for c in tbl.columns]
tbl["break-even bps vs 1/N"] = be
display(tbl.style.format("{:.2f}").background_gradient(cmap="RdYlGn", axis=None, subset=tbl.columns[:-1]))
tbl.to_csv("outputs/tables/cost_sensitivity.csv")
fig = plots.plot_cost_sensitivity(sens["sharpe"]); plots.save(fig, "16_cost_sensitivity"); plt.show()
""")

md(r"""
## 11. Is any of this statistically significant?

Jobson-Korkie test (Memmel correction) on monthly returns for H₀: Sharpe(strategy) = Sharpe(1/N).
With ~175 monthly observations and strategy correlations above 0.8, the power to detect
differences of 0.1–0.2 in annual Sharpe is low. That limitation is itself a finding.
""")
code(r"""
sig = an.significance_vs_benchmark(net, rf, "1/N Equal Weight")
display(sig.style.format({"diff_ann": "{:+.3f}", "z": "{:+.2f}", "p_value": "{:.3f}", "corr": "{:.2f}", "n_periods": "{:.0f}"})
        .background_gradient(subset=["p_value"], cmap="Greens_r", vmin=0, vmax=0.5))
sig.to_csv("outputs/tables/significance_vs_1N.csv")
""")

md(r"""
## 12. Where the risk sits, and what each method holds today

Risk contribution (share of portfolio variance) under each strategy's latest target weights,
using the Ledoit-Wolf covariance from the latest window. Then the current target weights —
the practical output of the pipeline.
""")
code(r"""
rc = an.risk_contribution_table(results, e_lw.cov)
fig = plots.plot_risk_contributions(rc); plots.save(fig, "17_risk_contributions"); plt.show()

latest = pd.DataFrame({k: r.target_weights.iloc[-1] for k, r in results.items()})
latest.index.name = f"target weights as of {first.target_weights.index[-1].date()}"
display(latest.style.format("{:.1%}").background_gradient(cmap="Blues", axis=None))
latest.to_csv("outputs/tables/latest_target_weights.csv")
sector_exp = sectors.reindex(columns=assets) @ latest
display(sector_exp.style.format("{:.1%}"))
""")
code(r"""
# Persist everything a reviewer would want to re-analyse
net.to_csv("outputs/tables/net_returns_daily.csv")
gross.to_csv("outputs/tables/gross_returns_daily.csv")
traded.to_csv("outputs/tables/traded_notional_daily.csv")
for k, r in results.items():
    safe = "".join(ch if ch.isalnum() else "_" for ch in k)
    r.target_weights.to_csv(f"outputs/tables/weights_{safe}.csv")
    if not r.diagnostics.empty:
        r.diagnostics.to_csv(f"outputs/tables/diagnostics_{safe}.csv")
print("saved:", sorted(os.listdir("outputs/tables"))[:8], "...")
print("figures:", len(os.listdir("outputs/figures")))
""")

# --------------------------------------------------------------------------- #
md(r"""
## 13. Interpretation

Write your own conclusions from the tables above; the template below is what the results
have typically shown on this universe and will need editing after each run.

**What the experiments show**

1. **Estimation error dominates.** On a 2-year window most expected returns have |t| < 2.
   The classical optimiser reacts to that noise by moving ~10% of the book (Section 6).
2. **Covariance shrinkage is not the main fix here.** With N/T ≈ 0.03 the optimal Ledoit-Wolf
   intensity is under 10%, so sample-covariance and LW mean-variance behave nearly identically.
   Shrinkage earns its keep at high N/T. Mean shrinkage, Black-Litterman and resampling reduce
   instability far more because they act on μ.
3. **1/N is a serious benchmark.** Out of sample and net of costs, the optimised strategies
   land within a narrow Sharpe band around equal weight, with turnover three to five times
   higher. None of the differences is significant at conventional levels (Section 11). This is
   consistent with DeMiguel, Garlappi & Uppal (2009).
4. **Costs decide the ranking.** At realistic institutional costs (5–10 bps) the ordering
   barely moves; at 25–50 bps the high-turnover optimisers fall below 1/N (Section 10).
   Turnover, not in-sample Sharpe, is the first number to look at.
5. **Minimum variance is a different bet.** It delivered the lowest volatility and a lower
   Sharpe over a period dominated by a mega-cap growth rally; its relative performance is
   regime-dependent and its sector concentration is the highest of any method.
6. **Risk parity's edge is structural, not return-driven.** Near-1/N turnover, no dependence
   on μ, and the most even risk contributions; its Sharpe tracks 1/N closely.

**Honest caveats**

- One universe, one sample period, survivorship in the universe choice (these 16 are today's
  large caps). Robustness across universes and start dates is the obvious next step.
- The momentum view in Black-Litterman is a demonstration of *systematic* view construction,
  not a claim that momentum plus BL beats the market.
- Costs are modelled as proportional; market impact for a large book would penalise the
  high-turnover methods more.
- The Sharpe test assumes approximately iid monthly returns; a HAC-robust test (Ledoit & Wolf
  2008) is the correct upgrade.
""")

md(r"""
## 14. Potential upgrades

| Upgrade | Why it matters |
|---|---|
| Factor-model covariance (PCA / Fama-French residual structure) | Scales to hundreds of assets where sample and even LW estimates degrade |
| Robust optimisation (Goldfarb-Iyengar uncertainty sets, CVaR objective) | Optimises the worst case over an estimation-error set rather than the point estimate |
| Hierarchical Risk Parity (López de Prado 2016) | Avoids covariance inversion entirely; compare stability with ERC |
| Idzorek confidence mapping for Black-Litterman | Express view confidence as a percentage instead of an Ω multiplier |
| Regime-conditional estimation (HMM / Markov switching) | Let μ and Σ depend on the detected regime rather than a fixed trailing window |
| Multi-period optimisation with transaction-cost-aware objectives (Gârleanu-Pedersen) | Trade toward the target gradually instead of a hard turnover cap |
| Ledoit-Wolf (2008) HAC Sharpe test, bootstrap confidence bands, White's reality check | Proper multiple-testing control across the strategy set |
| Broader universes (S&P 500 constituents with point-in-time membership) and rolling start dates | The single most important robustness check for any allocation study |
| Live paper-trading loop with a broker API and daily risk report | Turns the research code into an operating system |
""")

nb.cells = cells
out = ROOT / "notebooks" / "portfolio_optimisation.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
