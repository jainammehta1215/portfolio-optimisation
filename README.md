# Portfolio Optimisation: Mean-Variance, Black-Litterman and Robust Methods

**Asset-allocation track · Project 3 of 6** · the allocation engine of the series · see
[Project 1, Risk & Performance Analytics](https://github.com/jainammehta1215/portfolio-risk-analytics)
for reporting on any portfolio this produces

A research-grade Python framework that builds the classical Markowitz optimiser, demonstrates
quantitatively why it fails, applies the fixes used in institutional practice, and tests
everything out of sample with transaction costs against the benchmarks that are actually hard
to beat.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jainammehta1215/portfolio-optimisation/blob/main/notebooks/portfolio_optimisation.ipynb)
[![tests](https://github.com/jainammehta1215/portfolio-optimisation/actions/workflows/tests.yml/badge.svg)](https://github.com/jainammehta1215/portfolio-optimisation/actions)
`python 3.10+` · `cvxpy` · `pandas` · `scikit-learn` · 17 offline tests

---

## 1. Project overview

Mean-variance optimisation takes two inputs, expected returns and a covariance matrix, and
returns the portfolio with the best risk-return trade-off. In practice it returns the portfolio
that most aggressively exploits the *errors* in those inputs. This project:

* implements the classical optimiser with realistic constraints (long-only, position caps,
  sector caps, turnover limits) on a compiled convex engine;
* shows the instability directly by perturbing expected returns by one standard error and
  watching the weights move;
* applies four corrections: Ledoit-Wolf and constant-correlation covariance shrinkage,
  mean shrinkage, Michaud resampled frontiers, and Black-Litterman with market-implied
  equilibrium returns and systematic views;
* benchmarks all of them in a 15-year monthly walk-forward backtest, net of costs, against
  equal weight (1/N), minimum variance and risk parity;
* tests whether the Sharpe differences are statistically distinguishable from zero.

The result, on this universe and period, matches the published literature: after costs, none
of the optimised strategies significantly beats 1/N, and turnover is the variable that decides
the ranking as costs rise. The notebook is written so that finding is a feature, not an
embarrassment.

## 2. Real-world finance use case

* **Multi-asset and equity allocation teams** run exactly this comparison before adopting any
  optimiser. Black-Litterman (Goldman Sachs origin) and shrinkage covariance are default
  production choices; 1/N is the benchmark every proposal has to beat net of costs.
* **Manager due diligence** uses the instability experiment to ask whether a manager's
  allocation would survive a plausible change in their own return forecasts.
* **Risk teams** use the risk-contribution decomposition and turnover accounting to check
  that a strategy's realised behaviour matches its mandate. The full monthly reporting pack for
  any set of weights produced here (drawdown episodes, stress scenarios, calm-vs-stress
  correlation, HTML dashboard) is
  [Project 1](https://github.com/jainammehta1215/portfolio-risk-analytics) in this track.
* **Anyone building a systematic strategy** needs the backtest timing convention here
  (estimate on data to the close, trade at the close, drift until next rebalance, measure
  turnover against drifted holdings) to avoid the lookahead and cost errors that make retail
  backtests meaningless.

## 3. System architecture

```
                 ┌────────────────────────────────────────────────────────┐
                 │                      config.py                         │
                 │  universe · sectors · constraints · hyper-parameters   │
                 └────────────────────────────────────────────────────────┘
                                          │
      ┌───────────────┐        ┌──────────▼──────────┐        ┌──────────────────┐
      │ Yahoo Finance │──────► │      data.py        │ ◄──────│      FRED        │
      │ prices, shares│        │ cache · QC gate ·   │        │  DGS3MO (rf)     │
      └───────────────┘        │ point-in-time caps  │        └──────────────────┘
                               └──────────┬──────────┘
                                          │ returns, rf, caps, sector matrix
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
          ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
          │  estimators.py   │  │ black_litterman  │  │    optimisers.py     │
          │ μ̂ (hist/shrunk)  │─►│ Π · views · μ_BL │─►│ compiled cvxpy (DPP) │
          │ Σ̂ (sample/LW/CC) │  └──────────────────┘  │ MVO · max-Sharpe ·   │
          └──────────────────┘                        │ frontier · RP ·      │
                                                      │ Michaud resampling   │
                                                      └──────────┬───────────┘
                                                                 │ weights
                            ┌────────────────────┐    ┌──────────▼───────────┐
                            │    analysis.py     │    │     backtest.py      │
                            │ static experiments │    │ walk-forward · drift │
                            └─────────┬──────────┘    │ turnover · costs     │
                                      │               └──────────┬───────────┘
                                      ▼                          ▼
                            ┌──────────────────────────────────────────────┐
                            │           metrics.py  ·  plots.py            │
                            │  scorecard · cost curves · significance      │
                            └──────────────────────────────────────────────┘
```

Key design decisions:

* **Compile once, solve many.** Every convex program is built once per (universe, constraint
  set) with cvxpy's Disciplined Parametrized Programming. The covariance enters via its
  Cholesky factor so `sum_squares(Lᵀw)` stays DPP-compliant. Re-solves take ~2 ms, which is
  what makes 100-draw Michaud resampling at every one of 177 rebalances feasible.
* **Gross and traded stored separately.** Weights do not depend on the cost level, so the
  whole cost-sensitivity analysis is arithmetic on stored series.
* **No discretionary views in the backtest.** Black-Litterman uses a rules-based 12-1 momentum
  view so the historical test cannot be contaminated by hindsight.
* **Point-in-time market caps.** Current shares outstanding × historical price, not today's
  market caps applied to 2012.

## 4. Required APIs and data sources

| Source | What | Access |
|---|---|---|
| Yahoo Finance via `yfinance` | Adjusted daily closes 2010→; shares outstanding | Free, no key |
| FRED (`fredgraph.csv` endpoint) | `DGS3MO` 3-month Treasury constant maturity | Free, no key |
| Hard-coded fallback | Approximate shares outstanding if the live fetch fails | In `config.py` |
| Synthetic generator | Deterministic GBM with factor correlation for offline runs/tests | `data.synthetic_prices` |

No paid data is required. `USE_SYNTHETIC = True` in the notebook runs everything offline.

## 5. Required Python libraries

```
numpy pandas scipy scikit-learn cvxpy matplotlib yfinance requests tqdm
```
Dev: `pytest nbformat nbconvert ipykernel`. See `requirements.txt`. Solvers used: CLARABEL
(default), OSQP, SCS as fallbacks; all ship with cvxpy.

## 6. Folder / file structure

```
portfolio-optimisation/
├── README.md
├── requirements.txt
├── pyproject.toml
├── build_notebook.py                 assembles the notebook from src/ (keeps them in sync)
├── notebooks/
│   └── portfolio_optimisation.ipynb  self-contained Colab notebook (writes src/ itself)
├── src/portopt/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── estimators.py
│   ├── optimisers.py
│   ├── black_litterman.py
│   ├── backtest.py
│   ├── metrics.py
│   ├── analysis.py
│   └── plots.py
├── tests/
│   └── test_portopt.py               17 tests, synthetic data, ~2 s
├── data/cache/                       downloaded prices and rates (git-ignored)
└── outputs/
    ├── figures/                      17 PNGs produced by the notebook
    └── tables/                       summary, cost sensitivity, weights, diagnostics (CSV)
```

In Colab the notebook writes `src/portopt/` itself with `%%writefile` cells, so the same
structure exists whether you clone the repo or open the notebook cold.

## 7. Step-by-step build guide

1. **Config first.** Fix the universe, sector map and constraint levels before writing any
   estimator. Everything else reads from `config.py`.
2. **Data layer with a quality gate.** Download with retry and cache; fail loudly on duplicated
   dates, non-positive prices or thin history; forward-fill only isolated gaps. Build the
   risk-free series from FRED and align it to the trading calendar.
3. **Estimators.** Annualised sample mean/covariance, then Ledoit-Wolf (scikit-learn) and a
   port of the constant-correlation variant. Add mean shrinkage. Verify PSD and shrinkage
   ∈ [0, 1] in tests.
4. **Convex engine.** Define the constraint set once, compile four problems (utility MVO,
   target-return, max-Sharpe via Charnes-Cooper, max-return LP) and the log-barrier risk
   parity problem. Check `is_dcp(dpp=True)` for each.
5. **Michaud resampling** on top of the engine; **Black-Litterman** as a separate pure-math
   module with tests for the no-view identity and view direction.
6. **Backtest engine.** Rebalance schedule → per-rebalance window → weights → segment
   simulation with drift → turnover vs drifted holdings. Store gross and traded separately.
7. **Metrics and analysis.** Summary table, cost sensitivity, break-even, Sharpe test,
   instability experiment, frontier comparison, BL walkthrough.
8. **Plots**, each returning a Figure.
9. **Tests**, then **notebook** assembled from the modules by `build_notebook.py` so code and
   notebook cannot drift apart.
10. Execute the notebook top to bottom in a clean directory before publishing.

## 8. Data collection pipeline

`data.load_all()` returns prices, returns, risk-free, cap proxy, sector matrix and an issues
log. Prices: `yf.download(auto_adjust=True)` → tz-naive index → CSV cache keyed on ticker set
and date range. Risk-free: FRED CSV → percent yield → daily rate
`(1 + y/100)^(1/252) − 1` → forward-filled to the trading calendar → cached. Shares
outstanding: per-ticker `fast_info` with fallback. Cap proxy: shares × price panel. All network
calls retry three times with back-off; failures degrade with a logged error rather than a
silent default (the risk-free fallback is 0% with an explicit warning that Sharpe ratios will
be overstated).

## 9. Data cleaning and feature engineering

* Drop tickers with < 98% date coverage; forward-fill gaps ≤ 5 days; drop leading rows until
  every asset has a price.
* Simple daily returns (portfolio returns aggregate linearly; log returns do not).
* Estimation "features" per rebalance: annualised μ̂, Σ̂ under each estimator, standard errors
  of the mean (used in the instability experiment), 12-1 momentum ranks (Black-Litterman
  view), cap-weighted market return (implied δ).
* A guard flags any |daily return| > 50% for inspection.

## 10. Core models / algorithms

| Model | Formulation | Notes |
|---|---|---|
| Mean-variance utility | max μᵀw − (λ/2) wᵀΣw | μ pre-scaled by 1/λ to stay DPP |
| Minimum variance | min wᵀΣw | special case μ = 0 |
| Max Sharpe | min yᵀΣy s.t. (μ−r_f)ᵀy = 1, Σy = κ, constraints scaled by κ; w = y/κ | falls back to min-var if no positive excess return is feasible |
| Efficient frontier | min wᵀΣw s.t. μᵀw ≥ r, r swept from min-var return to max-return LP | |
| Risk parity | min ½yᵀΣy − Σ bᵢ log yᵢ; w = y/Σy (Spinu 2013) | exact ERC; SLSQP fallback when caps bind |
| Ledoit-Wolf shrinkage | Σ̂ = (1−α)S + α·target, analytic α | scaled-identity target (sklearn) and constant-correlation target (port of covCor.m) |
| Mean shrinkage | μ̃ = (1−s)μ̂ + s·mean(μ̂) | James-Stein style |
| Michaud resampling | simulate N(μ̂, Σ̂) histories, re-estimate, optimise, average weights (by rank for the frontier) | convexity of the feasible set keeps averages feasible |
| Black-Litterman | Π = δΣw_mkt; μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹[(τΣ)⁻¹Π + PᵀΩ⁻¹Q]; Σ_BL = Σ + [·]⁻¹ | He-Litterman Ω; δ from market Sharpe/variance, clipped to [1, 5]; systematic momentum view |
| Constraints | Σw = 1, w ≥ 0, w ≤ 20%, sector ≤ 35%, ‖w − w_prev‖₁ ≤ 50% | all optional |
| Sharpe test | Jobson-Korkie with Memmel correction on monthly returns | HAC version listed as an upgrade |

## 11. Visualisations and dashboard components

1. Normalised price history and correlation heatmap of the estimation window
2. Shrinkage intensity and condition number vs window length
3. Efficient frontiers: classical vs Ledoit-Wolf vs Michaud, with individual assets
4. Weight comparison bars across methods on one window
5. Instability box-plot grid (weight dispersion under one-standard-error perturbations)
6. Black-Litterman prior vs posterior returns; market vs prior vs posterior vs historical-mean weights
7. Strategy scorecard heat-map (rank-coloured)
8. Out-of-sample growth of $1 (log), drawdowns, rolling 3-year Sharpe, calendar-year returns
9. Stacked allocation-over-time panels for each strategy; annual turnover bars
10. Sharpe vs transaction-cost curves with break-even table
11. Risk-contribution bars under latest weights; current target weights and sector exposures

All figures are saved to `outputs/figures/`; all tables to `outputs/tables/`.

## 12. Performance metrics

CAGR · annualised volatility · Sharpe and Sortino (excess over the daily T-bill series) ·
maximum drawdown and duration · Calmar · daily historical VaR/CVaR 95% · skew · cost drag
(gross − net CAGR) · annual traded notional · average effective N (1/HHI) · average maximum
weight · weight instability (mean L1 change between consecutive targets) · Jobson-Korkie z and
p-value vs 1/N · same-cost break-even cost vs 1/N.

Typical output on the default universe (2012-02 → 2026-09, 10 bps):

| Strategy | CAGR | Vol | Sharpe | MaxDD | Turnover p.a. | Eff. N |
|---|---|---|---|---|---|---|
| 1/N Equal Weight | 20.6% | 16.3% | 1.12 | −33.7% | 59% | 16.0 |
| MVO Max-Sharpe (Sample) | 21.2% | 17.5% | 1.09 | −30.1% | 310% | 5.9 |
| MVO Max-Sharpe (Ledoit-Wolf) | 21.3% | 17.5% | 1.09 | −30.2% | 309% | 6.0 |
| MVO (LW + shrunk means) | 19.4% | 16.1% | 1.08 | −30.5% | 292% | 6.5 |
| Michaud Resampled | 20.8% | 16.6% | 1.12 | −30.3% | 180% | 8.7 |
| Black-Litterman (momentum views) | 20.2% | 17.4% | 1.04 | −28.6% | 258% | 9.2 |
| Minimum Variance (LW) | 12.7% | 14.1% | 0.80 | −35.3% | 115% | 7.1 |
| Risk Parity (LW) | 18.4% | 15.2% | 1.08 | −33.9% | 60% | 14.6 |

Only minimum variance differs from 1/N at p < 0.05, and in the wrong direction. Numbers will
move with the data end-date; the qualitative picture has been stable.

## 13. Final deliverables

* `notebooks/portfolio_optimisation.ipynb` — self-contained, runs top to bottom in Colab in
  roughly 3 minutes (about 45 s of which is the resampled strategy).
* `src/portopt/` — importable package; `pip install -e .`
* `tests/test_portopt.py` — 17 tests, offline, `PYTHONPATH=src pytest -q tests/`
* `outputs/figures/` — 17 publication-quality PNGs
* `outputs/tables/` — summary, cost sensitivity, significance, per-strategy weights and
  diagnostics, daily gross/net return series
* This README

## 14. Potential upgrades

* **Factor-model covariance** (statistical or Fama-French) for universes of hundreds of names.
* **Robust optimisation** with ellipsoidal uncertainty sets on μ (Goldfarb-Iyengar) or a CVaR
  objective.
* **Hierarchical Risk Parity** for a no-inversion allocation and a direct stability comparison
  with ERC.
* **Idzorek's confidence mapping** in Black-Litterman; multiple simultaneous views with
  correlated Ω.
* **Regime-conditional inputs** via HMM / Markov switching instead of a fixed trailing window.
* **Multi-period, cost-aware optimisation** (Gârleanu-Pedersen) replacing the hard turnover cap.
* **Ledoit-Wolf (2008) HAC Sharpe test**, bootstrap confidence bands and White's reality check
  across the whole strategy set.
* **Point-in-time S&P 500 membership** and rolling start dates, the single most important
  robustness check.
* **Live loop**: broker API, daily rebalancing report, drift alerts.

---

## Running it

```bash
git clone https://github.com/jainammehta1215/portfolio-optimisation.git && cd portfolio-optimisation
pip install -r requirements.txt
PYTHONPATH=src pytest -q tests/           # 17 passed
python build_notebook.py                  # regenerate the notebook from src/
jupyter nbconvert --to notebook --execute notebooks/portfolio_optimisation.ipynb
```

Or open the notebook in Colab and run all cells; it writes `src/portopt/` itself.

## Using your own companies

The universe is a list in `src/portopt/config.py` (or the `config.py` cell in the notebook). Any
symbol Yahoo Finance carries works, including non-US listings with their exchange suffix:
`RELIANCE.NS` (NSE India), `HSBA.L` (London), `7203.T` (Tokyo), `EMAAR.AE` (Dubai), `SAP.DE`
(Xetra). Look symbols up at finance.yahoo.com if unsure.

```python
UNIVERSE = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "HSBA.L", "7203.T", "ASML", "TSM"]
SECTOR_MAP = {}          # optional: anything missing is looked up on Yahoo automatically
```

What the pipeline handles for you:

* **Different exchange calendars.** Days on which fewer than half the universe traded are
  dropped (returns compound correctly across the gap); single-exchange holidays are
  forward-filled.
* **Different currencies.** Every price is converted to `DataConfig.base_currency` (USD by
  default) through Yahoo FX crosses before returns, market caps or anything else is computed.
  Pence and cents quotes are scaled to their major unit first.
* **Shares outstanding and sectors** are fetched per ticker; if Yahoo has no shares figure, add
  it to `FALLBACK_SHARES_OUTSTANDING_BN`.

What you must check yourself:

* **History.** A ticker listed after `DataConfig.start` is dropped with a message telling you
  its first price date; move `start` later to keep it.
* **Constraint feasibility.** The 20% position cap needs at least 5 names; the 35% sector cap
  needs at least 3 sectors. With fewer, raise the caps or set them to `None`.
* **Risk-free rate.** The default is the US 3-month T-bill, which is right for a USD base
  currency. For another base, change `fred_series` (or supply your own series).
* **Results are universe-specific.** On a 12-name India/UK/Japan/US list the optimisers beat
  1/N because two semiconductor names dominated the decade. That is not evidence the
  optimiser works; it is evidence the period had a winner. Run several universes and start
  dates before drawing conclusions.

## Asset-allocation track

Six projects that build one capability each and share a reporting layer. Completed ones are linked.

| # | Project | What it adds |
|---|---|---|
| 1 | [Portfolio Risk & Performance Analytics Dashboard](https://github.com/jainammehta1215/portfolio-risk-analytics) | Where the risk sits: Euler decomposition, benchmark-relative statistics, stress scenarios, HTML dashboard |
| 2 | [Multi-Factor Exposure Analyser (Fama-French)](https://github.com/jainammehta1215/factor-exposure-analyser) | Why assets co-move: factor betas, alpha after factor adjustment, style drift |
| 3 | **Portfolio Optimisation: Mean-Variance, Black-Litterman and Robust Methods** (this repo) | What weights to hold: Markowitz and its fixes, tested out of sample net of costs |
| 4 | [Risk Parity and Hierarchical Risk Parity](https://github.com/jainammehta1215/risk-parity-hrp) | Allocating by risk instead of capital; clustering instead of matrix inversion |
| 5 | [Macro Nowcasting and Recession Probability](https://github.com/jainammehta1215/macro-nowcasting) | The regime the allocation lives in: yield-curve probit, dynamic factor model, real-time vintages |
| 6 | Yield Curve Construction and Fixed-Income Immunisation | The rates side: bootstrapping, Nelson-Siegel, key-rate durations, liability matching |

A master repository will consolidate all six with a shared core once the track is complete.

## References

Black & Litterman (1992) *Global Portfolio Optimization* · He & Litterman (1999) *The
Intuition Behind Black-Litterman Model Portfolios* · Ledoit & Wolf (2003, 2004) covariance
shrinkage · Michaud (1998) *Efficient Asset Management* · Chopra & Ziemba (1993) on the
relative impact of errors in means vs variances · DeMiguel, Garlappi & Uppal (2009) *Optimal
Versus Naive Diversification* · Spinu (2013) risk parity convex formulation · Jobson & Korkie
(1981), Memmel (2003) Sharpe ratio tests.
