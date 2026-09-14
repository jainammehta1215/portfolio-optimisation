"""
portopt - Portfolio optimisation: mean-variance, Black-Litterman and robust methods.

Modules
-------
config           universe, constraints, model and backtest settings
data             Yahoo Finance / FRED loaders, market-cap proxy, quality checks
estimators       mean and covariance estimators (sample, Ledoit-Wolf, EWMA)
optimisers       compiled cvxpy engine: MVO, min-var, max-Sharpe, frontier, risk parity, Michaud
black_litterman  equilibrium returns, views, posterior
backtest         walk-forward engine with drift, turnover and costs
metrics          performance statistics and Sharpe difference test
analysis         static experiments (frontiers, instability, BL walkthrough, cost sensitivity)
plots            figures
"""
__version__ = "1.0.0"
