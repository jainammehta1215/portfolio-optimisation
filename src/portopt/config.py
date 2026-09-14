"""
Central configuration for the portfolio optimisation project.

Everything that a researcher might want to change lives here: the asset
universe, sector map, sample window, constraint set and backtest settings.
Nothing in the other modules hard-codes these values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
# Sixteen large-cap US equities spanning ten GICS sectors. Individual stocks
# rather than ETFs so that (a) market caps exist for Black-Litterman
# equilibrium weights and (b) sector caps are meaningful constraints.
UNIVERSE: List[str] = [
    "AAPL", "MSFT", "NVDA",          # Information Technology
    "AMZN", "HD",                    # Consumer Discretionary
    "JPM", "GS",                     # Financials
    "JNJ", "UNH",                    # Health Care
    "XOM", "CVX",                    # Energy
    "PG", "KO",                      # Consumer Staples
    "CAT",                           # Industrials
    "NEE",                           # Utilities
    "LIN",                           # Materials
]

SECTOR_MAP: Dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AMZN": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "JPM": "Financials", "GS": "Financials",
    "JNJ": "Health Care", "UNH": "Health Care",
    "XOM": "Energy", "CVX": "Energy",
    "PG": "Consumer Staples", "KO": "Consumer Staples",
    "CAT": "Industrials",
    "NEE": "Utilities",
    "LIN": "Materials",
}

# Fallback shares outstanding (billions) used ONLY if the live fetch fails.
# Approximate values; they are used to build a historical market-cap proxy
# (shares x price) so only relative magnitudes matter for equilibrium weights.
FALLBACK_SHARES_OUTSTANDING_BN: Dict[str, float] = {
    "AAPL": 14.6, "MSFT": 7.4, "NVDA": 24.4, "AMZN": 10.6, "HD": 1.0,
    "JPM": 2.8, "GS": 0.31, "JNJ": 2.4, "UNH": 0.91, "XOM": 4.3,
    "CVX": 1.75, "PG": 2.35, "KO": 4.3, "CAT": 0.47, "NEE": 2.06, "LIN": 0.47,
}

MARKET_PROXY: str = "SPY"          # used for CAPM-style diagnostics only


# --------------------------------------------------------------------------- #
# Dates and data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    start: str = "2010-01-01"
    end: Optional[str] = None          # None -> today
    cache_dir: str = "data/cache"
    fred_series: str = "DGS3MO"        # 3-month T-bill, constant maturity, %
    trading_days: int = 252
    min_history_fraction: float = 0.98  # drop tickers with < 98% of the dates


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #
@dataclass
class ConstraintConfig:
    long_only: bool = True
    max_weight: float = 0.20           # per-position cap
    max_sector_weight: float = 0.35    # per-sector cap
    max_turnover: float = 0.50         # sum |w_new - w_old| per rebalance (2 = unconstrained)


# --------------------------------------------------------------------------- #
# Estimation and model hyper-parameters
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    lookback_days: int = 504           # 2 years of daily data for mu / Sigma
    n_resample: int = 100              # Michaud bootstrap draws (backtest)
    n_resample_static: int = 300       # Michaud draws for the static frontier
    bl_tau: float = 0.05               # Black-Litterman scaling of prior uncertainty
    bl_default_delta: float = 2.5      # risk aversion if market-implied estimate is unusable
    bl_delta_bounds: tuple = (1.0, 5.0)
    bl_momentum_lookback: int = 252    # 12-1 momentum window (days)
    bl_momentum_skip: int = 21         # skip most recent month
    bl_momentum_view_ann: float = 0.05 # top basket beats bottom basket by 5% p.a.
    bl_top_bottom_k: int = 4           # basket size for the relative view
    risk_parity_max_iter: int = 500


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
@dataclass
class BacktestConfig:
    rebalance_freq: str = "ME"        # pandas offset alias: month end
    cost_bps: float = 10.0            # cost per unit of traded notional
    cost_grid_bps: List[float] = field(default_factory=lambda: [0, 5, 10, 25, 50, 100])
    initial_weights: str = "equal"    # starting point for the turnover constraint


DATA = DataConfig()
CONSTRAINTS = ConstraintConfig()
MODEL = ModelConfig()
BACKTEST = BacktestConfig()
