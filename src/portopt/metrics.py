"""
Performance and risk metrics for daily return series.

All annualisation uses 252 trading days. Sharpe and Sortino are computed on
returns in excess of the daily risk-free series, not a constant.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Single-series metrics
# --------------------------------------------------------------------------- #
def cagr(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    growth = float((1 + r).prod())
    return growth ** (periods / len(r)) - 1.0


def ann_vol(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(r.std(ddof=1) * np.sqrt(periods))


def sharpe(r: pd.Series, rf: Optional[pd.Series] = None, periods: int = TRADING_DAYS) -> float:
    ex = r - (rf.reindex(r.index).fillna(0.0) if rf is not None else 0.0)
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan


def sortino(r: pd.Series, rf: Optional[pd.Series] = None, periods: int = TRADING_DAYS) -> float:
    ex = r - (rf.reindex(r.index).fillna(0.0) if rf is not None else 0.0)
    downside = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2))
    return float(ex.mean() / downside * np.sqrt(periods)) if downside > 0 else np.nan


def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1 + r).cumprod()
    return wealth / wealth.cummax() - 1.0


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown_series(r).min())


def max_drawdown_duration_days(r: pd.Series) -> int:
    """Longest stretch (in trading days) below a previous peak."""
    dd = drawdown_series(r)
    underwater = dd < 0
    longest = current = 0
    for flag in underwater.values:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


def calmar(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(r)
    return float(cagr(r, periods) / abs(mdd)) if mdd < 0 else np.nan


def var_cvar(r: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """Historical daily VaR and CVaR (expected shortfall) at level alpha, as positive losses."""
    q = r.quantile(alpha)
    return float(-q), float(-r[r <= q].mean())


def tail_stats(r: pd.Series) -> Dict[str, float]:
    return dict(skew=float(stats.skew(r.dropna())), kurtosis=float(stats.kurtosis(r.dropna())))


# --------------------------------------------------------------------------- #
# Portfolio-structure metrics
# --------------------------------------------------------------------------- #
def annual_turnover(traded: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Average traded notional per year (sum|dw| summed over rebalances)."""
    return float(traded.sum() / len(traded) * periods)


def concentration(weights: pd.DataFrame) -> Dict[str, float]:
    hhi = (weights ** 2).sum(axis=1)
    return dict(avg_effective_n=float((1 / hhi).mean()),
                avg_max_weight=float(weights.max(axis=1).mean()),
                avg_n_positions=float((weights > 1e-4).sum(axis=1).mean()))


def weight_stability(weights: pd.DataFrame) -> float:
    """Mean L1 distance between consecutive target weight vectors."""
    return float(weights.diff().abs().sum(axis=1).iloc[1:].mean())


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #
def summary_table(net: pd.DataFrame, rf: pd.Series,
                  gross: Optional[pd.DataFrame] = None,
                  traded: Optional[pd.DataFrame] = None,
                  weights: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    rows = {}
    for col in net.columns:
        r = net[col].dropna()
        var5, cvar5 = var_cvar(r)
        row = {
            "CAGR": cagr(r),
            "Volatility": ann_vol(r),
            "Sharpe": sharpe(r, rf),
            "Sortino": sortino(r, rf),
            "Max Drawdown": max_drawdown(r),
            "Calmar": calmar(r),
            "Daily VaR 95%": var5,
            "Daily CVaR 95%": cvar5,
            "Skew": tail_stats(r)["skew"],
        }
        if gross is not None and col in gross:
            row["Cost Drag (CAGR)"] = cagr(gross[col].dropna()) - row["CAGR"]
        if traded is not None and col in traded:
            row["Annual Turnover"] = annual_turnover(traded[col])
        if weights is not None and col in weights:
            c = concentration(weights[col])
            row["Avg Effective N"] = c["avg_effective_n"]
            row["Avg Max Weight"] = c["avg_max_weight"]
            row["Weight Instability"] = weight_stability(weights[col])
        rows[col] = row
    return pd.DataFrame(rows).T


def format_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Human-readable percentages and 2-dp ratios for display."""
    pct = ["CAGR", "Volatility", "Max Drawdown", "Daily VaR 95%", "Daily CVaR 95%",
           "Cost Drag (CAGR)", "Annual Turnover", "Avg Max Weight", "Weight Instability"]
    out = df.copy()
    for c in out.columns:
        if c in pct:
            out[c] = out[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        else:
            out[c] = out[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    return out


def annual_returns(net: pd.DataFrame) -> pd.DataFrame:
    return net.groupby(net.index.year).apply(lambda x: (1 + x).prod() - 1)


# --------------------------------------------------------------------------- #
# Statistical comparison of Sharpe ratios
# --------------------------------------------------------------------------- #
def sharpe_difference_test(r1: pd.Series, r2: pd.Series, rf: Optional[pd.Series] = None,
                           freq: str = "ME") -> Dict[str, float]:
    """
    Jobson-Korkie (1981) test with the Memmel (2003) correction for
    H0: SR_1 = SR_2. Computed on period-aggregated returns (monthly by
    default) to reduce autocorrelation. Returns z-stat and two-sided p-value.

    This is a first-pass test; it assumes approximately iid returns. For a
    HAC-robust version see Ledoit & Wolf (2008).
    """
    df = pd.concat([r1, r2], axis=1).dropna()
    if rf is not None:
        rf_al = rf.reindex(df.index).fillna(0.0)
        df = df.sub(rf_al, axis=0)
    agg = (1 + df).resample(freq).prod() - 1
    agg = agg.dropna()
    x, y = agg.iloc[:, 0].values, agg.iloc[:, 1].values
    T = len(agg)
    m1, m2 = x.mean(), y.mean()
    s1, s2 = x.std(ddof=1), y.std(ddof=1)
    rho = np.corrcoef(x, y)[0, 1]
    sr1, sr2 = m1 / s1, m2 / s2
    theta = (2 - 2 * rho + 0.5 * (sr1 ** 2 + sr2 ** 2 - 2 * sr1 * sr2 * rho ** 2)) / T
    z = (sr1 - sr2) / np.sqrt(theta)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return dict(sr1_period=sr1, sr2_period=sr2, diff_ann=(sr1 - sr2) * np.sqrt(12 if freq == "ME" else 1),
                z=float(z), p_value=float(p), n_periods=T, corr=float(rho))
