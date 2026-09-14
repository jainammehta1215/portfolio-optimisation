"""
Visualisations. Every function returns a matplotlib Figure and never calls
plt.show(), so the notebook controls display and saving.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from . import metrics as met

PALETTE = ["#1f4e79", "#c0392b", "#e67e22", "#27ae60", "#8e44ad",
           "#16a085", "#7f8c8d", "#d4ac0d", "#2c3e50", "#e84393"]


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "figure.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.labelsize": 10,
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })


def save(fig: plt.Figure, name: str, outdir: str = "outputs/figures") -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    return path


_pct = PercentFormatter(1.0, decimals=0)


# --------------------------------------------------------------------------- #
# Static analysis figures
# --------------------------------------------------------------------------- #
def plot_frontiers(fc: Dict[str, object], rf_ann: float = 0.0,
                   title: str = "Efficient frontiers on one estimation window") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, label, style in [("sample", "Classical (sample estimates)", "-"),
                              ("ledoit_wolf", "Ledoit-Wolf covariance", "--"),
                              ("resampled", "Michaud resampled", "-.")]:
        fr = fc[key]
        ax.plot(fr["vol"], fr["ret"], style, lw=2, label=label)
    a = fc["assets"]
    ax.scatter(a["vol"], a["ret"], s=28, color="#7f8c8d", zorder=3)
    for t, row in a.iterrows():
        ax.annotate(t, (row["vol"], row["ret"]), fontsize=8, xytext=(4, 2), textcoords="offset points")
    ax.axhline(rf_ann, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Annualised volatility"); ax.set_ylabel("Annualised expected return")
    ax.xaxis.set_major_formatter(_pct); ax.yaxis.set_major_formatter(_pct)
    ax.set_title(title); ax.legend(loc="lower right")
    return fig


def plot_weight_comparison(weights: pd.DataFrame, title: str = "Portfolio weights by method") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5))
    weights.plot.bar(ax=ax, width=0.8)
    ax.yaxis.set_major_formatter(_pct); ax.set_ylabel("Weight"); ax.set_xlabel("")
    ax.set_title(title); ax.legend(ncol=2, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=0)
    return fig


def plot_instability(draws: Dict[str, np.ndarray], assets: List[str],
                     methods: Optional[List[str]] = None) -> plt.Figure:
    """Box plot of weight draws per asset, one panel per method."""
    methods = methods or list(draws)
    ncol = 2
    nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.2 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, methods):
        W = draws[m]
        ax.boxplot([W[:, j] for j in range(W.shape[1])], tick_labels=assets,
                   showfliers=False, medianprops=dict(color=PALETTE[1]))
        ax.set_title(m, fontsize=10); ax.yaxis.set_major_formatter(_pct)
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    for ax in axes[len(methods):]:
        ax.axis("off")
    fig.suptitle("Weight dispersion when expected returns are perturbed by one standard error",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    return fig


def plot_shrinkage_vs_window(df: pd.DataFrame) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(df.index, df["lw_shrinkage"], "o-", color=PALETTE[0], label="LW shrinkage intensity")
    ax1.set_ylabel("Shrinkage intensity", color=PALETTE[0]); ax1.set_xlabel("Estimation window (trading days)")
    ax1.set_xscale("log"); ax1.set_xticks(df.index); ax1.set_xticklabels(df.index)
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["cond_sample"], "s--", color=PALETTE[1], label="Condition number (sample)")
    ax2.plot(df.index, df["cond_ledoit_wolf"], "s--", color=PALETTE[3], label="Condition number (LW)")
    ax2.set_ylabel("Covariance condition number"); ax2.grid(False)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    ax1.set_title("Shrinkage matters most when the window is short relative to N")
    return fig


def plot_bl_returns(res, title: str = "Black-Litterman: equilibrium prior vs posterior") -> plt.Figure:
    df = res.summary()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(df)); w = 0.38
    ax.bar(x - w / 2, df["prior"], w, label="Prior (implied equilibrium, Pi)", color=PALETTE[6])
    ax.bar(x + w / 2, df["posterior"], w, label="Posterior (mu_BL)", color=PALETTE[0])
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=0)
    ax.yaxis.set_major_formatter(_pct); ax.set_ylabel("Annualised excess return")
    ax.set_title(title + f"   (delta = {res.delta:.2f})")
    if res.view_labels:
        ax.text(0.01, 0.97, "\n".join("View: " + v for v in res.view_labels), transform=ax.transAxes,
                va="top", fontsize=8, bbox=dict(boxstyle="round", fc="#f8f9fa", ec="#dee2e6"))
    ax.legend(loc="upper right")
    return fig


# --------------------------------------------------------------------------- #
# Backtest figures
# --------------------------------------------------------------------------- #
def plot_cumulative(net: pd.DataFrame, log_scale: bool = True,
                    title: str = "Out-of-sample growth of $1 (net of costs)") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    wealth = (1 + net).cumprod()
    for i, c in enumerate(wealth.columns):
        lw = 2.4 if "1/N" in c else 1.5
        ax.plot(wealth.index, wealth[c], lw=lw, label=c,
                color="black" if "1/N" in c else PALETTE[i % len(PALETTE)])
    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title(title); ax.legend(loc="upper left", fontsize=8)
    return fig


def plot_drawdowns(net: pd.DataFrame, title: str = "Drawdowns") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, c in enumerate(net.columns):
        dd = met.drawdown_series(net[c])
        ax.plot(dd.index, dd, lw=1.2, label=c, color="black" if "1/N" in c else PALETTE[i % len(PALETTE)])
    ax.yaxis.set_major_formatter(_pct); ax.set_title(title); ax.legend(fontsize=8, loc="lower left")
    return fig


def plot_rolling_sharpe(net: pd.DataFrame, rf: pd.Series, window: int = 756,
                        title: str = "Rolling 3-year Sharpe ratio") -> plt.Figure:
    ex = net.sub(rf.reindex(net.index).fillna(0.0), axis=0)
    rs = ex.rolling(window).mean() / ex.rolling(window).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, c in enumerate(rs.columns):
        ax.plot(rs.index, rs[c], lw=1.3, label=c, color="black" if "1/N" in c else PALETTE[i % len(PALETTE)])
    ax.axhline(0, color="grey", lw=0.8); ax.set_title(title); ax.legend(fontsize=8, ncol=2)
    return fig


def plot_allocation(weights: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4))
    cols = weights.columns
    cmap = plt.get_cmap("tab20")
    ax.stackplot(weights.index, [weights[c].values for c in cols], labels=cols,
                 colors=[cmap(i / max(len(cols) - 1, 1)) for i in range(len(cols))], alpha=0.9)
    ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(_pct); ax.set_title(title)
    ax.legend(ncol=8, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.margins(x=0)
    return fig


def plot_allocation_grid(results: Dict, names: List[str]) -> plt.Figure:
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 2.6 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("tab20")
    for ax, name in zip(axes, names):
        w = results[name].target_weights
        cols = w.columns
        ax.stackplot(w.index, [w[c].values for c in cols], labels=cols,
                     colors=[cmap(i / max(len(cols) - 1, 1)) for i in range(len(cols))], alpha=0.9)
        ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(_pct); ax.set_title(name, fontsize=10)
        ax.margins(x=0)
    axes[-1].legend(ncol=8, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.suptitle("Target allocations over time", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_cost_sensitivity(sharpe_tab: pd.DataFrame,
                          title: str = "Sharpe ratio versus transaction cost") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(sharpe_tab.index):
        ax.plot(sharpe_tab.columns, sharpe_tab.loc[name], "o-", lw=2 if "1/N" in name else 1.4,
                label=name, color="black" if "1/N" in name else PALETTE[i % len(PALETTE)])
    ax.set_xlabel("Cost per unit traded (bps)"); ax.set_ylabel("Sharpe (net)")
    ax.set_title(title); ax.legend(fontsize=8)
    return fig


def plot_turnover(traded: pd.DataFrame, title: str = "Annual traded notional (sum |dw|)") -> plt.Figure:
    annual = traded.groupby(traded.index.year).sum()
    annual = annual[annual.index > annual.index.min()]      # drop inception year (includes the 100% buy)
    fig, ax = plt.subplots(figsize=(11, 4))
    annual.plot.bar(ax=ax, width=0.85)
    ax.yaxis.set_major_formatter(_pct); ax.set_xlabel(""); ax.set_title(title); ax.legend(fontsize=8, ncol=2)
    plt.setp(ax.get_xticklabels(), rotation=0)
    return fig


def plot_risk_contributions(rc: pd.DataFrame, title: str = "Risk contribution by asset (latest weights)") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5))
    rc.plot.bar(ax=ax, width=0.8)
    ax.axhline(1 / len(rc), color="grey", ls=":", lw=1, label="Equal contribution")
    ax.yaxis.set_major_formatter(_pct); ax.set_xlabel(""); ax.set_title(title); ax.legend(fontsize=8, ncol=2)
    plt.setp(ax.get_xticklabels(), rotation=0)
    return fig


def plot_annual_returns(annual: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    annual.plot.bar(ax=ax, width=0.85)
    ax.yaxis.set_major_formatter(_pct); ax.set_xlabel(""); ax.set_title("Calendar-year returns (net)")
    ax.legend(fontsize=8, ncol=2); plt.setp(ax.get_xticklabels(), rotation=0)
    return fig


def plot_correlation(corr: pd.DataFrame, title: str = "Return correlation (estimation window)") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8); ax.set_yticklabels(corr.index, fontsize=8)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(corr.values[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046); ax.set_title(title)
    return fig


def plot_summary_heatmap(table: pd.DataFrame, cols: List[str]) -> plt.Figure:
    """Rank-coloured heatmap of the summary table (green = better)."""
    df = table[cols].copy()
    better_low = {"Volatility", "Max Drawdown", "Annual Turnover", "Cost Drag (CAGR)",
                  "Daily VaR 95%", "Daily CVaR 95%", "Avg Max Weight", "Weight Instability"}
    ranks = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for c in cols:
        series = df[c].abs() if c == "Max Drawdown" else df[c]
        ranks[c] = series.rank(ascending=(c in better_low))
    fig, ax = plt.subplots(figsize=(1.3 * len(cols) + 3, 0.5 * len(df) + 1.5))
    im = ax.imshow(ranks.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df.index, fontsize=8); ax.grid(False)
    for i in range(len(df)):
        for j, c in enumerate(cols):
            v = df.iloc[i, j]
            txt = f"{v:.1%}" if c in {"CAGR", "Volatility", "Max Drawdown", "Annual Turnover",
                                      "Cost Drag (CAGR)", "Avg Max Weight"} else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    ax.set_title("Strategy scorecard (colour = rank within column; green is better)")
    return fig
