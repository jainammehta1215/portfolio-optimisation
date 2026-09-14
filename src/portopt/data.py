"""
Data acquisition layer.

Responsibilities
----------------
* Download adjusted prices from Yahoo Finance with local caching.
* Download the risk-free rate from FRED (no API key required).
* Build a point-in-time market-cap proxy (current shares outstanding x
  historical price) for Black-Litterman equilibrium weights.
* Run data-quality checks and fail loudly on problems that would corrupt
  downstream estimates.
* Provide a synthetic fallback so the pipeline can be unit-tested offline.

Every public function returns pandas objects indexed by a tz-naive
DatetimeIndex and aligned to the trading calendar of the price data.
"""
from __future__ import annotations

import io
import logging
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from .config import (DATA, FALLBACK_SHARES_OUTSTANDING_BN, SECTOR_MAP,
                     UNIVERSE, DataConfig)

logger = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _cache_path(cfg: DataConfig, name: str) -> str:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    return os.path.join(cfg.cache_dir, name)


def _retry(fn, attempts: int = 3, wait: float = 2.0, label: str = "call"):
    """Retry a callable with linear back-off. Re-raises the last exception."""
    last_exc: Optional[Exception] = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we want to retry on anything
            last_exc = exc
            logger.warning("%s failed (attempt %d/%d): %s", label, i, attempts, exc)
            time.sleep(wait * i)
    assert last_exc is not None
    raise last_exc


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #
def load_prices(tickers: Iterable[str] = UNIVERSE,
                cfg: DataConfig = DATA,
                force_refresh: bool = False) -> pd.DataFrame:
    """
    Adjusted close prices, one column per ticker.

    Prices are cached to CSV keyed on the ticker set and date range.
    Adjusted closes (auto_adjust=True) already incorporate splits and
    dividends, so simple returns on them are total returns.
    """
    tickers = sorted(set(tickers))
    end = cfg.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cache = _cache_path(cfg, f"prices_{cfg.start}_{end}_{len(tickers)}.csv")

    if os.path.exists(cache) and not force_refresh:
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
        if set(prices.columns) == set(tickers):
            logger.info("Loaded %d tickers from cache %s", len(tickers), cache)
            return prices[tickers]

    import yfinance as yf  # imported lazily so tests can run without it

    def _download() -> pd.DataFrame:
        raw = yf.download(tickers, start=cfg.start, end=end,
                          auto_adjust=True, progress=False, threads=True)
        if raw.empty:
            raise RuntimeError("Yahoo Finance returned an empty frame")
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])
        return close

    prices = _retry(_download, label="yfinance download")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    prices.columns = [str(c) for c in prices.columns]
    prices = prices.reindex(columns=tickers)

    try:
        prices.to_csv(cache)
    except Exception as exc:  # read-only filesystem etc. - caching is best-effort
        logger.warning("Could not write price cache: %s", exc)
    return prices


def clean_prices(prices: pd.DataFrame,
                 cfg: DataConfig = DATA) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Data-quality gate.

    * Drops tickers with insufficient history.
    * Forward-fills isolated gaps (max 5 days) and reports them.
    * Drops any leading rows where not every ticker has a price.
    * Raises on non-positive prices or on duplicated dates.

    Returns the cleaned frame and a dict of issues found, for the report.
    """
    issues: Dict[str, str] = {}
    px = prices.copy()

    if px.index.duplicated().any():
        raise ValueError("Duplicated dates in price index")

    # 1. Cross-exchange calendars: drop days on which most of the universe was
    #    closed (e.g. an Indian holiday in a mixed India/US list). Dropping a
    #    row is safe for returns because pct_change compounds across the gap;
    #    forward-filling instead would create artificial zero-return days.
    row_cov = px.notna().mean(axis=1)
    dropped_days = int((row_cov < 0.5).sum())
    if dropped_days:
        issues["_calendar"] = f"dropped {dropped_days} days when <50% of tickers traded"
    px = px[row_cov >= 0.5]

    # 2. Fill the remaining isolated gaps (single-exchange holidays, bad ticks).
    gap_counts = px.isna().sum()
    px = px.ffill(limit=5)

    # 3. Tickers with insufficient history (listed after `start`, delisted, etc).
    coverage = px.notna().mean()
    thin = coverage[coverage < cfg.min_history_fraction]
    for t, c in thin.items():
        first = px[t].first_valid_index()
        issues[t] = (f"dropped: only {c:.1%} of dates have a price"
                     + (f" (first price {first.date()}; move DataConfig.start later to keep it)" if first is not None else ""))
    px = px.drop(columns=thin.index)
    for t, n in gap_counts.reindex(px.columns).items():
        if n > 0:
            issues[t] = issues.get(t, "") + f"filled {int(n)} missing price(s)"

    px = px.dropna(how="any")          # remove leading NaNs / anything unfillable

    if (px <= 0).any().any():
        bad = px.columns[(px <= 0).any()].tolist()
        raise ValueError(f"Non-positive prices found for {bad}")

    return px, issues


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns. Simple (not log) because portfolio returns aggregate linearly."""
    rets = prices.pct_change().iloc[1:]
    # Guard against corporate-action glitches that produce absurd single-day moves.
    extreme = (rets.abs() > 0.5)
    if extreme.any().any():
        n = int(extreme.sum().sum())
        logger.warning("%d daily returns exceed +/-50%%; inspect before trusting results", n)
    return rets


# --------------------------------------------------------------------------- #
# Currency conversion
# --------------------------------------------------------------------------- #
def fetch_currencies(tickers: Iterable[str]) -> pd.Series:
    """Quote currency per ticker from yfinance (e.g. USD, INR, JPY, GBp)."""
    out: Dict[str, str] = {}
    try:
        import yfinance as yf
        for t in tickers:
            try:
                out[t] = str(yf.Ticker(t).fast_info.get("currency") or "USD")
            except Exception as exc:  # noqa: BLE001
                logger.warning("currency lookup failed for %s; assuming USD: %s", t, exc)
                out[t] = "USD"
    except ImportError:
        out = {t: "USD" for t in tickers}
    return pd.Series(out, name="currency")


def convert_to_base_currency(prices: pd.DataFrame, base: str = "USD",
                             currencies: Optional[pd.Series] = None,
                             cfg: DataConfig = DATA) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Convert every column to `base` using Yahoo FX crosses (e.g. INRUSD=X).

    GBp (pence) and ZAc (cents) are scaled to their major unit first. FX
    series are forward-filled onto the price calendar. Returns the converted
    frame and a log of what was converted. No-op for single-currency
    universes already quoted in `base`.
    """
    log: Dict[str, str] = {}
    cur = currencies if currencies is not None else fetch_currencies(prices.columns)
    cur = cur.reindex(prices.columns).fillna(base)
    px = prices.copy()

    minor = {"GBp": ("GBP", 100.0), "ZAc": ("ZAR", 100.0), "ILA": ("ILS", 100.0)}
    for t, c in cur.items():
        if c in minor:
            major, div = minor[c]
            px[t] = px[t] / div
            cur[t] = major

    needed = sorted(set(cur) - {base})
    if not needed:
        return px, log

    import yfinance as yf
    end = cfg.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    for c in needed:
        pair = f"{c}{base}=X"
        def _dl(pair=pair):
            fx = yf.download(pair, start=cfg.start, end=end, auto_adjust=True, progress=False)
            if fx.empty:
                raise RuntimeError(f"no FX data for {pair}")
            fx = fx["Close"]
            return fx.iloc[:, 0] if isinstance(fx, pd.DataFrame) else fx
        try:
            fx = _retry(_dl, label=f"FX {pair}")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Could not download {pair}; cannot express {c} assets in {base}") from exc
        fx.index = pd.to_datetime(fx.index).tz_localize(None)
        fx = fx.reindex(px.index.union(fx.index)).ffill().reindex(px.index)
        cols = [t for t, cc in cur.items() if cc == c]
        px[cols] = px[cols].mul(fx, axis=0)
        log[c] = f"{len(cols)} ticker(s) converted via {pair}"
    return px, log


# --------------------------------------------------------------------------- #
# Risk-free rate
# --------------------------------------------------------------------------- #
def load_risk_free(index: pd.DatetimeIndex,
                   cfg: DataConfig = DATA,
                   force_refresh: bool = False) -> pd.Series:
    """
    Daily risk-free rate aligned to `index`, derived from a FRED yield series
    quoted in percent (annualised, bond-equivalent). Falls back to zero with a
    loud warning if FRED is unreachable, so the pipeline never silently uses a
    stale number.
    """
    cache = _cache_path(cfg, f"fred_{cfg.fred_series}.csv")
    try:
        if os.path.exists(cache) and not force_refresh:
            raw = pd.read_csv(cache, index_col=0, parse_dates=True)
        else:
            resp = _retry(lambda: requests.get(FRED_CSV_URL.format(series=cfg.fred_series), timeout=30),
                          label="FRED download")
            resp.raise_for_status()
            raw = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
            raw.to_csv(cache)
        yld = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    except Exception as exc:  # noqa: BLE001
        logger.error("Risk-free download failed (%s). Using 0%% - Sharpe ratios will be overstated.", exc)
        return pd.Series(0.0, index=index, name="rf_daily")

    yld = yld.reindex(index.union(yld.index)).ffill().reindex(index)
    yld = yld.fillna(0.0)
    daily = (1.0 + yld / 100.0) ** (1.0 / cfg.trading_days) - 1.0
    daily.name = "rf_daily"
    return daily


# --------------------------------------------------------------------------- #
# Market capitalisation proxy
# --------------------------------------------------------------------------- #
def load_shares_outstanding(tickers: Iterable[str]) -> pd.Series:
    """
    Current shares outstanding per ticker via yfinance fast_info, with a
    hard-coded fallback. Wrapped per-ticker so one failure does not sink all.
    """
    tickers = list(tickers)
    out: Dict[str, float] = {}
    try:
        import yfinance as yf
        for t in tickers:
            try:
                tk = yf.Ticker(t)
                fi = tk.fast_info
                shares = fi.get("shares") or fi.get("sharesOutstanding")
                if not shares:                      # derive from market cap / price
                    mcap, last = fi.get("marketCap"), fi.get("lastPrice")
                    if mcap and last:
                        shares = mcap / last
                if not shares:                      # slower .info endpoint as a last resort
                    shares = tk.info.get("sharesOutstanding")
                if shares and shares > 0:
                    out[t] = float(shares)
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("shares lookup failed for %s: %s", t, exc)
            out[t] = FALLBACK_SHARES_OUTSTANDING_BN.get(t, np.nan) * 1e9
    except ImportError:
        out = {t: FALLBACK_SHARES_OUTSTANDING_BN.get(t, np.nan) * 1e9 for t in tickers}

    s = pd.Series(out, name="shares_outstanding")
    if s.isna().any():
        missing = s.index[s.isna()].tolist()
        raise ValueError(
            f"No shares outstanding available for {missing}. Add them to "
            "config.FALLBACK_SHARES_OUTSTANDING_BN (in billions) or drop the ticker.")
    return s


def market_cap_proxy(prices: pd.DataFrame, shares: pd.Series) -> pd.DataFrame:
    """
    Historical market-cap proxy = current shares outstanding x historical price.

    This ignores buybacks and issuance over time, but it is point-in-time
    with respect to *prices*, which is what matters for avoiding lookahead in
    the equilibrium weights. Using today's market caps for 2012 rebalances
    would embed knowledge of which companies later grew.
    """
    shares = shares.reindex(prices.columns)
    return prices.mul(shares, axis=1)


def market_weights_at(caps: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Cap-weighted vector on a date (last available on or before `date`)."""
    row = caps.loc[:date].iloc[-1]
    return row / row.sum()


# --------------------------------------------------------------------------- #
# Sectors
# --------------------------------------------------------------------------- #
def fetch_sectors(tickers: Iterable[str]) -> Dict[str, str]:
    """Sector per ticker from yfinance's .info endpoint (slow; one call per ticker)."""
    found: Dict[str, str] = {}
    try:
        import yfinance as yf
        for t in tickers:
            try:
                sec = yf.Ticker(t).info.get("sector")
                if sec:
                    found[t] = str(sec)
            except Exception as exc:  # noqa: BLE001
                logger.warning("sector lookup failed for %s: %s", t, exc)
    except ImportError:
        pass
    return found


def sector_matrix(tickers: List[str],
                  sector_map: Dict[str, str] = SECTOR_MAP,
                  auto_lookup: bool = True) -> pd.DataFrame:
    """
    Binary (sectors x assets) matrix A such that A @ w = sector weights.

    Tickers missing from `sector_map` are looked up on Yahoo Finance when
    `auto_lookup` is True; anything still unknown goes to 'Other' with a
    warning. With only one or two sectors present the default 35% sector
    cap is infeasible, so raise the cap or turn it off in that case.
    """
    rows = {}
    missing = [t for t in tickers if t not in sector_map]
    fetched = fetch_sectors(missing) if (missing and auto_lookup) else {}
    for t in tickers:
        sec = sector_map.get(t) or fetched.get(t)
        if sec is None:
            logger.warning("No sector for %s; assigning 'Other'", t)
            sec = "Other"
        rows[t] = sec
    sectors = sorted(set(rows.values()))
    mat = pd.DataFrame(0.0, index=sectors, columns=tickers)
    for t, sec in rows.items():
        mat.loc[sec, t] = 1.0
    return mat


# --------------------------------------------------------------------------- #
# Synthetic fallback (offline tests / demos)
# --------------------------------------------------------------------------- #
def synthetic_prices(tickers: List[str] = UNIVERSE,
                     n_days: int = 2520,
                     seed: int = 7,
                     start: str = "2015-01-01") -> pd.DataFrame:
    """
    Geometric Brownian motion with a one-factor correlation structure.
    Deterministic given `seed`. Used only when live data is unavailable.
    """
    rng = np.random.default_rng(seed)
    n = len(tickers)
    beta = rng.uniform(0.6, 1.4, n)
    idio = rng.uniform(0.15, 0.35, n) / np.sqrt(252)
    mkt = rng.normal(0.0004, 0.011, n_days)
    eps = rng.normal(0, 1, (n_days, n)) * idio
    rets = np.outer(mkt, beta) + eps + rng.uniform(0.0, 0.0003, n)
    idx = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=tickers)


def load_all(cfg: DataConfig = DATA,
             tickers: Iterable[str] = UNIVERSE,
             use_synthetic: bool = False) -> Dict[str, object]:
    """
    One-call loader used by the notebook. Returns a dict with:
    prices, returns, rf, caps, sectors (matrix), shares, issues.
    """
    tickers = list(tickers)
    if use_synthetic:
        logger.warning("Using SYNTHETIC prices - results are illustrative only")
        prices = synthetic_prices(tickers)
        shares = pd.Series({t: FALLBACK_SHARES_OUTSTANDING_BN.get(t, 1.0) * 1e9 for t in tickers})
        issues: Dict[str, str] = {"_mode": "synthetic"}
    else:
        prices = load_prices(tickers, cfg)
        prices, issues = clean_prices(prices, cfg)
        prices, fx_log = convert_to_base_currency(prices, cfg.base_currency, cfg=cfg)
        prices = prices.dropna(how="any")
        issues.update({f"_fx_{k}": v for k, v in fx_log.items()})
        shares = load_shares_outstanding(prices.columns)

    returns = compute_returns(prices)
    rf = load_risk_free(returns.index, cfg) if not use_synthetic else pd.Series(0.02 / 252, index=returns.index)
    caps = market_cap_proxy(prices, shares)
    sectors = sector_matrix(list(prices.columns))
    return dict(prices=prices, returns=returns, rf=rf, caps=caps,
                sectors=sectors, shares=shares, issues=issues)
