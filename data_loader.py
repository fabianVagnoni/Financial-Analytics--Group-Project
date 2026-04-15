"""
Data download and log-return computation.
Source: Lecture 2 download pattern (used in every lecture notebook).

Downloads from yfinance and caches to CSV in data_cache/ to avoid
repeated API calls and Yahoo Finance rate limits.
"""

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf

from config import START_DATE, END_DATE

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")


def _cache_path(ticker, start, end):
    """Return path for a cached CSV file."""
    return os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}.csv")


def download_price(
    ticker: str,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.Series:
    """
    Download daily Close prices for a single ticker via yfinance.
    Caches result to CSV so subsequent calls don't hit the API.
    Returns pd.Series indexed by date, name = ticker.
    """
    cache = _cache_path(ticker, start, end)

    # Try cache first
    if os.path.exists(cache):
        price = pd.read_csv(cache, index_col=0, parse_dates=True).squeeze("columns")
        price.name = ticker
        return price

    # Download from yfinance with retry for rate limits
    df = pd.DataFrame()
    for attempt in range(3):
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
        )
        # Flatten MultiIndex columns if present (yfinance quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            break
        time.sleep(2 * (attempt + 1))  # back off: 2s, 4s, 6s

    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} ({start} to {end}).")

    df = df.dropna(subset=["Close"])
    price = df["Close"].astype(float)
    price.name = ticker

    # Cache to CSV
    os.makedirs(CACHE_DIR, exist_ok=True)
    price.to_csv(cache)

    return price


def compute_log_returns(price: pd.Series) -> pd.Series:
    """Compute log-returns: ln(P_t) - ln(P_{t-1})."""
    log_ret = np.log(price).diff().dropna()
    log_ret.name = price.name
    return log_ret


def load_single_stock(
    ticker: str,
    start: str = START_DATE,
    end: str = END_DATE,
) -> tuple:
    """
    Convenience wrapper: returns (price, log_ret) for one ticker.
    """
    price = download_price(ticker, start, end)
    log_ret = compute_log_returns(price)
    return price, log_ret


def load_multiple_stocks(
    tickers: list,
    start: str = START_DATE,
    end: str = END_DATE,
) -> tuple:
    """
    Download Close prices for multiple tickers, align dates, compute log-returns.
    Returns (prices_df, returns_df) both with columns = tickers.

    Adapted from Lecture 11:
        data = yf.download(tickers, ...)["Close"]
        returns = np.log(data / data.shift(1)).dropna()
    """
    # Use single-stock loader to leverage CSV cache
    series = {}
    for t in tickers:
        series[t] = download_price(t, start, end)

    prices_df = pd.DataFrame(series).dropna()
    returns_df = np.log(prices_df / prices_df.shift(1)).dropna()

    return prices_df, returns_df
