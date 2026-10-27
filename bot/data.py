"""Daily price data from Yahoo Finance, cached to disk."""
import os

import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")

# Liquid US ETFs covering stocks, bonds, gold, commodities, real estate.
UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]
CASH = "SHY"  # short-term treasuries, used as the "risk-off" holding


# For candlestick tests: broad ETFs + large US stocks. Note survivorship bias:
# these are today's winners, so absolute returns are flattering; we only
# compare pattern days against normal days for the same stocks.
PATTERN_UNIVERSE = UNIVERSE + [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
    "AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "XOM", "PG", "KO", "PFE",
    "WMT", "INTC", "CSCO", "ORCL", "IBM", "MRK", "CVX", "HD", "DIS", "MCD", "BA",
]


def load_ohlc(tickers, start="2000-01-01", refresh=False):
    """Dict of ticker -> adjusted OHLC DataFrame."""
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    missing = [t for t in tickers
               if refresh or not os.path.exists(os.path.join(CACHE, f"ohlc_{t}_{start}.csv"))]
    if missing:
        df = yf.download(missing, start=start, auto_adjust=True, progress=False, group_by="ticker")
        for t in missing:
            df[t].dropna().to_csv(os.path.join(CACHE, f"ohlc_{t}_{start}.csv"))
    for t in tickers:
        out[t] = pd.read_csv(os.path.join(CACHE, f"ohlc_{t}_{start}.csv"), index_col=0, parse_dates=True)
    return out


def load_prices(tickers, start="2005-01-01", refresh=False):
    """Adjusted close prices (dividends included), one column per ticker."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{'_'.join(sorted(tickers))}_{start}.csv")
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    df = df[tickers].dropna(how="all").ffill()
    df.to_csv(path)
    return df
