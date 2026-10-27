"""Weight-based daily backtester.

A strategy returns target portfolio weights per day, computed only from data
up to and including that day's close. Weights are applied from the NEXT day,
so there is no look-ahead. Costs are charged on every unit of turnover.
"""
import numpy as np
import pandas as pd

# Trading 212: no commission, 0.15% FX fee each way on USD assets for a GBP
# account, plus ~0.05% for spread/slippage.
COST_PER_TRADE = 0.0020


def run(prices, weights, start_cash=5000.0, cost=COST_PER_TRADE):
    rets = prices.pct_change().fillna(0.0)
    w = weights.reindex(prices.index).ffill().fillna(0.0)
    held = w.shift(1).fillna(0.0)  # trade at close, earn from next day
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    daily = (held * rets[held.columns]).sum(axis=1) - turnover * cost
    equity = start_cash * (1 + daily).cumprod()
    return equity, daily, turnover


def stats(equity, daily, turnover=None):
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = daily.std() * np.sqrt(252)
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    dd = (equity / equity.cummax() - 1).min()
    out = {
        "Final £": round(equity.iloc[-1]),
        "CAGR %": round(cagr * 100, 1),
        "Vol %": round(vol * 100, 1),
        "Sharpe": round(sharpe, 2),
        "MaxDD %": round(dd * 100, 1),
        "Calmar": round(cagr / -dd, 2) if dd < 0 else np.nan,
    }
    if turnover is not None:
        out["Trades/yr"] = round(turnover.sum() / years, 1)
    return out
