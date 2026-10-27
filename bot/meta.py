"""Adaptive meta-learner: learns which strategies to trust from their own
past (net-of-cost) results, and re-allocates between them every month.

Two learning rules:
  - "top-k": back the k strategies with the best trailing risk-adjusted return
  - "hedge": the multiplicative-weights / Hedge algorithm from online learning:
             each strategy's weight grows exponentially with its recent
             (decayed) performance, so trust shifts gradually rather than
             jumping, and no strategy is ever fully abandoned.
Allocation at month-end t uses strategy returns up to t only.
"""
import numpy as np
import pandas as pd

from .backtest import run
from .strategies import month_ends


def strategy_returns(prices, strategies):
    return pd.DataFrame({name: run(prices, w)[1] for name, w in strategies.items()})


def allocations(rets, rule="hedge", lookback=252, k=3, halflife=126, eta=8.0, warmup=252):
    """Monthly strategy allocations (rows = month-ends, cols = strategies)."""
    me = rets.index[month_ends(rets)]
    out = {}
    for d in me:
        hist = rets.loc[:d]
        if len(hist) < warmup:
            continue
        if rule == "topk":
            h = hist.iloc[-lookback:]
            score = h.mean() / (h.std() + 1e-9)
            best = score.nlargest(k).index
            out[d] = pd.Series({s: (1 / k if s in best else 0.0) for s in rets.columns})
        else:  # hedge
            wts = 0.5 ** (np.arange(len(hist))[::-1] / halflife)
            perf = (np.log1p(hist).mul(wts, axis=0)).sum() / wts.sum() * 252
            z = np.exp(eta * (perf - perf.max()))
            out[d] = z / z.sum()
    return pd.DataFrame(out).T


def combine(prices, strategies, alloc):
    """Turn strategy allocations into one set of daily asset weights."""
    a = alloc.reindex(prices.index).ffill().fillna(0.0)
    total = sum(strategies[s].mul(a[s], axis=0) for s in alloc.columns)
    return total.fillna(0.0)


def meta_strategy(prices, strategies, **kw):
    rets = strategy_returns(prices, strategies)
    alloc = allocations(rets, **kw)
    w = combine(prices, strategies, alloc)
    w.attrs["alloc"] = alloc
    return w
