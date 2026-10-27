"""Round 2 research: published tactical strategies, a walk-forward ML ranker
and adaptive meta-learners, all compared on the same out-of-sample window.

Honesty checks:
  - every ML/meta decision only uses information available at the time
  - two separate halves of the test window must both look good
  - a paired t-test vs the current live strategy says whether any
    improvement is distinguishable from luck
"""
import time
import warnings

import numpy as np
import pandas as pd

from bot import strategies as S
from bot.backtest import run, stats
from bot.data import CASH, EXTRA, UNIVERSE, load_prices
from bot.meta import meta_strategy
from bot.ml import ml_ranker

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230)
t0 = time.time()

prices = load_prices(UNIVERSE + EXTRA + [CASH], start="2005-01-01").dropna()
U = UNIVERSE
print(f"Data {prices.index[0].date()} -> {prices.index[-1].date()}, {prices.shape[1]} ETFs")

base = {
    "Buy & hold SPY": S.buy_hold(prices),
    "60/40": S.sixty_forty(prices),
    "Risk parity": S.risk_parity(prices),
    "SPY 200d trend": S.trend_200(prices),
    "SPY vol-target": S.vol_target(prices),
    "Time-series momentum": S.tsmom(prices, U),
    "Momentum top-5 (LIVE)": S.momentum_rotation(prices, U, top=5),
    "Momentum top-3": S.momentum_rotation(prices, U, top=3),
    "Vol-scaled momentum top-5": S.momentum_volscaled(prices, U, top=5),
    "Adaptive Asset Alloc (Butler)": S.adaptive_aa(prices, U, top=5),
    "VAA-G4 (Keller)": S.vaa(prices),
    "DAA (Keller)": S.daa(prices),
}
print(f"base strategies built ({time.time() - t0:.0f}s)")
ml = {
    "ML gradient boosting top-5": ml_ranker(prices, U, top=5, kind="gbm"),
    "ML ridge regression top-5": ml_ranker(prices, U, top=5, kind="ridge"),
    "ML GBM top-5, no trend filter": ml_ranker(prices, U, top=5, kind="gbm", trend_filter=False),
}
print(f"ML models trained walk-forward ({time.time() - t0:.0f}s)")

pool = {k: v for k, v in {**base, **ml}.items()}
meta = {
    "META hedge (all strategies)": meta_strategy(prices, pool, rule="hedge"),
    "META top-3 by 12m Sharpe": meta_strategy(prices, pool, rule="topk", k=3),
    "META hedge (no ML)": meta_strategy(prices, base, rule="hedge"),
}
print(f"meta-learners built ({time.time() - t0:.0f}s)")

ALL = {**base, **ml, **meta}
PERIODS = {
    "OUT-OF-SAMPLE 2011-2026": ("2011-05-01", None),
    "First half 2011-2018": ("2011-05-01", "2018-12-31"),
    "Second half 2019-2026": ("2019-01-01", None),
}
for period, (a, b) in PERIODS.items():
    rows = {}
    for name, w in ALL.items():
        eq, daily, to = run(prices.loc[a:b], w.loc[a:b])
        rows[name] = stats(eq, daily, to)
    p = prices.loc[a:b].index
    print(f"\n=== {period}  ({p[0].date()} -> {p[-1].date()}) ===")
    print(pd.DataFrame(rows).T.sort_values("Sharpe", ascending=False).to_string())

# Is anything better than the live strategy by more than luck?
a = "2011-05-01"
live_d = run(prices.loc[a:], ALL["Momentum top-5 (LIVE)"].loc[a:])[1]
spy_d = run(prices.loc[a:], ALL["Buy & hold SPY"].loc[a:])[1]
rows = {}
for name, w in ALL.items():
    d = run(prices.loc[a:], w.loc[a:])[1]
    m = (1 + d).resample("ME").prod() - 1
    def tstat(other):
        o = (1 + other).resample("ME").prod() - 1
        diff = m - o
        return diff.mean() / (diff.std() / np.sqrt(len(diff))) if diff.std() > 0 else 0
    rows[name] = {"t vs LIVE": round(tstat(live_d), 2), "t vs SPY": round(tstat(spy_d), 2)}
print("\n=== Paired t-stat of monthly return difference (|t| > 2 ~ not luck) ===")
print(pd.DataFrame(rows).T.sort_values("t vs LIVE", ascending=False).to_string())

al = meta["META hedge (all strategies)"].attrs["alloc"]
print("\n=== What the Hedge meta-learner trusts most (avg allocation, last 3y) ===")
print((al.loc[str(prices.index[-1].year - 3):].mean().sort_values(ascending=False) * 100)
      .round(1).to_string())
print("\nLatest allocation:")
print((al.iloc[-1].sort_values(ascending=False) * 100).round(1).head(6).to_string())

imp = ml["ML gradient boosting top-5"].attrs["predictions"]
print(f"\nML predictions made: {len(imp)} months.  Done in {time.time() - t0:.0f}s")
