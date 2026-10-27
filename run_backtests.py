"""Backtest every strategy vs buy-and-hold SPY, on the full history and on
separate in-sample / out-of-sample periods to check the edge is consistent."""
import warnings

import pandas as pd

from bot import strategies as S
from bot.backtest import run, stats
from bot.data import CASH, UNIVERSE, load_prices

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
prices = load_prices(UNIVERSE + SECTORS + [CASH], start="2005-01-01").dropna()
U = UNIVERSE
mom5 = S.momentum_rotation(prices, U, top=5)

STRATS = {
    "Buy & hold SPY": S.buy_hold(prices),
    "60/40 SPY/bonds": S.sixty_forty(prices),
    "SPY 200d trend": S.trend_200(prices),
    "SPY 50/200 golden cross": S.sma_cross(prices),
    "SPY vol-target 12%": S.vol_target(prices),
    "SPY turn-of-month": S.turn_of_month(prices),
    "SPY RSI(2) reversion": S.rsi2_reversion(prices),
    "Dual momentum (GEM)": S.dual_momentum(prices),
    "Risk parity": S.risk_parity(prices),
    "Time-series momentum": S.tsmom(prices, U),
    "Donchian breakout": S.donchian(prices, U),
    "Dip-buy (20d low, uptrend)": S.dip_buy(prices, U),
    "Sector momentum top-3": S.momentum_rotation(prices, SECTORS, top=3),
    "Momentum top-3": S.momentum_rotation(prices, U, top=3),
    "Momentum top-5 (LIVE)": mom5,
    "Momentum top-5 + vol-target": S.with_vol_target(mom5, prices),
    "50% mom-5 + 50% SPY": S.blend((0.5, mom5), (0.5, S.buy_hold(prices))),
}

PERIODS = {
    "FULL": ("2007-03-01", None),  # after every strategy's 1-year warm-up
    "2007-2015 (in-sample)": ("2007-03-01", "2015-12-31"),
    "2016-now (out-of-sample)": ("2016-01-01", None),
    "Last 3y": (str(prices.index[-1] - pd.DateOffset(years=3))[:10], None),
}

for period, (a, b) in PERIODS.items():
    rows = {}
    for name, w in STRATS.items():
        eq, daily, to = run(prices.loc[a:b], w.loc[a:b])
        rows[name] = stats(eq, daily, to)
    p = prices.loc[a:b].index
    print(f"\n=== {period}  ({p[0].date()} -> {p[-1].date()}) ===")
    print(pd.DataFrame(rows).T.sort_values("Sharpe", ascending=False).to_string())

# Robustness: does momentum still work if we change its settings? If only one
# exact setting works, it's probably luck (curve fitting).
print("\n=== Robustness grid: momentum rotation, Sharpe 2007-now (SPY = "
      f"{stats(*run(prices.loc['2007-03-01':], STRATS['Buy & hold SPY'].loc['2007-03-01':]))['Sharpe']}) ===")
grid = {}
for lbs, label in [((63, 126, 252), "3/6/12m"), ((126,), "6m"), ((252,), "12m"), ((21, 63, 126), "1/3/6m")]:
    grid[label] = {}
    for top in (2, 3, 4, 5, 6):
        w = S.momentum_rotation(prices, U, top=top, lookbacks=lbs).loc["2007-03-01":]
        s = stats(*run(prices.loc["2007-03-01":], w))
        grid[label][f"top {top}"] = f"{s['Sharpe']:.2f} / {s['CAGR %']:.1f}%"
print(pd.DataFrame(grid).T.to_string())
print("(cells: Sharpe / CAGR)")
