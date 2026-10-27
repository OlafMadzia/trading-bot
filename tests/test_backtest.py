import numpy as np
import pandas as pd
import pytest

from bot.backtest import run, stats
from bot.strategies import month_ends


def _flat(n=10):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"A": 100.0}, index=idx)


def test_trade_earns_only_from_the_next_day():
    """A position decided at day t's close must not earn day t's move."""
    p = _flat()
    p.iloc[5:, 0] = 110.0  # +10% jump on day 5
    w = pd.DataFrame(0.0, index=p.index, columns=["A"])
    w.iloc[5:, 0] = 1.0    # decide to buy on day 5 (after the jump is known)
    eq, _, _ = run(p, w, start_cash=100, cost=0)
    assert eq.iloc[-1] == pytest.approx(100)  # missed the jump: no look-ahead

    w.iloc[4, 0] = 1.0     # bought the day before: earns it
    eq, _, _ = run(p, w, start_cash=100, cost=0)
    assert eq.iloc[-1] == pytest.approx(110)


def test_costs_are_charged_on_turnover():
    p = _flat()
    w = pd.DataFrame(1.0, index=p.index, columns=["A"])
    eq, _, turnover = run(p, w, start_cash=100, cost=0.01)
    assert turnover.sum() == pytest.approx(1.0)        # one buy, then hold
    assert eq.iloc[-1] == pytest.approx(99.0)          # 1% cost on the buy


def test_stats_on_known_series():
    idx = pd.bdate_range("2020-01-01", periods=253)
    eq = pd.Series(np.linspace(100, 110, 253), index=idx)
    eq.iloc[100] = 80  # a 20%+ drawdown
    s = stats(eq, eq.pct_change().fillna(0))
    assert s["MaxDD %"] < -20
    assert s["Final £"] == 110


def test_month_ends_flags_last_trading_day():
    idx = pd.bdate_range("2024-01-01", "2024-03-31")
    me = month_ends(pd.DataFrame(index=idx, data={"A": 1.0}))
    flagged = list(idx[me.values].strftime("%Y-%m-%d"))
    assert flagged == ["2024-01-31", "2024-02-29", "2024-03-29"]
