import numpy as np
import pandas as pd
import pytest

from bot import league
from bot.data import UNIVERSE
from bot.ml import ml_ranker
from bot.strategies import buy_hold, month_ends
from tests.conftest import change_future


def test_ml_never_trains_on_the_future(prices):
    """Scrambling everything after a date must leave earlier ML picks unchanged:
    the model may only learn from months whose outcome was already known."""
    cut = prices.index[1300]
    a = ml_ranker(prices, UNIVERSE, top=5)
    b = ml_ranker(change_future(prices, cut), UNIVERSE, top=5)
    me = prices.index[month_ends(prices)]
    decided_before = me[me < cut]
    assert len(decided_before) > 36 + 3  # some real (post-warm-up) ML decisions are checked
    pd.testing.assert_frame_equal(a.loc[decided_before], b.loc[decided_before])


def _strategies(prices, better):
    """Champion = buy & hold A; challenger = buy & hold B."""
    p = prices[["SPY", "QQQ"]].copy()
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, (len(p), 2))
    drift = np.array([0.0002, 0.0002 + better])
    p[:] = 100 * np.exp(np.cumsum(noise + drift, axis=0))
    return p, {"Champ": buy_hold(p, "SPY"), "Challenger": buy_hold(p, "QQQ")}


def test_league_promotes_a_clearly_better_strategy(prices, monkeypatch):
    monkeypatch.setattr(league, "EVAL_YEARS", 6)
    p, strats = _strategies(prices, better=0.002)
    new, table = league.evaluate(p, strats, "Champ", cost=0.002)
    assert new == "Challenger"
    assert table.loc["Challenger", "t vs champion"] > league.PROMOTE_T


def test_league_ignores_a_strategy_that_is_only_luckily_ahead(prices, monkeypatch):
    monkeypatch.setattr(league, "EVAL_YEARS", 6)
    p, strats = _strategies(prices, better=0.00005)  # tiny edge, swamped by noise
    new, _ = league.evaluate(p, strats, "Champ", cost=0.002)
    assert new == "Champ"
