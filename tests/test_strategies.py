import numpy as np
import pytest

from bot import strategies as S
from bot.data import UNIVERSE
from bot.strategies import month_ends
from tests.conftest import change_future

U = UNIVERSE
STRATS = {
    "buy_hold": lambda p: S.buy_hold(p),
    "trend_200": S.trend_200,
    "momentum_top5": lambda p: S.momentum_rotation(p, U, top=5),
    "momentum_volscaled": lambda p: S.momentum_volscaled(p, U),
    "adaptive_aa": lambda p: S.adaptive_aa(p, U),
    "risk_parity": S.risk_parity,
    "tsmom": lambda p: S.tsmom(p, U),
    "vaa": S.vaa,
    "daa": S.daa,
    "dual_momentum": S.dual_momentum,
    "vol_target": S.vol_target,
}


@pytest.mark.parametrize("name", STRATS)
def test_weights_are_valid(prices, name):
    w = STRATS[name](prices)
    assert (w >= -1e-9).all().all(), "no short positions"
    assert (w.sum(axis=1) <= 1 + 1e-9).all(), "never more than 100% invested"


@pytest.mark.parametrize("name", STRATS)
def test_no_look_ahead(prices, name):
    """Changing prices AFTER a date must not change any decision made BEFORE it."""
    cut = prices.index[1000]
    a = STRATS[name](prices)
    b = STRATS[name](change_future(prices, cut))
    before = prices.index[prices.index < cut]
    np.testing.assert_allclose(a.loc[before].values, b.loc[before].values)


def test_momentum_picks_the_strongest_assets(prices):
    p = prices.copy()
    p["GLD"] = 100 * 1.002 ** np.arange(len(p))  # make gold the clear winner
    w = S.momentum_rotation(p, U, top=5)
    last_me = p.index[month_ends(p)][-2]
    assert w.loc[last_me, "GLD"] == pytest.approx(0.2)
