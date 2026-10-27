import numpy as np
import pandas as pd
import pytest

from bot.data import CASH, EXTRA, UNIVERSE


@pytest.fixture(scope="session")
def prices():
    """Synthetic daily prices (random walks with different drifts) so tests
    are fast, deterministic and never touch the network."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2010-01-01", "2016-12-31")
    cols = UNIVERSE + EXTRA + [CASH]
    drift = rng.uniform(-0.0002, 0.0006, len(cols))
    rets = rng.normal(drift, 0.01, (len(idx), len(cols)))
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)


def change_future(prices, cut, seed=0):
    """Rewrite history from `cut` onwards: an abrupt, different jump for every
    asset, then a fresh random walk. Any strategy that peeks at the future,
    even one day ahead, will make different decisions before `cut`."""
    rng = np.random.default_rng(seed)
    out = prices.copy()
    n = len(out.loc[cut:])
    jump = rng.uniform(0.3, 3.0, prices.shape[1])
    walk = np.exp(np.cumsum(rng.normal(0, 0.03, (n, prices.shape[1])), axis=0))
    out.loc[cut:] = out.loc[cut:].values * jump * walk
    return out
