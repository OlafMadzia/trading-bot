"""Walk-forward machine-learning asset ranker.

Every month-end it:
  1. builds features for each asset (momentum over several windows,
     volatility, trend, drawdown, cross-sectional ranks, market regime),
  2. retrains a model on ALL PAST months whose outcome is already known,
  3. predicts which assets will beat the average next month,
  4. holds the top-N.
The model never sees a month's outcome before that month has finished.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import CASH
from .strategies import month_ends

MIN_TRAIN_MONTHS = 36


def features(prices, date, assets, market="SPY"):
    p = prices.loc[:date]
    u = p[assets]
    last = u.iloc[-1]
    r = lambda n: last / u.iloc[-1 - n] - 1
    f = pd.DataFrame({
        "r21": r(21), "r63": r(63), "r126": r(126), "r252": r(252),
        "r252_skip1m": u.iloc[-22] / u.iloc[-253] - 1,
        "vol63": u.pct_change().iloc[-63:].std() * np.sqrt(252),
        "dist_sma200": last / u.iloc[-200:].mean() - 1,
        "dd252": last / u.iloc[-252:].max() - 1,
    })
    for c in ("r21", "r63", "r126", "r252"):
        f[c + "_rank"] = f[c].rank(pct=True)
    f["sharpe126"] = f["r126"] / (f["vol63"] + 1e-6)
    m = p[market]
    f["mkt_r63"] = m.iloc[-1] / m.iloc[-64] - 1
    f["mkt_vol21"] = m.pct_change().iloc[-21:].std() * np.sqrt(252)
    f["mkt_above_sma200"] = float(m.iloc[-1] > m.iloc[-200:].mean())
    return f


def make_model(kind):
    if kind == "gbm":
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150,
                                             min_samples_leaf=40, l2_regularization=1.0,
                                             random_state=0)
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def build_dataset(prices, assets):
    me = prices.index[month_ends(prices)][:-1]  # last row isn't a real month end
    me = me[me >= prices.index[260]]
    X, y, dates = [], [], []
    for i, d in enumerate(me):
        f = features(prices, d, assets)
        if i + 1 < len(me):
            fwd = prices.loc[me[i + 1], assets] / prices.loc[d, assets] - 1
            target = fwd - fwd.mean()  # beat-the-average, i.e. a ranking target
        else:
            target = pd.Series(np.nan, index=assets)
        X.append(f)
        y.append(target)
        dates.append(d)
    return dates, X, y


def ml_ranker(prices, assets, top=5, kind="gbm", trend_filter=True, refit_every=1):
    """Returns daily target weights like every other strategy."""
    assets = list(assets) + ([CASH] if CASH not in assets else [])
    dates, X, y = build_dataset(prices, assets)
    w = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    model, preds_log = None, {}
    for k, d in enumerate(dates):
        w.loc[d] = 0.0
        if k < MIN_TRAIN_MONTHS:
            w.loc[d, CASH] = 1.0
            continue
        if model is None or k % refit_every == 0:
            # train only on months whose outcome was known by date d
            Xtr = pd.concat(X[:k]).values
            ytr = pd.concat(y[:k]).values
            model = make_model(kind).fit(Xtr, ytr)
        pred = pd.Series(model.predict(X[k].values), index=assets)
        preds_log[d] = pred
        pick = pred.nlargest(top)
        if trend_filter:  # same safety rule as the live strategy
            ok = X[k]["dist_sma200"] > 0
            pick = pick[ok[pick.index] | (pick.index == CASH)]
        for t in pick.index:
            w.loc[d, t] = 1 / top
        w.loc[d, CASH] += 1 - w.loc[d].sum()
    w = w.ffill().fillna(0.0)
    w.attrs["predictions"] = pd.DataFrame(preds_log).T
    return w


def ml_live_pick(prices, assets, top=5, kind="gbm"):
    """For the live bot: train on everything known today, predict next month."""
    w = ml_ranker(prices, assets, top=top, kind=kind)
    return w.iloc[-1]
