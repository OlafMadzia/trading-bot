"""Candidate strategies. Each takes a price DataFrame and returns target weights.

Parameters are textbook defaults, deliberately NOT tuned to the data, so the
backtest isn't just a curve-fit of history.
"""
import pandas as pd

from .data import CASH


def month_ends(prices):
    """True on the last trading day of each month."""
    m = prices.index.to_series().dt.month
    return m != m.shift(-1)


def _rebalance_monthly(prices, fn):
    w = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for d in prices.index[month_ends(prices)]:
        w.loc[d] = 0.0
        hist = prices.loc[:d]
        if len(hist) < 253:  # need a year of history before trading
            continue
        for t, x in fn(hist).items():
            w.loc[d, t] = x
    return w.ffill().fillna(0.0)


def buy_hold(prices, ticker="SPY"):
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[ticker] = 1.0
    return w


def trend_200(prices, ticker="SPY"):
    """Hold SPY when it's above its 200-day average (checked monthly), else cash."""
    def pick(p):
        s = p[ticker]
        return {ticker: 1.0} if s.iloc[-1] > s.iloc[-200:].mean() else {CASH: 1.0}
    return _rebalance_monthly(prices, pick)


def dual_momentum(prices):
    """Gary Antonacci's GEM: US vs international stocks by 12-month return;
    if the winner hasn't beaten cash, hold bonds instead."""
    def pick(p):
        r = p.iloc[-1] / p.iloc[-253] - 1
        best = "SPY" if r["SPY"] >= r["EFA"] else "EFA"
        return {best: 1.0} if r[best] > r[CASH] else {"IEF": 1.0}
    return _rebalance_monthly(prices, pick)


def momentum_rotation(prices, universe, top=3, lookbacks=(63, 126, 252)):
    """Each month hold the top-N assets by average of 3/6/12-month return,
    but only those above their 200-day average; empty slots go to cash."""
    def pick(p):
        if len(p) < max(lookbacks) + 1:
            return {CASH: 1.0}
        u = p[universe]
        score = sum(u.iloc[-1] / u.iloc[-n] - 1 for n in lookbacks) / len(lookbacks)
        trend_ok = u.iloc[-1] > u.iloc[-200:].mean()
        winners = score[trend_ok & (score > 0)].nlargest(top).index
        w = {t: 1.0 / top for t in winners}
        if len(winners) < top:
            w[CASH] = (top - len(winners)) / top
        return w
    return _rebalance_monthly(prices, pick)


def _single(prices, ticker, pos):
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[ticker] = pos
    return w


def sma_cross(prices, ticker="SPY", fast=50, slow=200):
    """Golden cross: hold while the 50-day average is above the 200-day."""
    s = prices[ticker]
    return _single(prices, ticker, (s.rolling(fast).mean() > s.rolling(slow).mean()).astype(float))


def vol_target(prices, ticker="SPY", target=0.12, lookback=20):
    """Moreira & Muir (2017): scale exposure down when recent volatility is high."""
    vol = prices[ticker].pct_change().rolling(lookback).std() * (252 ** 0.5)
    return _single(prices, ticker, (target / vol).clip(upper=1.0).fillna(0.0))


def turn_of_month(prices, ticker="SPY"):
    """Hold only the last trading day + first 3 trading days of each month."""
    idx = prices.index.to_series()
    first = idx.groupby([idx.dt.year, idx.dt.month]).rank(method="first")
    last = idx.groupby([idx.dt.year, idx.dt.month]).rank(method="first", ascending=False)
    # weight set at close of day t is held on day t+1, so shift the window back a day
    hold = ((first <= 3) | (last == 1)).shift(-1).fillna(False)
    return _single(prices, ticker, hold.astype(float))


def sixty_forty(prices):
    def pick(p):
        return {"SPY": 0.6, "IEF": 0.4}
    return _rebalance_monthly(prices, pick)


def risk_parity(prices, assets=("SPY", "TLT", "GLD", "DBC", "IEF"), lookback=60):
    """Inverse-volatility weights, so each asset contributes similar risk."""
    def pick(p):
        vol = p[list(assets)].pct_change().iloc[-lookback:].std()
        inv = 1 / vol
        return (inv / inv.sum()).to_dict()
    return _rebalance_monthly(prices, pick)


def tsmom(prices, universe):
    """Moskowitz/Ooi/Pedersen time-series momentum: equal slots, each asset is
    held only if its own 12-month return is positive (else that slot is cash)."""
    def pick(p):
        r = p[universe].iloc[-1] / p[universe].iloc[-253] - 1
        w = {t: 1 / len(universe) for t in r[r > 0].index}
        w[CASH] = 1 - sum(w.values())
        return w
    return _rebalance_monthly(prices, pick)


def donchian(prices, universe, entry=55, exit_=20):
    """Turtle-style breakout per asset: buy a 55-day high, sell a 20-day low."""
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for t in universe:
        s = prices[t]
        hi, lo = s.rolling(entry).max(), s.rolling(exit_).min()
        pos, inpos = [], False
        for i in range(len(s)):
            if not inpos and s.iloc[i] >= hi.iloc[i]:
                inpos = True
            elif inpos and s.iloc[i] <= lo.iloc[i]:
                inpos = False
            pos.append(inpos)
        w[t] = pd.Series(pos, index=s.index).astype(float) / len(universe)
    return w


def dip_buy(prices, universe, hold=5, max_pos=4):
    """From the candlestick study: closes at a 20-day low in an uptrend (above
    the 200-day average) tend to bounce. Buy, hold N days, max 4 at a time."""
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    sig = pd.DataFrame({t: (prices[t] == prices[t].rolling(20).min()) &
                        (prices[t] > prices[t].rolling(200).mean()) for t in universe})
    open_until = {}
    for i, d in enumerate(prices.index):
        open_until = {t: e for t, e in open_until.items() if e > i}
        for t in sig.columns[sig.iloc[i].values]:
            if t not in open_until and len(open_until) < max_pos:
                open_until[t] = i + hold
        for t in open_until:
            w.iat[i, w.columns.get_loc(t)] = 1 / max_pos
    return w


def _mom13612(p):
    """Keller's fast momentum score: weights recent months more heavily."""
    r = lambda n: p.iloc[-1] / p.iloc[-1 - n] - 1
    return 12 * r(21) + 4 * r(63) + 2 * r(126) + r(252)


def vaa(prices, offensive=("SPY", "EFA", "EEM", "AGG"), defensive=("LQD", "IEF", "SHY")):
    """Keller & Keuning (2017) Vigilant Asset Allocation G4: if ANY offensive
    asset has negative momentum, the whole portfolio goes defensive."""
    def pick(p):
        off, de = _mom13612(p[list(offensive)]), _mom13612(p[list(defensive)])
        return {off.idxmax(): 1.0} if (off > 0).all() else {de.idxmax(): 1.0}
    return _rebalance_monthly(prices, pick)


def daa(prices, risky=("SPY", "IWM", "QQQ", "VGK", "EWJ", "EEM", "VNQ", "DBC",
                       "GLD", "TLT", "HYG", "LQD"),
        canary=("EEM", "AGG"), defensive=("SHY", "IEF", "LQD"), top=6):
    """Keller (2018) Defensive Asset Allocation: two 'canary' assets decide how
    much goes defensive (0, 50 or 100%); the rest goes to the top-6 risky."""
    def pick(p):
        bad = int((_mom13612(p[list(canary)]) <= 0).sum())
        def_frac = bad / 2
        w = {}
        if def_frac < 1:
            best = _mom13612(p[list(risky)]).nlargest(top).index
            w.update({t: (1 - def_frac) / top for t in best})
        if def_frac > 0:
            d = _mom13612(p[list(defensive)]).idxmax()
            w[d] = w.get(d, 0) + def_frac
        return w
    return _rebalance_monthly(prices, pick)


def adaptive_aa(prices, universe, top=5, lookback=126, vol_lb=60):
    """Butler et al. (2012) Adaptive Asset Allocation: pick the top momentum
    assets, then weight them by inverse volatility instead of equally."""
    def pick(p):
        u = p[universe]
        mom = u.iloc[-1] / u.iloc[-lookback] - 1
        best = mom[(mom > 0) & (u.iloc[-1] > u.iloc[-200:].mean())].nlargest(top).index
        if len(best) == 0:
            return {CASH: 1.0}
        inv = 1 / u[best].pct_change().iloc[-vol_lb:].std()
        w = (inv / inv.sum() * len(best) / top).to_dict()
        if len(best) < top:
            w[CASH] = 1 - sum(w.values())
        return w
    return _rebalance_monthly(prices, pick)


def momentum_volscaled(prices, universe, top=5):
    """Risk-adjusted momentum: rank by 6-12m return divided by volatility
    (Barroso & Santa-Clara style), with the same 200d trend filter."""
    def pick(p):
        u = p[universe]
        ret = (u.iloc[-1] / u.iloc[-126] - 1 + u.iloc[-1] / u.iloc[-252] - 1) / 2
        score = ret / (u.pct_change().iloc[-126:].std() * 252 ** 0.5)
        ok = (u.iloc[-1] > u.iloc[-200:].mean()) & (score > 0)
        winners = score[ok].nlargest(top).index
        w = {t: 1 / top for t in winners}
        if len(winners) < top:
            w[CASH] = (top - len(winners)) / top
        return w
    return _rebalance_monthly(prices, pick)


def with_vol_target(weights, prices, target=0.12, lookback=20):
    """Overlay: scale any strategy's weights to a target volatility."""
    rets = (weights.shift(1) * prices.pct_change()).sum(axis=1)
    vol = rets.rolling(lookback).std() * (252 ** 0.5)
    return weights.mul((target / vol).clip(upper=1.0).fillna(1.0), axis=0)


def blend(*parts):
    """Mix strategies, e.g. blend((0.5, w1), (0.5, w2))."""
    return sum(f * w for f, w in parts)


def rsi2_reversion(prices, ticker="SPY"):
    """Larry Connors' RSI(2): in an uptrend (above 200d MA), buy when RSI(2)
    is very oversold, sell when price closes above its 5-day average."""
    s = prices[ticker]
    delta = s.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    ma200, ma5 = s.rolling(200).mean(), s.rolling(5).mean()
    pos, inpos = [], False
    for i in range(len(s)):
        if not inpos and s.iloc[i] > ma200.iloc[i] and rsi.iloc[i] < 10:
            inpos = True
        elif inpos and s.iloc[i] > ma5.iloc[i]:
            inpos = False
        pos.append(1.0 if inpos else 0.0)
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[ticker] = pos  # out of the market = idle cash (avoids double FX fees)
    return w
