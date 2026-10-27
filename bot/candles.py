"""Candlestick and price-action pattern detectors.

Each detector takes one instrument's OHLC DataFrame and returns a boolean
Series that is True on the day the pattern COMPLETES (known at that close).
Definitions follow Nison / Bulkowski; reversal patterns require a prior
trend so they mean what traders say they mean.
"""
import pandas as pd


def _parts(d):
    o, h, l, c = d["Open"], d["High"], d["Low"], d["Close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, float("nan"))
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    avg_body = body.rolling(20).mean()
    return o, h, l, c, body, rng, upper, lower, avg_body


def downtrend(d, n=5):
    return d["Close"] < d["Close"].shift(n)


def uptrend(d, n=5):
    return d["Close"] > d["Close"].shift(n)


def patterns(d):
    o, h, l, c, body, rng, upper, lower, avg = _parts(d)
    bull, bear = c > o, c < o
    o1, c1, h1, l1, body1 = o.shift(1), c.shift(1), h.shift(1), l.shift(1), body.shift(1)
    o2, c2, body2 = o.shift(2), c.shift(2), body.shift(2)
    dn, up = downtrend(d).shift(1), uptrend(d).shift(1)
    small = body < 0.3 * rng
    doji = body < 0.1 * rng
    mid1 = (o1 + c1) / 2

    p = {
        # --- bullish reversal ---
        "Hammer": dn & small & (lower > 2 * body) & (upper < 0.1 * rng),
        "Bullish engulfing": dn & bear.shift(1) & bull & (o <= c1) & (c >= o1) & (body > body1),
        "Piercing line": dn & bear.shift(1) & (body1 > avg) & bull & (o < l1) & (c > mid1) & (c < o1),
        "Morning star": dn & (c2 < o2) & (body2 > avg) & (body1 < 0.3 * body2) & bull & (c > (o2 + c2) / 2),
        "Bullish harami": dn & bear.shift(1) & (body1 > avg) & bull & (o > c1) & (c < o1),
        "Dragonfly doji": dn & doji & (upper < 0.1 * rng) & (lower > 0.6 * rng),
        "Three white soldiers": bull & bull.shift(1) & bull.shift(2) & (c > c1) & (c1 > c2)
                                & (body > avg) & (body1 > avg) & (body2 > avg),
        # --- bearish reversal ---
        "Shooting star": up & small & (upper > 2 * body) & (lower < 0.1 * rng),
        "Hanging man": up & small & (lower > 2 * body) & (upper < 0.1 * rng),
        "Bearish engulfing": up & bull.shift(1) & bear & (o >= c1) & (c <= o1) & (body > body1),
        "Dark cloud cover": up & bull.shift(1) & (body1 > avg) & bear & (o > h1) & (c < mid1) & (c > o1),
        "Evening star": up & (c2 > o2) & (body2 > avg) & (body1 < 0.3 * body2) & bear & (c < (o2 + c2) / 2),
        "Bearish harami": up & bull.shift(1) & (body1 > avg) & bear & (o < c1) & (c > o1),
        "Gravestone doji": up & doji & (lower < 0.1 * rng) & (upper > 0.6 * rng),
        "Three black crows": bear & bear.shift(1) & bear.shift(2) & (c < c1) & (c1 < c2)
                             & (body > avg) & (body1 > avg) & (body2 > avg),
        # --- neutral / price action ---
        "Doji": doji,
        "Inside bar": (h < h1) & (l > l1),
        "Outside bar up": (h > h1) & (l < l1) & (c > h1),
        "NR7 (narrowest range of 7)": rng == rng.rolling(7).min(),
        "3 down closes": (c < c1) & (c1 < c2) & (c2 < c.shift(3)),
        "Gap down >1%": o < l1 * 0.99,
        "Gap up >1%": o > h1 * 1.01,
        "Close at 20d low": c == c.rolling(20).min(),
        "Close at 20d high": c == c.rolling(20).max(),
    }
    return {k: v.fillna(False).astype(bool) for k, v in p.items()}


# What each pattern is *supposed* to predict (+1 up, -1 down, 0 neutral)
EXPECTED = {
    "Hammer": 1, "Bullish engulfing": 1, "Piercing line": 1, "Morning star": 1,
    "Bullish harami": 1, "Dragonfly doji": 1, "Three white soldiers": 1,
    "Shooting star": -1, "Hanging man": -1, "Bearish engulfing": -1,
    "Dark cloud cover": -1, "Evening star": -1, "Bearish harami": -1,
    "Gravestone doji": -1, "Three black crows": -1,
}
