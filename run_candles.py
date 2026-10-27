"""Event study: do candlestick patterns predict returns?

For every pattern occurrence we buy at the NEXT day's open (the earliest you
could act) and measure the return after 1, 5 and 10 days. That is compared
against the average return of buying at any random open on the same
instrument, so a pattern only "works" if it beats a random entry. Results are
split into two eras to check the effect is stable, not a fluke.
"""
import warnings

import numpy as np
import pandas as pd

from bot.candles import EXPECTED, patterns
from bot.data import PATTERN_UNIVERSE, load_ohlc

warnings.filterwarnings("ignore")
HORIZONS = (1, 5, 10)
COST = 0.002  # round trip; pattern trades are short-term so costs matter
ERAS = {"2000-12": ("2000", "2012"), "2013-26": ("2013", "2026")}

data = load_ohlc(PATTERN_UNIVERSE)
rows = []  # one row per (pattern, era, horizon, instrument-event)
for tkr, d in data.items():
    pats = patterns(d)
    entry = d["Open"].shift(-1)
    fwd = {h: d["Close"].shift(-h) / entry - 1 for h in HORIZONS}
    for era, (a, b) in ERAS.items():
        for h in HORIZONS:
            f = fwd[h].loc[a:b]
            base = f.mean()
            for name, mask in pats.items():
                ev = f[mask.loc[a:b]].dropna()
                if len(ev):
                    rows.append(pd.DataFrame({"pattern": name, "era": era, "h": h,
                                              "excess": ev.values - base, "raw": ev.values}))
ev = pd.concat(rows)


def summarise(g):
    n = len(g)
    m = g["excess"].mean()
    t = m / (g["excess"].std() / np.sqrt(n)) if n > 1 else 0
    return pd.Series({"n": n, "excess %": m * 100, "t": t,
                      "win %": (g["raw"] > 0).mean() * 100})


res = ev.groupby(["pattern", "h", "era"]).apply(summarise).unstack("era")
pd.set_option("display.width", 250)
pd.set_option("display.float_format", "{:.2f}".format)

print(f"Instruments: {len(data)}   Baseline = random entry on the same instrument\n")
for h in HORIZONS:
    t = res.xs(h, level="h")
    t = t[[("n", "2000-12"), ("n", "2013-26"), ("excess %", "2000-12"), ("excess %", "2013-26"),
           ("t", "2000-12"), ("t", "2013-26"), ("win %", "2013-26")]]
    t.columns = ["n 00-12", "n 13-26", "exc% 00-12", "exc% 13-26", "t 00-12", "t 13-26", "win% 13-26"]
    t["expected"] = [EXPECTED.get(p, 0) for p in t.index]
    # "works" = right sign in BOTH eras, |t|>2 in both, and beats round-trip cost
    good = (np.sign(t["exc% 00-12"]) == np.sign(t["exc% 13-26"])) & \
           (t[["t 00-12", "t 13-26"]].abs().min(axis=1) > 2) & \
           (t[["exc% 00-12", "exc% 13-26"]].abs().min(axis=1) > COST * 100)
    t["consistent edge?"] = np.where(good, "YES", "")
    print(f"=== Holding {h} day(s) after the signal ===")
    print(t.sort_values("t 13-26").to_string(), "\n")
