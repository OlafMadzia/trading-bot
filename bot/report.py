"""Builds docs/index.html: a self-contained results dashboard for GitHub Pages.

Regenerated after every daily run. All data is embedded as JSON, so the page
needs no server and no external libraries.
"""
import csv
import datetime as dt
import json
import os

from . import strategies as S
from .backtest import run
from .data import CASH, EXTRA, ROOT, UNIVERSE, load_prices

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "docs", "index.html")


def _csv(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _backtest():
    """Momentum top-5 vs buy-and-hold S&P 500 from £5k, weekly points."""
    p = load_prices(UNIVERSE + EXTRA + [CASH], start="2005-01-01").dropna().loc["2008-04-01":]
    out = {}
    for name, w in {"Momentum top-5": S.momentum_rotation(p, UNIVERSE, top=5),
                    "S&P 500": S.buy_hold(p)}.items():
        eq = run(p, w)[0].resample("W").last()
        dd = eq / eq.cummax() - 1
        out[name] = {"eq": [[d.strftime("%Y-%m-%d"), round(v, 0)] for d, v in eq.items()],
                     "dd": [[d.strftime("%Y-%m-%d"), round(v * 100, 1)] for d, v in dd.items()]}
    return out


def build():
    eq = _csv("equity_log.csv")
    state = json.load(open(os.path.join(DATA, "state.json")))
    league = _csv("league.csv")
    trades = _csv("trades.csv")[-15:][::-1]
    champs = _csv("champion_history.csv")
    payload = {
        "updated": dt.datetime.utcnow().strftime("%d %b %Y, %H:%M UTC"),
        "equity": [{"date": r["date"], "bot": float(r["equity_gbp"]),
                    "spy": float(r["spy_equity_gbp"])} for r in eq],
        "holdings": [h.split(":") for h in (eq[-1]["holdings"].split(";") if eq else []) if h],
        "champion": state.get("champion", "Momentum top-5"),
        "start": state.get("start_date"),
        "start_equity": state.get("start_equity"),
        "cost": state.get("learned_cost"),
        "league": league,
        "trades": trades,
        "switches": sum(1 for c in champs if c["champion"] != c["previous"]),
        "months": len(champs),
        "backtest": _backtest(),
    }
    with open(os.path.join(os.path.dirname(__file__), "dashboard.html"), encoding="utf-8") as f:
        html = f.read().replace("/*__DATA__*/null", json.dumps(payload))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    return OUT


if __name__ == "__main__":
    print(build())
