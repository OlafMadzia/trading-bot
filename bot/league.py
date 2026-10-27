"""Champion / challenger league.

Every month all candidate strategies are re-scored on the latest data,
including every month the bot has actually traded, using the trading cost the
bot has measured from its own fills. The live strategy (the "champion") is only
replaced when a challenger beats it by more than luck can explain:

  1. paired t-stat of monthly returns vs the champion > PROMOTE_T, and
  2. a higher Sharpe in BOTH halves of the evaluation window, and
  3. a max drawdown no more than 5 points worse.

That makes the system adapt when something genuinely better emerges,
without lurching after whichever strategy had a lucky few months, which
the research (RESEARCH.md, round 2) showed destroys returns.
"""
import numpy as np
import pandas as pd

from . import strategies as S
from .backtest import run, stats
from .data import UNIVERSE
from .meta import meta_strategy
from .ml import ml_ranker

U = UNIVERSE
DEFAULT_CHAMPION = "Momentum top-5"
PROMOTE_T = 2.0
EVAL_YEARS = 10


def _base(p):
    return {
        "Buy & hold SPY": S.buy_hold(p),
        "60/40": S.sixty_forty(p),
        "Risk parity": S.risk_parity(p),
        "SPY 200d trend": S.trend_200(p),
        "SPY vol-target": S.vol_target(p),
        "Time-series momentum": S.tsmom(p, U),
        "Momentum top-5": S.momentum_rotation(p, U, top=5),
        "Momentum top-3": S.momentum_rotation(p, U, top=3),
        "Vol-scaled momentum top-5": S.momentum_volscaled(p, U, top=5),
        "Adaptive Asset Alloc": S.adaptive_aa(p, U, top=5),
        "VAA-G4": S.vaa(p),
        "DAA": S.daa(p),
    }


def all_strategies(prices):
    base = _base(prices)
    ml = {"ML gradient boosting top-5": ml_ranker(prices, U, top=5, kind="gbm")}
    meta = {
        "Meta-learner (gentle Hedge)": meta_strategy(prices, {**base, **ml}, rule="hedge", eta=2),
        "Meta-learner (top-3, 12m)": meta_strategy(prices, {**base, **ml}, rule="topk", k=3),
    }
    return {**base, **ml, **meta}


def _monthly(d):
    return (1 + d).resample("ME").prod() - 1


def evaluate(prices, strats, champion, cost):
    start = prices.index[-1] - pd.DateOffset(years=EVAL_YEARS)
    p = prices.loc[start:]
    mid = p.index[len(p) // 2]
    daily = {n: run(p, w.loc[start:], cost=cost)[1] for n, w in strats.items()}
    champ_m = _monthly(daily[champion])
    champ_dd = stats((1 + daily[champion]).cumprod(), daily[champion])["MaxDD %"]

    def sharpe(d):
        return d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else 0

    rows = []
    for n, d in daily.items():
        s = stats((1 + d).cumprod(), d)
        diff = _monthly(d) - champ_m
        t = diff.mean() / (diff.std() / np.sqrt(len(diff))) if diff.std() > 0 else 0.0
        sh1, sh2 = sharpe(d.loc[:mid]), sharpe(d.loc[mid:])
        c1, c2 = sharpe(daily[champion].loc[:mid]), sharpe(daily[champion].loc[mid:])
        qualifies = (n != champion and t > PROMOTE_T and sh1 > c1 and sh2 > c2
                     and s["MaxDD %"] >= champ_dd - 5)
        rows.append({"strategy": n, "CAGR %": s["CAGR %"], "Sharpe": s["Sharpe"],
                     "MaxDD %": s["MaxDD %"], "Sharpe 1st half": round(sh1, 2),
                     "Sharpe 2nd half": round(sh2, 2), "t vs champion": round(t, 2),
                     "promote?": qualifies})
    table = pd.DataFrame(rows).set_index("strategy").sort_values("Sharpe", ascending=False)
    winners = table[table["promote?"]]
    new_champ = winners["t vs champion"].idxmax() if len(winners) else champion
    return new_champ, table
