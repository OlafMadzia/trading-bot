"""Daily paper-trading runner.

Each month it re-runs the champion/challenger league (bot/league.py) to decide
which strategy to trade, using the trading cost it has learned from its own
fills. Signals come from long-history US ETFs; trades are placed in the
UK-listed (UCITS) equivalents that Trading 212 actually offers.
"""
import csv
import datetime as dt
import json
import os
import time

import yfinance as yf

from . import league
from . import strategies as S
from .backtest import COST_PER_TRADE
from .data import CASH, EXTRA, ROOT, UNIVERSE, load_prices
from .t212 import T212

# signal ticker -> (Trading 212 ticker, Yahoo symbol, quote currency)
MAP = {
    "SPY": ("VUSAl_EQ", "VUSA.L", "GBP"),
    "QQQ": ("EQQQl_EQ", "EQQQ.L", "GBX"),
    "IWM": ("R2USl_EQ", "R2US.L", "USD"),
    "EFA": ("XMWXl_EQ", "XMWX.L", "GBP"),
    "EEM": ("VFEMl_EQ", "VFEM.L", "GBP"),
    "TLT": ("IDTLl_EQ", "IDTL.L", "USD"),
    "IEF": ("IBTMl_EQ", "IBTM.L", "GBP"),
    "GLD": ("SGLNl_EQ", "SGLN.L", "GBX"),
    "DBC": ("CMODl_EQ", "CMOD.L", "USD"),
    "VNQ": ("IUSPl_EQ", "IUSP.L", "GBX"),
    "SHY": ("IBTSl_EQ", "IBTS.L", "GBP"),
    "AGG": ("IUAGl_EQ", "IUAG.L", "USD"),
    "LQD": ("LQGHl_EQ", "LQGH.L", "GBP"),
    "HYG": ("SHYUl_EQ", "SHYU.L", "GBP"),
    "VGK": ("VEURl_EQ", "VEUR.L", "GBP"),
    "EWJ": ("VJPNl_EQ", "VJPN.L", "GBP"),
}
T212_TO_SIG = {v[0]: k for k, v in MAP.items()}
BENCH = "VUSA.L"  # buy-and-hold S&P 500 benchmark
MIN_TRADE_GBP = 1.0     # T212's minimum order value
MIN_TRADE_FRAC = 0.03   # skip rebalancing trades smaller than 3% of equity
COST_PRIOR_WEIGHT = 20  # learned cost = mix of the 0.2% prior and observed fills

DATA = os.path.join(ROOT, "data")
STATE = os.path.join(DATA, "state.json")
EQUITY_LOG = os.path.join(DATA, "equity_log.csv")
TRADES_LOG = os.path.join(DATA, "trades.csv")
FILLS_LOG = os.path.join(DATA, "fills.csv")
LEAGUE_LOG = os.path.join(DATA, "league.csv")
CHAMPION_LOG = os.path.join(DATA, "champion_history.csv")


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    with open(os.path.join(ROOT, "logs", "bot.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def _append_csv(path, row):
    rows = _read_csv(path)
    if rows and set(row) <= set(rows[0]):
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=list(rows[0]), restval="").writerow(row)
    else:  # new file, or new columns: rewrite with the merged header
        _write_csv(path, rows + [row])


def _upsert_daily(path, row):
    """Append today's row, replacing an earlier row for the same date."""
    _write_csv(path, [r for r in _read_csv(path) if r["date"] != row["date"]] + [row])


def last_price(symbol):
    return float(yf.Ticker(symbol).fast_info.last_price)


def fx_rates():
    return {"GBP": 1.0, "GBX": 0.01,
            "USD": 1.0 / last_price("GBPUSD=X"), "EUR": 1.0 / last_price("GBPEUR=X")}


def equity_gbp(api, fx):
    """Account total minus anything the bot doesn't manage (e.g. manual trades)."""
    instruments = json.load(open(os.path.join(DATA, "instruments.json")))
    ccy = {i["ticker"]: i["currencyCode"] for i in instruments}
    other = sum(p["quantity"] * p["currentPrice"] * fx.get(ccy.get(p["ticker"]), 1.0)
                for p in api.positions() if p["ticker"] not in T212_TO_SIG)
    return api.cash()["total"] - other


def signal_prices():
    """Full daily history (completed bars only) and the latest month-end date."""
    prices = load_prices(UNIVERSE + EXTRA + [CASH], start="2005-01-01", refresh=True).dropna()
    if prices.index[-1].date() >= dt.date.today():
        prices = prices.iloc[:-1]  # drop today's still-forming bar
    me = prices.index[S.month_ends(prices)]
    # the final row is always flagged as a "month end"; ignore it unless the
    # month really has ended (the next bar would be in a new month)
    signal_date = me[-2] if me[-1] == prices.index[-1] and \
        prices.index[-1].month == dt.date.today().month else me[-1]
    return prices, signal_date


# ---------------------------------------------------------------- learning

def learn_costs(api, state):
    """Match our orders to Trading 212's fill records and measure the real
    cost of each trade: fees (e.g. FX conversion) + slippage vs the price we
    expected. The estimate starts at the 0.2% prior and moves towards what we
    actually observe as fills accumulate."""
    ours = {r["order_id"]: r for r in _read_csv(TRADES_LOG) if r.get("order_id")}
    known = {r["order_id"] for r in _read_csv(FILLS_LOG)}
    todo = set(ours) - known
    path, pages = "/equity/history/orders?limit=50", 0
    while todo and path and pages < 5:
        res = api._req("GET", path.replace("/api/v0", ""))
        for it in res.get("items", []):
            oid = str(it["order"]["id"])
            fill = it.get("fill")
            if oid not in todo or not fill or it["order"]["status"] != "FILLED":
                continue
            t = ours[oid]
            wi = fill["walletImpact"]
            value = abs(wi["netValue"])
            fees = -sum(x["quantity"] for x in wi.get("taxes", []))
            side = 1 if it["order"]["side"] == "BUY" else -1
            ref = float(t["ref_price"]) if t.get("ref_price") else None
            slip = side * (fill["price"] / ref - 1) if ref else 0.0
            _append_csv(FILLS_LOG, {
                "order_id": oid, "time": fill["filledAt"], "ticker": t["ticker"],
                "side": it["order"]["side"], "qty": fill["quantity"],
                "ref_price": ref or "", "fill_price": fill["price"],
                "value_gbp": round(value, 2), "fees_gbp": round(fees, 4),
                "cost_frac": round(fees / value + slip, 6) if value else 0.0})
            todo.discard(oid)
        path, pages = res.get("nextPagePath"), pages + 1
        time.sleep(11)  # history endpoint allows ~6 requests a minute

    costs = [float(r["cost_frac"]) for r in _read_csv(FILLS_LOG)]
    n = len(costs)
    observed = sum(costs) / n if n else COST_PER_TRADE
    learned = (n * observed + COST_PRIOR_WEIGHT * COST_PER_TRADE) / (n + COST_PRIOR_WEIGHT)
    learned = min(max(learned, 0.0005), 0.01)
    state["learned_cost"] = learned
    log(f"Cost learning: {n} fills, observed {observed:.3%}/trade -> using {learned:.3%}")
    return learned


def run_league(prices, state, cost, signal_date):
    champion = state.get("champion", league.DEFAULT_CHAMPION)
    strats = league.all_strategies(prices)
    new_champ, table = league.evaluate(prices, strats, champion, cost)
    table.insert(0, "as_of", signal_date.date().isoformat())
    table.to_csv(LEAGUE_LOG)
    top = table.head(5)
    log("League (last 10y, Sharpe): " + "; ".join(f"{n} {r.Sharpe}" for n, r in top.iterrows()))
    if new_champ != champion:
        log(f"PROMOTION: '{new_champ}' beat '{champion}' "
            f"(t={table.loc[new_champ, 't vs champion']}) -> new live strategy")
    else:
        log(f"Champion stays '{champion}' (no challenger beat it by more than luck)")
    _append_csv(CHAMPION_LOG, {"date": signal_date.date().isoformat(), "champion": new_champ,
                               "previous": champion, "cost_used": round(cost, 5)})
    state["champion"] = new_champ
    w = strats[new_champ].loc[signal_date]
    return {k: v for k, v in w.items() if v > 1e-4}


# ---------------------------------------------------------------- trading

def portfolio(api, fx):
    """Bot-managed holdings (ignores anything not in MAP) valued in GBP."""
    held = {}
    for p in api.positions():
        sig = T212_TO_SIG.get(p["ticker"])
        if sig:
            ccy = MAP[sig][2]
            held[sig] = {"qty": p["quantity"], "price": p["currentPrice"],
                         "gbp": p["quantity"] * p["currentPrice"] * fx[ccy]}
    return held


def place(api, sig, qty, note, ref_price):
    ticker = MAP[sig][0]
    decimals = 3
    for _ in range(8):
        q = int(qty * 10**decimals) / 10**decimals  # round towards zero
        if q == 0:
            return None
        try:
            res = api.market_order(ticker, q)
            log(f"ORDER {note} {ticker} qty={q} -> {res.get('status')}")
            _append_csv(TRADES_LOG, {"time": dt.datetime.now().isoformat(timespec="seconds"),
                                     "side": note, "signal": sig, "ticker": ticker,
                                     "qty": q, "status": res.get("status"),
                                     "order_id": res.get("id", ""), "ref_price": ref_price})
            time.sleep(1.5)
            return res
        except RuntimeError as e:
            msg = str(e).lower()
            if "insufficient" in msg and qty > 0:
                qty *= 0.97  # T212 reserves a buffer on non-GBP buys
            elif ("quantity" in msg or "precision" in msg) and decimals > 0:
                decimals -= 1
            else:
                raise
    return None


def rebalance(api, targets, fx):
    errors = 0
    api.wait_for_fills()
    held = portfolio(api, fx)
    equity = equity_gbp(api, fx)
    min_trade = max(MIN_TRADE_GBP, MIN_TRADE_FRAC * equity)

    # 1) sells first so the cash is there for the buys
    for sig, h in held.items():
        want = equity * targets.get(sig, 0.0)
        diff = want - h["gbp"]
        if diff < -min_trade or (want == 0 and h["gbp"] > 0):
            qty = h["qty"] if want == 0 else -diff / h["gbp"] * h["qty"]
            try:
                place(api, sig, -qty, "SELL", h["price"])
            except RuntimeError as e:
                errors += 1
                log(f"ERROR selling {sig}: {e}")

    api.wait_for_fills()
    held = portfolio(api, fx)

    # 2) buys; each one waits for the previous fill (T212 reserves extra cash
    # for pending orders) and is capped to the free cash with a fee buffer
    buys = {}
    for sig, wgt in targets.items():
        diff = equity * wgt - held.get(sig, {}).get("gbp", 0.0)
        if diff > min_trade:
            buys[sig] = diff
    for i, (sig, gbp) in enumerate(buys.items()):
        api.wait_for_fills()
        free = api.cash()["free"] * 0.985
        remaining = sum(list(buys.values())[i:])
        gbp = gbp * min(1.0, free / remaining)
        ccy, ysym = MAP[sig][2], MAP[sig][1]
        price = held[sig]["price"] if sig in held else last_price(ysym)
        try:
            place(api, sig, gbp / (price * fx[ccy]), "BUY", price)
        except RuntimeError as e:
            errors += 1
            log(f"ERROR buying {sig}: {e}")
    return errors


def run_daily():
    os.makedirs(DATA, exist_ok=True)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    state.setdefault("champion", league.DEFAULT_CHAMPION)
    api = T212()
    fx = fx_rates()

    prices, signal_date = signal_prices()
    if state.get("last_signal") != signal_date.date().isoformat():
        log(f"New month-end signal {signal_date.date()} -> learning + league + rebalance")
        cost = learn_costs(api, state)
        targets = run_league(prices, state, cost, signal_date)
        log(f"Targets ({state['champion']}): " +
            ", ".join(f"{k} {v:.0%}" for k, v in sorted(targets.items(), key=lambda x: -x[1])))
        errors = rebalance(api, targets, fx)
        if errors == 0:
            state["last_signal"] = signal_date.date().isoformat()
        else:
            log(f"{errors} order(s) failed; will retry next run")
    else:
        log(f"No new signal; holding ({state['champion']})")

    api.wait_for_fills()
    held = portfolio(api, fx)
    equity = equity_gbp(api, fx)
    bench_px = last_price(BENCH)
    if "start_equity" not in state:
        state.update(start_date=dt.date.today().isoformat(),
                     start_equity=equity, bench_start=bench_px)
    bot_ret = equity / state["start_equity"] - 1
    bench_ret = bench_px / state["bench_start"] - 1
    _upsert_daily(EQUITY_LOG, {
        "date": dt.date.today().isoformat(),
        "equity_gbp": round(equity, 2),
        "bot_return_pct": round(bot_ret * 100, 2),
        "spy_return_pct": round(bench_ret * 100, 2),
        "spy_equity_gbp": round(state["start_equity"] * (1 + bench_ret), 2),
        "holdings": ";".join(f"{k}:{h['gbp']:.2f}" for k, h in held.items()),
        "strategy": state["champion"],
    })
    json.dump(state, open(STATE, "w"), indent=2)
    log(f"Equity £{equity:,.2f}  bot {bot_ret:+.2%}  vs S&P 500 {bench_ret:+.2%}")
