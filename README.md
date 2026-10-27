# Trading bot (Trading 212 demo / paper trading)

**Strategy:** momentum top-5 ETF rotation. At each month end, rank 10 asset-class ETFs
(US large/tech/small caps, intl developed, emerging, long/mid bonds, gold,
commodities, real estate) by their average 3/6/12-month return. Hold the top 5
at 20% each, but only those above their 200-day average; any empty slots go
to short-term treasuries.

Backtest 2006–2026 (with fees): about 10% a year vs 11% for the S&P 500, but with a max drawdown of
-16% vs -55% and a better Sharpe ratio (0.85 vs 0.64).

## Files
- `run_backtests.py`: compare all strategies vs buy-and-hold SPY
- `run_daily.py` / `run_daily.bat`: daily run. Rebalances when a new month-end signal appears, then logs equity
- `data/equity_log.csv`: one row per day: bot vs S&P 500, both scaled to £5k
- `data/trades.csv`: every order placed
- `logs/bot.log`: full run log
- `.env`: API keys (never share or commit). `T212_ENV=demo` is enforced in code.

## Schedule (run once in a terminal)
```
schtasks /Create /TN "TradingBot Daily" /TR "C:\Users\prool\trading-bot\run_daily.bat" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 10:00 /F
```
