# Trading bot (Trading 212 demo / paper trading)

Runs automatically on GitHub Actions at 08:30 UTC every weekday (`.github/workflows/daily.yml`).

**How it decides:** at each month end it
1. measures its real trading costs from its own fills (`data/fills.csv`),
2. re-scores 15 strategies (momentum variants, risk parity, 60/40, Keller VAA/DAA,
   a walk-forward ML ranker, meta-learners) on the last 10 years (`data/league.csv`),
3. keeps the current strategy unless a challenger beats it by more than luck (t > 2,
   better in both halves, no much-worse drawdown) (`data/champion_history.csv`),
4. rebalances into the champion's portfolio.

Current champion: **Momentum top-5**. Research and results are in `RESEARCH.md`.

## Files
- `run_daily.py`: the daily run (GitHub runs this)
- `run_backtests.py`, `run_candles.py`, `run_research.py`: research scripts
- `bot/league.py`: champion/challenger selection; `bot/ml.py`: ML ranker; `bot/meta.py`: meta-learners
- `data/equity_log.csv`: one row per day: bot vs S&P 500
- `data/trades.csv`: every order placed
- `.env`: API keys for local runs (never commit). GitHub uses repo secrets instead.

Don't run `run_daily.py` locally while GitHub is also running it (duplicate trades).
Run `git pull` before editing locally.
