# Strategy research — October 2026

Everything here can be reproduced with `run_candles.py` and `run_backtests.py`.
Raw output is in `data/candle_results.txt` and `data/backtest_results.txt`.

## How we tested (so we don't fool ourselves)
- **No look-ahead:** signals use only data up to that day's close. Trades happen the next day.
- **Costs included:** 0.2% per trade by default. The cost sensitivity test also tries 0.05% and 0%.
- **Two eras:** each strategy has to work in 2007–2015 *and* 2016–2026. If an edge shows up in only one era, it's probably luck.
- **No parameter tuning:** textbook settings only. The robustness grid checks that nearby settings also work.
- **Benchmark:** buy-and-hold S&P 500. If a strategy can't beat that, we might as well just hold the index.

## 1. Candlestick patterns: they don't work
24 patterns, 40 instruments (index/sector ETFs and 30 large US stocks), 2000–2026,
about 400,000 pattern occurrences. Each one is compared with buying on a random day.

| Pattern | Supposed to predict | What actually happened (5 days later, 2013–26) |
|---|---|---|
| Bullish engulfing | Up | **Slightly down** vs random (-0.14%) |
| Morning star | Up | **Down** (-0.18%), in both eras |
| Hammer | Up | Nothing (+0.02%) |
| Three white soldiers | Up | Slightly down |
| Bearish engulfing | Down | **Up** (+0.17%) |
| Three black crows | Down | **Up** (+0.30%), the opposite of what it's supposed to mean |
| Shooting star, hanging man, dark cloud, evening star, doji | Various | Nothing distinguishable from random |

None of the 15 classic reversal patterns predicted what it's supposed to in both eras.
This matches the academic research (e.g. Marshall, Young & Rose 2006, *Journal of
Banking & Finance*: no value in candlestick strategies on US stocks).

**The one real effect: short-term mean reversion.** Stocks and ETFs that close at a
20-day low, or after 3 down days in a row, beat random entries over the next 5–10 days.
They did so in both eras (+0.2% to +0.4%, the strongest effect in the study). Closes at a
20-day *high* do slightly worse. Notice that "three black crows" "working" backwards is
just this same effect.

Caveat: pattern days cluster in market selloffs, so the t-statistics overstate how
reliable the effect is.

## 2. Trading strategies (2007–2026, starting from £5k)

| Strategy | Return/yr | Worst drop | Sharpe | Both eras? |
|---|---|---|---|---|
| **Momentum top-5 ETFs (LIVE)** | **10.6%** | **-16%** | **0.88** | ✅ 0.71 / 1.05 |
| Risk parity | 5.7% | -19% | 0.83 | ✅ but low return |
| 50% momentum + 50% S&P 500 | 11.2% | -30% | 0.82 | ✅ |
| 60/40 stocks/bonds | 8.4% | -31% | 0.78 | ✅ |
| S&P 500 200-day trend filter | 9.3% | -29% | 0.77 | ⚠️ weaker lately |
| Time-series momentum | 6.4% | -15% | 0.75 | ✅ but low return |
| S&P 500 volatility targeting | 8.4% | -30% | 0.75 | ⚠️ only 2016+ |
| Golden cross (50/200 MA) | 8.9% | -34% | 0.68 | ⚠️ |
| *Buy & hold S&P 500* | *11.0%* | *-55%* | *0.63* | |
| Sector momentum | 7.6% | -23% | 0.55 | ❌ |
| Dual momentum (GEM) | 5.5% | -38% | 0.42 | ❌ |
| Donchian/Turtle breakout | 1.7% | -8% | 0.32 | ❌ |
| RSI(2) dip buying | 1.0% | -19% | 0.19 | ❌ |
| Dip-buy (from the candle study) | -0.6% | -24% | -0.05 | ❌ |
| Turn-of-month | -2.3% | -43% | -0.25 | ❌ |

**Short-term strategies (RSI(2), dip-buying, turn-of-month) are eaten by costs.** Even
with *zero* costs and interest on idle cash they make only 4–6% a year, because they sit in cash most of the time.

**Momentum is robust.** Sharpe stays between 0.75 and 0.89 for any top-4 to top-6
setting and any lookback window (see the grid in `backtest_results.txt`). It isn't
one lucky setting.

## Conclusions
1. **Nothing reliably beats the S&P 500 on raw return.** What momentum top-5 does is give
   almost the same return with *less than a third* of the drawdown. That's the realistic edge
   available here: a smoother ride, not more money.
2. Candlestick patterns are folklore. We dropped them.
3. If you want more return and can stomach bigger drops, **50% momentum + 50% S&P 500**
   matched the S&P 500's return with about half the drawdown.
4. Live results will be noisier than the backtest. Judge them after 6–12 months, not weeks.

## Ideas not tested yet
- Momentum on individual stocks (needs survivorship-bias-free data; can't be done honestly with free data)
- Machine-learning models (high risk of overfitting; only worth trying if they beat momentum out-of-sample)
- Adding the momentum strategy's cash slot to UK gilts instead of US treasuries (removes the USD exposure)
