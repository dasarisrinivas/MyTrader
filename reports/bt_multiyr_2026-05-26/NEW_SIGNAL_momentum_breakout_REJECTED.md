# New signal search — momentum breakout (TESTED & REJECTED, 2026-05-26)

## Goal

Find a *new* long signal to lift trade frequency, data-driven, and ship it to
the live config if the edge was clear and consistent. Outcome: the candidate the
data pointed to does not survive faithful backtesting. **Nothing shipped. Live
config unchanged. The live bot never trades it (default-OFF, not in config.yaml).**

## How the candidate was chosen (data-driven)

Dumped engine features for all 26 months of 15m bars, restricted to RTH bars
(after the opening block) where the current ADX-12 book is **flat**, and screened
seven pre-specified long archetypes with an ATR stop/target forward-outcome proxy.

Every counter-trend / mean-reversion idea **lost** (VWAP fade, RSI-oversold
reclaim, range-low bounce, capitulation dip) — unsurprising for a trend strategy
on ES. The only positive archetype was **momentum**: a break of a rolling 16-bar
high with a clean EMA stack (EMA9>EMA21>EMA50). In the screen it looked strong:

| Screen (flat bars, idealized) | n | WR | exp/trade | total | by year |
|---|---|---|---|---|---|
| brk16 + EMA-stack | 405 | 49.6% | +$5.7 | +$2,288 | 2024 +376 / 2025 +1,728 / 2026 +184 |

## Why it was rejected — the faithful engine backtest

Implemented it as Signal H (`_check_momentum_breakout_long`, default-OFF flag),
routed through the **same** entry/stop/TP/manager machinery as every other signal,
and ran the live-faithful backtest across the same 26-month chunks:

| Chunk | MOM_BRK trades | WR | MOM_BRK PnL | Book before | Book after |
|---|---|---|---|---|---|
| c1 (2024) | 24 | 33.3% | −$365 | +$255 | +$63 |
| c2 (2025) | 22 | 31.8% | −$399 | +$852 | +$482 |
| c3 (2025-26) | 16 | 25.0% | −$208 | +$1,035 | +$830 |
| **Total** | **62** | **30.6%** | **−$973** | **+$2,142** | **+$1,371** |

The signal lost money in **all three** periods and degraded the book by ~$770.
The screen's ~50% win rate collapsed to ~31% in execution.

## Why the screen was misleading

- **No transaction costs** in the proxy; the engine charges commission + 1-tick
  slippage on every one of the 62 trades.
- **Idealized first-touch exit** in the proxy vs the engine's real stop / target /
  max-hold execution.
- **Entry timing:** the real trigger fills at the 15m *close* that breaks the high
  — often an already-extended bar — so a 1.5×ATR stop gets whipsawed. A 31% win
  rate is below the ~33% breakeven for a 2:1 payoff, even before costs.

## Conclusion

There is no free additive long signal here. The existing six long signals already
capture the available trend edge, and the one momentum pocket the screen flagged
is a proxy artifact. **The real frequency lever was the one already pulled:
`ft_adx_min` 18 → 12 (+44% trades, validated on 26 months).**

If more frequency is genuinely wanted, the honest paths are (a) a finer timeframe
(5m decision bars or a 5m-confirmation layer) or (b) accept current frequency —
both bigger architectural changes, not a quick signal add. A breakout *retest /
pullback* entry (instead of buying the extended breakout bar) would mostly
duplicate the existing EMA/trend-cont pullback signals, so it is unlikely to be
additive.

## Status of the code

`Signal H` stays in `es_fifteen_min.py` **default-OFF and frozen**, documented as a
rejected negative (consistent with the project's shadow-experiment discipline).
It is not in `config.yaml`, so the live bot never evaluates it. One genuine engine
fix was kept: `_close_final_position` now handles the 15m-only path (where `df_1m`
is empty) instead of crashing when a position is open at the data's end.

Trade list: `mom_brk_REJECTED_trades.csv`.
