# What to improve — from the latest-code replay (2026-05-24)

Latest code over Feb 2025–Jan 16 2026: **166 trades, 50.6% WR, +$66.60** (MES $5/pt). The book is
essentially breakeven. The loss is **not** spread across the system — it is concentrated in one
signal, and within that signal, in a few specific conditions.

## Bottom line

| Change | Trades | WR | Net PnL |
|---|---|---|---|
| As-is (latest code) | 166 | 50.6% | +$66.60 |
| TREND_CONT_LONG: skip opening 30 min **and** Fridays | 132 | 58.3% | **+$1,049.50** |
| …also require ATR ≥ 6 | 118 | 58.5% | +$996.80 |
| …also require ADX ≥ 30 | 73 | 61.6% | +$941.10 |
| Drop TREND_CONT_LONG entirely | 61 | 59.0% | +$706.10 |

**Two time-of-day filters on one signal turn +$66 into +$1,050 — a ~16× improvement — while still
trading 132 of 166 setups.** That is the headline recommendation.

## Where the money leaks: TREND_CONT_LONG

TREND_CONT_LONG is 105 of 166 trades (63% of volume), 45.7% WR, **−$639.50**. The other three
signals together make **+$706**. So TREND_CONT_LONG gives back almost everything the book earns.

Its loss is concentrated, and each pocket is **consistent across all quarters** (not a single bad
stretch, not one tail loss):

| Condition | n | WR | Net PnL | Consistency |
|---|---|---|---|---|
| Opening 30 min (hour 8 CT / 9:30–10:00 ET) | 17 | **11.8%** | −$668 | negative all 5 quarters; 15/17 losers; worst single −$97 |
| Fridays | 26 | **23.1%** | −$741 | negative all 4 quarters; 20/26 losers |
| Low ATR (< 6 pts) | 34 | 32.4% | −$543 | continuation with no volatility = no follow-through |
| ADX < 30 | 65 | ~42% | −$687 | only ADX ≥ 30 is positive (56.5%, +$47) |

These overlap heavily (Friday opens, quiet-tape opens), so they are quantified *jointly* in the
table above, not summed.

The realized R:R is ~1:1 (avg win +$52, avg loss −$51), so TREND_CONT_LONG needs >50% WR to profit
and only gets 45.7%. It is a **hit-rate** problem driven by taking the signal in the wrong
conditions, not a stop/target sizing problem.

## Recommendations, ranked by impact × robustness

1. **Block TREND_CONT_LONG in the opening 30 minutes and on Fridays.** Biggest, cleanest win
   (+$66 → +$1,050). Both are low-degrees-of-freedom *time* gates, the codebase already has the
   pattern (`LATE_AFTERNOON_BLOCK`, `MONDAY_BLOCK`), and both have an economic story: the first 30
   min is opening whipsaw/price discovery, and Fridays carry weekend de-risking with weak trend
   follow-through. The edge holds in every quarter — the strongest evidence against curve-fitting.

2. **Add an ATR floor (≈6 pts) and/or ADX floor (≈30) to TREND_CONT_LONG.** A continuation signal
   needs momentum to continue into; in quiet/weak-trend tape it just round-trips to the stop. These
   push WR to ~60–62% but cut more trades and are threshold-tuned, so treat them as secondary to the
   time gates and validate the exact cutoffs out-of-sample.

3. **If the filters don't survive validation, cut TREND_CONT_LONG.** Even naively removing it yields
   +$706 at 59% WR. Keeping it unfiltered is the one choice the data clearly rejects.

## What to keep (don't touch)

The three profitable signals are fine — leave them alone:

- **EMA9_PB_LONG** — 76.9% WR, +$349, only 2 stops in 13 trades. Strongest edge, but thin sample.
- **EMA21_PB_LONG** — 53.5% WR, +$266 (n=43), the most reliable workhorse.
- **OR_BREAK_LONG** — 60% WR, +$92, but n=5; too thin to conclude.

Scaling EMA9_PB_LONG is tempting but the sample (n=13) is too small to size up on yet — collect more
before increasing its allocation.

## Caveats / discipline

- **In-sample.** These thresholds were found on the replay set. Per the project's own discipline
  (box_rng_atr / expectancy-gate: pre-register, shadow-first), the time gates should be pre-registered
  and validated shadow/forward before shipping live. The cross-quarter consistency and economic
  rationale make them strong candidates, but that is not the same as out-of-sample proof.
- Replay used ES 1m as the MES price series (identical index; PnL already at MES $5/pt).
- Thin samples on EMA9_PB_LONG (13) and OR_BREAK_LONG (5) — directional, not conclusive.
- The replay's gate set may not mirror every live `config.yaml` flag (e.g. OR_BREAK_LONG fires here
  though it's disabled live); confirm live-parity before acting on the smaller signals.
