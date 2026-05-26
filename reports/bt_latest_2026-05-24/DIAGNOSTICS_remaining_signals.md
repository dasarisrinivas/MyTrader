# Residual diagnostics — remaining signals (post-fix book, 2026-05-25)

Run on the post-fix book (120 trades, after OPENING_BLOCK + FRIDAY_TREND_CONT_BLOCK). Goal: find residual soft spots in the signals other than the already-fixed `TREND_CONT_LONG` open/Friday losses.

Note: shorts are **disabled** in the current code → zero short trades to diagnose. `EMA9_PB_LONG` (n=13) and `OR_BREAK_LONG` (n=5) are too thin for reliable slicing.

## Finding 1 (strong): TREND_CONT_LONG needs ADX ≥ 25

Monotonic across ADX, n=67:

| ADX | n | WR | PnL |
|---|---|---|---|
| <20 | 25 | 48% | −$1 |
| 20–25 | 19 | 53% | −$23 |
| 25–30 | 10 | 60% | +$66 |
| 30+ | 13 | 69% | +$181 |

Below ADX 25: 44 trades, 50% WR, −$24 net — churn and exposure for no return. Trend continuation needs a real trend. **Recommend: ADX ≥ 25 floor on TREND_CONT_LONG.**

## Finding 2 (medium confidence, small sample): EMA21_PB_LONG loses on Fridays

8 Friday trades, 38% WR, −$112. Consistent with the Friday-weakness theme, but n=8 — treat as a monitor/test candidate, not a confident change. (The Friday block was originally scoped to TREND_CONT only because the *aggregate* of other signals is positive on Fridays; EMA21 specifically is not.)

## Projected impact

| Book | Trades | WR | PnL |
|---|---|---|---|
| Current post-fix | 120 | 59.2% | +$1,169 |
| + TREND_CONT_LONG ADX≥25 | 76 | 64.5% | +$1,194 |
| + that AND EMA21 skip-Friday | 68 | 67.6% | +$1,305 |

The ADX floor's value is **risk-adjusted, not PnL**: +$24 dollars but −44 trades (−37% exposure) and +5pts WR.

## Recommendation

1. Implement the ADX≥25 floor on `TREND_CONT_LONG` (strong, mechanistic, monotonic, n=67).
2. Hold the EMA21-Friday block as a monitored candidate (n=8 too small to commit).
3. Leave EMA9/OR_BREAK alone (samples too small).
4. Caution: the book is now 120 trades; each additional day/indicator filter raises curve-fit risk. Validate shadow/paper before sizing.

Separately: to evaluate re-enabling shorts, a backtest with shorts enabled would be needed (current data has none).
