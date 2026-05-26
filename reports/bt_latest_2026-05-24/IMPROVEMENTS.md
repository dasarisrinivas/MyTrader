# What to improve — from the latest-code replay (2026-05-24)

Baseline (latest code, Feb 2025–Jan 16 2026): **166 trades, 50.6% WR, +$66.60** (MES $5/pt). Barely above breakeven.

## Root cause: one signal, in three bad conditions

`TREND_CONT_LONG` is 63% of all volume and the only net-negative signal (−$639.50). Its losses are not spread evenly — they sit in three explainable buckets:

| Condition | Trades | WR | Net PnL |
|---|---|---|---|
| **Opening hour (8 AM CT / 9:30 ET)** | 17 | 11.8% | **−$668** |
| **Fridays** | 26 | 23.1% | **−$741** |
| Low ATR (<6) | 34 | 32.4% | −$543 |
| ADX < 30 | 82 | ~42% | all negative |
| — vs mid-day h9–13 | 82 | 53–67% | **+$201** |
| — vs ADX > 30 (real trend) | 18 | 61.1% | +$46 |

The signal is firing "continuation" trades when there is no trend to continue: at the open (pure whipsaw, before the opening range is even set), on Fridays (weekend de-risking, poor follow-through), and in low-volatility/low-ADX chop.

## The fix: two structural filters (no threshold-tuning)

| Scenario | Book trades | Book WR | **Book PnL** |
|---|---|---|---|
| Baseline (today) | 166 | 50.6% | +$66.60 |
| Drop `TREND_CONT_LONG` entirely | 61 | 59.0% | +$706 |
| `TREND_CONT_LONG`: mid-day only (9–13 CT) | 143 | 56.6% | +$907 |
| `TREND_CONT_LONG`: exclude Friday | 140 | 55.7% | +$808 |
| **`TREND_CONT_LONG`: 9–13 CT AND no Friday** | **128** | **59.4%** | **+$1,141** |

Two filters take the book from **+$67 → +$1,141 (≈17×)** and 50.6% → 59.4% WR. Quarterly robustness: the filtered book is **positive in all five quarters** (Q1 +$47, Q2 +$548, Q3 +$350, Q4 +$99, 2026Q1 +$98); the baseline was negative in three of five. So this is a consistent effect, not one lucky stretch.

## Recommended changes, prioritized

1. **Skip the opening 30 minutes for all signals.** Move the entry-window start from 9:30 ET → **10:00 ET** (9:00 CT). The 8 AM CT hour loses money on *every* signal (`TREND_CONT_LONG` −$668 **and** the other three −$225). This is the single biggest, cleanest lever and it's global. Mirrors the existing `LATE_AFTERNOON_BLOCK` pattern.

2. **Block `TREND_CONT_LONG` on Fridays.** Friday is specific to this signal (−$741); the pullback signals are fine on Friday (+$108), so do not block them. Mirrors the existing `MONDAY_BLOCK`.

3. **Require a real trend for `TREND_CONT_LONG`: ADX ≥ 30 (or skip ATR < 6).** Below ADX 30 it is a coin flip; only the strong-trend bucket pays. This is a secondary filter — apply only if #1+#2 alone aren't enough, since stacking it on top slightly over-filters (removes some winners).

4. **Consider demoting `TREND_CONT_LONG` to the smallest size** until it earns its keep. Even filtered it is the weakest edge per trade; the real money is in `EMA9_PB_LONG` (77% WR) and `EMA21_PB_LONG` (53.5%, +$266).

## Validation note (project discipline)

These filters are derived in-sample on 105 trades, but they are **time/day-structural and mechanistically grounded** (open whipsaw, Friday follow-through), corroborated across the other signals and across all four quarters — not curve-fit indicator thresholds. Still, consistent with the project's shadow-first practice (`docs/expectancy_gate_proposal.md`), implement #1–#2 as a pre-registered change and confirm on forward/paper before sizing up. Change #1 is a one-line config edit to the entry window; #2 is a Friday gate analogous to `MONDAY_BLOCK`.

Data: `all_trades_latest_code.csv`, `summary_by_signal.csv`.

---

## IMPLEMENTED & VERIFIED (2026-05-25)

Two config-gated blocks were added to `shree/strategies/es_fifteen_min.py`, mirroring the existing `MONDAY_BLOCK`/`LATE_AFTERNOON_BLOCK` (default-ON, asymmetric — they only remove trades):

- `OPENING_BLOCK` (`ft_opening_block_minutes=30`): blocks all signals in the first 30 min of RTH (9:30–9:59 ET).
- `FRIDAY_TREND_CONT_BLOCK` (`ft_friday_block_trend_cont=True`): blocks the trend-continuation family on Fridays only.

The full backtest was re-run with the change. Verified before/after (same engine, same data, MES $5/pt):

| Signal | Before (n / WR / PnL) | After (n / WR / PnL) |
|---|---|---|
| EMA9_PB_LONG | 13 / 76.9% / +$348.8 | 13 / 76.9% / +$348.8 |
| OR_BREAK_LONG | 5 / 60.0% / +$91.7 | 5 / 60.0% / +$91.7 |
| EMA21_PB_LONG | 43 / 53.5% / +$265.6 | 35 / 60.0% / +$506.0 |
| TREND_CONT_LONG | 105 / 45.7% / **−$639.5** | 67 / 55.2% / **+$222.9** |
| **ALL** | **166 / 50.6% / +$66.6** | **120 / 59.2% / +$1,169.5** |

Per-quarter after the change: **positive in all five** (Q1 +$24, Q2 +$624, Q3 +$196, Q4 +$173, 2026Q1 +$152). `TREND_CONT_LONG` flipped from −$640 to +$223; 46 trades removed, all net losers.

**Still to do before live:** confirm shadow/paper per project discipline before sizing up; the flags are default-ON in code but were not added to `config.yaml` (they follow the same getattr-default pattern as the Monday/late-afternoon blocks). Files: `all_trades_AFTER_fix.csv`.
