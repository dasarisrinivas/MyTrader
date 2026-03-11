# Signal Optimization — Remaining Phases

**Created:** Mar 3, 2026
**Status:** Phase 1+2 COMPLETE → Phase 3+4 PENDING

---

## ✅ Completed (Phase 1+2) — Deployed Mar 3, 2026

| # | Fix | File | Description |
|---|---|---|---|
| 1 | Remove MACD from A/B/D/E | `es_fifteen_min.py` | MACD histogram filter removed from signals A, B, D, E. Kept on C (shallow pullback) and F (trend continuation). |
| 2 | Widen ADX band [18,45] | `config.yaml` | `ft_adx_min: 20→18`, `ft_adx_max: 35→45`. Strong trends (ADX=42) and borderline (ADX=18) now qualify. |
| 5 | Remove RSI from A/D | `es_fifteen_min.py` | RSI overbought/oversold filter removed from pullback signals. Kept on C only. |
| 12 | VX additive scaling | `signal_processor.py` | Replaced multiplicative (conf × 0.7) with additive signal-type-aware tiers. #1 signal killer fixed. |
| 13 | Lower conf threshold | `config.yaml` | `min_confidence_for_trade: 0.50→0.40`. |
| 14 | Hybrid dampen cap | `signal_processor.py` | Oppose/uncertain dampen capped at -0.05 (was -0.10). |

---

## ⬜ Phase 3 — Regime-Aware Enhancements

### Fix #3: Day-Type Classifier Layer
**Priority:** HIGH — enables regime-aware parameter tuning
**Files:** New `shree/strategies/day_type_classifier.py`, wire into `signal_processor.py`

Classify each trading day into one of 3 regimes using early-session indicators:
- **TREND day:** ADX > 25 at bar 4+, price outside OR by > 0.5×ATR, EMA9/21 aligned
- **RANGE day:** ADX < 20, price oscillating within OR, no EMA alignment
- **BREAKOUT day:** Price gaps beyond prior day's range, or OR break with volume surge

Use the classification to tune downstream parameters:
- TREND: relax touch band, allow doji bars, increase F signal cap
- RANGE: tighten stops, prefer A/D pullbacks, reduce B/E weight
- BREAKOUT: boost B/E confidence, wider stops, reduce A/D weight

### Fix #6: ATR-Based Touch Band for Signals A/D
**Priority:** HIGH — most impactful single fix for pullback frequency
**Files:** `es_fifteen_min.py` → `_check_ema21_pullback()` and `_check_ema21_pullback_short()`

Currently: `bar_low <= ema21` (exact touch required)
Problem: Price misses EMA21 by 0.5–2 points regularly → signal never fires

Proposed: `bar_low <= ema21 + (ATR * 0.15)` — allow proximity within 15% of ATR
- ATR=10 → touch band = 1.5 pts (miss by up to $7.50 allowed)
- ATR=20 → touch band = 3.0 pts (volatile day, wider tolerance)
- ATR=5 → touch band = 0.75 pts (tight day, tight tolerance)

This adapts to market conditions automatically. The bar must still show bullish close + EMA alignment.

### Fix #7: Doji Allowance on Trend Days
**Priority:** MEDIUM — unlocks valid signals on indecision bars
**Files:** `es_fifteen_min.py` → `_check_ema21_pullback()` and `_check_ema21_pullback_short()`

Currently: Signal A requires `close > open` (bullish candle). Signal D requires `close < open`.
Problem: On trend days, a doji at EMA21 is a valid continuation signal but gets filtered.

Proposed: If day_type == TREND and `abs(close - open) < 0.5 * ATR * 0.10`:
- Allow doji bars if EMA alignment is strong (EMA21 > EMA50 by > ATR)
- Reduce confidence by -0.03 for doji (slight penalty but not a block)

Depends on: Fix #3 (day-type classifier)

### Fix #8: Increase Signal F Max Cap on Trend Days
**Priority:** MEDIUM
**Files:** `es_fifteen_min.py` → `_check_trend_continuation()`

Currently: F signal fires at most once per direction per day (or capped at 2).
On strong trend days, price runs from EMA21 multiple times.

Proposed: If day_type == TREND, allow up to 3 F signals per direction.
Depends on: Fix #3 (day-type classifier)

---

## ⬜ Phase 4 — Entry Pattern Expansion

### Fix #9: OR Continuation Entry Pattern
**Priority:** MEDIUM — captures follow-through after OR breaks
**Files:** `es_fifteen_min.py` → new `_check_or_continuation()` method

Currently: Signal B fires on the first OR High cross. If price pulls back to OR High and bounces, there's no re-entry signal.

Proposed new pattern:
1. OR breakout already occurred (B fired earlier in session)
2. Price pulls back to within 2 pts of OR High (support retest)
3. Bullish bar closes above OR High again
4. EMA9 > EMA21 still holds
→ Generate signal with `reason="OR_CONT_LONG"`, confidence 0.65 (slightly lower than B)

Mirror for short: OR breakdown retest at OR Low.

### Fix #10: Relax EMA9 > EMA21 Requirement for Signal B
**Priority:** LOW — reduces false negatives on crossover bars
**Files:** `es_fifteen_min.py` → `_check_or_breakout()`

Currently: `ema9 > ema21` is required for B. Problem: On the bar where price crosses OR High, the EMA9 may not have crossed EMA21 yet (lagging indicator).

Proposed: Allow B if EMA9 is within 0.5 pts of EMA21 (convergence, about to cross).
Alternative: Use EMA5 > EMA9 instead (faster signal).

### Fix #11: Sentiment-Trend Alignment Boost
**Priority:** LOW
**Files:** `signal_processor.py`

When sentiment direction matches signal direction (bullish sentiment + BUY signal), add +0.03 confidence boost instead of the current neutral (1.0x).

Currently sentiment only penalizes (REDUCE_SIZE at 0.7x, BLOCK). Adding a small positive feedback for alignment could help marginal signals pass the threshold.

---

## ⬜ Phase 5 — Expectancy / Risk-Math Fixes (NEW — Mar 11, 2026)

Identified from live trade analysis (see `docs/daily/2026-03-11.md` Fundamental Takeaway). These are **not** signal-quality fixes — they address the structural R:R and profit-protection math that determines whether the strategy has positive expected value at all.

> At the current running rate: 3W / 5L (37.5% WR) × ~1:1 R:R → **negative expected value per trade**.
> Regime and signal-quality fixes reduce frequency of bad trades but do not repair the underlying edge.

### Fix #12: Overnight Minimum R:R Gate
**Priority:** HIGH — directly addresses root cause identified Mar 11
**Files:** `config.yaml`, `shree/execution/components/order_coordinator.py`

Currently: `min_risk_reward_ratio: 1.0` applies uniformly across all sessions.
Problem: Overnight pullbacks at 1.03:1 R:R pass the gate but have negative EV at sub-50% win rate.

The current overnight SL/TP setup widens risk without widening reward:
- `ft_overnight_sl_mult: 1.2` (SL widens to 7.2 pts)
- `ft_overnight_tp_mult: 1.0` (TP stays at 8.0 pts)
- Effective overnight R:R ≈ 8.0 / 7.2 = **1.11:1** — barely above the floor

Proposed:
- Add `ft_overnight_min_rr: 1.7` config param (session-specific R:R floor)
- When outside RTH, require R:R ≥ 1.7 before placing order
- To achieve 1.7:1 with a 7.2 pt overnight SL, TP must be ≥ 12.2 pts
- Either raise `ft_overnight_tp_mult` to ~1.5, or tighten `ft_overnight_sl_mult` back toward 1.0

Break-even WR at 1.7:1 R:R = 1 / (1 + 1.7) ≈ **37%**, which matches current live WR.
Break-even WR at 2.0:1 R:R = 1 / (1 + 2.0) ≈ **33%**, which gives margin of safety.

### Fix #13: Breakeven / Profit-Lock on Open Trades
**Priority:** HIGH — prevents giving back unrealized profits (demonstrated by Trade 7, Mar 10)
**Files:** `shree/execution/components/exit_manager.py`

Currently: There is no trailing stop or breakeven trigger once a trade is open.
Problem: Trade 7 reached +$25 unrealized (at 11:00 PM) but was stopped out at the original SL (-$36.25) by midnight — a swing of $61.25 from peak.

Proposed:
- When unrealized P&L exceeds `breakeven_trigger_pts` (e.g. 3.0 pts / $15), move SL to entry + 1 tick (breakeven)
- When unrealized P&L exceeds `profit_lock_pts` (e.g. 5.0 pts / $25), move SL to lock in 50% of the gain
- Parameterize separately for overnight vs RTH (overnight could use lower threshold since moves are slower)
- Example: `ft_breakeven_trigger_pts: 3.0`, `ft_profit_lock_pts: 5.0`, `ft_profit_lock_pct: 0.5`

This is especially important for overnight trades where the bot holds through 1–2 hour reversals.

---

## Implementation Order (Recommended)

1. **Fix #12** (overnight R:R gate) — standalone, directly fixes root cause, no dependencies
2. **Fix #13** (breakeven / profit-lock) — standalone, prevents profit givebacks
3. **Fix #6** (ATR touch band) — ✅ already deployed Mar 9 as T2 regime-adaptive band
4. **Fix #3** (day-type classifier) — enables #7 and #8
5. **Fix #7** (doji allowance) — requires #3
6. **Fix #8** (F max cap on trend days) — requires #3
7. **Fix #9** (OR continuation) — standalone new pattern
8. **Fix #10** (EMA9/21 relaxation for B) — low risk, standalone
9. **Fix #11** (sentiment alignment boost) — low risk, standalone

---

## Timeline & Decision Gates

| Date | Milestone | Status | Action |
|---|---|---|---|
| **Mar 3** | Phase 1+2 deployed live | ✅ Done | Monitor signal fire rate, VX additive behavior, hybrid dampen |
| **Mar 3–14** | Observation period | ✅ Done | Data collected — see Mar 11 findings below |
| **Mar 10** | Touch-band review | ✅ Closed — Fix #6 already deployed Mar 9 | Near-miss count: **240 total** in live log. T2 regime-adaptive touch band active. |
| **Mar 11** | Expectancy analysis | ✅ Added Fix #12 + Fix #13 | Root-cause review: trades/day since go-live = **1.40/day**. Since Mar 9 = **2.00/day**. Near-miss count = 240. VX penalty on pullbacks firing correctly at -0.05. **New structural issue found:** negative EV at ~1:1 R:R + 37.5% WR. Fix #12 (overnight R:R gate) and Fix #13 (breakeven/profit-lock) added. |
| **Mar 17** | 2-week review | ⏳ Upcoming | Evaluate: Is trade frequency ≥ 2/day? Current: 2.00/day since Mar 9 (exactly at threshold). **Recommendation: start Fix #12 + Fix #13 now (expectancy fixes). Hold Fix #3 until Mar 17 review.** If trades/day drops below 2 again → start Fix #3 (day-type classifier). |
| **Mar 24** | Phase 3 checkpoint | ⏳ Pending | If Phase 3 started: deploy Fix #3, then #7 and #8. If not needed: skip to Phase 4 evaluation |
| **Apr 1** | Phase 4 evaluation | ⏳ Pending | Only if signal variety is insufficient after Phase 3. Start with **Fix #9 (OR continuation)** |
| **Ongoing** | Skip Phase 3+4 entirely if | — | Phase 1+2 achieves 2+ trades/day consistently with ~45-50% win rate |

### Quick Decision Commands
```bash
# How many trades/day since restart?
grep "order_placed" logs/reconcile.log | awk '{print $1}' | sort -u | uniq -c

# How many near-miss touch-band failures?
grep "A:low.*>touch\|D:high.*<touch" logs/live_trading.log | wc -l

# Signal fire rate (signals generated vs HOLD)
grep "Strategy Signal:" logs/live_trading.log | grep -c "BUY\|SELL"
grep "Strategy Signal:" logs/live_trading.log | grep -c "HOLD"

# VX additive behavior verification
grep "VX Adjustment\|VX Neutral" logs/live_trading.log | tail -10

# Confidence blocked count (should be lower than before)
grep "BLOCKED.*confidence" logs/live_trading.log | wc -l
```

---

## Validation Approach

Each phase should be validated with:
1. **Unit tests** — new test file per feature
2. **Backtest comparison** — run `python3 -m backtest.run --symbol MES --start 2025-02-01 --end 2026-01-31 --bar 15m` before and after
3. **Live observation** — deploy in paper mode (`ibkr_port: 4002`) for 2-3 sessions before live
4. **Log analysis** — grep for new signal types / diagnostic changes to verify fire rate
