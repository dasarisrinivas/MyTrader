# Signal Optimization — Remaining Phases

**Created:** Mar 3, 2026
**Last Updated:** Mar 16, 2026
**Status:** Phase 1+2 COMPLETE ✅ | Phase 3+4 PENDING | Phase 5 COMPLETE ✅ (Fix #12+#13+#14+#15) | Phase 6 IN PROGRESS (Fix #16 deployed) | Phase 7 PENDING DATA (Fix #17+#18)

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

> **✅ DEPLOYED Mar 11, 2026:** `ft_breakeven_trigger_pts: 3.0` active. Trigger lowered from `total_pnl > $50` to `pnl >= $15 (3pts)`. See `exit_manager.py` breakeven block.
> **⚠️ NEAR-MISS Mar 12:** Trade 4 (0b26cf0639ee) peaked at +$9.38 (~1.87 pts) — below the 3pt trigger — then reversed to full SL (-$34.99). 2pt trigger would have saved ~$45. → Feeds into Fix #15.

---

## ⬜ Phase 6 — Consecutive Loss & Exhaustion Guards (NEW — Mar 12, 2026)

Identified from live trade analysis (see `docs/daily/2026-03-12.md`). Four consecutive SL hits in 5 hours (19:15–01:34 CST) totaling **-$176.64**. All were high-confidence (0.742–0.903) SELL signals in a market that had already completed its primary 100-pt sell leg and entered overnight chop.

### Fix #14: Lower Consecutive-Loss Cooldown Trigger (3 losses, not 5)
**Priority:** 🔴 HIGH — would have saved ~$74 on Mar 12 night
**Files:** `shree/execution/live_trading_manager.py`, `config.yaml`

> **✅ DEPLOYED Mar 12, 2026:** `ft_consecutive_loss_trigger: 3` active. After 3 consecutive SL hits,
> `_extra_cooldown_until` is set in `_notify_position_closed`, blocking entries in `_should_block_new_entry`
> for `cooldown_on_consecutive_losses_minutes` (now 30 min). Counter resets on any win/breakeven.
> Note: `cooldown_on_consecutive_losses_minutes` was previously dead config — now wired up.

Break-even math: If we had stopped after Trade 2 at -$102.48, we'd have saved $74.16 (Trades 3+4).

### Fix #15: Overnight Breakeven Trigger — 2pts Instead of 3pts
**Priority:** 🔴 HIGH — direct follow-up to Fix #13 near-miss
**Files:** `config.yaml`, `shree/execution/components/exit_manager.py`

> **✅ DEPLOYED Mar 12, 2026:** `ft_overnight_breakeven_trigger_pts: 2.0` active.
> `exit_manager.py` now detects session via ET time: if outside 09:30–16:00 ET, uses 2 pt trigger
> instead of the RTH 3 pt trigger. RTH path unchanged.

### Fix #16: Trend Exhaustion Filter for TREND_CONT Signals
**Priority:** 🟡 MEDIUM — prevents re-entering spent moves
**Files:** `shree/strategies/es_fifteen_min.py` → `_check_trend_continuation_short()` / `_check_trend_continuation_long()`

> **✅ DEPLOYED Mar 12, 2026:** `ft_trend_exhaustion_atr_multiple: 8.0` active.
> Both `_check_trend_continuation_short` and `_check_trend_continuation_long` check the session
> move from OR midpoint `(or_high + or_low) / 2`. If `abs(move) / atr > 8.0`, signal returns None.
> Guard only activates when `_or_computed = True` (RTH OR available); overnight-only sessions exempt.
> Logs `TREND_EXHAUSTION_SHORT` / `TREND_EXHAUSTION_LONG` when triggered.

---

## ⬜ Phase 7 — Overnight Entry Quality (NEW — Mar 16, 2026)

Identified from live trade forensic: Mar 16 01:30 CT trade (Signal A, EMA21_PB_LONG) entered into strongly bearish sentiment and was stopped out after a thin-market reversal.

### Fix #17: Overnight Sentiment Conflict Block for BUY signals
**Priority:** 🟡 MEDIUM — needs more data before implementing
**Files:** `shree/execution/components/signal_processor.py` or `shree/strategies/es_fifteen_min.py`

**Forensic (Mar 16 01:30 CT):**
- Signal: EMA21_PB_LONG, ATR=5.4, ADX=23, RSI=58, MACD_h=+0.30
- Sentiment: Stocktwits=−0.23, Reddit=−0.49, VIX=−0.40 → **Combined=−0.33**
- Outcome: SL hit at 02:34 CT after reaching +$20.63 unrealized peak
- Same setup 90 min earlier (ATR=6.4, sentiment unknown) → TP hit cleanly

**Proposed guard:** Block overnight BUY signals when combined sentiment < −0.25. Block overnight SELL signals when combined sentiment > +0.25.

**⚠️ Hold criteria — do NOT implement until:**
1. At least 10 overnight A/D signal outcomes collected (currently only 2 confirmed)
2. Confirm the 00:00 CT TP-hit trade did NOT have sentiment < −0.25 (if it did, the guard would have blocked a winner — net negative)
3. Check: would this guard have blocked any of the Mar 10–12 overnight winners?

```bash
# Check sentiment at 00:00 CT entry (the winner)
grep "2026-03-16 00:00" logs/live_trading.log | grep "SENTIMENT\|Blended\|Combined"

# When enough overnight A/D trades accumulate, cross-reference sentiment vs outcome:
sqlite3 data/trade_journal.db "SELECT t.entry_time, t.direction, t.outcome, t.pnl FROM trades t WHERE strftime('%H', t.entry_time) < '09' ORDER BY t.entry_time DESC LIMIT 20"
```

### Fix #18: Overnight Minimum ATR Guard for A/D Signals ✅ DEPLOYED MAR 17 2026
**Priority:** ✅ COMPLETE
**Files:** `shree/strategies/es_fifteen_min.py` (Guard 3 in `generate()`), `shree/config/strategy.py`

**Evidence (Mar 17 forensic — 5 filled overnight A/D trades):**
- ATR=4.6 → SL_HIT −$40.00
- ATR=5.8 → SL_HIT −$36.25
- ATR=5.8 → SL_HIT −$33.75
- ATR=6.0 → SL_HIT −$34.99
- ATR=7.7 → TP_HIT +$37.50 (sole winner)

100% loss rate when ATR < 6.0 (3/3). Low ATR overnight = thin liquidity, SL easily clipped by random noise.

**Implementation:** Guard 3 in `es_fifteen_min.generate()` — blocks A/D/A-prime/D-prime overnight when ATR < `ft_overnight_min_atr_ad` (default 6.0). RTH-exempt (only active outside 09:30–16:00 ET). Set to 0 to disable.

**Config:** `ft_overnight_min_atr_ad: 6.0` in `one_minute:` section of `config.yaml`.
**Tests:** `tests/test_overnight_entry_guards.py` [19]–[24] (6 tests).

---

## Implementation Order (Recommended)

1. ~~**Fix #12** (overnight R:R gate)~~ ✅ **DEPLOYED Mar 11**
2. ~~**Fix #13** (breakeven / profit-lock)~~ ✅ **DEPLOYED Mar 11**
3. ~~**Fix #14** (consecutive-loss cooldown at 3)~~ ✅ **DEPLOYED Mar 12**
4. ~~**Fix #15** (overnight breakeven at 2pts)~~ ✅ **DEPLOYED Mar 12**
5. ~~**Fix #16** (trend exhaustion filter)~~ ✅ **DEPLOYED Mar 12**
6. **Fix #6** (ATR touch band) — ✅ already deployed Mar 9 as T2 regime-adaptive band
7. **Fix #3** (day-type classifier) — enables #7 and #8, pending Mar 17 gate decision
8. **Fix #7** (doji allowance) — requires #3
9. **Fix #8** (F max cap on trend days) — requires #3
10. **Fix #9** (OR continuation) — standalone new pattern
11. **Fix #10** (EMA9/21 relaxation for B) — low risk, standalone
10. **Fix #11** (sentiment alignment boost) — low risk, standalone
11. **Fix #17** (overnight sentiment conflict block) — pending 10 overnight outcomes
12. ~~**Fix #18** (overnight ATR floor for A/D)~~ — ✅ DEPLOYED MAR 17

---

## Timeline & Decision Gates

| Date | Milestone | Status | Action |
|---|---|---|---|
| **Mar 3** | Phase 1+2 deployed live | ✅ Done | Monitor signal fire rate, VX additive behavior, hybrid dampen |
| **Mar 3–14** | Observation period | ✅ Done | Data collected — see Mar 11 findings below |
| **Mar 10** | Touch-band review | ✅ Closed — Fix #6 already deployed Mar 9 | Near-miss count: **240 total** in live log. T2 regime-adaptive touch band active. |
| **Mar 11** | Expectancy analysis | ✅ Fix #12 + Fix #13 deployed | Root-cause review: negative EV at ~1:1 R:R + 37.5% WR. Fix #12 (overnight TP×1.5 + R:R gate) and Fix #13 (breakeven at 3pts) deployed. Both confirmed live in Mar 12 overnight logs. |
| **Mar 12** | Consecutive-loss crisis | ✅ Identified Fix #14+#15+#16 | 4 consecutive SL hits overnight (-$176.64). All high-conf (0.742–0.903) SELLs into a spent move. No cooldown fired (5-loss threshold never reached). Fix #13 near-miss at +$9.38 (2pts below trigger). Three new fixes identified: **Fix #14** (cooldown at 3), **Fix #15** (overnight BE at 2pts), **Fix #16** (exhaustion filter). |
| **Mar 12** | Phase 6 deployed | ✅ Done | Fix #14 (consecutive-loss cooldown), Fix #15 (overnight BE 2pt), Fix #16 (trend exhaustion filter), Bug fix (BREAKEVEN misclassification in finalize_trade) |
| **Mar 16** | Overnight sentinel forensic | ✅ Fix #18 deployed | Fix #17 (overnight sentiment block) still pending — only 5/10 filled outcomes. Fix #18 (overnight ATR floor) deployed: 3/3 low-ATR losses confirmed. |
| **Mar 17** | Fix #18 deployed + 2-week review | ✅ Fix #18 done, ⏳ review | Fix #18: block A/D overnight when ATR < 6.0. Evidence: 100% loss rate at low ATR (−$110). Trade frequency: 1.2–1.5/day < 2 threshold → Fix #3 gate met. |
| **Mar 24** | Phase 3 checkpoint | ⏳ Pending | If Phase 3 started: deploy Fix #3, then #7 and #8. |
| **Apr 1** | Phase 4 evaluation | ⏳ Pending | Only if signal variety insufficient after Phase 3. Start with Fix #9. |

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
