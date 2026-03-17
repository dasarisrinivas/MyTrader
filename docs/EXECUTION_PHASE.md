# EXECUTION PHASE: Production Hardening Plan

**Date:** February 2026  
**System:** MES 15m RTH Strategy (v3)  
**Capital:** $5,000  
**Validated Performance:** 174 trades, PF 2.18, +$4,129 (+82.6%), Sharpe 24.73, DD -6.64%, WR 65.5%  
**Overnight Decision:** PERMANENTLY FLAT (SNR 0.058, untradeable at $5K)

---

## Executive Summary

The 15m v3 strategy has been validated through comprehensive backtesting (Feb 2025 – Jan 2026). This document defines the production hardening phase: what changes are needed to run this strategy safely with real capital, how to scale positions as equity grows, and what kill-switches protect the account if the strategy degrades.

**Core principle:** The strategy logic is FROZEN. No signal changes, no parameter tweaks. All changes are guardrails, monitoring, and risk infrastructure.

---

## 1. System Architecture Changes

### 1.1 RTH Forced Flatten (CRITICAL — P0)

**Gap identified:** No time-based forced exit exists. Positions rely entirely on bracket orders (TP/SL) and `ft_max_hold_bars` (120 min time-stop). If a bracket doesn't fill due to an IB glitch and max-hold hasn't elapsed, a position could leak overnight.

**Implementation:** Hard exit at **15:50 ET** (10 minutes before RTH close).

Location: `shree/execution/components/exit_manager.py`

```
check_position_exit_logic():
  BEFORE any other exit check:
    if current_time >= 15:50 ET and position != 0:
        force_market_exit(reason="RTH_FORCED_FLATTEN")
        log_structured_event("RTH_FORCED_FLATTEN")
```

This is defense-in-depth. Normal operation: brackets or time-stop fire well before 15:50. The flatten is a backstop that should fire 0 times per month in normal operation.

**Config:** `rth_flatten_time_et: "15:50"` (new field in `one_minute` section)

### 1.2 Session Isolation (Already Implemented ✅)

The `TradingSessionManager` already gates strategy execution to RTH only for 15m:
- Non-RTH bars: exit-check only (brackets/time-stops)
- RTH bars: full `_process_trading_cycle()`
- Indicators: computed on ALL bars (24/7) per validated backtest methodology
- `_prev_close`: only updated during RTH (strategy-internal isolation)

No changes needed.

### 1.3 Entry Window (Already Implemented ✅)

Strategy already enforces 10:30–14:59 ET entry window via `ft_entry_start_hour/minute` and `ft_entry_end_hour/minute` in `es_fifteen_min.py`. Combined with the new 15:50 flatten, the latest possible entry is 14:45 ET (last 15m bar starting before 15:00) with a guaranteed exit by 15:50.

### 1.4 Avoid-Close Window (Already Implemented ✅)

`risk_gate.avoid_close_window_minutes: 20` blocks new entries within 20 minutes of 16:00 CT (15:00 ET). This is redundant with the strategy's 14:59 cutoff but provides a second layer.

---

## 2. Position Scaling Plan

### 2.1 Philosophy

MES = 1 contract minimum. No fractional sizing. Scaling is binary: 1 contract or N contracts.

The backtest was run on 1 contract × $5,000 starting capital. Adding contracts increases both return AND drawdown proportionally. The scaling plan gates on **equity milestones** with **drawdown gates** that force scale-down if triggered.

### 2.2 Scaling Tiers

| Tier | Equity Required | Contracts | Max Risk/Trade | Max Daily Loss | Max Weekly Loss |
|------|----------------|-----------|----------------|----------------|-----------------|
| 1    | $5,000–$7,499  | 1 MES     | $60 (1.2%)     | $250 (5.0%)    | $500 (10.0%)    |
| 2    | $7,500–$11,999 | 1 MES     | $75 (1.0%)     | $300 (4.0%)    | $600 (8.0%)     |
| 3    | $12,000–$19,999| 2 MES     | $120 (1.0%)    | $480 (4.0%)    | $960 (8.0%)     |
| 4    | $20,000+       | 3 MES     | $150 (0.75%)   | $600 (3.0%)    | $1,200 (6.0%)   |

### 2.3 Scale-Down Rules

- If equity drops below tier threshold: **immediately** scale down to the tier below
- If weekly drawdown gate triggers: scale to 1 contract until new week + equity recovery
- If daily loss limit triggers: halt trading for remainder of day, reset next morning
- **Hysteresis buffer:** Must be $500 above tier threshold for 5 trading days before scaling up (prevents ping-pong)

### 2.4 Implementation Plan (FUTURE — not in v1 deployment)

Position scaling requires:
1. Equity-aware contract calculator in `risk_gate.py`
2. Config tier table (YAML)
3. Hysteresis state tracking (persistent across restarts)

**For v1 deployment: hardcode 1 contract.** Scaling is a v2 feature once the system has 90+ days of live performance data.

---

## 3. Risk Governance

### 3.1 Daily Loss Limit

**Current:** `daily_max_loss_usd: 2000` (40% of $5K — DANGEROUS)  
**New:** `daily_max_loss_usd: 250` (5% of $5K)

When triggered:
- Block all new entries for the rest of the trading day
- Log `DAILY_LOSS_LIMIT` event
- Send Telegram alert
- Existing positions: let brackets manage (don't force-exit a potentially profitable position)

Already implemented in `risk_gate.evaluate_entry()` step 7. Config value just needs fixing.

### 3.2 Weekly Loss Limit (NEW)

**New:** `weekly_max_loss_usd: 500` (10% of $5K)

Implementation: Add to `RiskGateConfig` and `evaluate_entry()`. Track cumulative weekly P&L. Reset at Monday 09:30 ET (RTH open).

When triggered:
- Block all new entries until next Monday RTH open
- Log `WEEKLY_LOSS_LIMIT` event
- Send Telegram alert
- This is the "step back and assess" circuit breaker

### 3.3 Consecutive Loss Gate

**Current:** `max_consecutive_losses: 10` (too permissive — 10 losses × $60 avg = $600, 12% of account)  
**New:** `max_consecutive_losses: 5`

The backtest shows max 7 consecutive losses. Setting to 5 means if we hit 5 in a row (something the backtest rarely saw), we pause and investigate. Reset: next trading day or manual override.

### 3.4 Peak Drawdown Guard

**Current:** `peak_drawdown_pct: 4.0`, `peak_drawdown_action: halt`, `flatten_on_trigger: true`  
**Status:** ✅ Already well-configured. When equity drops 4% from high-water mark, halt trading and flatten.

### 3.5 Per-Trade Risk Cap

**Current:** `risk_per_trade_usd: 1000` (needed temporarily to not reject wide 15m stops during testing)  
**New:** `risk_per_trade_usd: 75` (1.5% of $5K, covers typical 1.5×ATR stop ≈ 15pt = $75)

The strategy's stops are ATR-based: `ft_pb_stop_mult: 1.5` × ATR. Typical ATR ≈ 8–12 pts → stop ≈ 12–18 pts → risk ≈ $60–$90. Setting `risk_per_trade_max: 125` covers high-vol days (ATR 16 → 24pt stop → $120).

**IMPORTANT:** `max_stop_points` must also be set correctly. The strategy can produce stops up to ~25 pts in high-vol regimes. Setting too tight rejects valid trades.

**New values:**
```yaml
risk_per_trade_usd: 75
risk_per_trade_min: 25
risk_per_trade_max: 125
min_stop_points: 6.0
max_stop_points: 25.0
```

### 3.6 Kill-Switch with Auto-Reset (NEW)

A manual/automated kill-switch that halts all trading. Unlike the daily/weekly limits, this requires explicit conditions to reset:

**Trigger conditions (any one):**
- Daily loss limit hit 3 days in a week
- Weekly loss limit hit
- Manual activation via config flag or Telegram command
- Peak drawdown triggered

**Auto-reset conditions (ALL must be true):**
- New trading day (after 09:30 ET)
- Equity is above previous day's close (recovering)
- At least 1 calendar day has passed since trigger
- Kill-switch was not manually activated (manual requires manual reset)

Implementation: Add `kill_switch_active: bool` and `kill_switch_trigger_ts` to `RiskGate` state.

---

## 4. Monitoring & Metrics

### 4.1 Key Metrics to Track

| Metric | Source | Alert Threshold |
|--------|--------|----------------|
| Daily P&L | `risk_gate.realized_pnl_today` | < -$200 (warning), < -$250 (halt) |
| Weekly P&L | New: cumulative weekly tracker | < -$400 (warning), < -$500 (halt) |
| Win Rate (rolling 20) | Trade outcomes | < 50% (warning) |
| Profit Factor (rolling 20) | Trade outcomes | < 1.2 (warning), < 1.0 (critical) |
| Avg Trade Duration | Entry/exit timestamps | > 90 min (unusual) |
| Bracket Fill Rate | Order tracker | < 95% (IB issue) |
| Consecutive Losses | `risk_gate._consecutive_losses` | >= 4 (warning), >= 5 (halt) |
| Position Leak Count | RTH flatten trigger count | > 0 (investigate) |
| Strategy Signal Count | Daily signal generation | 0 for 3+ days (investigate) |

### 4.2 Telegram Alerts

Already partially implemented. Extend to cover:
- Trade entry/exit with P&L
- Daily summary (trades, P&L, win rate)
- Risk gate blocks (which gate, why)
- Kill-switch activation/deactivation
- RTH flatten events (should be 0; any occurrence is notable)

### 4.3 Decision Log

Already implemented: `logs/decisions.csv`. Contains every signal evaluation, entry/exit, and risk gate block. Critical for post-session review.

### 4.4 Performance Degradation Detection

**Weekly review checklist:**
1. Is rolling 20-trade PF > 1.5? (backtest: 2.18)
2. Is rolling 20-trade WR > 55%? (backtest: 65.5%)
3. Is max drawdown < 8%? (backtest: 6.64%)
4. Are entry signals still generating during RTH? (not broken)
5. Is bracket fill rate > 95%? (IB connectivity)

**If PF drops below 1.0 for 30+ trades:** Strategy may be degrading. Pause live trading, investigate with new backtest data.

---

## 5. Overnight Quarantine Plan

### 5.1 Decision: Permanent Flat Overnight

Based on rigorous analysis (`scripts/analyze_overnight.py`, `scripts/analyze_overnight_deep.py`):

- Overnight drift: +2.4 pts/night, std 41.5 pts → **SNR 0.058** (noise is 17× signal)
- Median MAE: 15.25 pts → with $5K max stop of 2.5 pts, **87% of nights get stopped out**
- Best realistic overnight strategy (10pt stop): +$4,694/yr but 15.8% DD → unacceptable at $5K
- 30m overnight strategy backtest: PF 0.76, -$1,779, -39% DD → **deeply unprofitable**

**Conclusion:** No overnight positions. Ever. At $5K.

### 5.2 Protection Layers (Defense-in-Depth)

| Layer | Mechanism | Already Exists? |
|-------|-----------|----------------|
| 1. Strategy | Entry window 10:30–14:59 ET | ✅ Yes |
| 2. Risk Gate | Avoid-close window (20 min before 16:00 CT) | ✅ Yes |
| 3. Time-Stop | ft_max_hold_bars = 8 (120 min max hold) | ✅ Yes |
| 4. **RTH Flatten** | **Force market exit at 15:50 ET** | ❌ **NEW — IMPLEMENTING** |
| 5. Bracket Orders | TP/SL fire regardless of bot state | ✅ Yes |
| 6. Session Gate | Non-RTH bars: exit-only, no entries | ✅ Yes |

Layer 4 is the critical addition. With it, a position cannot survive past 15:50 ET regardless of bracket status.

### 5.3 Overnight Monitoring (Flat Verification)

Even though we're flat, the bot should:
- Verify position = 0 at 16:15 ET (after RTH close)
- If position ≠ 0: EMERGENCY CLOSE + critical alert
- Log "OVERNIGHT_FLAT_VERIFIED" or "OVERNIGHT_POSITION_LEAK" daily

### 5.4 Future: When to Revisit Overnight

Conditions to reconsider overnight trading:
1. Account equity > $20,000 (4× current)
2. RTH strategy has 6+ months live track record with PF > 1.5
3. New overnight edge identified with Sharpe > 1.0 in backtest
4. Position sizing allows stops of 15+ pts (median MAE) within 1% risk

---

## 6. Failure Modes & Mitigations

### 6.1 IB Connectivity Loss

**Risk:** Bot disconnects from IB Gateway. Bracket orders are server-side (survive disconnect), but time-stops and flatten logic don't fire.

**Mitigation:**
- Bracket orders (TP/SL) are placed on IB servers → survive bot crash
- Add heartbeat check: if no IB response for 60s → alert
- On reconnect: immediately check position, force flatten if outside RTH
- Manual fallback: IB mobile app for emergency close

### 6.2 Bracket Order Rejection

**Risk:** IB rejects bracket order (margin, price limits, etc.). Position entered without stop loss.

**Mitigation:**
- Current: `risk_gate.evaluate_entry()` validates margin before submission
- Add: if bracket confirmation not received within 5s of entry, cancel entry and flatten
- Add: periodic bracket verification (every 5 min while in position)

### 6.3 Strategy Degradation (Regime Change)

**Risk:** Market regime shifts (e.g., extended low-vol chop). Strategy win rate drops.

**Mitigation:**
- Consecutive loss gate (5 losses → halt)
- Weekly loss limit ($500 → halt)
- Peak drawdown (4% → halt + flatten)
- Weekly review checklist (Section 4.4)
- Strategy is NOT auto-adapted. Changes require manual backtest validation.

### 6.4 Slippage / Fill Quality

**Risk:** Live fills worse than backtest assumptions (backtest uses close price).

**Mitigation:**
- Bracket orders use LIMIT for TP (should improve vs market)
- Entry is market order (accept ~0.25–0.50 pt slippage)
- MES bid-ask spread: typically 0.25 pt during RTH
- Monitor: if avg slippage > 1.0 pt, investigate

### 6.5 Bot Crash Mid-Position

**Risk:** Python process crashes while position is open.

**Mitigation:**
- Bracket orders persist on IB server (TP/SL still active)
- On restart: warmup → detect existing position → manage as normal
- Already implemented: `_position_verified` check on warmup complete
- NEW: If restart happens outside RTH with position: emergency flatten

### 6.6 Config Drift

**Risk:** Config values accidentally changed, loosening risk controls.

**Mitigation:**
- Document canonical config values in this file (Section 3)
- Add config validation on startup: reject if daily_max_loss > 10% of stated capital
- Future: config file hashing + alert on change

---

## 7. Implementation Checklist

### Phase 1: Critical Safety (Deploy Immediately)
- [x] RTH forced flatten at 15:50 ET (`exit_manager.py`)
- [x] Fix `daily_max_loss_usd: 2000 → 250` (`config.yaml`)
- [x] Fix `max_consecutive_losses: 10 → 5` (`config.yaml`)
- [x] Fix `risk_per_trade_usd: 1000 → 75` + related bounds (`config.yaml`)
- [x] Add weekly loss limit to `RiskGateConfig` + `evaluate_entry()`
- [x] Add kill-switch with auto-reset to `RiskGate`

### Phase 2: Monitoring (Week 1 of Live)
- [ ] Extend Telegram alerts for all risk events
- [ ] Add overnight flat verification (16:15 ET check)
- [ ] Add rolling PF/WR tracker (20-trade window)
- [ ] Add bracket fill rate monitoring
- [ ] Daily summary report

### Phase 3: Scaling (After 90 Days Live)
- [ ] Implement equity-tier position calculator
- [ ] Add scaling config table to YAML
- [ ] Implement hysteresis buffer logic
- [ ] Backtest with 2-contract scaling

---

## 8. Canonical Config Values (v1 Deploy)

```yaml
risk_gate:
  max_contracts: 1
  risk_per_trade_usd: 75
  risk_per_trade_min: 25
  risk_per_trade_max: 125
  daily_max_loss_usd: 250
  weekly_max_loss_usd: 500       # NEW
  max_consecutive_losses: 5
  min_stop_points: 6.0
  max_stop_points: 25.0
  margin_buffer_usd: 1000
  avoid_close_window_minutes: 20
  avoid_close_enabled: true
  peak_drawdown_enabled: true
  peak_drawdown_pct: 4.0
  peak_drawdown_action: "halt"
  peak_drawdown_flatten_on_trigger: true
  peak_drawdown_reset_on_new_day: true

one_minute:
  # ... (unchanged strategy params)
  rth_flatten_time_et: "15:50"   # NEW: Hard RTH flatten time
  max_hold_minutes: 120
```

---

*This document is the authoritative reference for production deployment decisions. Strategy logic is FROZEN — only guardrails change.*
