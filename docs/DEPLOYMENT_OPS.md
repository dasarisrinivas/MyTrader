# DEPLOYMENT OPERATIONS: Live Validation, Scaling & Edge Monitoring

**Date:** February 2026  
**Prerequisite:** `EXECUTION_PHASE.md` — all Phase 1 items implemented and verified  
**Backtest Reference Period:** Feb 2025 – Jan 2026 (174 trades, 12 months)  
**Instrument:** MES, 1 contract, $5,000 capital  

---

## Backtest Reference Statistics (Ground Truth)

These numbers are the benchmark. Live performance is compared against them.

| Metric | Value | Source |
|--------|-------|--------|
| Total Trades | 174 | 12 months |
| Win Rate | 65.5% | 114W / 60L |
| Profit Factor | 2.18 | |
| Expectancy | $23.73/trade | |
| Avg Win | $66.89 | |
| Avg Loss | -$58.27 | |
| Max Single Win | $1,088.85 | |
| Max Single Loss | -$143.65 | |
| Std(PnL) | $109.42 | |
| Max Drawdown | -$288.80 (-5.8%) | |
| Max Consecutive Losses | 4 | |
| Avg Trades/Week | 3.8 | |
| Max Trades/Week | 9 | |
| Worst Week | -$229.70 | |
| Best Week | $1,808.00 | |
| Rolling-20 WR Range | 45.0% – 85.0% (mean 66.6%) | |
| Rolling-20 PF Range | 0.87 – 6.49 (mean 2.30) | |
| Losing Months | 3 of 12 (Feb, Sep, Dec) | |
| Worst Month | -$162.45 (Feb 2025) | |

---

## 1. LIVE VALIDATION PHASE (PRE-SCALING)

### 1.1 Minimum Sample Size

**Minimum: 50 live trades** before any scaling decision is evaluated.

Rationale:
- 50 trades ≈ 13 weeks at 3.8 trades/week average
- Provides statistically meaningful win rate estimate (95% CI width ≈ ±13%)
- Captures at least one losing streak (backtest max = 4)
- Crosses at least 2 monthly P&L cycles

**No scaling discussion before trade #50. Period.**

### 1.2 Live vs Backtest Comparison Metrics

Compute these at trade 25 (interim check) and trade 50 (gate):

| Metric | Backtest Ref | Acceptable Range | Action if Outside |
|--------|-------------|-------------------|-------------------|
| Win Rate | 65.5% | ≥ 50.0% | < 50%: pause, investigate |
| Profit Factor | 2.18 | ≥ 1.20 | < 1.20: pause, investigate |
| Expectancy | $23.73 | ≥ $5.00 | < $5: investigate fills/slippage |
| Avg Loss | -$58.27 | ≤ -$85.00 | Worse than -$85: investigate stop placement |
| Avg Slippage (entry) | 0 (backtest) | ≤ 1.0 pts ($5) | > 1.0 pt: investigate order type/timing |
| Max Consecutive Losses | 4 | ≤ 5 | = 5: kill-switch fires (by design) |
| Bracket Fill Rate | 100% | ≥ 95% | < 95%: IB issue, halt and debug |
| RTH Flatten Triggers | 0 | = 0 | > 0: investigate why brackets/time-stop failed |
| Trades/Week | 3.8 | 1.5 – 7.0 | < 1: signals not generating; > 7: something changed |

### 1.3 Slippage Measurement

Backtest enters at bar close. Live enters at market. Measure:

```
entry_slippage_pts = abs(live_fill_price - signal_price) 
```

Track per trade. Expected: 0.25–0.50 pts (1–2 ticks) for MES during RTH.

**Alert threshold:** Rolling-10 avg slippage > 0.75 pts.  
**Halt threshold:** Any single fill > 2.0 pts slippage.

### 1.4 Deviation Thresholds & Responses

| Condition | Severity | Action |
|-----------|----------|--------|
| WR < 50% after 30+ trades | WARNING | Review decision log, check signal fidelity |
| WR < 40% after 30+ trades | CRITICAL | Halt live. Re-backtest on recent data |
| PF < 1.0 after 30+ trades | CRITICAL | Halt live. Strategy edge may be gone |
| PF < 1.0 after 50+ trades | TERMINAL | Stop live. Do NOT resume without new validation |
| DD > 8% (> $400 on $5K) | CRITICAL | Kill-switch fires. Manual review before resume |
| DD > 10% ($500) | TERMINAL | Stop live. Full audit required |
| 0 signals for 5 consecutive trading days | WARNING | Check data feed, IB connection, strategy code |

### 1.5 Immediate Rollback to 1× Conditions

If at any point during scaled trading (Tier 2+), ANY of these occur:
1. Rolling-20 PF drops below 1.0
2. Rolling-20 WR drops below 45%
3. Weekly loss exceeds Tier 1 weekly limit ($500)
4. Kill-switch activates for any reason
5. Any manual override of risk gates is detected

→ **Immediately reduce to 1 contract. Remain at 1 contract for minimum 20 trades before re-evaluating.**

---

## 2. POSITION SCALING RULESET

### 2.1 MES Contract Sizing Constraint

MES minimum = 1 contract. No fractional contracts. Practical scaling options:
- **1 contract** = base risk
- **2 contracts** = 2× risk (next possible step)

"1.5× sizing" is implemented as **alternating**: trade 1 at 1 contract, trade 2 at 2 contracts, repeat. Over N trades, average size = 1.5 contracts.

### 2.2 Scaling Tiers

| Tier | Equity Gate | Contracts | Avg Risk/Trade | Daily Max Loss | Weekly Max Loss |
|------|------------|-----------|----------------|----------------|-----------------|
| **T1** | $5,000 – $7,499 | 1 | $75 | $250 (5.0%) | $500 (10.0%) |
| **T1.5** | $7,500 – $9,999 | Alternating 1/2 | ~$112 | $300 (4.0%) | $600 (8.0%) |
| **T2** | $10,000 – $14,999 | 2 | $150 | $400 (4.0%) | $800 (8.0%) |

### 2.3 Scale-Up Prerequisites (ALL must be true)

To move from T1 → T1.5:
1. Equity ≥ $7,500 (verified from IB account, not estimated)
2. Equity has been ≥ $7,500 for **10 consecutive trading days** (hysteresis)
3. ≥ 50 live trades completed at T1
4. Live PF ≥ 1.5 over most recent 30 trades
5. Live WR ≥ 55% over most recent 30 trades
6. No kill-switch activation in the past 20 trading days
7. Current drawdown from peak < 3%

To move from T1.5 → T2:
1. Equity ≥ $10,000 for **10 consecutive trading days**
2. ≥ 30 live trades completed at T1.5
3. Live PF ≥ 1.4 over most recent 30 trades
4. Live WR ≥ 55% over most recent 30 trades
5. No kill-switch activation in the past 20 trading days
6. Max single-trade loss at T1.5 did not exceed $200

### 2.4 Scale-Down Rules (ANY triggers immediate de-scale)

| Condition | Action |
|-----------|--------|
| Equity drops below current tier threshold | Drop to tier below. Immediate. |
| Daily loss limit hit | Drop to T1 for remainder of day (already halted by risk gate) |
| Weekly loss limit hit | Drop to T1 for remainder of week |
| Kill-switch activates | Drop to T1. Remain at T1 for 20 trades after reset |
| Rolling-20 PF < 1.0 | Drop to T1. Remain until PF > 1.3 for 20 trades |
| 2 losing weeks in a row (at T1.5 or T2) | Drop one tier |

### 2.5 Cooldown After Scaling Events

| Event | Cooldown |
|-------|----------|
| Scale up (any tier) | No trades for remainder of current day. Resume next RTH open |
| Scale down (any trigger) | Immediate. No cooldown on de-scaling (safety first) |
| Scale down → scale up attempt | Minimum 20 trades at lower tier before re-evaluation |

### 2.6 Alternating 1/2 Contract Logic (T1.5)

```
if tier == T1.5:
    if trade_count_at_tier % 2 == 0:
        contracts = 1
    else:
        contracts = 2
```

This averages 1.5× exposure. If a loss occurs on the 2-contract trade, the next trade is 1 contract (natural de-risking after loss).

### 2.7 Configuration Changes Per Tier

When scaling:
- `max_contracts` → update to tier value
- `daily_max_loss_usd` → update to tier value
- `weekly_max_loss_usd` → update to tier value
- `risk_per_trade_usd` → scale proportionally
- `risk_per_trade_max` → scale proportionally
- **All other strategy parameters: UNCHANGED**

---

## 3. PSYCHOLOGICAL & OPERATIONAL SAFETY

### 3.1 Operator Risk Detection

The primary risk to this system is the operator, not the market. Specific threats:

| Threat | Detection | Prevention |
|--------|-----------|------------|
| Revenge trading (manual entry after loss) | Order tracker shows order not from bot | Bot is sole order source; manual orders = violation |
| Widening stops after entry | Bracket modification log ≠ bot-initiated | Log all bracket modifications with source tag |
| Disabling risk gates | Config hash changes on startup | Startup validator: reject if risk params outside bounds |
| Overriding kill-switch prematurely | Kill-switch deactivation logged with timestamp | Minimum 1-day cooldown before manual reset |
| Increasing position size manually | Position > `max_contracts` detected | Position monitor alerts if qty > expected |
| Trading outside bot (other instruments) | N/A for this system | Separate account for discretionary (if any) |

### 3.2 Human Interference Alerts

Log and alert on ALL of the following:
1. **Config file modified while bot is running** — hash check on each cycle
2. **Position quantity differs from expected** — qty > max_contracts or position exists when bot thinks flat
3. **Orders exist that bot didn't place** — order tracker cross-reference
4. **Kill-switch manually deactivated** — log timestamp, require reason string
5. **Risk gate params changed between sessions** — diff config on startup vs last known config

### 3.3 Mandatory Stop Conditions (Non-P&L)

These trigger a halt regardless of profitability:

| Condition | Action |
|-----------|--------|
| Bot has not placed a trade in 10+ trading days | Halt. Investigate signal generation |
| IB Gateway disconnected > 30 minutes during RTH | Halt. Verify position state. Alert operator |
| Bracket fill rate < 90% over any 10-trade window | Halt. IB order routing issue |
| RTH forced flatten triggered even once | Halt day. Investigate why normal exits failed |
| Config validation fails on startup | Do not start. Fix config first |
| Account equity < $3,000 (40% loss from start) | Permanent halt. Full strategy review required |

### 3.4 Separation of Roles

Even for a single operator:
- **"Designer" role**: Can modify strategy params, but only via backtest → validate → deploy cycle
- **"Operator" role**: Can start/stop bot, monitor alerts, activate kill-switch. Cannot modify strategy params while live
- In practice: no config changes while bot is running. Stop → change → validate → restart.

---

## 4. METRIC-DRIVEN EDGE MONITORING

### 4.1 Weekly Metrics (computed every Friday 16:00 ET)

| Metric | Computation | Backtest Benchmark |
|--------|-------------|-------------------|
| Trades this week | Count | 3.8 avg |
| Win rate this week | Wins/Total | 65.5% |
| PF this week | GrossWin/GrossLoss | 2.18 |
| Total P&L this week | Sum(realized_pnl) | $89.75 avg |
| Max single loss | Min(realized_pnl) | -$143.65 worst |
| Consecutive losses (current streak) | Counter | 4 max |
| Drawdown from equity peak | (equity - peak) / peak | -5.8% max |

### 4.2 Monthly Metrics (computed 1st of each month)

| Metric | Computation | Backtest Benchmark |
|--------|-------------|-------------------|
| Monthly P&L | Sum | $344/mo avg ($4,129/12) |
| Monthly trade count | Count | 14.5 avg (174/12) |
| Monthly WR | Wins/Total | 65.5% |
| Monthly PF | GrossWin/GrossLoss | 2.18 |
| Losing months in last 6 | Count where P&L < 0 | ≤ 2 of 6 (backtest had 3/12) |
| Sharpe (annualized) | mean(daily_ret)/std(daily_ret) × √252 | 24.73 |

### 4.3 Edge Decay vs Normal Variance

**Normal variance (expected, no action):**
- Individual losing weeks (backtest worst: -$229.70)
- Individual losing months (backtest worst: -$162.45)
- Win rate dropping to 50% over a 20-trade window (backtest min was 45%)
- PF dropping to 1.0 over a 20-trade window (backtest min was 0.87)
- 4 consecutive losses (backtest max)

**Edge decay signals (requires investigation):**

| Signal | Threshold | Backtest Never Saw |
|--------|-----------|-------------------|
| Rolling-30 WR < 45% | Below any 30-trade backtest window | Investigate |
| Rolling-30 PF < 0.90 | Sustained unprofitability | Pause live |
| 3 losing months in a row | Backtest max was 1 consecutive | Pause live |
| 5+ consecutive losses | Backtest max was 4 | Kill-switch fires |
| Expectancy < $0 over 40+ trades | Strategy is net-negative | Stop live |
| Max DD exceeds 10% | Nearly 2× backtest worst | Stop live |

**Statistical test for edge decay:**

At the 50-trade mark and every 25 trades thereafter, compute:
```
z = (live_WR - 0.655) / sqrt(0.655 * 0.345 / N)
```
Where N = number of live trades. If z < -2.0 (p < 0.023, one-tailed), the live win rate is statistically significantly worse than backtest at the 95% confidence level.

**Critical thresholds by sample size:**

| Live Trades | WR that triggers z < -2.0 |
|-------------|--------------------------|
| 30 | < 48.2% |
| 50 | < 52.1% |
| 75 | < 54.5% |
| 100 | < 55.9% |

If live WR crosses the z < -2.0 threshold: **halt trading, full review.**

### 4.4 Regime Change Detection

The strategy is designed for trending RTH sessions (ADX 20-35). It does not work in:
- Low-vol chop (ADX < 15 sustained)
- Crash/panic (ATR > 40 sustained, gap moves)

**Regime indicators to monitor (not trade on — observe only):**
- VIX > 35 sustained for 3+ days: elevated caution
- ES daily ATR drops below 20 pts for 10+ days: low-vol regime, fewer signals expected
- ES daily ATR exceeds 80 pts for 3+ days: crisis regime, wider stops will hit risk caps
- Fed announcement days: expect wider slippage, consider manual halt

**No automated regime-based trading changes.** The strategy's ADX filter (20-35) and ATR-based stops already adapt somewhat. If regimes make the strategy structurally unprofitable, the PF/WR degradation metrics will catch it.

---

## 5. KILL-SWITCH DRILLS

### 5.1 Drill Schedule

Run each drill **before first live trade** and **monthly thereafter**.

### 5.2 Drill Scenarios

#### Drill 1: IB Gateway Disconnect (Simulated)

**Setup:** Bot running, no position. Kill IB Gateway process.  
**Expected behavior:**
1. Bot detects disconnect within 60s
2. `_broadcast_error()` fires
3. Telegram alert: "IB DISCONNECTED"
4. Bot enters reconnection loop
5. No new trades attempted
6. Bracket orders on IB server: unaffected (they persist)

**Verify:**
- [ ] Alert received within 2 minutes
- [ ] No orphan orders placed during disconnect
- [ ] Bot reconnects and resumes when Gateway restarts
- [ ] Position state is correct after reconnect

#### Drill 2: IB Gateway Disconnect WITH Open Position

**Setup:** Bot running, in a position (paper account). Kill IB Gateway.  
**Expected behavior:**
1. Same as Drill 1
2. Bracket orders (TP/SL) remain active on IB server
3. On reconnect: bot detects existing position, resumes exit management
4. If disconnect spans RTH close: bracket SL is the backstop

**Verify:**
- [ ] Brackets survived the disconnect (check IB TWS/mobile)
- [ ] Position detected correctly on reconnect
- [ ] If time-stop would have fired during disconnect, bot exits on reconnect

#### Drill 3: Daily Loss Limit Hit

**Setup:** Paper account. Manually create losing trades until daily P&L < -$250.  
**Expected behavior:**
1. `risk_gate.evaluate_entry()` returns `DAILY_LOSS_LIMIT`
2. No new entries for rest of day
3. Telegram alert: "DAILY_LOSS_LIMIT"
4. Next morning: entries resume (if equity permits)

**Verify:**
- [ ] Block message appears in logs
- [ ] Alert received
- [ ] No entries attempted after block
- [ ] Reset works next day

#### Drill 4: Kill-Switch Activation (Manual)

**Setup:** Bot running, flat.  
**Expected behavior:**
1. Operator calls `risk_gate.activate_kill_switch(reason="DRILL", manual=True)`
2. All entries blocked
3. Log: "KILL SWITCH ACTIVATED: DRILL (manual=True)"
4. Auto-reset does NOT fire (manual kill = manual reset)
5. Operator calls `risk_gate.deactivate_kill_switch()`
6. Entries resume

**Verify:**
- [ ] Kill-switch blocks entries immediately
- [ ] Auto-reset correctly skipped for manual kills
- [ ] Manual deactivation works
- [ ] Log trail is complete

#### Drill 5: RTH Forced Flatten (Simulated)

**Setup:** Paper account. Enter position manually at 15:45 ET. Disable bracket orders (or set TP/SL far away).  
**Expected behavior:**
1. At 15:50 ET: `RTH_FORCED_FLATTEN` triggers
2. Market order to close position
3. Log: "RTH_FORCED_FLATTEN triggered"
4. Structured event logged
5. This is a NOTABLE event — should never happen in normal operation

**Verify:**
- [ ] Flatten fires between 15:50:00 and 15:50:30 ET
- [ ] Position is flat by 15:51 ET
- [ ] Alert sent
- [ ] Correct exit reason in trade log

#### Drill 6: Bot Crash and Restart

**Setup:** Bot running with open position (paper). `kill -9` the Python process.  
**Expected behavior:**
1. Bot process dies. Brackets on IB server remain active
2. Restart bot
3. Warmup phase collects bars
4. After warmup: detects existing position
5. Resumes exit management (time-stop, bracket monitoring)

**Verify:**
- [ ] Brackets survived crash (verify in IB TWS)
- [ ] Bot detects position on restart
- [ ] No duplicate entry attempted
- [ ] If outside RTH on restart with position: emergency flatten fires

### 5.3 Drill Verification Checklist (Post-Drill)

After EVERY drill, confirm:
- [ ] Account position = 0 (flat)
- [ ] No orphan orders on IB
- [ ] Kill-switch is in expected state (active or inactive)
- [ ] Bot is in expected state (running or stopped)
- [ ] Logs contain complete audit trail of the drill
- [ ] Telegram alerts were received for every triggered event

---

## 6. FUTURE OPTIONALITY (NO ACTION NOW)

These are hard prerequisites that must be met before expanding scope. They are not goals or plans — they are gates.

### 6.1 Increasing Account Size (Adding Capital)

**Prerequisites (ALL required):**
1. ≥ 100 live trades at current capital level
2. Live PF ≥ 1.3 over the full live sample
3. Live WR ≥ 55% over the full live sample
4. Max live drawdown was < 8%
5. No kill-switch activations in last 40 trading days
6. Added capital comes from external deposit, NOT from "earned" P&L counting
7. All scaling tiers re-validated with new capital base (recalculate % thresholds)

**What changes:**
- All dollar-based risk limits scale proportionally
- Percentage-based limits remain fixed
- `max_contracts` may increase per tier table
- Strategy parameters: UNCHANGED

### 6.2 Trading ES Instead of MES

**Prerequisites (ALL required):**
1. Account equity ≥ $25,000 (minimum for 1 ES contract margin + 2× buffer)
2. ≥ 200 live MES trades with PF ≥ 1.3
3. Separate ES backtest confirming identical signal behavior (same data, same results scaled by 50/5)
4. ES bracket order testing on paper: confirm TP/SL fill behavior matches MES
5. Slippage measurement on ES paper: confirm ≤ 0.25 pts avg
6. Risk per trade on ES: $250 per point × stop_pts. A 15pt stop = $3,750 risk. Account must support this within 2% risk rule.

**Minimum account for ES at 2% risk per trade:**
- Typical stop: 15 pts × $50/pt = $750 risk
- 2% rule: $750 / 0.02 = **$37,500 minimum**
- With margin buffer: **$40,000+**

**Do not trade ES below $40K account.**

### 6.3 Reconsidering Overnight Trading

**Prerequisites (ALL required):**
1. Account equity ≥ $20,000 (allows 15pt stops within 0.375% risk)
2. RTH strategy has ≥ 6 months live track record with PF ≥ 1.5
3. New overnight edge identified with:
   - Backtest Sharpe > 1.0
   - Backtest PF > 1.5
   - Max DD < 10%
   - Sample size ≥ 200 trades
4. Overnight strategy uses SEPARATE capital allocation (not shared with RTH)
5. Combined max risk (RTH + overnight) ≤ 2% of total equity per trade
6. Overnight strategy has independent kill-switch and daily/weekly limits
7. 30-day paper trading validation of overnight strategy before live

**Current state:** Overnight is structurally untradeable at $5K. SNR = 0.058. Median MAE = 15.25 pts vs max affordable stop of 2.5 pts. This is not a close call.

### 6.4 Adding a Second Strategy

**Prerequisites (ALL required):**
1. Current strategy has ≥ 6 months live track record
2. New strategy backtested on same data period with:
   - PF ≥ 1.5
   - DD < 10%
   - Low correlation with existing strategy (correlation of daily P&L < 0.3)
3. Combined risk budget: both strategies share the same daily/weekly limits
4. No increase in total contracts beyond current tier
5. Strategies must not generate conflicting signals (e.g., long and short simultaneously)
6. 30-day paper trading validation of new strategy in isolation

### 6.5 Automation of Scaling

**Prerequisites (ALL required):**
1. ≥ 50 trades completed at each tier being automated
2. All scaling transitions logged and reviewed manually for at least 3 cycles
3. De-scaling logic tested under simulated stress (drawdown sequences)
4. Manual override always available to force de-scale
5. Automated scale-up requires next-day confirmation (no intra-day scale-up)

**Until then:** All scaling decisions are manual, logged, and require overnight reflection.

---

## Appendix A: Weekly Review Template

```
Week of: ____________
Equity start: $________  Equity end: $________  Change: $________

Trades: ____  Wins: ____  Losses: ____  WR: ____%
PF: ____  Expectancy: $____/trade
Max single loss: $____  Max consecutive losses: ____
Slippage avg: ____ pts  Bracket fill rate: ____%

Risk events:
  Daily limit hit:    [ ] Yes  [ ] No
  Weekly limit hit:   [ ] Yes  [ ] No
  Kill-switch fired:  [ ] Yes  [ ] No
  RTH flatten fired:  [ ] Yes  [ ] No
  Config changed:     [ ] Yes  [ ] No

Current tier: T____  Contracts: ____
Scale change this week: [ ] None  [ ] Up  [ ] Down  Reason: ________

Edge health:
  Rolling-20 WR: ____%  (threshold: >50%)
  Rolling-20 PF: ____   (threshold: >1.2)
  Rolling-30 z-score: ____ (threshold: > -2.0)

Notes:
_______________________________________________
_______________________________________________

Decision: [ ] Continue  [ ] Pause  [ ] Scale Down  [ ] Full Review
```

---

## Appendix B: Startup Validation Checklist

Run before EVERY bot start:

```
[ ] IB Gateway running and connected
[ ] Account shows correct equity ($____)
[ ] Position = 0 (flat)
[ ] No orphan orders
[ ] Config hash matches expected (no unauthorized changes)
[ ] risk_gate values: daily_max=$250, weekly_max=$500, max_consec=5
[ ] max_contracts = ____ (matches current tier)
[ ] rth_flatten_time_et = "15:50"
[ ] peak_drawdown_enabled = true, pct = 4.0
[ ] kill_switch_active = false (or explain why true)
[ ] Current time is RTH or pre-RTH (do not start during overnight)
[ ] Telegram alerts test: send test message, confirm receipt
[ ] Last backtest result matches reference (PF 2.18, 174 trades)
```

---

*This document is auditable. All thresholds are derived from backtest data. Default action under uncertainty: reduce risk.*
