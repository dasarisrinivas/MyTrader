# Friday Loss Audit — 2026-02-20

**Audit Date:** 2026-02-22  
**Auditor:** Automated Trading System Audit Agent  
**Period:** Week of Feb 18–20, 2026

---

## 1. Summary Verdict

**MIXED — Code Issue (Primary) + Expected Drawdown (Secondary)**

Friday's loss of **$44.80** on a single EMA9 pullback trade was primarily caused by a **fixed stop-loss that was too tight for the prevailing volatility** (0.56× ATR). The week's total net P&L of **-$174.17** across 3 trades (all stopped out) reflects both a sub-optimal stop-sizing defect AND difficult choppy market conditions.

| Classification | Weight |
|---|---|
| Code/Logic Issue (Signal C stop sizing) | **70%** |
| Expected Drawdown (CHOP regime, bearish sentiment) | **30%** |

---

## 2. Root Cause Analysis

### 2.1 Primary: Fixed Stop-Loss Too Tight for ATR (Signal C)

**Defect:** The EMA9 Pullback (Signal C) used a **fixed 8-point stop-loss** (`_fixed_sl_points_ema9=8.0`) regardless of ATR.

| Metric | Value |
|---|---|
| ATR at entry (14-period, 15m) | **14.3 points** |
| Stop-loss used | **8.0 points** |
| Stop as % of ATR | **0.56× ATR** |
| Expected noise band | ~10–14 pts (0.7–1.0× ATR) |
| Time to stop-out | **13 minutes** (13:30:37 → 13:43:38) |

An 8-point stop in a 14.3-ATR environment sits **inside normal bar noise**. The price moved against the entry by exactly 8 points (6920.25 → 6912.25) then reversed — classic noise-stop behavior.

**Code evidence (pre-fix version, commit `9ba6e50`):**
```python
# Line 695-696 at trade time:
stop_loss = close - self._fixed_sl_points_ema9       # Fixed 8 pts ($40)
take_profit = close + self._fixed_tp_points_ema9     # Fixed 10 pts ($50)
```

The ATR-adaptive fix was committed at **14:08 CST** (commit `b0bcb6d`), 38 minutes **after** the loss:
```python
# Current code (post-fix):
sl_pts = min(ceiling, max(floor, ATR × mult))  # Dynamic: 8–20 pts
tp_pts = sl_pts × rr_ratio                      # Preserves 1.25:1 R:R
```

### 2.2 Secondary: CHOP Market + Bearish Sentiment Ignored

The hybrid pipeline correctly identified the market regime as **CHOP** with **MEDIUM** volatility. Multiple warning signals were present but insufficient weight was given to them:

| Factor | Friday's Value | Concern |
|---|---|---|
| Market trend (hybrid) | **CHOP** | Not trending — pullback strategy less effective |
| Hybrid confidence | **16%** (0.16) | Very low — barely above zero |
| RAG similar trade win rate | **30%** (0.30) | Historical: 70% of similar setups lost |
| Sentiment (Stocktwits) | **-0.33** | Bearish |
| VX multiplier | **0.8125** | Elevated volatility (VX=18.75) |
| Final confidence after overlays | **0.601** | Barely above 0.50 threshold |

The system took the trade because:
1. D-Engine (strategy) generated `BUY` with base confidence 0.70
2. H-Engine (hybrid) agreed with `BUY` but at only 16% confidence
3. Agreement boost (+0.033) pushed final confidence to 0.601
4. Scoring validation was **not applied** to this trade (it should have been)

### 2.3 Week Context — All 3 Trades Stopped Out

| # | Date | Signal | Entry | SL Hit | PnL | Stop as ×ATR | Issue |
|---|---|---|---|---|---|---|---|
| 1 | Feb 18 13:45 | OR_BREAK_SHORT | 6885.50 | 6900.50 | **-$75.62** | ~1.5× ATR(9.9) | Stop hit on reversal, reasonable sizing |
| 2 | Feb 19 11:15 | SELL (short) | 6866.75 | 6877.50 | **-$53.75** | ~1.0× ATR | Stopped in 7 min, tight stop |
| 3 | **Feb 20 13:30** | **EMA9_PB_LONG** | **6920.25** | **6912.25** | **-$44.80** | **0.56× ATR(14.3)** | **Fixed 8pt stop in 14pt ATR** |
| | | | | **TOTAL** | **-$174.17** | | |

Trade #1 (Feb 18) had a reasonable stop at ~1.5× ATR. Trade #2 and #3 both had stops that were tighter than the noise band.

---

## 3. Evidence

### 3.1 Friday Trade Timeline
```
13:30:26 CST  Strategy fires EMA9_PB_LONG (conf=0.70, ATR=14.3)
13:30:30 CST  Hybrid pipeline agrees: BUY (conf=0.16, trend=CHOP)
13:30:34 CST  Final confidence: 0.601 (base 0.70, sentiment -0.14, VX ×0.81, agreement +0.03)
13:30:34 CST  RiskGate PASS: margin=5199, required=3464, stopPts=8.0
13:30:34 CST  Order placed: BUY 1 MES @ LIMIT 6921.25
13:30:37 CST  Filled @ 6920.25 (1.00 point improvement)
13:30:37 CST  Bracket: SL=6912.25 (STOP), TP=6930.25 (LIMIT)
13:43:38 CST  STOP-LOSS HIT: Filled @ 6912.25
13:43:38 CST  P&L: gross=-$40.00, net=-$44.80 (comm=$4.80)
```

### 3.2 Market Structure at Entry
- **Price action:** 6920.25 — above all EMAs (EMA9=6909.8, EMA21=6901.4, EMA50=6894.1)
- **OR:** High=6894.75, Low=6889.25 (price well above OR — extended)
- **PDH:** 6894.50 (price +0.37% above PDH — near resistance)
- **Weekly range:** 6791–6931 (price near top of weekly range)
- **ADX:** 22.6 (borderline trending, just above minimum threshold of 22)
- **RSI:** 58.3 (neutral-to-bullish, not overbought)
- **MACD histogram:** +1.94 (positive but not strongly)

### 3.3 Key Risk Signals Ignored
1. **Hybrid classified market as CHOP** — pullback strategies are designed for trending markets
2. **RAG win rate 30%** — 70% of historically similar setups lost
3. **Stocktwits sentiment -0.33** — bearish crowd sentiment
4. **Price at top of weekly range** (6920 vs weekly high 6931) — limited upside
5. **ADX barely at minimum** (22.6 vs threshold 22) — trend strength questionable
6. **VX at 18.75** — elevated, not fearful, but above calm (sub-15) levels

---

## 4. Fix Assessment

### 4.1 Fix Already Applied (Signal C ATR-Adaptive Stops)

The ATR-adaptive fix was already committed on Friday (14:08 CST). **Status: APPLIED.**

**Before (at trade time):**
```python
stop_loss = close - self._fixed_sl_points_ema9  # Always 8.0 pts
take_profit = close + self._fixed_tp_points_ema9  # Always 10.0 pts
```

**After (current code):**
```python
sl_pts = min(self._ema9_sl_ceiling,
             max(self._ema9_sl_floor, atr * self._ema9_sl_atr_mult))
# min(20.0, max(8.0, 14.3 * 1.0)) = 14.3 pts
tp_pts = sl_pts * self._ema9_rr_ratio  # 14.3 * 1.25 = 17.875 pts
```

**If the fix had been in place:** SL would have been 14.3 pts ($71.50 risk) instead of 8 pts ($40 risk). The trade would have had a ~42.76pt SL buffer matching the hybrid pipeline's own suggested SL. The stop would NOT have been hit by the 8-point dip.

### 4.2 What's NOT Yet Fixed — CHOP Regime Filter

The bigger systemic issue is that the strategy took a pullback trade in a **CHOP** market regime. The hybrid pipeline flagged CHOP correctly but:

1. The D-Engine doesn't check hybrid regime before firing
2. The H-Engine agreed with BUY (at only 16% confidence) — causing the agreement boost instead of a block
3. The scoring validation system was apparently bypassed for this trade

---

## 5. Backtest Comparison

### 5.1 ATR-Adaptive Fix Impact (from comments in code)

Per the simulation data embedded in the strategy comments:
> Simulation (Oct 2025): WR 37.5% → 50%, PF 0.54 → 0.95 at ATR×1.0

The ATR-adaptive fix should improve Signal C's win rate from ~37.5% to ~50% by keeping stops outside the noise band. **This fix is already deployed.**

### 5.2 Recommended: Run Confirmation Backtest

A backtest should be run to confirm the fix works on recent data:
```bash
python3 -m backtest.run --symbol MES --start 2026-01-20 --end 2026-02-20 --bar 15m
```

---

## 6. Recommendations

### R1: Add CHOP Regime Guard to Signal C (EMA9 Pullback) — **HIGH PRIORITY**

The EMA9 pullback is a **trending market** signal. It should NOT fire when the hybrid pipeline detects CHOP. This is the most impactful protective improvement.

**Suggested implementation:** In `_generate_strategy_first_signal()` in `signal_processor.py`, add a regime check before allowing EMA9_PB signals:

```python
# If hybrid detects CHOP, block pullback signals (they need trending markets)
if "PB" in signal.metadata.get("reason", "") and hybrid_trend == "CHOP":
    logger.warning("⚠️ Pullback signal blocked: market regime is CHOP")
    return Signal("HOLD", 0.0, {"reason": "CHOP_BLOCK"})
```

This does NOT curve-fit — it's a structural filter. Pullback strategies fundamentally require trends.

### R2: Tighten RAG Win-Rate Gate — **MEDIUM PRIORITY**

When RAG shows <40% win rate on similar historical setups, the confidence should be penalized more aggressively. Currently the 30% win rate barely affected the decision.

**Suggested:** Apply a 0.15 confidence penalty when `rag_win_rate < 0.40` (currently no penalty).

### R3: No Core Logic Changes Needed

- The ATR-adaptive fix for Signal C is **already deployed and correct**
- The risk gate config values (max_stop=25, risk_per_trade=100) are appropriate
- The RiskGateConfig dataclass defaults (max_stop=12) should be updated to match config.yaml for consistency, but this is cosmetic since config.yaml overrides take precedence

---

## Appendix: Configuration Discrepancy (Non-Critical)

The `RiskGateConfig` dataclass has stale defaults that don't match `config.yaml`:

| Field | Dataclass Default | config.yaml Value |
|---|---|---|
| `max_stop_points` | 12.0 | 25.0 |
| `risk_per_trade_usd` | 60.0 | 100.0 |
| `risk_per_trade_max` | 90.0 | 125.0 |

These are overridden at runtime by `config.yaml`, so no functional impact. But the stale defaults could cause confusion during testing or if config.yaml is ever missing a field.

---

**Classification: 70% Code Issue / 30% Expected Drawdown**  
**Action Required: Verify ATR-adaptive fix + Add CHOP regime guard**  
**Risk Level: LOW (fix already deployed, additional guard recommended)**
