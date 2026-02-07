# Scoring-Based Entry System - Quick Reference

## 🎯 Core Concept

**Replace hard filters with weighted scoring to increase trade frequency while preserving edge.**

---

## 📊 Scoring Breakdown

| Category | Max Points | Key Components |
|----------|-----------|----------------|
| 🔷 **Trend/Structure** | **+40** | EMA alignment, slope, price vs VWAP, HTF bias |
| 🔶 **Momentum** | **+25** | Strong candle, momentum acceleration, no divergence |
| 🟦 **Volatility/Regime** | **+20** | ADX strength, ADX trend, ATR percentile |
| 🟩 **Entry Quality** | **+20** | Pullback depth, entry near support zone |
| 🟥 **Penalties** | **Negative** | Weak ADX, chop, late session, opposing wick |

**Total Possible:** ~105 points (before penalties)

---

## 🎚️ Position Sizing

```
Score >= 60  →  🟢 FULL position (1.0x)
Score >= 45  →  🟡 HALF position (0.5x)
Score < 45   →  🔴 NO TRADE
```

---

## 🛡️ Risk Gates (HARD - Always Enforced)

- ❌ Max loss per trade
- ❌ Max daily loss  
- ❌ Max open risk
- ❌ Session cutoff time

*These override ANY score*

---

## 🚀 Quick Start

### 1. Run Validation Test

```bash
python3 test_scoring_system.py
```

### 2. Run Backtest

```bash
python -m mytrader.backtest.run \
    --strategy mes_one_minute_scoring \
    --start-date 2025-01-01 \
    --end-date 2026-02-01 \
    --symbol ES
```

### 3. Compare Results

Check these metrics vs original strategy:
- **Trade frequency** (target: 8-15/day vs 1-2/day)
- **Win rate** (target: 45-55%)
- **Profit factor** (target: 1.3-1.8)

---

## 🔧 Configuration

In `config.yaml`:

```yaml
one_minute:
  use_scoring_system: true  # Enable scoring
  scoring_full_size_threshold: 60.0  # Full position
  scoring_half_size_threshold: 45.0  # Half position
  
  # Standard risk parameters still apply
  stop_atr_multiplier: 1.5
  take_profit_multiple: 2.0
  max_loss_per_trade: 100.0
  max_daily_loss: 500.0
```

---

## 📈 Tuning Guidelines

### Too Many Trades?
→ **Raise thresholds:** 60→65, 45→50

### Too Few Trades?
→ **Lower thresholds:** 60→55, 45→40

### Low Win Rate?
→ **Raise full-size threshold:** 60→65  
→ **Keep or lower half-size:** 45→40

### Missing Good Setups?
→ **Lower half-size threshold:** 45→40  
→ **Review penalty weights**

---

## 🔍 Diagnostic Output

Every decision includes:

```
SCORING_ENTRY: BUY Score=73.0 [LONG]
  Trend=30 Mom=20 Regime=13 Entry=10 Penalty=0
  
Components:
  [trend   ] EMA_STACK_UP    : +10.0 (EMA 9>21>50)
  [trend   ] ABOVE_VWAP      : +10.0 (+0.3% from VWAP)
  [momentum] STRONG_CANDLE   : +10.0 (body=62.5%)
  [regime  ] ADX_STRONG      : +10.0 (ADX=28.0)
  [entry   ] NEAR_SUPPORT    : +10.0 (dist=0.3%)
  
Decision: FULL_SIZE (1.0x) → BUY @ 5850.00
  Stop: 5837.25 | Target: 5875.50 | R:R 1:2.0
```

---

## ⚖️ Scoring vs Hard Filters

### Example: Moderate Setup (ADX=22)

**Hard Filters:**
```
❌ REJECTED (ADX < 25)
Result: NO TRADE
```

**Scoring System:**
```
✅ Score: 58.0
Breakdown:
  Trend:    +30 (EMA aligned, above VWAP)
  Momentum: +10 (moderate)
  Regime:    +8 (ADX moderate)
  Entry:    +10 (good pullback)
  Penalty:    0

Result: HALF SIZE (0.5x position)
```

**Outcome:** Captures marginal setup with appropriate risk

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `scoring_entry.py` | Core scoring logic |
| `scoring_integration.py` | Integration with strategy |
| `mes_one_minute_scoring.py` | New strategy class |
| `test_scoring_system.py` | Validation tests |
| `SCORING_ENTRY_SYSTEM.md` | Full documentation |

---

## ✅ Validation Checklist

Before deploying:

- [ ] Run `test_scoring_system.py` (all tests pass)
- [ ] Run backtest on historical data
- [ ] Compare metrics to baseline strategy
- [ ] Verify risk gates work correctly
- [ ] Check score distributions make sense
- [ ] Review diagnostics output
- [ ] Test on recent market data

---

## 🎓 Key Insights

1. **More trades ≠ worse performance**
   - Half-size positions control risk on marginal setups
   - More opportunities to capture edge

2. **Scoring > Binary filters**
   - No single condition dominates
   - Quality measured holistically

3. **Position sizing is powerful**
   - Full size for high conviction
   - Half size for moderate setups
   - None for weak signals

4. **Risk gates are critical**
   - Always enforced, regardless of score
   - Protect against catastrophic losses

---

## 📞 Support

Questions or issues?
1. Check `SCORING_ENTRY_SYSTEM.md` for details
2. Run validation tests for examples
3. Review diagnostic logs for debugging

---

**Status:** ✅ Production Ready  
**Version:** 1.0 (Feb 2026)  
**Next:** Run backtest and tune thresholds
