# Signal Generation Fixes - December 9, 2025

## Problem Statement

The trading bot produced **ZERO signals** despite SPY Futures moving ±0.2% up and down throughout the day. All candles were classified as `trend=CHOP, vol=LOW` with `confidence=0.00`.

## Root Cause Analysis

### 1. Overly Strict Trend Detection
The original `RuleEngine` used binary trend classification:
```python
# OLD (too strict)
if price > ema_9 > ema_20:
    result.market_trend = "UPTREND"
elif price < ema_9 < ema_20:
    result.market_trend = "DOWNTREND"
else:
    result.market_trend = "CHOP"  # Everything else!
```

This required **perfect EMA alignment** (`price > ema_9 > ema_20`), which rarely happens during:
- Micro movements (0.1%-0.3%)
- Choppy days with small oscillations
- Early trend formation before EMAs align

### 2. ATR Filter Blocking Low-Vol Days
The ATR minimum filter (0.15) was blocking signals on low-volatility days where ATR drops below the threshold.

### 3. Signal Threshold Too High
The signal threshold of 40 required the trend_weight (30) to be earned plus additional momentum. Without a trend classification, the bot couldn't reach the threshold.

## Fixes Implemented

### 1. Enhanced Trend Detection
**File:** `mytrader/rag/hybrid_rag_pipeline.py`

Added multi-level trend classification:
- **UPTREND/DOWNTREND**: Full EMA alignment (unchanged)
- **MICRO_UP/MICRO_DOWN**: 0.1%+ price change with price above/below short EMA
- **WEAK_UP/WEAK_DOWN**: Price above/below EMA with positive/negative EMA slope

```python
# NEW (more sensitive)
if price > ema_9 > ema_20:
    result.market_trend = "UPTREND"
elif price < ema_9 < ema_20:
    result.market_trend = "DOWNTREND"
elif pct_change >= 0.1 and price > ema_9:
    result.market_trend = "MICRO_UP"
elif pct_change <= -0.1 and price < ema_9:
    result.market_trend = "MICRO_DOWN"
elif price > ema_9 and ema_diff_pct > 0.02:
    result.market_trend = "WEAK_UP"
elif price < ema_9 and ema_diff_pct < -0.02:
    result.market_trend = "WEAK_DOWN"
else:
    result.market_trend = "CHOP"
```

### 2. Relaxed ATR Filter
**File:** `mytrader/rag/hybrid_rag_pipeline.py`

- Low-vol days: ATR minimum reduced to 0.05 (warn only, don't block)
- Normal days: ATR minimum remains at 0.15

### 3. Scalp Mode for Low-Volatility
**File:** `mytrader/rag/hybrid_rag_pipeline.py`

- Added `TradeAction.SCALP_BUY` and `TradeAction.SCALP_SELL`
- Scalp threshold: 20 (vs normal 40)
- Scalps generate signals on micro-trends

### 4. Partial Credit for Micro-Trends
**File:** `mytrader/rag/hybrid_rag_pipeline.py`

```python
if result.market_trend == "UPTREND":
    buy_score += self.trend_weight  # 100%
elif result.market_trend == "MICRO_UP":
    buy_score += self.trend_weight * 0.7  # 70%
elif result.market_trend == "WEAK_UP":
    buy_score += self.trend_weight * 0.4  # 40%
```

### 5. Scalp Risk Management
**File:** `mytrader/execution/live_trading_manager.py`

- Scalp trades use tighter stops (60% of normal)
- Scalp targets are smaller (50% of normal)
- Falls back to ATR * 0.75 for stop if no pipeline params

## Test Results

| Scenario | Price Change | Old Result | New Result |
|----------|--------------|------------|------------|
| Micro Up | +0.15% | CHOP → HOLD | MICRO_UP → SCALP_BUY (31) |
| Micro Down | -0.12% | CHOP → HOLD | MICRO_DOWN → SCALP_SELL (31) |
| Weak Up | +0.02% | CHOP → HOLD | UPTREND → SCALP_BUY (57.5) |

## Files Modified

1. `mytrader/rag/hybrid_rag_pipeline.py`
   - Enhanced `RuleEngine.evaluate()` with micro-trend detection
   - Added `SCALP_BUY`, `SCALP_SELL` to `TradeAction` enum
   - Relaxed ATR filter for low-vol days
   - Added scalp mode with lower threshold

2. `mytrader/execution/live_trading_manager.py`
   - Handle `SCALP_BUY`/`SCALP_SELL` signals
   - Tighter stop/target for scalp trades
   - Updated exit logic to recognize scalp signals

3. `mytrader/rag/enhanced_signal_engine.py` (NEW)
   - Full-featured enhanced signal engine with pattern detection
   - Can be used as drop-in replacement for more sophisticated analysis

## Configuration

Current config in `config.yaml`:
```yaml
hybrid:
  enabled: true
  atr_min: 0.15  # Relaxed to 0.05 in scalp mode
  signal_threshold: 40  # Lowered to 20 in scalp mode
```

## Expected Behavior

On low-volatility days (±0.2% moves):
1. System detects `volatility_regime: LOW`
2. Activates scalp mode (threshold = 20)
3. Micro-trends (0.1%+) generate SCALP_BUY/SCALP_SELL signals
4. Tighter risk management (60% stop, 50% target)
5. ATR filter warns but doesn't block

## Monitoring

Watch for these log messages:
- `"Hybrid Pipeline: SCALP_BUY"` - Scalp signal generated
- `"trend=MICRO_UP"` or `"trend=MICRO_DOWN"` - Micro-trend detected
- `"Using SCALP risk params"` - Tighter stop/target applied
- `"VERY_LOW_ATR"` - ATR warning (not blocking)
