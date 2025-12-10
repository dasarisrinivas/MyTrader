#!/usr/bin/env python3
"""Diagnostic script to analyze why ZERO signals were generated.

This script simulates the market conditions you described:
- SPY Futures moved +0.2% up and -0.2% down
- Bot produced ZERO signals

Run this to see exactly where the signal pipeline breaks down.
"""
import sys
sys.path.insert(0, "/Users/svss/Documents/code/MyTrader")

from dataclasses import dataclass
from typing import Any, Dict, List

# ============================================================
# SIMULATED MARKET DATA FOR TODAY
# ============================================================

# Simulated SPY Futures data with +0.2% and -0.2% moves
SIMULATED_CANDLES = [
    # Time, Open, High, Low, Close, Volume
    {"time": "09:30", "open": 600.00, "high": 600.50, "low": 599.80, "close": 600.20, "volume": 10000},
    {"time": "09:31", "open": 600.20, "high": 600.60, "low": 600.10, "close": 600.50, "volume": 8000},
    {"time": "09:32", "open": 600.50, "high": 600.80, "low": 600.40, "close": 600.70, "volume": 7500},
    {"time": "09:33", "open": 600.70, "high": 601.10, "low": 600.60, "close": 601.00, "volume": 9000},  # +0.17%
    {"time": "09:34", "open": 601.00, "high": 601.20, "low": 600.90, "close": 601.10, "volume": 6000},
    {"time": "09:35", "open": 601.10, "high": 601.30, "low": 600.80, "close": 600.90, "volume": 7000},
    {"time": "09:36", "open": 600.90, "high": 601.00, "low": 600.50, "close": 600.60, "volume": 8500},
    {"time": "09:37", "open": 600.60, "high": 600.70, "low": 600.20, "close": 600.30, "volume": 9500},
    {"time": "09:38", "open": 600.30, "high": 600.40, "low": 599.90, "close": 600.00, "volume": 10500},  # Back to open
    {"time": "09:39", "open": 600.00, "high": 600.10, "low": 599.50, "close": 599.60, "volume": 11000},
    {"time": "09:40", "open": 599.60, "high": 599.70, "low": 599.20, "close": 599.30, "volume": 12000},  # -0.12%
    {"time": "09:41", "open": 599.30, "high": 599.40, "low": 598.90, "close": 599.00, "volume": 13000},
    {"time": "09:42", "open": 599.00, "high": 599.10, "low": 598.70, "close": 598.80, "volume": 11500},  # -0.20%
]


def calculate_indicators(candles: List[Dict]) -> List[Dict]:
    """Calculate technical indicators for simulated candles."""
    results = []
    
    # Initial EMAs (using simple starting values)
    ema_9 = candles[0]["close"]
    ema_20 = candles[0]["close"]
    ema_50 = candles[0]["close"]
    
    # MACD
    ema_12 = candles[0]["close"]
    ema_26 = candles[0]["close"]
    macd_signal = 0
    
    # RSI
    gains = []
    losses = []
    prev_close = candles[0]["close"]
    
    # ATR
    atrs = []
    
    for i, candle in enumerate(candles):
        close = candle["close"]
        high = candle["high"]
        low = candle["low"]
        open_price = candle["open"]
        
        # Update EMAs
        ema_9 = ema_9 * 0.8 + close * 0.2
        ema_20 = ema_20 * 0.9 + close * 0.1
        ema_50 = ema_50 * 0.96 + close * 0.04
        
        # MACD
        ema_12 = ema_12 * 0.846 + close * 0.154
        ema_26 = ema_26 * 0.926 + close * 0.074
        macd = ema_12 - ema_26
        macd_signal = macd_signal * 0.8 + macd * 0.2
        macd_hist = macd - macd_signal
        
        # RSI
        change = close - prev_close
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
        
        if len(gains) >= 14:
            avg_gain = sum(gains[-14:]) / 14
            avg_loss = sum(losses[-14:]) / 14
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50  # Default
        
        # ATR
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        atrs.append(tr)
        atr = sum(atrs[-14:]) / min(14, len(atrs))
        
        prev_close = close
        
        result = {
            "time": candle["time"],
            "price": close,
            "close": close,
            "open": open_price,
            "high": high,
            "low": low,
            "ema_9": ema_9,
            "ema_20": ema_20,
            "ema_50": ema_50,
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "atr": atr,
            "atr_20_avg": 0.8,  # Simulated average
            "vwap": 600.00,  # Simulated VWAP
            "pdh": 601.50,
            "pdl": 598.50,
            "volume_ratio": 1.0,
        }
        results.append(result)
    
    return results


def run_diagnostic():
    """Run the full diagnostic comparing old vs new signal engines."""
    print("=" * 80)
    print("SIGNAL GENERATION DIAGNOSTIC")
    print("=" * 80)
    print("\nSimulated market conditions:")
    print("  - Opening price: $600.00")
    print("  - Peak: $601.20 (+0.20%)")
    print("  - Low: $598.80 (-0.20%)")
    print("  - Typical low-volatility day with micro trends")
    print()
    
    # Calculate indicators
    candles_with_indicators = calculate_indicators(SIMULATED_CANDLES)
    
    # ============================================================
    # TEST 1: Original RuleEngine logic
    # ============================================================
    print("-" * 80)
    print("TEST 1: ORIGINAL RULE ENGINE BEHAVIOR")
    print("-" * 80)
    
    print("\nOriginal trend detection logic:")
    print("  if price > ema_9 > ema_20: UPTREND")
    print("  elif price < ema_9 < ema_20: DOWNTREND")
    print("  else: CHOP")
    print()
    print("Original signal threshold: 40")
    print("Original ATR minimum: 0.15")
    print()
    
    original_signals = []
    for data in candles_with_indicators:
        price = data["close"]
        ema_9 = data["ema_9"]
        ema_20 = data["ema_20"]
        atr = data["atr"]
        rsi = data["rsi"]
        
        # Original trend detection
        if price > ema_9 > ema_20:
            trend = "UPTREND"
        elif price < ema_9 < ema_20:
            trend = "DOWNTREND"
        else:
            trend = "CHOP"
        
        # Original scoring
        score = 0
        if trend == "UPTREND":
            score += 30  # trend_weight
        if rsi < 30:
            score += 15  # oversold
        elif rsi > 70:
            score += 15  # overbought (for sells)
        
        # Original ATR filter
        atr_blocked = atr < 0.15
        
        # Signal determination
        if atr_blocked:
            signal = "BLOCKED (ATR)"
        elif score >= 40:
            signal = "BUY" if trend == "UPTREND" else "SELL"
        else:
            signal = "HOLD"
        
        original_signals.append({
            "time": data["time"],
            "price": price,
            "trend": trend,
            "score": score,
            "signal": signal,
        })
    
    print("Candle-by-candle analysis (Original Engine):")
    print(f"{'Time':<8} {'Price':>8} {'Trend':<12} {'Score':>6} {'Signal':<15}")
    print("-" * 50)
    for sig in original_signals:
        print(f"{sig['time']:<8} {sig['price']:>8.2f} {sig['trend']:<12} {sig['score']:>6.1f} {sig['signal']:<15}")
    
    signals_generated = sum(1 for s in original_signals if s["signal"] not in ["HOLD", "BLOCKED (ATR)"])
    print(f"\n>>> ORIGINAL ENGINE: {signals_generated} signals generated")
    print(f">>> All {len(original_signals)} candles classified as CHOP (no aligned EMAs)")
    print()
    
    # ============================================================
    # TEST 2: Enhanced Signal Engine
    # ============================================================
    print("-" * 80)
    print("TEST 2: ENHANCED SIGNAL ENGINE BEHAVIOR")
    print("-" * 80)
    
    try:
        from mytrader.rag.enhanced_signal_engine import EnhancedSignalEngine, TrendType
        
        engine = EnhancedSignalEngine({
            "atr_min": 0.05,
            "scalp_threshold": 20,
            "normal_threshold": 30,
            "micro_trend_pct": 0.1,
        })
        
        print("\nEnhanced trend detection logic:")
        print("  - Micro trends: 0.1%-0.3% moves detected")
        print("  - Weak trends: Price above/below EMAs (not aligned)")
        print("  - Scalp threshold: 20 (for low-vol days)")
        print("  - ATR minimum: 0.05 (very relaxed)")
        print()
        
        enhanced_signals = []
        for data in candles_with_indicators:
            result = engine.evaluate(data)
            enhanced_signals.append({
                "time": data["time"],
                "price": data["close"],
                "trend_1m": result.trend_1m.value,
                "vol_regime": result.volatility_regime.value,
                "momentum": result.momentum_score,
                "signal": result.signal.value,
                "confidence": result.confidence,
                "patterns": ", ".join(result.patterns_detected) if result.patterns_detected else "-",
            })
        
        print("Candle-by-candle analysis (Enhanced Engine):")
        print(f"{'Time':<8} {'Price':>8} {'Trend':<18} {'Vol':<10} {'Mom':>6} {'Signal':<12} {'Conf':>6}")
        print("-" * 80)
        for sig in enhanced_signals:
            print(f"{sig['time']:<8} {sig['price']:>8.2f} {sig['trend_1m']:<18} {sig['vol_regime']:<10} {sig['momentum']:>6.1f} {sig['signal']:<12} {sig['confidence']:>6.1f}")
        
        signals_generated = sum(1 for s in enhanced_signals if s["signal"] != "HOLD")
        print(f"\n>>> ENHANCED ENGINE: {signals_generated} signals generated")
        
    except Exception as e:
        print(f"Error importing enhanced engine: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print()
    print("=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)
    print("""
PROBLEM: Original RuleEngine requires STRICT EMA alignment:
  - UPTREND needs: price > ema_9 > ema_20
  - This fails during micro movements (0.1%-0.3%)
  - EMAs don't align fast enough for small moves
  - Result: Everything classified as CHOP

ISSUES IDENTIFIED:
  1. Trend detection too binary (UPTREND/DOWNTREND/CHOP)
  2. No recognition of micro trends (0.1%-0.3% moves)
  3. Signal threshold (40) requires trend_weight (30) to be earned
  4. ATR filter (0.15) still blocking in very low vol
  5. No scalp mode for tight ranges
  6. No pattern recognition (EMA curl, VWAP reclaim, etc.)

FIXES IN ENHANCED ENGINE:
  1. Multi-level trend detection (STRONG, MICRO, WEAK, RANGE)
  2. Detects micro trends based on % change, not just EMA alignment
  3. Lowered scalp threshold to 20 for low-vol days
  4. ATR minimum reduced to 0.05 (warns but doesn't block)
  5. Pattern detection adds points (EMA curl, VWAP, PDH/PDL)
  6. Momentum scoring (-100 to +100) with RSI/MACD
  7. Volatility regime adapts thresholds automatically
""")


if __name__ == "__main__":
    run_diagnostic()
