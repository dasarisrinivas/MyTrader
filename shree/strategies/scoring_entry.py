"""Scoring-Based Entry System for ES/MES 1-Minute Strategy.

This module replaces hard rejection filters with a weighted scoring system
to increase trade frequency while preserving edge through intelligent weighting.

CORE PHILOSOPHY:
- Hard filters reject 99% of setups → Scoring system weighs conditions
- Each condition adds/subtracts points rather than blocking entirely
- Risk containment remains as HARD gates (non-negotiable)
- More trades, better position sizing, preserved edge

SCORING BREAKDOWN:
1. Trend/Structure (max +40 points)
   - EMA alignment
   - EMA slope direction
   - Price vs VWAP
   - Higher timeframe bias

2. Momentum (max +25 points)
   - Strong momentum candle
   - Momentum acceleration
   - No divergence

3. Volatility/Regime (max +20 points)
   - ADX strength
   - ADX trend
   - ATR percentile

4. Pullback/Entry Quality (max +20 points)
   - Pullback depth
   - Entry near support zone

5. Penalties (negative scoring)
   - Weak ADX (< 15): -15
   - Chop/range regime: -10
   - Late session/lunch: -5
   - Large opposing wick: -5

POSITION SIZING:
- Score >= 60: Full position (1.0x)
- Score >= 45: Half position (0.5x)
- Score < 45: No trade

RISK GATES (HARD):
- Max loss per trade
- Max daily loss
- Max open risk
- Session cutoff times

Author: Senior Quantitative Trading Engineer - Feb 2026
"""
from dataclasses import dataclass, field
from datetime import time, datetime
from typing import Dict, Optional, List, Tuple, Any
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger


class PositionSize(Enum):
    """Position sizing based on signal score."""
    NONE = 0.0
    HALF = 0.5
    FULL = 1.0


@dataclass
class ScoreComponent:
    """Individual scoring component with value and reason."""
    name: str
    value: float
    reason: str
    category: str  # 'trend', 'momentum', 'regime', 'entry', 'penalty'


@dataclass
class SignalScore:
    """Complete signal scoring with component breakdown and diagnostics.
    
    Attributes:
        total_score: Final weighted score (0-100+ scale, penalties can go negative)
        components: List of all scoring components
        direction: 'LONG' or 'SHORT'
        position_size: PositionSize enum
        reasons: Human-readable summary of key contributors
        metadata: Additional diagnostic data
    """
    total_score: float
    components: List[ScoreComponent] = field(default_factory=list)
    direction: str = "NONE"
    position_size: PositionSize = PositionSize.NONE
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_component(self, name: str, value: float, reason: str, category: str):
        """Add a scoring component."""
        self.components.append(ScoreComponent(name, value, reason, category))
        self.total_score += value
        
    def get_category_score(self, category: str) -> float:
        """Get total score for a specific category."""
        return sum(c.value for c in self.components if c.category == category)
    
    def get_breakdown(self) -> Dict[str, float]:
        """Get score breakdown by category."""
        return {
            'trend': self.get_category_score('trend'),
            'momentum': self.get_category_score('momentum'),
            'regime': self.get_category_score('regime'),
            'entry': self.get_category_score('entry'),
            'penalty': self.get_category_score('penalty'),
            'total': self.total_score
        }
    
    def format_diagnostic(self) -> str:
        """Format comprehensive diagnostic string for logging."""
        breakdown = self.get_breakdown()
        components_str = " | ".join([
            f"{c.name}={c.value:+.1f} ({c.reason})"
            for c in self.components
        ])
        return (
            f"Score={self.total_score:.1f} [{self.direction}] "
            f"Trend={breakdown['trend']:.0f} Mom={breakdown['momentum']:.0f} "
            f"Regime={breakdown['regime']:.0f} Entry={breakdown['entry']:.0f} "
            f"Penalty={breakdown['penalty']:.0f} | {components_str}"
        )


def calculate_trend_structure_score(
    data: Dict[str, Any],
    prev_data: Optional[Dict[str, Any]] = None
) -> List[ScoreComponent]:
    """Calculate trend/structure score (max +40 points).
    
    Components:
    - EMA fast > EMA slow: +10
    - EMA slope positive: +10
    - Price above VWAP: +10
    - Higher-TF bias aligned: +10
    
    Args:
        data: Current bar data with indicators
        prev_data: Previous bar data for slope calculations
        
    Returns:
        List of ScoreComponent objects
    """
    components = []
    
    # Extract values
    close = float(data.get('close', 0))
    ema_9 = float(data.get('EMA_9', close))
    ema_21 = float(data.get('EMA_21', close))
    ema_50 = float(data.get('EMA_50', ema_21))
    vwap = float(data.get('SESSION_VWAP', close))
    trend_label_htf = data.get('trend_label_htf', 'UNKNOWN')
    
    # Determine primary direction from 1m EMAs
    direction = "LONG" if ema_9 > ema_21 else "SHORT"
    
    # 1. EMA alignment (fast > slow)
    # FEB 3 2026: Added penalty for trading against EMA stack
    ema_stack_aligned = False
    if direction == "LONG":
        if ema_9 > ema_21 and ema_21 > ema_50:
            components.append(ScoreComponent(
                "EMA_STACK_UP", 10.0, "EMA 9>21>50", "trend"
            ))
            ema_stack_aligned = True
        elif ema_9 > ema_21:
            components.append(ScoreComponent(
                "EMA_PARTIAL_UP", 5.0, "EMA 9>21 only", "trend"
            ))
        elif ema_9 < ema_21 and ema_21 < ema_50:
            # Trading LONG but EMA stack is DOWN - strong penalty
            components.append(ScoreComponent(
                "COUNTER_EMA_STACK", -15.0, "LONG but EMA 9<21<50", "penalty"
            ))
    else:  # SHORT
        if ema_9 < ema_21 and ema_21 < ema_50:
            components.append(ScoreComponent(
                "EMA_STACK_DOWN", 10.0, "EMA 9<21<50", "trend"
            ))
            ema_stack_aligned = True
        elif ema_9 < ema_21:
            components.append(ScoreComponent(
                "EMA_PARTIAL_DOWN", 5.0, "EMA 9<21 only", "trend"
            ))
        elif ema_9 > ema_21 and ema_21 > ema_50:
            # Trading SHORT but EMA stack is UP - strong penalty
            components.append(ScoreComponent(
                "COUNTER_EMA_STACK", -15.0, "SHORT but EMA 9>21>50", "penalty"
            ))
    
    # 2. EMA slope (compare to previous bar)
    if prev_data is not None:
        prev_ema_21 = float(prev_data.get('EMA_21', ema_21))
        ema_slope = ema_21 - prev_ema_21
        
        if direction == "LONG" and ema_slope > 0:
            components.append(ScoreComponent(
                "EMA_SLOPE_UP", 10.0, f"slope={ema_slope:.2f}", "trend"
            ))
        elif direction == "SHORT" and ema_slope < 0:
            components.append(ScoreComponent(
                "EMA_SLOPE_DOWN", 10.0, f"slope={ema_slope:.2f}", "trend"
            ))
        elif abs(ema_slope) < 0.1:
            # Flat slope - meaningful penalty (no directional conviction)
            # FEB 2026 FIX: Increased from -2 to -8. Flat EMA slope on 1m MES
            # means the market is going nowhere; trend entries here chop out.
            components.append(ScoreComponent(
                "EMA_FLAT", -8.0, "slope near zero", "trend"
            ))
    
    # 3. Price vs VWAP
    if direction == "LONG" and close > vwap:
        distance_pct = ((close - vwap) / vwap) * 100
        components.append(ScoreComponent(
            "ABOVE_VWAP", 10.0, f"+{distance_pct:.1f}% from VWAP", "trend"
        ))
    elif direction == "SHORT" and close < vwap:
        distance_pct = ((vwap - close) / vwap) * 100
        components.append(ScoreComponent(
            "BELOW_VWAP", 10.0, f"-{distance_pct:.1f}% from VWAP", "trend"
        ))
    elif direction == "LONG" and close < vwap:
        # Price below VWAP on LONG - warning but not full penalty
        components.append(ScoreComponent(
            "BELOW_VWAP_WARN", -3.0, "price below VWAP", "trend"
        ))
    elif direction == "SHORT" and close > vwap:
        components.append(ScoreComponent(
            "ABOVE_VWAP_WARN", -3.0, "price above VWAP", "trend"
        ))
    
    # 4. Higher timeframe alignment
    if trend_label_htf != 'UNKNOWN':
        if direction == "LONG" and trend_label_htf in ("UPTREND", "LONG"):
            components.append(ScoreComponent(
                "HTF_ALIGNED_LONG", 10.0, f"HTF={trend_label_htf}", "trend"
            ))
        elif direction == "SHORT" and trend_label_htf in ("DOWNTREND", "SHORT"):
            components.append(ScoreComponent(
                "HTF_ALIGNED_SHORT", 10.0, f"HTF={trend_label_htf}", "trend"
            ))
        elif direction == "LONG" and trend_label_htf in ("DOWNTREND", "SHORT"):
            # Counter-trend - penalty
            components.append(ScoreComponent(
                "HTF_COUNTER", -5.0, f"HTF={trend_label_htf} vs LONG", "trend"
            ))
        elif direction == "SHORT" and trend_label_htf in ("UPTREND", "LONG"):
            components.append(ScoreComponent(
                "HTF_COUNTER", -5.0, f"HTF={trend_label_htf} vs SHORT", "trend"
            ))
    
    return components


def calculate_momentum_score(
    data: Dict[str, Any],
    recent_bars: Optional[pd.DataFrame] = None
) -> List[ScoreComponent]:
    """Calculate momentum score (max +25 points).
    
    Components:
    - Strong momentum candle: +10
    - Momentum increasing vs prior bars: +10
    - No immediate momentum divergence: +5
    
    Args:
        data: Current bar data
        recent_bars: Last 10-20 bars for momentum comparison
        
    Returns:
        List of ScoreComponent objects
    """
    components = []
    
    # Extract values
    close = float(data.get('close', 0))
    open_price = float(data.get('open', close))
    high = float(data.get('high', close))
    low = float(data.get('low', close))
    rsi = float(data.get('RSI_14', 50))
    macd_hist = float(data.get('MACD_hist', 0))
    
    direction = "LONG" if close > open_price else "SHORT"
    candle_body = abs(close - open_price)
    candle_range = high - low
    
    # 1. Strong momentum candle
    if candle_range > 0:
        body_ratio = candle_body / candle_range
        
        if direction == "LONG" and body_ratio > 0.6:
            components.append(ScoreComponent(
                "STRONG_BULL_CANDLE", 10.0, f"body={body_ratio:.1%}", "momentum"
            ))
        elif direction == "SHORT" and body_ratio > 0.6:
            components.append(ScoreComponent(
                "STRONG_BEAR_CANDLE", 10.0, f"body={body_ratio:.1%}", "momentum"
            ))
        elif body_ratio > 0.4:
            # Decent candle
            components.append(ScoreComponent(
                "MODERATE_CANDLE", 5.0, f"body={body_ratio:.1%}", "momentum"
            ))
        else:
            # Weak/doji candle
            components.append(ScoreComponent(
                "WEAK_CANDLE", -3.0, f"body={body_ratio:.1%}", "momentum"
            ))
    
    # 2. Momentum increasing (check recent bars)
    if recent_bars is not None and len(recent_bars) >= 5:
        try:
            # Compare current MACD_hist to recent average
            recent_macd = recent_bars['MACD_hist'].tail(5).dropna()
            if len(recent_macd) >= 3:
                recent_avg = recent_macd.mean()
                
                if direction == "LONG":
                    if macd_hist > recent_avg and macd_hist > 0:
                        components.append(ScoreComponent(
                            "MOM_INCREASING", 10.0, f"MACD_hist={macd_hist:.2f}", "momentum"
                        ))
                    elif macd_hist > 0:
                        components.append(ScoreComponent(
                            "MOM_POSITIVE", 5.0, f"MACD_hist={macd_hist:.2f}", "momentum"
                        ))
                else:  # SHORT
                    if macd_hist < recent_avg and macd_hist < 0:
                        components.append(ScoreComponent(
                            "MOM_INCREASING", 10.0, f"MACD_hist={macd_hist:.2f}", "momentum"
                        ))
                    elif macd_hist < 0:
                        components.append(ScoreComponent(
                            "MOM_NEGATIVE", 5.0, f"MACD_hist={macd_hist:.2f}", "momentum"
                        ))
        except Exception as e:
            logger.debug(f"Could not calculate momentum trend: {e}")
    
    # 3. No momentum divergence (RSI alignment with direction)
    if direction == "LONG":
        if rsi >= 55:
            components.append(ScoreComponent(
                "RSI_ALIGNED", 5.0, f"RSI={rsi:.0f}", "momentum"
            ))
        elif rsi < 45:
            # Bearish RSI on LONG setup - divergence warning
            components.append(ScoreComponent(
                "RSI_DIVERGENCE", -5.0, f"RSI={rsi:.0f} low", "momentum"
            ))
    else:  # SHORT
        if rsi <= 45:
            components.append(ScoreComponent(
                "RSI_ALIGNED", 5.0, f"RSI={rsi:.0f}", "momentum"
            ))
        elif rsi > 55:
            # Bullish RSI on SHORT setup - divergence warning
            components.append(ScoreComponent(
                "RSI_DIVERGENCE", -5.0, f"RSI={rsi:.0f} high", "momentum"
            ))
    
    return components


def calculate_regime_score(
    data: Dict[str, Any],
    atr_percentile: Optional[float] = None
) -> List[ScoreComponent]:
    """Calculate volatility/regime score (max +20 points).
    
    Components:
    - ADX > 25: +10
    - ADX rising: +5
    - ATR in medium/high percentile: +5
    
    Args:
        data: Current bar data
        atr_percentile: Current ATR percentile (0-1)
        
    Returns:
        List of ScoreComponent objects
    """
    components = []
    
    # Extract values
    adx = float(data.get('ADX_14', 0))
    
    # 1. ADX strength
    if adx >= 25:
        components.append(ScoreComponent(
            "ADX_STRONG", 10.0, f"ADX={adx:.1f}", "regime"
        ))
    elif adx >= 20:
        components.append(ScoreComponent(
            "ADX_MODERATE", 5.0, f"ADX={adx:.1f}", "regime"
        ))
    elif adx >= 15:
        components.append(ScoreComponent(
            "ADX_WEAK", 0.0, f"ADX={adx:.1f}", "regime"
        ))
    # Note: ADX < 15 penalty handled in penalty section
    
    # 2. ADX trend (requires previous bar data, handled via metadata)
    adx_rising = data.get('adx_rising', None)
    if adx_rising is True:
        components.append(ScoreComponent(
            "ADX_RISING", 5.0, "ADX trending up", "regime"
        ))
    elif adx_rising is False:
        components.append(ScoreComponent(
            "ADX_FALLING", -2.0, "ADX weakening", "regime"
        ))
    
    # 3. ATR percentile
    if atr_percentile is not None:
        if atr_percentile >= 0.75:
            components.append(ScoreComponent(
                "ATR_HIGH", 5.0, f"ATR_pct={atr_percentile:.1%}", "regime"
            ))
        elif atr_percentile >= 0.50:
            components.append(ScoreComponent(
                "ATR_MEDIUM", 3.0, f"ATR_pct={atr_percentile:.1%}", "regime"
            ))
        elif atr_percentile < 0.25:
            components.append(ScoreComponent(
                "ATR_LOW_WARN", -2.0, f"ATR_pct={atr_percentile:.1%}", "regime"
            ))
    
    return components


def calculate_entry_quality_score(
    data: Dict[str, Any],
    recent_bars: Optional[pd.DataFrame] = None
) -> List[ScoreComponent]:
    """Calculate pullback/entry quality score (max +20 points).
    
    Components:
    - Pullback depth within tolerance zone: +10
    - Entry near EMA/VWAP zone: +10
    
    Args:
        data: Current bar data
        recent_bars: Recent bars for pullback detection
        
    Returns:
        List of ScoreComponent objects
    """
    components = []
    
    # Extract values
    close = float(data.get('close', 0))
    ema_9 = float(data.get('EMA_9', close))
    ema_21 = float(data.get('EMA_21', close))
    vwap = float(data.get('SESSION_VWAP', close))
    
    direction = "LONG" if ema_9 > ema_21 else "SHORT"
    
    # 1. Pullback depth detection
    if recent_bars is not None and len(recent_bars) >= 10:
        try:
            recent_highs = recent_bars['high'].tail(10)
            recent_lows = recent_bars['low'].tail(10)
            
            if direction == "LONG":
                # Look for pullback from recent high
                recent_high = recent_highs.max()
                pullback_depth = (recent_high - close) / recent_high if recent_high > 0 else 0
                
                if 0.002 <= pullback_depth <= 0.015:  # 0.2% to 1.5% pullback
                    components.append(ScoreComponent(
                        "PULLBACK_IDEAL", 10.0, f"depth={pullback_depth:.1%}", "entry"
                    ))
                elif 0.015 < pullback_depth <= 0.03:  # 1.5% to 3% pullback
                    components.append(ScoreComponent(
                        "PULLBACK_MODERATE", 5.0, f"depth={pullback_depth:.1%}", "entry"
                    ))
                elif pullback_depth > 0.05:  # Too deep
                    components.append(ScoreComponent(
                        "PULLBACK_DEEP", -3.0, f"depth={pullback_depth:.1%}", "entry"
                    ))
            else:  # SHORT
                # Look for bounce from recent low
                recent_low = recent_lows.min()
                bounce_depth = (close - recent_low) / recent_low if recent_low > 0 else 0
                
                if 0.002 <= bounce_depth <= 0.015:
                    components.append(ScoreComponent(
                        "BOUNCE_IDEAL", 10.0, f"depth={bounce_depth:.1%}", "entry"
                    ))
                elif 0.015 < bounce_depth <= 0.03:
                    components.append(ScoreComponent(
                        "BOUNCE_MODERATE", 5.0, f"depth={bounce_depth:.1%}", "entry"
                    ))
                elif bounce_depth > 0.05:
                    components.append(ScoreComponent(
                        "BOUNCE_HIGH", -3.0, f"depth={bounce_depth:.1%}", "entry"
                    ))
        except Exception as e:
            logger.debug(f"Could not calculate pullback depth: {e}")
    
    # 2. Entry near support zone (EMA/VWAP)
    if direction == "LONG":
        # Check distance from EMA21 or VWAP (whichever is lower as support)
        support_level = min(ema_21, vwap)
        distance_from_support = (close - support_level) / support_level if support_level > 0 else 0
        
        if distance_from_support <= 0.003:  # Within 0.3%
            components.append(ScoreComponent(
                "NEAR_SUPPORT", 10.0, f"dist={distance_from_support:.1%}", "entry"
            ))
        elif distance_from_support <= 0.008:  # Within 0.8%
            components.append(ScoreComponent(
                "CLOSE_SUPPORT", 5.0, f"dist={distance_from_support:.1%}", "entry"
            ))
    else:  # SHORT
        # Check distance from EMA21 or VWAP (whichever is higher as resistance)
        resistance_level = max(ema_21, vwap)
        distance_from_resistance = (resistance_level - close) / resistance_level if resistance_level > 0 else 0
        
        if distance_from_resistance <= 0.003:
            components.append(ScoreComponent(
                "NEAR_RESISTANCE", 10.0, f"dist={distance_from_resistance:.1%}", "entry"
            ))
        elif distance_from_resistance <= 0.008:
            components.append(ScoreComponent(
                "CLOSE_RESISTANCE", 5.0, f"dist={distance_from_resistance:.1%}", "entry"
            ))
    
    return components


def calculate_penalty_score(
    data: Dict[str, Any],
    timestamp: Optional[datetime] = None
) -> List[ScoreComponent]:
    """Calculate penalty scores (negative points).
    
    Penalties:
    - ADX < 15: -15
    - Chop/range regime: -10
    - Late session or lunch hour: -5
    - Large opposing wick: -5
    
    Args:
        data: Current bar data
        timestamp: Current timestamp for session checks
        
    Returns:
        List of ScoreComponent objects (negative values)
    """
    components = []
    
    # Extract values
    adx = float(data.get('ADX_14', 0))
    trend_label = data.get('trend_label', 'UNKNOWN')
    close = float(data.get('close', 0))
    open_price = float(data.get('open', close))
    high = float(data.get('high', close))
    low = float(data.get('low', close))
    
    # 1. Very weak ADX - SEVERE penalty
    # FEB 2026 FIX: Increased from -15 to -25. ADX < 15 means no trend whatsoever.
    # Trend-following entries in this regime have historically high failure rates.
    # The penalty must be large enough to prevent FULL_SIZE entries in chop.
    if adx < 15:
        components.append(ScoreComponent(
            "ADX_TOO_LOW", -25.0, f"ADX={adx:.1f}", "penalty"
        ))
    
    # 2. Chop/range regime - substantial penalty
    # FEB 2026 FIX: Increased from -10 to -20. When market is classified as CHOP/RANGING,
    # trend-following entries have the worst expectancy. This penalty combined with
    # ADX_TOO_LOW ensures chop produces scores well below full-size threshold.
    if trend_label in ('CHOP', 'RANGING'):
        components.append(ScoreComponent(
            "CHOP_REGIME", -20.0, f"regime={trend_label}", "penalty"
        ))
    
    # 3. Late session or lunch hour
    if timestamp is not None:
        try:
            t = timestamp.time() if hasattr(timestamp, 'time') else time(12, 0)
            
            # Lunch hour: 11:30 - 13:00 CST — worst MES window for trend entries
            # FEB 2026 FIX: Increased from -5 to -10. This is the #1 chop zone.
            if time(11, 30) <= t < time(13, 0):
                components.append(ScoreComponent(
                    "LUNCH_HOUR", -10.0, f"time={t.strftime('%H:%M')}", "penalty"
                ))
            # Late session: after 14:30 CST
            elif t >= time(14, 30):
                components.append(ScoreComponent(
                    "LATE_SESSION", -5.0, f"time={t.strftime('%H:%M')}", "penalty"
                ))
        except Exception as e:
            logger.debug(f"Could not check session time: {e}")
    
    # 4. Large opposing wick
    direction = "LONG" if close > open_price else "SHORT"
    candle_body = abs(close - open_price)
    candle_range = high - low
    
    if candle_range > 0 and candle_body > 0:
        if direction == "LONG":
            # Check upper wick on bullish candle
            upper_wick = high - max(close, open_price)
            upper_wick_ratio = upper_wick / candle_body
            
            if upper_wick_ratio > 1.5:  # Upper wick > 1.5x body
                components.append(ScoreComponent(
                    "LARGE_UPPER_WICK", -5.0, f"ratio={upper_wick_ratio:.1f}", "penalty"
                ))
        else:  # SHORT
            # Check lower wick on bearish candle
            lower_wick = min(close, open_price) - low
            lower_wick_ratio = lower_wick / candle_body
            
            if lower_wick_ratio > 1.5:  # Lower wick > 1.5x body
                components.append(ScoreComponent(
                    "LARGE_LOWER_WICK", -5.0, f"ratio={lower_wick_ratio:.1f}", "penalty"
                ))
    
    return components


def calculate_level_trap_score(
    data: Dict[str, Any],
    recent_bars: Optional[pd.DataFrame] = None,
    key_levels: Optional[Dict[str, Optional[float]]] = None,
) -> List[ScoreComponent]:
    """Detect false-breakout / level-trap patterns near key S/R levels.

    FEB 20 2026: New penalty component.  A "level trap" occurs when price
    wicks through a significant support/resistance level but the candle
    *closes back inside* — a classic sign of a fake-out that catches
    breakout traders.  In elevated-VIX, geopolitically-tense sessions
    (like today) these traps are especially common.

    Detection logic (per level):
        1. The candle high/low breached the level.
        2. The candle close is back on the "wrong" side of the level
           (i.e. the breakout failed).
        3. Penalty scales with how far the wick extended beyond the level
           relative to the candle range (deeper wick = stronger trap signal).

    Also checks recent bars (last 3) for *multi-bar* false breakouts:
    a prior bar closed beyond the level but the current bar reversed back.

    Args:
        data: Current bar OHLC + indicators.
        recent_bars: Last 5-10 bars for multi-bar trap detection.
        key_levels: Dict of level_name → price, e.g.
            {"PDH": 6922.0, "PDL": 6860.0, "R1": 6900.0, ...}.
            When ``None``, the function is a no-op.

    Returns:
        List of ScoreComponent penalties (all negative or empty).
    """
    components: List[ScoreComponent] = []

    if not key_levels:
        return components

    close = float(data.get("close", 0))
    high = float(data.get("high", close))
    low = float(data.get("low", close))
    candle_range = high - low
    if candle_range <= 0:
        return components

    for level_name, level_price in key_levels.items():
        if level_price is None:
            continue
        level_price = float(level_price)

        # ── Resistance trap (wick above, close below) ────────────
        if high > level_price and close < level_price:
            wick_beyond = high - level_price
            wick_ratio = wick_beyond / candle_range
            # Scale penalty: -8 base, up to -15 for deep wicks
            penalty = -8.0 - min(7.0, wick_ratio * 10.0)
            components.append(ScoreComponent(
                "LEVEL_TRAP_RESIST",
                round(penalty, 1),
                f"wicked above {level_name}={level_price:.1f} (wick_ratio={wick_ratio:.2f})",
                "penalty",
            ))

        # ── Support trap (wick below, close above) ───────────────
        if low < level_price and close > level_price:
            wick_beyond = level_price - low
            wick_ratio = wick_beyond / candle_range
            penalty = -8.0 - min(7.0, wick_ratio * 10.0)
            components.append(ScoreComponent(
                "LEVEL_TRAP_SUPPORT",
                round(penalty, 1),
                f"wicked below {level_name}={level_price:.1f} (wick_ratio={wick_ratio:.2f})",
                "penalty",
            ))

    # ── Multi-bar false breakout (recent bar closed beyond, current reversed) ──
    if recent_bars is not None and len(recent_bars) >= 2:
        try:
            for level_name, level_price in key_levels.items():
                if level_price is None:
                    continue
                level_price = float(level_price)
                prev_close = float(recent_bars["close"].iloc[-2])

                # Prior bar closed above resistance, current bar fell back below
                if prev_close > level_price and close < level_price:
                    components.append(ScoreComponent(
                        "MULTI_BAR_TRAP_RESIST",
                        -10.0,
                        f"prev closed above {level_name}={level_price:.1f}, now reversed",
                        "penalty",
                    ))

                # Prior bar closed below support, current bar bounced back above
                if prev_close < level_price and close > level_price:
                    components.append(ScoreComponent(
                        "MULTI_BAR_TRAP_SUPPORT",
                        -10.0,
                        f"prev closed below {level_name}={level_price:.1f}, now reversed",
                        "penalty",
                    ))
        except Exception as e:
            logger.debug(f"Multi-bar trap detection skipped: {e}")

    return components


def calculate_signal_score(
    data: Dict[str, Any],
    prev_data: Optional[Dict[str, Any]] = None,
    recent_bars: Optional[pd.DataFrame] = None,
    timestamp: Optional[datetime] = None,
    atr_percentile: Optional[float] = None,
    key_levels: Optional[Dict[str, Optional[float]]] = None,
) -> SignalScore:
    """Calculate comprehensive signal score from all components.
    
    This is the main entry point for the scoring system. It aggregates
    scores from all categories and returns a complete SignalScore object.
    
    Args:
        data: Current bar data with all indicators
        prev_data: Previous bar data for slope calculations
        recent_bars: Recent 10-20 bars for momentum/pullback analysis
        timestamp: Current timestamp for session checks
        atr_percentile: Current ATR percentile (0-1)
        key_levels: Optional dict of S/R level_name → price for false-breakout
            trap detection (e.g. {"PDH": 6922.0, "S1": 6860.0}).
        
    Returns:
        SignalScore object with total score and component breakdown
    """
    score = SignalScore(total_score=0.0)
    
    # Determine direction from EMA alignment
    close = float(data.get('close', 0))
    ema_9 = float(data.get('EMA_9', close))
    ema_21 = float(data.get('EMA_21', close))
    score.direction = "LONG" if ema_9 > ema_21 else "SHORT"
    
    # FEB 6 2026 FIX: HTF override must happen BEFORE component scoring.
    # Previously this ran AFTER components were scored, causing 20.7% of trades
    # to execute in the opposite direction of their scored components
    # (e.g., LONG trade with EMA_STACK_DOWN components).
    # 
    # When 1m and 15m disagree, skip the trade instead of forcing a direction.
    # The old override created incoherent signals: SHORT-quality components
    # used to justify LONG trades. Better to simply not trade when timeframes
    # conflict — this is a natural filter, not a lost opportunity.
    trend_label_htf = data.get('trend_label_htf', 'UNKNOWN')
    if trend_label_htf in ("UPTREND", "LONG") and score.direction == "SHORT":
        # 1m says SHORT but 15m says UPTREND → conflicting timeframes, skip
        score.metadata['htf_conflict'] = True
        score.metadata['htf_direction'] = trend_label_htf
        score.metadata['1m_direction'] = score.direction
        # Apply a heavy penalty instead of overriding direction.
        # This will likely push the score below threshold, filtering it out.
        score.total_score -= 30.0
        score.components.append(ScoreComponent(
            "HTF_CONFLICT", -30.0, f"1m={score.direction} vs 15m={trend_label_htf}", "penalty"
        ))
    elif trend_label_htf in ("DOWNTREND", "SHORT") and score.direction == "LONG":
        # 1m says LONG but 15m says DOWNTREND → conflicting timeframes, skip
        score.metadata['htf_conflict'] = True
        score.metadata['htf_direction'] = trend_label_htf
        score.metadata['1m_direction'] = score.direction
        score.total_score -= 30.0
        score.components.append(ScoreComponent(
            "HTF_CONFLICT", -30.0, f"1m={score.direction} vs 15m={trend_label_htf}", "penalty"
        ))
    elif trend_label_htf in ("UPTREND", "LONG") and score.direction == "LONG":
        # Both timeframes agree on LONG — bonus
        score.metadata['htf_aligned'] = True
    elif trend_label_htf in ("DOWNTREND", "SHORT") and score.direction == "SHORT":
        # Both timeframes agree on SHORT — bonus
        score.metadata['htf_aligned'] = True
    
    # Calculate ADX rising for regime score
    if prev_data is not None:
        prev_adx = float(prev_data.get('ADX_14', 0))
        curr_adx = float(data.get('ADX_14', 0))
        data['adx_rising'] = curr_adx > prev_adx
    
    # Aggregate all components
    trend_components = calculate_trend_structure_score(data, prev_data)
    momentum_components = calculate_momentum_score(data, recent_bars)
    regime_components = calculate_regime_score(data, atr_percentile)
    entry_components = calculate_entry_quality_score(data, recent_bars)
    penalty_components = calculate_penalty_score(data, timestamp)
    level_trap_components = calculate_level_trap_score(data, recent_bars, key_levels)
    
    # Add all components to score
    for component in (trend_components + momentum_components + 
                     regime_components + entry_components + penalty_components +
                     level_trap_components):
        score.components.append(component)
        score.total_score += component.value
    
    # Build reasons list (top contributors)
    positive_components = sorted(
        [c for c in score.components if c.value > 0],
        key=lambda x: x.value,
        reverse=True
    )[:3]
    negative_components = sorted(
        [c for c in score.components if c.value < 0],
        key=lambda x: x.value
    )[:2]
    
    score.reasons = [c.name for c in positive_components + negative_components]
    
    # Store breakdown in metadata
    score.metadata = score.get_breakdown()
    score.metadata['direction'] = score.direction
    score.metadata['timestamp'] = timestamp
    
    return score


def should_enter_trade(
    score: SignalScore,
    min_full_size_score: float = 60.0,
    min_half_size_score: float = 45.0
) -> Tuple[PositionSize, str]:
    """Determine if trade should be entered and at what size.
    
    Position sizing thresholds:
    - Score >= 60: Full position (1.0x)
    - Score >= 45: Half position (0.5x)
    - Score < 45: No trade
    
    Args:
        score: SignalScore object from calculate_signal_score()
        min_full_size_score: Minimum score for full position (default 60)
        min_half_size_score: Minimum score for half position (default 45)
        
    Returns:
        Tuple of (PositionSize, reason_string)
    """
    if score.total_score >= min_full_size_score:
        score.position_size = PositionSize.FULL
        return (
            PositionSize.FULL,
            f"FULL_SIZE: score={score.total_score:.1f} (>={min_full_size_score})"
        )
    elif score.total_score >= min_half_size_score:
        score.position_size = PositionSize.HALF
        return (
            PositionSize.HALF,
            f"HALF_SIZE: score={score.total_score:.1f} (>={min_half_size_score})"
        )
    else:
        score.position_size = PositionSize.NONE
        return (
            PositionSize.NONE,
            f"NO_TRADE: score={score.total_score:.1f} (<{min_half_size_score})"
        )


def check_risk_gates(
    data: Dict[str, Any],
    timestamp: Optional[datetime] = None,
    current_position_pnl: float = 0.0,
    daily_pnl: float = 0.0,
    open_risk: float = 0.0,
    max_loss_per_trade: float = 100.0,
    max_daily_loss: float = 500.0,
    max_open_risk: float = 300.0,
    session_end_time: Optional[time] = None
) -> Tuple[bool, str]:
    """Check hard risk containment gates.
    
    These gates are NON-NEGOTIABLE and override any signal score.
    Even a perfect score (100+) must respect these limits.
    
    Args:
        data: Current bar data
        timestamp: Current timestamp
        current_position_pnl: Current open position P&L
        daily_pnl: Today's realized P&L
        open_risk: Current open risk (unrealized)
        max_loss_per_trade: Maximum allowed loss per trade
        max_daily_loss: Maximum allowed daily loss
        max_open_risk: Maximum allowed open risk
        session_end_time: Session cutoff time (no new trades after)
        
    Returns:
        Tuple of (is_allowed, reason)
    """
    # 1. Max loss per trade
    if abs(current_position_pnl) >= max_loss_per_trade:
        return False, f"MAX_TRADE_LOSS: ${current_position_pnl:.2f}"
    
    # 2. Max daily loss
    if daily_pnl <= -max_daily_loss:
        return False, f"MAX_DAILY_LOSS: ${daily_pnl:.2f}"
    
    # 3. Max open risk
    if open_risk >= max_open_risk:
        return False, f"MAX_OPEN_RISK: ${open_risk:.2f}"
    
    # 4. Session cutoff
    if timestamp is not None and session_end_time is not None:
        try:
            t = timestamp.time() if hasattr(timestamp, 'time') else None
            if t is not None and t >= session_end_time:
                return False, f"SESSION_CUTOFF: {t.strftime('%H:%M')} >= {session_end_time.strftime('%H:%M')}"
        except Exception as e:
            logger.warning(f"Could not check session cutoff: {e}")
    
    return True, "RISK_GATES_PASSED"
