"""SELL exhaustion entry module — shorts on confirmed exhaustion.

Only enters when ``MarketState.is_exhaustion == True`` near PDH/VWAP
extremes with rejection candle confirmation.

Inputs:
    market_state, data, recent_bars.

Outputs:
    EntrySignal: ``SELL`` on exhaustion, or ``HOLD``.

Side-effects:
    None — pure analysis, no I/O.
"""
from __future__ import annotations

from datetime import datetime, time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..market_state import MarketPhase, MarketStateResult, TrendDirection
from .session_time import SessionWindow
from .signals import EntrySignal, PullbackAnalysis


class SellExhaustionModule:
    """SELL exhaustion module - shorts only on confirmed exhaustion.
    
    ENTRY CRITERIA:
    - EXHAUSTION = TRUE (RSI falling from high + MACD slope negative)
    - Near PDH or VWAP deviation extreme
    - EMA_STACK_UP = false OR MACD slope < 0
    - NOT in STRONG_TREND_UP
    - Rejection candle or failed high confirmation
    
    EXIT CRITERIA:
    - Target: PDL / VWAP / measured move
    - Stop: Above rejection high or recent swing high
    
    WHY THIS WORKS:
    - Only shorting exhaustion avoids fighting momentum
    - Exhaustion = momentum loss, not just "high RSI"
    - Rejection confirmation reduces false signals
    - Hard blocks prevent shorting acceptance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize SELL exhaustion module."""
        config = config or {}
        
        # Exhaustion thresholds
        self.rsi_exhaustion_min = config.get("rsi_exhaustion_min", 65)
        self.macd_slope_threshold = config.get("macd_slope_threshold", 0.0)
        
        # Location requirements
        self.require_near_level = config.get("require_near_level", True)
        self.vwap_extreme_atr = config.get("vwap_extreme_atr", 2.0)
        
        # Confirmation requirements
        self.require_rejection_candle = config.get("require_rejection_candle", True)
        self.rejection_wick_ratio = config.get("rejection_wick_ratio", 0.5)
        
        # Trends to avoid shorting
        self.avoid_strong_trend = config.get("avoid_strong_trend", True)
        
        # Risk management
        self.stop_atr_mult = config.get("stop_atr_mult", 1.5)
        self.target_risk_mult = config.get("target_risk_mult", 2.0)
        self.min_stop_points = config.get("min_stop_points", 3.25)
        
        logger.info(
            f"SellExhaustionModule initialized: "
            f"RSI_exhaust>={self.rsi_exhaustion_min}, "
            f"require_rejection={self.require_rejection_candle}"
        )
    
    def evaluate(
        self,
        market_state: MarketStateResult,
        data: Dict[str, Any],
        recent_bars: Optional[pd.DataFrame] = None
    ) -> EntrySignal:
        """Evaluate for SELL exhaustion entry.
        
        Args:
            market_state: Current market state from detector
            data: Current bar data with indicators
            recent_bars: Recent price history for pattern detection
            
        Returns:
            EntrySignal with SELL, HOLD, or blocked signal
        """
        # === CHECK IF SHORT IS ALLOWED ===
        if not market_state.allow_short:
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"SHORT_BLOCKED: {'; '.join(market_state.block_reasons) if market_state.block_reasons else market_state.phase.value}",
                entry_type="BLOCKED"
            )
        
        # === REQUIRE EXHAUSTION ===
        if not market_state.is_exhaustion:
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason="NO_EXHAUSTION: Wait for momentum loss signals",
                entry_type="WAIT"
            )
        
        # === CHECK PHASE ===
        allowed_phases = [MarketPhase.EXHAUSTION_UP, MarketPhase.CHOP]
        if market_state.phase not in allowed_phases:
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"WRONG_PHASE: {market_state.phase.value}, need exhaustion",
                entry_type="WAIT"
            )
        
        # === AVOID STRONG UPTREND ===
        if self.avoid_strong_trend:
            if market_state.trend == TrendDirection.STRONG_TREND_UP:
                return EntrySignal(
                    action="HOLD",
                    confidence=0.0,
                    reason="STRONG_TREND_UP: Don't short strong momentum",
                    entry_type="BLOCKED"
                )
        
        # === EXTRACT PRICE DATA ===
        price = float(data.get("close", data.get("price", 0)))
        ema9 = float(data.get("ema_9", data.get("EMA_9", price)))
        ema21 = float(data.get("ema_21", data.get("EMA_21", price)))
        vwap = float(data.get("vwap", data.get("SESSION_VWAP", price)))
        atr = float(data.get("atr", data.get("ATR_14", 1.0)))
        pdh = float(data.get("pdh", data.get("PDH", 0)))
        pdl = float(data.get("pdl", data.get("PDL", 0)))
        
        open_price = float(data.get("open", price))
        high = float(data.get("high", price))
        low = float(data.get("low", price))
        
        # === CHECK LOCATION ===
        location_score = 0.0
        location_reasons = []
        
        if self.require_near_level:
            # Near PDH (resistance)
            if market_state.near_pdh:
                location_score += 40
                location_reasons.append("NEAR_PDH")
            
            # VWAP extreme deviation
            if market_state.vwap_deviation_extreme:
                location_score += 35
                location_reasons.append("VWAP_EXTREME")
            
            # Above VWAP by significant amount
            if vwap > 0:
                vwap_dist_atr = (price - vwap) / atr if atr > 0 else 0
                if vwap_dist_atr > 1.5:
                    location_score += 25
                    location_reasons.append("ABOVE_VWAP_1.5ATR")
            
            if location_score < 25:
                return EntrySignal(
                    action="HOLD",
                    confidence=0.0,
                    reason=f"POOR_LOCATION: score={location_score:.0f}, not near resistance",
                    entry_type="WAIT"
                )
        
        # === CHECK FOR REJECTION CANDLE ===
        rejection_score = 0.0
        rejection_reasons = []
        
        if self.require_rejection_candle:
            candle_range = high - low if high > low else 0.01
            upper_wick = high - max(open_price, price)
            body = abs(price - open_price)
            
            # Rejection: long upper wick, closes near low
            if upper_wick / candle_range > self.rejection_wick_ratio:
                rejection_score += 35
                rejection_reasons.append("UPPER_WICK_REJECTION")
            
            # Bearish close
            if price < open_price:
                rejection_score += 20
                rejection_reasons.append("BEARISH_CLOSE")
                
                # Strong bearish body
                if body / candle_range > 0.5:
                    rejection_score += 15
                    rejection_reasons.append("STRONG_BODY")
            
            # Failed at PDH
            if pdh > 0 and high >= pdh * 0.999 and price < pdh:
                rejection_score += 25
                rejection_reasons.append("PDH_REJECTION")
            
            if rejection_score < 30:
                return EntrySignal(
                    action="HOLD",
                    confidence=0.0,
                    reason=f"NO_REJECTION: score={rejection_score:.0f}, need confirmation",
                    entry_type="WAIT"
                )
        
        # === CHECK EMA/MACD BREAKDOWN ===
        breakdown_score = 0.0
        
        # EMA stack broken or breaking
        if not market_state.ema_stack_up:
            breakdown_score += 25
        
        # MACD slope negative
        if market_state.macd_slope_down:
            breakdown_score += 30
        
        # Price below EMA9 (first sign of weakness)
        if price < ema9:
            breakdown_score += 20
        
        # === CALCULATE STOPS AND TARGETS ===
        # Stop above rejection high or recent swing high
        stop_candidates = [high + atr * 0.5]  # Above current bar
        
        if recent_bars is not None and len(recent_bars) >= 3:
            swing_high = recent_bars['high'].tail(5).max()
            stop_candidates.append(swing_high + atr * 0.3)
        
        if pdh > 0 and high >= pdh * 0.99:
            stop_candidates.append(pdh + atr * 0.5)
        
        stop_loss = min(stop_candidates)  # Tightest reasonable stop
        
        # Enforce minimum stop distance
        min_stop_dist = max(self.min_stop_points, atr * 0.5)
        if stop_loss - price < min_stop_dist:
            stop_loss = price + min_stop_dist
        
        # Target based on R:R
        risk = stop_loss - price
        reward = risk * self.target_risk_mult
        take_profit = price - reward
        
        # Adjust target to VWAP or PDL if nearby
        if vwap > 0 and vwap < price and vwap > take_profit:
            take_profit = vwap + atr * 0.25  # Just above VWAP
        if pdl > 0 and pdl < price and pdl > take_profit:
            take_profit = pdl + atr * 0.25  # Just above PDL
        
        # === CALCULATE CONFIDENCE ===
        base_confidence = 0.55  # Lower base for shorts (harder trade)
        
        # Adjust based on exhaustion score
        confidence_adj = market_state.exhaustion_score / 250  # 0 to 0.4
        
        # Adjust based on location and rejection
        confidence_adj += (location_score + rejection_score) / 200  # 0 to 0.5
        
        # Adjust based on breakdown
        confidence_adj += breakdown_score / 150  # 0 to 0.5
        
        # Penalty if EMA stack still up (risky short)
        if market_state.ema_stack_up:
            confidence_adj -= 0.15
        
        final_confidence = max(0.50, min(0.90, base_confidence + confidence_adj))
        
        return EntrySignal(
            action="SELL",
            confidence=final_confidence,
            reason=f"SELL_EXHAUSTION: {', '.join(location_reasons + rejection_reasons)}",
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_type="EXHAUSTION",
            metadata={
                "location_score": location_score,
                "rejection_score": rejection_score,
                "breakdown_score": breakdown_score,
                "exhaustion_score": market_state.exhaustion_score,
                "rsi": market_state.rsi_value,
                "trend_score": market_state.trend_score,
                "phase": market_state.phase.value
            }
        )


# =============================================================================
# EVENING CONTINUATION MODULE - LATE RTH HIGH-MOMENTUM CONTINUATION
# =============================================================================

