"""Trade Entry Modules - BUY Continuation and SELL Exhaustion.

This module provides separate, symmetric entry logic for:
1. BUY CONTINUATION: Buy pullbacks in bullish acceptance (PRIORITY)
2. SELL EXHAUSTION: Sell only when exhaustion is confirmed

PRINCIPLE:
- BUY continuation has PRIORITY over any SELL logic
- SELL exhaustion allowed ONLY if MarketState.is_exhaustion == TRUE
- No signal flipping on the same candle

MORNING RTH OPTIMIZATION (9:30-11:00 CST):
- This is the highest-probability window for trend continuation
- Stricter acceptance requirements but higher confidence signals
- Block all reversal logic before 10:15 CST if trend is bullish

Author: Trading System Refactor - Jan 27, 2026
"""
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, Optional, Tuple, Any, List
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger

from .market_state import (
    MarketStateDetector,
    MarketStateResult,
    MarketPhase,
    TrendDirection,
    create_market_state_detector
)


class SessionWindow(Enum):
    """Trading session windows for different strategies."""
    PRE_MARKET = "PRE_MARKET"           # Before 9:30 CST
    MORNING_OPEN = "MORNING_OPEN"       # 9:30-10:00 CST (volatile, careful)
    MORNING_PRIME = "MORNING_PRIME"     # 9:30-10:45 CST (best continuation for BUY)
    MIDDAY = "MIDDAY"                   # 10:45-14:00 CST (chop, range/reversion only)
    AFTERNOON = "AFTERNOON"             # 14:00-15:00 CST (possible trend)
    CLOSE = "CLOSE"                     # 15:00-16:00 CST (position squaring)
    OVERNIGHT = "OVERNIGHT"             # After 16:00 CST


class SessionTimeManager:
    """
    Centralized session time logic for all entry modules.
    
    Session Windows (CST):
    - PRE_MARKET: Before 09:30
    - MORNING_OPEN: 09:30-10:00 (volatile open, careful)
    - MORNING_PRIME: 09:30-10:45 (best for BUY_CONTINUATION)
    - MIDDAY: 10:45-14:00 (chop zone, RANGE/MEAN_REVERSION only)
    - AFTERNOON: 14:00-15:00 (possible trend resumption)
    - CLOSE: 15:00-16:00 (position squaring)
    - OVERNIGHT: After 16:00
    
    Key cutoffs:
    - buy_continuation_cutoff: 10:45 CST (after this, disable BUY_CONTINUATION)
    - reversal_block_until: 10:15 CST (no reversal trades before this)
    """
    
    # Session boundaries (CST)
    PRE_MARKET_END = time(9, 30)
    MORNING_OPEN_END = time(10, 0)
    
    # MORNING_PRIME: Best window for BUY_CONTINUATION
    # Changed from 09:30-11:30 to 09:30-10:45
    MORNING_PRIME_START = time(9, 30)
    MORNING_PRIME_END = time(10, 45)
    BUY_CONTINUATION_CUTOFF = time(10, 45)  # After this, disable BUY_CONTINUATION
    
    # MIDDAY: Chop zone, only RANGE/MEAN_REVERSION allowed
    MIDDAY_START = time(10, 45)
    MIDDAY_END = time(14, 0)
    
    # Afternoon/Close
    AFTERNOON_START = time(14, 0)
    AFTERNOON_END = time(15, 0)
    CLOSE_START = time(15, 0)
    CLOSE_END = time(16, 0)
    
    # Special cutoffs
    REVERSAL_BLOCK_UNTIL = time(10, 15)  # No reversal trades before this if trend is bullish
    
    @classmethod
    def get_session_window(cls, timestamp: Optional[datetime]) -> SessionWindow:
        """
        Determine current session window from timestamp.
        
        Args:
            timestamp: Current datetime (should be in CST or will be treated as local)
            
        Returns:
            SessionWindow enum value
        """
        if timestamp is None:
            return SessionWindow.MORNING_PRIME  # Default to prime time
        
        try:
            t = timestamp.time() if hasattr(timestamp, 'time') else time(10, 0)
        except Exception:
            return SessionWindow.MORNING_PRIME
        
        if t < cls.PRE_MARKET_END:
            return SessionWindow.PRE_MARKET
        elif t < cls.MORNING_PRIME_END:
            # 09:30-10:45 is MORNING_PRIME (includes MORNING_OPEN for continuity)
            return SessionWindow.MORNING_PRIME
        elif t < cls.MIDDAY_END:
            return SessionWindow.MIDDAY
        elif t < cls.AFTERNOON_END:
            return SessionWindow.AFTERNOON
        elif t < cls.CLOSE_END:
            return SessionWindow.CLOSE
        else:
            return SessionWindow.OVERNIGHT
    
    @classmethod
    def is_buy_continuation_allowed(cls, timestamp: Optional[datetime]) -> Tuple[bool, str]:
        """
        Check if BUY_CONTINUATION is allowed based on time.
        
        BUY_CONTINUATION is only allowed during MORNING_PRIME (09:30-10:45 CST).
        After 10:45 CST, only RANGE/MEAN_REVERSION modules are active.
        
        Args:
            timestamp: Current datetime
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        if timestamp is None:
            return True, "no_timestamp"
        
        try:
            t = timestamp.time() if hasattr(timestamp, 'time') else time(10, 0)
        except Exception:
            return True, "time_parse_error"
        
        if t < cls.MORNING_PRIME_START:
            return False, f"PRE_MARKET: {t.strftime('%H:%M')} < 09:30 CST"
        elif t <= cls.BUY_CONTINUATION_CUTOFF:
            return True, f"MORNING_PRIME: {t.strftime('%H:%M')} within 09:30-10:45 CST"
        else:
            return False, f"BUY_CUTOFF_REACHED: {t.strftime('%H:%M')} > 10:45 CST (RANGE/REVERSION only)"

    @classmethod
    def is_short_continuation_allowed(cls, timestamp: Optional[datetime]) -> Tuple[bool, str]:
        """
        Check if SHORT_CONTINUATION is allowed based on time.

        Mirror of BUY_CONTINUATION timing (MORNING_PRIME only).
        """
        if timestamp is None:
            return True, "no_timestamp"

        try:
            t = timestamp.time() if hasattr(timestamp, 'time') else time(10, 0)
        except Exception:
            return True, "time_parse_error"

        if t < cls.MORNING_PRIME_START:
            return False, f"PRE_MARKET: {t.strftime('%H:%M')} < 09:30 CST"
        elif t <= cls.BUY_CONTINUATION_CUTOFF:
            return True, f"MORNING_PRIME: {t.strftime('%H:%M')} within 09:30-10:45 CST"
        else:
            return False, f"SHORT_CUTOFF_REACHED: {t.strftime('%H:%M')} > 10:45 CST (RANGE/REVERSION only)"
    
    @classmethod
    def is_range_reversion_allowed(cls, timestamp: Optional[datetime]) -> Tuple[bool, str]:
        """
        Check if RANGE/MEAN_REVERSION is allowed based on time.
        
        RANGE_REVERSION is allowed during:
        - MIDDAY: 10:45-14:00 CST (primary window)
        - Can also be active 09:45-14:30 per RangeReversionModule's own logic
        
        Args:
            timestamp: Current datetime
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        if timestamp is None:
            return True, "no_timestamp"
        
        try:
            t = timestamp.time() if hasattr(timestamp, 'time') else time(12, 0)
        except Exception:
            return True, "time_parse_error"
        
        # Range reversion has its own timing logic in RangeReversionModule
        # This is a softer check - primarily active in MIDDAY
        if cls.MIDDAY_START <= t <= cls.MIDDAY_END:
            return True, f"MIDDAY_RANGE_WINDOW: {t.strftime('%H:%M')} within 10:45-14:00 CST"
        elif time(9, 45) <= t <= time(14, 30):
            return True, f"EXTENDED_RANGE_WINDOW: {t.strftime('%H:%M')} within 09:45-14:30 CST"
        else:
            return False, f"OUTSIDE_RANGE_WINDOW: {t.strftime('%H:%M')}"
    
    @classmethod
    def is_reversal_blocked(cls, timestamp: Optional[datetime]) -> Tuple[bool, str]:
        """
        Check if reversal trades should be blocked based on time.
        
        Block all reversal logic before 10:15 CST.
        
        Args:
            timestamp: Current datetime
            
        Returns:
            Tuple of (is_blocked, reason)
        """
        if timestamp is None:
            return False, "no_timestamp"
        
        try:
            t = timestamp.time() if hasattr(timestamp, 'time') else time(10, 30)
        except Exception:
            return False, "time_parse_error"
        
        if t < cls.REVERSAL_BLOCK_UNTIL:
            return True, f"REVERSAL_BLOCKED: {t.strftime('%H:%M')} < 10:15 CST"
        else:
            return False, f"REVERSAL_ALLOWED: {t.strftime('%H:%M')} >= 10:15 CST"


@dataclass
class PullbackAnalysis:
    """Detailed pullback analysis result."""
    is_valid: bool = False
    score: float = 0.0
    depth_pct: float = 0.0
    touch_level: str = ""  # "EMA9", "EMA21", "VWAP", "NONE"
    confirmation: str = ""  # "BULLISH_ENGULF", "STRONG_CLOSE", "HIGHER_LOW", "NONE"
    reasons: List[str] = field(default_factory=list)


@dataclass
class EntrySignal:
    """Signal from entry module."""
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0
    reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_type: str = ""  # "CONTINUATION", "EXHAUSTION", "PULLBACK", etc.
    session_window: str = ""  # Current session window
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def is_actionable(self) -> bool:
        return self.action in ("BUY", "SELL") and self.confidence > 0.5


class BuyContinuationModule:
    """BUY continuation module - buys pullbacks in acceptance.
    
    OPTIMIZED FOR MORNING RTH (9:30-11:00 CST)
    
    ACCEPTANCE CONDITIONS (ALL must be true for entry):
    - EMA stack up (EMA9 > EMA21 > EMA50 or EMA9 > EMA21)
    - MACD histogram positive AND slope increasing (or > +0.20)
    - RSI in bullish continuation zone (55-70), not falling
    - Price above VWAP or reclaiming VWAP after shallow pullback
    - ADX > 25 (trend strength confirmed) - optional but boosts confidence
    
    ENTRY REQUIREMENTS:
    - MarketState.is_acceptance == TRUE
    - MarketState.is_exhaustion == FALSE
    - Pullback into EMA9 / EMA21 / VWAP (no deep retracement)
    - Bullish confirmation candle (engulfing / strong close)
    
    HARD BLOCKS:
    - NEVER short during ACCEPTANCE (handled by MarketStateDetector)
    - Block BUY if RSI > 72 (overextension risk)
    - Block BUY if price > max_vwap_distance without pullback
    - Block all reversal logic before 10:15 CST if trend is bullish
    
    STOPS & TARGETS:
    - Stop: Below pullback low OR EMA21 (whichever gives better R:R)
    - Max stop size enforced (default 8 ticks / 2 points on MES)
    - Target: PDH, range high, or measured move (minimum 2:1 R:R)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize BUY continuation module with morning RTH optimization."""
        config = config or {}
        
        # === RSI THRESHOLDS ===
        self.rsi_min = config.get("rsi_min", 55)           # Floor for continuation zone
        self.rsi_max = config.get("rsi_max", 70)           # Ceiling for continuation zone
        self.rsi_overextension = config.get("rsi_overextension", 72)  # Hard block above this
        self.rsi_ideal_min = config.get("rsi_ideal_min", 58)  # Sweet spot floor
        self.rsi_ideal_max = config.get("rsi_ideal_max", 65)  # Sweet spot ceiling
        
        # === MACD THRESHOLDS ===
        self.macd_acceptance_threshold = config.get("macd_acceptance_threshold", 0.20)
        self.macd_slope_min = config.get("macd_slope_min", 0.0)  # Must be flat or rising
        
        # === ADX THRESHOLD ===
        self.adx_trend_threshold = config.get("adx_trend_threshold", 25)
        self.adx_strong_threshold = config.get("adx_strong_threshold", 30)
        
        # === PULLBACK PARAMETERS ===
        self.pullback_depth_min_pct = config.get("pullback_depth_min_pct", 0.05)  # Min 0.05% pullback
        self.pullback_depth_max_pct = config.get("pullback_depth_max_pct", 0.50)  # Max 0.50% pullback
        self.max_vwap_distance_atr = config.get("max_vwap_distance_atr", 2.0)     # Max ATR from VWAP without pullback
        self.ema_proximity_pct = config.get("ema_proximity_pct", 0.15)            # % distance to count as "near"
        
        # === CONFIRMATION REQUIREMENTS ===
        self.require_bullish_close = config.get("require_bullish_close", True)
        self.require_higher_low = config.get("require_higher_low", False)
        self.min_candle_body_ratio = config.get("min_candle_body_ratio", 0.4)     # Body/range ratio for confirmation
        self.engulfing_required = config.get("engulfing_required", False)
        
        # === RISK MANAGEMENT ===
        self.stop_atr_mult = config.get("stop_atr_mult", 1.5)
        self.target_risk_mult = config.get("target_risk_mult", 2.0)      # Minimum 2:1 R:R
        self.min_stop_points = config.get("min_stop_points", 3.25)       # Min stop distance
        self.max_stop_points = config.get("max_stop_points", 8.0)        # Max stop distance (8 ticks)
        self.min_rr_ratio = config.get("min_rr_ratio", 2.0)              # Minimum R:R required

        # === HIGHER TIMEFRAME ALIGNMENT ===
        # Require 15m trend alignment to avoid buying into higher-timeframe downtrends.
        self.require_htf_alignment = config.get("require_htf_alignment", True)
        
        # === SESSION TIMING (CST) ===
        self.morning_open_start = time(9, 30)
        self.morning_open_end = time(10, 0)
        self.morning_prime_start = time(10, 0)
        self.morning_prime_end = time(11, 0)
        self.reversal_block_until = time(10, 15)  # No reversal trades before this
        self.midday_start = time(11, 0)
        self.midday_end = time(14, 0)
        
        # === MINIMUM SCORES ===
        self.min_pullback_score = config.get("min_pullback_score", 35)
        self.min_confidence = config.get("min_confidence", 0.55)
        
        # === TRACKING ===
        self._last_signal_time: Optional[datetime] = None
        self._consecutive_holds = 0
        
        logger.info(
            f"BuyContinuationModule initialized: "
            f"RSI=[{self.rsi_min}-{self.rsi_max}], "
            f"MACD_threshold={self.macd_acceptance_threshold}, "
            f"ADX_trend={self.adx_trend_threshold}, "
            f"R:R_min={self.min_rr_ratio}"
        )
    
    def _get_session_window(self, timestamp: Optional[datetime]) -> SessionWindow:
        """Determine current session window for strategy selection."""
        if timestamp is None:
            return SessionWindow.MORNING_PRIME  # Default to prime time
        
        # Convert to local time if needed
        try:
            if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                # Assume CST if not specified
                local_ts = timestamp
            else:
                local_ts = timestamp
            t = local_ts.time() if hasattr(local_ts, 'time') else time(10, 0)
        except Exception:
            t = time(10, 0)
        
        if t < self.morning_open_start:
            return SessionWindow.PRE_MARKET
        elif t < self.morning_open_end:
            return SessionWindow.MORNING_OPEN
        elif t < self.morning_prime_end:
            return SessionWindow.MORNING_PRIME
        elif t < self.midday_end:
            return SessionWindow.MIDDAY
        elif t < time(15, 0):
            return SessionWindow.AFTERNOON
        elif t < time(16, 0):
            return SessionWindow.CLOSE
        else:
            return SessionWindow.OVERNIGHT
    
    def _is_reversal_blocked(self, timestamp: Optional[datetime], market_state: MarketStateResult) -> Tuple[bool, str]:
        """Check if reversal trades should be blocked based on time and trend.
        
        Block all reversal logic before 10:15 CST if trend is bullish.
        """
        if timestamp is None:
            return False, ""
        
        try:
            t = timestamp.time() if hasattr(timestamp, 'time') else time(10, 30)
        except Exception:
            return False, ""
        
        # Before 10:15 CST with bullish trend = block reversals
        if t < self.reversal_block_until:
            if market_state.trend in [TrendDirection.STRONG_TREND_UP, TrendDirection.TREND_UP]:
                return True, f"REVERSAL_BLOCKED: Before {self.reversal_block_until} with bullish trend"
            if market_state.is_acceptance and not market_state.is_exhaustion:
                return True, f"REVERSAL_BLOCKED: Before {self.reversal_block_until} in acceptance"
        
        return False, ""
    
    def _analyze_pullback(
        self,
        price: float,
        open_price: float,
        high: float,
        low: float,
        ema9: float,
        ema21: float,
        ema50: float,
        vwap: float,
        atr: float,
        recent_bars: Optional[pd.DataFrame]
    ) -> PullbackAnalysis:
        """Analyze pullback quality and confirmation signals.
        
        Returns detailed analysis of whether current bar represents
        a valid pullback entry opportunity.
        """
        result = PullbackAnalysis()
        
        if price <= 0 or atr <= 0:
            return result
        
        # === CALCULATE DISTANCES ===
        ema9_dist_pct = abs(price - ema9) / price * 100 if ema9 > 0 else 999
        ema21_dist_pct = abs(price - ema21) / price * 100 if ema21 > 0 else 999
        vwap_dist_pct = abs(price - vwap) / price * 100 if vwap > 0 else 999
        vwap_dist_atr = abs(price - vwap) / atr if vwap > 0 else 999
        
        # === CHECK EMA9 PULLBACK ===
        # Price near EMA9 or bar touched EMA9
        ema9_touch = low <= ema9 <= high if ema9 > 0 else False
        ema9_near = ema9_dist_pct < self.ema_proximity_pct and price >= ema9
        
        if ema9_touch:
            result.score += 35
            result.touch_level = "EMA9"
            result.reasons.append("EMA9_BOUNCE")
        elif ema9_near:
            result.score += 25
            if not result.touch_level:
                result.touch_level = "EMA9"
            result.reasons.append("EMA9_SUPPORT")
        
        # === CHECK EMA21 PULLBACK ===
        ema21_touch = low <= ema21 <= high if ema21 > 0 else False
        ema21_near = ema21_dist_pct < self.ema_proximity_pct * 1.5 and price >= ema21
        
        if ema21_touch:
            result.score += 30
            if not result.touch_level:
                result.touch_level = "EMA21"
            result.reasons.append("EMA21_BOUNCE")
        elif ema21_near:
            result.score += 20
            if not result.touch_level:
                result.touch_level = "EMA21"
            result.reasons.append("EMA21_SUPPORT")
        
        # === CHECK VWAP PULLBACK ===
        vwap_touch = low <= vwap <= high if vwap > 0 else False
        vwap_near = vwap_dist_pct < self.ema_proximity_pct and price >= vwap
        vwap_reclaim = (low < vwap < price) if vwap > 0 else False  # Dipped below, closed above
        
        if vwap_reclaim:
            result.score += 35
            if not result.touch_level:
                result.touch_level = "VWAP"
            result.reasons.append("VWAP_RECLAIM")
        elif vwap_touch:
            result.score += 30
            if not result.touch_level:
                result.touch_level = "VWAP"
            result.reasons.append("VWAP_BOUNCE")
        elif vwap_near:
            result.score += 20
            if not result.touch_level:
                result.touch_level = "VWAP"
            result.reasons.append("VWAP_SUPPORT")
        
        # === CANDLE CONFIRMATION ===
        candle_range = high - low if high > low else 0.01
        candle_body = abs(price - open_price)
        body_ratio = candle_body / candle_range
        is_bullish = price > open_price
        
        # Bullish engulfing pattern
        if recent_bars is not None and len(recent_bars) >= 2:
            prev_bar = recent_bars.iloc[-2]
            prev_open = float(prev_bar.get('open', 0))
            prev_close = float(prev_bar.get('close', 0))
            prev_was_bearish = prev_close < prev_open
            
            # Current bar engulfs previous bearish bar
            if is_bullish and prev_was_bearish:
                if price > prev_open and open_price < prev_close:
                    result.score += 20
                    result.confirmation = "BULLISH_ENGULF"
                    result.reasons.append("ENGULFING")
        
        # Strong bullish close (body > 50% of range)
        if is_bullish and body_ratio > self.min_candle_body_ratio:
            if result.confirmation != "BULLISH_ENGULF":
                result.confirmation = "STRONG_CLOSE"
            result.score += 15
            result.reasons.append("BULLISH_CLOSE")
        elif is_bullish:
            result.score += 5
            result.reasons.append("WEAK_BULLISH")
        
        # === HIGHER LOW CHECK ===
        if recent_bars is not None and len(recent_bars) >= 3:
            recent_lows = recent_bars['low'].tail(3).values
            if len(recent_lows) >= 3:
                # Current low > previous low
                if low > recent_lows[-2]:
                    result.score += 10
                    if not result.confirmation:
                        result.confirmation = "HIGHER_LOW"
                    result.reasons.append("HIGHER_LOW")
                # Sequence of higher lows
                if recent_lows[-2] > recent_lows[-3] and low > recent_lows[-2]:
                    result.score += 5
                    result.reasons.append("HL_SEQUENCE")
        
        # === CALCULATE PULLBACK DEPTH ===
        if recent_bars is not None and len(recent_bars) >= 5:
            recent_high = recent_bars['high'].tail(5).max()
            if recent_high > low:
                result.depth_pct = (recent_high - low) / recent_high * 100
                
                # Ideal pullback depth (0.1% - 0.4%)
                if self.pullback_depth_min_pct <= result.depth_pct <= self.pullback_depth_max_pct:
                    result.score += 10
                    result.reasons.append(f"IDEAL_DEPTH({result.depth_pct:.2f}%)")
                elif result.depth_pct > self.pullback_depth_max_pct:
                    result.score -= 10
                    result.reasons.append(f"DEEP_PULLBACK({result.depth_pct:.2f}%)")
        
        # === DETERMINE VALIDITY ===
        result.is_valid = (
            result.score >= self.min_pullback_score and
            result.touch_level != "" and
            len(result.reasons) >= 2  # Need multiple confirmations
        )
        
        return result
    
    def _calculate_stops_and_targets(
        self,
        price: float,
        ema21: float,
        atr: float,
        pdh: float,
        recent_bars: Optional[pd.DataFrame],
        pullback_analysis: PullbackAnalysis
    ) -> Tuple[float, float, bool, str]:
        """Calculate stop loss and take profit with R:R validation.
        
        Returns:
            (stop_loss, take_profit, is_valid, reason)
        """
        # === STOP LOSS CANDIDATES ===
        stop_candidates = []
        
        # Below pullback low (from recent bars)
        if recent_bars is not None and len(recent_bars) >= 3:
            pullback_low = recent_bars['low'].tail(3).min()
            stop_candidates.append(("PULLBACK_LOW", pullback_low - atr * 0.3))
        
        # Below EMA21
        if ema21 > 0:
            stop_candidates.append(("EMA21", ema21 - atr * 0.3))
        
        # ATR-based stop
        stop_candidates.append(("ATR", price - atr * self.stop_atr_mult))
        
        # Select tightest reasonable stop
        if stop_candidates:
            # Sort by stop level (highest = tightest)
            stop_candidates.sort(key=lambda x: x[1], reverse=True)
            stop_type, stop_loss = stop_candidates[0]
        else:
            stop_loss = price - atr * self.stop_atr_mult
            stop_type = "DEFAULT"
        
        # === ENFORCE STOP DISTANCE LIMITS ===
        stop_distance = price - stop_loss
        
        # Minimum stop distance
        if stop_distance < self.min_stop_points:
            stop_loss = price - self.min_stop_points
            stop_distance = self.min_stop_points
        
        # Maximum stop distance
        if stop_distance > self.max_stop_points:
            stop_loss = price - self.max_stop_points
            stop_distance = self.max_stop_points
        
        # === TARGET CALCULATION ===
        risk = stop_distance
        min_reward = risk * self.min_rr_ratio
        
        # Target candidates
        target_candidates = []
        
        # PDH target
        if pdh > 0 and pdh > price:
            pdh_reward = pdh - price
            if pdh_reward >= min_reward:
                target_candidates.append(("PDH", pdh - atr * 0.25))  # Just below PDH
        
        # Measured move (2:1 R:R)
        measured_target = price + (risk * self.target_risk_mult)
        target_candidates.append(("MEASURED", measured_target))
        
        # Extended target (3:1 R:R) if strong setup
        if pullback_analysis.score >= 50:
            extended_target = price + (risk * 3.0)
            target_candidates.append(("EXTENDED", extended_target))
        
        # Select best target
        if target_candidates:
            # Prefer PDH if within range, otherwise measured move
            pdh_targets = [t for t in target_candidates if t[0] == "PDH"]
            if pdh_targets:
                target_type, take_profit = pdh_targets[0]
            else:
                target_type, take_profit = target_candidates[0]
        else:
            take_profit = price + min_reward
            target_type = "MIN_RR"
        
        # === VALIDATE R:R ===
        actual_reward = take_profit - price
        actual_rr = actual_reward / risk if risk > 0 else 0
        
        if actual_rr < self.min_rr_ratio:
            return stop_loss, take_profit, False, f"INSUFFICIENT_RR: {actual_rr:.2f} < {self.min_rr_ratio}"
        
        return stop_loss, take_profit, True, f"{stop_type}_STOP, {target_type}_TARGET, RR={actual_rr:.2f}"
    
    def evaluate(
        self,
        market_state: MarketStateResult,
        data: Dict[str, Any],
        recent_bars: Optional[pd.DataFrame] = None,
        timestamp: Optional[datetime] = None
    ) -> EntrySignal:
        """Evaluate for BUY continuation entry.
        
        Args:
            market_state: Current market state from detector
            data: Current bar data with indicators
            recent_bars: Recent price history for pullback detection
            timestamp: Current timestamp for session filtering
            
        Returns:
            EntrySignal with BUY, HOLD, or blocked signal
        """
        # === DETERMINE SESSION WINDOW ===
        session = self._get_session_window(timestamp)
        session_str = session.value
        
        # Initialize metadata
        metadata = {
            "session_window": session_str,
            "is_acceptance": market_state.is_acceptance,
            "is_exhaustion": market_state.is_exhaustion,
            "trend_score": market_state.trend_score,
            "continuation_score": market_state.continuation_score,
            "exhaustion_score": market_state.exhaustion_score,
            "allow_buy": market_state.allow_buy,
            "allow_short": market_state.allow_short,
            "phase": market_state.phase.value,
            "trend": market_state.trend.value,
        }
        
        # === HARD BLOCK: BUY NOT ALLOWED ===
        if not market_state.allow_buy:
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"BUY_BLOCKED: {market_state.phase.value}",
                entry_type="BLOCKED",
                session_window=session_str,
                metadata=metadata
            )
        
        # === HARD BLOCK: EXHAUSTION PRESENT ===
        # Don't buy into exhaustion - wait for resolution
        if market_state.is_exhaustion:
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason="EXHAUSTION_PRESENT: Momentum loss detected, wait for resolution",
                entry_type="CAUTION",
                session_window=session_str,
                metadata=metadata
            )
        
        # === REQUIRE ACCEPTANCE FOR CONTINUATION ===
        if not market_state.is_acceptance:
            # Allow weaker entries in trending markets (but lower confidence)
            if market_state.trend not in [TrendDirection.STRONG_TREND_UP, TrendDirection.TREND_UP, TrendDirection.WEAK_UP]:
                return EntrySignal(
                    action="HOLD",
                    confidence=0.0,
                    reason=f"NOT_ACCEPTANCE: phase={market_state.phase.value}, trend={market_state.trend.value}",
                    entry_type="WAIT",
                    session_window=session_str,
                    metadata=metadata
                )
        
        # === EXTRACT PRICE DATA ===
        price = float(data.get("close", data.get("price", 0)))
        open_price = float(data.get("open", price))
        high = float(data.get("high", price))
        low = float(data.get("low", price))
        
        ema9 = float(data.get("ema_9", data.get("EMA_9", price)))
        ema21 = float(data.get("ema_21", data.get("EMA_21", price)))
        ema50 = float(data.get("ema_50", data.get("EMA_50", ema21)))
        vwap = float(data.get("vwap", data.get("SESSION_VWAP", price)))
        
        atr = float(data.get("atr", data.get("ATR_14", 1.0)))
        adx = float(data.get("adx", data.get("ADX_14", 20)))
        macd_hist = float(data.get("macd_hist", data.get("MACD_hist", 0)))
        
        pdh = float(data.get("pdh", data.get("PDH", 0)))
        pdl = float(data.get("pdl", data.get("PDL", 0)))
        
        rsi = market_state.rsi_value
        
        metadata.update({
            "rsi": rsi,
            "adx": adx,
            "macd_hist": macd_hist,
            "price": price,
            "ema9": ema9,
            "ema21": ema21,
            "vwap": vwap,
            "atr": atr,
        })

        # === HIGHER TIMEFRAME ALIGNMENT (15m EMA trend) ===
        if self.require_htf_alignment:
            htf_ema21 = float(data.get("15m_ema21", data.get("15m_EMA_21", 0)))
            htf_ema50 = float(data.get("15m_ema50", data.get("15m_EMA_50", 0)))
            metadata["htf_ema21"] = htf_ema21
            metadata["htf_ema50"] = htf_ema50

            if htf_ema21 > 0 and htf_ema50 > 0 and htf_ema21 < htf_ema50:
                return EntrySignal(
                    action="HOLD",
                    confidence=0.0,
                    reason="HTF_DOWNTREND: 15m EMA21 < EMA50",
                    entry_type="BLOCKED",
                    session_window=session_str,
                    metadata=metadata
                )
        
        # === HARD BLOCK: RSI OVEREXTENSION ===
        if rsi > self.rsi_overextension:
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"RSI_OVEREXTENDED: {rsi:.1f} > {self.rsi_overextension} (chase risk)",
                entry_type="BLOCKED",
                session_window=session_str,
                metadata=metadata
            )
        
        # === HARD BLOCK: TOO FAR FROM VWAP WITHOUT PULLBACK ===
        if vwap > 0 and atr > 0:
            vwap_dist_atr = (price - vwap) / atr
            if vwap_dist_atr > self.max_vwap_distance_atr:
                return EntrySignal(
                    action="HOLD",
                    confidence=0.0,
                    reason=f"VWAP_DISTANCE: {vwap_dist_atr:.1f} ATR > {self.max_vwap_distance_atr} (need pullback)",
                    entry_type="WAIT",
                    session_window=session_str,
                    metadata=metadata
                )
        
        # === RSI ZONE CHECK ===
        rsi_in_zone = self.rsi_min <= rsi <= self.rsi_max
        rsi_in_ideal = self.rsi_ideal_min <= rsi <= self.rsi_ideal_max
        
        if not rsi_in_zone:
            # Allow slightly outside range with reduced confidence
            if rsi < self.rsi_min - 5:
                return EntrySignal(
                    action="HOLD",
                    confidence=0.0,
                    reason=f"RSI_TOO_LOW: {rsi:.1f} < {self.rsi_min - 5} (wait for momentum)",
                    entry_type="WAIT",
                    session_window=session_str,
                    metadata=metadata
                )
            # RSI between rsi_max and rsi_overextension - proceed with caution
        
        # === ANALYZE PULLBACK ===
        pullback = self._analyze_pullback(
            price=price,
            open_price=open_price,
            high=high,
            low=low,
            ema9=ema9,
            ema21=ema21,
            ema50=ema50,
            vwap=vwap,
            atr=atr,
            recent_bars=recent_bars
        )
        
        metadata["pullback_score"] = pullback.score
        metadata["pullback_level"] = pullback.touch_level
        metadata["pullback_confirmation"] = pullback.confirmation
        metadata["pullback_depth_pct"] = pullback.depth_pct
        metadata["pullback_reasons"] = pullback.reasons
        
        if not pullback.is_valid:
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"NO_VALID_PULLBACK: score={pullback.score:.0f}, level={pullback.touch_level or 'NONE'}",
                entry_type="WAIT",
                session_window=session_str,
                metadata=metadata
            )
        
        # === CALCULATE STOPS AND TARGETS ===
        stop_loss, take_profit, rr_valid, rr_reason = self._calculate_stops_and_targets(
            price=price,
            ema21=ema21,
            atr=atr,
            pdh=pdh,
            recent_bars=recent_bars,
            pullback_analysis=pullback
        )
        
        metadata["stop_loss"] = stop_loss
        metadata["take_profit"] = take_profit
        metadata["stop_reason"] = rr_reason
        
        if not rr_valid:
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=rr_reason,
                entry_type="BLOCKED",
                session_window=session_str,
                metadata=metadata
            )
        
        # === CALCULATE CONFIDENCE ===
        base_confidence = 0.55
        
        # Session bonus
        if session == SessionWindow.MORNING_PRIME:
            base_confidence += 0.10  # Best window for continuation
        elif session == SessionWindow.MORNING_OPEN:
            base_confidence += 0.05  # Good but volatile
        elif session == SessionWindow.MIDDAY:
            base_confidence -= 0.10  # Chop zone
        
        # Acceptance bonus
        if market_state.is_acceptance:
            base_confidence += 0.10
        
        # RSI zone bonus
        if rsi_in_ideal:
            base_confidence += 0.08
        elif rsi_in_zone:
            base_confidence += 0.04
        
        # ADX trend strength bonus
        if adx >= self.adx_strong_threshold:
            base_confidence += 0.08
        elif adx >= self.adx_trend_threshold:
            base_confidence += 0.04
        
        # MACD confirmation bonus
        if macd_hist > self.macd_acceptance_threshold:
            base_confidence += 0.06
        elif macd_hist > 0:
            base_confidence += 0.03
        
        # Pullback quality bonus
        pullback_bonus = (pullback.score - self.min_pullback_score) / 100
        base_confidence += min(0.15, pullback_bonus)
        
        # Engulfing pattern bonus
        if pullback.confirmation == "BULLISH_ENGULF":
            base_confidence += 0.05
        
        # Continuation score bonus
        if market_state.continuation_score > 60:
            base_confidence += 0.05
        
        # Cap confidence
        final_confidence = min(0.95, max(self.min_confidence, base_confidence))
        
        metadata["confidence_breakdown"] = {
            "base": 0.55,
            "session": session_str,
            "acceptance": market_state.is_acceptance,
            "rsi_in_ideal": rsi_in_ideal,
            "adx": adx,
            "macd_hist": macd_hist,
            "pullback_score": pullback.score,
            "final": final_confidence
        }
        
        # === GENERATE SIGNAL ===
        reason_parts = [
            f"BUY_CONTINUATION",
            f"{pullback.touch_level}_PULLBACK",
        ]
        if pullback.confirmation:
            reason_parts.append(pullback.confirmation)
        if session == SessionWindow.MORNING_PRIME:
            reason_parts.append("MORNING_PRIME")
        if market_state.is_acceptance:
            reason_parts.append("ACCEPTANCE")
        
        return EntrySignal(
            action="BUY",
            confidence=final_confidence,
            reason=" | ".join(reason_parts),
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_type="CONTINUATION",
            session_window=session_str,
            metadata=metadata
        )


class ShortContinuationModule:
    """SHORT continuation module - sells pullbacks in bearish acceptance.

    Mirrors BUY_CONTINUATION logic for downtrend pullbacks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # === RSI THRESHOLDS (bearish zone) ===
        self.rsi_min = config.get("rsi_min", 30)
        self.rsi_max = config.get("rsi_max", 45)
        self.rsi_overextension = config.get("rsi_overextension", 28)
        self.rsi_ideal_min = config.get("rsi_ideal_min", 35)
        self.rsi_ideal_max = config.get("rsi_ideal_max", 42)

        # === MACD THRESHOLDS (bearish) ===
        self.macd_acceptance_threshold = config.get("macd_acceptance_threshold", -0.20)
        self.macd_slope_min = config.get("macd_slope_min", 0.0)

        # === ADX THRESHOLD ===
        self.adx_trend_threshold = config.get("adx_trend_threshold", 25)
        self.adx_strong_threshold = config.get("adx_strong_threshold", 30)

        # === PULLBACK PARAMETERS ===
        self.pullback_depth_min_pct = config.get("pullback_depth_min_pct", 0.05)
        self.pullback_depth_max_pct = config.get("pullback_depth_max_pct", 0.50)
        self.max_vwap_distance_atr = config.get("max_vwap_distance_atr", 2.0)
        self.ema_proximity_pct = config.get("ema_proximity_pct", 0.15)

        # === CONFIRMATION REQUIREMENTS ===
        self.require_bearish_close = config.get("require_bearish_close", True)
        self.require_lower_high = config.get("require_lower_high", False)
        self.min_candle_body_ratio = config.get("min_candle_body_ratio", 0.4)
        self.engulfing_required = config.get("engulfing_required", False)

        # === RISK MANAGEMENT ===
        self.stop_atr_mult = config.get("stop_atr_mult", 1.5)
        self.target_risk_mult = config.get("target_risk_mult", 2.0)
        self.min_stop_points = config.get("min_stop_points", 3.25)
        self.max_stop_points = config.get("max_stop_points", 8.0)
        self.min_rr_ratio = config.get("min_rr_ratio", 2.0)

        # === HIGHER TIMEFRAME ALIGNMENT ===
        self.require_htf_alignment = config.get("require_htf_alignment", True)

        # === MINIMUM SCORES ===
        self.min_pullback_score = config.get("min_pullback_score", 35)
        self.min_confidence = config.get("min_confidence", 0.55)

        logger.info(
            f"ShortContinuationModule initialized: "
            f"RSI=[{self.rsi_min}-{self.rsi_max}], "
            f"MACD_threshold={self.macd_acceptance_threshold}, "
            f"ADX_trend={self.adx_trend_threshold}, "
            f"R:R_min={self.min_rr_ratio}"
        )

    def _get_session_window(self, timestamp: Optional[datetime]) -> SessionWindow:
        if timestamp is None:
            return SessionWindow.MORNING_PRIME

        try:
            if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                local_ts = timestamp
            else:
                local_ts = timestamp
            t = local_ts.time() if hasattr(local_ts, 'time') else time(10, 0)
        except Exception:
            t = time(10, 0)

        if t < time(9, 30):
            return SessionWindow.PRE_MARKET
        elif t < time(10, 0):
            return SessionWindow.MORNING_OPEN
        elif t < time(11, 0):
            return SessionWindow.MORNING_PRIME
        elif t < time(14, 0):
            return SessionWindow.MIDDAY
        elif t < time(15, 0):
            return SessionWindow.AFTERNOON
        elif t < time(16, 0):
            return SessionWindow.CLOSE
        else:
            return SessionWindow.OVERNIGHT

    def _analyze_pullback_short(
        self,
        price: float,
        open_price: float,
        high: float,
        low: float,
        ema9: float,
        ema21: float,
        ema50: float,
        vwap: float,
        atr: float,
        recent_bars: Optional[pd.DataFrame]
    ) -> PullbackAnalysis:
        result = PullbackAnalysis()

        if price <= 0 or atr <= 0:
            return result

        ema9_dist_pct = abs(price - ema9) / price * 100 if ema9 > 0 else 999
        ema21_dist_pct = abs(price - ema21) / price * 100 if ema21 > 0 else 999
        vwap_dist_pct = abs(price - vwap) / price * 100 if vwap > 0 else 999
        vwap_dist_atr = abs(price - vwap) / atr if vwap > 0 else 999

        ema9_touch = low <= ema9 <= high if ema9 > 0 else False
        ema9_near = ema9_dist_pct < self.ema_proximity_pct and price <= ema9

        if ema9_touch:
            result.score += 35
            result.touch_level = "EMA9"
            result.reasons.append("EMA9_REJECT")
        elif ema9_near:
            result.score += 25
            if not result.touch_level:
                result.touch_level = "EMA9"
            result.reasons.append("EMA9_RESISTANCE")

        ema21_touch = low <= ema21 <= high if ema21 > 0 else False
        ema21_near = ema21_dist_pct < self.ema_proximity_pct * 1.5 and price <= ema21

        if ema21_touch:
            result.score += 30
            if not result.touch_level:
                result.touch_level = "EMA21"
            result.reasons.append("EMA21_REJECT")
        elif ema21_near:
            result.score += 20
            if not result.touch_level:
                result.touch_level = "EMA21"
            result.reasons.append("EMA21_RESISTANCE")

        vwap_touch = low <= vwap <= high if vwap > 0 else False
        vwap_near = vwap_dist_pct < self.ema_proximity_pct and price <= vwap
        vwap_reject = (high > vwap > price) if vwap > 0 else False

        if vwap_reject:
            result.score += 35
            if not result.touch_level:
                result.touch_level = "VWAP"
            result.reasons.append("VWAP_REJECT")
        elif vwap_touch:
            result.score += 30
            if not result.touch_level:
                result.touch_level = "VWAP"
            result.reasons.append("VWAP_REJECT")
        elif vwap_near:
            result.score += 20
            if not result.touch_level:
                result.touch_level = "VWAP"
            result.reasons.append("VWAP_RESISTANCE")

        candle_range = high - low if high > low else 0.01
        candle_body = abs(price - open_price)
        body_ratio = candle_body / candle_range
        is_bearish = price < open_price

        if recent_bars is not None and len(recent_bars) >= 2:
            prev_bar = recent_bars.iloc[-2]
            prev_open = float(prev_bar.get('open', 0))
            prev_close = float(prev_bar.get('close', 0))
            prev_was_bullish = prev_close > prev_open

            if is_bearish and prev_was_bullish:
                if open_price > prev_close and price < prev_open:
                    result.score += 20
                    result.confirmation = "BEARISH_ENGULF"
                    result.reasons.append("ENGULFING")

        if is_bearish and body_ratio > self.min_candle_body_ratio:
            if result.confirmation != "BEARISH_ENGULF":
                result.confirmation = "STRONG_CLOSE"
            result.score += 15
            result.reasons.append("BEARISH_CLOSE")
        elif is_bearish:
            result.score += 5
            result.reasons.append("WEAK_BEARISH")

        if recent_bars is not None and len(recent_bars) >= 3:
            recent_highs = recent_bars['high'].tail(3).values
            if len(recent_highs) >= 3:
                if high < recent_highs[-2]:
                    result.score += 10
                    if not result.confirmation:
                        result.confirmation = "LOWER_HIGH"
                    result.reasons.append("LOWER_HIGH")
                if recent_highs[-2] < recent_highs[-3] and high < recent_highs[-2]:
                    result.score += 5
                    result.reasons.append("LH_SEQUENCE")

        if recent_bars is not None and len(recent_bars) >= 5:
            recent_low = recent_bars['low'].tail(5).min()
            if recent_low > 0:
                result.depth_pct = (high - recent_low) / recent_low * 100

                if self.pullback_depth_min_pct <= result.depth_pct <= self.pullback_depth_max_pct:
                    result.score += 10
                    result.reasons.append(f"IDEAL_DEPTH({result.depth_pct:.2f}%)")
                elif result.depth_pct > self.pullback_depth_max_pct:
                    result.score -= 10
                    result.reasons.append(f"DEEP_PULLBACK({result.depth_pct:.2f}%)")

        result.is_valid = (
            result.score >= self.min_pullback_score and
            result.touch_level != "" and
            len(result.reasons) >= 2
        )

        return result

    def _calculate_stops_and_targets_short(
        self,
        price: float,
        ema21: float,
        atr: float,
        pdl: float,
        recent_bars: Optional[pd.DataFrame],
        pullback_analysis: PullbackAnalysis
    ) -> Tuple[float, float, bool, str]:
        stop_candidates = []

        if recent_bars is not None and len(recent_bars) >= 3:
            pullback_high = recent_bars['high'].tail(3).max()
            stop_candidates.append(("PULLBACK_HIGH", pullback_high + atr * 0.3))

        if ema21 > 0:
            stop_candidates.append(("EMA21", ema21 + atr * 0.3))

        stop_candidates.append(("ATR", price + atr * self.stop_atr_mult))

        if stop_candidates:
            stop_candidates.sort(key=lambda x: x[1])
            stop_type, stop_loss = stop_candidates[0]
        else:
            stop_loss = price + atr * self.stop_atr_mult
            stop_type = "DEFAULT"

        stop_distance = stop_loss - price

        if stop_distance < self.min_stop_points:
            stop_loss = price + self.min_stop_points
            stop_distance = self.min_stop_points

        if stop_distance > self.max_stop_points:
            stop_loss = price + self.max_stop_points
            stop_distance = self.max_stop_points

        risk = stop_distance
        min_reward = risk * self.min_rr_ratio

        target_candidates = []

        if pdl > 0 and pdl < price:
            pdl_reward = price - pdl
            if pdl_reward >= min_reward:
                target_candidates.append(("PDL", pdl + atr * 0.25))

        measured_target = price - (risk * self.target_risk_mult)
        target_candidates.append(("MEASURED", measured_target))

        if pullback_analysis.score >= 50:
            extended_target = price - (risk * 3.0)
            target_candidates.append(("EXTENDED", extended_target))

        if target_candidates:
            pdl_targets = [t for t in target_candidates if t[0] == "PDL"]
            if pdl_targets:
                target_type, take_profit = pdl_targets[0]
            else:
                target_type, take_profit = target_candidates[0]
        else:
            take_profit = price - min_reward
            target_type = "MIN_RR"

        actual_reward = price - take_profit
        actual_rr = actual_reward / risk if risk > 0 else 0

        if actual_rr < self.min_rr_ratio:
            return stop_loss, take_profit, False, f"INSUFFICIENT_RR: {actual_rr:.2f} < {self.min_rr_ratio}"

        return stop_loss, take_profit, True, f"{stop_type}_STOP, {target_type}_TARGET, RR={actual_rr:.2f}"

    def evaluate(
        self,
        market_state: MarketStateResult,
        data: Dict[str, Any],
        recent_bars: Optional[pd.DataFrame] = None,
        timestamp: Optional[datetime] = None
    ) -> EntrySignal:
        session = self._get_session_window(timestamp)
        session_str = session.value

        metadata = {
            "session_window": session_str,
            "is_acceptance": market_state.is_acceptance,
            "is_exhaustion": market_state.is_exhaustion,
            "trend_score": market_state.trend_score,
            "continuation_score": market_state.continuation_score,
            "exhaustion_score": market_state.exhaustion_score,
            "allow_buy": market_state.allow_buy,
            "allow_short": market_state.allow_short,
            "phase": market_state.phase.value,
            "trend": market_state.trend.value,
        }

        if not market_state.allow_short:
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"SHORT_BLOCKED: {market_state.phase.value}",
                entry_type="BLOCKED",
                session_window=session_str,
                metadata=metadata
            )

        if market_state.is_exhaustion:
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason="EXHAUSTION_PRESENT: Momentum loss detected, wait for resolution",
                entry_type="CAUTION",
                session_window=session_str,
                metadata=metadata
            )

        if not market_state.is_acceptance:
            if market_state.trend not in [TrendDirection.STRONG_TREND_DOWN, TrendDirection.TREND_DOWN, TrendDirection.WEAK_DOWN]:
                return EntrySignal(
                    action="HOLD",
                    confidence=0.0,
                    reason=f"NOT_ACCEPTANCE: phase={market_state.phase.value}, trend={market_state.trend.value}",
                    entry_type="WAIT",
                    session_window=session_str,
                    metadata=metadata
                )

        price = float(data.get("close", data.get("price", 0)))
        open_price = float(data.get("open", price))
        high = float(data.get("high", price))
        low = float(data.get("low", price))

        ema9 = float(data.get("ema_9", data.get("EMA_9", price)))
        ema21 = float(data.get("ema_21", data.get("EMA_21", price)))
        ema50 = float(data.get("ema_50", data.get("EMA_50", ema21)))
        vwap = float(data.get("vwap", data.get("SESSION_VWAP", price)))

        atr = float(data.get("atr", data.get("ATR_14", 1.0)))
        adx = float(data.get("adx", data.get("ADX_14", 20)))
        macd_hist = float(data.get("macd_hist", data.get("MACD_hist", 0)))

        pdh = float(data.get("pdh", data.get("PDH", 0)))
        pdl = float(data.get("pdl", data.get("PDL", 0)))

        rsi = market_state.rsi_value

        metadata.update({
            "rsi": rsi,
            "adx": adx,
            "macd_hist": macd_hist,
            "price": price,
            "ema9": ema9,
            "ema21": ema21,
            "vwap": vwap,
            "atr": atr,
        })

        if self.require_htf_alignment:
            htf_ema21 = float(data.get("15m_ema21", data.get("15m_EMA_21", 0)))
            htf_ema50 = float(data.get("15m_ema50", data.get("15m_EMA_50", 0)))
            metadata["htf_ema21"] = htf_ema21
            metadata["htf_ema50"] = htf_ema50

            if htf_ema21 > 0 and htf_ema50 > 0 and htf_ema21 > htf_ema50:
                return EntrySignal(
                    action="HOLD",
                    confidence=0.0,
                    reason="HTF_UPTREND: 15m EMA21 > EMA50",
                    entry_type="BLOCKED",
                    session_window=session_str,
                    metadata=metadata
                )

        if rsi < self.rsi_overextension:
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"RSI_OVERSOLD: {rsi:.1f} < {self.rsi_overextension} (rebound risk)",
                entry_type="BLOCKED",
                session_window=session_str,
                metadata=metadata
            )

        if vwap > 0 and atr > 0:
            vwap_dist_atr = (vwap - price) / atr
            if vwap_dist_atr > self.max_vwap_distance_atr:
                return EntrySignal(
                    action="HOLD",
                    confidence=0.0,
                    reason=f"VWAP_DISTANCE: {vwap_dist_atr:.1f} ATR > {self.max_vwap_distance_atr} (need pullback)",
                    entry_type="WAIT",
                    session_window=session_str,
                    metadata=metadata
                )

        rsi_in_zone = self.rsi_min <= rsi <= self.rsi_max
        rsi_in_ideal = self.rsi_ideal_min <= rsi <= self.rsi_ideal_max

        if not rsi_in_zone:
            if rsi > self.rsi_max + 5:
                return EntrySignal(
                    action="HOLD",
                    confidence=0.0,
                    reason=f"RSI_TOO_HIGH: {rsi:.1f} > {self.rsi_max + 5} (wait for momentum)",
                    entry_type="WAIT",
                    session_window=session_str,
                    metadata=metadata
                )

        pullback = self._analyze_pullback_short(
            price=price,
            open_price=open_price,
            high=high,
            low=low,
            ema9=ema9,
            ema21=ema21,
            ema50=ema50,
            vwap=vwap,
            atr=atr,
            recent_bars=recent_bars
        )

        metadata["pullback_score"] = pullback.score
        metadata["pullback_level"] = pullback.touch_level
        metadata["pullback_confirmation"] = pullback.confirmation
        metadata["pullback_depth_pct"] = pullback.depth_pct
        metadata["pullback_reasons"] = pullback.reasons

        if not pullback.is_valid:
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"NO_VALID_PULLBACK: score={pullback.score:.0f}, level={pullback.touch_level or 'NONE'}",
                entry_type="WAIT",
                session_window=session_str,
                metadata=metadata
            )

        stop_loss, take_profit, rr_valid, rr_reason = self._calculate_stops_and_targets_short(
            price=price,
            ema21=ema21,
            atr=atr,
            pdl=pdl,
            recent_bars=recent_bars,
            pullback_analysis=pullback
        )

        metadata["stop_loss"] = stop_loss
        metadata["take_profit"] = take_profit
        metadata["stop_reason"] = rr_reason

        if not rr_valid:
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=rr_reason,
                entry_type="BLOCKED",
                session_window=session_str,
                metadata=metadata
            )

        base_confidence = 0.55

        if session == SessionWindow.MORNING_PRIME:
            base_confidence += 0.10
        elif session == SessionWindow.MORNING_OPEN:
            base_confidence += 0.05
        elif session == SessionWindow.MIDDAY:
            base_confidence -= 0.10

        if market_state.is_acceptance:
            base_confidence += 0.10

        if rsi_in_ideal:
            base_confidence += 0.08
        elif rsi_in_zone:
            base_confidence += 0.04

        if adx >= self.adx_strong_threshold:
            base_confidence += 0.08
        elif adx >= self.adx_trend_threshold:
            base_confidence += 0.04

        if macd_hist < self.macd_acceptance_threshold:
            base_confidence += 0.06
        elif macd_hist < 0:
            base_confidence += 0.03

        pullback_bonus = (pullback.score - self.min_pullback_score) / 100
        base_confidence += min(0.15, pullback_bonus)

        if pullback.confirmation == "BEARISH_ENGULF":
            base_confidence += 0.05

        if market_state.continuation_score > 60:
            base_confidence += 0.05

        final_confidence = min(0.95, max(self.min_confidence, base_confidence))

        metadata["confidence_breakdown"] = {
            "base": 0.55,
            "session": session_str,
            "acceptance": market_state.is_acceptance,
            "rsi_in_ideal": rsi_in_ideal,
            "adx": adx,
            "macd_hist": macd_hist,
            "pullback_score": pullback.score,
            "final": final_confidence
        }

        reason_parts = [
            "SHORT_CONTINUATION",
            f"{pullback.touch_level}_PULLBACK",
        ]
        if pullback.confirmation:
            reason_parts.append(pullback.confirmation)
        if session == SessionWindow.MORNING_PRIME:
            reason_parts.append("MORNING_PRIME")
        if market_state.is_acceptance:
            reason_parts.append("ACCEPTANCE")

        return EntrySignal(
            action="SELL",
            confidence=final_confidence,
            reason=" | ".join(reason_parts),
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_type="CONTINUATION",
            session_window=session_str,
            metadata=metadata
        )


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

@dataclass
class EveningPatternAnalysis:
    """Analysis result for evening continuation patterns.
    
    Only two patterns are valid:
    - BREAK_AND_HOLD: Price breaks level, next candle holds above
    - MICRO_PULLBACK: Shallow pullback (<= 20%), stays above VWAP
    """
    is_valid: bool = False
    pattern_type: str = ""  # "BREAK_HOLD", "MICRO_PULLBACK", "NONE"
    score: float = 0.0
    breakout_level: float = 0.0
    pullback_depth_pct: float = 0.0
    volume_confirmed: bool = False
    reasons: List[str] = field(default_factory=list)


class EveningContinuationModule:
    """EVENING continuation module - high-momentum continuation in late RTH.
    
    ═══════════════════════════════════════════════════════════════════════════
    PURPOSE:
    This module captures ONLY strong, high-momentum continuation during late RTH
    (14:00-16:00 CST) when pullbacks are unreliable and chop risk is elevated.
    
    DESIGN PHILOSOPHY:
    - Trade LESS frequently than morning continuation
    - Require STRONGER confirmation (ADX >= 32, trend_score >= 70)
    - Use TIGHTER stops (max 2.75 pts) and SMALLER targets (2.0-2.5R)
    - Skip most setups by design — DISCIPLINE over frequency
    
    ═══════════════════════════════════════════════════════════════════════════
    VALID SESSION WINDOWS:
    - AFTERNOON (14:00-15:00 CST)
    - CLOSE (15:00-16:00 CST)
    - NO new entries after 15:40 CST (manage-only mode)
    
    ═══════════════════════════════════════════════════════════════════════════
    ENTRY PATTERNS (ONLY TWO ALLOWED):
    
    TYPE A - BREAK_AND_HOLD:
        - Break above PDH / range high / last impulse high
        - Next candle HOLDS above breakout level
        - Close in top 30% of candle
        - Volume >= session median
    
    TYPE B - MICRO_PULLBACK:
        - Pullback depth <= 20% of last impulse
        - Does NOT touch EMA21
        - Holds above VWAP
        - Followed by strong bullish close
    
    ═══════════════════════════════════════════════════════════════════════════
    HARD BLOCKS:
    - RSI > 68 (distribution risk in late session)
    - RSI < 55 (momentum loss)
    - Stop distance > 2.75 points
    - Exhaustion present
    - Any session outside AFTERNOON/CLOSE
    - Time > 15:40 CST
    
    ═══════════════════════════════════════════════════════════════════════════
    EXPLICITLY DISALLOWED:
    - EMA21 or VWAP pullbacks (morning module territory)
    - Deep retracements (> 20%)
    - RSI divergence plays
    - Inside-bar compression without volume
    - Chop / overlapping candles
    - Any SELL or reversal logic
    
    ═══════════════════════════════════════════════════════════════════════════
    Author: Trading System Enhancement - Jan 27, 2026
    """
    
    # =========================================================================
    # CONFIGURATION CONSTANTS - DO NOT RELAX THESE THRESHOLDS
    # =========================================================================
    
    # Session timing (CST)
    EVENING_START = time(14, 0)   # 14:00 CST - AFTERNOON starts
    EVENING_END = time(16, 0)     # 16:00 CST - RTH close
    CUTOFF_TIME = time(15, 40)    # 15:40 CST - No new entries after this
    
    # RSI requirements - STRICT zone for late-day momentum
    RSI_MIN = 55       # Below this = momentum loss, skip
    RSI_MAX = 68       # Above this = distribution risk, skip
    RSI_IDEAL_MIN = 58 # Ideal continuation zone
    RSI_IDEAL_MAX = 65 # Ideal continuation zone
    
    # Trend requirements - STRONGER than morning
    ADX_MIN = 32              # Higher than morning (25) - need confirmed trend
    MACD_MIN = 0.30           # Higher than morning (0.20) - need strong momentum
    TREND_SCORE_MIN = 70      # Higher threshold for late-day
    
    # Risk management - TIGHTER than morning
    MAX_STOP_DISTANCE = 2.75  # Maximum stop in points
    MIN_STOP_DISTANCE = 1.50  # Minimum stop for realistic execution
    PARTIAL_TARGET_RR = 1.5   # Mandatory partial at 1.5R
    FINAL_TARGET_MIN_RR = 2.0 # Minimum final target
    FINAL_TARGET_MAX_RR = 2.5 # Maximum final target (no extended targets)
    
    # Pullback requirements
    MAX_PULLBACK_DEPTH_PCT = 0.20  # Maximum 20% retracement
    
    # Confidence thresholds
    BASE_CONFIDENCE = 0.50
    MIN_CONFIDENCE = 0.65     # Must meet this to trigger
    MAX_CONFIDENCE = 0.78     # Cap - late day is always uncertain
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize evening continuation module.
        
        Args:
            config: Optional configuration overrides (use with caution)
        """
        config = config or {}
        
        # Allow config overrides but use strict defaults
        self.rsi_min = config.get("rsi_min", self.RSI_MIN)
        self.rsi_max = config.get("rsi_max", self.RSI_MAX)
        self.adx_min = config.get("adx_min", self.ADX_MIN)
        self.macd_min = config.get("macd_min", self.MACD_MIN)
        self.trend_score_min = config.get("trend_score_min", self.TREND_SCORE_MIN)
        self.max_stop_distance = config.get("max_stop_distance", self.MAX_STOP_DISTANCE)
        
        # Track statistics
        self.signals_generated = 0
        self.signals_blocked = 0
        
        logger.info(
            f"[EVENING_CONTINUATION] Module initialized: "
            f"RSI=[{self.rsi_min}-{self.rsi_max}], "
            f"ADX>={self.adx_min}, MACD>={self.macd_min}, "
            f"trend_score>={self.trend_score_min}, "
            f"max_stop={self.max_stop_distance}pts"
        )
    
    def _get_session_window(self, timestamp: Optional[datetime]) -> SessionWindow:
        """Determine current session window from timestamp.
        
        Returns SessionWindow enum, defaulting to OVERNIGHT if outside RTH.
        """
        if timestamp is None:
            # Default to midday (not valid for evening module)
            return SessionWindow.MIDDAY
        
        current_time = timestamp.time()
        
        # Check session boundaries
        if current_time < time(9, 30):
            return SessionWindow.PRE_MARKET
        elif current_time < time(10, 0):
            return SessionWindow.MORNING_OPEN
        elif current_time < time(11, 0):
            return SessionWindow.MORNING_PRIME
        elif current_time < time(14, 0):
            return SessionWindow.MIDDAY
        elif current_time < time(15, 0):
            return SessionWindow.AFTERNOON
        elif current_time < time(16, 0):
            return SessionWindow.CLOSE
        else:
            return SessionWindow.OVERNIGHT
    
    def _is_valid_evening_session(self, timestamp: Optional[datetime]) -> Tuple[bool, str]:
        """Check if current time is valid for evening continuation.
        
        Returns:
            Tuple of (is_valid, reason)
        """
        if timestamp is None:
            return False, "NO_TIMESTAMP"
        
        current_time = timestamp.time()
        
        # Must be in AFTERNOON or CLOSE session
        if current_time < self.EVENING_START:
            return False, f"TOO_EARLY: {current_time} < 14:00 CST"
        
        if current_time >= self.EVENING_END:
            return False, f"RTH_CLOSED: {current_time} >= 16:00 CST"
        
        # No new entries after 15:40 CST
        if current_time >= self.CUTOFF_TIME:
            return False, f"CUTOFF_REACHED: {current_time} >= 15:40 CST (manage-only)"
        
        return True, "VALID_EVENING_SESSION"
    
    def _analyze_break_and_hold(
        self,
        price: float,
        open_price: float,
        high: float,
        low: float,
        pdh: float,
        recent_bars: Optional[pd.DataFrame],
        volume: float,
        session_median_volume: float
    ) -> EveningPatternAnalysis:
        """Analyze for Break-and-Hold continuation pattern.
        
        TYPE A PATTERN:
        - Price broke above a key level (PDH, range high, impulse high)
        - Current candle HOLDS above that breakout level
        - Close is in top 30% of candle range
        - Volume confirms (>= session median)
        
        WHY THIS WORKS:
        - Late-day breakouts with hold = institutional commitment
        - Top 30% close = buyers in control
        - Volume confirmation = real participation, not noise
        
        Args:
            price: Current close price
            open_price: Current open price
            high: Current high
            low: Current low  
            pdh: Previous day high (key level)
            recent_bars: Recent price history
            volume: Current bar volume
            session_median_volume: Session median for comparison
            
        Returns:
            EveningPatternAnalysis with pattern details
        """
        result = EveningPatternAnalysis()
        result.reasons = []
        
        # Need recent bars for proper analysis
        if recent_bars is None or len(recent_bars) < 3:
            result.reasons.append("INSUFFICIENT_BARS")
            return result
        
        # Identify potential breakout levels
        breakout_levels = []
        
        # Level 1: PDH (most significant)
        if pdh > 0:
            breakout_levels.append(("PDH", pdh))
        
        # Level 2: Recent swing high (last 10 bars)
        if len(recent_bars) >= 10:
            recent_high = recent_bars['high'].tail(10).max()
            if recent_high > 0:
                breakout_levels.append(("SWING_HIGH", recent_high))
        
        # Level 3: Last impulse high (highest of last 5 bars)
        if len(recent_bars) >= 5:
            impulse_high = recent_bars['high'].tail(5).max()
            if impulse_high > 0 and impulse_high not in [l[1] for l in breakout_levels]:
                breakout_levels.append(("IMPULSE_HIGH", impulse_high))
        
        if not breakout_levels:
            result.reasons.append("NO_BREAKOUT_LEVELS")
            return result
        
        # Check for break-and-hold pattern
        pattern_score = 0.0
        best_level_name = ""
        best_level_price = 0.0
        
        for level_name, level_price in breakout_levels:
            # Did we break above this level?
            if high > level_price:
                # Are we HOLDING above it? (close above level)
                if price > level_price:
                    # Calculate hold margin
                    hold_margin = (price - level_price) / level_price if level_price > 0 else 0
                    
                    if hold_margin > 0:
                        level_score = 30.0  # Base score for break-and-hold
                        
                        # Bonus for PDH (most significant level)
                        if level_name == "PDH":
                            level_score += 15.0
                            result.reasons.append("PDH_BREAKOUT")
                        elif level_name == "SWING_HIGH":
                            level_score += 10.0
                            result.reasons.append("SWING_HIGH_BREAKOUT")
                        else:
                            level_score += 5.0
                            result.reasons.append("IMPULSE_BREAKOUT")
                        
                        if level_score > pattern_score:
                            pattern_score = level_score
                            best_level_name = level_name
                            best_level_price = level_price
        
        if pattern_score == 0:
            result.reasons.append("NO_BREAK_AND_HOLD")
            return result
        
        # Check close position in candle (must be top 30%)
        candle_range = high - low if high > low else 0.01
        close_position = (price - low) / candle_range
        
        if close_position < 0.70:  # Not in top 30%
            result.reasons.append(f"WEAK_CLOSE_POSITION: {close_position:.1%} (need >= 70%)")
            return result
        
        pattern_score += 15.0  # Bonus for strong close position
        result.reasons.append(f"STRONG_CLOSE: top {(1 - close_position):.0%}")
        
        # Check volume confirmation
        if session_median_volume > 0 and volume >= session_median_volume:
            pattern_score += 15.0
            result.volume_confirmed = True
            result.reasons.append("VOLUME_CONFIRMED")
        elif volume > 0:
            result.reasons.append("VOLUME_BELOW_MEDIAN")
            # Still allow but don't add bonus
        
        # Bullish candle check (close > open)
        if price > open_price:
            pattern_score += 10.0
            result.reasons.append("BULLISH_CANDLE")
        else:
            result.reasons.append("BEARISH_CANDLE_WARNING")
            pattern_score -= 10.0  # Penalty for bearish close
        
        # Validate pattern score
        if pattern_score >= 50.0:
            result.is_valid = True
            result.pattern_type = "BREAK_HOLD"
            result.score = pattern_score
            result.breakout_level = best_level_price
        
        return result
    
    def _analyze_micro_pullback(
        self,
        price: float,
        open_price: float,
        high: float,
        low: float,
        ema9: float,
        ema21: float,
        vwap: float,
        atr: float,
        recent_bars: Optional[pd.DataFrame]
    ) -> EveningPatternAnalysis:
        """Analyze for Micro Pullback continuation pattern.
        
        TYPE B PATTERN:
        - Pullback depth <= 20% of last impulse
        - Does NOT touch EMA21 (too deep for late-day)
        - Holds above VWAP (institutional anchor)
        - Current candle shows strong bullish close
        
        WHY THIS WORKS:
        - Shallow pullbacks in strong trends = eager buyers
        - EMA21 touch in late-day often = trend ending
        - VWAP hold = institutional support
        
        Args:
            price: Current close price
            open_price: Current open price  
            high: Current high
            low: Current low
            ema9: EMA 9 value
            ema21: EMA 21 value
            vwap: Session VWAP
            atr: ATR for normalization
            recent_bars: Recent price history
            
        Returns:
            EveningPatternAnalysis with pattern details
        """
        result = EveningPatternAnalysis()
        result.reasons = []
        
        # Need recent bars for impulse measurement
        if recent_bars is None or len(recent_bars) < 5:
            result.reasons.append("INSUFFICIENT_BARS")
            return result
        
        # Calculate last impulse (highest high of recent bars)
        impulse_high = recent_bars['high'].tail(10).max() if len(recent_bars) >= 10 else high
        impulse_low = recent_bars['low'].tail(5).min() if len(recent_bars) >= 5 else low
        impulse_range = impulse_high - impulse_low if impulse_high > impulse_low else 1.0
        
        # Calculate pullback depth
        pullback_depth = impulse_high - low  # How far we pulled back from high
        pullback_depth_pct = pullback_depth / impulse_range if impulse_range > 0 else 1.0
        
        result.pullback_depth_pct = pullback_depth_pct
        
        # HARD CHECK: Pullback must be <= 20%
        if pullback_depth_pct > self.MAX_PULLBACK_DEPTH_PCT:
            result.reasons.append(f"PULLBACK_TOO_DEEP: {pullback_depth_pct:.1%} > 20%")
            return result
        
        result.reasons.append(f"SHALLOW_PULLBACK: {pullback_depth_pct:.1%}")
        pattern_score = 25.0  # Base score for shallow pullback
        
        # HARD CHECK: Must NOT touch EMA21 (too deep for evening)
        if low <= ema21:
            result.reasons.append("TOUCHED_EMA21: Pullback too deep for evening")
            return result
        
        result.reasons.append("ABOVE_EMA21")
        pattern_score += 15.0
        
        # HARD CHECK: Must hold above VWAP
        if price < vwap:
            result.reasons.append("BELOW_VWAP: Lost institutional anchor")
            return result
        
        # Bonus if low held above VWAP (not just close)
        if low > vwap:
            result.reasons.append("LOW_ABOVE_VWAP")
            pattern_score += 15.0
        else:
            result.reasons.append("CLOSE_ABOVE_VWAP")
            pattern_score += 10.0
        
        # Check for strong bullish close (recovery from pullback)
        candle_range = high - low if high > low else 0.01
        close_position = (price - low) / candle_range
        
        if close_position >= 0.60:  # Close in upper 40%
            result.reasons.append(f"BULLISH_RECOVERY: close at {close_position:.0%}")
            pattern_score += 15.0
        elif close_position >= 0.50:
            result.reasons.append("NEUTRAL_CLOSE")
            pattern_score += 5.0
        else:
            result.reasons.append("WEAK_CLOSE_POSITION")
            return result  # Reject weak closes
        
        # Bullish candle (close > open)
        if price > open_price:
            result.reasons.append("BULLISH_CANDLE")
            pattern_score += 10.0
        
        # Price above EMA9 (immediate trend)
        if price > ema9:
            result.reasons.append("ABOVE_EMA9")
            pattern_score += 10.0
        
        # Validate pattern score
        if pattern_score >= 50.0:
            result.is_valid = True
            result.pattern_type = "MICRO_PULLBACK"
            result.score = pattern_score
        
        return result
    
    def _calculate_stops_and_targets(
        self,
        price: float,
        pattern: EveningPatternAnalysis,
        atr: float,
        low: float,
        recent_bars: Optional[pd.DataFrame]
    ) -> Tuple[float, float, float, bool, str]:
        """Calculate stop loss and targets for evening entry.
        
        RISK RULES (STRICT):
        - Maximum stop distance: 2.75 points
        - Minimum stop distance: 1.50 points (realistic execution)
        - Mandatory partial at 1.5R
        - Final target: 2.0R to 2.5R (NO extended targets)
        
        WHY TIGHTER STOPS:
        - Late-day moves are faster and more volatile
        - Less time for recovery if wrong
        - Smaller wins but higher win rate
        
        Args:
            price: Entry price
            pattern: Pattern analysis result
            atr: Current ATR
            low: Current bar low
            recent_bars: Recent price history
            
        Returns:
            Tuple of (stop_loss, partial_target, final_target, is_valid, reason)
        """
        # Determine stop location based on pattern
        if pattern.pattern_type == "BREAK_HOLD":
            # Stop below breakout level or current low
            stop_candidates = [
                pattern.breakout_level - atr * 0.3,  # Below breakout
                low - atr * 0.25,                     # Below current bar
            ]
        else:  # MICRO_PULLBACK
            # Stop below pullback low
            stop_candidates = [
                low - atr * 0.25,  # Below current bar low
            ]
            
            # Add recent swing low if available
            if recent_bars is not None and len(recent_bars) >= 3:
                swing_low = recent_bars['low'].tail(3).min()
                stop_candidates.append(swing_low - atr * 0.2)
        
        # Use tightest stop that's still valid
        stop_loss = max(stop_candidates)  # Closest to price
        
        # Calculate risk
        risk = price - stop_loss
        
        # HARD CHECK: Enforce maximum stop distance
        if risk > self.MAX_STOP_DISTANCE:
            return 0, 0, 0, False, f"STOP_TOO_WIDE: {risk:.2f}pts > {self.MAX_STOP_DISTANCE}pts"
        
        # HARD CHECK: Enforce minimum stop distance
        if risk < self.MIN_STOP_DISTANCE:
            # Widen stop to minimum
            stop_loss = price - self.MIN_STOP_DISTANCE
            risk = self.MIN_STOP_DISTANCE
        
        # Calculate targets
        partial_target = price + (risk * self.PARTIAL_TARGET_RR)  # 1.5R
        
        # Final target between 2.0R and 2.5R based on pattern strength
        if pattern.score >= 70:
            final_rr = self.FINAL_TARGET_MAX_RR  # 2.5R for strong patterns
        else:
            final_rr = self.FINAL_TARGET_MIN_RR  # 2.0R for standard patterns
        
        final_target = price + (risk * final_rr)
        
        reason = f"STOP={risk:.2f}pts, PARTIAL=1.5R, FINAL={final_rr}R"
        
        return stop_loss, partial_target, final_target, True, reason
    
    def _calculate_confidence(
        self,
        pattern: EveningPatternAnalysis,
        market_state: MarketStateResult,
        rsi: float,
        adx: float,
        macd_hist: float
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate confidence score for evening entry.
        
        BASE: 50%
        
        BONUSES:
        - ADX >= 35: +8%
        - MACD >= 0.35: +6%
        - Break-and-hold pattern: +10%
        - Volume expansion: +6%
        - RSI in ideal zone [58-65]: +4%
        
        RULES:
        - Minimum to trade: 65%
        - Maximum cap: 78% (late-day uncertainty)
        
        Args:
            pattern: Pattern analysis result
            market_state: Current market state
            rsi: Current RSI value
            adx: Current ADX value
            macd_hist: Current MACD histogram
            
        Returns:
            Tuple of (confidence, breakdown_dict)
        """
        confidence = self.BASE_CONFIDENCE
        breakdown = {"base": self.BASE_CONFIDENCE}
        
        # ADX bonus (strong trend)
        if adx >= 35:
            confidence += 0.08
            breakdown["adx_35+"] = 0.08
        elif adx >= self.adx_min:
            confidence += 0.04
            breakdown["adx_32+"] = 0.04
        
        # MACD bonus (strong momentum)
        if macd_hist >= 0.35:
            confidence += 0.06
            breakdown["macd_0.35+"] = 0.06
        elif macd_hist >= self.macd_min:
            confidence += 0.03
            breakdown["macd_0.30+"] = 0.03
        
        # Pattern bonus
        if pattern.pattern_type == "BREAK_HOLD":
            confidence += 0.10
            breakdown["break_hold_pattern"] = 0.10
        elif pattern.pattern_type == "MICRO_PULLBACK":
            confidence += 0.06
            breakdown["micro_pullback_pattern"] = 0.06
        
        # Volume confirmation bonus
        if pattern.volume_confirmed:
            confidence += 0.06
            breakdown["volume_confirmed"] = 0.06
        
        # RSI in ideal zone bonus
        if self.RSI_IDEAL_MIN <= rsi <= self.RSI_IDEAL_MAX:
            confidence += 0.04
            breakdown["rsi_ideal_zone"] = 0.04
        
        # Trend score bonus
        if market_state.trend_score >= 80:
            confidence += 0.04
            breakdown["trend_score_80+"] = 0.04
        
        # Apply caps
        final_confidence = max(0.0, min(self.MAX_CONFIDENCE, confidence))
        breakdown["final"] = final_confidence
        
        return final_confidence, breakdown
    
    def evaluate(
        self,
        market_state: MarketStateResult,
        data: Dict[str, Any],
        recent_bars: Optional[pd.DataFrame] = None,
        timestamp: Optional[datetime] = None
    ) -> EntrySignal:
        """Evaluate for evening continuation entry.
        
        This method implements STRICT filtering with default NO_SIGNAL behavior.
        Entry is allowed ONLY when ALL criteria pass.
        
        EVALUATION ORDER:
        1. Session validation (AFTERNOON/CLOSE only, before 15:40 CST)
        2. Market state requirements (acceptance, no exhaustion)
        3. Indicator requirements (ADX, MACD, RSI, trend_score)
        4. Pattern detection (BREAK_HOLD or MICRO_PULLBACK)
        5. Risk validation (stop <= 2.75 points)
        6. Confidence validation (>= 65%)
        
        Args:
            market_state: Current market state from detector
            data: Current bar data with indicators
            recent_bars: Recent price history for pattern detection
            timestamp: Current timestamp for session filtering
            
        Returns:
            EntrySignal with BUY or HOLD (never SELL)
        """
        # Initialize metadata for logging
        metadata = {
            "module": "EVENING_CONTINUATION",
            "timestamp": str(timestamp) if timestamp else "UNKNOWN",
        }
        
        # =====================================================================
        # GATE 1: SESSION VALIDATION (HARD BLOCK)
        # =====================================================================
        session_valid, session_reason = self._is_valid_evening_session(timestamp)
        session = self._get_session_window(timestamp)
        metadata["session_window"] = session.value
        metadata["session_valid"] = session_valid
        
        if not session_valid:
            self.signals_blocked += 1
            logger.debug(f"[EVENING_CONTINUATION] BLOCKED: {session_reason}")
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"SESSION_INVALID: {session_reason}",
                entry_type="NO_SIGNAL",
                session_window=session.value,
                metadata=metadata
            )
        
        # =====================================================================
        # GATE 2: MARKET STATE REQUIREMENTS (ALL REQUIRED)
        # =====================================================================
        
        # REQUIRE: is_acceptance == True
        if not market_state.is_acceptance:
            self.signals_blocked += 1
            logger.debug(f"[EVENING_CONTINUATION] BLOCKED: NOT_ACCEPTANCE (phase={market_state.phase.value})")
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"NOT_ACCEPTANCE: phase={market_state.phase.value}",
                entry_type="NO_SIGNAL",
                session_window=session.value,
                metadata=metadata
            )
        
        # REQUIRE: is_exhaustion == False
        if market_state.is_exhaustion:
            self.signals_blocked += 1
            logger.debug("[EVENING_CONTINUATION] BLOCKED: EXHAUSTION_PRESENT")
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason="EXHAUSTION_PRESENT: Momentum loss detected",
                entry_type="NO_SIGNAL",
                session_window=session.value,
                metadata=metadata
            )
        
        # REQUIRE: EMA_STACK_UP == True
        if not market_state.ema_stack_up:
            self.signals_blocked += 1
            logger.debug("[EVENING_CONTINUATION] BLOCKED: EMA_STACK_NOT_UP")
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason="EMA_STACK_NOT_UP: Trend structure broken",
                entry_type="NO_SIGNAL",
                session_window=session.value,
                metadata=metadata
            )
        
        # =====================================================================
        # EXTRACT PRICE DATA
        # =====================================================================
        price = float(data.get("close", data.get("price", 0)))
        open_price = float(data.get("open", price))
        high = float(data.get("high", price))
        low = float(data.get("low", price))
        
        ema9 = float(data.get("ema_9", data.get("EMA_9", price)))
        ema21 = float(data.get("ema_21", data.get("EMA_21", price)))
        vwap = float(data.get("vwap", data.get("SESSION_VWAP", price)))
        atr = float(data.get("atr", data.get("ATR_14", 1.0)))
        adx = float(data.get("adx", data.get("ADX_14", 20)))
        macd_hist = float(data.get("macd_hist", data.get("MACD_hist", 0)))
        
        pdh = float(data.get("pdh", data.get("PDH", 0)))
        
        volume = float(data.get("volume", data.get("VOLUME", 0)))
        session_median_volume = float(data.get("session_median_volume", volume * 0.8))
        
        rsi = market_state.rsi_value
        trend_score = market_state.trend_score
        
        metadata.update({
            "price": price,
            "rsi": rsi,
            "adx": adx,
            "macd_hist": macd_hist,
            "trend_score": trend_score,
            "vwap": vwap,
            "ema21": ema21,
        })
        
        # =====================================================================
        # GATE 3: PRICE POSITION REQUIREMENTS
        # =====================================================================
        
        # REQUIRE: Price ABOVE VWAP
        if price < vwap:
            self.signals_blocked += 1
            logger.debug(f"[EVENING_CONTINUATION] BLOCKED: BELOW_VWAP ({price:.2f} < {vwap:.2f})")
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"BELOW_VWAP: {price:.2f} < {vwap:.2f}",
                entry_type="NO_SIGNAL",
                session_window=session.value,
                metadata=metadata
            )
        
        # REQUIRE: Price ABOVE EMA21
        if price < ema21:
            self.signals_blocked += 1
            logger.debug(f"[EVENING_CONTINUATION] BLOCKED: BELOW_EMA21 ({price:.2f} < {ema21:.2f})")
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"BELOW_EMA21: {price:.2f} < {ema21:.2f}",
                entry_type="NO_SIGNAL",
                session_window=session.value,
                metadata=metadata
            )
        
        # =====================================================================
        # GATE 4: INDICATOR REQUIREMENTS (ALL REQUIRED)
        # =====================================================================
        
        # REQUIRE: ADX >= 32
        if adx < self.adx_min:
            self.signals_blocked += 1
            logger.debug(f"[EVENING_CONTINUATION] BLOCKED: ADX_TOO_LOW ({adx:.1f} < {self.adx_min})")
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"ADX_TOO_LOW: {adx:.1f} < {self.adx_min}",
                entry_type="NO_SIGNAL",
                session_window=session.value,
                metadata=metadata
            )
        
        # REQUIRE: MACD >= 0.30
        if macd_hist < self.macd_min:
            self.signals_blocked += 1
            logger.debug(f"[EVENING_CONTINUATION] BLOCKED: MACD_TOO_LOW ({macd_hist:.2f} < {self.macd_min})")
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"MACD_TOO_LOW: {macd_hist:.2f} < {self.macd_min}",
                entry_type="NO_SIGNAL",
                session_window=session.value,
                metadata=metadata
            )
        
        # REQUIRE: trend_score >= 70
        if trend_score < self.trend_score_min:
            self.signals_blocked += 1
            logger.debug(f"[EVENING_CONTINUATION] BLOCKED: TREND_SCORE_LOW ({trend_score:.0f} < {self.trend_score_min})")
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"TREND_SCORE_LOW: {trend_score:.0f} < {self.trend_score_min}",
                entry_type="NO_SIGNAL",
                session_window=session.value,
                metadata=metadata
            )
        
        # =====================================================================
        # GATE 5: RSI REQUIREMENTS (STRICT ZONE)
        # =====================================================================
        
        # HARD BLOCK: RSI > 68 (distribution risk in late session)
        if rsi > self.rsi_max:
            self.signals_blocked += 1
            logger.debug(f"[EVENING_CONTINUATION] BLOCKED: RSI_TOO_HIGH ({rsi:.1f} > {self.rsi_max}) - distribution risk")
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"RSI_TOO_HIGH: {rsi:.1f} > {self.rsi_max} (distribution risk)",
                entry_type="NO_SIGNAL",
                session_window=session.value,
                metadata=metadata
            )
        
        # HARD BLOCK: RSI < 55 (momentum loss)
        if rsi < self.rsi_min:
            self.signals_blocked += 1
            logger.debug(f"[EVENING_CONTINUATION] BLOCKED: RSI_TOO_LOW ({rsi:.1f} < {self.rsi_min}) - momentum loss")
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"RSI_TOO_LOW: {rsi:.1f} < {self.rsi_min} (momentum loss)",
                entry_type="NO_SIGNAL",
                session_window=session.value,
                metadata=metadata
            )
        
        # =====================================================================
        # GATE 6: PATTERN DETECTION (MUST MATCH ONE)
        # =====================================================================
        
        # Try Break-and-Hold pattern first (higher priority)
        break_hold = self._analyze_break_and_hold(
            price=price,
            open_price=open_price,
            high=high,
            low=low,
            pdh=pdh,
            recent_bars=recent_bars,
            volume=volume,
            session_median_volume=session_median_volume
        )
        
        # Try Micro Pullback pattern
        micro_pullback = self._analyze_micro_pullback(
            price=price,
            open_price=open_price,
            high=high,
            low=low,
            ema9=ema9,
            ema21=ema21,
            vwap=vwap,
            atr=atr,
            recent_bars=recent_bars
        )
        
        # Select best valid pattern
        pattern: Optional[EveningPatternAnalysis] = None
        
        if break_hold.is_valid and micro_pullback.is_valid:
            # Both valid - pick higher score
            pattern = break_hold if break_hold.score >= micro_pullback.score else micro_pullback
        elif break_hold.is_valid:
            pattern = break_hold
        elif micro_pullback.is_valid:
            pattern = micro_pullback
        
        if pattern is None:
            self.signals_blocked += 1
            reject_reasons = break_hold.reasons + micro_pullback.reasons
            logger.debug(f"[EVENING_CONTINUATION] BLOCKED: NO_VALID_PATTERN - {', '.join(reject_reasons[:3])}")
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"NO_VALID_PATTERN: {', '.join(reject_reasons[:3])}",
                entry_type="NO_SIGNAL",
                session_window=session.value,
                metadata=metadata
            )
        
        metadata["pattern_type"] = pattern.pattern_type
        metadata["pattern_score"] = pattern.score
        metadata["pattern_reasons"] = pattern.reasons
        
        # =====================================================================
        # GATE 7: RISK VALIDATION (STOP MUST BE <= 2.75 PTS)
        # =====================================================================
        
        stop_loss, partial_target, final_target, risk_valid, risk_reason = self._calculate_stops_and_targets(
            price=price,
            pattern=pattern,
            atr=atr,
            low=low,
            recent_bars=recent_bars
        )
        
        if not risk_valid:
            self.signals_blocked += 1
            logger.debug(f"[EVENING_CONTINUATION] BLOCKED: {risk_reason}")
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=risk_reason,
                entry_type="NO_SIGNAL",
                session_window=session.value,
                metadata=metadata
            )
        
        metadata["stop_loss"] = stop_loss
        metadata["partial_target"] = partial_target
        metadata["final_target"] = final_target
        metadata["risk_reason"] = risk_reason
        
        # =====================================================================
        # GATE 8: CONFIDENCE VALIDATION (MUST BE >= 65%)
        # =====================================================================
        
        confidence, confidence_breakdown = self._calculate_confidence(
            pattern=pattern,
            market_state=market_state,
            rsi=rsi,
            adx=adx,
            macd_hist=macd_hist
        )
        
        metadata["confidence_breakdown"] = confidence_breakdown
        
        if confidence < self.MIN_CONFIDENCE:
            self.signals_blocked += 1
            logger.debug(f"[EVENING_CONTINUATION] BLOCKED: CONFIDENCE_TOO_LOW ({confidence:.1%} < {self.MIN_CONFIDENCE:.1%})")
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"CONFIDENCE_TOO_LOW: {confidence:.1%} < {self.MIN_CONFIDENCE:.1%}",
                entry_type="NO_SIGNAL",
                session_window=session.value,
                metadata=metadata
            )
        
        # =====================================================================
        # ALL GATES PASSED - GENERATE SIGNAL
        # =====================================================================
        
        self.signals_generated += 1
        
        # Build reason string with tags
        reason_parts = [
            f"[EVENING_CONTINUATION]",
            f"[{pattern.pattern_type}]",
        ]
        
        if pattern.volume_confirmed:
            reason_parts.append("VOLUME_OK")
        
        reason_parts.append(f"RSI={rsi:.0f}")
        reason_parts.append(f"ADX={adx:.0f}")
        reason_parts.append(f"CONF={confidence:.0%}")
        
        logger.info(
            f"[EVENING_CONTINUATION] SIGNAL GENERATED: "
            f"{pattern.pattern_type} | confidence={confidence:.1%} | "
            f"stop={price - stop_loss:.2f}pts | "
            f"session={session.value}"
        )
        
        return EntrySignal(
            action="BUY",
            confidence=confidence,
            reason=" | ".join(reason_parts),
            stop_loss=stop_loss,
            take_profit=final_target,  # Use final target as primary
            entry_type=f"EVENING_{pattern.pattern_type}",
            session_window=session.value,
            metadata=metadata
        )


# =============================================================================
# EVENING SELL CONTINUATION MODULE - VERY STRICT LATE RTH SELL CONTINUATION
# =============================================================================

@dataclass
class EveningSellPatternAnalysis:
    """Analysis result for evening SELL continuation patterns.
    
    Only two patterns are valid:
    - BREAKDOWN_HOLD: Price breaks below level, next candle fails to reclaim
    - BEAR_FLAG: Shallow retrace of impulse down, breakdown continues
    """
    is_valid: bool = False
    pattern_type: str = ""  # "BREAKDOWN_HOLD", "BEAR_FLAG", "NONE"
    score: float = 0.0
    breakdown_level: float = 0.0
    flag_depth_pct: float = 0.0
    volume_confirmed: bool = False
    reasons: List[str] = field(default_factory=list)


class EveningSellContinuationModule:
    """EVENING SELL continuation module - VERY STRICT late RTH sell continuation.
    
    ═══════════════════════════════════════════════════════════════════════════
    PURPOSE:
    Capture late-day SELL continuation ONLY AFTER FAILED ACCEPTANCE.
    This is NOT a fade-the-high module - it requires confirmed structural 
    breakdown AFTER acceptance has failed.
    
    WHY THIS MODULE IS VERY STRICT:
    - Late-day SELL continuation is high-risk (short covering, squeezes)
    - Most "exhaustion" signals in strong trends are false signals
    - Only trade AFTER structure has already broken down
    - Designed to be RARE - maybe 1-2 signals per week maximum
    
    ═══════════════════════════════════════════════════════════════════════════
    PRECONDITIONS (ALL REQUIRED - no exceptions):
    
    1. PHASE REQUIREMENT:
       - market_phase == EXHAUSTION_DOWN (NOT EXHAUSTION_UP)
       - phase_age >= 3 bars (avoid first-bar fake exhaustion)
       - phase_confirmed == True
    
    2. ACCEPTANCE MUST BE GONE:
       - is_acceptance == False
       - ema_stack_up == False
       - EMA9 < EMA21 < EMA50 (OR EMA flattening + breakdown)
    
    3. MOMENTUM REQUIREMENTS:
       - ADX >= 28 (trend strength)
       - MACD <= -0.20 (bearish momentum)
       - RSI in range [35–50] (NOT oversold - short covering risk)
       - trend_score <= -60 (confirmed bearish)
       - VWAP slope <= 0 (downward pressure)
    
    ═══════════════════════════════════════════════════════════════════════════
    SESSION CONSTRAINTS:
    - Valid ONLY during AFTERNOON (14:00–15:00 CST)
    - Hard cutoff: NO entries after 15:30 CST
    - DISABLED during MORNING and MIDDAY
    
    ═══════════════════════════════════════════════════════════════════════════
    ENTRY PATTERNS (ONE required):
    
    TYPE A - BREAKDOWN_HOLD:
        - Breakdown below VWAP / EMA21 / value low
        - Next candle FAILS to reclaim level
        - Close in bottom 30% of range
        - Volume >= session median
    
    TYPE B - BEAR_FLAG:
        - Flag retrace <= 30% of impulse down
        - Retrace holds below VWAP
        - No bullish engulfing candles
        - Breakdown candle with strong bearish close
    
    ═══════════════════════════════════════════════════════════════════════════
    HARD BLOCKS (ABSOLUTE - will NEVER be relaxed):
    
    1. RSI < 32 (short-covering risk - too oversold)
    2. Price extended > 1.8 ATR below VWAP (stretched too far)
    3. Bullish divergence detected (MACD rising while price falling)
    4. phase_age > 20 bars (late exhaustion decay - reversal likely)
    5. Any acceptance signal reappears (EMA stack reforms)
    6. Confidence < 68%
    
    ═══════════════════════════════════════════════════════════════════════════
    RISK MANAGEMENT:
    - Maximum stop: 2.5 points
    - Stop: Above flag high or VWAP
    - Target 1: 1.5R (mandatory partial)
    - Target 2: 2.0R–2.5R based on impulse strength
    - Minimum R:R = 2.0
    
    ═══════════════════════════════════════════════════════════════════════════
    Author: Trading System Enhancement - Jan 27, 2026
    """
    
    # =========================================================================
    # CONFIGURATION CONSTANTS - DO NOT RELAX THESE THRESHOLDS
    # =========================================================================
    
    # Session timing (CST) - STRICTER than buy evening module
    EVENING_START = time(14, 0)   # 14:00 CST - AFTERNOON starts
    EVENING_END = time(15, 30)    # 15:30 CST - EARLIER cutoff than BUY (15:40)
    
    # Phase requirements
    MIN_PHASE_AGE = 3             # Bars in EXHAUSTION_DOWN before entry allowed
    MAX_PHASE_AGE = 20            # After 20 bars, exhaustion is decaying
    
    # RSI requirements - STRICT zone, avoid short-covering
    RSI_MIN = 35                  # Below this = short-covering risk
    RSI_MAX = 50                  # Above this = not enough momentum
    RSI_OVERSOLD_BLOCK = 32       # HARD BLOCK below this
    
    # Momentum requirements - STRONGER than morning modules
    ADX_MIN = 28
    MACD_MAX = -0.20              # Must be negative (bearish)
    TREND_SCORE_MAX = -60         # Must be bearish
    
    # VWAP requirements
    MAX_VWAP_EXTENSION_ATR = 1.8  # Can't be too far below VWAP
    
    # Pattern requirements
    MAX_FLAG_DEPTH_PCT = 0.30     # Bear flag can't retrace more than 30%
    
    # Risk management
    MAX_STOP_DISTANCE = 2.5       # Points
    MIN_STOP_DISTANCE = 1.25      # Points
    PARTIAL_TARGET_RR = 1.5
    FINAL_TARGET_MIN_RR = 2.0
    FINAL_TARGET_MAX_RR = 2.5
    
    # Confidence
    BASE_CONFIDENCE = 0.60
    MIN_CONFIDENCE = 0.68         # HIGHER than BUY evening module (0.65)
    MAX_CONFIDENCE = 0.75         # LOWER cap than BUY evening module (0.78)
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize evening SELL continuation module."""
        config = config or {}
        
        self.rsi_min = config.get("rsi_min", self.RSI_MIN)
        self.rsi_max = config.get("rsi_max", self.RSI_MAX)
        self.adx_min = config.get("adx_min", self.ADX_MIN)
        self.macd_max = config.get("macd_max", self.MACD_MAX)
        self.trend_score_max = config.get("trend_score_max", self.TREND_SCORE_MAX)
        self.min_phase_age = config.get("min_phase_age", self.MIN_PHASE_AGE)
        self.max_phase_age = config.get("max_phase_age", self.MAX_PHASE_AGE)
        
        # Statistics
        self.signals_generated = 0
        self.signals_blocked = 0
        
        logger.info(
            f"[EVENING_SELL_CONTINUATION] Module initialized: "
            f"RSI=[{self.rsi_min}-{self.rsi_max}], "
            f"ADX>={self.adx_min}, MACD<={self.macd_max}, "
            f"trend_score<={self.trend_score_max}, "
            f"phase_age=[{self.min_phase_age}-{self.max_phase_age}]"
        )
    
    def _is_valid_session(self, timestamp: Optional[datetime]) -> Tuple[bool, str]:
        """Check if current time is valid for evening SELL continuation.
        
        Returns:
            Tuple of (is_valid, reason)
        """
        if timestamp is None:
            return False, "NO_TIMESTAMP"
        
        current_time = timestamp.time()
        
        # Must be in AFTERNOON session only (more restrictive than BUY)
        if current_time < self.EVENING_START:
            return False, f"TOO_EARLY: {current_time} < 14:00 CST"
        
        if current_time >= self.EVENING_END:
            return False, f"CUTOFF_REACHED: {current_time} >= 15:30 CST"
        
        return True, "VALID_AFTERNOON_SESSION"
    
    def _detect_bullish_divergence(
        self,
        price: float,
        macd_hist: float,
        recent_bars: Optional[pd.DataFrame]
    ) -> bool:
        """Detect bullish divergence (price making lower lows, MACD making higher lows).
        
        WHY THIS BLOCKS ENTRY:
        Bullish divergence suggests selling pressure is weakening.
        Entering a SELL here has high probability of failure.
        """
        if recent_bars is None or len(recent_bars) < 5:
            return False
        
        # Compare current vs 5 bars ago
        recent_lows = recent_bars['low'].tail(5).values
        
        # Price making lower lows?
        price_lower_low = price < recent_lows.min()
        
        # For MACD divergence, we'd need historical MACD values
        # For now, use the MACD slope as a proxy
        # If MACD is rising while price is falling = bullish divergence
        if len(recent_bars) >= 3:
            macd_cols = ['MACD_hist', 'macd_hist']
            macd_col = None
            for col in macd_cols:
                if col in recent_bars.columns:
                    macd_col = col
                    break
            
            if macd_col:
                recent_macd = recent_bars[macd_col].tail(3).values
                macd_rising = recent_macd[-1] > recent_macd[0]  # Current > 3 bars ago
                
                if price_lower_low and macd_rising:
                    return True
        
        return False
    
    def _analyze_breakdown_hold(
        self,
        price: float,
        open_price: float,
        high: float,
        low: float,
        ema21: float,
        vwap: float,
        pdl: float,
        recent_bars: Optional[pd.DataFrame],
        volume: float,
        session_median_volume: float
    ) -> EveningSellPatternAnalysis:
        """Analyze for Breakdown-and-Hold continuation pattern.
        
        TYPE A PATTERN:
        - Price broke below a key level (EMA21, VWAP, PDL)
        - Current candle FAILS to reclaim level (stays below)
        - Close is in bottom 30% of candle range
        - Volume confirms
        
        WHY THIS WORKS:
        - Failed reclaim = sellers still in control
        - Bottom 30% close = bears winning the candle
        - Volume = real participation in breakdown
        """
        result = EveningSellPatternAnalysis()
        result.reasons = []
        
        # Need recent bars
        if recent_bars is None or len(recent_bars) < 3:
            result.reasons.append("INSUFFICIENT_BARS")
            return result
        
        # Identify breakdown levels (must be below all)
        breakdown_levels = []
        
        # Level 1: EMA21 (most important for trend)
        if ema21 > 0:
            breakdown_levels.append(("EMA21", ema21))
        
        # Level 2: VWAP
        if vwap > 0:
            breakdown_levels.append(("VWAP", vwap))
        
        # Level 3: PDL (significant support)
        if pdl > 0:
            breakdown_levels.append(("PDL", pdl))
        
        if not breakdown_levels:
            result.reasons.append("NO_BREAKDOWN_LEVELS")
            return result
        
        # Check for breakdown-and-hold pattern
        pattern_score = 0.0
        best_level_name = ""
        best_level_price = 0.0
        
        for level_name, level_price in breakdown_levels:
            # Are we BELOW this level?
            if price < level_price:
                # Did we fail to reclaim? (high didn't reach back above)
                if high < level_price:
                    level_score = 30.0  # Base score for breakdown-hold
                    
                    if level_name == "EMA21":
                        level_score += 15.0
                        result.reasons.append("BELOW_EMA21")
                    elif level_name == "VWAP":
                        level_score += 12.0
                        result.reasons.append("BELOW_VWAP")
                    elif level_name == "PDL":
                        level_score += 18.0  # PDL is very significant
                        result.reasons.append("BELOW_PDL")
                    
                    if level_score > pattern_score:
                        pattern_score = level_score
                        best_level_name = level_name
                        best_level_price = level_price
        
        if pattern_score == 0:
            result.reasons.append("NO_BREAKDOWN_HOLD")
            return result
        
        # Check close position in candle (must be bottom 30%)
        candle_range = high - low if high > low else 0.01
        close_position = (price - low) / candle_range
        
        if close_position > 0.30:  # Not in bottom 30%
            result.reasons.append(f"WEAK_CLOSE_POSITION: {close_position:.1%} (need <= 30%)")
            return result
        
        pattern_score += 15.0
        result.reasons.append(f"STRONG_BEARISH_CLOSE: bottom {close_position:.0%}")
        
        # Check volume
        if session_median_volume > 0 and volume >= session_median_volume:
            pattern_score += 12.0
            result.volume_confirmed = True
            result.reasons.append("VOLUME_CONFIRMED")
        
        # Bearish candle (close < open)
        if price < open_price:
            pattern_score += 8.0
            result.reasons.append("BEARISH_CANDLE")
        else:
            result.reasons.append("DOJI_OR_BULLISH_WARNING")
            pattern_score -= 10.0
        
        # Validate score
        if pattern_score >= 50.0:
            result.is_valid = True
            result.pattern_type = "BREAKDOWN_HOLD"
            result.score = pattern_score
            result.breakdown_level = best_level_price
        
        return result
    
    def _analyze_bear_flag(
        self,
        price: float,
        open_price: float,
        high: float,
        low: float,
        vwap: float,
        atr: float,
        recent_bars: Optional[pd.DataFrame]
    ) -> EveningSellPatternAnalysis:
        """Analyze for Bear Flag continuation pattern.
        
        TYPE B PATTERN:
        - Recent impulse down created swing low
        - Current bar is a retrace that:
          - Does NOT exceed 30% of impulse
          - Holds below VWAP
          - Has no bullish engulfing candles
        - Current candle shows bearish continuation (breakdown)
        
        WHY THIS WORKS:
        - Shallow retrace = sellers eager to continue
        - Below VWAP = institutional selling pressure
        - Breakdown from flag = continuation confirmed
        """
        result = EveningSellPatternAnalysis()
        result.reasons = []
        
        if recent_bars is None or len(recent_bars) < 5:
            result.reasons.append("INSUFFICIENT_BARS")
            return result
        
        # Find impulse (lowest low in recent bars)
        impulse_low = recent_bars['low'].tail(10).min() if len(recent_bars) >= 10 else low
        impulse_high = recent_bars['high'].tail(10).max() if len(recent_bars) >= 10 else high
        impulse_range = impulse_high - impulse_low if impulse_high > impulse_low else 1.0
        
        # Calculate retrace depth
        retrace_high = recent_bars['high'].tail(3).max()  # Recent 3 bars high
        retrace_from_low = retrace_high - impulse_low
        retrace_depth_pct = retrace_from_low / impulse_range if impulse_range > 0 else 1.0
        
        result.flag_depth_pct = retrace_depth_pct
        
        # HARD CHECK: Retrace must be <= 30%
        if retrace_depth_pct > self.MAX_FLAG_DEPTH_PCT:
            result.reasons.append(f"RETRACE_TOO_DEEP: {retrace_depth_pct:.1%} > 30%")
            return result
        
        result.reasons.append(f"SHALLOW_RETRACE: {retrace_depth_pct:.1%}")
        pattern_score = 25.0
        
        # HARD CHECK: Must hold below VWAP
        if price > vwap:
            result.reasons.append("ABOVE_VWAP: Flag broken")
            return result
        
        result.reasons.append("BELOW_VWAP")
        pattern_score += 15.0
        
        # Check for bullish engulfing in recent bars (would invalidate)
        for i in range(min(3, len(recent_bars) - 1)):
            idx = -(i + 1)
            bar = recent_bars.iloc[idx]
            prev_bar = recent_bars.iloc[idx - 1] if abs(idx - 1) <= len(recent_bars) else bar
            
            # Bullish engulfing: current close > prev open, current open < prev close
            bar_close = bar.get('close', bar.get('price', 0))
            bar_open = bar.get('open', bar_close)
            prev_close = prev_bar.get('close', prev_bar.get('price', 0))
            prev_open = prev_bar.get('open', prev_close)
            
            if bar_close > prev_open and bar_open < prev_close and bar_close > bar_open:
                result.reasons.append("BULLISH_ENGULFING_DETECTED: Flag invalidated")
                return result
        
        pattern_score += 10.0
        result.reasons.append("NO_BULLISH_ENGULFING")
        
        # Current candle should be bearish breakdown
        if price < open_price:
            pattern_score += 12.0
            result.reasons.append("BEARISH_BREAKDOWN_CANDLE")
        else:
            result.reasons.append("NO_BREAKDOWN_CANDLE")
            return result
        
        # Close in lower half of range
        candle_range = high - low if high > low else 0.01
        close_position = (price - low) / candle_range
        
        if close_position <= 0.40:
            pattern_score += 10.0
            result.reasons.append("STRONG_BEARISH_CLOSE")
        
        if pattern_score >= 50.0:
            result.is_valid = True
            result.pattern_type = "BEAR_FLAG"
            result.score = pattern_score
        
        return result
    
    def _calculate_stops_and_targets(
        self,
        price: float,
        pattern: EveningSellPatternAnalysis,
        vwap: float,
        atr: float,
        high: float,
        recent_bars: Optional[pd.DataFrame]
    ) -> Tuple[float, float, float, bool, str]:
        """Calculate stop loss and targets for evening SELL entry.
        
        RISK RULES:
        - Max stop: 2.5 points (TIGHTER than BUY evening)
        - Stop: Above flag high or VWAP
        - Partial at 1.5R, final at 2.0R-2.5R
        """
        # Determine stop location
        stop_candidates = []
        
        if pattern.pattern_type == "BREAKDOWN_HOLD":
            # Stop above breakdown level
            stop_candidates.append(pattern.breakdown_level + atr * 0.3)
            stop_candidates.append(high + atr * 0.25)
        else:  # BEAR_FLAG
            # Stop above flag high
            if recent_bars is not None and len(recent_bars) >= 3:
                flag_high = recent_bars['high'].tail(3).max()
                stop_candidates.append(flag_high + atr * 0.2)
            stop_candidates.append(high + atr * 0.25)
        
        # Also consider VWAP as stop reference
        if vwap > price:
            stop_candidates.append(vwap + atr * 0.2)
        
        # Use closest stop (tightest that's still valid)
        stop_loss = min(stop_candidates) if stop_candidates else high + atr * 0.3
        
        # Calculate risk
        risk = stop_loss - price
        
        # HARD CHECK: Max stop distance
        if risk > self.MAX_STOP_DISTANCE:
            return 0, 0, 0, False, f"STOP_TOO_WIDE: {risk:.2f}pts > {self.MAX_STOP_DISTANCE}pts"
        
        # Enforce minimum
        if risk < self.MIN_STOP_DISTANCE:
            stop_loss = price + self.MIN_STOP_DISTANCE
            risk = self.MIN_STOP_DISTANCE
        
        # Calculate targets (SELL = targets below entry)
        partial_target = price - (risk * self.PARTIAL_TARGET_RR)
        
        # Final target based on pattern strength
        if pattern.score >= 70:
            final_rr = self.FINAL_TARGET_MAX_RR
        else:
            final_rr = self.FINAL_TARGET_MIN_RR
        
        final_target = price - (risk * final_rr)
        
        reason = f"STOP={risk:.2f}pts, PARTIAL=1.5R, FINAL={final_rr}R"
        
        return stop_loss, partial_target, final_target, True, reason
    
    def _calculate_confidence(
        self,
        pattern: EveningSellPatternAnalysis,
        market_state: MarketStateResult,
        rsi: float,
        adx: float,
        macd_hist: float
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate confidence for evening SELL entry.
        
        BASE: 60%
        
        BONUSES:
        - Clean phase transition: +8%
        - Strong breakdown structure: +8%
        - Volume expansion: +6%
        - ADX > 32: +6%
        
        CAP: 75% (LOWER than BUY evening - sells are harder)
        """
        confidence = self.BASE_CONFIDENCE
        breakdown = {"base": self.BASE_CONFIDENCE}
        
        # Phase transition bonus (clean exhaustion)
        if market_state.phase_confirmed:
            confidence += 0.08
            breakdown["phase_confirmed"] = 0.08
        
        # Breakdown structure bonus
        if pattern.pattern_type == "BREAKDOWN_HOLD":
            confidence += 0.08
            breakdown["breakdown_hold"] = 0.08
        elif pattern.pattern_type == "BEAR_FLAG":
            confidence += 0.06
            breakdown["bear_flag"] = 0.06
        
        # Volume bonus
        if pattern.volume_confirmed:
            confidence += 0.06
            breakdown["volume"] = 0.06
        
        # ADX bonus
        if adx >= 32:
            confidence += 0.06
            breakdown["adx_32+"] = 0.06
        elif adx >= self.adx_min:
            confidence += 0.03
            breakdown["adx_28+"] = 0.03
        
        # RSI in ideal zone bonus
        if 38 <= rsi <= 45:
            confidence += 0.04
            breakdown["rsi_ideal"] = 0.04
        
        # Strong MACD bonus
        if macd_hist <= -0.30:
            confidence += 0.04
            breakdown["macd_strong"] = 0.04
        
        # Cap confidence
        final_confidence = min(self.MAX_CONFIDENCE, confidence)
        breakdown["final"] = final_confidence
        
        return final_confidence, breakdown
    
    def evaluate(
        self,
        market_state: MarketStateResult,
        data: Dict[str, Any],
        recent_bars: Optional[pd.DataFrame] = None,
        timestamp: Optional[datetime] = None
    ) -> EntrySignal:
        """Evaluate for evening SELL continuation entry.
        
        This method implements VERY STRICT filtering. Entry is allowed ONLY when
        ALL preconditions pass AND a valid pattern is detected.
        
        CRITICAL: This module can ONLY trigger during EXHAUSTION_DOWN phase
        that has been confirmed for at least 3 bars.
        
        Args:
            market_state: Current market state (must be EXHAUSTION_DOWN)
            data: Current bar data with indicators
            recent_bars: Recent price history
            timestamp: Current timestamp
            
        Returns:
            EntrySignal with SELL or HOLD (never BUY)
        """
        metadata = {
            "module": "EVENING_SELL_CONTINUATION",
            "timestamp": str(timestamp) if timestamp else "UNKNOWN",
        }
        
        # =====================================================================
        # GATE 1: SESSION VALIDATION (AFTERNOON only, before 15:30)
        # =====================================================================
        session_valid, session_reason = self._is_valid_session(timestamp)
        
        if not session_valid:
            self.signals_blocked += 1
            logger.debug(f"[EVENING_SELL_CONTINUATION] BLOCKED: {session_reason}")
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"SESSION_INVALID: {session_reason}",
                entry_type="NO_SIGNAL",
                session_window="AFTERNOON",
                metadata=metadata
            )
        
        # =====================================================================
        # GATE 2: PHASE REQUIREMENT (MUST be EXHAUSTION_DOWN)
        # =====================================================================
        
        # HARD REQUIREMENT: Must be in EXHAUSTION_DOWN phase
        if market_state.phase != MarketPhase.EXHAUSTION_DOWN:
            self.signals_blocked += 1
            logger.debug(
                f"[EVENING_SELL_CONTINUATION] BLOCKED: WRONG_PHASE "
                f"(need EXHAUSTION_DOWN, got {market_state.phase.value})"
            )
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"WRONG_PHASE: need EXHAUSTION_DOWN, got {market_state.phase.value}",
                entry_type="NO_SIGNAL",
                session_window="AFTERNOON",
                metadata=metadata
            )
        
        # Check phase age (must be confirmed, not too old)
        if market_state.phase_age < self.min_phase_age:
            self.signals_blocked += 1
            logger.debug(
                f"[EVENING_SELL_CONTINUATION] BLOCKED: PHASE_TOO_YOUNG "
                f"({market_state.phase_age} < {self.min_phase_age})"
            )
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"PHASE_TOO_YOUNG: {market_state.phase_age} bars < {self.min_phase_age}",
                entry_type="NO_SIGNAL",
                session_window="AFTERNOON",
                metadata=metadata
            )
        
        if market_state.phase_age > self.max_phase_age:
            self.signals_blocked += 1
            logger.debug(
                f"[EVENING_SELL_CONTINUATION] BLOCKED: PHASE_TOO_OLD "
                f"({market_state.phase_age} > {self.max_phase_age})"
            )
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"PHASE_DECAY: {market_state.phase_age} bars > {self.max_phase_age} (reversal likely)",
                entry_type="NO_SIGNAL",
                session_window="AFTERNOON",
                metadata=metadata
            )
        
        # =====================================================================
        # GATE 3: ACCEPTANCE MUST BE GONE
        # =====================================================================
        
        if market_state.is_acceptance:
            self.signals_blocked += 1
            logger.debug("[EVENING_SELL_CONTINUATION] BLOCKED: ACCEPTANCE_PRESENT")
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason="ACCEPTANCE_PRESENT: Cannot SELL during acceptance",
                entry_type="NO_SIGNAL",
                session_window="AFTERNOON",
                metadata=metadata
            )
        
        if market_state.ema_stack_up:
            self.signals_blocked += 1
            logger.debug("[EVENING_SELL_CONTINUATION] BLOCKED: EMA_STACK_UP")
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason="EMA_STACK_UP: Bullish structure present",
                entry_type="NO_SIGNAL",
                session_window="AFTERNOON",
                metadata=metadata
            )
        
        # =====================================================================
        # EXTRACT PRICE DATA
        # =====================================================================
        price = float(data.get("close", data.get("price", 0)))
        open_price = float(data.get("open", price))
        high = float(data.get("high", price))
        low = float(data.get("low", price))
        
        ema9 = float(data.get("ema_9", data.get("EMA_9", price)))
        ema21 = float(data.get("ema_21", data.get("EMA_21", price)))
        ema50 = float(data.get("ema_50", data.get("EMA_50", price)))
        vwap = float(data.get("vwap", data.get("SESSION_VWAP", price)))
        atr = float(data.get("atr", data.get("ATR_14", 1.0)))
        adx = float(data.get("adx", data.get("ADX_14", 20)))
        macd_hist = float(data.get("macd_hist", data.get("MACD_hist", 0)))
        
        pdl = float(data.get("pdl", data.get("PDL", 0)))
        
        volume = float(data.get("volume", data.get("VOLUME", 0)))
        session_median_volume = float(data.get("session_median_volume", volume * 0.8))
        
        rsi = market_state.rsi_value
        trend_score = market_state.trend_score
        vwap_slope = market_state.vwap_slope
        
        metadata.update({
            "price": price,
            "rsi": rsi,
            "adx": adx,
            "macd_hist": macd_hist,
            "trend_score": trend_score,
            "phase_age": market_state.phase_age,
        })
        
        # =====================================================================
        # GATE 4: EMA STRUCTURE (must be bearish or flattening + breakdown)
        # =====================================================================
        
        # Prefer bearish EMA stack, but allow flattening with breakdown
        ema_bearish = market_state.ema_stack_down
        ema_breaking_down = (
            not market_state.ema_stack_up and
            price < ema9 < ema21 and  # At least breaking
            market_state.ema_flattening
        )
        
        if not ema_bearish and not ema_breaking_down:
            self.signals_blocked += 1
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason="EMA_STRUCTURE_INVALID: Need bearish stack or breakdown",
                entry_type="NO_SIGNAL",
                session_window="AFTERNOON",
                metadata=metadata
            )
        
        # =====================================================================
        # GATE 5: VWAP SLOPE (must be negative or flat)
        # =====================================================================
        
        if vwap_slope > 0.001:  # Allow tiny positive slope
            self.signals_blocked += 1
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"VWAP_SLOPE_POSITIVE: {vwap_slope:.4f} > 0",
                entry_type="NO_SIGNAL",
                session_window="AFTERNOON",
                metadata=metadata
            )
        
        # =====================================================================
        # GATE 6: MOMENTUM REQUIREMENTS
        # =====================================================================
        
        # ADX check
        if adx < self.adx_min:
            self.signals_blocked += 1
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"ADX_TOO_LOW: {adx:.1f} < {self.adx_min}",
                entry_type="NO_SIGNAL",
                session_window="AFTERNOON",
                metadata=metadata
            )
        
        # MACD must be negative
        if macd_hist > self.macd_max:
            self.signals_blocked += 1
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"MACD_NOT_BEARISH: {macd_hist:.2f} > {self.macd_max}",
                entry_type="NO_SIGNAL",
                session_window="AFTERNOON",
                metadata=metadata
            )
        
        # Trend score must be bearish
        if trend_score > self.trend_score_max:
            self.signals_blocked += 1
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"TREND_NOT_BEARISH: {trend_score:.0f} > {self.trend_score_max}",
                entry_type="NO_SIGNAL",
                session_window="AFTERNOON",
                metadata=metadata
            )
        
        # =====================================================================
        # GATE 7: RSI REQUIREMENTS (STRICT - avoid short covering)
        # =====================================================================
        
        # HARD BLOCK: Too oversold
        if rsi < self.RSI_OVERSOLD_BLOCK:
            self.signals_blocked += 1
            logger.debug(
                f"[EVENING_SELL_CONTINUATION] BLOCKED: RSI_OVERSOLD "
                f"({rsi:.1f} < {self.RSI_OVERSOLD_BLOCK}) - short covering risk"
            )
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"RSI_OVERSOLD: {rsi:.1f} < {self.RSI_OVERSOLD_BLOCK} (short covering risk)",
                entry_type="NO_SIGNAL",
                session_window="AFTERNOON",
                metadata=metadata
            )
        
        # RSI zone check
        if rsi < self.rsi_min or rsi > self.rsi_max:
            self.signals_blocked += 1
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"RSI_OUT_OF_ZONE: {rsi:.1f} not in [{self.rsi_min}-{self.rsi_max}]",
                entry_type="NO_SIGNAL",
                session_window="AFTERNOON",
                metadata=metadata
            )
        
        # =====================================================================
        # GATE 8: VWAP EXTENSION CHECK
        # =====================================================================
        
        if vwap > 0 and atr > 0:
            vwap_dist_atr = (vwap - price) / atr  # Positive = price below VWAP
            if vwap_dist_atr > self.MAX_VWAP_EXTENSION_ATR:
                self.signals_blocked += 1
                return EntrySignal(
                    action="HOLD",
                    confidence=0.0,
                    reason=f"TOO_EXTENDED: {vwap_dist_atr:.1f} ATR below VWAP > {self.MAX_VWAP_EXTENSION_ATR}",
                    entry_type="NO_SIGNAL",
                    session_window="AFTERNOON",
                    metadata=metadata
                )
        
        # =====================================================================
        # GATE 9: BULLISH DIVERGENCE CHECK
        # =====================================================================
        
        if self._detect_bullish_divergence(price, macd_hist, recent_bars):
            self.signals_blocked += 1
            logger.debug("[EVENING_SELL_CONTINUATION] BLOCKED: BULLISH_DIVERGENCE")
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason="BULLISH_DIVERGENCE: MACD rising while price falling",
                entry_type="NO_SIGNAL",
                session_window="AFTERNOON",
                metadata=metadata
            )
        
        # =====================================================================
        # GATE 10: PATTERN DETECTION
        # =====================================================================
        
        # Try Breakdown-and-Hold pattern
        breakdown = self._analyze_breakdown_hold(
            price=price,
            open_price=open_price,
            high=high,
            low=low,
            ema21=ema21,
            vwap=vwap,
            pdl=pdl,
            recent_bars=recent_bars,
            volume=volume,
            session_median_volume=session_median_volume
        )
        
        # Try Bear Flag pattern
        bear_flag = self._analyze_bear_flag(
            price=price,
            open_price=open_price,
            high=high,
            low=low,
            vwap=vwap,
            atr=atr,
            recent_bars=recent_bars
        )
        
        # Select best pattern
        pattern: Optional[EveningSellPatternAnalysis] = None
        
        if breakdown.is_valid and bear_flag.is_valid:
            pattern = breakdown if breakdown.score >= bear_flag.score else bear_flag
        elif breakdown.is_valid:
            pattern = breakdown
        elif bear_flag.is_valid:
            pattern = bear_flag
        
        if pattern is None:
            self.signals_blocked += 1
            reject_reasons = breakdown.reasons + bear_flag.reasons
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"NO_VALID_PATTERN: {', '.join(reject_reasons[:3])}",
                entry_type="NO_SIGNAL",
                session_window="AFTERNOON",
                metadata=metadata
            )
        
        metadata["pattern_type"] = pattern.pattern_type
        metadata["pattern_score"] = pattern.score
        metadata["pattern_reasons"] = pattern.reasons
        
        # =====================================================================
        # GATE 11: RISK VALIDATION
        # =====================================================================
        
        stop_loss, partial_target, final_target, risk_valid, risk_reason = self._calculate_stops_and_targets(
            price=price,
            pattern=pattern,
            vwap=vwap,
            atr=atr,
            high=high,
            recent_bars=recent_bars
        )
        
        if not risk_valid:
            self.signals_blocked += 1
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=risk_reason,
                entry_type="NO_SIGNAL",
                session_window="AFTERNOON",
                metadata=metadata
            )
        
        metadata["stop_loss"] = stop_loss
        metadata["partial_target"] = partial_target
        metadata["final_target"] = final_target
        
        # =====================================================================
        # GATE 12: CONFIDENCE VALIDATION
        # =====================================================================
        
        confidence, confidence_breakdown = self._calculate_confidence(
            pattern=pattern,
            market_state=market_state,
            rsi=rsi,
            adx=adx,
            macd_hist=macd_hist
        )
        
        metadata["confidence_breakdown"] = confidence_breakdown
        
        if confidence < self.MIN_CONFIDENCE:
            self.signals_blocked += 1
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"CONFIDENCE_TOO_LOW: {confidence:.1%} < {self.MIN_CONFIDENCE:.1%}",
                entry_type="NO_SIGNAL",
                session_window="AFTERNOON",
                metadata=metadata
            )
        
        # =====================================================================
        # ALL GATES PASSED - GENERATE SELL SIGNAL
        # =====================================================================
        
        self.signals_generated += 1
        
        reason_parts = [
            "[EVENING_SELL_CONTINUATION]",
            f"[{pattern.pattern_type}]",
        ]
        
        if pattern.volume_confirmed:
            reason_parts.append("VOLUME_OK")
        
        reason_parts.append(f"RSI={rsi:.0f}")
        reason_parts.append(f"phase_age={market_state.phase_age}")
        reason_parts.append(f"CONF={confidence:.0%}")
        
        logger.info(
            f"[EVENING_SELL_CONTINUATION] SIGNAL GENERATED: "
            f"{pattern.pattern_type} | confidence={confidence:.1%} | "
            f"stop={stop_loss - price:.2f}pts | "
            f"phase_age={market_state.phase_age}"
        )
        
        return EntrySignal(
            action="SELL",
            confidence=confidence,
            reason=" | ".join(reason_parts),
            stop_loss=stop_loss,
            take_profit=final_target,
            entry_type=f"EVENING_SELL_{pattern.pattern_type}",
            session_window="AFTERNOON",
            metadata=metadata
        )


class IntegratedEntryManager:
    """Manages all entry modules: continuation, evening, range reversion, and exhaustion.
    
    This class provides a unified interface for the MES strategy to:
    1. Evaluate market state (acceptance vs exhaustion)
    2. Generate appropriate entry signals
    3. Ensure hard blocks are respected
    4. Maintain trade frequency while fixing bias
    
    PRIORITY ORDER (by session context):
    1. BuyContinuationModule (morning/prime continuation - HIGHEST PRIORITY during AM)
    2. ShortContinuationModule (mirror continuation on downtrends)
    3. EveningContinuationModule (late RTH BUY momentum - AFTERNOON/CLOSE sessions)
    4. EveningSellContinuationModule (late RTH SELL continuation - ONLY in EXHAUSTION_DOWN)
    5. RangeReversionModule (mean reversion on CHOP/RANGE days - ONLY if all above return NO_SIGNAL)
    6. SellExhaustionModule (reversal on confirmed exhaustion - LOWEST PRIORITY)
    
    RangeReversionModule ONLY fires when:
    - ALL continuation modules return NO_SIGNAL
    - Market is in CHOP/RANGE regime (ADX < 22, no trend)
    - Time is 09:45-14:30 CST
    - is_acceptance == False AND is_exhaustion == False
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize integrated entry manager."""
        config = config or {}
        
        # Create submodules (IN PRIORITY ORDER)
        self.state_detector = create_market_state_detector(config.get("state_config"))
        self.buy_module = BuyContinuationModule(config.get("buy_config"))
        self.short_module = ShortContinuationModule(config.get("short_config"))
        self.evening_module = EveningContinuationModule(config.get("evening_config"))
        self.evening_sell_module = EveningSellContinuationModule(config.get("evening_sell_config"))
        
        # Import and create RangeReversionModule
        try:
            from mytrader.strategies.range_reversion_module import RangeReversionModule
            self.range_reversion_module = RangeReversionModule(config.get("range_config"))
            self._has_range_module = True
            logger.info("RangeReversionModule loaded successfully")
        except ImportError as e:
            logger.warning(f"RangeReversionModule not available: {e}")
            self.range_reversion_module = None
            self._has_range_module = False
        
        self.sell_module = SellExhaustionModule(config.get("sell_config"))
        
        # Tracking
        self.last_signal: Optional[EntrySignal] = None
        self.consecutive_holds = 0
        self.max_consecutive_holds = config.get("max_consecutive_holds", 50)
        
        # DAILY_TREND_CONFIRMATION gate tracking
        # Once gate fails for a session, BUY_CONTINUATION is blocked
        self._daily_trend_gate_evaluated = False
        self._daily_trend_confirmed = True  # Start True, set False if gate fails
        self._daily_trend_gate_session_date: Optional[str] = None
        self._daily_trend_gate_details: Dict[str, Any] = {}

        # SHORT daily trend gate tracking
        self._daily_trend_gate_short_evaluated = False
        self._daily_trend_short_confirmed = True
        self._daily_trend_gate_short_session_date: Optional[str] = None
        self._daily_trend_gate_short_details: Dict[str, Any] = {}
        
        # Global confidence filter - minimum confidence required for any trade
        self._min_confidence_threshold = config.get("min_confidence_threshold", 0.78)
        
        # ONE-LOSS-PER-DIRECTION rule tracking
        # If a trade is stopped out, block further trades in that direction for the session
        self._session_stopout: Dict[str, bool] = {"BUY": False, "SELL": False}
        self._session_stopout_date: Optional[str] = None
        
        # RANGE EXPANSION validation config
        # Continuation trades require range expansion to ensure we're entering during active periods
        self._range_expansion_mult = config.get("range_expansion_mult", 1.2)  # Current range > 1.2 * avg
        self._range_lookback = config.get("range_lookback", 10)  # Average of last 10 candles
        self._atr_increasing_bars = config.get("atr_increasing_bars", 3)  # ATR must increase for N bars
        
        # FAILED TREND DAY guardrail
        # If first continuation trade stops out + ADX < 22 + price in VWAP chop zone
        # -> Disable continuation logic for rest of day, switch to CHOP/RANGE regime
        self._failed_trend_day = False
        self._failed_trend_day_date: Optional[str] = None
        self._first_continuation_stopout = False  # Track if first continuation trade stopped out
        self._failed_trend_adx_threshold = config.get("failed_trend_adx_threshold", 22.0)
        self._vwap_chop_zone_pct = config.get("vwap_chop_zone_pct", 0.001)  # 0.1% of price around VWAP
        
        logger.info(
            f"IntegratedEntryManager initialized with Morning BUY + SHORT + Evening BUY + SELL + "
            f"RangeReversion={'ON' if self._has_range_module else 'OFF'} + "
            f"min_confidence={self._min_confidence_threshold}"
        )
    
    def _apply_confidence_filter(
        self, 
        signal: "EntrySignal",
        module_name: str
    ) -> Tuple["EntrySignal", bool]:
        """
        Apply global confidence filter to a signal.
        
        If confidence < min_confidence_threshold (default 0.78), the signal
        is converted to HOLD and the trade is skipped.
        
        Args:
            signal: The entry signal to filter
            module_name: Name of the module that generated the signal (for logging)
            
        Returns:
            Tuple of (filtered_signal, was_filtered)
            - filtered_signal: Original signal if passed, HOLD signal if filtered
            - was_filtered: True if signal was suppressed due to low confidence
        """
        if not signal.is_actionable:
            return signal, False
        
        if signal.confidence < self._min_confidence_threshold:
            logger.warning(
                f"🚫 LOW_CONFIDENCE_FILTER: {module_name} signal suppressed | "
                f"action={signal.action} confidence={signal.confidence:.2f} < "
                f"threshold={self._min_confidence_threshold} | reason={signal.reason}"
            )
            
            filtered_signal = EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"LOW_CONFIDENCE_FILTER: {module_name} confidence {signal.confidence:.2f} < {self._min_confidence_threshold}",
                entry_type="FILTERED",
                session_window=signal.session_window,
                metadata={
                    "original_action": signal.action,
                    "original_confidence": signal.confidence,
                    "original_reason": signal.reason,
                    "filter_reason": "LOW_CONFIDENCE_FILTER",
                    "module": module_name,
                }
            )
            return filtered_signal, True
        
        return signal, False

    def _attach_entry_metadata(
        self,
        signal: "EntrySignal",
        module_name: str,
        data: Optional[Dict[str, Any]] = None
    ) -> "EntrySignal":
        """
        Attach module attribution to entry metadata.

        Adds `entry_module`, `entry_reason`, and `entry_type` to the signal metadata
        to support downstream analytics of losses by module.
        """
        if signal.metadata is None:
            signal.metadata = {}

        signal.metadata.setdefault("entry_module", module_name)
        signal.metadata.setdefault("entry_reason", signal.reason)
        signal.metadata.setdefault("entry_type", signal.entry_type)

        if data:
            trend_label = data.get("trend_label")
            trend_label_htf = data.get("trend_label_htf")
            if trend_label is not None:
                signal.metadata.setdefault("trend_label", trend_label)
            if trend_label_htf is not None:
                signal.metadata.setdefault("trend_label_htf", trend_label_htf)
        return signal
    
    def record_stopout(
        self, 
        direction: str, 
        timestamp: Optional[datetime] = None,
        entry_type: Optional[str] = None
    ) -> None:
        """
        Record that a trade was stopped out in a given direction.
        
        This implements the ONE-LOSS-PER-DIRECTION rule:
        - If a trade is stopped out (stop_loss exit)
        - Block all further trades in that direction for the session
        
        Also tracks FAILED_TREND_DAY:
        - If first continuation trade stops out, set flag for regime check
        
        Args:
            direction: "BUY" or "SELL" - the direction that was stopped out
            timestamp: Current timestamp (for session tracking)
            entry_type: Type of entry (e.g., "BUY_CONTINUATION", "EVENING_CONTINUATION")
        """
        # Normalize direction
        normalized = direction.upper()
        if "BUY" in normalized or normalized in ("LONG", "SCALP_BUY"):
            dir_key = "BUY"
        elif "SELL" in normalized or normalized in ("SHORT", "SCALP_SELL"):
            dir_key = "SELL"
        else:
            logger.warning(f"Unknown direction for stopout: {direction}")
            return
        
        # Update session date tracking
        session_date = None
        if timestamp:
            session_date = timestamp.strftime("%Y-%m-%d")
        
        # Reset if new session
        if session_date and session_date != self._session_stopout_date:
            self._session_stopout = {"BUY": False, "SELL": False}
            self._session_stopout_date = session_date
        
        # Track first continuation stopout for FAILED_TREND_DAY check
        # Check if this is a continuation trade type
        is_continuation = entry_type and "CONTINUATION" in entry_type.upper()
        if is_continuation and not self._first_continuation_stopout:
            self._first_continuation_stopout = True
            logger.warning(
                f"⚠️ FAILED_TREND_DAY: First continuation trade stopped out | "
                f"entry_type={entry_type} direction={dir_key} | "
                f"Regime check will occur on next bar"
            )
        
        # Record the stopout
        self._session_stopout[dir_key] = True
        logger.warning(
            f"🛑 ONE_LOSS_PER_DIRECTION: {dir_key} stopped out | "
            f"All {dir_key} trades blocked for remainder of session {session_date}"
        )
    
    def _check_direction_blocked(
        self, 
        signal: "EntrySignal",
        timestamp: Optional[datetime] = None
    ) -> Tuple["EntrySignal", bool]:
        """
        Check if a signal's direction is blocked due to prior stopout.
        
        ONE-LOSS-PER-DIRECTION rule:
        - If we were stopped out in this direction earlier in the session
        - Block all further trades in that direction
        
        Args:
            signal: The entry signal to check
            timestamp: Current timestamp (for session tracking)
            
        Returns:
            Tuple of (signal, was_blocked)
            - If blocked: (HOLD signal, True)
            - If not blocked: (original signal, False)
        """
        if not signal.is_actionable:
            return signal, False
        
        # Determine session date
        session_date = None
        if timestamp:
            session_date = timestamp.strftime("%Y-%m-%d")
        
        # Reset stopouts on new session
        if session_date and session_date != self._session_stopout_date:
            self._session_stopout = {"BUY": False, "SELL": False}
            self._session_stopout_date = session_date
            logger.debug(f"ONE_LOSS_PER_DIRECTION: Reset for new session {session_date}")
        
        # Determine direction from signal action
        action = signal.action.upper()
        if "BUY" in action or action == "LONG":
            dir_key = "BUY"
        elif "SELL" in action or action == "SHORT":
            dir_key = "SELL"
        else:
            # Unknown action type, don't block
            return signal, False
        
        # Check if this direction is blocked
        if self._session_stopout.get(dir_key, False):
            logger.warning(
                f"🚫 ONE_LOSS_PER_DIRECTION: {signal.action} blocked | "
                f"Already stopped out in {dir_key} direction this session"
            )
            
            blocked_signal = EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"ONE_LOSS_PER_DIRECTION: {dir_key} blocked after stopout",
                entry_type="BLOCKED",
                session_window=signal.session_window,
                metadata={
                    "original_action": signal.action,
                    "original_confidence": signal.confidence,
                    "original_reason": signal.reason,
                    "filter_reason": "ONE_LOSS_PER_DIRECTION",
                    "blocked_direction": dir_key,
                }
            )
            return blocked_signal, True
        
        return signal, False
    
    def _check_failed_trend_day(
        self,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        FAILED_TREND_DAY regime guardrail.
        
        If ALL conditions are met:
        1. First continuation trade of the day stopped out
        2. ADX < 22 (weak/no trend)
        3. Price is in VWAP chop zone (within 0.1% of VWAP)
        
        Then: Mark day as FAILED_TREND_DAY, disable continuation logic for rest of session.
        
        Args:
            data: Current bar data with indicators
            timestamp: Current timestamp (for session tracking)
            
        Returns:
            Tuple of (is_failed_trend_day: bool, details: dict)
        """
        details = {
            "first_continuation_stopout": self._first_continuation_stopout,
            "adx": 0.0,
            "adx_threshold": self._failed_trend_adx_threshold,
            "adx_below_threshold": False,
            "close": 0.0,
            "vwap": 0.0,
            "vwap_distance_pct": 0.0,
            "vwap_chop_zone_pct": self._vwap_chop_zone_pct,
            "in_vwap_chop_zone": False,
            "failed_trend_day": False,
            "reason": None,
        }
        
        # Determine session date
        session_date = None
        if timestamp:
            session_date = timestamp.strftime("%Y-%m-%d")
        
        # Reset on new session ONLY if date has changed from a previously set date
        # (Don't reset if date was never set - this is first check of session)
        if session_date and self._failed_trend_day_date is not None and session_date != self._failed_trend_day_date:
            self._failed_trend_day = False
            self._first_continuation_stopout = False
            logger.debug(f"FAILED_TREND_DAY: Reset for new session {session_date}")
        
        # Always update the session date
        if session_date:
            self._failed_trend_day_date = session_date
        
        # Update the details with current state
        details["first_continuation_stopout"] = self._first_continuation_stopout
        
        # If already marked as failed trend day, return early
        if self._failed_trend_day:
            details["failed_trend_day"] = True
            details["reason"] = "ALREADY_MARKED_FAILED"
            return True, details
        
        # CONDITION 1: First continuation trade stopped out
        if not self._first_continuation_stopout:
            details["reason"] = "NO_CONTINUATION_STOPOUT_YET"
            return False, details
        
        # CONDITION 2: ADX < 22 (weak/no trend)
        adx = data.get("adx", data.get("ADX", data.get("ADX_14", 0.0)))
        try:
            adx = float(adx) if adx is not None else 0.0
        except (TypeError, ValueError):
            adx = 0.0
        details["adx"] = adx
        details["adx_below_threshold"] = adx < self._failed_trend_adx_threshold
        
        if adx >= self._failed_trend_adx_threshold:
            details["reason"] = f"ADX_STRONG: {adx:.1f} >= {self._failed_trend_adx_threshold}"
            return False, details
        
        # CONDITION 3: Price in VWAP chop zone (within 0.1% of VWAP)
        close = data.get("close", data.get("Close", 0.0))
        vwap = data.get("vwap", data.get("VWAP", 0.0))
        try:
            close = float(close) if close is not None else 0.0
            vwap = float(vwap) if vwap is not None else 0.0
        except (TypeError, ValueError):
            close = 0.0
            vwap = 0.0
        
        details["close"] = close
        details["vwap"] = vwap
        
        if vwap > 0 and close > 0:
            vwap_distance_pct = abs(close - vwap) / vwap
            details["vwap_distance_pct"] = vwap_distance_pct
            details["in_vwap_chop_zone"] = vwap_distance_pct <= self._vwap_chop_zone_pct
            
            if vwap_distance_pct > self._vwap_chop_zone_pct:
                details["reason"] = f"NOT_IN_CHOP_ZONE: {vwap_distance_pct*100:.3f}% > {self._vwap_chop_zone_pct*100:.1f}%"
                return False, details
        else:
            # Can't evaluate VWAP condition, don't trigger
            details["reason"] = "VWAP_DATA_MISSING"
            return False, details
        
        # ALL CONDITIONS MET - Mark as FAILED TREND DAY
        self._failed_trend_day = True
        details["failed_trend_day"] = True
        details["reason"] = (
            f"FAILED_TREND_DAY: stopout + ADX={adx:.1f}<{self._failed_trend_adx_threshold} + "
            f"VWAP_chop={vwap_distance_pct*100:.3f}%"
        )
        
        logger.warning(
            f"🚨 FAILED_TREND_DAY ACTIVATED | "
            f"First continuation stopped out + ADX={adx:.1f} < {self._failed_trend_adx_threshold} + "
            f"Price in VWAP chop zone ({vwap_distance_pct*100:.3f}% from VWAP) | "
            f"Continuation logic DISABLED for rest of session {session_date}"
        )
        
        return True, details
    
    def _is_continuation_blocked_by_regime(
        self,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """
        Check if continuation trades are blocked due to FAILED_TREND_DAY regime.
        
        Returns:
            Tuple of (is_blocked: bool, reason: str)
        """
        # Check and potentially activate failed trend day
        is_failed, details = self._check_failed_trend_day(data, timestamp)
        
        if is_failed:
            return True, f"FAILED_TREND_DAY: {details['reason']}"
        
        return False, ""
    
    def _check_range_expansion(
        self,
        data: Dict[str, Any],
        recent_bars: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        RANGE_EXPANSION validation for continuation trades.
        
        Continuation trades require range expansion to ensure we're entering 
        during active market periods, not during low-volatility/dead zones.
        
        Two conditions (either must pass):
        1. Current candle range > 1.2 * average range of last 10 candles
        2. ATR(14) is increasing for at least 3 consecutive candles
        
        Args:
            data: Current bar data with OHLC and indicators
            recent_bars: Recent price history (must have at least 10+ bars)
            
        Returns:
            Tuple of (passed: bool, details: dict)
            - passed: True if range expansion is confirmed
            - details: Dict with calculation details for logging
        """
        details = {
            "current_range": 0.0,
            "avg_range": 0.0,
            "range_ratio": 0.0,
            "range_expansion_mult": self._range_expansion_mult,
            "cond1_range_expanded": False,
            "atr_values": [],
            "atr_increasing_count": 0,
            "atr_increasing_bars_required": self._atr_increasing_bars,
            "cond2_atr_increasing": False,
            "passed": False,
            "pass_reason": None,
        }
        
        # Get current bar's range
        current_high = data.get("high", data.get("High", 0.0))
        current_low = data.get("low", data.get("Low", 0.0))
        try:
            current_high = float(current_high) if current_high is not None else 0.0
            current_low = float(current_low) if current_low is not None else 0.0
        except (TypeError, ValueError):
            current_high = 0.0
            current_low = 0.0
        
        current_range = current_high - current_low
        details["current_range"] = current_range
        
        # === CONDITION 1: Current range > 1.2 * average range of last 10 candles ===
        if recent_bars is not None and len(recent_bars) >= self._range_lookback:
            # Get last N bars (excluding current)
            lookback_bars = recent_bars.tail(self._range_lookback + 1).head(self._range_lookback)
            
            # Calculate ranges for lookback bars
            if "high" in lookback_bars.columns and "low" in lookback_bars.columns:
                high_col = "high"
                low_col = "low"
            elif "High" in lookback_bars.columns and "Low" in lookback_bars.columns:
                high_col = "High"
                low_col = "Low"
            else:
                high_col = None
                low_col = None
            
            if high_col and low_col:
                lookback_ranges = lookback_bars[high_col] - lookback_bars[low_col]
                avg_range = lookback_ranges.mean()
                details["avg_range"] = avg_range
                
                if avg_range > 0:
                    range_ratio = current_range / avg_range
                    details["range_ratio"] = range_ratio
                    
                    if range_ratio > self._range_expansion_mult:
                        details["cond1_range_expanded"] = True
                        details["passed"] = True
                        details["pass_reason"] = f"RANGE_EXPANDED: {range_ratio:.2f}x > {self._range_expansion_mult}x"
        
        # === CONDITION 2: ATR increasing for N consecutive candles ===
        # Check if ATR has been increasing
        if recent_bars is not None and len(recent_bars) >= self._atr_increasing_bars + 1:
            # Look for ATR column
            atr_col = None
            for col in ["atr", "ATR", "atr_14", "ATR_14"]:
                if col in recent_bars.columns:
                    atr_col = col
                    break
            
            if atr_col is not None:
                # Get last N+1 ATR values to check N increases
                recent_atr = recent_bars[atr_col].tail(self._atr_increasing_bars + 1).tolist()
                details["atr_values"] = recent_atr
                
                # Count consecutive increases
                increasing_count = 0
                for i in range(1, len(recent_atr)):
                    if recent_atr[i] > recent_atr[i-1]:
                        increasing_count += 1
                    else:
                        increasing_count = 0  # Reset on any decrease
                
                details["atr_increasing_count"] = increasing_count
                
                if increasing_count >= self._atr_increasing_bars:
                    details["cond2_atr_increasing"] = True
                    if not details["passed"]:  # Don't overwrite if cond1 already passed
                        details["passed"] = True
                        details["pass_reason"] = f"ATR_INCREASING: {increasing_count} consecutive bars"
        
        # If neither condition passed, set failure reason
        if not details["passed"]:
            details["pass_reason"] = (
                f"NO_RANGE_EXPANSION: range_ratio={details['range_ratio']:.2f} < {self._range_expansion_mult}, "
                f"ATR_incr={details['atr_increasing_count']} < {self._atr_increasing_bars}"
            )
        
        return details["passed"], details
    
    def _apply_range_expansion_filter(
        self,
        signal: "EntrySignal",
        data: Dict[str, Any],
        recent_bars: Optional[pd.DataFrame],
        module_name: str
    ) -> Tuple["EntrySignal", bool]:
        """
        Apply range expansion filter to a continuation signal.
        
        Continuation trades require range expansion to ensure we're not
        entering during low-volatility periods.
        
        Args:
            signal: The entry signal to filter
            data: Current bar data
            recent_bars: Recent price history
            module_name: Name of the module that generated the signal
            
        Returns:
            Tuple of (filtered_signal, was_filtered)
            - If filtered: (HOLD signal, True)
            - If not filtered: (original signal, False)
        """
        if not signal.is_actionable:
            return signal, False
        
        # Check range expansion
        passed, details = self._check_range_expansion(data, recent_bars)
        
        if not passed:
            logger.warning(
                f"🚫 NO_RANGE_EXPANSION: {module_name} signal suppressed | "
                f"action={signal.action} | {details['pass_reason']}"
            )
            
            filtered_signal = EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"NO_RANGE_EXPANSION: {module_name} blocked - {details['pass_reason']}",
                entry_type="FILTERED",
                session_window=signal.session_window,
                metadata={
                    "original_action": signal.action,
                    "original_confidence": signal.confidence,
                    "original_reason": signal.reason,
                    "filter_reason": "NO_RANGE_EXPANSION",
                    "module": module_name,
                    "range_details": details,
                }
            )
            return filtered_signal, True
        
        logger.debug(
            f"✅ RANGE_EXPANSION OK: {module_name} | {details['pass_reason']}"
        )
        return signal, False

    def _apply_trend_alignment_filter(
        self,
        signal: "EntrySignal",
        data: Dict[str, Any],
        module_name: str,
        market_state: Optional[MarketStateResult] = None
    ) -> Tuple["EntrySignal", bool]:
        """
        Block continuation signals that oppose the higher-level trend label.

        Uses `trend_label` from the strategy (e.g., UPTREND/DOWNTREND/CHOP).
        If missing, this filter is a no-op.
        """
        if not signal.is_actionable:
            return signal, False

        trend_label_raw = data.get("trend_label_htf", data.get("trend_label"))
        if not trend_label_raw:
            return signal, False

        trend_label = str(trend_label_raw).upper()
        action = signal.action.upper()

        block_reason = None
        if trend_label == "CHOP" and "CONTINUATION" in module_name:
            block_reason = "CHOP_REGIME"
        elif trend_label == "UPTREND" and action.startswith("SELL"):
            block_reason = "UPTREND_BLOCKS_SELL"
        elif trend_label == "DOWNTREND" and action.startswith("BUY"):
            block_reason = "DOWNTREND_BLOCKS_BUY"

        if block_reason:
            filtered_signal = EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"TREND_MISMATCH: {module_name} blocked ({block_reason})",
                entry_type="FILTERED",
                session_window=signal.session_window,
                metadata={
                    "original_action": signal.action,
                    "original_confidence": signal.confidence,
                    "original_reason": signal.reason,
                    "filter_reason": "TREND_MISMATCH",
                    "trend_label": trend_label,
                    "module": module_name,
                }
            )
            return filtered_signal, True

        return signal, False
    
    def _evaluate_daily_trend_gate(
        self, 
        data: Dict[str, Any], 
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        DAILY_TREND_CONFIRMATION gate.
        
        Evaluates 3 conditions (at least 2 must be true):
        1. VWAP slope positive on 5m timeframe
        2. EMA21 > EMA50 on 5m or 15m timeframe
        3. ADX > 20
        
        Once this gate fails for a session, BUY_CONTINUATION is blocked.
        The gate is evaluated once per session (resets on new day).
        
        Returns:
            True if daily trend is confirmed (>=2 conditions met)
        """
        # Determine session date
        session_date = None
        if timestamp:
            session_date = timestamp.strftime("%Y-%m-%d")
        elif "timestamp" in data:
            try:
                ts = pd.to_datetime(data["timestamp"])
                session_date = ts.strftime("%Y-%m-%d")
            except Exception:
                pass
        
        # Reset gate on new session/day
        if session_date and session_date != self._daily_trend_gate_session_date:
            self._daily_trend_gate_evaluated = False
            self._daily_trend_confirmed = True
            self._daily_trend_gate_session_date = session_date
            self._daily_trend_gate_details = {}
            logger.debug(f"DAILY_TREND_GATE: Reset for new session {session_date}")
        
        # If already evaluated and failed, return cached result
        if self._daily_trend_gate_evaluated and not self._daily_trend_confirmed:
            return self._daily_trend_confirmed
        
        # Extract indicators
        # VWAP slope: prefer 5m, fall back to computing from price/vwap relationship
        vwap_slope_5m = data.get("5m_vwap_slope", data.get("vwap_slope", 0.0))
        
        # EMA values: prefer 5m/15m, fall back to 1m EMAs if not available
        # This ensures the gate can work with backtest data that only has 1m indicators
        ema21_5m = data.get("5m_ema21", data.get("5m_EMA_21", data.get("ema_21", data.get("EMA_21", 0.0))))
        ema50_5m = data.get("5m_ema50", data.get("5m_EMA_50", data.get("ema_50", data.get("EMA_50", 0.0))))
        ema21_15m = data.get("15m_ema21", data.get("15m_EMA_21", 0.0))
        ema50_15m = data.get("15m_ema50", data.get("15m_EMA_50", 0.0))
        
        # ADX - critical for trend confirmation
        adx = data.get("adx", data.get("ADX", data.get("ADX_14", 0.0)))
        
        # Safely convert to float
        try:
            vwap_slope_5m = float(vwap_slope_5m) if vwap_slope_5m is not None else 0.0
            ema21_5m = float(ema21_5m) if ema21_5m is not None else 0.0
            ema50_5m = float(ema50_5m) if ema50_5m is not None else 0.0
            ema21_15m = float(ema21_15m) if ema21_15m is not None else 0.0
            ema50_15m = float(ema50_15m) if ema50_15m is not None else 0.0
            adx = float(adx) if adx is not None else 0.0
        except (TypeError, ValueError):
            vwap_slope_5m = 0.0
            ema21_5m = 0.0
            ema50_5m = 0.0
            ema21_15m = 0.0
            ema50_15m = 0.0
            adx = 0.0
        
        # === CONDITION 1: VWAP slope positive OR price above VWAP ===
        # Relaxed: If no VWAP slope data, check if price > VWAP as proxy
        vwap_slope_threshold = 0.0001
        price = data.get("close", data.get("price", 0.0))
        vwap = data.get("vwap", data.get("SESSION_VWAP", 0.0))
        try:
            price = float(price) if price is not None else 0.0
            vwap = float(vwap) if vwap is not None else 0.0
        except (TypeError, ValueError):
            price = 0.0
            vwap = 0.0
        
        # VWAP condition: slope positive OR price > VWAP (bullish bias)
        cond1_vwap_positive = (vwap_slope_5m > vwap_slope_threshold) or (price > vwap > 0)
        
        # === CONDITION 2: EMA21 > EMA50 on 5m OR 15m OR 1m (fallback) ===
        cond2_ema_5m = (ema21_5m > ema50_5m) if (ema21_5m > 0 and ema50_5m > 0) else False
        cond2_ema_15m = (ema21_15m > ema50_15m) if (ema21_15m > 0 and ema50_15m > 0) else False
        cond2_ema_bullish = cond2_ema_5m or cond2_ema_15m
        
        # === CONDITION 3: ADX > 20 ===
        cond3_adx_trending = adx > 20
        
        # Count conditions met
        conditions_met = sum([cond1_vwap_positive, cond2_ema_bullish, cond3_adx_trending])
        gate_passed = conditions_met >= 2
        
        # Store details for logging/debugging
        self._daily_trend_gate_details = {
            "vwap_slope_5m": vwap_slope_5m,
            "vwap_slope_threshold": vwap_slope_threshold,
            "price": price,
            "vwap": vwap,
            "cond1_vwap_positive": cond1_vwap_positive,
            "ema21_5m": ema21_5m,
            "ema50_5m": ema50_5m,
            "ema21_15m": ema21_15m,
            "ema50_15m": ema50_15m,
            "cond2_ema_5m": cond2_ema_5m,
            "cond2_ema_15m": cond2_ema_15m,
            "cond2_ema_bullish": cond2_ema_bullish,
            "adx": adx,
            "cond3_adx_trending": cond3_adx_trending,
            "conditions_met": conditions_met,
            "gate_passed": gate_passed,
        }
        
        # Update tracking
        self._daily_trend_gate_evaluated = True
        self._daily_trend_confirmed = gate_passed
        
        # Log the gate evaluation
        if gate_passed:
            logger.info(
                f"✅ DAILY_TREND_GATE PASSED: {conditions_met}/3 conditions met "
                f"(VWAP_positive={cond1_vwap_positive}, EMA_bullish={cond2_ema_bullish}, ADX>20={cond3_adx_trending})"
            )
        else:
            logger.warning(
                f"❌ DAILY_TREND_GATE FAILED: {conditions_met}/3 conditions met "
                f"(VWAP_positive={cond1_vwap_positive}, EMA_bullish={cond2_ema_bullish}, ADX>20={cond3_adx_trending}) "
                f"- BUY_CONTINUATION blocked for session"
            )
        
        return gate_passed

    def _evaluate_daily_trend_gate_short(
        self,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        DAILY_TREND_CONFIRMATION gate for SHORT continuation.

        Evaluates 3 conditions (at least 2 must be true):
        1. VWAP slope negative on 5m timeframe (or price below VWAP)
        2. EMA21 < EMA50 on 5m or 15m timeframe
        3. ADX > 20
        """
        session_date = None
        if timestamp:
            session_date = timestamp.strftime("%Y-%m-%d")
        elif "timestamp" in data:
            try:
                ts = pd.to_datetime(data["timestamp"])
                session_date = ts.strftime("%Y-%m-%d")
            except Exception:
                pass

        if session_date and session_date != self._daily_trend_gate_short_session_date:
            self._daily_trend_gate_short_evaluated = False
            self._daily_trend_short_confirmed = True
            self._daily_trend_gate_short_session_date = session_date
            self._daily_trend_gate_short_details = {}
            logger.debug(f"DAILY_TREND_GATE_SHORT: Reset for new session {session_date}")

        if self._daily_trend_gate_short_evaluated and not self._daily_trend_short_confirmed:
            return self._daily_trend_short_confirmed

        vwap_slope_5m = data.get("5m_vwap_slope", data.get("vwap_slope", 0.0))

        ema21_5m = data.get("5m_ema21", data.get("5m_EMA_21", data.get("ema_21", data.get("EMA_21", 0.0))))
        ema50_5m = data.get("5m_ema50", data.get("5m_EMA_50", data.get("ema_50", data.get("EMA_50", 0.0))))
        ema21_15m = data.get("15m_ema21", data.get("15m_EMA_21", 0.0))
        ema50_15m = data.get("15m_ema50", data.get("15m_EMA_50", 0.0))

        adx = data.get("adx", data.get("ADX", data.get("ADX_14", 0.0)))

        try:
            vwap_slope_5m = float(vwap_slope_5m) if vwap_slope_5m is not None else 0.0
            ema21_5m = float(ema21_5m) if ema21_5m is not None else 0.0
            ema50_5m = float(ema50_5m) if ema50_5m is not None else 0.0
            ema21_15m = float(ema21_15m) if ema21_15m is not None else 0.0
            ema50_15m = float(ema50_15m) if ema50_15m is not None else 0.0
            adx = float(adx) if adx is not None else 0.0
        except (TypeError, ValueError):
            vwap_slope_5m = 0.0
            ema21_5m = 0.0
            ema50_5m = 0.0
            ema21_15m = 0.0
            ema50_15m = 0.0
            adx = 0.0

        vwap_slope_threshold = 0.0001
        price = data.get("close", data.get("price", 0.0))
        vwap = data.get("vwap", data.get("SESSION_VWAP", 0.0))
        try:
            price = float(price) if price is not None else 0.0
            vwap = float(vwap) if vwap is not None else 0.0
        except (TypeError, ValueError):
            price = 0.0
            vwap = 0.0

        cond1_vwap_negative = (vwap_slope_5m < -vwap_slope_threshold) or (price < vwap and vwap > 0)

        cond2_ema_5m = (ema21_5m < ema50_5m) if (ema21_5m > 0 and ema50_5m > 0) else False
        cond2_ema_15m = (ema21_15m < ema50_15m) if (ema21_15m > 0 and ema50_15m > 0) else False
        cond2_ema_bearish = cond2_ema_5m or cond2_ema_15m

        cond3_adx_trending = adx > 20

        conditions_met = sum([cond1_vwap_negative, cond2_ema_bearish, cond3_adx_trending])
        gate_passed = conditions_met >= 2

        self._daily_trend_gate_short_details = {
            "vwap_slope_5m": vwap_slope_5m,
            "vwap_slope_threshold": vwap_slope_threshold,
            "price": price,
            "vwap": vwap,
            "cond1_vwap_negative": cond1_vwap_negative,
            "ema21_5m": ema21_5m,
            "ema50_5m": ema50_5m,
            "ema21_15m": ema21_15m,
            "ema50_15m": ema50_15m,
            "cond2_ema_5m": cond2_ema_5m,
            "cond2_ema_15m": cond2_ema_15m,
            "cond2_ema_bearish": cond2_ema_bearish,
            "adx": adx,
            "cond3_adx_trending": cond3_adx_trending,
            "conditions_met": conditions_met,
            "gate_passed": gate_passed,
        }

        self._daily_trend_gate_short_evaluated = True
        self._daily_trend_short_confirmed = gate_passed

        if gate_passed:
            logger.info(
                f"✅ DAILY_TREND_GATE_SHORT PASSED: {conditions_met}/3 conditions met "
                f"(VWAP_negative={cond1_vwap_negative}, EMA_bearish={cond2_ema_bearish}, ADX>20={cond3_adx_trending})"
            )
        else:
            logger.warning(
                f"❌ DAILY_TREND_GATE_SHORT FAILED: {conditions_met}/3 conditions met "
                f"(VWAP_negative={cond1_vwap_negative}, EMA_bearish={cond2_ema_bearish}, ADX>20={cond3_adx_trending}) "
                f"- SHORT_CONTINUATION blocked for session"
            )

        return gate_passed
    
    def evaluate(
        self,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        prev_data: Optional[Dict[str, Any]] = None,
        recent_bars: Optional[pd.DataFrame] = None,
        session_type: str = "RTH"
    ) -> Tuple[EntrySignal, MarketStateResult]:
        """Evaluate market and generate entry signal.
        
        Args:
            data: Current bar data with indicators
            timestamp: Current timestamp
            prev_data: Previous bar data
            recent_bars: Recent price history
            session_type: "RTH" or "OVERNIGHT"
            
        Returns:
            Tuple of (EntrySignal, MarketStateResult)
        """
        # === 1. DETECT MARKET STATE ===
        market_state = self.state_detector.evaluate(
            data=data,
            timestamp=timestamp,
            prev_data=prev_data,
            session_type=session_type
        )
        
        # === 1.5. DAILY TREND CONFIRMATION GATE (blocks BUY_CONTINUATION if failed) ===
        # This gate evaluates once per session. If it fails, BUY_CONTINUATION is blocked.
        daily_trend_ok = self._evaluate_daily_trend_gate(data, timestamp)

        # === 1.5b. DAILY TREND CONFIRMATION GATE (SHORT) ===
        daily_trend_short_ok = self._evaluate_daily_trend_gate_short(data, timestamp)
        
        # === 1.6. SESSION TIME CHECK FOR BUY_CONTINUATION ===
        # BUY_CONTINUATION is only allowed during MORNING_PRIME (09:30-10:45 CST)
        # After 10:45 CST, only RANGE/MEAN_REVERSION modules are active
        buy_time_ok, buy_time_reason = SessionTimeManager.is_buy_continuation_allowed(timestamp)

        # === 1.6b. SESSION TIME CHECK FOR SHORT_CONTINUATION ===
        short_time_ok, short_time_reason = SessionTimeManager.is_short_continuation_allowed(timestamp)
        
        # === 1.7. FAILED_TREND_DAY REGIME CHECK ===
        # If first continuation stopout + ADX < 22 + VWAP chop zone -> disable continuation
        regime_blocked, regime_reason = self._is_continuation_blocked_by_regime(data, timestamp)
        
        # === 2. TRY BUY CONTINUATION FIRST ===
        # (Buying pullbacks in acceptance is higher probability)
        # GATED: Only evaluate if daily trend gate passed AND within time window AND regime OK
        # Initialize buy_signal to NO_SIGNAL in case gate blocks it
        buy_signal = EntrySignal(
            action="HOLD",
            confidence=0.0,
            reason="DAILY_TREND_GATE_BLOCKED",
            entry_type="NONE",
            session_window="NONE"
        )
        
        if regime_blocked:
            # FAILED_TREND_DAY - all continuation disabled
            buy_signal = EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"REGIME_BLOCKED: {regime_reason}",
                entry_type="BLOCKED",
                session_window=SessionTimeManager.get_session_window(timestamp).value if timestamp else "NONE"
            )
            logger.debug(f"BUY_CONTINUATION skipped: {regime_reason}")
        elif not buy_time_ok:
            # Time cutoff reached - BUY_CONTINUATION disabled
            buy_signal = EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"BUY_TIME_CUTOFF: {buy_time_reason}",
                entry_type="BLOCKED",
                session_window=SessionTimeManager.get_session_window(timestamp).value
            )
            logger.debug(f"BUY_CONTINUATION skipped: {buy_time_reason}")
        elif daily_trend_ok:
            buy_signal = self.buy_module.evaluate(
                market_state=market_state,
                data=data,
                recent_bars=recent_bars,
                timestamp=timestamp
            )
            
            if buy_signal.is_actionable:
                trend_filtered_signal, trend_filtered = self._apply_trend_alignment_filter(
                    buy_signal, data, "BUY_CONTINUATION", market_state
                )
                if trend_filtered:
                    buy_signal = trend_filtered_signal
                else:
                    # Apply range expansion filter for continuation trades
                    range_filtered_signal, range_filtered = self._apply_range_expansion_filter(
                        buy_signal, data, recent_bars, "BUY_CONTINUATION"
                    )
                    if range_filtered:
                        buy_signal = range_filtered_signal
                    else:
                        # Apply global confidence filter
                        filtered_signal, was_filtered = self._apply_confidence_filter(buy_signal, "BUY_CONTINUATION")
                        if not was_filtered:
                            # Check ONE_LOSS_PER_DIRECTION rule
                            final_signal, was_blocked = self._check_direction_blocked(filtered_signal, timestamp)
                            if not was_blocked:
                                final_signal = self._attach_entry_metadata(final_signal, "BUY_CONTINUATION", data)
                                self.last_signal = final_signal
                                self.consecutive_holds = 0
                                return final_signal, market_state
                            else:
                                buy_signal = final_signal
                        else:
                            # Update buy_signal to filtered version for hold reason tracking
                            buy_signal = filtered_signal
        else:
            logger.debug(
                "BUY_CONTINUATION skipped: DAILY_TREND_GATE failed "
                f"(details: {self._daily_trend_gate_details})"
            )
        
        # === 3. TRY SHORT CONTINUATION (ONLY IF BUY RETURNED NO_SIGNAL) ===
        short_signal = EntrySignal(
            action="HOLD",
            confidence=0.0,
            reason="DAILY_TREND_GATE_SHORT_BLOCKED",
            entry_type="NONE",
            session_window="NONE"
        )

        if regime_blocked:
            short_signal = EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"REGIME_BLOCKED: {regime_reason}",
                entry_type="BLOCKED",
                session_window=SessionTimeManager.get_session_window(timestamp).value if timestamp else "NONE"
            )
            logger.debug(f"SHORT_CONTINUATION skipped: {regime_reason}")
        elif not short_time_ok:
            short_signal = EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"SHORT_TIME_CUTOFF: {short_time_reason}",
                entry_type="BLOCKED",
                session_window=SessionTimeManager.get_session_window(timestamp).value
            )
            logger.debug(f"SHORT_CONTINUATION skipped: {short_time_reason}")
        elif daily_trend_short_ok:
            short_signal = self.short_module.evaluate(
                market_state=market_state,
                data=data,
                recent_bars=recent_bars,
                timestamp=timestamp
            )

            if short_signal.is_actionable:
                trend_filtered_signal, trend_filtered = self._apply_trend_alignment_filter(
                    short_signal, data, "SHORT_CONTINUATION", market_state
                )
                if trend_filtered:
                    short_signal = trend_filtered_signal
                else:
                    range_filtered_signal, range_filtered = self._apply_range_expansion_filter(
                        short_signal, data, recent_bars, "SHORT_CONTINUATION"
                    )
                    if range_filtered:
                        short_signal = range_filtered_signal
                    else:
                        filtered_signal, was_filtered = self._apply_confidence_filter(short_signal, "SHORT_CONTINUATION")
                        if not was_filtered:
                            final_signal, was_blocked = self._check_direction_blocked(filtered_signal, timestamp)
                            if not was_blocked:
                                final_signal = self._attach_entry_metadata(final_signal, "SHORT_CONTINUATION", data)
                                self.last_signal = final_signal
                                self.consecutive_holds = 0
                                return final_signal, market_state
                            else:
                                short_signal = final_signal
                        else:
                            short_signal = filtered_signal
        else:
            logger.debug(
                "SHORT_CONTINUATION skipped: DAILY_TREND_GATE_SHORT failed "
                f"(details: {self._daily_trend_gate_short_details})"
            )

        # === 4. TRY EVENING CONTINUATION (ONLY IF BUY/SHORT RETURNED NO_SIGNAL) ===
        # Evening module is isolated and only active in AFTERNOON/CLOSE sessions
        # It will self-filter based on time and return NO_SIGNAL outside valid windows
        # GATED: Blocked if FAILED_TREND_DAY regime is active
        evening_signal = EntrySignal(
            action="HOLD",
            confidence=0.0,
            reason="NO_SIGNAL",
            entry_type="NONE",
            session_window="NONE"
        )
        
        if regime_blocked:
            evening_signal = EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"REGIME_BLOCKED: {regime_reason}",
                entry_type="BLOCKED",
                session_window="EVENING"
            )
            logger.debug(f"EVENING_CONTINUATION skipped: {regime_reason}")
        else:
            evening_signal = self.evening_module.evaluate(
                market_state=market_state,
                data=data,
                recent_bars=recent_bars,
                timestamp=timestamp
            )
            
            if evening_signal.is_actionable:
                trend_filtered_signal, trend_filtered = self._apply_trend_alignment_filter(
                    evening_signal, data, "EVENING_CONTINUATION", market_state
                )
                if trend_filtered:
                    evening_signal = trend_filtered_signal
                else:
                    # Apply range expansion filter for continuation trades
                    range_filtered_signal, range_filtered = self._apply_range_expansion_filter(
                        evening_signal, data, recent_bars, "EVENING_CONTINUATION"
                    )
                    if range_filtered:
                        evening_signal = range_filtered_signal
                    else:
                        # Apply global confidence filter
                        filtered_signal, was_filtered = self._apply_confidence_filter(evening_signal, "EVENING_CONTINUATION")
                        if not was_filtered:
                            # Check ONE_LOSS_PER_DIRECTION rule
                            final_signal, was_blocked = self._check_direction_blocked(filtered_signal, timestamp)
                            if not was_blocked:
                                final_signal = self._attach_entry_metadata(final_signal, "EVENING_CONTINUATION", data)
                                self.last_signal = final_signal
                                self.consecutive_holds = 0
                                return final_signal, market_state
                            else:
                                evening_signal = final_signal
                        else:
                            evening_signal = filtered_signal
        
    # === 5. TRY EVENING SELL CONTINUATION (ONLY IN EXHAUSTION_DOWN) ===
        # This is STRICT - only fires when confirmed exhaustion in downtrend
        # Must have phase_age >= 3 bars to ensure we're not catching falling knives
        # GATED: Blocked if FAILED_TREND_DAY regime is active
        evening_sell_signal = EntrySignal(
            action="HOLD",
            confidence=0.0,
            reason="NO_SIGNAL",
            entry_type="NONE",
            session_window="NONE"
        )
        
        if regime_blocked:
            evening_sell_signal = EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"REGIME_BLOCKED: {regime_reason}",
                entry_type="BLOCKED",
                session_window="EVENING"
            )
            logger.debug(f"EVENING_SELL_CONTINUATION skipped: {regime_reason}")
        else:
            evening_sell_signal = self.evening_sell_module.evaluate(
                market_state=market_state,
                data=data,
                recent_bars=recent_bars,
                timestamp=timestamp
            )
            
            if evening_sell_signal.is_actionable:
                trend_filtered_signal, trend_filtered = self._apply_trend_alignment_filter(
                    evening_sell_signal, data, "EVENING_SELL_CONTINUATION", market_state
                )
                if trend_filtered:
                    evening_sell_signal = trend_filtered_signal
                else:
                    # Apply range expansion filter for continuation trades
                    range_filtered_signal, range_filtered = self._apply_range_expansion_filter(
                        evening_sell_signal, data, recent_bars, "EVENING_SELL_CONTINUATION"
                    )
                    if range_filtered:
                        evening_sell_signal = range_filtered_signal
                    else:
                        # Apply global confidence filter
                        filtered_signal, was_filtered = self._apply_confidence_filter(evening_sell_signal, "EVENING_SELL_CONTINUATION")
                        if not was_filtered:
                            # Check ONE_LOSS_PER_DIRECTION rule
                            final_signal, was_blocked = self._check_direction_blocked(filtered_signal, timestamp)
                            if not was_blocked:
                                final_signal = self._attach_entry_metadata(final_signal, "EVENING_SELL_CONTINUATION", data)
                                self.last_signal = final_signal
                                self.consecutive_holds = 0
                                return final_signal, market_state
                            else:
                                evening_sell_signal = final_signal
                        else:
                            evening_sell_signal = filtered_signal
        
    # === 6. TRY RANGE REVERSION (ONLY ON CHOP/RANGE DAYS) ===
        # This module is ISOLATED - only fires when:
        # - ALL continuation modules returned NO_SIGNAL
        # - Market is in CHOP/RANGE regime (low ADX, no trend)
        # - Time is 09:45-14:30 CST
        range_signal = None
        if self._has_range_module and self.range_reversion_module is not None:
            # Check if any continuation module had a signal (for priority guard)
            continuation_had_signal = (
                buy_signal.is_actionable or
                short_signal.is_actionable or
                evening_signal.is_actionable or
                evening_sell_signal.is_actionable
            )
            
            range_signal = self.range_reversion_module.evaluate(
                market_state=market_state,
                data=data,
                recent_bars=recent_bars,
                timestamp=timestamp,
                continuation_signal_active=continuation_had_signal
            )
            
            if range_signal.is_actionable:
                # Convert RangeEntrySignal to EntrySignal for consistency
                converted_signal = EntrySignal(
                    action=range_signal.action,
                    confidence=range_signal.confidence,
                    reason=range_signal.reason,
                    entry_type=range_signal.entry_type,
                    session_window="RANGE_REVERSION",
                    stop_loss=range_signal.stop_loss,
                    take_profit=range_signal.take_profit_1,
                    metadata=range_signal.metadata
                )
                # Apply global confidence filter
                filtered_signal, was_filtered = self._apply_confidence_filter(converted_signal, "RANGE_REVERSION")
                if not was_filtered:
                    # Check ONE_LOSS_PER_DIRECTION rule
                    final_signal, was_blocked = self._check_direction_blocked(filtered_signal, timestamp)
                    if not was_blocked:
                        final_signal = self._attach_entry_metadata(final_signal, "RANGE_REVERSION", data)
                        self.last_signal = final_signal
                        self.consecutive_holds = 0
                        return final_signal, market_state
                    else:
                        range_signal = final_signal
                else:
                    # Update range_signal reason for hold tracking
                    range_signal = filtered_signal
        
    # === 7. TRY SELL EXHAUSTION (REVERSAL) ===
        # (Only if exhaustion is confirmed)
        sell_signal = self.sell_module.evaluate(
            market_state=market_state,
            data=data,
            recent_bars=recent_bars
        )
        
        if sell_signal.is_actionable:
            # Apply global confidence filter
            filtered_signal, was_filtered = self._apply_confidence_filter(sell_signal, "SELL_EXHAUSTION")
            if not was_filtered:
                # Check ONE_LOSS_PER_DIRECTION rule
                final_signal, was_blocked = self._check_direction_blocked(filtered_signal, timestamp)
                if not was_blocked:
                    final_signal = self._attach_entry_metadata(final_signal, "SELL_EXHAUSTION", data)
                    self.last_signal = final_signal
                    self.consecutive_holds = 0
                    return final_signal, market_state
                else:
                    sell_signal = final_signal
            else:
                sell_signal = filtered_signal
        
    # === 8. NO TRADE ===
        self.consecutive_holds += 1
        
        # Combine reasons (include evening and range reasons if applicable)
        range_reason = range_signal.reason if range_signal else "N/A"
        if evening_signal.session_window in ["AFTERNOON", "CLOSE"]:
            hold_reason = (
                f"BUY: {buy_signal.reason} | "
                f"SHORT: {short_signal.reason} | "
                f"EVE_BUY: {evening_signal.reason} | "
                f"EVE_SELL: {evening_sell_signal.reason} | "
                f"RANGE: {range_reason} | "
                f"SELL: {sell_signal.reason}"
            )
        else:
            hold_reason = (
                f"BUY: {buy_signal.reason} | "
                f"SHORT: {short_signal.reason} | "
                f"RANGE: {range_reason} | "
                f"SELL: {sell_signal.reason}"
            )
        
        hold_signal = EntrySignal(
            action="HOLD",
            confidence=0.0,
            reason=hold_reason,
            entry_type="WAIT",
            metadata={
                "market_phase": market_state.phase.value,
                "trend": market_state.trend.value,
                "trend_score": market_state.trend_score,
                "is_acceptance": market_state.is_acceptance,
                "is_exhaustion": market_state.is_exhaustion,
                "allow_buy": market_state.allow_buy,
                "allow_short": market_state.allow_short,
                "consecutive_holds": self.consecutive_holds
            }
        )
        
        self.last_signal = hold_signal
        return hold_signal, market_state
    
    def get_allowed_trades(self, market_state: MarketStateResult) -> Dict[str, bool]:
        """Get dictionary of allowed trades for current state."""
        return {
            "buy": market_state.allow_buy,
            "short": market_state.allow_short,
            "is_acceptance": market_state.is_acceptance,
            "is_exhaustion": market_state.is_exhaustion
        }


def create_entry_manager(config: Optional[Dict[str, Any]] = None) -> IntegratedEntryManager:
    """Factory function to create entry manager."""
    return IntegratedEntryManager(config)
