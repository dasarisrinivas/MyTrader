"""Evening BUY continuation module — high-momentum continuation in late RTH.

Active only in AFTERNOON (14:00-15:00 CST) and CLOSE (15:00-16:00 CST)
session windows.  Requires stronger confirmation (ADX >= 32) and uses
tighter stops than morning continuation.

Inputs:
    market_state, data, recent_bars, timestamp.

Outputs:
    EntrySignal: ``BUY`` with tighter risk or ``HOLD``.

Side-effects:
    None — pure analysis, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..market_state import MarketStateResult, TrendDirection, MarketPhase
from .session_time import SessionWindow, SessionTimeManager
from .signals import EntrySignal


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

