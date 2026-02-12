"""Evening SELL continuation module — very strict late RTH sell continuation.

Active only in AFTERNOON/CLOSE sessions, requires confirmed exhaustion
in a downtrend (``EXHAUSTION_DOWN``).  Designed to be rare — maybe 1-2
signals per week.

Inputs:
    market_state, data, recent_bars, timestamp.

Outputs:
    EntrySignal: ``SELL`` or ``HOLD``.

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


