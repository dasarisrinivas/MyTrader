"""Market State Detection - Acceptance vs Exhaustion Framework.

This module provides clear separation between:
1. ACCEPTANCE: Market accepting prices at a level (continuation likely)
2. EXHAUSTION: Market rejecting further price movement (reversal possible)

CORE PRINCIPLE:
- NEVER short acceptance (bullish strength)
- ONLY short exhaustion (bearish reversal setup)
- ALWAYS buy on acceptance pullbacks

STATE TRANSITIONS:
- ACCEPTANCE → EXHAUSTION requires confirmed momentum loss
- EXHAUSTION → RESET requires return to neutral
- RESET → ACCEPTANCE requires fresh trend structure

This fixes the sell-side bias where bullish signals (EMA_STACK_UP, MACD_POS, RSI_HIGH)
were incorrectly interpreted as SHORT signals, causing repeated losses.

Author: Trading System Refactor - Jan 2026
"""
from dataclasses import dataclass, field
from datetime import time, datetime
from enum import Enum
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd
from loguru import logger


class TrendDirection(Enum):
    """Directional trend state (not strength)."""
    STRONG_TREND_UP = "STRONG_TREND_UP"      # Full EMA stack + momentum
    TREND_UP = "TREND_UP"                     # Above EMAs, positive slope
    WEAK_UP = "WEAK_UP"                       # Above some EMAs
    NEUTRAL = "NEUTRAL"                       # No clear direction
    WEAK_DOWN = "WEAK_DOWN"                   # Below some EMAs
    TREND_DOWN = "TREND_DOWN"                 # Below EMAs, negative slope
    STRONG_TREND_DOWN = "STRONG_TREND_DOWN"   # Full bearish stack + momentum


class MarketPhase(Enum):
    """Market phase for trade selection.
    
    STATE MACHINE:
    - ACCEPTANCE_UP/DOWN: Trend continuation phase, trade with trend only
    - EXHAUSTION_UP/DOWN: Reversal setup phase, counter-trend allowed
    - RESET: Neutral state, no entries allowed (wait for fresh structure)
    - CHOP: No clear phase, low confidence trades only
    - SQUEEZE: Strong momentum, never fade
    """
    ACCEPTANCE_UP = "ACCEPTANCE_UP"       # Bullish continuation - BUY only
    ACCEPTANCE_DOWN = "ACCEPTANCE_DOWN"   # Bearish continuation - SELL only
    EXHAUSTION_UP = "EXHAUSTION_UP"       # Topping - SELL possible
    EXHAUSTION_DOWN = "EXHAUSTION_DOWN"   # Bottoming - BUY possible
    RESET = "RESET"                       # Neutral - NO entries allowed
    CHOP = "CHOP"                         # No clear phase - avoid trading
    SQUEEZE = "SQUEEZE"                   # Strong momentum squeeze - avoid fading


@dataclass
class PhaseTransition:
    """Tracks phase transition details for state machine enforcement.
    
    WHY THIS EXISTS:
    - Prevents premature SELL entries before exhaustion is confirmed
    - Ensures minimum bar count before acting on phase change
    - Provides audit trail for debugging trade decisions
    """
    from_phase: MarketPhase = MarketPhase.CHOP
    to_phase: MarketPhase = MarketPhase.CHOP
    transition_bar: int = 0
    transition_reason: str = ""
    transition_timestamp: Optional[datetime] = None


@dataclass
class MarketStateResult:
    """Complete market state analysis.
    
    This replaces the simple trend_score with separate directional and
    exhaustion components for clear trade decision logic.
    """
    # === PRIMARY FLAGS (use these for trade decisions) ===
    is_acceptance: bool = False       # True = market accepting direction, continuation likely
    is_exhaustion: bool = False       # True = reversal conditions present
    allow_buy: bool = False           # True = BUY trades permitted
    allow_short: bool = False         # True = SHORT trades permitted
    
    # === MARKET PHASE ===
    phase: MarketPhase = MarketPhase.CHOP
    trend: TrendDirection = TrendDirection.NEUTRAL
    
    # === PHASE TRANSITION TRACKING ===
    phase_age: int = 0                # Bars since last phase transition
    last_transition: Optional[PhaseTransition] = None
    phase_confirmed: bool = False     # True if phase has been stable for min bars
    
    # === COMPONENT SCORES (for debugging/logging) ===
    trend_score: float = 0.0          # -100 to +100, directional strength only
    exhaustion_score: float = 0.0     # 0 to 100, higher = more exhaustion signals
    continuation_score: float = 0.0   # 0 to 100, higher = better continuation setup
    
    # === INDICATOR VALUES ===
    ema_stack_up: bool = False        # Price > EMA9 > EMA21 > EMA50
    ema_stack_down: bool = False      # Price < EMA9 < EMA21 < EMA50
    ema_flattening: bool = False      # EMAs converging (trend weakening)
    macd_positive: bool = False       # MACD histogram > threshold
    macd_negative: bool = False       # MACD histogram < -threshold
    macd_slope_up: bool = False       # MACD increasing
    macd_slope_down: bool = False     # MACD decreasing
    macd_declining_bars: int = 0      # Consecutive bars of MACD decline
    rsi_value: float = 50.0
    rsi_rising: bool = False
    rsi_falling: bool = False
    rsi_slope: float = 0.0            # Rolling RSI slope
    vwap_slope: float = 0.0           # VWAP direction indicator
    price_above_vwap: bool = False
    price_above_ema50: bool = False
    
    # === LEVEL PROXIMITY ===
    near_pdh: bool = False            # Near previous day high
    near_pdl: bool = False            # Near previous day low
    near_vwap: bool = False           # Near VWAP
    vwap_deviation_extreme: bool = False  # >2 ATR from VWAP
    vwap_deviation_atr: float = 0.0   # ATR distance from VWAP
    
    # === HARD BLOCKS ===
    short_blocked: bool = False       # SHORT is blocked by rules
    buy_blocked: bool = False         # BUY is blocked by rules
    block_reasons: List[str] = field(default_factory=list)
    
    # === REASONING ===
    reasoning: str = ""


class MarketStateDetector:
    """Detects market acceptance vs exhaustion for trade direction decisions.
    
    KEY CONCEPTS:
    
    ACCEPTANCE (Continuation Setup):
    - EMA_STACK_UP = true (Price > EMA9 > EMA21 > EMA50)
    - MACD_POS > 0.20
    - RSI between 55 and 70 (strong but not overbought)
    - Price above VWAP or EMA50
    Meaning: Market is accepting higher/lower prices, continuation likely
    Action: BUY pullbacks (bullish) or SELL pullbacks (bearish)
    
    EXHAUSTION (Reversal Setup):
    - RSI >= 65 AND RSI falling (losing momentum)
    - OR MACD slope <= 0 (flattening despite high price)
    - OR failed high (higher high rejected within N bars)
    Meaning: Market rejecting further extension, reversal possible
    Action: Consider counter-trend trades ONLY with exhaustion confirmation
    
    STATE TRANSITIONS:
    - ACCEPTANCE → EXHAUSTION: RSI >= 70 AND falling, MACD declining N bars, near PDH
    - EXHAUSTION → RESET: RSI returns to 45-55, MACD crosses zero or flattens
    - RESET → ACCEPTANCE: Fresh EMA stack, new impulse, volume expansion
    
    HARD RULES:
    1. NEVER SHORT ACCEPTANCE - if EMA_STACK_UP and MACD_POS > 0.20, SHORT = BLOCKED
    2. MORNING RTH PROTECTION - first 45 min + bullish setup = no shorts
    3. SQUEEZE PROTECTION - RSI > 75 and MACD > 0.30 = no shorts
    4. PHASE CONFIRMATION - min bars required before acting on phase change
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize market state detector with configurable thresholds."""
        config = config or {}
        
        # === ACCEPTANCE THRESHOLDS ===
        self.macd_acceptance_threshold = config.get("macd_acceptance_threshold", 0.20)
        self.rsi_acceptance_min = config.get("rsi_acceptance_min", 55)
        self.rsi_acceptance_max = config.get("rsi_acceptance_max", 70)
        
        # === EXHAUSTION THRESHOLDS ===
        self.rsi_exhaustion_threshold = config.get("rsi_exhaustion_threshold", 65)
        self.macd_slope_exhaustion = config.get("macd_slope_exhaustion", 0.0)
        self.failed_high_bars = config.get("failed_high_bars", 5)
        
        # === PHASE TRANSITION THRESHOLDS ===
        self.macd_declining_bars_threshold = config.get("macd_declining_bars_threshold", 3)
        self.phase_confirmation_bars = config.get("phase_confirmation_bars", 3)
        self.rsi_neutral_min = config.get("rsi_neutral_min", 45)
        self.rsi_neutral_max = config.get("rsi_neutral_max", 55)
        self.rsi_exhaustion_up_threshold = config.get("rsi_exhaustion_up_threshold", 70)
        self.rsi_exhaustion_down_threshold = config.get("rsi_exhaustion_down_threshold", 30)
        
        # === SQUEEZE THRESHOLDS (never fade) ===
        self.rsi_squeeze_threshold = config.get("rsi_squeeze_threshold", 75)
        self.macd_squeeze_threshold = config.get("macd_squeeze_threshold", 0.30)
        
        # === MORNING PROTECTION ===
        self.morning_protection_minutes = config.get("morning_protection_minutes", 45)
        self.morning_protection_hour = config.get("morning_protection_hour", 10)
        self.morning_protection_minute = config.get("morning_protection_minute", 15)
        
        # === VWAP DEVIATION ===
        self.vwap_extreme_atr_mult = config.get("vwap_extreme_atr_mult", 2.0)
        
        # === LEVEL PROXIMITY ===
        self.level_proximity_pct = config.get("level_proximity_pct", 0.15)
        
        # === PHASE TRACKING STATE ===
        self._current_phase = MarketPhase.CHOP
        self._phase_age = 0
        self._last_transition: Optional[PhaseTransition] = None
        self._bar_count = 0
        self._macd_declining_count = 0
        self._prev_macd = 0.0
        self._rsi_history: List[float] = []
        self._macd_history: List[float] = []
        self._vwap_history: List[float] = []
        
        logger.info(
            f"MarketStateDetector initialized: "
            f"macd_accept={self.macd_acceptance_threshold}, "
            f"rsi_exhaust={self.rsi_exhaustion_threshold}, "
            f"phase_confirm_bars={self.phase_confirmation_bars}"
        )
    
    def evaluate(
        self,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        prev_data: Optional[Dict[str, Any]] = None,
        session_type: str = "RTH"
    ) -> MarketStateResult:
        """Evaluate market state for acceptance vs exhaustion.
        
        Args:
            data: Current bar data with indicators
            timestamp: Current timestamp (for morning protection)
            prev_data: Previous bar data (for slope calculations)
            session_type: "RTH" or "OVERNIGHT"
            
        Returns:
            MarketStateResult with all flags and scores
        """
        result = MarketStateResult()
        
        # Increment bar count for phase tracking
        self._bar_count += 1
        
        # === EXTRACT INDICATORS ===
        price = float(data.get("close", data.get("price", 0)))
        ema9 = float(data.get("ema_9", data.get("EMA_9", price)))
        ema21 = float(data.get("ema_21", data.get("EMA_21", price)))
        ema50 = float(data.get("ema_50", data.get("EMA_50", price)))
        vwap = float(data.get("vwap", data.get("SESSION_VWAP", price)))
        
        rsi = float(data.get("rsi", data.get("RSI_14", 50)))
        macd_hist = float(data.get("macd_hist", data.get("MACD_hist", 0)))
        atr = float(data.get("atr", data.get("ATR_14", 1.0)))
        
        pdh = float(data.get("pdh", data.get("PDH", 0)))
        pdl = float(data.get("pdl", data.get("PDL", 0)))
        
        # Get previous values for slope calculation
        prev_rsi = 50.0
        prev_macd = 0.0
        if prev_data:
            prev_rsi = float(prev_data.get("rsi", prev_data.get("RSI_14", rsi)))
            prev_macd = float(prev_data.get("macd_hist", prev_data.get("MACD_hist", macd_hist)))
        else:
            prev_macd = self._prev_macd
        
        result.rsi_value = rsi
        
        # === UPDATE HISTORY FOR SLOPE CALCULATIONS ===
        self._rsi_history.append(rsi)
        self._macd_history.append(macd_hist)
        self._vwap_history.append(vwap)
        
        # Keep only recent history (10 bars)
        max_history = 10
        if len(self._rsi_history) > max_history:
            self._rsi_history = self._rsi_history[-max_history:]
        if len(self._macd_history) > max_history:
            self._macd_history = self._macd_history[-max_history:]
        if len(self._vwap_history) > max_history:
            self._vwap_history = self._vwap_history[-max_history:]
        
        # Calculate RSI slope (rolling 3-bar)
        if len(self._rsi_history) >= 3:
            result.rsi_slope = (self._rsi_history[-1] - self._rsi_history[-3]) / 2.0
        
        # Calculate VWAP slope
        if len(self._vwap_history) >= 3 and self._vwap_history[-3] > 0:
            result.vwap_slope = (self._vwap_history[-1] - self._vwap_history[-3]) / self._vwap_history[-3]
        
        # Track MACD declining bars
        if macd_hist < prev_macd:
            self._macd_declining_count += 1
        else:
            self._macd_declining_count = 0
        result.macd_declining_bars = self._macd_declining_count
        
        self._prev_macd = macd_hist
        
        # === 1. EMA STACK DETECTION ===
        # Bullish stack: Price > EMA9 > EMA21 > EMA50
        result.ema_stack_up = (
            price > ema9 > ema21 > ema50 and
            ema9 > 0 and ema21 > 0 and ema50 > 0
        )
        # Bearish stack: Price < EMA9 < EMA21 < EMA50
        result.ema_stack_down = (
            price < ema9 < ema21 < ema50 and
            ema9 > 0 and ema21 > 0 and ema50 > 0
        )
        
        # EMA flattening detection (EMAs converging)
        if ema9 > 0 and ema21 > 0:
            ema_spread = abs(ema9 - ema21) / ema21
            result.ema_flattening = ema_spread < 0.001  # Very close EMAs
        
        # === 2. MACD ANALYSIS ===
        result.macd_positive = macd_hist > self.macd_acceptance_threshold
        result.macd_negative = macd_hist < -self.macd_acceptance_threshold
        result.macd_slope_up = macd_hist > prev_macd
        result.macd_slope_down = macd_hist < prev_macd
        
        # === 3. RSI ANALYSIS ===
        result.rsi_rising = rsi > prev_rsi
        result.rsi_falling = rsi < prev_rsi
        
        # === 4. PRICE POSITION ===
        result.price_above_vwap = price > vwap if vwap > 0 else False
        result.price_above_ema50 = price > ema50 if ema50 > 0 else False
        
        # === 5. LEVEL PROXIMITY ===
        if pdh > 0:
            pdh_dist_pct = abs(price - pdh) / price * 100
            result.near_pdh = pdh_dist_pct < self.level_proximity_pct
        if pdl > 0:
            pdl_dist_pct = abs(price - pdl) / price * 100
            result.near_pdl = pdl_dist_pct < self.level_proximity_pct
        if vwap > 0:
            vwap_dist_pct = abs(price - vwap) / price * 100
            result.near_vwap = vwap_dist_pct < self.level_proximity_pct
            # Extreme deviation from VWAP
            vwap_dist_atr = abs(price - vwap) / atr if atr > 0 else 0
            result.vwap_deviation_extreme = vwap_dist_atr > self.vwap_extreme_atr_mult
            result.vwap_deviation_atr = vwap_dist_atr
        
        # === 6. ACCEPTANCE DETECTION ===
        # Bullish Acceptance: Strong uptrend, continuation likely
        bullish_acceptance = (
            result.ema_stack_up and
            result.macd_positive and
            self.rsi_acceptance_min <= rsi <= self.rsi_acceptance_max and
            (result.price_above_vwap or result.price_above_ema50)
        )
        
        # Bearish Acceptance: Strong downtrend, continuation likely
        bearish_acceptance = (
            result.ema_stack_down and
            result.macd_negative and
            (100 - self.rsi_acceptance_max) <= rsi <= (100 - self.rsi_acceptance_min) and
            not result.price_above_vwap and not result.price_above_ema50
        )
        
        result.is_acceptance = bullish_acceptance or bearish_acceptance
        
        # === 7. EXHAUSTION DETECTION ===
        # Bullish Exhaustion (topping): High RSI falling, MACD flattening
        bullish_exhaustion = (
            rsi >= self.rsi_exhaustion_threshold and
            result.rsi_falling and
            (result.macd_slope_down or macd_hist <= self.macd_slope_exhaustion)
        )
        
        # Bearish Exhaustion (bottoming): Low RSI rising, MACD recovering
        bearish_exhaustion = (
            rsi <= (100 - self.rsi_exhaustion_threshold) and
            result.rsi_rising and
            (result.macd_slope_up or macd_hist >= -self.macd_slope_exhaustion)
        )
        
        result.is_exhaustion = bullish_exhaustion or bearish_exhaustion
        
        # === 8. SQUEEZE DETECTION (never fade) ===
        is_squeeze = (
            rsi > self.rsi_squeeze_threshold and
            macd_hist > self.macd_squeeze_threshold
        )
        
        # === 9. TREND DIRECTION ===
        result.trend = self._classify_trend(
            price, ema9, ema21, ema50, macd_hist, rsi,
            result.ema_stack_up, result.ema_stack_down
        )
        
        # === 10. TREND SCORE (directional only) ===
        result.trend_score = self._calculate_trend_score(
            price, ema9, ema21, ema50, macd_hist, rsi,
            result.ema_stack_up, result.ema_stack_down
        )
        
        # === 11. EXHAUSTION SCORE ===
        result.exhaustion_score = self._calculate_exhaustion_score(
            rsi, macd_hist, result.rsi_falling, result.macd_slope_down,
            result.near_pdh, bullish_exhaustion, bearish_exhaustion
        )
        
        # === 12. CONTINUATION SCORE ===
        result.continuation_score = self._calculate_continuation_score(
            result.ema_stack_up, result.ema_stack_down,
            result.macd_positive, result.macd_negative,
            rsi, result.price_above_vwap, result.near_vwap
        )
        
        # === 13. MARKET PHASE (with transition validation) ===
        result.phase = self._determine_phase(
            bullish_acceptance, bearish_acceptance,
            bullish_exhaustion, bearish_exhaustion,
            is_squeeze, result.trend,
            result, timestamp
        )
        
        # === 14. HARD BLOCK RULES ===
        self._apply_hard_blocks(
            result, timestamp, session_type,
            bullish_acceptance, is_squeeze, macd_hist
        )
        
        # === 15. DETERMINE ALLOWED TRADES ===
        self._determine_allowed_trades(
            result, bullish_acceptance, bearish_acceptance,
            bullish_exhaustion, bearish_exhaustion, is_squeeze
        )
        
        # === 16. BUILD REASONING ===
        result.reasoning = self._build_reasoning(result)
        
        return result
    
    def _classify_trend(
        self, price: float, ema9: float, ema21: float, ema50: float,
        macd_hist: float, rsi: float,
        ema_stack_up: bool, ema_stack_down: bool
    ) -> TrendDirection:
        """Classify trend direction (not strength for reversal)."""
        if ema_stack_up and macd_hist > 0.3 and rsi > 60:
            return TrendDirection.STRONG_TREND_UP
        elif ema_stack_up:
            return TrendDirection.TREND_UP
        elif price > ema21 > ema50:
            return TrendDirection.WEAK_UP
        elif ema_stack_down and macd_hist < -0.3 and rsi < 40:
            return TrendDirection.STRONG_TREND_DOWN
        elif ema_stack_down:
            return TrendDirection.TREND_DOWN
        elif price < ema21 < ema50:
            return TrendDirection.WEAK_DOWN
        else:
            return TrendDirection.NEUTRAL
    
    def _calculate_trend_score(
        self, price: float, ema9: float, ema21: float, ema50: float,
        macd_hist: float, rsi: float,
        ema_stack_up: bool, ema_stack_down: bool
    ) -> float:
        """Calculate trend score: -100 (bearish) to +100 (bullish).
        
        This score represents DIRECTIONAL STRENGTH only.
        High positive = strong bullish (avoid shorts)
        High negative = strong bearish (avoid longs)
        """
        score = 0.0
        
        # EMA Stack: +/- 30 points
        if ema_stack_up:
            score += 30
        elif ema_stack_down:
            score -= 30
        elif price > ema9 > ema21:
            score += 15
        elif price < ema9 < ema21:
            score -= 15
        elif price > ema9:
            score += 5
        elif price < ema9:
            score -= 5
        
        # MACD: +/- 25 points
        macd_normalized = max(-25, min(25, macd_hist * 50))
        score += macd_normalized
        
        # RSI Trend Position: +/- 20 points
        # RSI 50 = neutral, >50 = bullish bias, <50 = bearish bias
        # But NOT as reversal signal - just directional bias
        rsi_bias = (rsi - 50) * 0.4  # +/-20 max
        score += rsi_bias
        
        # Price vs EMA50: +/- 15 points
        if ema50 > 0:
            ema50_dist = (price - ema50) / ema50 * 100
            score += max(-15, min(15, ema50_dist * 30))
        
        # Momentum confirmation: +/- 10 points
        if macd_hist > 0 and rsi > 55:
            score += 10
        elif macd_hist < 0 and rsi < 45:
            score -= 10
        
        return max(-100, min(100, score))
    
    def _calculate_exhaustion_score(
        self, rsi: float, macd_hist: float,
        rsi_falling: bool, macd_slope_down: bool,
        near_pdh: bool, bullish_exhaustion: bool, bearish_exhaustion: bool
    ) -> float:
        """Calculate exhaustion score: 0 (no exhaustion) to 100 (strong exhaustion)."""
        score = 0.0
        
        # Bullish exhaustion signals
        if rsi > 70:
            score += 20
        elif rsi > 65:
            score += 10
        
        if rsi_falling and rsi > 60:
            score += 15
        
        if macd_slope_down and macd_hist > 0:
            score += 15  # MACD rolling over while still positive = divergence
        
        if near_pdh:
            score += 10  # Near resistance
        
        # Bearish exhaustion signals
        if rsi < 30:
            score += 20
        elif rsi < 35:
            score += 10
        
        # Confirmation multiplier
        if bullish_exhaustion or bearish_exhaustion:
            score *= 1.5
        
        return min(100, score)
    
    def _calculate_continuation_score(
        self, ema_stack_up: bool, ema_stack_down: bool,
        macd_positive: bool, macd_negative: bool,
        rsi: float, price_above_vwap: bool, near_vwap: bool
    ) -> float:
        """Calculate continuation score: 0 (no setup) to 100 (strong continuation)."""
        score = 0.0
        
        # EMA alignment
        if ema_stack_up or ema_stack_down:
            score += 30
        
        # MACD confirmation
        if (ema_stack_up and macd_positive) or (ema_stack_down and macd_negative):
            score += 25
        
        # RSI in continuation zone (not extreme)
        if 45 <= rsi <= 65:
            score += 15
        elif 35 <= rsi <= 75:
            score += 10
        
        # VWAP position
        if ema_stack_up and price_above_vwap:
            score += 15
        elif ema_stack_down and not price_above_vwap:
            score += 15
        
        # Pullback opportunity
        if near_vwap:
            score += 15  # Good entry location
        
        return min(100, score)
    
    def _determine_phase(
        self, bullish_acceptance: bool, bearish_acceptance: bool,
        bullish_exhaustion: bool, bearish_exhaustion: bool,
        is_squeeze: bool, trend: TrendDirection,
        result: MarketStateResult,
        timestamp: Optional[datetime] = None
    ) -> MarketPhase:
        """Determine current market phase with transition validation.
        
        STATE TRANSITION RULES:
        
        ACCEPTANCE → EXHAUSTION requires:
        - RSI >= 70 (up) or <= 30 (down) AND rolling RSI slope < 0 (up) / > 0 (down)
        - MACD histogram declining for N bars (configurable, default = 3)
        - Price rejection near PDH / value high / VWAP deviation extreme
        
        EXHAUSTION → RESET requires:
        - RSI returns to neutral zone (45–55)
        - MACD histogram crosses zero OR flattens
        - Price re-enters VWAP zone
        
        RESET → ACCEPTANCE only allowed after:
        - Fresh EMA stack
        - New impulse leg (confirmed by MACD direction)
        """
        # Determine raw phase
        raw_phase = MarketPhase.CHOP
        transition_reason = ""
        
        if is_squeeze:
            raw_phase = MarketPhase.SQUEEZE
            transition_reason = "SQUEEZE: RSI > 75 + MACD > 0.30"
        elif bullish_acceptance:
            raw_phase = MarketPhase.ACCEPTANCE_UP
            transition_reason = "ACCEPTANCE_UP: EMA stack + MACD+ + RSI in zone"
        elif bearish_acceptance:
            raw_phase = MarketPhase.ACCEPTANCE_DOWN
            transition_reason = "ACCEPTANCE_DOWN: EMA bearish stack + MACD- + RSI in zone"
        elif bullish_exhaustion:
            raw_phase = MarketPhase.EXHAUSTION_UP
            transition_reason = "EXHAUSTION_UP: RSI high + falling + MACD rolling over"
        elif bearish_exhaustion:
            raw_phase = MarketPhase.EXHAUSTION_DOWN
            transition_reason = "EXHAUSTION_DOWN: RSI low + rising + MACD recovering"
        
        # === VALIDATE STATE TRANSITIONS ===
        new_phase = self._validate_phase_transition(
            current_phase=self._current_phase,
            proposed_phase=raw_phase,
            result=result,
            transition_reason=transition_reason,
            timestamp=timestamp
        )
        
        # Update tracking
        if new_phase != self._current_phase:
            self._last_transition = PhaseTransition(
                from_phase=self._current_phase,
                to_phase=new_phase,
                transition_bar=self._bar_count,
                transition_reason=transition_reason,
                transition_timestamp=timestamp
            )
            self._current_phase = new_phase
            self._phase_age = 0
            logger.debug(f"[PHASE_TRANSITION] {self._last_transition.from_phase.value} → {new_phase.value}: {transition_reason}")
        else:
            self._phase_age += 1
        
        # Update result with phase tracking
        result.phase_age = self._phase_age
        result.last_transition = self._last_transition
        result.phase_confirmed = self._phase_age >= self.phase_confirmation_bars
        
        return new_phase
    
    def _validate_phase_transition(
        self,
        current_phase: MarketPhase,
        proposed_phase: MarketPhase,
        result: MarketStateResult,
        transition_reason: str,
        timestamp: Optional[datetime]
    ) -> MarketPhase:
        """Validate that phase transition is allowed by state machine rules.
        
        WHY THIS EXISTS:
        - Prevents premature SELL entries before exhaustion is confirmed
        - Ensures ACCEPTANCE → EXHAUSTION requires multiple confirming factors
        - Forces a RESET period before re-entering ACCEPTANCE
        
        This is the key guard that prevents false SELL signals in strong trends.
        """
        # If same phase, no validation needed
        if proposed_phase == current_phase:
            return proposed_phase
        
        # === ACCEPTANCE → EXHAUSTION TRANSITION ===
        # Very strict - requires confirmed momentum loss
        if current_phase in [MarketPhase.ACCEPTANCE_UP, MarketPhase.ACCEPTANCE_DOWN]:
            if proposed_phase in [MarketPhase.EXHAUSTION_UP, MarketPhase.EXHAUSTION_DOWN]:
                # Validate: RSI must be extreme AND falling/rising
                if current_phase == MarketPhase.ACCEPTANCE_UP:
                    rsi_valid = (
                        result.rsi_value >= self.rsi_exhaustion_up_threshold and 
                        result.rsi_slope < 0
                    )
                else:  # ACCEPTANCE_DOWN
                    rsi_valid = (
                        result.rsi_value <= self.rsi_exhaustion_down_threshold and 
                        result.rsi_slope > 0
                    )
                
                # Validate: MACD must be declining for N bars
                macd_valid = result.macd_declining_bars >= self.macd_declining_bars_threshold
                
                # Validate: Near key level (PDH/PDL or VWAP extreme)
                level_valid = result.near_pdh or result.near_pdl or result.vwap_deviation_extreme
                
                if rsi_valid and macd_valid and level_valid:
                    logger.info(
                        f"[PHASE_TRANSITION] ACCEPTANCE→EXHAUSTION validated: "
                        f"RSI={result.rsi_value:.1f} slope={result.rsi_slope:.2f}, "
                        f"MACD_decline={result.macd_declining_bars}, level_near={level_valid}"
                    )
                    return proposed_phase
                else:
                    # Transition blocked - stay in acceptance
                    logger.debug(
                        f"[PHASE_TRANSITION] ACCEPTANCE→EXHAUSTION BLOCKED: "
                        f"rsi_valid={rsi_valid}, macd_valid={macd_valid}, level_valid={level_valid}"
                    )
                    return current_phase
            
            # ACCEPTANCE → CHOP: Allow direct transition
            elif proposed_phase == MarketPhase.CHOP:
                return proposed_phase
            
            # ACCEPTANCE → RESET: Not typical, but allow
            elif proposed_phase == MarketPhase.RESET:
                return proposed_phase
        
        # === EXHAUSTION → RESET TRANSITION ===
        # Required before re-entering acceptance
        elif current_phase in [MarketPhase.EXHAUSTION_UP, MarketPhase.EXHAUSTION_DOWN]:
            if proposed_phase in [MarketPhase.ACCEPTANCE_UP, MarketPhase.ACCEPTANCE_DOWN]:
                # Cannot go directly EXHAUSTION → ACCEPTANCE
                # Must go through RESET first
                # Check if should transition to RESET instead
                rsi_neutral = self.rsi_neutral_min <= result.rsi_value <= self.rsi_neutral_max
                macd_neutral = abs(result.macd_positive is False and result.macd_negative is False)
                near_vwap = result.near_vwap
                
                if rsi_neutral or (macd_neutral and near_vwap):
                    logger.info(f"[PHASE_TRANSITION] EXHAUSTION→RESET (blocking direct ACCEPTANCE)")
                    return MarketPhase.RESET
                else:
                    # Stay in exhaustion
                    return current_phase
            
            elif proposed_phase == MarketPhase.RESET:
                # Validate RESET conditions
                rsi_neutral = self.rsi_neutral_min <= result.rsi_value <= self.rsi_neutral_max
                
                if rsi_neutral:
                    return MarketPhase.RESET
                else:
                    return current_phase
            
            # EXHAUSTION → CHOP: Allow
            elif proposed_phase == MarketPhase.CHOP:
                return proposed_phase
        
        # === RESET → ACCEPTANCE TRANSITION ===
        # Requires fresh trend structure
        elif current_phase == MarketPhase.RESET:
            if proposed_phase in [MarketPhase.ACCEPTANCE_UP, MarketPhase.ACCEPTANCE_DOWN]:
                # Validate: Fresh EMA stack required
                ema_valid = result.ema_stack_up or result.ema_stack_down
                
                # Validate: Momentum confirmation
                macd_valid = result.macd_positive or result.macd_negative
                
                if ema_valid and macd_valid:
                    logger.info(f"[PHASE_TRANSITION] RESET→ACCEPTANCE: Fresh structure confirmed")
                    return proposed_phase
                else:
                    return current_phase
        
        # === DEFAULT: Allow transition ===
        return proposed_phase
    
    def _apply_hard_blocks(
        self, result: MarketStateResult,
        timestamp: Optional[datetime],
        session_type: str,
        bullish_acceptance: bool,
        is_squeeze: bool,
        macd_hist: float
    ) -> None:
        """Apply hard block rules that prevent certain trades.
        
        HARD BLOCK RULES:
        1. NEVER SHORT ACCEPTANCE - bullish acceptance = SHORT blocked
        2. MORNING RTH PROTECTION - first 45 min of RTH with bullish setup = SHORT blocked
        3. SQUEEZE PROTECTION - RSI > 75 and MACD > 0.30 = SHORT blocked
        """
        
        # === RULE 1: NEVER SHORT ACCEPTANCE ===
        # If market is in bullish acceptance, shorting is suicide
        if result.ema_stack_up and result.macd_positive:
            result.short_blocked = True
            result.block_reasons.append(
                "ACCEPTANCE_BLOCK: EMA stack up + MACD positive = continuation, not reversal"
            )
        
        # === RULE 2: MORNING RTH PROTECTION ===
        if timestamp and session_type == "RTH":
            try:
                # Convert to local time if needed
                if hasattr(timestamp, 'tz') and timestamp.tz is not None:
                    local_ts = timestamp.astimezone()
                else:
                    local_ts = timestamp
                
                t = local_ts.time() if hasattr(local_ts, 'time') else time(0, 0)
                morning_cutoff = time(self.morning_protection_hour, self.morning_protection_minute)
                
                if t < morning_cutoff:
                    # Morning session: more cautious with shorts
                    if result.ema_stack_up and macd_hist > 0.15:
                        result.short_blocked = True
                        result.block_reasons.append(
                            f"MORNING_PROTECTION: Before {morning_cutoff} + bullish setup"
                        )
            except Exception as e:
                logger.warning(f"Morning protection check failed: {e}")
        
        # === RULE 3: SQUEEZE PROTECTION ===
        if is_squeeze:
            result.short_blocked = True
            result.block_reasons.append(
                f"SQUEEZE_BLOCK: RSI {result.rsi_value:.1f} > 75 + MACD > 0.30 = momentum squeeze"
            )
    
    def _determine_allowed_trades(
        self, result: MarketStateResult,
        bullish_acceptance: bool, bearish_acceptance: bool,
        bullish_exhaustion: bool, bearish_exhaustion: bool,
        is_squeeze: bool
    ) -> None:
        """Determine which trades are allowed based on market state.
        
        TRADE RULES:
        - ACCEPTANCE_UP: BUY allowed, SHORT blocked
        - ACCEPTANCE_DOWN: SHORT allowed, BUY blocked  
        - EXHAUSTION_UP: SHORT allowed (if not blocked), BUY on pullback
        - EXHAUSTION_DOWN: BUY allowed, SHORT on rally
        - SQUEEZE: No trades (fading squeeze is dangerous)
        - CHOP: Both allowed but low confidence
        """
        
        # Start with defaults
        result.allow_buy = True
        result.allow_short = True
        
        # Apply phase-based rules
        if bullish_acceptance:
            # Strong uptrend - only buy pullbacks
            result.allow_buy = True
            result.allow_short = False  # Will be blocked by hard rules anyway
            
        elif bearish_acceptance:
            # Strong downtrend - only sell rallies
            result.allow_buy = False
            result.allow_short = True
            
        elif bullish_exhaustion:
            # Market may be topping - shorts possible if exhaustion confirmed
            # BUT only if not blocked by hard rules
            result.allow_buy = True  # Could still buy dips if exhaustion fails
            result.allow_short = result.is_exhaustion and not result.short_blocked
            
        elif bearish_exhaustion:
            # Market may be bottoming - buys possible
            result.allow_buy = result.is_exhaustion
            result.allow_short = True  # Could still short rallies if exhaustion fails
            
        elif is_squeeze:
            # Momentum squeeze - avoid fading
            result.allow_buy = False  # Don't buy into parabolic
            result.allow_short = False  # Don't fade squeeze
            
        else:
            # Chop - both allowed but low confidence
            result.allow_buy = True
            result.allow_short = True
        
        # Apply hard blocks (override above)
        if result.short_blocked:
            result.allow_short = False
        if result.buy_blocked:
            result.allow_buy = False
    
    def _build_reasoning(self, result: MarketStateResult) -> str:
        """Build human-readable reasoning."""
        parts = []
        
        parts.append(f"Phase={result.phase.value}")
        parts.append(f"Trend={result.trend.value}")
        parts.append(f"TrendScore={result.trend_score:.0f}")
        
        if result.is_acceptance:
            parts.append("ACCEPTANCE")
        if result.is_exhaustion:
            parts.append("EXHAUSTION")
        
        parts.append(f"RSI={result.rsi_value:.1f}")
        
        if result.ema_stack_up:
            parts.append("EMA_STACK_UP")
        elif result.ema_stack_down:
            parts.append("EMA_STACK_DOWN")
        
        if result.macd_positive:
            parts.append("MACD+")
        elif result.macd_negative:
            parts.append("MACD-")
        
        parts.append(f"BUY={'✓' if result.allow_buy else '✗'}")
        parts.append(f"SHORT={'✓' if result.allow_short else '✗'}")
        
        if result.block_reasons:
            parts.append(f"BLOCKS={len(result.block_reasons)}")
        
        return " | ".join(parts)


def create_market_state_detector(config: Optional[Dict[str, Any]] = None) -> MarketStateDetector:
    """Factory function to create market state detector."""
    return MarketStateDetector(config)
