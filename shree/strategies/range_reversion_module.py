"""
Range Reversion Module - Mean Reversion Trading for CHOP/RANGE Days

This module trades mean reversion ONLY on CHOP/RANGE days:
- Regime-isolated: Only active when NO trend detected
- Capital-safe: Strict stop losses and position sizing
- Low-frequency: Max 3 trades/day with cooldown after losses

ACTIVATION REQUIREMENTS (ALL must be met):
1. trend in [RANGE, CHOP]
2. ADX < 22
3. ema_stack_flat == True
4. is_acceptance == False
5. is_exhaustion == False
6. range_width <= 30.0 points (last 60-90 mins)
7. Time: 09:45-14:30 CST

TRADE LOCATION (EXTREMES ONLY):
- LONG: price <= VWAP - 1.0*ATR OR near session low, RSI <= 35 rising
- SHORT: price >= VWAP + 1.0*ATR OR near session high, RSI >= 65 falling
- NEVER trade middle 40% of range

Author: Shree Strategy Team
Created: 2026-01-29
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

# Import from market_state and entry_modules
from shree.strategies.market_state import (
    MarketStateResult,
    MarketPhase,
    TrendDirection,
)


class RangeTradeType(Enum):
    """Type of range reversion trade."""
    LONG_RANGE = "LONG_RANGE"
    SHORT_RANGE = "SHORT_RANGE"
    NO_SIGNAL = "NO_SIGNAL"


class RangePattern(Enum):
    """Entry confirmation pattern types."""
    REJECTION_WICK = "REJECTION_WICK"
    INSIDE_BAR_BREAK = "INSIDE_BAR_BREAK"
    MICRO_DOUBLE_BOTTOM = "MICRO_DOUBLE_BOTTOM"
    MICRO_DOUBLE_TOP = "MICRO_DOUBLE_TOP"
    VOLUME_DIVERGENCE = "VOLUME_DIVERGENCE"
    NO_PATTERN = "NO_PATTERN"


@dataclass
class RangeEntrySignal:
    """Signal output from RangeReversionModule."""
    action: str = "HOLD"  # "BUY", "SELL", "HOLD"
    confidence: float = 0.0
    reason: str = ""
    entry_type: str = "NO_SIGNAL"
    trade_type: RangeTradeType = RangeTradeType.NO_SIGNAL
    pattern: RangePattern = RangePattern.NO_PATTERN
    session_window: str = "RANGE_REVERSION"
    
    # Risk parameters
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None  # VWAP target
    take_profit_2: Optional[float] = None  # VWAP +/- 0.25 ATR
    risk_reward: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_actionable(self) -> bool:
        """Check if signal is tradeable."""
        return self.action in ["BUY", "SELL"] and self.confidence >= 0.65


@dataclass
class RangeStructure:
    """Tracks range structure for the session."""
    session_high: float = 0.0
    session_low: float = float('inf')
    range_width: float = 0.0
    range_mid: float = 0.0
    upper_40_boundary: float = 0.0  # Top of middle 40%
    lower_40_boundary: float = 0.0  # Bottom of middle 40%
    vwap: float = 0.0
    is_valid_range: bool = False


@dataclass  
class TradeTracking:
    """Tracks daily trades and loss streaks."""
    long_trades_today: int = 0
    short_trades_today: int = 0
    total_trades_today: int = 0
    consecutive_losses: int = 0
    last_loss_time: Optional[datetime] = None
    last_trade_date: Optional[str] = None  # Reset tracking on new day
    blocked_until: Optional[datetime] = None


class RangeReversionModule:
    """
    Mean Reversion Module for CHOP/RANGE Days
    
    This module is COMPLETELY ISOLATED from trend-following logic:
    - Only activates when market is in RANGE/CHOP with low ADX
    - Trades at extremes only (near VWAP bands or session highs/lows)
    - Requires entry confirmation patterns
    - Strict risk management (2.0-2.5 pt stops, 1.5 R:R minimum)
    
    PRIORITY: Evaluated ONLY if continuation modules return NO_SIGNAL.
    """
    
    # =========================================================================
    # CONFIGURATION CONSTANTS (NON-NEGOTIABLE)
    # =========================================================================
    
    # Activation thresholds
    ADX_MAX = 22.0                    # Must be below this for ranging
    RANGE_WIDTH_MAX = 30.0            # Max range in points
    RANGE_LOOKBACK_BARS = 60          # ~60-90 min lookback for range calc
    
    # Time windows (CST)
    START_TIME = time(9, 45)          # No trades before 09:45 CST
    END_TIME = time(14, 30)           # No trades after 14:30 CST
    
    # Trade location thresholds
    VWAP_ATR_THRESHOLD = 1.0          # Must be 1.0 ATR from VWAP
    SESSION_EXTREME_THRESHOLD = 1.5   # pts from session high/low
    
    # RSI thresholds
    RSI_LONG_MAX = 35.0               # RSI <= 35 for longs
    RSI_SHORT_MIN = 65.0              # RSI >= 65 for shorts
    
    # Risk management
    MAX_STOP_LOSS = 2.5               # Maximum stop in points
    MIN_STOP_LOSS = 2.0               # Minimum stop in points
    MIN_RISK_REWARD = 1.5             # Minimum R:R ratio
    
    # Frequency controls
    MAX_TRADES_PER_SIDE = 2           # Max longs OR shorts per day
    MAX_TOTAL_TRADES = 3              # Max total trades per day
    COOLDOWN_AFTER_LOSS_MINUTES = 15  # Minutes to wait after loss
    MAX_CONSECUTIVE_LOSSES = 2        # Block after this many losses
    
    # Confidence model
    BASE_CONFIDENCE = 0.60
    CONFIDENCE_STRONG_REJECTION = 0.10
    CONFIDENCE_RSI_DIVERGENCE = 0.08
    CONFIDENCE_VOLUME_SPIKE = 0.05
    CONFIDENCE_WIDE_RANGE_PENALTY = -0.10  # If range > 28 pts
    MIN_CONFIDENCE = 0.65
    MAX_CONFIDENCE = 0.72
    
    # Entry filter
    VWAP_REENTRY_ATR_BAND = 0.25      # Must close within VWAP ± 0.25 ATR
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize RangeReversionModule."""
        config = config or {}
        
        # Override defaults from config
        self.adx_max = config.get("adx_max", self.ADX_MAX)
        self.range_width_max = config.get("range_width_max", self.RANGE_WIDTH_MAX)
        self.min_confidence = config.get("min_confidence", self.MIN_CONFIDENCE)
        self.max_confidence = config.get("max_confidence", self.MAX_CONFIDENCE)
        
        # Tracking
        self.trade_tracking = TradeTracking()
        self.range_structure = RangeStructure()
        
        # Statistics
        self.signals_generated = 0
        self.signals_blocked = 0
        self.activation_failures: Dict[str, int] = {}
        
        logger.info(
            f"[RANGE_REVERSION] Module initialized: "
            f"ADX<{self.adx_max}, range<{self.range_width_max}pts, "
            f"time={self.START_TIME}-{self.END_TIME}, "
            f"max_trades={self.MAX_TOTAL_TRADES}/day"
        )
    
    # =========================================================================
    # MAIN EVALUATION METHOD
    # =========================================================================
    
    def evaluate(
        self,
        market_state: MarketStateResult,
        data: Dict[str, Any],
        recent_bars: Optional[pd.DataFrame] = None,
        timestamp: Optional[datetime] = None,
        continuation_signal_active: bool = False
    ) -> RangeEntrySignal:
        """
        Evaluate for range reversion entry.
        
        Args:
            market_state: Current market state from MarketStateDetector
            data: Current bar data with indicators
            recent_bars: DataFrame of recent price history
            timestamp: Current timestamp
            continuation_signal_active: True if ANY continuation module has signal
            
        Returns:
            RangeEntrySignal with trade details or NO_SIGNAL
        """
        # Initialize metadata
        metadata = {
            "module": "RANGE_REVERSION",
            "timestamp": timestamp.isoformat() if timestamp else None,
        }
        
        # Reset tracking on new day
        self._check_day_reset(timestamp)
        
        # =====================================================================
        # PRIORITY GUARD: Skip if continuation module has signal
        # =====================================================================
        if continuation_signal_active:
            self.signals_blocked += 1
            self._track_failure("CONTINUATION_PRIORITY")
            return RangeEntrySignal(
                action="HOLD",
                confidence=0.0,
                reason="CONTINUATION_PRIORITY: Continuation module has valid signal",
                entry_type="NO_SIGNAL",
                metadata=metadata
            )
        
        # =====================================================================
        # GATE 1: TIME WINDOW CHECK
        # =====================================================================
        if timestamp:
            current_time = timestamp.time()
            
            if current_time < self.START_TIME:
                self.signals_blocked += 1
                self._track_failure("TOO_EARLY")
                return RangeEntrySignal(
                    action="HOLD",
                    confidence=0.0,
                    reason=f"TOO_EARLY: {current_time} < {self.START_TIME}",
                    entry_type="NO_SIGNAL",
                    metadata=metadata
                )
            
            if current_time >= self.END_TIME:
                self.signals_blocked += 1
                self._track_failure("TOO_LATE")
                return RangeEntrySignal(
                    action="HOLD",
                    confidence=0.0,
                    reason=f"TOO_LATE: {current_time} >= {self.END_TIME}",
                    entry_type="NO_SIGNAL",
                    metadata=metadata
                )
        
        # =====================================================================
        # GATE 2: TRADE FREQUENCY LIMITS
        # =====================================================================
        frequency_block = self._check_frequency_limits(timestamp)
        if frequency_block:
            self.signals_blocked += 1
            return RangeEntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=frequency_block,
                entry_type="NO_SIGNAL",
                metadata=metadata
            )
        
        # =====================================================================
        # GATE 3: MARKET REGIME CHECK (NON-NEGOTIABLE)
        # =====================================================================
        regime_check = self._check_market_regime(market_state, data)
        if regime_check:
            self.signals_blocked += 1
            return RangeEntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=regime_check,
                entry_type="NO_SIGNAL",
                metadata=metadata
            )
        
        # =====================================================================
        # GATE 4: RANGE STRUCTURE VALIDATION
        # =====================================================================
        if recent_bars is not None and len(recent_bars) >= 30:
            self._calculate_range_structure(recent_bars, data)
        else:
            # Use session high/low from data if no recent_bars
            self._calculate_range_from_data(data)
        
        if not self.range_structure.is_valid_range:
            self.signals_blocked += 1
            self._track_failure("INVALID_RANGE")
            return RangeEntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"INVALID_RANGE: width={self.range_structure.range_width:.1f} > {self.range_width_max}",
                entry_type="NO_SIGNAL",
                metadata=metadata
            )
        
        # Add range info to metadata
        metadata.update({
            "session_high": self.range_structure.session_high,
            "session_low": self.range_structure.session_low,
            "range_width": self.range_structure.range_width,
            "vwap": self.range_structure.vwap,
        })
        
        # =====================================================================
        # GATE 5: TRADE LOCATION (EXTREMES ONLY)
        # =====================================================================
        trade_type, location_reason = self._determine_trade_location(data)
        
        if trade_type == RangeTradeType.NO_SIGNAL:
            self.signals_blocked += 1
            self._track_failure("BAD_LOCATION")
            return RangeEntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"BAD_LOCATION: {location_reason}",
                entry_type="NO_SIGNAL",
                metadata=metadata
            )
        
        metadata["trade_type"] = trade_type.value
        metadata["location_reason"] = location_reason
        
        # =====================================================================
        # GATE 6: ENTRY CONFIRMATION PATTERN
        # =====================================================================
        pattern, pattern_confidence = self._detect_entry_pattern(
            data, recent_bars, trade_type
        )
        
        if pattern == RangePattern.NO_PATTERN:
            self.signals_blocked += 1
            self._track_failure("NO_PATTERN")
            return RangeEntrySignal(
                action="HOLD",
                confidence=0.0,
                reason="NO_PATTERN: No valid entry confirmation detected",
                entry_type="NO_SIGNAL",
                trade_type=trade_type,
                metadata=metadata
            )
        
        metadata["pattern"] = pattern.value
        
        # =====================================================================
        # GATE 7: VWAP RE-ENTRY / REJECTION FILTER
        # =====================================================================
        vwap_filter = self._check_vwap_reentry_filter(data, trade_type)
        if vwap_filter:
            self.signals_blocked += 1
            self._track_failure("VWAP_FILTER")
            return RangeEntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=vwap_filter,
                entry_type="NO_SIGNAL",
                trade_type=trade_type,
                pattern=pattern,
                metadata=metadata
            )
        
        # =====================================================================
        # CALCULATE RISK PARAMETERS
        # =====================================================================
        risk_params = self._calculate_risk_parameters(data, trade_type, recent_bars)
        
        if risk_params is None:
            self.signals_blocked += 1
            self._track_failure("BAD_RISK")
            return RangeEntrySignal(
                action="HOLD",
                confidence=0.0,
                reason="BAD_RISK: Stop too wide or R:R insufficient",
                entry_type="NO_SIGNAL",
                trade_type=trade_type,
                pattern=pattern,
                metadata=metadata
            )
        
        stop_loss, tp1, tp2, risk_reward = risk_params
        
        # =====================================================================
        # CALCULATE CONFIDENCE
        # =====================================================================
        confidence = self._calculate_confidence(
            data, pattern, pattern_confidence, recent_bars
        )
        
        # Check confidence bounds
        if confidence < self.min_confidence:
            self.signals_blocked += 1
            self._track_failure("LOW_CONFIDENCE")
            return RangeEntrySignal(
                action="HOLD",
                confidence=confidence,
                reason=f"LOW_CONFIDENCE: {confidence:.1%} < {self.min_confidence:.1%}",
                entry_type="NO_SIGNAL",
                trade_type=trade_type,
                pattern=pattern,
                metadata=metadata
            )
        
        # Cap at max confidence
        confidence = min(confidence, self.max_confidence)
        
        # =====================================================================
        # GENERATE SIGNAL
        # =====================================================================
        close = float(data.get("close", 0))
        rsi = float(data.get("rsi", 50))
        adx = float(data.get("adx", 25))
        vwap = float(data.get("vwap", close))
        vwap_distance = close - vwap
        
        action = "BUY" if trade_type == RangeTradeType.LONG_RANGE else "SELL"
        entry_type = f"RANGE_{trade_type.value}"
        
        # Update metadata
        metadata.update({
            "entry_price": close,
            "stop_loss": stop_loss,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
            "risk_reward": risk_reward,
            "rsi": rsi,
            "adx": adx,
            "vwap_distance": vwap_distance,
            "range_width": self.range_structure.range_width,
        })
        
        self.signals_generated += 1
        
        logger.info(
            f"[RANGE_REVERSION] [{trade_type.value}] SIGNAL GENERATED: "
            f"{action} | pattern={pattern.value} | conf={confidence:.1%} | "
            f"stop={abs(close - stop_loss):.2f}pts | R:R={risk_reward:.2f} | "
            f"RSI={rsi:.1f} | ADX={adx:.1f}"
        )
        
        return RangeEntrySignal(
            action=action,
            confidence=confidence,
            reason=(
                f"[RANGE_REVERSION] [{trade_type.value}] | "
                f"[{pattern.value}] | "
                f"RSI={rsi:.1f} | ADX={adx:.1f} | "
                f"range={self.range_structure.range_width:.1f}pts | "
                f"CONF={confidence:.0%}"
            ),
            entry_type=entry_type,
            trade_type=trade_type,
            pattern=pattern,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_reward=risk_reward,
            metadata=metadata
        )
    
    # =========================================================================
    # MARKET REGIME VALIDATION
    # =========================================================================
    
    def _check_market_regime(
        self,
        market_state: MarketStateResult,
        data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Check if market is in valid RANGE/CHOP regime.
        
        Returns:
            None if valid, error string if blocked
        """
        # Check trend direction
        trend = market_state.trend
        valid_trends = [
            TrendDirection.NEUTRAL,
            TrendDirection.WEAK_UP,
            TrendDirection.WEAK_DOWN,
        ]
        
        # Also check phase
        phase = market_state.phase
        
        # Block if in acceptance (trending)
        if market_state.is_acceptance:
            self._track_failure("IS_ACCEPTANCE")
            return f"IS_ACCEPTANCE: Market accepting trend direction (phase={phase.value})"
        
        # Block if in exhaustion (reversal setup for trend module)
        if market_state.is_exhaustion:
            self._track_failure("IS_EXHAUSTION")
            return f"IS_EXHAUSTION: Reversal setup active (phase={phase.value})"
        
        # Block if strong trend detected
        strong_trends = [
            TrendDirection.STRONG_TREND_UP,
            TrendDirection.STRONG_TREND_DOWN,
            TrendDirection.TREND_UP,
            TrendDirection.TREND_DOWN,
        ]
        if trend in strong_trends:
            self._track_failure("STRONG_TREND")
            return f"STRONG_TREND: trend={trend.value} (need RANGE/CHOP)"
        
        # Check ADX
        adx = float(data.get("adx", 30))
        if adx >= self.adx_max:
            self._track_failure("ADX_TOO_HIGH")
            return f"ADX_TOO_HIGH: {adx:.1f} >= {self.adx_max}"
        
        # Check EMA stack flatness
        ema9 = float(data.get("ema9", 0))
        ema21 = float(data.get("ema21", 0))
        ema50 = float(data.get("ema50", 0))
        
        if ema9 > 0 and ema21 > 0 and ema50 > 0:
            # Calculate EMA spread as % of price
            ema_spread = abs(ema9 - ema50) / ema50 * 100 if ema50 > 0 else 0
            
            # If EMAs are spread more than 0.3%, not flat enough
            if ema_spread > 0.30:
                self._track_failure("EMA_NOT_FLAT")
                return f"EMA_NOT_FLAT: EMA spread {ema_spread:.2f}% > 0.30%"
        
        # Check if phase suggests ranging
        ranging_phases = [
            MarketPhase.CHOP,
            MarketPhase.RESET,
            MarketPhase.SQUEEZE,
        ]
        
        if phase not in ranging_phases:
            # Allow if trend is weak and not accepting
            if trend not in valid_trends:
                self._track_failure("WRONG_PHASE")
                return f"WRONG_PHASE: phase={phase.value}, trend={trend.value}"
        
        return None  # All checks passed
    
    # =========================================================================
    # RANGE STRUCTURE CALCULATION
    # =========================================================================
    
    def _calculate_range_structure(
        self,
        recent_bars: pd.DataFrame,
        data: Dict[str, Any]
    ) -> None:
        """Calculate range structure from recent bars."""
        # Use last 60-90 bars for range calculation
        lookback = min(len(recent_bars), self.RANGE_LOOKBACK_BARS)
        range_bars = recent_bars.tail(lookback)
        
        session_high = float(range_bars["high"].max())
        session_low = float(range_bars["low"].min())
        range_width = session_high - session_low
        range_mid = (session_high + session_low) / 2
        
        # Calculate middle 40% boundaries
        # Middle 40% = 30% to 70% of range
        lower_40_boundary = session_low + (range_width * 0.30)
        upper_40_boundary = session_low + (range_width * 0.70)
        
        vwap = float(data.get("vwap", range_mid))
        
        self.range_structure = RangeStructure(
            session_high=session_high,
            session_low=session_low,
            range_width=range_width,
            range_mid=range_mid,
            upper_40_boundary=upper_40_boundary,
            lower_40_boundary=lower_40_boundary,
            vwap=vwap,
            is_valid_range=(range_width <= self.range_width_max and range_width > 5.0)
        )
    
    def _calculate_range_from_data(self, data: Dict[str, Any]) -> None:
        """Calculate range structure from current data (fallback)."""
        session_high = float(data.get("session_high", data.get("pdh", 0)))
        session_low = float(data.get("session_low", data.get("pdl", 0)))
        
        if session_high <= 0 or session_low <= 0:
            # Use current price as fallback
            close = float(data.get("close", 0))
            atr = float(data.get("atr", 5))
            session_high = close + atr * 2
            session_low = close - atr * 2
        
        range_width = session_high - session_low
        range_mid = (session_high + session_low) / 2
        
        lower_40_boundary = session_low + (range_width * 0.30)
        upper_40_boundary = session_low + (range_width * 0.70)
        
        vwap = float(data.get("vwap", range_mid))
        
        self.range_structure = RangeStructure(
            session_high=session_high,
            session_low=session_low,
            range_width=range_width,
            range_mid=range_mid,
            upper_40_boundary=upper_40_boundary,
            lower_40_boundary=lower_40_boundary,
            vwap=vwap,
            is_valid_range=(range_width <= self.range_width_max and range_width > 5.0)
        )
    
    # =========================================================================
    # TRADE LOCATION LOGIC
    # =========================================================================
    
    def _determine_trade_location(
        self,
        data: Dict[str, Any]
    ) -> Tuple[RangeTradeType, str]:
        """
        Determine if price is at a valid trade location (extremes only).
        
        Returns:
            Tuple of (RangeTradeType, reason_string)
        """
        close = float(data.get("close", 0))
        vwap = float(data.get("vwap", close))
        atr = float(data.get("atr", 5))
        rsi = float(data.get("rsi", 50))
        macd_hist = float(data.get("macd_hist", data.get("MACD_hist", 0)))
        
        # Get previous MACD for contraction check
        prev_macd = float(data.get("prev_macd_hist", macd_hist))
        
        # Calculate distances
        vwap_distance = close - vwap
        vwap_atr_distance = abs(vwap_distance) / atr if atr > 0 else 0
        
        distance_to_high = self.range_structure.session_high - close
        distance_to_low = close - self.range_structure.session_low
        
        # Check if in middle 40% (FORBIDDEN ZONE)
        if (self.range_structure.lower_40_boundary <= close <= 
            self.range_structure.upper_40_boundary):
            return RangeTradeType.NO_SIGNAL, "MID_RANGE: Price in middle 40% of range"
        
        # =====================================================================
        # LONG ZONE CHECK
        # =====================================================================
        in_long_zone = False
        long_reasons = []
        
        # Condition 1: Price <= VWAP - 1.0 * ATR
        if vwap_distance <= -self.VWAP_ATR_THRESHOLD * atr:
            in_long_zone = True
            long_reasons.append(f"VWAP_OVERSOLD ({vwap_atr_distance:.2f} ATR below)")
        
        # Condition 2: Near session low
        if distance_to_low <= self.SESSION_EXTREME_THRESHOLD:
            in_long_zone = True
            long_reasons.append(f"NEAR_SESSION_LOW ({distance_to_low:.2f}pts)")
        
        if in_long_zone:
            # RSI check: must be <= 35 AND rising
            rsi_rising = float(data.get("rsi_slope", 0)) > 0 or rsi > float(data.get("prev_rsi", rsi - 1))
            
            if rsi > self.RSI_LONG_MAX:
                return RangeTradeType.NO_SIGNAL, f"RSI_NOT_OVERSOLD: {rsi:.1f} > {self.RSI_LONG_MAX}"
            
            if not rsi_rising:
                return RangeTradeType.NO_SIGNAL, f"RSI_NOT_RISING: RSI={rsi:.1f} still falling"
            
            # MACD contraction check (bear momentum weakening)
            macd_contracting = macd_hist > prev_macd  # Less negative = contracting
            if not macd_contracting and macd_hist < -0.1:
                return RangeTradeType.NO_SIGNAL, "MACD_EXPANDING: Bear momentum still expanding"
            
            return RangeTradeType.LONG_RANGE, " | ".join(long_reasons)
        
        # =====================================================================
        # SHORT ZONE CHECK
        # =====================================================================
        in_short_zone = False
        short_reasons = []
        
        # Condition 1: Price >= VWAP + 1.0 * ATR
        if vwap_distance >= self.VWAP_ATR_THRESHOLD * atr:
            in_short_zone = True
            short_reasons.append(f"VWAP_OVERBOUGHT ({vwap_atr_distance:.2f} ATR above)")
        
        # Condition 2: Near session high
        if distance_to_high <= self.SESSION_EXTREME_THRESHOLD:
            in_short_zone = True
            short_reasons.append(f"NEAR_SESSION_HIGH ({distance_to_high:.2f}pts)")
        
        if in_short_zone:
            # RSI check: must be >= 65 AND falling
            rsi_falling = float(data.get("rsi_slope", 0)) < 0 or rsi < float(data.get("prev_rsi", rsi + 1))
            
            if rsi < self.RSI_SHORT_MIN:
                return RangeTradeType.NO_SIGNAL, f"RSI_NOT_OVERBOUGHT: {rsi:.1f} < {self.RSI_SHORT_MIN}"
            
            if not rsi_falling:
                return RangeTradeType.NO_SIGNAL, f"RSI_NOT_FALLING: RSI={rsi:.1f} still rising"
            
            # MACD contraction check (bull momentum weakening)
            macd_contracting = macd_hist < prev_macd  # Less positive = contracting
            if not macd_contracting and macd_hist > 0.1:
                return RangeTradeType.NO_SIGNAL, "MACD_EXPANDING: Bull momentum still expanding"
            
            return RangeTradeType.SHORT_RANGE, " | ".join(short_reasons)
        
        # Not at any extreme
        return RangeTradeType.NO_SIGNAL, "NOT_AT_EXTREME: Price not at valid entry zone"
    
    # =========================================================================
    # ENTRY PATTERN DETECTION
    # =========================================================================
    
    def _detect_entry_pattern(
        self,
        data: Dict[str, Any],
        recent_bars: Optional[pd.DataFrame],
        trade_type: RangeTradeType
    ) -> Tuple[RangePattern, float]:
        """
        Detect entry confirmation pattern.
        
        Returns:
            Tuple of (RangePattern, pattern_confidence_boost)
        """
        if recent_bars is None or len(recent_bars) < 5:
            return RangePattern.NO_PATTERN, 0.0
        
        # Get recent bars for pattern detection
        bars = recent_bars.tail(5)
        current_bar = bars.iloc[-1]
        prev_bar = bars.iloc[-2]
        
        close = float(current_bar.get("close", data.get("close", 0)))
        open_price = float(current_bar.get("open", close))
        high = float(current_bar.get("high", close))
        low = float(current_bar.get("low", close))
        
        body = abs(close - open_price)
        upper_wick = high - max(close, open_price)
        lower_wick = min(close, open_price) - low
        candle_range = high - low
        
        # =====================================================================
        # PATTERN 1: REJECTION WICK (PIN BAR)
        # =====================================================================
        if candle_range > 0:
            if trade_type == RangeTradeType.LONG_RANGE:
                # Long rejection: lower wick >= 2x body
                if lower_wick >= body * 2 and lower_wick >= candle_range * 0.5:
                    logger.debug(
                        f"[RANGE_REVERSION] REJECTION_WICK detected: "
                        f"lower_wick={lower_wick:.2f}, body={body:.2f}"
                    )
                    return RangePattern.REJECTION_WICK, self.CONFIDENCE_STRONG_REJECTION
            
            elif trade_type == RangeTradeType.SHORT_RANGE:
                # Short rejection: upper wick >= 2x body
                if upper_wick >= body * 2 and upper_wick >= candle_range * 0.5:
                    logger.debug(
                        f"[RANGE_REVERSION] REJECTION_WICK detected: "
                        f"upper_wick={upper_wick:.2f}, body={body:.2f}"
                    )
                    return RangePattern.REJECTION_WICK, self.CONFIDENCE_STRONG_REJECTION
        
        # =====================================================================
        # PATTERN 2: INSIDE BAR BREAK
        # =====================================================================
        prev_high = float(prev_bar.get("high", high))
        prev_low = float(prev_bar.get("low", low))
        
        # Current bar is inside previous bar
        is_inside_bar = high <= prev_high and low >= prev_low
        
        if is_inside_bar:
            if trade_type == RangeTradeType.LONG_RANGE and close > open_price:
                # Bullish inside bar break
                logger.debug("[RANGE_REVERSION] INSIDE_BAR_BREAK detected (bullish)")
                return RangePattern.INSIDE_BAR_BREAK, 0.05
            
            elif trade_type == RangeTradeType.SHORT_RANGE and close < open_price:
                # Bearish inside bar break
                logger.debug("[RANGE_REVERSION] INSIDE_BAR_BREAK detected (bearish)")
                return RangePattern.INSIDE_BAR_BREAK, 0.05
        
        # =====================================================================
        # PATTERN 3: MICRO DOUBLE BOTTOM/TOP (2-3 bars)
        # =====================================================================
        if len(bars) >= 3:
            lows_3 = [float(bars.iloc[i].get("low", 0)) for i in range(-3, 0)]
            highs_3 = [float(bars.iloc[i].get("high", 0)) for i in range(-3, 0)]
            
            if trade_type == RangeTradeType.LONG_RANGE:
                # Double bottom: two similar lows within 1 pt
                if len(lows_3) >= 2:
                    min_low = min(lows_3)
                    second_low = sorted(lows_3)[1] if len(lows_3) > 1 else min_low
                    if abs(min_low - second_low) <= 1.0:
                        logger.debug(
                            f"[RANGE_REVERSION] MICRO_DOUBLE_BOTTOM detected: "
                            f"lows={lows_3}"
                        )
                        return RangePattern.MICRO_DOUBLE_BOTTOM, 0.06
            
            elif trade_type == RangeTradeType.SHORT_RANGE:
                # Double top: two similar highs within 1 pt
                if len(highs_3) >= 2:
                    max_high = max(highs_3)
                    second_high = sorted(highs_3, reverse=True)[1] if len(highs_3) > 1 else max_high
                    if abs(max_high - second_high) <= 1.0:
                        logger.debug(
                            f"[RANGE_REVERSION] MICRO_DOUBLE_TOP detected: "
                            f"highs={highs_3}"
                        )
                        return RangePattern.MICRO_DOUBLE_TOP, 0.06
        
        # =====================================================================
        # PATTERN 4: VOLUME DIVERGENCE
        # =====================================================================
        if "volume" in bars.columns:
            volumes = bars["volume"].values
            avg_volume = np.mean(volumes[:-1]) if len(volumes) > 1 else volumes[-1]
            current_volume = float(volumes[-1])
            
            # Volume spike at extreme (1.5x average)
            if current_volume >= avg_volume * 1.5:
                logger.debug(
                    f"[RANGE_REVERSION] VOLUME_DIVERGENCE detected: "
                    f"vol={current_volume:.0f}, avg={avg_volume:.0f}"
                )
                return RangePattern.VOLUME_DIVERGENCE, self.CONFIDENCE_VOLUME_SPIKE
        
        return RangePattern.NO_PATTERN, 0.0
    
    # =========================================================================
    # VWAP RE-ENTRY / REJECTION FILTER
    # =========================================================================
    
    def _check_vwap_reentry_filter(
        self,
        data: Dict[str, Any],
        trade_type: RangeTradeType
    ) -> Optional[str]:
        """
        Check if price shows rejection back toward VWAP.
        
        For range reversion, we want to see:
        - LONG: Close above open (bullish candle) AND close > low + 40% of candle range
        - SHORT: Close below open (bearish candle) AND close < high - 40% of candle range
        
        This confirms the rejection is happening at the extreme.
        
        Returns:
            None if filter passes, error string if blocked
        """
        close = float(data.get("close", 0))
        open_price = float(data.get("open", close))
        high = float(data.get("high", close))
        low = float(data.get("low", close))
        vwap = float(data.get("vwap", close))
        
        candle_range = high - low
        if candle_range <= 0:
            return "VWAP_FILTER: Invalid candle (no range)"
        
        if trade_type == RangeTradeType.LONG_RANGE:
            # For LONG: Need bullish candle showing rejection from lows
            # Close should be in upper 60% of candle
            close_position = (close - low) / candle_range
            
            if close < open_price:
                # Bearish candle at lows - no rejection yet
                return f"VWAP_FILTER: Bearish candle at lows (no rejection)"
            
            if close_position < 0.40:
                # Close too close to low - weak rejection
                return f"VWAP_FILTER: Weak rejection (close at {close_position:.0%} of range)"
            
            # Good rejection - close in upper 60% with bullish candle
            return None
            
        elif trade_type == RangeTradeType.SHORT_RANGE:
            # For SHORT: Need bearish candle showing rejection from highs
            # Close should be in lower 60% of candle
            close_position = (close - low) / candle_range
            
            if close > open_price:
                # Bullish candle at highs - no rejection yet
                return f"VWAP_FILTER: Bullish candle at highs (no rejection)"
            
            if close_position > 0.60:
                # Close too close to high - weak rejection
                return f"VWAP_FILTER: Weak rejection (close at {close_position:.0%} of range)"
            
            # Good rejection - close in lower 60% with bearish candle
            return None
        
        return None
    
    # =========================================================================
    # RISK PARAMETER CALCULATION
    # =========================================================================
    
    def _calculate_risk_parameters(
        self,
        data: Dict[str, Any],
        trade_type: RangeTradeType,
        recent_bars: Optional[pd.DataFrame]
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Calculate stop loss and take profit levels.
        
        Returns:
            Tuple of (stop_loss, tp1, tp2, risk_reward) or None if invalid
        """
        close = float(data.get("close", 0))
        vwap = float(data.get("vwap", close))
        atr = float(data.get("atr", 5))
        
        # Find rejection wick or session extreme for stop placement
        if recent_bars is not None and len(recent_bars) >= 3:
            recent_3 = recent_bars.tail(3)
            swing_low = float(recent_3["low"].min())
            swing_high = float(recent_3["high"].max())
        else:
            swing_low = close - atr
            swing_high = close + atr
        
        if trade_type == RangeTradeType.LONG_RANGE:
            # Stop below swing low or session low + buffer
            stop_candidate_1 = swing_low - 0.5  # Below recent swing
            stop_candidate_2 = self.range_structure.session_low - 0.5  # Below session low
            stop_loss = max(stop_candidate_1, stop_candidate_2)  # Use tighter stop
            
            stop_distance = close - stop_loss
            
            # Validate stop distance
            if stop_distance > self.MAX_STOP_LOSS:
                # Try to use a tighter stop
                stop_loss = close - self.MAX_STOP_LOSS
                stop_distance = self.MAX_STOP_LOSS
            
            if stop_distance < self.MIN_STOP_LOSS:
                stop_loss = close - self.MIN_STOP_LOSS
                stop_distance = self.MIN_STOP_LOSS
            
            # Targets
            tp1 = vwap  # VWAP target
            tp2 = vwap + (atr * self.VWAP_REENTRY_ATR_BAND)  # VWAP + 0.25 ATR
            
            # Calculate R:R based on TP1 (conservative target)
            reward = tp1 - close
            risk_reward = reward / stop_distance if stop_distance > 0 else 0
            
        else:  # SHORT_RANGE
            # Stop above swing high or session high + buffer
            stop_candidate_1 = swing_high + 0.5  # Above recent swing
            stop_candidate_2 = self.range_structure.session_high + 0.5  # Above session high
            stop_loss = min(stop_candidate_1, stop_candidate_2)  # Use tighter stop
            
            stop_distance = stop_loss - close
            
            # Validate stop distance
            if stop_distance > self.MAX_STOP_LOSS:
                stop_loss = close + self.MAX_STOP_LOSS
                stop_distance = self.MAX_STOP_LOSS
            
            if stop_distance < self.MIN_STOP_LOSS:
                stop_loss = close + self.MIN_STOP_LOSS
                stop_distance = self.MIN_STOP_LOSS
            
            # Targets
            tp1 = vwap  # VWAP target
            tp2 = vwap - (atr * self.VWAP_REENTRY_ATR_BAND)  # VWAP - 0.25 ATR
            
            # Calculate R:R based on TP1 (conservative target)
            reward = close - tp1
            risk_reward = reward / stop_distance if stop_distance > 0 else 0
        
        # Validate risk/reward
        if risk_reward < self.MIN_RISK_REWARD:
            logger.debug(
                f"[RANGE_REVERSION] R:R too low: {risk_reward:.2f} < {self.MIN_RISK_REWARD}"
            )
            return None
        
        # Final stop distance check
        final_stop_distance = abs(close - stop_loss)
        if final_stop_distance > self.MAX_STOP_LOSS:
            logger.debug(
                f"[RANGE_REVERSION] Stop too wide: {final_stop_distance:.2f} > {self.MAX_STOP_LOSS}"
            )
            return None
        
        return stop_loss, tp1, tp2, risk_reward
    
    # =========================================================================
    # CONFIDENCE CALCULATION
    # =========================================================================
    
    def _calculate_confidence(
        self,
        data: Dict[str, Any],
        pattern: RangePattern,
        pattern_boost: float,
        recent_bars: Optional[pd.DataFrame]
    ) -> float:
        """Calculate signal confidence (SEPARATE from continuation logic)."""
        confidence = self.BASE_CONFIDENCE
        
        # Add pattern boost
        confidence += pattern_boost
        
        # RSI divergence check
        rsi = float(data.get("rsi", 50))
        prev_rsi = float(data.get("prev_rsi", rsi))
        close = float(data.get("close", 0))
        
        if recent_bars is not None and len(recent_bars) >= 3:
            # Check for RSI divergence
            prices = recent_bars.tail(3)["close"].values
            if "rsi" in recent_bars.columns:
                rsis = recent_bars.tail(3)["rsi"].values
                
                # Bullish divergence: price making lower lows, RSI making higher lows
                if prices[-1] < prices[0] and rsis[-1] > rsis[0]:
                    confidence += self.CONFIDENCE_RSI_DIVERGENCE
                
                # Bearish divergence: price making higher highs, RSI making lower highs
                elif prices[-1] > prices[0] and rsis[-1] < rsis[0]:
                    confidence += self.CONFIDENCE_RSI_DIVERGENCE
        
        # Volume spike bonus
        if pattern == RangePattern.VOLUME_DIVERGENCE:
            confidence += self.CONFIDENCE_VOLUME_SPIKE
        
        # Wide range penalty
        if self.range_structure.range_width > 28.0:
            confidence += self.CONFIDENCE_WIDE_RANGE_PENALTY
        
        return confidence
    
    # =========================================================================
    # FREQUENCY LIMIT CHECKS
    # =========================================================================
    
    def _check_frequency_limits(
        self,
        timestamp: Optional[datetime]
    ) -> Optional[str]:
        """
        Check trade frequency limits.
        
        Returns:
            None if allowed, error string if blocked
        """
        # Check consecutive loss block
        if self.trade_tracking.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            self._track_failure("CONSECUTIVE_LOSSES")
            return (
                f"CONSECUTIVE_LOSSES: {self.trade_tracking.consecutive_losses} losses, "
                f"trading blocked"
            )
        
        # Check cooldown after loss
        if (self.trade_tracking.last_loss_time and timestamp and
            self.trade_tracking.blocked_until):
            if timestamp < self.trade_tracking.blocked_until:
                remaining = (self.trade_tracking.blocked_until - timestamp).seconds // 60
                self._track_failure("LOSS_COOLDOWN")
                return f"LOSS_COOLDOWN: {remaining} minutes remaining"
        
        # Check total trades per day
        if self.trade_tracking.total_trades_today >= self.MAX_TOTAL_TRADES:
            self._track_failure("MAX_DAILY_TRADES")
            return f"MAX_DAILY_TRADES: {self.trade_tracking.total_trades_today} trades today"
        
        return None
    
    def _check_day_reset(self, timestamp: Optional[datetime]) -> None:
        """Reset tracking on new day."""
        if timestamp:
            current_date = timestamp.strftime("%Y-%m-%d")
            if self.trade_tracking.last_trade_date != current_date:
                self.trade_tracking = TradeTracking(last_trade_date=current_date)
                logger.info(f"[RANGE_REVERSION] Day reset: {current_date}")
    
    # =========================================================================
    # TRADE RESULT TRACKING
    # =========================================================================
    
    def record_trade_result(
        self,
        trade_type: RangeTradeType,
        is_winner: bool,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record trade result for frequency tracking."""
        if trade_type == RangeTradeType.LONG_RANGE:
            self.trade_tracking.long_trades_today += 1
        elif trade_type == RangeTradeType.SHORT_RANGE:
            self.trade_tracking.short_trades_today += 1
        
        self.trade_tracking.total_trades_today += 1
        
        if is_winner:
            self.trade_tracking.consecutive_losses = 0
            self.trade_tracking.blocked_until = None
        else:
            self.trade_tracking.consecutive_losses += 1
            self.trade_tracking.last_loss_time = timestamp
            if timestamp:
                self.trade_tracking.blocked_until = (
                    timestamp + timedelta(minutes=self.COOLDOWN_AFTER_LOSS_MINUTES)
                )
            
            logger.warning(
                f"[RANGE_REVERSION] Loss recorded: "
                f"consecutive={self.trade_tracking.consecutive_losses}, "
                f"cooldown until {self.trade_tracking.blocked_until}"
            )
    
    # =========================================================================
    # STATISTICS TRACKING
    # =========================================================================
    
    def _track_failure(self, reason: str) -> None:
        """Track activation failure reasons."""
        self.activation_failures[reason] = self.activation_failures.get(reason, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get module statistics."""
        return {
            "signals_generated": self.signals_generated,
            "signals_blocked": self.signals_blocked,
            "activation_failures": self.activation_failures,
            "trades_today": {
                "long": self.trade_tracking.long_trades_today,
                "short": self.trade_tracking.short_trades_today,
                "total": self.trade_tracking.total_trades_today,
            },
            "consecutive_losses": self.trade_tracking.consecutive_losses,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_range_reversion_module(
    config: Optional[Dict[str, Any]] = None
) -> RangeReversionModule:
    """Factory function to create RangeReversionModule."""
    return RangeReversionModule(config)
