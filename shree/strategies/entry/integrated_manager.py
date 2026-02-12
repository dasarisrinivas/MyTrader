"""Integrated entry manager — routes signals to the correct entry module.

Provides a unified interface for evaluating market state and generating
entry signals across all entry modules (BUY continuation, SHORT continuation,
evening continuation, range reversion, sell exhaustion).

Inputs:
    data (dict): Current bar with indicators.
    timestamp: Current time.
    prev_data: Previous bar data.
    recent_bars: Recent price history.
    session_type: "RTH" or "OVERNIGHT".

Outputs:
    Tuple[EntrySignal, MarketStateResult] — the chosen signal and current state.

Side-effects:
    Mutates internal tracking state (consecutive holds, daily trend gate, etc.).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from ..market_state import (
    MarketStateResult,
    MarketPhase,
    TrendDirection,
    create_market_state_detector,
)
from .session_time import SessionWindow, SessionTimeManager
from .signals import EntrySignal
from .buy_continuation import BuyContinuationModule
from .short_continuation import ShortContinuationModule
from .sell_exhaustion import SellExhaustionModule
from .evening_buy import EveningContinuationModule
from .evening_sell import EveningSellContinuationModule


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
            from shree.strategies.range_reversion_module import RangeReversionModule
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
