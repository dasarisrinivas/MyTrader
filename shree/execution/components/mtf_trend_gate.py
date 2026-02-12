"""Multi-Timeframe Trend Gate with State Machine.

Created: Jan 12, 2026
Purpose: Enforce proper multi-timeframe discipline to prevent:
- Immediate re-entry after position close
- 1-minute candles from determining trend direction
- Trades without 15m + 30m trend alignment

Core Principles:
1. Fresh Evaluation After Position Close
   - Reset all cached trend/signal/bias state after any position close
   - Treat next potential trade as completely fresh decision cycle

2. Multi-Timeframe Gating (Hard Rule)
   - ALL timeframes (1m, 5m, 15m, 30m) must be evaluated together
   - If any timeframe data is missing, stale, or contradictory -> NO TRADE

3. Trend Authority Hierarchy
   - 15m candle = PRIMARY trend authority
   - 30m candle = CONFIRMATION trend
   - 5m candle = Alignment and momentum confirmation
   - 1m candle = ENTRY TIMING ONLY (NEVER determines trend)

4. Trend Agreement Rules
   - LONG: 15m=bullish, 30m=bullish|neutral, 5m=bullish|pullback, 1m=entry trigger
   - SHORT: 15m=bearish, 30m=bearish|neutral, 5m=bearish|pullback, 1m=entry trigger

5. Cooldown After Position Close
   - Wait for at least ONE full 15-minute candle close before re-entry
   - Prevent immediate re-entry even if 1m signals fire
"""
from __future__ import annotations

import enum
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Tuple

from ...utils.logger import logger
from ...utils.timezone_utils import now_cst, format_cst


class TradingState(enum.Enum):
    """State machine states for trading flow."""
    IDLE = "IDLE"  # No position, not actively seeking entry
    WAITING_FOR_TREND_ALIGNMENT = "WAITING_FOR_TREND_ALIGNMENT"  # Waiting for MTF trends to align
    WAITING_FOR_ENTRY_TRIGGER = "WAITING_FOR_ENTRY_TRIGGER"  # Trends aligned, waiting for 1m trigger
    IN_POSITION = "IN_POSITION"  # Currently holding a position
    COOLDOWN = "COOLDOWN"  # Waiting for cooldown after position close


@dataclass
class TimeframeTrend:
    """Trend information for a single timeframe."""
    timeframe: str  # "1m", "5m", "15m", "30m"
    trend: str  # "UPTREND", "DOWNTREND", "NEUTRAL", "UNKNOWN"
    confidence: float  # 0.0 to 1.0
    last_updated: datetime = field(default_factory=now_cst)
    candle_close_time: Optional[datetime] = None  # When the last candle closed
    ema_value: Optional[float] = None
    price_vs_ema: Optional[str] = None  # "ABOVE", "BELOW", "AT"
    is_stale: bool = False
    
    @property
    def age_seconds(self) -> float:
        """How old is this trend data."""
        return (now_cst() - self.last_updated).total_seconds()
    
    def is_bullish(self) -> bool:
        return self.trend in ("UPTREND", "BULLISH", "MICRO_UP")
    
    def is_bearish(self) -> bool:
        return self.trend in ("DOWNTREND", "BEARISH", "MICRO_DOWN")
    
    def is_neutral(self) -> bool:
        return self.trend in ("NEUTRAL", "RANGING", "UNKNOWN")


@dataclass
class MTFTrendSnapshot:
    """Complete multi-timeframe trend snapshot."""
    tf_1m: TimeframeTrend
    tf_5m: TimeframeTrend
    tf_15m: TimeframeTrend
    tf_30m: TimeframeTrend
    timestamp: datetime = field(default_factory=now_cst)
    
    def is_complete(self) -> bool:
        """Check if all timeframes have valid data."""
        for tf in [self.tf_1m, self.tf_5m, self.tf_15m, self.tf_30m]:
            if tf.trend == "UNKNOWN" or tf.is_stale:
                return False
        return True
    
    def get_stale_timeframes(self) -> List[str]:
        """Return list of stale timeframes."""
        stale = []
        for tf in [self.tf_1m, self.tf_5m, self.tf_15m, self.tf_30m]:
            if tf.is_stale or tf.trend == "UNKNOWN":
                stale.append(tf.timeframe)
        return stale
    
    def check_long_alignment(self) -> Tuple[bool, str]:
        """Check if trends support a LONG trade.
        
        Rules:
        - 15m must be bullish (PRIMARY authority)
        - 30m must be bullish OR neutral (CONFIRMATION)
        - 5m must be bullish OR pullback within bullish structure
        - 1m provides entry trigger only (ignored for trend)
        """
        reasons = []
        
        # 15m is PRIMARY authority - must be bullish
        if not self.tf_15m.is_bullish():
            reasons.append(f"15m_NOT_BULLISH({self.tf_15m.trend})")
            return False, "; ".join(reasons)
        
        # 30m is CONFIRMATION - must be bullish or neutral
        if self.tf_30m.is_bearish():
            reasons.append(f"30m_BEARISH({self.tf_30m.trend})")
            return False, "; ".join(reasons)
        
        # 5m is alignment - must be bullish or neutral (pullback allowed)
        # Only block if 5m is strongly bearish
        if self.tf_5m.is_bearish() and self.tf_5m.confidence > 0.6:
            reasons.append(f"5m_STRONGLY_BEARISH({self.tf_5m.trend})")
            return False, "; ".join(reasons)
        
        return True, "LONG_ALIGNED"
    
    def check_short_alignment(self) -> Tuple[bool, str]:
        """Check if trends support a SHORT trade.
        
        Rules:
        - 15m must be bearish (PRIMARY authority)
        - 30m must be bearish OR neutral (CONFIRMATION)
        - 5m must be bearish OR pullback within bearish structure
        - 1m provides entry trigger only (ignored for trend)
        """
        reasons = []
        
        # 15m is PRIMARY authority - must be bearish
        if not self.tf_15m.is_bearish():
            reasons.append(f"15m_NOT_BEARISH({self.tf_15m.trend})")
            return False, "; ".join(reasons)
        
        # 30m is CONFIRMATION - must be bearish or neutral
        if self.tf_30m.is_bullish():
            reasons.append(f"30m_BULLISH({self.tf_30m.trend})")
            return False, "; ".join(reasons)
        
        # 5m is alignment - must be bearish or neutral (pullback allowed)
        # Only block if 5m is strongly bullish
        if self.tf_5m.is_bullish() and self.tf_5m.confidence > 0.6:
            reasons.append(f"5m_STRONGLY_BULLISH({self.tf_5m.trend})")
            return False, "; ".join(reasons)
        
        return True, "SHORT_ALIGNED"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "1m": {"trend": self.tf_1m.trend, "conf": self.tf_1m.confidence, "stale": self.tf_1m.is_stale},
            "5m": {"trend": self.tf_5m.trend, "conf": self.tf_5m.confidence, "stale": self.tf_5m.is_stale},
            "15m": {"trend": self.tf_15m.trend, "conf": self.tf_15m.confidence, "stale": self.tf_15m.is_stale},
            "30m": {"trend": self.tf_30m.trend, "conf": self.tf_30m.confidence, "stale": self.tf_30m.is_stale},
            "complete": self.is_complete(),
        }


@dataclass 
class PositionCloseEvent:
    """Record of a position close for cooldown tracking."""
    close_time: datetime
    close_reason: str  # "TP", "SL", "MANUAL", "TIMEOUT", "TREND_FLIP"
    direction: str  # "LONG", "SHORT"
    pnl: float
    candle_15m_at_close: Optional[datetime] = None  # 15m candle timestamp at close


class MTFTrendGate:
    """Multi-Timeframe Trend Gate with State Machine.
    
    This gate enforces:
    1. No trade without full MTF alignment (1m, 5m, 15m, 30m)
    2. 15m is the PRIMARY trend authority
    3. 1m NEVER determines trend direction
    4. Cooldown after position close (wait for 15m candle)
    5. Fresh evaluation after every position close
    
    State Machine:
    - IDLE: No position, not seeking entry
    - WAITING_FOR_TREND_ALIGNMENT: Seeking MTF alignment
    - WAITING_FOR_ENTRY_TRIGGER: Trends aligned, waiting for 1m entry
    - IN_POSITION: Holding a position
    - COOLDOWN: Post-close cooldown period
    """
    
    # Staleness thresholds per timeframe (seconds)
    STALE_THRESHOLDS = {
        "1m": 90,    # 1.5 minutes
        "5m": 360,   # 6 minutes
        "15m": 1080, # 18 minutes
        "30m": 2160, # 36 minutes
    }
    
    def __init__(
        self,
        min_15m_candles_after_close: int = 1,
        require_full_mtf: bool = True,
    ):
        """Initialize the MTF Trend Gate.
        
        Args:
            min_15m_candles_after_close: Minimum 15m candles to wait after position close
            require_full_mtf: If True, require all 4 timeframes; if False, only require 15m+5m
        """
        self.min_15m_candles_after_close = min_15m_candles_after_close
        self.require_full_mtf = require_full_mtf
        
        # State machine
        self._state = TradingState.IDLE
        self._state_changed_at = now_cst()
        
        # Trend data storage (keyed by timeframe)
        self._trends: Dict[str, TimeframeTrend] = {}
        
        # Position close tracking
        self._last_position_close: Optional[PositionCloseEvent] = None
        self._position_close_history: Deque[PositionCloseEvent] = deque(maxlen=50)
        
        # 15m candle tracking for cooldown
        self._last_15m_candle_close: Optional[datetime] = None
        self._15m_candles_since_close: int = 0
        
        # Decision cache (invalidated on state change)
        self._cached_snapshot: Optional[MTFTrendSnapshot] = None
        self._snapshot_valid_until: Optional[datetime] = None
        
        logger.info(
            f"MTFTrendGate initialized: require_full_mtf={require_full_mtf}, "
            f"min_15m_candles_after_close={min_15m_candles_after_close}"
        )
    
    @property
    def state(self) -> TradingState:
        return self._state
    
    @property
    def state_duration_seconds(self) -> float:
        return (now_cst() - self._state_changed_at).total_seconds()
    
    def _transition_state(self, new_state: TradingState, reason: str = "") -> None:
        """Transition to a new state with logging."""
        old_state = self._state
        if old_state == new_state:
            return
        
        self._state = new_state
        self._state_changed_at = now_cst()
        self._cached_snapshot = None  # Invalidate cache
        
        logger.info(
            f"📊 MTF STATE TRANSITION: {old_state.value} -> {new_state.value} "
            f"(reason: {reason})"
        )
    
    def update_trend(
        self,
        timeframe: str,
        trend: str,
        confidence: float = 0.5,
        ema_value: Optional[float] = None,
        price_vs_ema: Optional[str] = None,
        candle_close_time: Optional[datetime] = None,
    ) -> None:
        """Update trend data for a timeframe.
        
        Args:
            timeframe: "1m", "5m", "15m", or "30m"
            trend: "UPTREND", "DOWNTREND", "NEUTRAL", etc.
            confidence: 0.0 to 1.0
            ema_value: Current EMA value for this timeframe
            price_vs_ema: "ABOVE", "BELOW", or "AT"
            candle_close_time: When the candle closed that generated this trend
        """
        tf_trend = TimeframeTrend(
            timeframe=timeframe,
            trend=trend.upper(),
            confidence=confidence,
            last_updated=now_cst(),
            candle_close_time=candle_close_time,
            ema_value=ema_value,
            price_vs_ema=price_vs_ema,
            is_stale=False,
        )
        self._trends[timeframe] = tf_trend
        
        # Special handling for 15m candle close (cooldown tracking)
        if timeframe == "15m" and candle_close_time:
            if self._last_15m_candle_close != candle_close_time:
                self._last_15m_candle_close = candle_close_time
                if self._state == TradingState.COOLDOWN:
                    self._15m_candles_since_close += 1
                    logger.info(
                        f"📊 15m candle closed during cooldown: "
                        f"{self._15m_candles_since_close}/{self.min_15m_candles_after_close}"
                    )
        
        # Invalidate snapshot cache
        self._cached_snapshot = None
        
        logger.debug(
            f"MTF trend updated: {timeframe}={trend} (conf={confidence:.2f}, "
            f"ema={ema_value}, price_vs_ema={price_vs_ema})"
        )
    
    def _check_staleness(self) -> None:
        """Mark stale timeframes based on age thresholds."""
        for tf, threshold in self.STALE_THRESHOLDS.items():
            if tf in self._trends:
                trend = self._trends[tf]
                trend.is_stale = trend.age_seconds > threshold
    
    def get_trend_snapshot(self) -> MTFTrendSnapshot:
        """Get current multi-timeframe trend snapshot."""
        self._check_staleness()
        
        def get_or_default(tf: str) -> TimeframeTrend:
            if tf in self._trends:
                return self._trends[tf]
            return TimeframeTrend(
                timeframe=tf,
                trend="UNKNOWN",
                confidence=0.0,
                is_stale=True,
            )
        
        snapshot = MTFTrendSnapshot(
            tf_1m=get_or_default("1m"),
            tf_5m=get_or_default("5m"),
            tf_15m=get_or_default("15m"),
            tf_30m=get_or_default("30m"),
            timestamp=now_cst(),
        )
        
        self._cached_snapshot = snapshot
        return snapshot
    
    def on_position_opened(self, direction: str) -> None:
        """Called when a new position is opened.
        
        Args:
            direction: "LONG" or "SHORT"
        """
        self._transition_state(TradingState.IN_POSITION, f"Opened {direction} position")
        logger.info(f"📊 MTF Gate: Position opened ({direction})")
    
    def on_position_closed(
        self,
        close_reason: str,
        direction: str,
        pnl: float,
    ) -> None:
        """Called when a position is closed - triggers cooldown.
        
        Args:
            close_reason: "TP", "SL", "MANUAL", "TIMEOUT", "TREND_FLIP"
            direction: "LONG" or "SHORT"
            pnl: Realized P&L
        """
        close_event = PositionCloseEvent(
            close_time=now_cst(),
            close_reason=close_reason,
            direction=direction,
            pnl=pnl,
            candle_15m_at_close=self._last_15m_candle_close,
        )
        self._last_position_close = close_event
        self._position_close_history.append(close_event)
        
        # Reset 15m candle counter for cooldown
        self._15m_candles_since_close = 0
        
        # Clear all cached trend state for fresh evaluation
        self._clear_trend_state()
        
        self._transition_state(
            TradingState.COOLDOWN,
            f"Position closed ({close_reason}, pnl=${pnl:.2f})"
        )
        
        logger.info(
            f"📊 MTF Gate: Position closed ({close_reason}), "
            f"entering COOLDOWN (need {self.min_15m_candles_after_close} x 15m candles)"
        )
    
    def _clear_trend_state(self) -> None:
        """Clear all cached trend state for fresh evaluation after position close."""
        # Mark all trends as stale but preserve the data for reference
        for tf in self._trends.values():
            tf.is_stale = True
        
        self._cached_snapshot = None
        
        logger.info("📊 MTF Gate: Cleared trend state for fresh evaluation")
    
    def is_cooldown_complete(self) -> bool:
        """Check if cooldown period is complete."""
        if self._state != TradingState.COOLDOWN:
            return True
        
        # Need at least N 15m candle closes since position close
        if self._15m_candles_since_close >= self.min_15m_candles_after_close:
            return True
        
        # Fallback: if no 15m candles, use time-based (15 minutes)
        if self._last_position_close:
            elapsed = (now_cst() - self._last_position_close.close_time).total_seconds()
            if elapsed >= 15 * 60:  # 15 minutes minimum
                return True
        
        return False
    
    def get_cooldown_remaining(self) -> Tuple[int, str]:
        """Get cooldown remaining (candles needed, time estimate).
        
        Returns:
            Tuple of (candles_needed, time_estimate_string)
        """
        if self._state != TradingState.COOLDOWN:
            return 0, "Not in cooldown"
        
        candles_needed = max(0, self.min_15m_candles_after_close - self._15m_candles_since_close)
        
        if candles_needed > 0:
            time_estimate = f"~{candles_needed * 15} minutes"
        else:
            time_estimate = "Ready"
        
        return candles_needed, time_estimate
    
    def evaluate_entry(
        self,
        proposed_action: str,
        sentiment_bias: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Evaluate if an entry is allowed based on MTF trends and state.
        
        This is the MAIN gate method that enforces all rules.
        
        Args:
            proposed_action: "BUY" or "SELL"
            sentiment_bias: Optional sentiment direction ("BULLISH", "BEARISH", "NEUTRAL")
        
        Returns:
            Tuple of (allowed, reason, metadata)
        """
        metadata: Dict[str, Any] = {
            "state": self._state.value,
            "state_duration_seconds": self.state_duration_seconds,
        }
        
        # 1. Check state machine
        if self._state == TradingState.IN_POSITION:
            return False, "ALREADY_IN_POSITION", metadata
        
        if self._state == TradingState.COOLDOWN:
            if not self.is_cooldown_complete():
                candles_needed, time_est = self.get_cooldown_remaining()
                metadata["cooldown_candles_needed"] = candles_needed
                metadata["cooldown_time_estimate"] = time_est
                return False, f"COOLDOWN_ACTIVE({candles_needed} x 15m candles remaining)", metadata
            else:
                # Cooldown complete, transition to waiting for alignment
                self._transition_state(TradingState.WAITING_FOR_TREND_ALIGNMENT, "Cooldown complete")
        
        # 2. Get trend snapshot
        snapshot = self.get_trend_snapshot()
        metadata["trends"] = snapshot.to_dict()
        
        # 3. Check for missing/stale data
        stale_tfs = snapshot.get_stale_timeframes()
        if stale_tfs and self.require_full_mtf:
            metadata["stale_timeframes"] = stale_tfs

            # Observability: explain exactly why each timeframe is considered stale
            try:
                debug_rows = []
                for tf in ("1m", "5m", "15m", "30m"):
                    tf_obj = getattr(snapshot, f"tf_{tf}")
                    threshold = self.STALE_THRESHOLDS.get(tf)
                    debug_rows.append(
                        f"{tf}: trend={tf_obj.trend}, stale={tf_obj.is_stale}, age_s={tf_obj.age_seconds:.0f}, "
                        f"threshold_s={threshold}, last_updated={tf_obj.last_updated}, candle_close={tf_obj.candle_close_time}"
                    )
                logger.info(
                    "🧊 MTF stale breakdown -> {}",
                    " | ".join(debug_rows),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"MTF stale breakdown unavailable: {exc}")
            return False, f"STALE_TIMEFRAME_DATA({', '.join(stale_tfs)})", metadata
        
        # 4. Check MTF alignment (15m PRIMARY, 30m CONFIRMATION)
        is_buy = proposed_action.upper() in ("BUY", "SCALP_BUY")
        is_sell = proposed_action.upper() in ("SELL", "SCALP_SELL")
        
        if is_buy:
            aligned, align_reason = snapshot.check_long_alignment()
            metadata["alignment_check"] = "LONG"
        elif is_sell:
            aligned, align_reason = snapshot.check_short_alignment()
            metadata["alignment_check"] = "SHORT"
        else:
            return False, f"INVALID_ACTION({proposed_action})", metadata
        
        metadata["alignment_result"] = align_reason
        
        if not aligned:
            # Block with detailed reason
            return False, f"MTF_NOT_ALIGNED: {align_reason}", metadata
        
        # 5. Check sentiment agreement (if provided)
        if sentiment_bias:
            sentiment_agrees = self._check_sentiment_agreement(
                proposed_action, sentiment_bias
            )
            metadata["sentiment_check"] = {
                "bias": sentiment_bias,
                "agrees": sentiment_agrees,
            }
            
            if not sentiment_agrees:
                return False, f"SENTIMENT_DISAGREES({sentiment_bias} vs {proposed_action})", metadata
        
        # 6. All checks passed - update state
        if self._state != TradingState.WAITING_FOR_ENTRY_TRIGGER:
            self._transition_state(
                TradingState.WAITING_FOR_ENTRY_TRIGGER,
                f"MTF aligned for {proposed_action}"
            )
        
        # Log the successful alignment
        logger.info(
            f"✅ MTF ENTRY ALLOWED: {proposed_action} "
            f"(15m={snapshot.tf_15m.trend}, 30m={snapshot.tf_30m.trend}, "
            f"5m={snapshot.tf_5m.trend})"
        )
        
        return True, "ENTRY_ALLOWED", metadata
    
    def _check_sentiment_agreement(
        self,
        proposed_action: str,
        sentiment_bias: str,
    ) -> bool:
        """Check if sentiment agrees with proposed action.
        
        Rules:
        - BUY requires BULLISH or NEUTRAL sentiment
        - SELL requires BEARISH or NEUTRAL sentiment
        - NEUTRAL sentiment allows any direction
        """
        sentiment = sentiment_bias.upper()
        action = proposed_action.upper()
        
        if sentiment == "NEUTRAL":
            return True
        
        if action in ("BUY", "SCALP_BUY"):
            return sentiment == "BULLISH"
        
        if action in ("SELL", "SCALP_SELL"):
            return sentiment == "BEARISH"
        
        return True  # Unknown action, allow
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get current state summary for logging/monitoring."""
        snapshot = self.get_trend_snapshot()
        
        cooldown_candles, cooldown_time = (0, "N/A")
        if self._state == TradingState.COOLDOWN:
            cooldown_candles, cooldown_time = self.get_cooldown_remaining()
        
        return {
            "state": self._state.value,
            "state_duration_seconds": self.state_duration_seconds,
            "trends": snapshot.to_dict(),
            "is_mtf_complete": snapshot.is_complete(),
            "stale_timeframes": snapshot.get_stale_timeframes(),
            "cooldown": {
                "candles_remaining": cooldown_candles,
                "time_estimate": cooldown_time,
                "15m_candles_since_close": self._15m_candles_since_close,
            },
            "last_position_close": {
                "time": self._last_position_close.close_time.isoformat() if self._last_position_close else None,
                "reason": self._last_position_close.close_reason if self._last_position_close else None,
            },
        }
    
    def reset(self, reason: str = "manual_reset") -> None:
        """Reset the gate to IDLE state."""
        self._trends.clear()
        self._cached_snapshot = None
        self._last_position_close = None
        self._15m_candles_since_close = 0
        self._transition_state(TradingState.IDLE, reason)
        logger.warning(f"📊 MTF Gate RESET: {reason}")
