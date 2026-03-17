"""Event Detector for triggering Bedrock analysis.

This module detects specific events that should trigger Bedrock LLM analysis:
- Market open summary (X minutes after open)
- Market close summary
- Volatility spike (configurable threshold)
- News keyword detection (CPI, FOMC, rate, inflation, recession, etc.)
- Manual trigger (via Flask endpoint)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, time as dt_time, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import logger


@dataclass
class EventPayload:
    """Payload for Bedrock analysis trigger."""
    trigger_type: str
    reason: str
    timestamp: datetime
    
    # Market data
    symbol: str = "MES"
    current_price: float = 0.0
    price_change_pct: float = 0.0
    
    # Indicators (last N minutes)
    momentum: float = 0.0
    atr: float = 0.0
    volatility: float = 0.0
    rsi: float = 50.0
    vix: Optional[float] = None
    
    # Position state
    position: int = 0
    unrealized_pnl: float = 0.0
    
    # News
    news_headlines: List[str] = field(default_factory=list)
    
    # Recent price history (for context)
    recent_prices: List[float] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trigger_type": self.trigger_type,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "current_price": self.current_price,
            "price_change_pct": self.price_change_pct,
            "momentum": self.momentum,
            "atr": self.atr,
            "volatility": self.volatility,
            "rsi": self.rsi,
            "vix": self.vix,
            "position": self.position,
            "unrealized_pnl": self.unrealized_pnl,
            "news_headlines": self.news_headlines,
            "recent_prices": self.recent_prices[-20:],  # Last 20 prices
            "metadata": self.metadata,
        }


class EventDetector:
    """Detects events that should trigger Bedrock analysis.
    
    The detector runs on each tick loop iteration but only triggers
    Bedrock calls when specific conditions are met.
    """
    
    # Market hours for ES/MES futures (Chicago time, UTC-6 winter / UTC-5 summer)
    # ES trades nearly 24 hours, but we focus on RTH: 8:30 AM - 3:00 PM CT
    MARKET_OPEN_HOUR = 8
    MARKET_OPEN_MINUTE = 30
    MARKET_CLOSE_HOUR = 15
    MARKET_CLOSE_MINUTE = 0
    
    # News keywords that should trigger analysis
    NEWS_KEYWORDS = [
        "cpi", "fomc", "rate", "inflation", "recession",
        "fed", "powell", "jobs", "employment", "gdp",
        "treasury", "yield", "tariff", "trade war",
        "earnings", "guidance", "outlook",
    ]
    
    def __init__(
        self,
        symbol: str = "MES",
        volatility_spike_threshold: float = 2.0,  # ATR multiplier
        minutes_after_open: int = 5,
        minutes_before_close: int = 5,
        min_interval_seconds: int = 60,
        cooldown_seconds: int = 300,  # 5 minutes between same trigger type
    ):
        """Initialize event detector.
        
        Args:
            symbol: Trading symbol (MES or ES)
            volatility_spike_threshold: ATR multiplier for volatility spike detection
            minutes_after_open: Minutes after market open for summary trigger
            minutes_before_close: Minutes before market close for summary trigger
            min_interval_seconds: Minimum interval between any Bedrock calls
            cooldown_seconds: Cooldown between same trigger type
        """
        self.symbol = symbol
        self.volatility_spike_threshold = volatility_spike_threshold
        self.minutes_after_open = minutes_after_open
        self.minutes_before_close = minutes_before_close
        self.min_interval_seconds = min_interval_seconds
        self.cooldown_seconds = cooldown_seconds
        
        # Tracking state
        self._last_trigger_time: Optional[datetime] = None
        self._last_trigger_type: Dict[str, datetime] = {}  # {trigger_type: last_time}
        self._market_open_triggered_today = False
        self._market_close_triggered_today = False
        self._last_check_date: Optional[str] = None
        self._manual_trigger_pending = False
        self._manual_trigger_notes: Optional[str] = None
        
        # Baseline volatility (rolling average)
        self._baseline_atr: Optional[float] = None
        self._atr_history: List[float] = []
        
        logger.info(
            f"EventDetector initialized: symbol={symbol}, "
            f"vol_spike_threshold={volatility_spike_threshold}x ATR"
        )
    
    def _reset_daily_state(self) -> None:
        """Reset daily state flags."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        if self._last_check_date != today:
            self._market_open_triggered_today = False
            self._market_close_triggered_today = False
            self._last_check_date = today
            logger.debug(f"Reset daily state for {today}")
    
    def _is_within_cooldown(self, trigger_type: str) -> bool:
        """Check if trigger type is within cooldown period.
        
        Args:
            trigger_type: Type of trigger
            
        Returns:
            True if within cooldown
        """
        if trigger_type in self._last_trigger_type:
            elapsed = (datetime.now(timezone.utc) - self._last_trigger_type[trigger_type]).total_seconds()
            if elapsed < self.cooldown_seconds:
                return True
        return False
    
    def _is_within_min_interval(self) -> bool:
        """Check if within minimum interval since last trigger.
        
        Returns:
            True if within minimum interval
        """
        if self._last_trigger_time:
            elapsed = (datetime.now(timezone.utc) - self._last_trigger_time).total_seconds()
            if elapsed < self.min_interval_seconds:
                return True
        return False
    
    def _record_trigger(self, trigger_type: str) -> None:
        """Record trigger for cooldown tracking.
        
        Args:
            trigger_type: Type of trigger
        """
        now = datetime.now(timezone.utc)
        self._last_trigger_time = now
        self._last_trigger_type[trigger_type] = now
    
    def _get_market_times(self) -> Tuple[dt_time, dt_time]:
        """Get market open and close times in UTC.
        
        Returns:
            Tuple of (open_time, close_time) in UTC
        """
        # Approximate Chicago to UTC conversion (6 hours in winter, 5 in summer)
        # For simplicity, using 6 hours offset
        utc_offset = 6
        
        open_hour = (self.MARKET_OPEN_HOUR + utc_offset) % 24
        close_hour = (self.MARKET_CLOSE_HOUR + utc_offset) % 24
        
        return (
            dt_time(open_hour, self.MARKET_OPEN_MINUTE),
            dt_time(close_hour, self.MARKET_CLOSE_MINUTE)
        )
    
    def _check_market_open_trigger(self, now: datetime) -> Optional[str]:
        """Check if market open trigger should fire.
        
        Args:
            now: Current UTC time
            
        Returns:
            Trigger reason or None
        """
        if self._market_open_triggered_today:
            return None
        
        open_time, _ = self._get_market_times()
        current_time = now.time()
        
        # Calculate trigger window (minutes_after_open after market open)
        trigger_time = dt_time(
            open_time.hour,
            open_time.minute + self.minutes_after_open
        )
        
        # Check if we're in the trigger window (within 2 minutes)
        trigger_dt = now.replace(
            hour=trigger_time.hour,
            minute=trigger_time.minute,
            second=0,
            microsecond=0
        )
        
        time_diff = abs((now - trigger_dt).total_seconds())
        
        if time_diff <= 120:  # Within 2 minute window
            return f"Market open summary ({self.minutes_after_open} min after open)"
        
        return None
    
    def _check_market_close_trigger(self, now: datetime) -> Optional[str]:
        """Check if market close trigger should fire.
        
        Args:
            now: Current UTC time
            
        Returns:
            Trigger reason or None
        """
        if self._market_close_triggered_today:
            return None
        
        _, close_time = self._get_market_times()
        
        # Calculate trigger window (minutes_before_close before market close)
        trigger_hour = close_time.hour
        trigger_minute = close_time.minute - self.minutes_before_close
        
        if trigger_minute < 0:
            trigger_hour -= 1
            trigger_minute += 60
        
        trigger_dt = now.replace(
            hour=trigger_hour,
            minute=trigger_minute,
            second=0,
            microsecond=0
        )
        
        time_diff = abs((now - trigger_dt).total_seconds())
        
        if time_diff <= 120:  # Within 2 minute window
            return f"Market close summary ({self.minutes_before_close} min before close)"
        
        return None
    
    def _check_volatility_spike(self, snapshot: Dict) -> Optional[str]:
        """Check if volatility spike occurred.
        
        Args:
            snapshot: Market snapshot with indicators
            
        Returns:
            Trigger reason or None
        """
        atr = snapshot.get("atr", 0.0)
        
        if atr <= 0:
            return None
        
        # Update ATR history for baseline calculation
        self._atr_history.append(atr)
        if len(self._atr_history) > 100:
            self._atr_history = self._atr_history[-100:]
        
        # Calculate baseline (average of last 50 ATR values)
        if len(self._atr_history) >= 20:
            self._baseline_atr = sum(self._atr_history[-50:]) / min(50, len(self._atr_history))
        
        if self._baseline_atr and self._baseline_atr > 0:
            spike_ratio = atr / self._baseline_atr
            
            if spike_ratio >= self.volatility_spike_threshold:
                return f"Volatility spike detected: {spike_ratio:.1f}x baseline ATR"
        
        return None
    
    def _check_news_trigger(self, snapshot: Dict) -> Optional[str]:
        """Check if news keywords detected.
        
        Args:
            snapshot: Market snapshot with news headlines
            
        Returns:
            Trigger reason or None
        """
        headlines = snapshot.get("news_headlines", [])
        
        if not headlines:
            return None
        
        # Combine headlines and search for keywords
        text = " ".join(headlines).lower()
        
        detected_keywords = []
        for keyword in self.NEWS_KEYWORDS:
            if re.search(r'\b' + keyword + r'\b', text):
                detected_keywords.append(keyword)
        
        if detected_keywords:
            return f"News detected: {', '.join(detected_keywords[:3])}"
        
        return None
    
    def set_manual_trigger(self, notes: Optional[str] = None) -> None:
        """Set manual trigger (from Flask endpoint).
        
        Args:
            notes: Optional notes for the trigger
        """
        self._manual_trigger_pending = True
        self._manual_trigger_notes = notes
        logger.info(f"Manual Bedrock trigger set: {notes or 'No notes'}")
    
    def should_call_bedrock(
        self,
        snapshot: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[EventPayload]]:
        """Check if Bedrock should be called.
        
        This is the main entry point called from the trading loop.
        
        Args:
            snapshot: Market snapshot with:
                - current_price: float
                - momentum: float
                - atr: float
                - volatility: float
                - rsi: float
                - vix: Optional[float]
                - position: int
                - unrealized_pnl: float
                - news_headlines: List[str]
                - recent_prices: List[float]
                
        Returns:
            Tuple of (should_trigger, reason, payload)
        """
        now = datetime.now(timezone.utc)
        
        # Reset daily state if new day
        self._reset_daily_state()
        
        # Check minimum interval
        if self._is_within_min_interval():
            return False, None, None
        
        trigger_type: Optional[str] = None
        reason: Optional[str] = None
        
        # Priority 1: Manual trigger (always takes precedence)
        if self._manual_trigger_pending:
            trigger_type = "manual"
            reason = f"Manual trigger: {self._manual_trigger_notes or 'User requested'}"
            self._manual_trigger_pending = False
            self._manual_trigger_notes = None
        
        # Priority 2: Market open summary
        elif not self._is_within_cooldown("market_open"):
            market_open_reason = self._check_market_open_trigger(now)
            if market_open_reason:
                trigger_type = "market_open"
                reason = market_open_reason
                self._market_open_triggered_today = True
        
        # Priority 3: Market close summary
        elif not self._is_within_cooldown("market_close"):
            market_close_reason = self._check_market_close_trigger(now)
            if market_close_reason:
                trigger_type = "market_close"
                reason = market_close_reason
                self._market_close_triggered_today = True
        
        # Priority 4: Volatility spike
        elif not self._is_within_cooldown("volatility_spike"):
            vol_reason = self._check_volatility_spike(snapshot)
            if vol_reason:
                trigger_type = "volatility_spike"
                reason = vol_reason
        
        # Priority 5: News detection
        elif not self._is_within_cooldown("news"):
            news_reason = self._check_news_trigger(snapshot)
            if news_reason:
                trigger_type = "news"
                reason = news_reason
        
        # No trigger
        if not trigger_type:
            return False, None, None
        
        # Record trigger
        self._record_trigger(trigger_type)
        
        # Build payload
        payload = EventPayload(
            trigger_type=trigger_type,
            reason=reason,
            timestamp=now,
            symbol=self.symbol,
            current_price=snapshot.get("current_price", 0.0),
            price_change_pct=snapshot.get("price_change_pct", 0.0),
            momentum=snapshot.get("momentum", 0.0),
            atr=snapshot.get("atr", 0.0),
            volatility=snapshot.get("volatility", 0.0),
            rsi=snapshot.get("rsi", 50.0),
            vix=snapshot.get("vix"),
            position=snapshot.get("position", 0),
            unrealized_pnl=snapshot.get("unrealized_pnl", 0.0),
            news_headlines=snapshot.get("news_headlines", []),
            recent_prices=snapshot.get("recent_prices", []),
        )
        
        logger.info(f"Bedrock trigger: {trigger_type} - {reason}")
        
        return True, reason, payload
    
    def get_status(self) -> Dict[str, Any]:
        """Get detector status.
        
        Returns:
            Status dictionary
        """
        return {
            "symbol": self.symbol,
            "volatility_threshold": self.volatility_spike_threshold,
            "baseline_atr": self._baseline_atr,
            "market_open_triggered": self._market_open_triggered_today,
            "market_close_triggered": self._market_close_triggered_today,
            "last_trigger_time": self._last_trigger_time.isoformat() if self._last_trigger_time else None,
            "cooldown_seconds": self.cooldown_seconds,
            "min_interval_seconds": self.min_interval_seconds,
            "manual_trigger_pending": self._manual_trigger_pending,
        }


# Factory function for creating detector with config
def create_event_detector(
    symbol: str = "MES",
    config: Optional[Dict] = None
) -> EventDetector:
    """Create event detector from config.
    
    Args:
        symbol: Trading symbol
        config: Optional config dict with:
            - volatility_spike_threshold
            - minutes_after_open
            - minutes_before_close
            - min_interval_seconds
            - cooldown_seconds
            
    Returns:
        Configured EventDetector
    """
    if config is None:
        config = {}
    
    return EventDetector(
        symbol=symbol,
        volatility_spike_threshold=config.get("volatility_spike_threshold", 2.0),
        minutes_after_open=config.get("minutes_after_open", 5),
        minutes_before_close=config.get("minutes_before_close", 5),
        min_interval_seconds=config.get("min_interval_seconds", 60),
        cooldown_seconds=config.get("cooldown_seconds", 300),
    )
