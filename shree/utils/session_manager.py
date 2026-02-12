"""
Session Manager for ES/MES 24-Hour Trading
Provides session-aware configuration for different trading periods.

CME ES/MES Sessions (Central Time):
- RTH (Regular Trading Hours): 8:30 AM - 3:00 PM CT
- Evening Session: 5:00 PM - 11:00 PM CT  
- Overnight Session: 11:00 PM - 3:00 AM CT
- Pre-Market Session: 3:00 AM - 8:30 AM CT
- Daily Maintenance: 4:00 PM - 5:00 PM CT (NO TRADING)

Weekend Closure: Friday 4:00 PM CT - Sunday 5:00 PM CT
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Dict, Optional, Any
import os

from .timezone_utils import CST, now_cst
from .logger import logger


class TradingSession(Enum):
    """Trading session types for ES/MES."""
    RTH = "RTH"                    # Regular Trading Hours (most liquid)
    EVENING = "EVENING"            # Post-RTH evening session
    OVERNIGHT = "OVERNIGHT"        # Late night / early morning
    PRE_MARKET = "PRE_MARKET"      # Before RTH
    MAINTENANCE = "MAINTENANCE"    # Daily CME maintenance window
    WEEKEND = "WEEKEND"            # Market closed


@dataclass
class SessionConfig:
    """Session-specific trading parameters."""
    session: TradingSession
    
    # Position sizing
    max_contracts: int = 1
    position_size_factor: float = 1.0  # Multiplier for normal size
    
    # Confidence thresholds
    min_confidence: float = 0.55
    confidence_boost: float = 0.0  # Added to signals during favorable sessions
    
    # Risk management
    atr_multiplier_sl: float = 3.0
    atr_multiplier_tp: float = 6.0  # 2:1 R:R
    min_risk_reward: float = 1.5
    max_stop_points: float = 12.0
    
    # Trade frequency
    max_trades_per_hour: int = 2
    max_trades_per_session: int = 8
    cooldown_after_loss_minutes: int = 20
    cooldown_after_consecutive_losses_minutes: int = 45
    
    # Entry filters
    require_5m_confirmation: bool = True
    require_trend_alignment: bool = True
    spread_gate_ticks: float = 2.0  # Max spread to allow entry
    
    # Session-specific adjustments
    rsi_buffer: float = 0.0  # Added to RSI thresholds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "session": self.session.value,
            "max_contracts": self.max_contracts,
            "position_size_factor": self.position_size_factor,
            "min_confidence": self.min_confidence,
            "atr_multiplier_sl": self.atr_multiplier_sl,
            "min_risk_reward": self.min_risk_reward,
            "max_trades_per_hour": self.max_trades_per_hour,
            "cooldown_after_loss_minutes": self.cooldown_after_loss_minutes,
            "require_5m_confirmation": self.require_5m_confirmation,
            "spread_gate_ticks": self.spread_gate_ticks,
        }


# Default session configurations
DEFAULT_SESSION_CONFIGS: Dict[TradingSession, SessionConfig] = {
    TradingSession.RTH: SessionConfig(
        session=TradingSession.RTH,
        max_contracts=1,
        position_size_factor=1.0,
        min_confidence=0.55,
        atr_multiplier_sl=3.0,
        atr_multiplier_tp=6.0,
        min_risk_reward=1.5,
        max_stop_points=12.0,
        max_trades_per_hour=2,
        max_trades_per_session=8,
        cooldown_after_loss_minutes=20,
        cooldown_after_consecutive_losses_minutes=45,
        require_5m_confirmation=True,
        require_trend_alignment=True,
        spread_gate_ticks=2.0,
        rsi_buffer=0.0,
    ),
    TradingSession.EVENING: SessionConfig(
        session=TradingSession.EVENING,
        max_contracts=1,
        position_size_factor=1.0,  # Full size OK in evening
        min_confidence=0.60,  # Slightly higher threshold
        atr_multiplier_sl=3.5,  # Wider stops
        atr_multiplier_tp=7.0,  # Maintain 2:1 R:R
        min_risk_reward=2.0,  # Require better setups
        max_stop_points=12.0,
        max_trades_per_hour=1,  # Slower pace
        max_trades_per_session=4,
        cooldown_after_loss_minutes=30,
        cooldown_after_consecutive_losses_minutes=60,
        require_5m_confirmation=True,
        require_trend_alignment=True,
        spread_gate_ticks=2.0,
        rsi_buffer=3.0,  # Relaxed RSI thresholds
    ),
    TradingSession.OVERNIGHT: SessionConfig(
        session=TradingSession.OVERNIGHT,
        max_contracts=1,
        position_size_factor=1.0,  # Conservative
        min_confidence=0.70,  # Highest threshold
        atr_multiplier_sl=4.0,  # Widest stops
        atr_multiplier_tp=8.0,  # Maintain 2:1 R:R
        min_risk_reward=2.0,
        max_stop_points=12.0,
        max_trades_per_hour=1,
        max_trades_per_session=2,  # Very limited
        cooldown_after_loss_minutes=45,
        cooldown_after_consecutive_losses_minutes=90,
        require_5m_confirmation=True,
        require_trend_alignment=True,
        spread_gate_ticks=3.0,  # Allow slightly wider spreads
        rsi_buffer=5.0,  # Most relaxed RSI
    ),
    TradingSession.PRE_MARKET: SessionConfig(
        session=TradingSession.PRE_MARKET,
        max_contracts=1,
        position_size_factor=1.0,
        min_confidence=0.65,
        atr_multiplier_sl=3.5,
        atr_multiplier_tp=7.0,
        min_risk_reward=2.0,
        max_stop_points=12.0,
        max_trades_per_hour=1,
        max_trades_per_session=2,
        cooldown_after_loss_minutes=30,
        cooldown_after_consecutive_losses_minutes=60,
        require_5m_confirmation=True,
        require_trend_alignment=True,
        spread_gate_ticks=2.0,
        rsi_buffer=3.0,
    ),
    TradingSession.MAINTENANCE: SessionConfig(
        session=TradingSession.MAINTENANCE,
        max_contracts=0,  # NO TRADING
        position_size_factor=0.0,
        min_confidence=1.0,  # Impossible threshold
        atr_multiplier_sl=0.0,
        atr_multiplier_tp=0.0,
        min_risk_reward=99.0,
        max_stop_points=0.0,
        max_trades_per_hour=0,
        max_trades_per_session=0,
        cooldown_after_loss_minutes=999,
        cooldown_after_consecutive_losses_minutes=999,
        require_5m_confirmation=True,
        require_trend_alignment=True,
        spread_gate_ticks=0.0,
        rsi_buffer=0.0,
    ),
    TradingSession.WEEKEND: SessionConfig(
        session=TradingSession.WEEKEND,
        max_contracts=0,  # NO TRADING
        position_size_factor=0.0,
        min_confidence=1.0,
        atr_multiplier_sl=0.0,
        atr_multiplier_tp=0.0,
        min_risk_reward=99.0,
        max_stop_points=0.0,
        max_trades_per_hour=0,
        max_trades_per_session=0,
        cooldown_after_loss_minutes=999,
        cooldown_after_consecutive_losses_minutes=999,
        require_5m_confirmation=True,
        require_trend_alignment=True,
        spread_gate_ticks=0.0,
        rsi_buffer=0.0,
    ),
}


class SessionManager:
    """
    Manages trading sessions and provides session-aware configurations.
    
    Key responsibilities:
    - Determine current trading session
    - Provide session-specific risk parameters
    - Track session transitions and reset counters
    - Block trading during maintenance/weekends
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize session manager.
        
        Args:
            config: Optional config overrides for session parameters
        """
        self.config = config or {}
        self._session_configs = dict(DEFAULT_SESSION_CONFIGS)
        
        # Apply any config overrides
        self._apply_config_overrides()
        
        # Session tracking
        self._current_session: Optional[TradingSession] = None
        self._session_start_time: Optional[datetime] = None
        self._session_trade_count: int = 0
        self._session_loss_count: int = 0
        self._daily_trade_count: int = 0
        self._daily_loss_count: int = 0
        self._consecutive_losses: int = 0
        self._last_daily_reset: Optional[datetime] = None
        
        # Time boundaries (in CST)
        self._rth_start = time(8, 30)
        self._rth_end = time(15, 0)
        self._maintenance_start = time(16, 0)
        self._maintenance_end = time(17, 0)
        self._evening_start = time(17, 0)
        self._evening_end = time(23, 0)
        self._overnight_end = time(3, 0)
        self._premarket_end = time(8, 30)
    
    def _apply_config_overrides(self) -> None:
        """Apply configuration overrides to session configs from YAML.
        
        Reads from futures_trading.active_sessions in config.yaml.
        Maps lowercase YAML keys (rth, evening, etc.) to TradingSession enums.
        """
        # Map YAML session names (lowercase) to TradingSession enum values
        yaml_to_enum = {
            "rth": TradingSession.RTH,
            "evening": TradingSession.EVENING,
            "overnight": TradingSession.OVERNIGHT,
            "premarket": TradingSession.PRE_MARKET,
            "pre_market": TradingSession.PRE_MARKET,
            "maintenance": TradingSession.MAINTENANCE,
            "weekend": TradingSession.WEEKEND,
        }
        
        # Read from futures_trading.active_sessions (the actual YAML structure)
        futures_config = self.config.get("futures_trading", {})
        active_sessions = futures_config.get("active_sessions", {})
        
        if not active_sessions:
            logger.debug("No active_sessions config found in futures_trading")
            return
        
        for session_name, overrides in active_sessions.items():
            if not isinstance(overrides, dict):
                continue
                
            # Map YAML key to enum
            session = yaml_to_enum.get(session_name.lower())
            if session is None:
                logger.warning(f"Unknown session in config: {session_name}")
                continue
                
            if session not in self._session_configs:
                logger.warning(f"No default config for session: {session.value}")
                continue
            
            # Apply overrides to session config
            session_cfg = self._session_configs[session]
            for key, value in overrides.items():
                # Skip non-config keys like start_time, end_time, timezone
                if key in ("start_time", "end_time", "timezone", "trading_blocked", "reason"):
                    continue
                    
                if hasattr(session_cfg, key):
                    # Validate confidence is 0-1 scale
                    if key == "min_confidence" and value > 1.0:
                        logger.warning(f"min_confidence {value} > 1.0, assuming 0-100 scale, converting to {value/100}")
                        value = value / 100.0
                    setattr(session_cfg, key, value)
                    logger.debug(f"Session {session.value}: set {key}={value}")
                else:
                    logger.debug(f"Session {session.value}: unknown config key '{key}'")
    
    def get_current_session(self, now: Optional[datetime] = None) -> TradingSession:
        """
        Determine the current trading session.
        
        Args:
            now: Current time (defaults to now_cst())
            
        Returns:
            Current TradingSession enum value
        """
        if now is None:
            now = now_cst()
        
        # Ensure we're in CST
        if now.tzinfo is None:
            now = now.replace(tzinfo=CST)
        else:
            now = now.astimezone(CST)
        
        current_time = now.time()
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        
        # Weekend check - CME closed Friday 4 PM CT to Sunday 5 PM CT
        # Friday after 4 PM CT = weekend (market closed after maintenance)
        if weekday == 4 and current_time >= self._maintenance_start:  # Friday after 4 PM
            return TradingSession.WEEKEND
        if weekday == 5:  # Saturday
            return TradingSession.WEEKEND
        if weekday == 6 and current_time < self._evening_start:  # Sunday before 5 PM
            return TradingSession.WEEKEND
        
        # Daily maintenance (4 PM - 5 PM CT)
        if self._maintenance_start <= current_time < self._maintenance_end:
            return TradingSession.MAINTENANCE
        
        # RTH (8:30 AM - 3:00 PM CT)
        if self._rth_start <= current_time < self._rth_end:
            return TradingSession.RTH
        
        # Evening (5:00 PM - 11:00 PM CT)
        if self._evening_start <= current_time < self._evening_end:
            return TradingSession.EVENING
        
        # Overnight (11:00 PM - 3:00 AM CT)
        if current_time >= self._evening_end or current_time < self._overnight_end:
            return TradingSession.OVERNIGHT
        
        # Pre-market (3:00 AM - 8:30 AM CT)
        if self._overnight_end <= current_time < self._premarket_end:
            return TradingSession.PRE_MARKET
        
        # Close window (3:00 PM - 4:00 PM CT) - treat as RTH with caution
        if self._rth_end <= current_time < self._maintenance_start:
            return TradingSession.RTH  # Still RTH, but avoid_close_window should block
        
        return TradingSession.RTH  # Fallback
    
    def get_session_config(self, session: Optional[TradingSession] = None) -> SessionConfig:
        """
        Get configuration for a specific session.
        
        Args:
            session: Session to get config for (defaults to current session)
            
        Returns:
            SessionConfig for the requested session
        """
        if session is None:
            session = self.get_current_session()
        
        return self._session_configs.get(session, self._session_configs[TradingSession.RTH])
    
    def is_trading_allowed(self, now: Optional[datetime] = None) -> tuple[bool, str]:
        """
        Check if trading is allowed at current time.
        
        Args:
            now: Current time (defaults to now_cst())
            
        Returns:
            Tuple of (allowed, reason)
        """
        session = self.get_current_session(now)
        config = self.get_session_config(session)
        
        if session == TradingSession.MAINTENANCE:
            return False, "CME daily maintenance window (4-5 PM CT)"
        
        if session == TradingSession.WEEKEND:
            return False, "Market closed for weekend"
        
        if config.max_contracts == 0:
            return False, f"Trading disabled for {session.value} session"
        
        # Check session trade limit
        if self._session_trade_count >= config.max_trades_per_session:
            return False, f"Session trade limit reached ({self._session_trade_count}/{config.max_trades_per_session})"
        
        # Check consecutive losses lockout
        max_consecutive = self.config.get("max_consecutive_losses", 3)
        if self._consecutive_losses >= max_consecutive:
            return False, f"Consecutive loss lockout ({self._consecutive_losses} losses)"
        
        return True, "OK"
    
    def check_daily_reset(self, now: Optional[datetime] = None) -> bool:
        """
        Check if daily counters should be reset (at 5 PM CT - CME settlement).
        
        Args:
            now: Current time
            
        Returns:
            True if reset was performed
        """
        if now is None:
            now = now_cst()
        
        reset_hour = 17  # 5 PM CT
        
        # Check if we've crossed the reset time since last reset
        if self._last_daily_reset is None:
            self._last_daily_reset = now
            return False
        
        # Different day or crossed 5 PM
        if now.date() > self._last_daily_reset.date():
            if now.hour >= reset_hour or self._last_daily_reset.hour < reset_hour:
                self._perform_daily_reset(now)
                return True
        elif now.hour >= reset_hour and self._last_daily_reset.hour < reset_hour:
            self._perform_daily_reset(now)
            return True
        
        return False
    
    def _perform_daily_reset(self, now: datetime) -> None:
        """Perform daily counter reset."""
        logger.info(
            "📅 Daily reset at {} CT | Trades: {} | Losses: {} | Consecutive: {}",
            now.strftime("%Y-%m-%d %H:%M"),
            self._daily_trade_count,
            self._daily_loss_count,
            self._consecutive_losses,
        )
        
        self._daily_trade_count = 0
        self._daily_loss_count = 0
        self._consecutive_losses = 0
        self._session_trade_count = 0
        self._session_loss_count = 0
        self._last_daily_reset = now
    
    def on_session_change(self, new_session: TradingSession) -> None:
        """
        Handle session transition.
        
        Args:
            new_session: The new session being entered
        """
        if self._current_session != new_session:
            old_session = self._current_session
            self._current_session = new_session
            self._session_start_time = now_cst()
            self._session_trade_count = 0
            self._session_loss_count = 0
            
            logger.info(
                "🔄 Session change: {} → {} at {}",
                old_session.value if old_session else "INIT",
                new_session.value,
                self._session_start_time.strftime("%H:%M CT"),
            )
    
    def record_trade(self, is_win: bool) -> None:
        """
        Record a completed trade.
        
        Args:
            is_win: Whether the trade was profitable
        """
        self._session_trade_count += 1
        self._daily_trade_count += 1
        
        if not is_win:
            self._session_loss_count += 1
            self._daily_loss_count += 1
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0  # Reset on win
    
    def get_status(self) -> Dict[str, Any]:
        """Get current session manager status for logging."""
        session = self.get_current_session()
        config = self.get_session_config(session)
        allowed, reason = self.is_trading_allowed()
        
        return {
            "session": session.value,
            "trading_allowed": allowed,
            "reason": reason,
            "session_trades": self._session_trade_count,
            "session_trade_limit": config.max_trades_per_session,
            "daily_trades": self._daily_trade_count,
            "consecutive_losses": self._consecutive_losses,
            "config": config.to_dict(),
        }


# Singleton instance
_session_manager: Optional[SessionManager] = None


def get_session_manager(config: Optional[Dict[str, Any]] = None) -> SessionManager:
    """Get or create the singleton session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(config)
    return _session_manager


def reset_session_manager() -> None:
    """Reset the singleton session manager (for testing)."""
    global _session_manager
    _session_manager = None
