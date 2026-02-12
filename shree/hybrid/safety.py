"""Safety Manager - Throttles, cooldowns, and emergency stops.

Implements production safety guards for the trading bot.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import logger


@dataclass
class SafetyCheck:
    """Result of a safety check."""
    
    is_safe: bool
    reason: str
    check_type: str  # "cooldown", "order_limit", "pnl_limit", "emergency_stop"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "reason": self.reason,
            "check_type": self.check_type,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OrderRecord:
    """Record of a placed order for tracking."""
    
    timestamp: datetime
    action: str
    quantity: int
    price: float
    order_id: Optional[str] = None


class SafetyManager:
    """Production safety manager with throttles, limits, and emergency stops.
    
    Safety features:
    1. Trade cooldown: Minimum time between trades
    2. Order limit: Maximum orders per rolling window
    3. P&L limit: Maximum loss before emergency stop
    4. Emergency stop: Manual or automatic halt
    """
    
    def __init__(
        self,
        cooldown_minutes: int = 5,
        max_orders_per_window: int = 3,
        window_minutes: int = 15,
        max_pnl_drop_pct: float = 2.5,
        initial_capital: float = 100000.0,
        dry_run: bool = False,
    ):
        """Initialize safety manager.
        
        Args:
            cooldown_minutes: Minimum minutes between trades
            max_orders_per_window: Maximum orders allowed in rolling window
            window_minutes: Rolling window size in minutes
            max_pnl_drop_pct: Maximum P&L drop (percentage) before emergency stop
            initial_capital: Initial capital for P&L calculations
            dry_run: If True, operate in dry-run mode (no real orders)
        """
        self.cooldown_seconds = cooldown_minutes * 60
        self.max_orders_per_window = max_orders_per_window
        self.window_seconds = window_minutes * 60
        self.max_pnl_drop_pct = max_pnl_drop_pct
        self.initial_capital = initial_capital
        self.dry_run = dry_run
        
        # State tracking
        self._last_trade_time: Optional[datetime] = None
        self._order_history: List[OrderRecord] = []
        self._current_pnl: float = 0.0
        self._peak_pnl: float = 0.0
        self._emergency_stop: bool = False
        self._emergency_reason: str = ""
        
        # Statistics
        self._checks_performed = 0
        self._trades_blocked = 0
        self._trades_allowed = 0
        
        logger.info(
            f"SafetyManager initialized: cooldown={cooldown_minutes}min, "
            f"max_orders={max_orders_per_window}/{window_minutes}min, "
            f"max_drop={max_pnl_drop_pct}%, dry_run={dry_run}"
        )
    
    def check_all(self) -> SafetyCheck:
        """Perform all safety checks.
        
        Returns:
            SafetyCheck with overall result
        """
        self._checks_performed += 1
        
        # Check emergency stop first
        if self._emergency_stop:
            return SafetyCheck(
                is_safe=False,
                reason=f"Emergency stop active: {self._emergency_reason}",
                check_type="emergency_stop",
            )
        
        # Check cooldown
        cooldown_check = self.check_cooldown()
        if not cooldown_check.is_safe:
            self._trades_blocked += 1
            return cooldown_check
        
        # Check order limit
        order_check = self.check_order_limit()
        if not order_check.is_safe:
            self._trades_blocked += 1
            return order_check
        
        # Check P&L limit
        pnl_check = self.check_pnl_limit()
        if not pnl_check.is_safe:
            self._trades_blocked += 1
            return pnl_check
        
        self._trades_allowed += 1
        return SafetyCheck(
            is_safe=True,
            reason="All safety checks passed",
            check_type="all",
        )
    
    def check_cooldown(self) -> SafetyCheck:
        """Check if cooldown period has elapsed.
        
        Returns:
            SafetyCheck for cooldown
        """
        if self._last_trade_time is None:
            return SafetyCheck(
                is_safe=True,
                reason="No previous trade",
                check_type="cooldown",
            )
        
        elapsed = (datetime.now(timezone.utc) - self._last_trade_time).total_seconds()
        remaining = self.cooldown_seconds - elapsed
        
        if remaining > 0:
            return SafetyCheck(
                is_safe=False,
                reason=f"Cooldown: {remaining:.0f}s remaining",
                check_type="cooldown",
            )
        
        return SafetyCheck(
            is_safe=True,
            reason="Cooldown elapsed",
            check_type="cooldown",
        )
    
    def check_order_limit(self) -> SafetyCheck:
        """Check if order limit for rolling window is exceeded.
        
        Returns:
            SafetyCheck for order limit
        """
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self.window_seconds)
        
        # Count recent orders
        recent_orders = [
            o for o in self._order_history
            if o.timestamp > window_start
        ]
        
        if len(recent_orders) >= self.max_orders_per_window:
            return SafetyCheck(
                is_safe=False,
                reason=f"Order limit: {len(recent_orders)}/{self.max_orders_per_window} in window",
                check_type="order_limit",
            )
        
        return SafetyCheck(
            is_safe=True,
            reason=f"Orders in window: {len(recent_orders)}/{self.max_orders_per_window}",
            check_type="order_limit",
        )
    
    def check_pnl_limit(self) -> SafetyCheck:
        """Check if P&L drop exceeds limit.
        
        Returns:
            SafetyCheck for P&L limit
        """
        if self._peak_pnl <= 0:
            return SafetyCheck(
                is_safe=True,
                reason="No peak P&L recorded",
                check_type="pnl_limit",
            )
        
        # Calculate drawdown from peak
        drawdown = self._peak_pnl - self._current_pnl
        drawdown_pct = (drawdown / self.initial_capital) * 100
        
        if drawdown_pct >= self.max_pnl_drop_pct:
            # Trigger emergency stop
            self._emergency_stop = True
            self._emergency_reason = f"P&L drawdown {drawdown_pct:.2f}% >= {self.max_pnl_drop_pct}%"
            
            return SafetyCheck(
                is_safe=False,
                reason=self._emergency_reason,
                check_type="pnl_limit",
            )
        
        return SafetyCheck(
            is_safe=True,
            reason=f"Drawdown: {drawdown_pct:.2f}%",
            check_type="pnl_limit",
        )
    
    def record_trade(
        self,
        action: str,
        quantity: int,
        price: float,
        order_id: Optional[str] = None,
    ):
        """Record a trade for tracking.
        
        Args:
            action: "BUY" or "SELL"
            quantity: Number of contracts
            price: Fill price
            order_id: Optional order ID
        """
        now = datetime.now(timezone.utc)
        
        self._order_history.append(OrderRecord(
            timestamp=now,
            action=action,
            quantity=quantity,
            price=price,
            order_id=order_id,
        ))
        
        self._last_trade_time = now
        
        # Clean old history (keep last 100)
        if len(self._order_history) > 100:
            self._order_history = self._order_history[-100:]
        
        logger.info(f"Trade recorded: {action} {quantity} @ {price}")
    
    def update_pnl(self, realized_pnl: float, unrealized_pnl: float = 0.0):
        """Update current P&L.
        
        Args:
            realized_pnl: Realized P&L
            unrealized_pnl: Unrealized P&L
        """
        self._current_pnl = realized_pnl + unrealized_pnl
        
        # Update peak
        if self._current_pnl > self._peak_pnl:
            self._peak_pnl = self._current_pnl
    
    def trigger_emergency_stop(self, reason: str):
        """Manually trigger emergency stop.
        
        Args:
            reason: Reason for emergency stop
        """
        self._emergency_stop = True
        self._emergency_reason = reason
        logger.critical(f"EMERGENCY STOP triggered: {reason}")
    
    def reset_emergency_stop(self):
        """Reset emergency stop (requires manual intervention)."""
        self._emergency_stop = False
        self._emergency_reason = ""
        logger.warning("Emergency stop RESET - trading enabled")
    
    def get_cooldown_remaining(self) -> float:
        """Get remaining cooldown in seconds."""
        if self._last_trade_time is None:
            return 0.0
        
        elapsed = (datetime.now(timezone.utc) - self._last_trade_time).total_seconds()
        return max(0.0, self.cooldown_seconds - elapsed)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get safety statistics.
        
        Returns:
            Dictionary of safety stats
        """
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self.window_seconds)
        recent_orders = [o for o in self._order_history if o.timestamp > window_start]
        
        return {
            "dry_run": self.dry_run,
            "emergency_stop": self._emergency_stop,
            "emergency_reason": self._emergency_reason,
            "cooldown_remaining_s": self.get_cooldown_remaining(),
            "orders_in_window": len(recent_orders),
            "max_orders_per_window": self.max_orders_per_window,
            "current_pnl": self._current_pnl,
            "peak_pnl": self._peak_pnl,
            "drawdown_pct": ((self._peak_pnl - self._current_pnl) / self.initial_capital * 100) if self._peak_pnl > 0 else 0,
            "checks_performed": self._checks_performed,
            "trades_blocked": self._trades_blocked,
            "trades_allowed": self._trades_allowed,
        }
