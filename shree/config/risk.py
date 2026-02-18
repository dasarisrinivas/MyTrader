"""Risk and trading execution configuration — RiskGateConfig, TradingConfig."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os
from datetime import time

from .strategy import EntryFilterConfig

@dataclass
class RiskGateConfig:
    """Hard risk/margin gate for MES."""

    # JAN 18 2026: Updated RiskGate limits to support RTH Volatility Strategy
    # Increased caps to allow wider stops (up to 50 pts) for volatile sessions
    risk_contracts_env: str = os.environ.get("MAX_MES_CONTRACTS", "1")
    max_contracts: int = field(default_factory=lambda: int(os.environ.get("MAX_MES_CONTRACTS", "1")))
    
    # Base risk per trade raised to $250 to accommodate 50pt stops (1 contract * $5 * 50pts)
    risk_per_trade_usd: float = field(default_factory=lambda: float(os.environ.get("RISK_PER_TRADE_USD", "250")))
    risk_per_trade_min: float = 25.0
    risk_per_trade_max: float = 250.0 # Increased from 75 to 250
    min_stop_points: float = 3.0      # Slightly increased floor
    
    # Daily loss raised to $750 (3 max loss trades) to prevent instant lockout
    daily_max_loss_usd: float = field(default_factory=lambda: float(os.environ.get("DAILY_MAX_LOSS_USD", "750")))
    margin_buffer_usd: float = field(default_factory=lambda: float(os.environ.get("MARGIN_BUFFER_USD", "1000")))
    initial_margin_long: float = field(default_factory=lambda: float(os.environ.get("MES_INITIAL_MARGIN_LONG", "2464")))
    initial_margin_short: float = field(default_factory=lambda: float(os.environ.get("MES_INITIAL_MARGIN_SHORT", "2305.6")))
    avoid_close_window_minutes: int = field(default_factory=lambda: int(os.environ.get("AVOID_CLOSE_WINDOW_MINUTES", "20")))
    avoid_close_enabled: bool = field(
        default_factory=lambda: os.environ.get("AVOID_CLOSE_WINDOW_ENABLED", "true").lower() not in {"0", "false", "no"}
    )
    intraday_close_time: time = time(16, 0)  # CME close (4 PM CT)
    maintenance_start: time = time(16, 0)  # CME maintenance start CT
    maintenance_end: time = time(17, 0)    # CME maintenance end CT
    tick_size: float = 0.25
    max_stop_points: float = 50.0  # Increased from 12.0 to 50.0 for RTH volatility
    max_consecutive_losses: int = 3
    peak_drawdown_enabled: bool = False
    peak_drawdown_pct: float = 4.0
    peak_drawdown_action: str = "halt"  # "halt" or "tighten"
    peak_drawdown_tighten_multiplier: float = 0.5
    peak_drawdown_stop_buffer_points: float = 0.5
    peak_drawdown_flatten_on_trigger: bool = True
    peak_drawdown_reset_on_new_day: bool = False

    def bounded_risk_usd(self) -> float:
        """Clamp risk-per-trade to a safe range."""
        raw = self.risk_per_trade_usd
        return min(self.risk_per_trade_max, max(self.risk_per_trade_min, raw))


@dataclass
class TradingConfig:
    max_position_size: int = field(default_factory=lambda: int(os.environ.get("MAX_POSITION_SIZE", "5")))
    contracts_per_order: int = 1
    max_daily_loss: float = field(default_factory=lambda: float(os.environ.get("MAX_DAILY_LOSS", "2000.0")))
    max_loss_per_trade: float = field(default_factory=lambda: float(os.environ.get("MAX_LOSS_PER_TRADE", "1250.0")))
    max_daily_trades: int = 20
    initial_capital: float = field(default_factory=lambda: float(os.environ.get("INITIAL_CAPITAL", "100000.0")))
    contract_multiplier: float = 50.0
    stop_loss_ticks: float = 10.0
    take_profit_ticks: float = 20.0
    tick_size: float = 0.25
    tick_value: float = 12.5
    commission_per_contract: float = 2.4
    
    # Position sizing method: "fixed_fraction" or "kelly"
    position_sizing_method: str = "fixed_fraction"
    # Risk percentage per trade for fixed fractional sizing (0.005 = 0.5%, 0.01 = 1%)
    risk_per_trade_pct: float = 0.005
    
    # Risk/reward guardrail
    min_risk_reward_ratio: float = 1.0  # Minimum R:R to allow entry (1.0 = 1:1)

    # Safety parameters
    disaster_stop_pct: float = 0.007
    max_trade_duration_minutes: int = 60
    trade_cooldown_minutes: int = 5
    order_lock_timeout_seconds: int = 300
    pending_order_timeout_seconds: int = 180
    enable_ib_api_debug_log: bool = field(
        default_factory=lambda: os.environ.get("IB_API_DEBUG_LOG", "False").lower() in {"1", "true", "yes"}
    )
    ib_api_log_path: str = field(
        default_factory=lambda: os.environ.get("IB_API_LOG_PATH", "logs/ib_api_wire.log")
    )
    reset_state_on_start: bool = False
    
    # Margin safety (used by PositionManager)
    initial_margin_long: float = field(default_factory=lambda: float(os.environ.get("MES_INITIAL_MARGIN_LONG", "2464")))
    initial_margin_short: float = field(default_factory=lambda: float(os.environ.get("MES_INITIAL_MARGIN_SHORT", "2305.6")))

    def get_validated_cooldown(self, min_val: int = 1, max_val: int = 60) -> int:
        """Return cooldown bounded to [min_val, max_val], log warning if out of bounds."""
        val = getattr(self, 'trade_cooldown_minutes', 5)
        if val < min_val:
            import logging
            logging.warning(f"trade_cooldown_minutes {val} too low; using {min_val}")
            return min_val
        if val > max_val:
            import logging
            logging.warning(f"trade_cooldown_minutes {val} too high; using {max_val}")
            return max_val
        if val > 30:
            import logging
            logging.warning(f"trade_cooldown_minutes unusually high: {val}")
        return val
    
    # Indicator warm-up
    min_bars_for_signals: int = 200
    
    # Market regime filter thresholds
    min_atr_threshold: float = 0.5
    max_spread_ticks: int = 1
    max_loop_latency_seconds: float = 3.0
    
    # Weighted voting thresholds
    min_weighted_confidence: float = 0.70
    confidence_threshold: float = field(default_factory=lambda: float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7")))
    min_confidence_for_trade: float = 0.60
    min_stop_distance_ticks: int = 4

    # Startup entry gating (prevents immediate post-restart entries)
    # 0 disables. These gates should only affect NEW entries; exits remain allowed.
    startup_grace_period_seconds: int = 0
    startup_min_completed_bars: int = 0

    # Hard Safety Constraints
    max_contracts_limit: int = field(default_factory=lambda: int(os.environ.get("MAX_CONTRACTS", "5")))
    margin_limit_pct: float = field(default_factory=lambda: float(os.environ.get("MARGIN_LIMIT_PCT", "0.80")))
    decision_min_interval_seconds: int = field(default_factory=lambda: int(os.environ.get("DECISION_MIN_INTERVAL_SECONDS", "30")))
    order_retry_limit: int = field(default_factory=lambda: int(os.environ.get("ORDER_RETRY_LIMIT", "3")))
    contract_month_offset: int = field(default_factory=lambda: int(os.environ.get("CONTRACT_MONTH_OFFSET", "0")))
    allow_naked_orders: bool = field(
        default_factory=lambda: os.environ.get("ALLOW_NAKED_ORDERS", "False").lower()
        in {"1", "true", "yes"}
    )
    enforce_market_hours: bool = field(
        default_factory=lambda: os.environ.get("ENFORCE_MARKET_HOURS", "True").lower()
        in {"1", "true", "yes"}
    )
    cancel_orders_on_startup: bool = field(
        default_factory=lambda: os.environ.get("CANCEL_ORDERS_ON_STARTUP", "False").lower()
        in {"1", "true", "yes"}
    )
    
    # Optional entry filter tuning
    entry_filters: EntryFilterConfig = field(default_factory=EntryFilterConfig)



