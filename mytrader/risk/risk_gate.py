"""Hard risk and margin gate for MES trading."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Dict, Optional, Tuple
import math
import os

from ..utils.logger import logger
from ..utils.timezone_utils import CST, now_cst


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).lower() in {"1", "true", "yes", "on"}


@dataclass
class RiskGateConfig:
    """Configuration for MES hard risk gate."""

    max_contracts: int = field(default_factory=lambda: int(os.environ.get("MAX_MES_CONTRACTS", "1")))
    risk_per_trade_usd: float = field(default_factory=lambda: float(os.environ.get("RISK_PER_TRADE_USD", "60")))
    risk_per_trade_min: float = 25.0
    risk_per_trade_max: float = 75.0  # $75 max = 15 point stop
    # JAN 8 2026 FIX: Raised min_stop_points from 4.0 to 6.0
    # Analysis showed 83% stop-loss hit rate with 4-point stops due to normal
    # market noise of 3-5 points. 6 points gives breathing room.
    min_stop_points: float = 6.0
    max_stop_points: float = 12.0  # 12 points max = $60 risk cap
    margin_buffer_usd: float = field(default_factory=lambda: float(os.environ.get("MARGIN_BUFFER_USD", "1000")))
    initial_margin_long: float = field(default_factory=lambda: float(os.environ.get("MES_INITIAL_MARGIN_LONG", "2464")))
    initial_margin_short: float = field(default_factory=lambda: float(os.environ.get("MES_INITIAL_MARGIN_SHORT", "2305.6")))
    daily_max_loss_usd: float = field(default_factory=lambda: float(os.environ.get("DAILY_MAX_LOSS_USD", "150")))
    max_consecutive_losses: int = field(default_factory=lambda: int(os.environ.get("MAX_CONSECUTIVE_LOSSES", "3")))
    avoid_close_window_minutes: int = field(default_factory=lambda: int(os.environ.get("AVOID_CLOSE_WINDOW_MINUTES", "60")))
    avoid_close_enabled: bool = field(default_factory=lambda: _env_bool("AVOID_CLOSE_WINDOW_ENABLED", True))
    intraday_close_time: time = time(16, 0)  # End of close window CT (maintenance start)
    maintenance_start: time = time(16, 0)  # CME maintenance start CT
    maintenance_end: time = time(17, 0)    # CME maintenance end CT
    tick_size: float = 0.25
    # Peak-to-trough drawdown guard (high-water mark trailing stop for equity)
    peak_drawdown_enabled: bool = field(default_factory=lambda: _env_bool("PEAK_DRAWDOWN_ENABLED", False))
    peak_drawdown_pct: float = field(default_factory=lambda: float(os.environ.get("PEAK_DRAWDOWN_PCT", "4.0")))
    peak_drawdown_action: str = field(default_factory=lambda: os.environ.get("PEAK_DRAWDOWN_ACTION", "halt"))
    peak_drawdown_tighten_multiplier: float = field(
        default_factory=lambda: float(os.environ.get("PEAK_DRAWDOWN_TIGHTEN_MULT", "0.5"))
    )
    peak_drawdown_stop_buffer_points: float = field(
        default_factory=lambda: float(os.environ.get("PEAK_DRAWDOWN_STOP_BUFFER_POINTS", "0.5"))
    )
    peak_drawdown_flatten_on_trigger: bool = field(
        default_factory=lambda: _env_bool("PEAK_DRAWDOWN_FLATTEN", True)
    )
    peak_drawdown_reset_on_new_day: bool = field(
        default_factory=lambda: _env_bool("PEAK_DRAWDOWN_RESET_ON_NEW_DAY", False)
    )

    def bounded_risk_usd(self) -> float:
        raw = self.risk_per_trade_usd
        return min(self.risk_per_trade_max, max(self.risk_per_trade_min, raw))


@dataclass
class RiskGateResult:
    allowed: bool
    reason: str
    levels: Dict[str, float]


class RiskGate:
    """Evaluates hard risk/margin guardrails before entry."""

    def __init__(self, config: RiskGateConfig):
        self.config = config
        self._consecutive_losses: int = 0
        self._equity_high_water: Optional[float] = None
        self._equity_base: Optional[float] = None
        self._drawdown_active: bool = False
        self._drawdown_trigger_ts: Optional[datetime] = None
        self._last_equity: Optional[float] = None

    @staticmethod
    def _is_valid_number(val: Optional[float]) -> bool:
        return val is not None and isinstance(val, (int, float)) and math.isfinite(val)

    def _round_to_tick(self, value: float) -> float:
        tick = max(self.config.tick_size, 1e-6)
        return round(value / tick) * tick

    def _extract_equity(self, account_state: Dict[str, float]) -> Optional[float]:
        for key in (
            "account_equity",
            "net_liquidation",
            "equity",
            "available_funds",
            "excess_liquidity",
        ):
            value = account_state.get(key)
            if self._is_valid_number(value):
                return float(value)
        return None

    def update_equity(self, equity: Optional[float], timestamp: Optional[datetime] = None) -> None:
        if not self.config.peak_drawdown_enabled:
            return
        if equity is None or not self._is_valid_number(equity):
            return

        equity = float(equity)
        self._last_equity = equity
        if self._equity_high_water is None:
            self._equity_high_water = equity
            self._equity_base = equity
            self._drawdown_active = False
            self._drawdown_trigger_ts = None
            return

        if equity >= self._equity_high_water:
            self._equity_high_water = equity
            self._equity_base = equity
            self._drawdown_active = False
            self._drawdown_trigger_ts = None
            return

        threshold = -abs(self.config.peak_drawdown_pct) / 100.0
        drawdown_pct = (equity - self._equity_high_water) / self._equity_high_water

        if drawdown_pct <= threshold and not self._drawdown_active:
            self._drawdown_active = True
            self._drawdown_trigger_ts = timestamp or now_cst()
            logger.warning(
                "🚫 Peak drawdown triggered: equity {:.2f} vs high water {:.2f} ({:.2f}%)",
                equity,
                self._equity_high_water,
                drawdown_pct * 100.0,
            )

    @property
    def drawdown_active(self) -> bool:
        return self._drawdown_active

    @property
    def high_water_equity(self) -> Optional[float]:
        return self._equity_high_water

    @property
    def base_equity(self) -> Optional[float]:
        return self._equity_base

    def drawdown_status(self) -> Dict[str, Optional[float]]:
        if self._equity_high_water is None:
            return {"high_water": None, "base": None, "drawdown_pct": None}
        drawdown_pct = None
        if self._equity_high_water and self._last_equity is not None:
            drawdown_pct = (self._last_equity - self._equity_high_water) / self._equity_high_water
        return {
            "high_water": self._equity_high_water,
            "base": self._equity_base,
            "drawdown_pct": drawdown_pct,
        }

    def reset_drawdown(self) -> None:
        self._equity_high_water = None
        self._equity_base = None
        self._drawdown_active = False
        self._drawdown_trigger_ts = None
        self._last_equity = None

    def _check_close_window(self, now: datetime) -> bool:
        if not self.config.avoid_close_enabled:
            return False
        try:
            cst_now = now.astimezone(CST) if now.tzinfo else now_cst()
        except Exception:
            cst_now = now_cst()
        cutoff = datetime.combine(cst_now.date(), self.config.intraday_close_time, tzinfo=CST)
        window_start = cutoff - timedelta(minutes=max(0, self.config.avoid_close_window_minutes))
        # FIX: Only block if within the window (between window_start and cutoff), not after cutoff
        return window_start <= cst_now <= cutoff

    def _check_maintenance_window(self, now: datetime) -> bool:
        """Check if current time is in CME maintenance window (4-5 PM CT)."""
        try:
            cst_now = now.astimezone(CST) if now.tzinfo else now_cst()
        except Exception:
            cst_now = now_cst()
        current_time = cst_now.time()
        return self.config.maintenance_start <= current_time < self.config.maintenance_end

    def record_trade_result(self, is_win: bool) -> None:
        """Record trade outcome for consecutive loss tracking."""
        if is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            logger.warning(f"📉 Consecutive losses: {self._consecutive_losses}")

    def reset_consecutive_losses(self) -> None:
        """Reset consecutive loss counter (e.g., at daily reset)."""
        self._consecutive_losses = 0

    def evaluate_entry(
        self,
        action: str,
        quantity: int,
        entry_price: float,
        atr: float,
        account_state: Dict[str, float],
        current_position: int,
        now: Optional[datetime],
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> RiskGateResult:
        """Return whether an entry is allowed and any adjusted levels."""
        levels: Dict[str, float] = {}

        # 0) Maintenance window check (HARD BLOCK)
        if now and self._check_maintenance_window(now):
            reason = "MAINTENANCE_WINDOW"
            logger.warning("🚫 RiskGate block: CME maintenance window (4-5 PM CT)")
            return RiskGateResult(False, reason, levels)

        # 0.5) Consecutive loss lockout
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            reason = f"CONSECUTIVE_LOSS_LOCKOUT:{self._consecutive_losses}>={self.config.max_consecutive_losses}"
            logger.warning("🚫 RiskGate block: {}", reason)
            return RiskGateResult(False, reason, levels)

        # 0.75) Peak drawdown guard (high-water mark trailing)
        equity = self._extract_equity(account_state)
        self.update_equity(equity, now)
        if self.config.peak_drawdown_enabled and self._drawdown_active:
            action_mode = (self.config.peak_drawdown_action or "halt").lower()
            if action_mode == "halt":
                return RiskGateResult(False, "PEAK_DRAWDOWN_LOCKOUT", levels)

        # 1) Position cap (no pyramiding)
        projected = current_position + (quantity if action.upper().startswith("BUY") else -quantity)
        if abs(projected) > self.config.max_contracts:
            reason = f"POSITION_LIMIT:{projected}>{self.config.max_contracts}"
            logger.warning("🚫 RiskGate block: {}", reason)
            return RiskGateResult(False, reason, levels)

        # 2) Stop/tp presence and direction
        if not self._is_valid_number(stop_loss) or not self._is_valid_number(take_profit):
            reason = "INVALID_PROTECTION"
            logger.warning("🚫 RiskGate block: missing/invalid SL/TP")
            return RiskGateResult(False, reason, levels)

        stop_loss = float(stop_loss)
        take_profit = float(take_profit)

        is_buy = action.upper() in {"BUY", "SCALP_BUY"}
        if is_buy:
            if not (stop_loss < entry_price < take_profit):
                return RiskGateResult(False, "BRACKET_DIRECTION", levels)
        else:
            if not (take_profit < entry_price < stop_loss):
                return RiskGateResult(False, "BRACKET_DIRECTION", levels)

        # 3) Peak drawdown tighten mode (reduce stop distance before validation)
        if self.config.peak_drawdown_enabled and self._drawdown_active:
            action_mode = (self.config.peak_drawdown_action or "halt").lower()
            if action_mode == "tighten":
                stop_distance = abs(entry_price - stop_loss)
                tighten_mult = max(0.1, min(1.0, self.config.peak_drawdown_tighten_multiplier))
                tightened_distance = stop_distance * tighten_mult
                tightened_distance = max(self.config.min_stop_points, tightened_distance)
                new_distance = min(stop_distance, tightened_distance)
                if is_buy:
                    stop_loss = entry_price - new_distance
                else:
                    stop_loss = entry_price + new_distance
                levels["drawdown_tighten_distance"] = new_distance

        # 4) Tick alignment
        stop_loss = self._round_to_tick(stop_loss)
        take_profit = self._round_to_tick(take_profit)
        levels["stop_loss"] = stop_loss
        levels["take_profit"] = take_profit

    # 5) Risk per trade sizing vs stop distance
        # Calculate maximum allowed stop based on risk budget
        max_stop_points_from_risk = self.config.bounded_risk_usd() / 5.0  # $5 per point for MES
        levels["max_stop_from_risk"] = max_stop_points_from_risk
        actual_points = abs(entry_price - stop_loss)
        levels["actual_stop_points"] = actual_points
        
        # Check minimum stop distance (Jan 2026: prevent stops too tight for noise)
        if actual_points < self.config.min_stop_points:
            logger.warning(
                "🚫 RiskGate: Stop {:.2f} pts < min {:.2f} pts",
                actual_points,
                self.config.min_stop_points,
            )
            return RiskGateResult(False, "STOP_TOO_TIGHT", levels)
        
        # Check maximum stop distance (config hard cap)
        if actual_points > self.config.max_stop_points:
            logger.warning(
                "🚫 RiskGate: Stop {:.2f} pts > hard cap {:.2f} pts",
                actual_points,
                self.config.max_stop_points,
            )
            return RiskGateResult(False, "STOP_EXCEEDS_CAP", levels)
        
        # Check maximum stop distance (don't risk more than budget allows)
        if actual_points > max_stop_points_from_risk:
            logger.warning(
                "🚫 RiskGate: Stop {:.2f} pts > max {:.2f} pts (risk budget ${:.0f})",
                actual_points,
                max_stop_points_from_risk,
                self.config.bounded_risk_usd(),
            )
            return RiskGateResult(False, "STOP_TOO_WIDE", levels)
        
        levels["stop_points"] = actual_points
        levels["tp_points"] = abs(take_profit - entry_price)

        # 6) Margin buffer
        available = account_state.get("available_funds") or account_state.get("excess_liquidity")
        if available is None:
            return RiskGateResult(False, "ACCOUNT_UNAVAILABLE", levels)
        required_margin = (
            self.config.initial_margin_long if is_buy else self.config.initial_margin_short
        ) + self.config.margin_buffer_usd
        if available < required_margin:
            reason = f"INSUFFICIENT_MARGIN:{available:.2f}<{required_margin:.2f}"
            logger.warning("🚫 RiskGate block: {}", reason)
            return RiskGateResult(False, reason, levels)
        levels["required_margin"] = required_margin
        levels["available_funds"] = available

        # 7) Daily kill switch
        realized_today = account_state.get("realized_pnl_today", 0.0)
        if realized_today <= -abs(self.config.daily_max_loss_usd):
            return RiskGateResult(False, "DAILY_LOSS_LIMIT", levels)

        # 8) Avoid close window
        if now and self._check_close_window(now):
            return RiskGateResult(False, "NEAR_SESSION_CLOSE", levels)

        if self.config.peak_drawdown_enabled:
            drawdown_pct = None
            if self._equity_high_water:
                drawdown_pct = (equity - self._equity_high_water) / self._equity_high_water if equity is not None else None
            levels["drawdown_active"] = float(self._drawdown_active)
            if self._equity_high_water is not None:
                levels["drawdown_high_water"] = float(self._equity_high_water)
            if drawdown_pct is not None:
                levels["drawdown_pct"] = float(drawdown_pct)

        return RiskGateResult(True, "OK", levels)
