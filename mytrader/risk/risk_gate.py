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
    risk_per_trade_usd: float = field(default_factory=lambda: float(os.environ.get("RISK_PER_TRADE_USD", "50")))
    risk_per_trade_min: float = 25.0
    risk_per_trade_max: float = 75.0
    min_stop_points: float = 2.0
    margin_buffer_usd: float = field(default_factory=lambda: float(os.environ.get("MARGIN_BUFFER_USD", "1000")))
    initial_margin_long: float = field(default_factory=lambda: float(os.environ.get("MES_INITIAL_MARGIN_LONG", "2464")))
    initial_margin_short: float = field(default_factory=lambda: float(os.environ.get("MES_INITIAL_MARGIN_SHORT", "2305.6")))
    daily_max_loss_usd: float = field(default_factory=lambda: float(os.environ.get("DAILY_MAX_LOSS_USD", "150")))
    avoid_close_window_minutes: int = field(default_factory=lambda: int(os.environ.get("AVOID_CLOSE_WINDOW_MINUTES", "20")))
    avoid_close_enabled: bool = field(default_factory=lambda: _env_bool("AVOID_CLOSE_WINDOW_ENABLED", True))
    intraday_close_time: time = time(15, 0)  # RTH close CT
    tick_size: float = 0.25

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

    @staticmethod
    def _is_valid_number(val: Optional[float]) -> bool:
        return val is not None and isinstance(val, (int, float)) and math.isfinite(val)

    def _round_to_tick(self, value: float) -> float:
        tick = max(self.config.tick_size, 1e-6)
        return round(value / tick) * tick

    def _check_close_window(self, now: datetime) -> bool:
        if not self.config.avoid_close_enabled:
            return False
        try:
            cst_now = now.astimezone(CST) if now.tzinfo else now_cst()
        except Exception:
            cst_now = now_cst()
        cutoff = datetime.combine(cst_now.date(), self.config.intraday_close_time, tzinfo=CST)
        window_start = cutoff - timedelta(minutes=max(0, self.config.avoid_close_window_minutes))
        return cst_now >= window_start

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

        # 1) Position cap (no pyramiding)
        projected = current_position + (quantity if action.upper().startswith("BUY") else -quantity)
        if abs(projected) > self.config.max_contracts:
            reason = f"POSITION_LIMIT:{projected}>{self.config.max_contracts}"
            logger.warning("🚫 RiskGate block: %s", reason)
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

        # 3) Tick alignment
        stop_loss = self._round_to_tick(stop_loss)
        take_profit = self._round_to_tick(take_profit)
        levels["stop_loss"] = stop_loss
        levels["take_profit"] = take_profit

        # 4) Risk per trade sizing vs stop distance
        # Calculate maximum allowed stop based on risk budget
        max_stop_points_from_risk = self.config.bounded_risk_usd() / 5.0  # $5 per point for MES
        levels["max_stop_from_risk"] = max_stop_points_from_risk
        actual_points = abs(entry_price - stop_loss)
        levels["actual_stop_points"] = actual_points
        
        # Check minimum stop distance (Jan 2026: prevent stops too tight for noise)
        if actual_points < self.config.min_stop_points:
            logger.warning(
                "🚫 RiskGate: Stop %.2f pts < min %.2f pts",
                actual_points, self.config.min_stop_points
            )
            return RiskGateResult(False, "STOP_TOO_TIGHT", levels)
        
        # Check maximum stop distance (don't risk more than budget allows)
        if actual_points > max_stop_points_from_risk:
            logger.warning(
                "🚫 RiskGate: Stop %.2f pts > max %.2f pts (risk budget $%.0f)",
                actual_points, max_stop_points_from_risk, self.config.bounded_risk_usd()
            )
            return RiskGateResult(False, "STOP_TOO_WIDE", levels)
        
        levels["stop_points"] = actual_points
        levels["tp_points"] = abs(take_profit - entry_price)

        # 5) Margin buffer
        available = account_state.get("available_funds") or account_state.get("excess_liquidity")
        if available is None:
            return RiskGateResult(False, "ACCOUNT_UNAVAILABLE", levels)
        required_margin = (
            self.config.initial_margin_long if is_buy else self.config.initial_margin_short
        ) + self.config.margin_buffer_usd
        if available < required_margin:
            reason = f"INSUFFICIENT_MARGIN:{available:.2f}<{required_margin:.2f}"
            logger.warning("🚫 RiskGate block: %s", reason)
            return RiskGateResult(False, reason, levels)
        levels["required_margin"] = required_margin
        levels["available_funds"] = available

        # 6) Daily kill switch
        realized_today = account_state.get("realized_pnl_today", 0.0)
        if realized_today <= -abs(self.config.daily_max_loss_usd):
            return RiskGateResult(False, "DAILY_LOSS_LIMIT", levels)

        # 7) Avoid close window
        if now and self._check_close_window(now):
            return RiskGateResult(False, "NEAR_SESSION_CLOSE", levels)

        return RiskGateResult(True, "OK", levels)
