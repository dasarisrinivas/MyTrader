"""Trade decision evaluation and sizing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from ...utils.logger import logger

if TYPE_CHECKING:  # pragma: no cover
    from ..live_trading_manager import LiveTradingManager


@dataclass
class TradingContext:
    filters_passed: bool
    filters_applied: List[str]
    active_orders: int
    current_position: Optional[object]
    min_confidence: float


@dataclass
class DecisionOutcome:
    allow: bool
    reason: Optional[str] = None
    exit_only: bool = False
    exit_quantity: int = 0


@dataclass
class ExitDecision:
    should_exit: bool
    action: str = "HOLD"
    quantity: int = 0
    reason: Optional[str] = None


@dataclass
class StopLevels:
    stop_loss: float
    take_profit: float
    meta: Dict[str, float] = field(default_factory=dict)


@dataclass
class PositionAdjustment:
    scale: str
    size_multiplier: float
    reason: Optional[str] = None


@dataclass
class MarketState:
    tradable: bool
    reason: Optional[str] = None
    issues: List[str] = field(default_factory=list)


class TradeDecisionEngine:
    """Evaluates buy/sell/hold actions."""

    def __init__(self, manager: "LiveTradingManager"):  # noqa: F821 (forward reference)
        self.manager = manager

    def should_enter_trade(self, signal, context: TradingContext) -> DecisionOutcome:
        """Determine if a trade should proceed based on filters and state."""
        if signal.action == "HOLD":
            return DecisionOutcome(False, reason="HOLD")

        if not context.filters_passed:
            reason_text = ", ".join(context.filters_applied) if context.filters_applied else "filter_block"
            return DecisionOutcome(False, reason=reason_text)

        if context.active_orders > 0:
            return DecisionOutcome(False, reason="ACTIVE_ORDERS")

        pos = context.current_position
        if pos and getattr(pos, "quantity", 0) != 0:
            is_buy_signal = signal.action in ["BUY", "SCALP_BUY"]
            is_sell_signal = signal.action in ["SELL", "SCALP_SELL"]
            qty = abs(getattr(pos, "quantity", 0))
            if (pos.quantity > 0 and is_sell_signal) or (pos.quantity < 0 and is_buy_signal):
                return DecisionOutcome(False, reason="EXIT_REQUIRED", exit_only=True, exit_quantity=qty)
            return DecisionOutcome(False, reason="POSITION_OPEN")

        if signal.confidence < context.min_confidence:
            return DecisionOutcome(False, reason="CONFIDENCE_BELOW_MIN")

        return DecisionOutcome(True)

    def calculate_position_size(self, signal, metadata: Dict[str, float]) -> int:
        """Compute position size using risk manager and metadata scaling."""
        m = self.manager
        risk_stats = m.risk.get_statistics()
        qty = m.risk.position_size(
            m.settings.trading.initial_capital,
            signal.confidence,
            win_rate=risk_stats.get("win_rate"),
            avg_win=risk_stats.get("avg_win"),
            avg_loss=risk_stats.get("avg_loss"),
        )

        scaler = float(metadata.get("position_scaler", 1.0))
        if scaler > 0:
            qty = max(1, int(round(qty * scaler)))

        qty = min(qty, m.settings.trading.max_position_size)

        logger.info("🛠️ Position sizing -> qty=%d (scaler=%.2f)", qty, scaler)
        return qty

    def evaluate_exit_signals(self) -> ExitDecision:
        """Evaluate whether existing positions should be flattened."""
        m = self.manager
        position_qty = getattr(m.status, "current_position", 0)
        if position_qty == 0:
            return ExitDecision(False, reason="NO_POSITION")

        active_orders = m.executor.get_active_order_count(sync=True) if m.executor else 0
        if active_orders > 0:
            return ExitDecision(False, reason="PENDING_ORDERS")

        tracker = getattr(m, "tracker", None)
        unrealized = getattr(tracker, "unrealized_pnl", None) if tracker else m.status.unrealized_pnl
        daily_pnl = getattr(tracker, "daily_pnl", None) if tracker else m.status.daily_pnl

        max_daily_loss = getattr(m.settings.trading, "max_daily_loss", None)
        max_loss_per_trade = getattr(m.settings.trading, "max_loss_per_trade", None)

        exit_action = "SELL" if position_qty > 0 else "BUY"
        qty = abs(position_qty)

        if max_daily_loss is not None and daily_pnl is not None and daily_pnl <= -abs(max_daily_loss):
            return ExitDecision(True, action=exit_action, quantity=qty, reason="DAILY_LOSS_LIMIT")

        if max_loss_per_trade is not None and unrealized is not None and unrealized <= -abs(max_loss_per_trade):
            return ExitDecision(True, action=exit_action, quantity=qty, reason="TRADE_LOSS_LIMIT")

        return ExitDecision(False, action=exit_action, quantity=qty, reason="HOLD")

    def calculate_dynamic_stops(self) -> StopLevels:
        """Derive dynamic protective levels from recent context."""
        m = self.manager
        price = (
            m.status.current_price
            or (m.price_history[-1]["close"] if m.price_history else 0.0)
        )
        action = m.status.last_signal or "BUY"

        if not m.risk:
            return StopLevels(stop_loss=price, take_profit=price, meta={"reason": "NO_RISK_MANAGER"})

        atr = 0.0
        if m.current_trade_features:
            atr = float(m.current_trade_features.get("atr", 0.0))
        if atr <= 0 and m.price_history:
            atr = float(m.price_history[-1].get("ATR_14", 0.0) or 0.0)
        atr = max(atr, 1e-6)

        direction = "long" if action in ["BUY", "SCALP_BUY"] else "short"
        stop_loss, take_profit = m.risk.calculate_dynamic_stops(
            entry_price=price,
            current_atr=atr,
            direction=direction,  # type: ignore[arg-type]
        )
        meta = {"atr_used": atr, "direction": direction}
        return StopLevels(stop_loss=stop_loss, take_profit=take_profit, meta=meta)

    def assess_position_scaling(self) -> PositionAdjustment:
        """Suggest position scaling based on recent performance."""
        m = self.manager
        if not m.risk:
            return PositionAdjustment(scale="hold", size_multiplier=1.0, reason="NO_RISK_MANAGER")

        stats = m.risk.get_statistics()
        win_rate = stats.get("win_rate", 0.0) or 0.0
        losing_trades = stats.get("losing_trades", 0)
        size_multiplier = 1.0
        reason = "BASELINE"
        scale = "hold"

        if win_rate >= 0.6 and stats.get("total_trades", 0) >= 10:
            size_multiplier = 1.25
            scale = "increase"
            reason = "POSITIVE_WIN_RATE"
        elif losing_trades >= 3:
            size_multiplier = 0.75
            scale = "decrease"
            reason = "LOSING_STREAK"

        max_size = getattr(m.settings.trading, "max_position_size", size_multiplier)
        size_multiplier = min(size_multiplier, max_size)
        return PositionAdjustment(scale=scale, size_multiplier=size_multiplier, reason=reason)

    def validate_market_conditions(self) -> MarketState:
        """Validate whether market/risk constraints allow trading."""
        m = self.manager
        issues: List[str] = []

        if not m.risk or not m.tracker:
            return MarketState(tradable=True, issues=issues)

        stats = m.risk.get_statistics()
        if stats.get("daily_loss", 0) >= getattr(m.settings.trading, "max_daily_loss", float("inf")):
            issues.append("DAILY_LOSS_LIMIT")

        if stats.get("daily_trade_count", 0) >= getattr(m.settings.trading, "max_daily_trades", float("inf")):
            issues.append("DAILY_TRADE_LIMIT")

        portfolio_heat = stats.get("portfolio_heat", 0.0) or 0.0
        max_heat = getattr(m.settings.trading, "margin_limit_pct", 100) * 100
        if portfolio_heat >= max_heat:
            issues.append("PORTFOLIO_HEAT")

        tradable = len(issues) == 0
        reason = None if tradable else ", ".join(issues)
        return MarketState(tradable=tradable, reason=reason, issues=issues)
