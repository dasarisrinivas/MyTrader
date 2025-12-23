"""Trade decision evaluation and sizing."""
from __future__ import annotations

from dataclasses import dataclass
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
