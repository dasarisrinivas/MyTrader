"""Order tracking, metadata, and trade lifecycle helpers."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..learning.trade_learning import ExecutionMetrics, TradeLearningPayload
from ..utils.logger import logger
from ..utils.timezone_utils import now_cst


class OrderCoordinator:
    """Encapsulates order metadata and trade lifecycle persistence."""

    def __init__(self, manager: "LiveTradingManager"):  # noqa: F821 (forward reference)
        self.manager = manager

    def prepare_order_metadata(
        self,
        base_metadata: Optional[Dict[str, Any]],
        entry_price: float,
        strategy_name: str,
    ) -> Dict[str, Any]:
        metadata = dict(base_metadata or {})
        metadata.setdefault("trade_cycle_id", self.manager._current_cycle_id)
        candle_ts = self.manager._last_candle_processed
        if isinstance(candle_ts, datetime):
            metadata.setdefault("bar_close_timestamp", candle_ts.isoformat())
        else:
            metadata.setdefault(
                "bar_close_timestamp",
                now_cst().replace(second=0, microsecond=0).isoformat(),
            )
        metadata.setdefault(
            "signal_id",
            metadata.get("signal_id") or metadata.get("signal_type") or self.manager._current_cycle_id,
        )
        metadata.setdefault("strategy_name", strategy_name)
        metadata["entry_price"] = entry_price
        metadata.setdefault("entry_price_bucket", self.bucket_entry_price(entry_price))
        return metadata

    def bucket_entry_price(self, price: float) -> float:
        tick_size = max(getattr(self.manager.settings.trading, "tick_size", 0.25), 1e-6)
        bucket_ticks = max(1, getattr(self.manager.settings.trading, "min_stop_distance_ticks", 4))
        bucket_size = tick_size * bucket_ticks
        return round(price / bucket_size) * bucket_size

    def register_trade_entry(
        self,
        cycle_id: Optional[str],
        action: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata = metadata or {}
        cycle_id = cycle_id or self.manager._current_cycle_id or uuid.uuid4().hex[:12]

        self.manager._current_entry_cycle_id = cycle_id
        context = self.manager._cycle_context.get(cycle_id, {})
        is_long = action in ("BUY", "SCALP_BUY")
        self.manager._open_trade_context = {
            "cycle_id": cycle_id,
            "action": action,
            "is_long": is_long,
            "quantity": quantity,
            "entry_price": entry_price,
            "entry_time": now_cst().isoformat(),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "metadata": metadata,
            "regime": context.get("regime"),
            "volatility": context.get("volatility"),
            "aws": context.get("aws"),
            "signal_confidence": context.get("signal_confidence"),
            "signal_type": context.get("signal_type"),
            "features": self.manager.current_trade_features or {},
        }
        reason_codes = context.get("reason_codes", set())
        self.manager._active_reason_codes = set(reason_codes)

        logger.info(
            "📝 Registered trade entry: cycle_id=%s, action=%s, qty=%s, entry=%.2f",
            cycle_id,
            action,
            quantity,
            entry_price,
        )

    async def finalize_trade(
        self,
        exit_price: float,
        exit_time: datetime,
        realized_pnl: float,
    ) -> None:
        if not self.manager._open_trade_context or not self.manager.learning_recorder:
            return
        ctx = self.manager._open_trade_context
        direction = 1 if ctx.get("is_long") else -1
        contract = getattr(self.manager.settings.data, "ibkr_symbol", "ES")
        payload = TradeLearningPayload(
            trade_cycle_id=ctx["cycle_id"],
            symbol=contract,
            contract=contract,
            quantity=ctx["quantity"],
            side=ctx["action"],
            entry_time=ctx["entry_time"],
            entry_price=ctx["entry_price"],
            exit_time=exit_time.isoformat(),
            exit_price=exit_price,
            stop_loss=ctx["stop_loss"],
            take_profit=ctx["take_profit"],
            signal_type=ctx.get("signal_type"),
            signal_confidence=ctx.get("signal_confidence"),
            regime=ctx.get("regime"),
            volatility=ctx.get("volatility"),
            advisory=ctx.get("aws"),
            risk_parameters={
                "distance_points": abs(ctx["entry_price"] - ctx["stop_loss"]),
                "target_points": abs(ctx["take_profit"] - ctx["entry_price"]),
            },
            features=ctx.get("features") or {},
            execution_metrics=ExecutionMetrics(),
            reason_codes=sorted(self.manager._active_reason_codes),
            outcome="LOSS" if realized_pnl < 0 else "WIN" if realized_pnl > 0 else "BREAKEVEN",
            pnl=realized_pnl,
        )
        self.manager.learning_recorder.record_outcome(payload)
        if self.manager._local_kb:
            try:
                self.manager._local_kb.record_trade(payload)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Local KB record skipped: {exc}")
        candles = self.manager.price_history[-120:]
        history_ctx = {
            "regime": ctx.get("regime"),
            "volatility": ctx.get("volatility"),
            "symbol": contract,
        }
        if candles:
            self.manager.learning_recorder.record_history_snapshot(
                payload.trade_cycle_id,
                candles,
                history_ctx,
            )
        self.manager._open_trade_context = None
        self.manager._active_reason_codes.clear()
        self.manager._cycle_context.pop(payload.trade_cycle_id, None)
        if self.manager._current_entry_cycle_id == payload.trade_cycle_id:
            self.manager._current_entry_cycle_id = None

    def add_reason_code(self, code: str) -> None:
        if not code:
            return
        self.manager._active_reason_codes.add(code)
        cycle_ctx = self.manager._cycle_context.get(self.manager._current_cycle_id)
        if cycle_ctx is not None:
            if "reason_codes" not in cycle_ctx or not isinstance(cycle_ctx["reason_codes"], set):
                cycle_ctx["reason_codes"] = set()
            cycle_ctx["reason_codes"].add(code)
