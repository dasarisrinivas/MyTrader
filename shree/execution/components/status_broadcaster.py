"""Websocket / callback broadcasting helpers."""

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict


class StatusBroadcaster:
    """Handles emitting status/signal/order updates to external listeners."""

    def __init__(self, manager: "LiveTradingManager"):  # noqa: F821 (forward reference)
        self.manager = manager

    async def broadcast_status(self):
        if self.manager.on_status_update:
            await self.manager.on_status_update(asdict(self.manager.status))

    async def broadcast_signal(self, signal, price: float):
        if self.manager.on_signal_generated:
            await self.manager.on_signal_generated(
                {
                    "action": signal.action,
                    "confidence": signal.confidence,
                    "price": price,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metadata": signal.metadata if isinstance(signal.metadata, dict) else {},
                }
            )

    async def broadcast_order_update(self, order_data: Dict):
        if self.manager.on_order_update:
            order_data["timestamp"] = datetime.now(timezone.utc).isoformat()
            await self.manager.on_order_update(order_data)

    async def broadcast_error(self, error_msg: str):
        if self.manager.on_error:
            await self.manager.on_error(
                {
                    "error": error_msg,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

    def update_status_from_tracker(self) -> None:
        tracker = self.manager.tracker
        if not tracker:
            return
        self.manager.status.daily_pnl = getattr(tracker, "daily_pnl", self.manager.status.daily_pnl)
        tracker_unrealized = getattr(tracker, "unrealized_pnl", None)
        if tracker_unrealized is not None:
            self.manager.status.unrealized_pnl = tracker_unrealized
