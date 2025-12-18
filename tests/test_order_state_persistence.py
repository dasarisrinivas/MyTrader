from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock

from mytrader.config import Settings
from mytrader.execution.ib_executor import TradeExecutor
from mytrader.execution.live_trading_manager import LiveTradingManager
from mytrader.execution.position_manager import DecisionResult
from mytrader.monitoring.order_tracker import OrderTracker


class _EventHook:
    def __iadd__(self, handler):
        return self

    def __isub__(self, handler):
        return self

    def emit(self, *args, **kwargs):
        return None


class PassiveIB:
    """Minimal IB stub for persistence/idempotency tests."""

    def __init__(self):
        self._next_id = 9000
        self.orderStatusEvent = _EventHook()
        self.execDetailsEvent = _EventHook()

    def isConnected(self):
        return True

    def openTrades(self):
        return []

    def positions(self):
        return []

    async def accountSummaryAsync(self):
        return []

    async def reqContractDetailsAsync(self, contract):
        detail = SimpleNamespace(
            contract=SimpleNamespace(
                localSymbol=contract.symbol,
                lastTradeDateOrContractMonth="202512",
            )
        )
        return [detail]

    def placeOrder(self, contract, order):
        if not getattr(order, "orderId", 0):
            order.orderId = self._next_id
            self._next_id += 1
        status = SimpleNamespace(status="Submitted", filled=0, remaining=order.totalQuantity, avgFillPrice=0.0)
        return SimpleNamespace(contract=contract, order=order, orderStatus=status)

    def cancelOrder(self, order):
        return None

    def waitOnUpdate(self):
        return None


def test_order_tracker_signature_persistence(tmp_path):
    db_path = tmp_path / "orders_state.db"
    tracker = OrderTracker(db_path=db_path)
    tracker.record_submission_signature(
        signature="sig-1",
        symbol="ES",
        action="BUY",
        quantity=1,
        price_bucket=100.0,
        bar_timestamp="2025-01-01T10:00:00",
        signal_id="cycle-1",
        strategy_name="legacy",
    )
    assert tracker.signature_exists("sig-1", ttl_seconds=600)
    assert not tracker.signature_exists("sig-1", ttl_seconds=0)


def test_order_tracker_last_trade_time(tmp_path):
    db_path = tmp_path / "orders_state.db"
    tracker = OrderTracker(db_path=db_path)
    ts = datetime.now(timezone.utc).replace(microsecond=0)
    tracker.record_last_trade_time("ES", ts)
    loaded = tracker.get_last_trade_time("ES")
    assert loaded == ts


@pytest.mark.asyncio
async def test_duplicate_submission_blocks_after_persistence(tmp_path, monkeypatch):
    settings = Settings()
    ib = PassiveIB()
    executor = TradeExecutor(ib, settings.trading, settings.data.ibkr_symbol)
    executor.order_tracker = OrderTracker(db_path=tmp_path / "orders_state.db")
    monkeypatch.setattr(
        executor.position_manager,
        "can_place_order",
        AsyncMock(return_value=DecisionResult(allowed_contracts=1, reason="ok")),
    )
    monkeypatch.setattr(
        executor,
        "_confirm_protective_orders",
        AsyncMock(return_value=True),
    )
    metadata = {
        "bar_close_timestamp": "2025-01-01T10:00:00",
        "signal_id": "cycle-a",
        "entry_price_bucket": 100.0,
        "strategy_name": "legacy",
        "trade_cycle_id": "cycle-a",
        "entry_price": 100.0,
    }
    signature = executor.generate_submission_signature("BUY", 1, metadata)
    executor.record_submission(signature, "BUY", 1, metadata)
    result = await executor.place_order(
        action="BUY",
        quantity=1,
        limit_price=100.0,
        stop_loss=99.0,
        take_profit=101.0,
        metadata=dict(metadata),
    )
    assert result.status == "Cancelled"
    assert "duplicate" in (result.message or "").lower()


def test_live_manager_restores_cooldown_state(tmp_path):
    settings = Settings()
    tracker = OrderTracker(db_path=tmp_path / "orders_state.db")
    ts = datetime.now(timezone.utc).replace(microsecond=0)
    tracker.record_last_trade_time(settings.data.ibkr_symbol, ts)
    manager = LiveTradingManager(settings, simulation_mode=True)
    manager.executor = SimpleNamespace(order_tracker=tracker)
    manager._load_persistent_cooldown_state()
    assert manager._last_trade_time == ts
