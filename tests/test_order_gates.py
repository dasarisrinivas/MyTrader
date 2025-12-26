import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest

from mytrader.config import TradingConfig
from mytrader.execution.components.order_coordinator import OrderCoordinator
from mytrader.execution.ib_executor import TradeExecutor


class DummyIB:
    def __init__(self, positions=None):
        self._positions = positions or []

    def positions(self):
        return self._positions

    def isConnected(self):
        return False


class StubManager:
    def __init__(self):
        trading = TradingConfig()
        data = SimpleNamespace(ibkr_symbol="MES")
        self.settings = SimpleNamespace(trading=trading, data=data)
        self._cooldown_seconds = 60
        self._last_trade_time = None
        self.executor = SimpleNamespace(get_current_position=self._get_position)
        self._position = None

    async def _get_position(self):
        return self._position


class OrderGateTests(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        asyncio.set_event_loop(None)
        self.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_protective_guard_blocks_missing_levels(self):
        executor = TradeExecutor(ib=DummyIB(), config=TradingConfig(), symbol="MES")
        result = self._run(
            executor.place_order(
                action="BUY",
                quantity=1,
                limit_price=100.0,
                stop_loss=None,
                take_profit=101.0,
                metadata={"bar_close_timestamp": "2024-01-01T00:00:00"},
                entry_price=100.0,
            )
        )
        self.assertEqual(result.status, "Cancelled")
        self.assertIn("Protective guard blocked order", result.message)

    def test_executor_dedupes_same_bar_and_action(self):
        executor = TradeExecutor(ib=DummyIB(), config=TradingConfig(), symbol="MES")
        metadata = {"bar_close_timestamp": "2024-01-01T00:00:00", "entry_price": 100.0}
        first = self._run(
            executor.place_order(
                action="BUY",
                quantity=1,
                limit_price=100.0,
                stop_loss=99.0,
                take_profit=101.0,
                metadata=dict(metadata),
                entry_price=100.0,
            )
        )
        self.assertEqual(first.status, "Cancelled")
        dup = self._run(
            executor.place_order(
                action="BUY",
                quantity=1,
                limit_price=100.0,
                stop_loss=99.0,
                take_profit=101.0,
                metadata=dict(metadata),
                entry_price=100.0,
            )
        )
        self.assertEqual(dup.status, "Cancelled")
        self.assertIn("Duplicate signal submission blocked", dup.message)

    def test_entry_gate_blocks_cooldown(self):
        manager = StubManager()
        manager._last_trade_time = datetime.now(timezone.utc) - timedelta(seconds=10)
        coordinator = OrderCoordinator(manager)
        allowed, reason = self._run(
            coordinator.enforce_entry_gates(
                "BUY",
                {"bar_close_timestamp": "2024-01-01T00:00:00"},
            )
        )
        self.assertFalse(allowed)
        self.assertIn("COOLDOWN", reason)

    def test_entry_gate_blocks_open_position(self):
        manager = StubManager()
        manager._position = SimpleNamespace(quantity=1)
        coordinator = OrderCoordinator(manager)
        allowed, reason = self._run(
            coordinator.enforce_entry_gates(
                "SELL",
                {"bar_close_timestamp": "2024-01-01T00:00:00"},
            )
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "POSITION_OPEN")


if __name__ == "__main__":
    unittest.main()
