import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import pandas as pd

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
        self.assertIn("Not connected", result.message)

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
        self.assertIn("Not connected", dup.message)

    def test_entry_gate_blocks_cooldown(self):
        manager = StubManager()
        manager._last_trade_time = datetime.now(timezone.utc) - timedelta(seconds=10)
        coordinator = OrderCoordinator(manager)
        allowed, reason, _ = self._run(
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
        allowed, reason, _ = self._run(
            coordinator.enforce_entry_gates(
                "SELL",
                {"bar_close_timestamp": "2024-01-01T00:00:00"},
            )
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "POSITION_OPEN")

    def test_execute_trade_respects_risk_gate_block(self):
        manager = StubManager()
        manager._current_cycle_id = "cycle"
        manager._add_reason_code = lambda *args, **kwargs: None
        manager._validate_entry_guard = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("entry guard should not run when risk gate blocks")
        )
        manager._enforce_risk_gate = AsyncMock(return_value=(False, {}))
        manager.risk = SimpleNamespace(
            can_trade=lambda qty: True,
            get_statistics=lambda: {},
            position_size=lambda *args, **kwargs: 1,
        )
        manager.trade_decision_engine = SimpleNamespace(calculate_position_size=lambda *args, **kwargs: 1)
        manager.risk_controller = SimpleNamespace(
            calculate_stop_loss=lambda **kwargs: (99.0, 101.0, {}),
        )
        manager._min_confidence_for_trade = 0.6
        manager._min_stop_distance = 0.25
        manager._trade_time_lock = None
        manager.price_history = []
        manager.current_trade_features = {}
        manager._cycle_context = {}
        manager._active_reason_codes = set()
        manager._current_entry_cycle_id = None
        manager._last_candle_processed = None
        manager.status = SimpleNamespace(current_position=0)
        manager.executor = SimpleNamespace(
            is_order_locked=lambda: False,
            get_current_position=AsyncMock(return_value=None),
        )
        manager._broadcast_order_update = AsyncMock()
        manager._broadcast_error = AsyncMock()

        features = pd.DataFrame(
            [
                {
                    "close": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "volume": 1000,
                    "ATR_14": 2.0,
                    "RSI_14": 50,
                    "MACD": 0.1,
                }
            ]
        )
        signal = SimpleNamespace(action="BUY", confidence=0.7, metadata={"stop_loss": 99.0, "take_profit": 101.0})

        with patch(
            "mytrader.execution.components.order_coordinator.detect_market_regime",
            return_value=(SimpleNamespace(value="TEST"), 1.0),
        ), patch(
            "mytrader.execution.components.order_coordinator.get_regime_parameters",
            return_value={"volatility": "MED"},
        ), patch.object(
            OrderCoordinator, "enforce_entry_gates", AsyncMock(return_value=(True, "OK", "key"))
        ):
            coordinator = OrderCoordinator(manager)
            manager.order_coordinator = coordinator
            result = self._run(
                coordinator.execute_trade_with_risk_checks(
                    signal,
                    current_price=100.0,
                    features=features,
                )
            )
            self.assertIsNone(result)
            manager._enforce_risk_gate.assert_awaited()


if __name__ == "__main__":
    unittest.main()
