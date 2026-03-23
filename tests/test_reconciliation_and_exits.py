import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock

from shree.config import TradingConfig
from shree.execution.ib_executor import TradeExecutor
from shree.execution.live_trading_manager import LiveTradingManager
from shree.execution.components.exit_manager import ExitManager


class DummyPosition:
    def __init__(self, symbol: str, qty: int, avg_cost: float, sec_type: str = "FUT", multiplier: str = "5"):
        self.contract = SimpleNamespace(symbol=symbol, secType=sec_type, multiplier=multiplier)
        self.position = qty
        self.avgCost = avg_cost
        self.unrealizedPNL = 0.0


class DummyIB:
    def __init__(self, positions):
        self._positions = positions

    def positions(self):
        return self._positions


class ReconcileAndExitTests(unittest.TestCase):
    def test_reconcile_uses_multiplier_for_futures(self):
        # Create a fresh event loop to avoid contamination from prior asyncio.run() calls
        # in the same test session (asyncio.run() closes the loop when done, which leaves
        # asyncio.Lock() unable to get a current loop in subsequent tests).
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            ib = DummyIB([DummyPosition("MES", -3, 34915.0, "FUT", "5")])
            executor = TradeExecutor(ib=ib, config=TradingConfig(), symbol="MES")
            loop.run_until_complete(executor._reconcile_positions())
            pos = executor.positions["MES"]
            self.assertAlmostEqual(pos.avg_cost, 6983.0, delta=1.0)
            self.assertAlmostEqual(pos.market_value, pos.quantity * pos.avg_cost * 5, places=2)
        finally:
            loop.close()

    def test_reconcile_preserves_existing_position_timestamp(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            ib = DummyIB([DummyPosition("MES", 1, 33370.0, "FUT", "5")])
            executor = TradeExecutor(ib=ib, config=TradingConfig(), symbol="MES")
            original_ts = datetime.utcnow() - timedelta(hours=3)
            executor.positions["MES"] = SimpleNamespace(
                quantity=1,
                timestamp=original_ts,
                stop_loss=6660.25,
                take_profit=6695.25,
                trailing_atr_multiplier=None,
                trailing_percent=None,
                atr_value=11.0,
                entry_metadata={"reason": "EMA21_PB_LONG"},
            )

            loop.run_until_complete(executor._reconcile_positions())
            pos = executor.positions["MES"]
            self.assertEqual(pos.timestamp, original_ts)
            self.assertEqual(pos.stop_loss, 6660.25)
            self.assertEqual(pos.take_profit, 6695.25)
            self.assertEqual(pos.entry_metadata, {"reason": "EMA21_PB_LONG"})
        finally:
            loop.close()

    def _make_bare_manager(self):
        """Return a bare LiveTradingManager with only the attributes needed by ExitManager."""
        manager = LiveTradingManager.__new__(LiveTradingManager)
        manager.settings = None
        manager.contract_spec = None
        manager._exit_mgr = ExitManager(manager)
        return manager

    def test_exit_signal_ignores_notional_entry_cost(self):
        manager = self._make_bare_manager()
        current_price = 6985.0
        position = SimpleNamespace(avg_cost=34915.0, timestamp=datetime.utcnow())

        normalized = LiveTradingManager._normalize_entry_price(manager, position.avg_cost, current_price)
        self.assertAlmostEqual(normalized, 6983.0, delta=1.0)

        signal = LiveTradingManager._generate_exit_signal_for_short(manager, current_price, position)
        self.assertIsNone(signal)

    def test_exit_checks_skip_when_gap_implausible(self):
        manager = self._make_bare_manager()
        current_price = 5000.0
        position = SimpleNamespace(avg_cost=20000.0, timestamp=datetime.utcnow())

        signal = LiveTradingManager._generate_exit_signal_for_long(manager, current_price, position)
        self.assertIsNone(signal)

    def test_max_hold_exit_not_suppressed_by_active_orders(self):
        manager = LiveTradingManager.__new__(LiveTradingManager)
        manager._active_timeframe = "15m"
        manager._ft_max_hold_minutes = 120
        manager.price_history = []
        manager.one_minute_cfg = None
        manager.status = SimpleNamespace(last_atr=0.0)
        manager.settings = None
        manager.contract_spec = SimpleNamespace(point_value=5)
        manager._normalize_entry_price = lambda avg_cost, current_price: avg_cost

        executor = MagicMock()
        executor.get_current_position = AsyncMock(
            return_value=SimpleNamespace(quantity=1, avg_cost=100.0, timestamp=datetime.utcnow() - timedelta(hours=3))
        )
        executor.get_current_price = AsyncMock(return_value=101.0)
        executor.update_trailing_stops = AsyncMock(return_value=None)
        executor.get_active_order_count.return_value = 2
        manager.executor = executor

        exit_mgr = ExitManager(manager)
        exit_mgr.execute_position_exit = AsyncMock(return_value=None)
        exit_mgr.place_exit_order = AsyncMock(return_value=None)

        handled = asyncio.run(exit_mgr.check_position_exit_signals(101.0))

        self.assertTrue(handled)
        exit_mgr.execute_position_exit.assert_awaited_once()
        exit_mgr.place_exit_order.assert_not_called()

    def test_thesis_reversal_exit_triggers_when_trend_cont_long_thesis_breaks(self):
        manager = LiveTradingManager.__new__(LiveTradingManager)
        manager._active_timeframe = "15m"
        manager._ft_max_hold_minutes = 120
        manager.price_history = [{"ATR_14": 10.0}]
        manager.one_minute_cfg = None
        manager.status = SimpleNamespace(last_atr=10.0)
        manager.settings = SimpleNamespace(
            trading=SimpleNamespace(
                ft_thesis_reversal_exit_enabled=True,
                ft_thesis_reversal_arm_profit_pts=2.0,
                ft_thesis_reversal_arm_stop_fraction=0.35,
                ft_breakeven_trigger_pts=99.0,
            )
        )
        manager.contract_spec = SimpleNamespace(point_value=5)
        manager._normalize_entry_price = lambda avg_cost, current_price: avg_cost
        manager._open_trade_context = {
            "cycle_id": "abc123",
            "signal_type": "TREND_CONT_LONG",
            "stop_loss": 95.0,
            "take_profit": 112.0,
            "features": {"atr": 10.0},
        }
        manager.current_trade_features = {"ema_9": 98.0, "ema_21": 100.0, "macd_hist": -1.0, "adx": 18.0, "rsi": 48.0}

        executor = MagicMock()
        position = SimpleNamespace(quantity=1, avg_cost=100.0, timestamp=datetime.utcnow())
        executor.get_current_position = AsyncMock(return_value=position)
        executor.get_current_price = AsyncMock(side_effect=[103.0, 103.0, 96.0, 96.0])
        executor.update_trailing_stops = AsyncMock(return_value=None)
        executor.get_active_order_count.return_value = 0
        executor.get_active_bracket_levels.return_value = {"take_profit": 112.0, "stop_loss": 95.0}
        manager.executor = executor

        exit_mgr = ExitManager(manager)
        exit_mgr.execute_position_exit = AsyncMock(return_value=None)
        exit_mgr.place_exit_order = AsyncMock(return_value=None)

        first = asyncio.run(exit_mgr.check_position_exit_signals(None))
        second = asyncio.run(exit_mgr.check_position_exit_signals(None))

        self.assertFalse(first)
        self.assertTrue(second)
        exit_mgr.execute_position_exit.assert_awaited_once()
        called_signal = exit_mgr.execute_position_exit.await_args.args[0]
        self.assertEqual(called_signal["reason"], "THESIS_REVERSAL_EXIT")

    def test_thesis_reversal_exit_does_not_trigger_without_thesis_invalidation(self):
        manager = LiveTradingManager.__new__(LiveTradingManager)
        manager._active_timeframe = "15m"
        manager._ft_max_hold_minutes = 120
        manager.price_history = [{"ATR_14": 10.0}]
        manager.one_minute_cfg = None
        manager.status = SimpleNamespace(last_atr=10.0)
        manager.settings = SimpleNamespace(
            trading=SimpleNamespace(
                ft_thesis_reversal_exit_enabled=True,
                ft_thesis_reversal_arm_profit_pts=2.0,
                ft_thesis_reversal_arm_stop_fraction=0.35,
                ft_breakeven_trigger_pts=99.0,
            )
        )
        manager.contract_spec = SimpleNamespace(point_value=5)
        manager._normalize_entry_price = lambda avg_cost, current_price: avg_cost
        manager._open_trade_context = {
            "cycle_id": "xyz789",
            "signal_type": "TREND_CONT_LONG",
            "stop_loss": 95.0,
            "take_profit": 112.0,
            "features": {"atr": 10.0},
        }
        manager.current_trade_features = {"ema_9": 104.0, "ema_21": 101.0, "macd_hist": 0.4, "adx": 28.0, "rsi": 57.0}

        executor = MagicMock()
        position = SimpleNamespace(quantity=1, avg_cost=100.0, timestamp=datetime.utcnow())
        executor.get_current_position = AsyncMock(return_value=position)
        executor.get_current_price = AsyncMock(side_effect=[103.0, 103.0, 99.0, 99.0])
        executor.update_trailing_stops = AsyncMock(return_value=None)
        executor.get_active_order_count.return_value = 0
        executor.get_active_bracket_levels.return_value = {"take_profit": 112.0, "stop_loss": 95.0}
        manager.executor = executor

        exit_mgr = ExitManager(manager)
        exit_mgr.execute_position_exit = AsyncMock(return_value=None)
        exit_mgr.place_exit_order = AsyncMock(return_value=None)

        first = asyncio.run(exit_mgr.check_position_exit_signals(None))
        handled = asyncio.run(exit_mgr.check_position_exit_signals(None))

        self.assertFalse(first)
        self.assertFalse(handled)
        exit_mgr.execute_position_exit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
