import asyncio
from datetime import datetime
from types import SimpleNamespace
import unittest

from mytrader.config import TradingConfig
from mytrader.execution.ib_executor import TradeExecutor
from mytrader.execution.live_trading_manager import LiveTradingManager


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
        ib = DummyIB([DummyPosition("MES", -3, 34915.0, "FUT", "5")])
        executor = TradeExecutor(ib=ib, config=TradingConfig(), symbol="MES")

        asyncio.run(executor._reconcile_positions())

        pos = executor.positions["MES"]
        self.assertAlmostEqual(pos.avg_cost, 6983.0, delta=1.0)
        self.assertAlmostEqual(pos.market_value, pos.quantity * pos.avg_cost * 5, places=2)

    def test_exit_signal_ignores_notional_entry_cost(self):
        manager = LiveTradingManager.__new__(LiveTradingManager)
        current_price = 6985.0
        position = SimpleNamespace(avg_cost=34915.0, timestamp=datetime.utcnow())

        normalized = LiveTradingManager._normalize_entry_price(manager, position.avg_cost, current_price)
        self.assertAlmostEqual(normalized, 6983.0, delta=1.0)

        signal = LiveTradingManager._generate_exit_signal_for_short(manager, current_price, position)
        self.assertIsNone(signal)

    def test_exit_checks_skip_when_gap_implausible(self):
        manager = LiveTradingManager.__new__(LiveTradingManager)
        current_price = 5000.0
        position = SimpleNamespace(avg_cost=20000.0, timestamp=datetime.utcnow())

        signal = LiveTradingManager._generate_exit_signal_for_long(manager, current_price, position)
        self.assertIsNone(signal)


if __name__ == "__main__":
    unittest.main()
