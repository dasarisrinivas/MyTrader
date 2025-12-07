"""
Unit Tests for LiveDataManager
===============================
Tests for live data subscriptions, tick-to-candle conversion, and event handling.
"""
import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.mock_ib_server import MockIB, MockContract, MockTicker


class TestTickToCandleRebuild(unittest.TestCase):
    """
    Test: test_tick_to_candle_rebuild
    
    Feed ticks and expect candles with correct OHLC values.
    """
    
    def test_candle_building_basic(self):
        """Test basic candle building from ticks."""
        from mytrader.data.live_data_manager import CandleBuilder, NormalizedTick, TickType
        
        builder = CandleBuilder(symbol="ES", interval_seconds=60)
        
        # Create a series of ticks within a 1-minute window
        base_time = datetime(2025, 12, 7, 10, 0, 0, tzinfo=timezone.utc)
        
        ticks = [
            NormalizedTick(timestamp=base_time + timedelta(seconds=0), symbol="ES", last=5000.0, last_size=1),
            NormalizedTick(timestamp=base_time + timedelta(seconds=10), symbol="ES", last=5002.0, last_size=2),
            NormalizedTick(timestamp=base_time + timedelta(seconds=20), symbol="ES", last=4998.0, last_size=1),  # Low
            NormalizedTick(timestamp=base_time + timedelta(seconds=30), symbol="ES", last=5005.0, last_size=3),  # High
            NormalizedTick(timestamp=base_time + timedelta(seconds=50), symbol="ES", last=5001.0, last_size=1),  # Close
        ]
        
        # Process ticks
        completed_candles = []
        for tick in ticks:
            result = builder.process_tick(tick)
            if result:
                completed_candles.append(result)
        
        # Get current candle
        current = builder.get_current_candle()
        
        # Verify candle OHLC
        self.assertIsNotNone(current, "Should have a current candle")
        self.assertEqual(current.open, 5000.0, "Open should be first tick price")
        self.assertEqual(current.high, 5005.0, "High should be highest price")
        self.assertEqual(current.low, 4998.0, "Low should be lowest price")
        self.assertEqual(current.close, 5001.0, "Close should be last tick price")
        self.assertEqual(current.tick_count, 5, "Should have 5 ticks")
        self.assertEqual(current.volume, 8, "Total volume should be 8")
    
    def test_candle_completion_on_new_period(self):
        """Test that candle completes when new period starts."""
        from mytrader.data.live_data_manager import CandleBuilder, NormalizedTick
        
        builder = CandleBuilder(symbol="ES", interval_seconds=60)
        
        # First candle ticks (minute 10:00)
        t1 = datetime(2025, 12, 7, 10, 0, 0, tzinfo=timezone.utc)
        builder.process_tick(NormalizedTick(timestamp=t1, symbol="ES", last=5000.0, last_size=1))
        builder.process_tick(NormalizedTick(timestamp=t1 + timedelta(seconds=30), symbol="ES", last=5010.0, last_size=1))
        
        # New minute starts - should complete previous candle
        t2 = datetime(2025, 12, 7, 10, 1, 0, tzinfo=timezone.utc)
        completed = builder.process_tick(NormalizedTick(timestamp=t2, symbol="ES", last=5015.0, last_size=1))
        
        # Should return completed candle
        self.assertIsNotNone(completed, "Should have completed candle")
        self.assertEqual(completed.open, 5000.0)
        self.assertEqual(completed.close, 5010.0)
        
        # Current candle should be new
        current = builder.get_current_candle()
        self.assertEqual(current.open, 5015.0)
        self.assertEqual(current.close, 5015.0)
    
    def test_rebuild_candles_from_ticks(self):
        """Test rebuilding candles from historical tick data."""
        from mytrader.data.live_data_manager import CandleBuilder, NormalizedTick
        
        builder = CandleBuilder(symbol="ES", interval_seconds=60)
        
        # Create 3 minutes of tick data
        ticks = []
        for minute in range(3):
            for second in [0, 20, 40]:
                ts = datetime(2025, 12, 7, 10, minute, second, tzinfo=timezone.utc)
                price = 5000.0 + minute * 10 + second / 10
                ticks.append(NormalizedTick(timestamp=ts, symbol="ES", last=price, last_size=1))
        
        # Rebuild candles
        candles = builder.rebuild_candles_from_ticks(ticks)
        
        # Should have 3 candles
        self.assertEqual(len(candles), 3, "Should have 3 candles")
        
        # Verify each candle
        for i, candle in enumerate(candles):
            self.assertEqual(candle.tick_count, 3, f"Candle {i} should have 3 ticks")
            self.assertEqual(candle.interval_seconds, 60)
    
    def test_empty_tick_handling(self):
        """Test handling of ticks with no last price."""
        from mytrader.data.live_data_manager import CandleBuilder, NormalizedTick
        
        builder = CandleBuilder(symbol="ES", interval_seconds=60)
        
        ts = datetime(2025, 12, 7, 10, 0, 0, tzinfo=timezone.utc)
        
        # Tick with no last price should be ignored
        result = builder.process_tick(NormalizedTick(timestamp=ts, symbol="ES", last=None, bid=5000.0, ask=5001.0))
        self.assertIsNone(result)
        self.assertIsNone(builder.get_current_candle())
        
        # Tick with last price should work
        builder.process_tick(NormalizedTick(timestamp=ts, symbol="ES", last=5000.5, last_size=1))
        self.assertIsNotNone(builder.get_current_candle())


class TestLiveDataManager(unittest.TestCase):
    """Test LiveDataManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_ib = MockIB()
        self.mock_ib._connected = True
    
    def test_subscription(self):
        """Test subscribing to market data."""
        from mytrader.data.live_data_manager import LiveDataManager, LiveDataConfig
        
        config = LiveDataConfig(log_dir=self.temp_dir)
        
        async def run_test():
            manager = LiveDataManager(self.mock_ib, config)
            contract = MockContract(symbol="ES")
            success = await manager.subscribe(contract)
            return success, manager.get_subscribed_symbols()
        
        result, symbols = asyncio.run(run_test())
        
        self.assertTrue(result, "Subscription should succeed")
        self.assertIn("ES", symbols)
    
    def test_callback_registration(self):
        """Test registering and unregistering callbacks."""
        from mytrader.data.live_data_manager import LiveDataManager, LiveDataConfig, NormalizedTick
        
        async def run_test():
            config = LiveDataConfig(log_dir=self.temp_dir)
            manager = LiveDataManager(self.mock_ib, config)
            
            callback_data = []
            
            def on_tick(tick: NormalizedTick):
                callback_data.append(tick)
            
            # Register callback
            manager.register_on_tick(on_tick)
            count_after_register = len(manager._on_tick_callbacks)
            
            # Unregister callback
            manager.unregister_on_tick(on_tick)
            count_after_unregister = len(manager._on_tick_callbacks)
            
            return count_after_register, count_after_unregister
        
        count_reg, count_unreg = asyncio.run(run_test())
        self.assertEqual(count_reg, 1)
        self.assertEqual(count_unreg, 0)
    
    def test_connection_status(self):
        """Test connection status tracking."""
        from mytrader.data.live_data_manager import LiveDataManager, LiveDataConfig
        
        async def run_test():
            config = LiveDataConfig(log_dir=self.temp_dir)
            manager = LiveDataManager(self.mock_ib, config)
            
            # Wait for connection
            connected = await manager.wait_for_connection(timeout=1.0)
            
            final_status = manager.get_connection_status()
            
            return connected, final_status.connected
        
        wait_result, final = asyncio.run(run_test())
        
        # Should successfully connect since mock_ib is already connected
        self.assertTrue(wait_result)
        self.assertTrue(final)
    
    def test_backpressure_handling(self):
        """Test backpressure when event queue is full."""
        from mytrader.data.live_data_manager import LiveDataManager, LiveDataConfig
        
        async def run_test():
            config = LiveDataConfig(
                log_dir=self.temp_dir,
                max_event_queue_size=10,
                backpressure_drop_count=5,
            )
            manager = LiveDataManager(self.mock_ib, config)
            
            # Fill the queue manually beyond capacity
            # The deque has maxlen so it auto-truncates
            for i in range(15):
                manager._event_queue.append({"type": "test", "data": i})
            
            return manager.get_event_queue_size()
        
        queue_size = asyncio.run(run_test())
        
        # Queue should be at max size (10) due to deque maxlen
        self.assertEqual(queue_size, 10)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestNormalizedDataStructures(unittest.TestCase):
    """Test normalized data structures."""
    
    def test_normalized_tick_to_dict(self):
        """Test NormalizedTick serialization."""
        from mytrader.data.live_data_manager import NormalizedTick, TickType
        
        tick = NormalizedTick(
            timestamp=datetime(2025, 12, 7, 10, 0, 0, tzinfo=timezone.utc),
            symbol="ES",
            bid=5000.0,
            ask=5001.0,
            last=5000.5,
            last_size=1,
            tick_type=TickType.LAST,
        )
        
        d = tick.to_dict()
        
        self.assertEqual(d["symbol"], "ES")
        self.assertEqual(d["bid"], 5000.0)
        self.assertEqual(d["tick_type"], "LAST")
        self.assertIn("timestamp", d)
    
    def test_normalized_quote(self):
        """Test NormalizedQuote mid and spread calculation."""
        from mytrader.data.live_data_manager import NormalizedQuote
        
        quote = NormalizedQuote(
            timestamp=datetime.now(timezone.utc),
            symbol="ES",
            bid=5000.0,
            ask=5001.0,
            bid_size=100,
            ask_size=50,
        )
        
        self.assertEqual(quote.mid, 5000.5)
        self.assertEqual(quote.spread, 1.0)
    
    def test_normalized_candle(self):
        """Test NormalizedCandle structure."""
        from mytrader.data.live_data_manager import NormalizedCandle
        
        candle = NormalizedCandle(
            timestamp=datetime.now(timezone.utc),
            symbol="ES",
            open=5000.0,
            high=5010.0,
            low=4995.0,
            close=5005.0,
            volume=1000,
            tick_count=50,
            interval_seconds=60,
        )
        
        self.assertEqual(candle.interval_seconds, 60)
        self.assertEqual(candle.high - candle.low, 15.0)


class TestIdempotencySignature(unittest.TestCase):
    """Test idempotency signature generation."""
    
    def test_signature_consistency(self):
        """Test that same inputs produce same signature."""
        from mytrader.execution.reconcile import ReconcileManager, ReconcileConfig
        from tests.mock_ib_server import MockIB
        
        config = ReconcileConfig()
        ib = MockIB()
        manager = ReconcileManager(ib, config)
        
        ts = datetime(2025, 12, 7, 10, 0, 5, tzinfo=timezone.utc)
        
        sig1 = manager.generate_submission_signature("ES", 1, 5000.0, "BUY", ts)
        sig2 = manager.generate_submission_signature("ES", 1, 5000.0, "BUY", ts)
        
        self.assertEqual(sig1, sig2, "Same inputs should produce same signature")
    
    def test_signature_differs_on_different_inputs(self):
        """Test that different inputs produce different signatures."""
        from mytrader.execution.reconcile import ReconcileManager, ReconcileConfig
        from tests.mock_ib_server import MockIB
        
        config = ReconcileConfig()
        ib = MockIB()
        manager = ReconcileManager(ib, config)
        
        ts = datetime(2025, 12, 7, 10, 0, 5, tzinfo=timezone.utc)
        
        sig1 = manager.generate_submission_signature("ES", 1, 5000.0, "BUY", ts)
        sig2 = manager.generate_submission_signature("ES", 2, 5000.0, "BUY", ts)  # Different qty
        sig3 = manager.generate_submission_signature("ES", 1, 5001.0, "BUY", ts)  # Different price
        sig4 = manager.generate_submission_signature("ES", 1, 5000.0, "SELL", ts)  # Different side
        
        self.assertNotEqual(sig1, sig2)
        self.assertNotEqual(sig1, sig3)
        self.assertNotEqual(sig1, sig4)
    
    def test_timestamp_rounding(self):
        """Test that timestamps are rounded to 10-second windows."""
        from mytrader.execution.reconcile import ReconcileManager, ReconcileConfig
        from tests.mock_ib_server import MockIB
        
        config = ReconcileConfig()
        ib = MockIB()
        manager = ReconcileManager(ib, config)
        
        # Two timestamps within same 10-second window
        ts1 = datetime(2025, 12, 7, 10, 0, 3, tzinfo=timezone.utc)
        ts2 = datetime(2025, 12, 7, 10, 0, 7, tzinfo=timezone.utc)
        
        sig1 = manager.generate_submission_signature("ES", 1, 5000.0, "BUY", ts1)
        sig2 = manager.generate_submission_signature("ES", 1, 5000.0, "BUY", ts2)
        
        self.assertEqual(sig1, sig2, "Timestamps in same 10s window should produce same signature")
        
        # Different window
        ts3 = datetime(2025, 12, 7, 10, 0, 15, tzinfo=timezone.utc)
        sig3 = manager.generate_submission_signature("ES", 1, 5000.0, "BUY", ts3)
        
        self.assertNotEqual(sig1, sig3, "Different 10s windows should produce different signatures")


if __name__ == "__main__":
    unittest.main()
