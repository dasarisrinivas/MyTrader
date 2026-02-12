"""Unit tests for VX futures feed module."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestVxConfig:
    """Tests for VxConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        from shree.data.vx_futures_feed import VxConfig
        
        config = VxConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 7497
        assert config.client_id == 71
        assert config.market_data_type == 1
        assert config.stale_seconds == 120
        assert config.conservative_on_stale is False
        assert config.extreme_threshold == 30.0
        assert config.elevated_threshold == 20.0
        assert config.max_retries == 5
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
    
    def test_custom_values(self):
        """Test custom configuration values."""
        from shree.data.vx_futures_feed import VxConfig
        
        config = VxConfig(
            host="192.168.1.1",
            port=7496,
            client_id=100,
            extreme_threshold=35.0,
            elevated_threshold=25.0,
        )
        assert config.host == "192.168.1.1"
        assert config.port == 7496
        assert config.client_id == 100
        assert config.extreme_threshold == 35.0
        assert config.elevated_threshold == 25.0


class TestVxState:
    """Tests for VxState dataclass."""
    
    def test_default_state(self):
        """Test default state values."""
        from shree.data.vx_futures_feed import VxState
        
        state = VxState()
        assert state.price is None
        assert state.last_update is None
        assert state.contract_symbol is None
        assert state.error_count == 0
    
    def test_state_with_values(self):
        """Test state with values."""
        from shree.data.vx_futures_feed import VxState
        
        now = datetime.now()
        state = VxState(
            price=25.50,
            last_update=now,
            contract_symbol="VXH5",
            error_count=2,
        )
        assert state.price == 25.50
        assert state.last_update == now
        assert state.contract_symbol == "VXH5"
        assert state.error_count == 2


class TestVxFuturesFeedInit:
    """Tests for VxFuturesFeed initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        from shree.data.vx_futures_feed import VxFuturesFeed
        
        feed = VxFuturesFeed()
        assert feed.config.host == "127.0.0.1"
        assert feed.config.port == 7497
        assert feed.config.client_id == 71
        assert feed._running is False
        assert feed._thread is None
        assert feed._front_contract is None
        assert feed._ticker is None
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        from shree.data.vx_futures_feed import VxFuturesFeed
        
        feed = VxFuturesFeed(
            host="localhost",
            port=7496,
            client_id=99,
            extreme_threshold=32.0,
            elevated_threshold=22.0,
        )
        assert feed.config.host == "localhost"
        assert feed.config.port == 7496
        assert feed.config.client_id == 99
        assert feed.config.extreme_threshold == 32.0
        assert feed.config.elevated_threshold == 22.0


class TestVolatilityMultiplier:
    """Tests for volatility multiplier calculation."""
    
    @pytest.fixture
    def feed(self):
        """Create a VxFuturesFeed instance for testing."""
        from shree.data.vx_futures_feed import VxFuturesFeed
        return VxFuturesFeed()
    
    def test_extreme_vix_multiplier(self, feed):
        """Test multiplier when VIX is at extreme levels (>= 30)."""
        # VX >= 30 should return 0.4
        multiplier = feed._calculate_multiplier(35.0, is_stale=False)
        assert multiplier == 0.4
        
        multiplier = feed._calculate_multiplier(30.0, is_stale=False)
        assert multiplier == 0.4
    
    def test_elevated_vix_multiplier(self, feed):
        """Test multiplier when VIX is elevated (>= 20, < 30)."""
        # VX >= 20 and < 30 should return 0.7
        multiplier = feed._calculate_multiplier(25.0, is_stale=False)
        assert multiplier == 0.7
        
        multiplier = feed._calculate_multiplier(20.0, is_stale=False)
        assert multiplier == 0.7
        
        multiplier = feed._calculate_multiplier(29.99, is_stale=False)
        assert multiplier == 0.7
    
    def test_normal_vix_multiplier(self, feed):
        """Test multiplier when VIX is normal (< 20)."""
        # VX < 20 should return 1.0
        multiplier = feed._calculate_multiplier(15.0, is_stale=False)
        assert multiplier == 1.0
        
        multiplier = feed._calculate_multiplier(19.99, is_stale=False)
        assert multiplier == 1.0
        
        multiplier = feed._calculate_multiplier(12.0, is_stale=False)
        assert multiplier == 1.0
    
    def test_none_price_returns_neutral(self, feed):
        """Test that None price returns neutral multiplier."""
        multiplier = feed._calculate_multiplier(None, is_stale=False)
        assert multiplier == 1.0
    
    def test_zero_price_returns_neutral(self, feed):
        """Test that zero price returns neutral multiplier."""
        multiplier = feed._calculate_multiplier(0.0, is_stale=False)
        assert multiplier == 1.0
        
        multiplier = feed._calculate_multiplier(-5.0, is_stale=False)
        assert multiplier == 1.0
    
    def test_stale_with_conservative_mode(self):
        """Test that stale data with conservative mode returns 0.5."""
        from shree.data.vx_futures_feed import VxFuturesFeed
        
        feed = VxFuturesFeed(conservative_on_stale=True)
        multiplier = feed._calculate_multiplier(15.0, is_stale=True)
        assert multiplier == 0.5
    
    def test_stale_without_conservative_mode(self, feed):
        """Test that stale data without conservative mode uses price."""
        # Default conservative_on_stale is False
        multiplier = feed._calculate_multiplier(15.0, is_stale=True)
        assert multiplier == 1.0  # Normal price, no conservative penalty
        
        multiplier = feed._calculate_multiplier(25.0, is_stale=True)
        assert multiplier == 0.7  # Elevated price still applied
    
    def test_custom_thresholds(self):
        """Test custom threshold values."""
        from shree.data.vx_futures_feed import VxFuturesFeed
        
        feed = VxFuturesFeed(
            extreme_threshold=40.0,
            elevated_threshold=25.0,
        )
        
        # 35 is now below extreme (40) but above elevated (25)
        multiplier = feed._calculate_multiplier(35.0, is_stale=False)
        assert multiplier == 0.7
        
        # 22 is now below elevated (25)
        multiplier = feed._calculate_multiplier(22.0, is_stale=False)
        assert multiplier == 1.0


class TestStaleDetection:
    """Tests for stale data detection."""
    
    def test_no_update_is_stale(self):
        """Test that no update means stale."""
        from shree.data.vx_futures_feed import VxFuturesFeed
        
        feed = VxFuturesFeed()
        # No update has been received
        assert feed.is_stale() is True
    
    def test_recent_update_not_stale(self):
        """Test that recent update is not stale."""
        from shree.data.vx_futures_feed import VxFuturesFeed
        
        feed = VxFuturesFeed(stale_seconds=120)
        # Simulate recent update
        feed._state.last_update = datetime.now()
        assert feed.is_stale() is False
    
    def test_old_update_is_stale(self):
        """Test that old update is stale."""
        from shree.data.vx_futures_feed import VxFuturesFeed
        
        feed = VxFuturesFeed(stale_seconds=120)
        # Simulate old update (3 minutes ago)
        feed._state.last_update = datetime.now() - timedelta(seconds=180)
        assert feed.is_stale() is True
    
    def test_custom_stale_threshold(self):
        """Test custom stale threshold."""
        from shree.data.vx_futures_feed import VxFuturesFeed
        
        feed = VxFuturesFeed(stale_seconds=60)
        # 45 seconds ago - should not be stale with 60s threshold
        feed._state.last_update = datetime.now() - timedelta(seconds=45)
        assert feed.is_stale() is False
        
        # 90 seconds ago - should be stale with 60s threshold
        feed._state.last_update = datetime.now() - timedelta(seconds=90)
        assert feed.is_stale() is True


class TestGetLatest:
    """Tests for get_latest method."""
    
    def test_get_latest_with_data(self):
        """Test get_latest returns correct data."""
        from shree.data.vx_futures_feed import VxFuturesFeed
        
        feed = VxFuturesFeed()
        now = datetime.now()
        feed._state.price = 22.50
        feed._state.last_update = now
        feed._state.contract_symbol = "VXH5"
        
        result = feed.get_latest()
        assert result["price"] == 22.50
        assert result["contract"] == "VXH5"
        assert result["is_stale"] is False
        assert result["multiplier"] == 0.7  # 22.50 >= 20
        assert result["error_count"] == 0
    
    def test_get_latest_no_data(self):
        """Test get_latest with no data."""
        from shree.data.vx_futures_feed import VxFuturesFeed
        
        feed = VxFuturesFeed()
        
        result = feed.get_latest()
        assert result["price"] is None
        assert result["last_update"] is None
        assert result["contract"] is None
        assert result["is_stale"] is True
        assert result["multiplier"] == 1.0  # No price -> neutral


class TestGetVxPrice:
    """Tests for get_vx_price method."""
    
    def test_get_price_with_value(self):
        """Test get_vx_price with a value."""
        from shree.data.vx_futures_feed import VxFuturesFeed
        
        feed = VxFuturesFeed()
        feed._state.price = 18.75
        
        assert feed.get_vx_price() == 18.75
    
    def test_get_price_without_value(self):
        """Test get_vx_price with no value."""
        from shree.data.vx_futures_feed import VxFuturesFeed
        
        feed = VxFuturesFeed()
        assert feed.get_vx_price() is None


class TestPriceFromTicker:
    """Tests for _get_price_from_ticker method."""
    
    def test_no_ticker_returns_none(self):
        """Test that no ticker returns None."""
        from shree.data.vx_futures_feed import VxFuturesFeed
        
        feed = VxFuturesFeed()
        feed._ticker = None
        assert feed._get_price_from_ticker() is None
    
    def test_last_price(self):
        """Test extraction of last price."""
        from shree.data.vx_futures_feed import VxFuturesFeed
        
        feed = VxFuturesFeed()
        ticker = MagicMock()
        ticker.marketPrice.return_value = float('nan')  # Invalid
        ticker.last = 21.50
        ticker.bid = None
        ticker.ask = None
        feed._ticker = ticker
        
        assert feed._get_price_from_ticker() == 21.50
    
    def test_midpoint_fallback(self):
        """Test fallback to bid/ask midpoint."""
        from shree.data.vx_futures_feed import VxFuturesFeed
        
        feed = VxFuturesFeed()
        ticker = MagicMock()
        ticker.marketPrice.return_value = float('nan')
        ticker.last = None
        ticker.bid = 20.0
        ticker.ask = 22.0
        feed._ticker = ticker
        
        assert feed._get_price_from_ticker() == 21.0  # midpoint
    
    def test_bid_only_fallback(self):
        """Test fallback to bid only."""
        from shree.data.vx_futures_feed import VxFuturesFeed
        
        feed = VxFuturesFeed()
        ticker = MagicMock()
        ticker.marketPrice.return_value = float('nan')
        ticker.last = None
        ticker.bid = 20.0
        ticker.ask = None
        feed._ticker = ticker
        
        assert feed._get_price_from_ticker() == 20.0


class TestModuleLevelFunctions:
    """Tests for module-level singleton functions."""
    
    def test_get_vx_feed_returns_none_initially(self):
        """Test get_vx_feed returns None before init."""
        from shree.data.vx_futures_feed import get_vx_feed, shutdown_vx_feed
        
        # Ensure clean state
        shutdown_vx_feed()
        assert get_vx_feed() is None
    
    def test_init_and_get_vx_feed(self):
        """Test init_vx_feed creates singleton."""
        from shree.data.vx_futures_feed import (
            init_vx_feed,
            get_vx_feed,
            shutdown_vx_feed,
        )
        
        # Initialize
        feed = init_vx_feed(host="127.0.0.1", port=7497, client_id=99)
        
        # Get should return the same instance
        assert get_vx_feed() is feed
        assert feed.config.client_id == 99
        
        # Cleanup
        shutdown_vx_feed()
        assert get_vx_feed() is None


class TestThreadSafety:
    """Tests for thread-safety of state access."""
    
    def test_concurrent_state_access(self):
        """Test that concurrent access to state is safe."""
        from shree.data.vx_futures_feed import VxFuturesFeed
        
        feed = VxFuturesFeed()
        errors: List[Exception] = []
        
        def reader():
            for _ in range(100):
                try:
                    _ = feed.get_vx_price()
                    _ = feed.is_stale()
                    _ = feed.get_volatility_multiplier()
                except Exception as e:
                    errors.append(e)
        
        def writer():
            for i in range(100):
                try:
                    with feed._lock:
                        feed._state.price = 15.0 + (i % 20)
                        feed._state.last_update = datetime.now()
                except Exception as e:
                    errors.append(e)
        
        threads = [
            threading.Thread(target=reader) for _ in range(3)
        ] + [
            threading.Thread(target=writer) for _ in range(2)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread errors: {errors}"


class TestVolatilityMultiplierIntegration:
    """Integration tests for get_volatility_multiplier."""
    
    def test_full_multiplier_flow(self):
        """Test the full flow from state update to multiplier."""
        from shree.data.vx_futures_feed import VxFuturesFeed
        
        feed = VxFuturesFeed()
        
        # Initially no data -> 1.0
        assert feed.get_volatility_multiplier() == 1.0
        
        # Set normal VIX -> 1.0
        feed._state.price = 15.0
        feed._state.last_update = datetime.now()
        assert feed.get_volatility_multiplier() == 1.0
        
        # Set elevated VIX -> 0.7
        feed._state.price = 25.0
        feed._state.last_update = datetime.now()
        assert feed.get_volatility_multiplier() == 0.7
        
        # Set extreme VIX -> 0.4
        feed._state.price = 35.0
        feed._state.last_update = datetime.now()
        assert feed.get_volatility_multiplier() == 0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
