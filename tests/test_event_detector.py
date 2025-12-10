"""Tests for Event Detector.

Tests cover:
- Market open/close triggers
- Volatility spike detection
- News keyword detection
- Manual triggers
- Cooldown logic
- Payload construction
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

from mytrader.llm.event_detector import (
    EventDetector,
    EventPayload,
    create_event_detector,
)


@pytest.fixture
def event_detector():
    """Create a fresh event detector for each test."""
    return EventDetector(
        symbol="MES",
        volatility_spike_threshold=2.0,
        minutes_after_open=5,
        minutes_before_close=5,
        min_interval_seconds=60,
        cooldown_seconds=300,
    )


@pytest.fixture
def base_snapshot():
    """Create a base market snapshot."""
    return {
        "current_price": 5375.00,
        "price_change_pct": 0.01,
        "momentum": 0.02,
        "atr": 5.0,
        "volatility": 0.015,
        "rsi": 55.0,
        "vix": 18.5,
        "position": 0,
        "unrealized_pnl": 0.0,
        "news_headlines": [],
        "recent_prices": [5370, 5372, 5374, 5375],
    }


class TestEventDetectorInitialization:
    """Tests for EventDetector initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        detector = EventDetector()
        
        assert detector.symbol == "MES"
        assert detector.volatility_spike_threshold == 2.0
        assert detector.min_interval_seconds == 60
        assert detector.cooldown_seconds == 300
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        detector = EventDetector(
            symbol="ES",
            volatility_spike_threshold=3.0,
            minutes_after_open=10,
            cooldown_seconds=600,
        )
        
        assert detector.symbol == "ES"
        assert detector.volatility_spike_threshold == 3.0
        assert detector.minutes_after_open == 10
        assert detector.cooldown_seconds == 600
    
    def test_factory_function(self):
        """Test create_event_detector factory function."""
        config = {
            "volatility_spike_threshold": 2.5,
            "min_interval_seconds": 30,
        }
        
        detector = create_event_detector(symbol="ES", config=config)
        
        assert detector.symbol == "ES"
        assert detector.volatility_spike_threshold == 2.5
        assert detector.min_interval_seconds == 30


class TestManualTrigger:
    """Tests for manual trigger functionality."""
    
    def test_set_manual_trigger(self, event_detector, base_snapshot):
        """Test setting a manual trigger."""
        event_detector.set_manual_trigger(notes="Testing manual trigger")
        
        should_trigger, reason, payload = event_detector.should_call_bedrock(base_snapshot)
        
        assert should_trigger is True
        assert "manual" in reason.lower()
        assert payload is not None
        assert payload.trigger_type == "manual"
    
    def test_manual_trigger_clears_after_use(self, event_detector, base_snapshot):
        """Test that manual trigger clears after being used."""
        event_detector.set_manual_trigger(notes="Test")
        
        # First call should trigger
        should_trigger1, _, _ = event_detector.should_call_bedrock(base_snapshot)
        assert should_trigger1 is True
        
        # Wait for min interval
        event_detector._last_trigger_time = None
        
        # Second call should not trigger (manual was consumed)
        should_trigger2, _, _ = event_detector.should_call_bedrock(base_snapshot)
        assert should_trigger2 is False
    
    def test_manual_trigger_takes_precedence(self, event_detector, base_snapshot):
        """Test that manual trigger takes precedence over other triggers."""
        # Set up conditions that would trigger volatility spike
        event_detector._baseline_atr = 5.0
        event_detector._atr_history = [5.0] * 50
        base_snapshot["atr"] = 15.0  # 3x baseline
        
        # Set manual trigger
        event_detector.set_manual_trigger(notes="Manual override")
        
        should_trigger, reason, payload = event_detector.should_call_bedrock(base_snapshot)
        
        assert should_trigger is True
        assert payload.trigger_type == "manual"


class TestVolatilitySpike:
    """Tests for volatility spike detection."""
    
    def test_volatility_spike_detected(self, event_detector, base_snapshot):
        """Test volatility spike detection."""
        # Build up baseline ATR
        event_detector._atr_history = [5.0] * 50
        event_detector._baseline_atr = 5.0
        
        # Set high ATR (2.5x baseline)
        base_snapshot["atr"] = 12.5
        
        # Clear cooldown
        event_detector._last_trigger_time = None
        event_detector._last_trigger_type = {}
        
        should_trigger, reason, payload = event_detector.should_call_bedrock(base_snapshot)
        
        assert should_trigger is True
        assert "volatility" in reason.lower()
        assert payload.trigger_type == "volatility_spike"
    
    def test_no_spike_when_within_threshold(self, event_detector, base_snapshot):
        """Test no trigger when volatility is within threshold."""
        # Build up baseline ATR
        event_detector._atr_history = [5.0] * 50
        event_detector._baseline_atr = 5.0
        
        # Set ATR just below threshold (1.5x baseline)
        base_snapshot["atr"] = 7.5
        
        event_detector._last_trigger_time = None
        event_detector._last_trigger_type = {}
        
        should_trigger, reason, payload = event_detector.should_call_bedrock(base_snapshot)
        
        assert should_trigger is False
    
    def test_baseline_atr_calculation(self, event_detector, base_snapshot):
        """Test baseline ATR is calculated correctly."""
        # No baseline initially
        assert event_detector._baseline_atr is None
        
        # Add ATR values
        for atr_value in range(20):
            base_snapshot["atr"] = 5.0 + atr_value * 0.1
            event_detector._check_volatility_spike(base_snapshot)
        
        # Baseline should now be calculated
        assert event_detector._baseline_atr is not None
        assert 5.0 <= event_detector._baseline_atr <= 7.0


class TestNewsDetection:
    """Tests for news keyword detection."""
    
    def test_news_keyword_detected(self, event_detector, base_snapshot):
        """Test news keyword detection."""
        base_snapshot["news_headlines"] = [
            "Fed Chair Powell speaks on inflation outlook",
            "Markets await FOMC decision",
        ]
        
        event_detector._last_trigger_time = None
        event_detector._last_trigger_type = {}
        
        should_trigger, reason, payload = event_detector.should_call_bedrock(base_snapshot)
        
        assert should_trigger is True
        assert "news" in reason.lower()
        assert "inflation" in reason.lower() or "fomc" in reason.lower() or "powell" in reason.lower()
    
    def test_no_trigger_for_irrelevant_news(self, event_detector, base_snapshot):
        """Test no trigger for irrelevant news."""
        base_snapshot["news_headlines"] = [
            "Local weather forecast",
            "Sports update",
        ]
        
        event_detector._last_trigger_time = None
        event_detector._last_trigger_type = {}
        
        should_trigger, reason, payload = event_detector.should_call_bedrock(base_snapshot)
        
        # Should not trigger (no keywords match)
        assert should_trigger is False or payload.trigger_type != "news"
    
    def test_multiple_keywords_in_news(self, event_detector, base_snapshot):
        """Test detection of multiple keywords."""
        base_snapshot["news_headlines"] = [
            "CPI data shows inflation rising, Fed rate hike expected",
        ]
        
        event_detector._last_trigger_time = None
        event_detector._last_trigger_type = {}
        
        should_trigger, reason, payload = event_detector.should_call_bedrock(base_snapshot)
        
        assert should_trigger is True
        assert payload.trigger_type == "news"


class TestCooldownLogic:
    """Tests for cooldown and interval logic."""
    
    def test_min_interval_enforced(self, event_detector, base_snapshot):
        """Test minimum interval between triggers."""
        event_detector.set_manual_trigger(notes="First trigger")
        
        # First trigger
        should_trigger1, _, _ = event_detector.should_call_bedrock(base_snapshot)
        assert should_trigger1 is True
        
        # Immediate second call should be blocked by min interval
        event_detector.set_manual_trigger(notes="Second trigger")
        should_trigger2, _, _ = event_detector.should_call_bedrock(base_snapshot)
        assert should_trigger2 is False
    
    def test_trigger_type_cooldown(self, event_detector, base_snapshot):
        """Test cooldown between same trigger types."""
        # Trigger volatility spike
        event_detector._baseline_atr = 5.0
        event_detector._atr_history = [5.0] * 50
        base_snapshot["atr"] = 12.5
        
        event_detector._last_trigger_time = None
        event_detector._last_trigger_type = {}
        
        # First trigger
        should_trigger1, _, _ = event_detector.should_call_bedrock(base_snapshot)
        assert should_trigger1 is True
        
        # Clear min interval but not trigger-specific cooldown
        event_detector._last_trigger_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        
        # Second call should be blocked by trigger-specific cooldown
        should_trigger2, _, _ = event_detector.should_call_bedrock(base_snapshot)
        assert should_trigger2 is False
    
    def test_different_trigger_types_have_separate_cooldowns(self, event_detector, base_snapshot):
        """Test that different trigger types have separate cooldowns."""
        # Setup for volatility spike
        event_detector._baseline_atr = 5.0
        event_detector._atr_history = [5.0] * 50
        base_snapshot["atr"] = 12.5
        
        event_detector._last_trigger_time = None
        event_detector._last_trigger_type = {}
        
        # Trigger volatility spike
        should_trigger1, _, payload1 = event_detector.should_call_bedrock(base_snapshot)
        assert should_trigger1 is True
        assert payload1.trigger_type == "volatility_spike"
        
        # Clear min interval
        event_detector._last_trigger_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        
        # Manual trigger should work (different type)
        event_detector.set_manual_trigger(notes="Manual after volatility")
        should_trigger2, _, payload2 = event_detector.should_call_bedrock(base_snapshot)
        assert should_trigger2 is True
        assert payload2.trigger_type == "manual"


class TestEventPayload:
    """Tests for EventPayload construction."""
    
    def test_payload_contains_all_fields(self, event_detector, base_snapshot):
        """Test that payload contains all required fields."""
        event_detector.set_manual_trigger(notes="Test payload")
        
        should_trigger, reason, payload = event_detector.should_call_bedrock(base_snapshot)
        
        assert payload is not None
        assert payload.trigger_type == "manual"
        assert payload.symbol == "MES"
        assert payload.current_price == 5375.00
        assert payload.rsi == 55.0
        assert payload.atr == 5.0
        assert payload.position == 0
    
    def test_payload_to_dict(self, event_detector, base_snapshot):
        """Test payload serialization to dict."""
        event_detector.set_manual_trigger(notes="Test")
        
        _, _, payload = event_detector.should_call_bedrock(base_snapshot)
        
        payload_dict = payload.to_dict()
        
        assert isinstance(payload_dict, dict)
        assert "trigger_type" in payload_dict
        assert "timestamp" in payload_dict
        assert "current_price" in payload_dict
        assert "recent_prices" in payload_dict
    
    def test_payload_truncates_recent_prices(self, event_detector, base_snapshot):
        """Test that payload truncates recent prices to last 20."""
        base_snapshot["recent_prices"] = list(range(50))
        
        event_detector.set_manual_trigger(notes="Test")
        _, _, payload = event_detector.should_call_bedrock(base_snapshot)
        
        payload_dict = payload.to_dict()
        
        assert len(payload_dict["recent_prices"]) <= 20


class TestDailyStateReset:
    """Tests for daily state management."""
    
    def test_market_open_triggered_today_flag(self, event_detector, base_snapshot):
        """Test market open triggered flag."""
        assert event_detector._market_open_triggered_today is False
        
        # Simulate market open trigger (would need to mock time)
        event_detector._market_open_triggered_today = True
        
        # Check that it persists within same day
        event_detector._reset_daily_state()
        assert event_detector._market_open_triggered_today is True
    
    def test_daily_reset_on_new_day(self, event_detector, base_snapshot):
        """Test daily flags reset on new day."""
        event_detector._market_open_triggered_today = True
        event_detector._market_close_triggered_today = True
        event_detector._last_check_date = "2025-12-06"  # Yesterday
        
        # Reset should clear flags
        event_detector._reset_daily_state()
        
        assert event_detector._market_open_triggered_today is False
        assert event_detector._market_close_triggered_today is False


class TestGetStatus:
    """Tests for status retrieval."""
    
    def test_get_status_returns_all_fields(self, event_detector):
        """Test status contains all expected fields."""
        status = event_detector.get_status()
        
        assert "symbol" in status
        assert "volatility_threshold" in status
        assert "baseline_atr" in status
        assert "market_open_triggered" in status
        assert "market_close_triggered" in status
        assert "cooldown_seconds" in status
        assert "manual_trigger_pending" in status
    
    def test_status_reflects_current_state(self, event_detector):
        """Test status reflects current state."""
        event_detector.set_manual_trigger(notes="Test")
        event_detector._baseline_atr = 5.5
        
        status = event_detector.get_status()
        
        assert status["manual_trigger_pending"] is True
        assert status["baseline_atr"] == 5.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
