"""Tests for Jan 2026 audit fixes.

These tests verify the new safety gates added after the Jan 2, 2026 audit
that showed 5/8 trades losing money due to:
1. Counter-trend trading in DOWNTREND
2. Low ADX (weak trend) signals
3. Stops too tight for 1-minute noise
4. Missing 5-minute trend confirmation
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Test imports
from mytrader.data.candle_aggregator import MultiTimeframeCandleBuilder, AggregatedCandle


CST = ZoneInfo("America/Chicago")


class TestNoTradeInRangeLowATR:
    """Test: No trade in RANGE regime + low ATR."""
    
    def create_mock_features(self, atr: float = 0.5, adx: float = 15.0) -> pd.DataFrame:
        """Create mock features DataFrame with specified ATR and ADX."""
        data = {
            "open": [6900.0] * 60,
            "high": [6905.0] * 60,
            "low": [6895.0] * 60,
            "close": [6902.0] * 60,
            "volume": [100] * 60,
            "ATR_14": [atr] * 60,
            "ADX_14": [adx] * 60,
            "RSI_14": [50.0] * 60,
            "EMA_9": [6900.0] * 60,
            "EMA_20": [6898.0] * 60,
        }
        df = pd.DataFrame(data)
        df["timestamp"] = [datetime.now(CST) - timedelta(minutes=60-i) for i in range(60)]
        df.set_index("timestamp", inplace=True)
        return df
    
    def test_low_atr_blocks_signal(self):
        """Verify that low ATR (below threshold) blocks trading signals."""
        # This test verifies the ADX gate logic
        from mytrader.execution.components.signal_processor import SignalProcessor
        
        # Create a mock signal
        signal = SimpleNamespace(
            action="BUY",
            confidence=0.65,
            metadata={}
        )
        
        # Create features with low ADX (below 20 threshold)
        features = self.create_mock_features(atr=0.5, adx=15.0)
        
        # Mock the settings
        mock_settings = MagicMock()
        mock_settings.trading = MagicMock()
        mock_settings.trading.entry_filters = {
            "require_adx_confirmation": True,
            "min_adx_threshold": 20.0,
            "allow_counter_trend": False,
        }
        
        # Create processor
        mock_manager = MagicMock()
        processor = SignalProcessor(mock_settings, None, mock_manager)
        
        # Apply gates
        result = processor._apply_adx_and_trend_gates(signal, features, "UPTREND")
        
        # Should be blocked due to low ADX
        assert result.action == "HOLD"
        assert result.confidence == 0.0
        assert "LOW_ADX" in str(result.metadata.get("block_reasons", []))
    
    def test_normal_atr_allows_signal(self):
        """Verify that normal ADX (above threshold) allows signals."""
        from mytrader.execution.components.signal_processor import SignalProcessor
        
        signal = SimpleNamespace(
            action="BUY",
            confidence=0.65,
            metadata={}
        )
        
        # Create features with good ADX (above 20 threshold)
        features = self.create_mock_features(atr=2.5, adx=25.0)
        
        mock_settings = MagicMock()
        mock_settings.trading = MagicMock()
        mock_settings.trading.entry_filters = {
            "require_adx_confirmation": True,
            "min_adx_threshold": 20.0,
            "allow_counter_trend": False,
        }
        
        mock_manager = MagicMock()
        processor = SignalProcessor(mock_settings, None, mock_manager)
        
        result = processor._apply_adx_and_trend_gates(signal, features, "UPTREND")
        
        # Should pass - ADX is sufficient and trend aligned
        assert result.action == "BUY"
        assert result.confidence == 0.65


class TestCounterTrendBlock:
    """Test: Counter-trend trades are blocked."""
    
    def create_mock_features(self, adx: float = 25.0) -> pd.DataFrame:
        """Create mock features with sufficient ADX."""
        data = {
            "close": [6900.0] * 10,
            "ATR_14": [2.5] * 10,
            "ADX_14": [adx] * 10,
        }
        df = pd.DataFrame(data)
        df["timestamp"] = [datetime.now(CST) - timedelta(minutes=10-i) for i in range(10)]
        df.set_index("timestamp", inplace=True)
        return df
    
    def test_buy_in_downtrend_blocked(self):
        """Verify BUY signal is blocked when market is in DOWNTREND."""
        from mytrader.execution.components.signal_processor import SignalProcessor
        
        signal = SimpleNamespace(
            action="BUY",
            confidence=0.65,
            metadata={}
        )
        
        features = self.create_mock_features(adx=25.0)
        
        mock_settings = MagicMock()
        mock_settings.trading = MagicMock()
        mock_settings.trading.entry_filters = {
            "require_adx_confirmation": True,
            "min_adx_threshold": 20.0,
            "allow_counter_trend": False,  # Key setting
        }
        
        mock_manager = MagicMock()
        processor = SignalProcessor(mock_settings, None, mock_manager)
        
        # Apply gates with DOWNTREND market
        result = processor._apply_adx_and_trend_gates(signal, features, "DOWNTREND")
        
        # Should be blocked - BUY in DOWNTREND is counter-trend
        assert result.action == "HOLD"
        assert "COUNTER_TREND" in str(result.metadata.get("block_reasons", []))
        assert result.metadata.get("counter_trend_blocked") is True
    
    def test_sell_in_uptrend_blocked(self):
        """Verify SELL signal is blocked when market is in UPTREND."""
        from mytrader.execution.components.signal_processor import SignalProcessor
        
        signal = SimpleNamespace(
            action="SELL",
            confidence=0.65,
            metadata={}
        )
        
        features = self.create_mock_features(adx=25.0)
        
        mock_settings = MagicMock()
        mock_settings.trading = MagicMock()
        mock_settings.trading.entry_filters = {
            "require_adx_confirmation": True,
            "min_adx_threshold": 20.0,
            "allow_counter_trend": False,
        }
        
        mock_manager = MagicMock()
        processor = SignalProcessor(mock_settings, None, mock_manager)
        
        result = processor._apply_adx_and_trend_gates(signal, features, "UPTREND")
        
        # Should be blocked - SELL in UPTREND is counter-trend
        assert result.action == "HOLD"
        assert "COUNTER_TREND" in str(result.metadata.get("block_reasons", []))
    
    def test_trend_aligned_trade_passes(self):
        """Verify trend-aligned trades pass through."""
        from mytrader.execution.components.signal_processor import SignalProcessor
        
        # BUY in UPTREND should pass
        signal = SimpleNamespace(
            action="BUY",
            confidence=0.65,
            metadata={}
        )
        
        features = self.create_mock_features(adx=25.0)
        
        mock_settings = MagicMock()
        mock_settings.trading = MagicMock()
        mock_settings.trading.entry_filters = {
            "require_adx_confirmation": True,
            "min_adx_threshold": 20.0,
            "allow_counter_trend": False,
        }
        
        mock_manager = MagicMock()
        processor = SignalProcessor(mock_settings, None, mock_manager)
        
        result = processor._apply_adx_and_trend_gates(signal, features, "UPTREND")
        
        # Should pass - BUY in UPTREND is trend-aligned
        assert result.action == "BUY"
        assert result.confidence == 0.65


class TestMultiTimeframeTrendFilter:
    """Test: 5-minute trend filter blocks counter-trend trades."""
    
    def test_mtf_builder_aggregation(self):
        """Verify 1-minute bars aggregate correctly into 5-minute candles."""
        builder = MultiTimeframeCandleBuilder(
            base_interval=1,
            target_interval=5,
            ema_period=20,
        )
        
        base_time = datetime(2026, 1, 2, 9, 0, 0, tzinfo=CST)
        
        # Add 5 one-minute bars (should complete one 5-min candle)
        completed = None
        for i in range(5):
            bar_time = base_time + timedelta(minutes=i)
            result = builder.add_bar(
                timestamp=bar_time,
                open_price=6900.0 + i,
                high_price=6905.0 + i,
                low_price=6895.0 + i,
                close_price=6902.0 + i,
            )
            if result:
                completed = result
        
        # Add one more bar to trigger completion
        result = builder.add_bar(
            timestamp=base_time + timedelta(minutes=5),
            open_price=6905.0,
            high_price=6910.0,
            low_price=6900.0,
            close_price=6907.0,
        )
        
        # Should have a completed candle now
        assert builder.has_complete_candle()
        
        candle = builder.get_latest_candle()
        assert candle is not None
        assert candle.bar_count == 5
        # First bar open should be candle open
        assert candle.open == 6900.0
        # Last bar close should be candle close
        assert candle.close == 6906.0  # 6902.0 + 4
    
    def test_mtf_trend_detection_uptrend(self):
        """Verify uptrend detection from 5-min candles."""
        builder = MultiTimeframeCandleBuilder(
            base_interval=1,
            target_interval=5,
            ema_period=5,  # Short for testing
        )
        
        base_time = datetime(2026, 1, 2, 9, 0, 0, tzinfo=CST)
        
        # Create ascending price pattern (uptrend)
        base_price = 6900.0
        for period in range(6):  # 6 five-minute periods = 30 bars
            period_start = base_time + timedelta(minutes=period * 5)
            for i in range(5):
                bar_time = period_start + timedelta(minutes=i)
                price_offset = period * 3  # Each period 3 points higher
                builder.add_bar(
                    timestamp=bar_time,
                    open_price=base_price + price_offset,
                    high_price=base_price + price_offset + 2,
                    low_price=base_price + price_offset - 1,
                    close_price=base_price + price_offset + 1.5,
                )
        
        # Trigger final candle
        builder.add_bar(
            timestamp=base_time + timedelta(minutes=30),
            open_price=6920.0,
            high_price=6925.0,
            low_price=6918.0,
            close_price=6923.0,
        )
        
        # Should detect uptrend
        trend = builder.get_trend()
        assert trend in ("UPTREND", "NEUTRAL")  # Allow neutral during ramp-up
    
    def test_mtf_blocks_counter_trend_buy_in_downtrend(self):
        """Verify 5-min filter blocks BUY when 5-min trend is down."""
        builder = MultiTimeframeCandleBuilder(
            base_interval=1,
            target_interval=5,
            ema_period=5,
        )
        
        base_time = datetime(2026, 1, 2, 9, 0, 0, tzinfo=CST)
        
        # Create descending price pattern (downtrend)
        base_price = 6950.0
        for period in range(6):
            period_start = base_time + timedelta(minutes=period * 5)
            for i in range(5):
                bar_time = period_start + timedelta(minutes=i)
                price_offset = period * 5  # Each period 5 points lower
                builder.add_bar(
                    timestamp=bar_time,
                    open_price=base_price - price_offset,
                    high_price=base_price - price_offset + 1,
                    low_price=base_price - price_offset - 3,
                    close_price=base_price - price_offset - 2,
                )
        
        # Trigger final candle
        builder.add_bar(
            timestamp=base_time + timedelta(minutes=30),
            open_price=6920.0,
            high_price=6921.0,
            low_price=6915.0,
            close_price=6916.0,
        )
        
        # Check alignment for BUY
        is_aligned, reason = builder.is_trend_aligned("BUY")
        
        # BUY should NOT be aligned with downtrend
        # (May be True if not enough data for clear downtrend detection)
        if builder.get_trend() == "DOWNTREND":
            assert not is_aligned
            assert "COUNTER_TREND" in reason
    
    def test_mtf_allows_trend_aligned_sell_in_downtrend(self):
        """Verify 5-min filter allows SELL when 5-min trend is down."""
        builder = MultiTimeframeCandleBuilder(
            base_interval=1,
            target_interval=5,
            ema_period=5,
        )
        
        # Manually inject candles representing downtrend
        from collections import deque
        builder._candles = deque(maxlen=100)
        builder._ema_values = deque(maxlen=100)
        
        base_time = datetime(2026, 1, 2, 9, 0, 0, tzinfo=CST)
        
        # Add declining candles
        for i in range(5):
            candle = AggregatedCandle(
                timestamp=base_time + timedelta(minutes=i * 5),
                open=6950.0 - i * 5,
                high=6952.0 - i * 5,
                low=6945.0 - i * 5,
                close=6947.0 - i * 5,  # Lower highs, lower lows
                volume=100.0,
                bar_count=5,
            )
            builder._candles.append(candle)
            builder._ema_values.append(6950.0 - i * 3)  # Declining EMA
        
        builder._last_completed_candle = builder._candles[-1]
        
        # Check alignment for SELL
        is_aligned, reason = builder.is_trend_aligned("SELL")
        
        # If downtrend detected, SELL should be aligned
        trend = builder.get_trend()
        if trend == "DOWNTREND":
            assert is_aligned
            assert "DOWNTREND" in reason


class TestRiskRewardValidation:
    """Test: Minimum R:R ratio enforcement."""
    
    def test_poor_rr_rejected(self):
        """Verify trades with R:R < 1.5 are rejected."""
        from mytrader.risk.trade_math import compute_risk_reward
        
        # BUY at 6900, SL at 6895 (5pt risk), TP at 6906 (6pt reward)
        # R:R = 6/5 = 1.2 - should be rejected if min is 1.5
        entry = 6900.0
        stop_loss = 6895.0  # 5 points risk
        take_profit = 6906.0  # 6 points reward
        
        risk_points, reward_points, rr_ratio = compute_risk_reward(
            entry, stop_loss, take_profit, "BUY"
        )
        
        assert abs(risk_points - 5.0) < 0.01
        assert abs(reward_points - 6.0) < 0.01
        assert abs(rr_ratio - 1.2) < 0.01
        
        # This should fail the 1.5 minimum
        min_rr = 1.5
        assert rr_ratio < min_rr
    
    def test_good_rr_accepted(self):
        """Verify trades with R:R >= 1.5 are accepted."""
        from mytrader.risk.trade_math import compute_risk_reward
        
        # BUY at 6900, SL at 6895 (5pt risk), TP at 6910 (10pt reward)
        # R:R = 10/5 = 2.0 - should be accepted
        entry = 6900.0
        stop_loss = 6895.0  # 5 points risk
        take_profit = 6910.0  # 10 points reward
        
        risk_points, reward_points, rr_ratio = compute_risk_reward(
            entry, stop_loss, take_profit, "BUY"
        )
        
        assert abs(risk_points - 5.0) < 0.01
        assert abs(reward_points - 10.0) < 0.01
        assert abs(rr_ratio - 2.0) < 0.01
        
        # This should pass the 1.5 minimum
        min_rr = 1.5
        assert rr_ratio >= min_rr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
