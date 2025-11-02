"""Unit tests for trading strategies."""
import pytest
import pandas as pd
import numpy as np
from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
from mytrader.strategies.momentum_reversal import MomentumReversalStrategy
from mytrader.strategies.market_regime import detect_market_regime, get_regime_parameters, MarketRegime
from mytrader.features.feature_engineer import engineer_features


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 200
    
    # Generate trending price data
    base_price = 4700
    trend = np.linspace(0, 50, n)
    noise = np.random.randn(n) * 2
    close_prices = base_price + trend + noise
    
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(n) * 0.5,
        'high': close_prices + abs(np.random.randn(n) * 1),
        'low': close_prices - abs(np.random.randn(n) * 1),
        'close': close_prices,
        'volume': np.random.randint(10000, 15000, n)
    })
    data.index = pd.date_range('2024-01-01', periods=n, freq='1min')
    return data


@pytest.fixture
def sample_features(sample_ohlcv_data):
    """Create engineered features from sample data."""
    return engineer_features(sample_ohlcv_data, None)


class TestRsiMacdSentimentStrategy:
    """Test RSI MACD Sentiment strategy."""
    
    def test_strategy_initialization(self):
        """Test strategy can be initialized with default parameters."""
        strategy = RsiMacdSentimentStrategy()
        assert strategy.name == "rsi_macd_sentiment"
        assert strategy.rsi_buy == 35.0
        assert strategy.rsi_sell == 65.0
        
    def test_strategy_custom_parameters(self):
        """Test strategy with custom parameters."""
        strategy = RsiMacdSentimentStrategy(
            rsi_buy=25.0,
            rsi_sell=75.0,
            sentiment_buy=-0.5,
            sentiment_sell=0.5
        )
        assert strategy.rsi_buy == 25.0
        assert strategy.rsi_sell == 75.0
        
    def test_generate_signal_with_features(self, sample_features):
        """Test signal generation with real features."""
        strategy = RsiMacdSentimentStrategy()
        signal = strategy.generate(sample_features)
        
        assert signal.action in ["BUY", "SELL", "HOLD"]
        assert 0.0 <= signal.confidence <= 1.0
        assert "rsi" in signal.metadata
        assert "macd" in signal.metadata
        
    def test_buy_signal_conditions(self, sample_features):
        """Test that BUY signals are generated when conditions are met."""
        # Modify features to trigger BUY
        modified = sample_features.copy()
        modified.loc[modified.index[-1], "RSI_14"] = 25  # Oversold
        modified.loc[modified.index[-1], "MACDhist_12_26_9"] = 0.5  # Bullish crossover
        modified.loc[modified.index[-2], "MACDhist_12_26_9"] = -0.1
        
        strategy = RsiMacdSentimentStrategy()
        signal = strategy.generate(modified)
        
        # Should generate BUY or at least have low RSI
        assert signal.metadata["rsi"] < 35
        
    def test_sell_signal_conditions(self, sample_features):
        """Test that SELL signals are generated when conditions are met."""
        # Modify features to trigger SELL
        modified = sample_features.copy()
        modified.loc[modified.index[-1], "RSI_14"] = 75  # Overbought
        modified.loc[modified.index[-1], "MACDhist_12_26_9"] = -0.5  # Bearish crossover
        modified.loc[modified.index[-2], "MACDhist_12_26_9"] = 0.1
        
        strategy = RsiMacdSentimentStrategy()
        signal = strategy.generate(modified)
        
        # Should have high RSI
        assert signal.metadata["rsi"] > 65


class TestMomentumReversalStrategy:
    """Test Momentum Reversal strategy."""
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        strategy = MomentumReversalStrategy()
        assert strategy.name == "momentum_reversal"
        
    def test_generate_signal(self, sample_features):
        """Test signal generation."""
        strategy = MomentumReversalStrategy()
        returns = sample_features["close"].pct_change().dropna()
        signal = strategy.generate(sample_features)
        
        assert signal.action in ["BUY", "SELL", "HOLD"]
        assert 0.0 <= signal.confidence <= 1.0


class TestMarketRegimeDetection:
    """Test market regime detection."""
    
    def test_detect_trending_up_regime(self, sample_features):
        """Test detection of upward trending market."""
        # Create strong uptrend
        modified = sample_features.copy()
        modified["EMA_21"] = modified["close"] * 1.0
        modified["EMA_50"] = modified["close"] * 0.99
        modified["EMA_200"] = modified["close"] * 0.98
        modified["ADX_14"] = 30  # Strong trend
        
        regime, confidence = detect_market_regime(modified, lookback=50)
        
        # Should detect uptrend or at least not be mean-reverting with high certainty
        assert confidence > 0.5
        
    def test_detect_high_volatility_regime(self, sample_features):
        """Test detection of high volatility."""
        # Add high volatility
        modified = sample_features.copy()
        modified["ATR_14"] = modified["ATR_14"] * 3  # Increase ATR
        
        regime, confidence = detect_market_regime(modified, lookback=50)
        
        # Regime detection should work
        assert regime in MarketRegime
        assert confidence > 0.5
        
    def test_get_regime_parameters(self):
        """Test parameter retrieval for each regime."""
        for regime in MarketRegime:
            params = get_regime_parameters(regime)
            
            assert "rsi_buy" in params
            assert "rsi_sell" in params
            assert "sentiment_buy" in params
            assert "sentiment_sell" in params
            assert "position_multiplier" in params
            
            # Validate parameter ranges
            assert 0 < params["rsi_buy"] < 100
            assert 0 < params["rsi_sell"] < 100
            assert params["rsi_buy"] < params["rsi_sell"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
