"""Unit tests for LLM integration modules."""
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd
import numpy as np

from mytrader.llm.data_schema import (
    TradingContext,
    TradeRecommendation,
    TradeOutcome,
)
from mytrader.llm.bedrock_client import BedrockClient
from mytrader.llm.trade_advisor import TradeAdvisor
from mytrader.llm.trade_logger import TradeLogger
from mytrader.llm.sentiment_aggregator import SentimentAggregator
from mytrader.strategies.base import Signal
from mytrader.strategies.llm_enhanced_strategy import LLMEnhancedStrategy


class TestDataSchema:
    """Test data schema classes."""
    
    def test_trading_context_creation(self):
        """Test TradingContext creation and serialization."""
        context = TradingContext(
            symbol="ES",
            current_price=4950.0,
            timestamp=datetime(2024, 1, 1, 12, 0),
            rsi=35.0,
            macd=0.5,
            macd_signal=0.3,
            macd_hist=0.2,
            atr=10.0,
            sentiment_score=0.2,
        )
        
        assert context.symbol == "ES"
        assert context.current_price == 4950.0
        assert context.rsi == 35.0
        
        # Test serialization
        data_dict = context.to_dict()
        assert isinstance(data_dict, dict)
        assert data_dict["symbol"] == "ES"
        assert data_dict["technical_indicators"]["rsi"] == 35.0
    
    def test_trade_recommendation_creation(self):
        """Test TradeRecommendation creation and serialization."""
        rec = TradeRecommendation(
            trade_decision="BUY",
            confidence=0.85,
            suggested_position_size=2,
            reasoning="Strong oversold signal with positive sentiment",
            key_factors=["RSI < 30", "Positive MACD", "Bullish sentiment"],
            model_name="claude-3-sonnet",
        )
        
        assert rec.trade_decision == "BUY"
        assert rec.confidence == 0.85
        assert len(rec.key_factors) == 3
        
        # Test serialization
        data_dict = rec.to_dict()
        assert isinstance(data_dict, dict)
        assert data_dict["trade_decision"] == "BUY"
        
        # Test deserialization
        rec2 = TradeRecommendation.from_dict(data_dict)
        assert rec2.trade_decision == rec.trade_decision
        assert rec2.confidence == rec.confidence
    
    def test_trade_outcome_creation(self):
        """Test TradeOutcome creation."""
        outcome = TradeOutcome(
            order_id=12345,
            symbol="ES",
            timestamp=datetime.utcnow(),
            action="BUY",
            quantity=2,
            entry_price=4950.0,
            exit_price=4960.0,
            realized_pnl=500.0,
            outcome="WIN",
        )
        
        assert outcome.order_id == 12345
        assert outcome.action == "BUY"
        assert outcome.realized_pnl == 500.0
        assert outcome.outcome == "WIN"
        
        # Test serialization
        data_dict = outcome.to_dict()
        assert isinstance(data_dict, dict)
        assert data_dict["realized_pnl"] == 500.0


class TestBedrockClient:
    """Test AWS Bedrock client (mocked)."""
    
    @patch("mytrader.llm.bedrock_client.boto3")
    def test_bedrock_client_initialization(self, mock_boto3):
        """Test Bedrock client initialization."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        
        client = BedrockClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
        
        assert client.model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert client.region_name == "us-east-1"
        mock_boto3.client.assert_called_once()
    
    @patch("mytrader.llm.bedrock_client.boto3")
    def test_get_trade_recommendation(self, mock_boto3):
        """Test getting trade recommendation from LLM."""
        # Mock AWS response
        mock_client = MagicMock()
        mock_response = {
            "body": MagicMock()
        }
        mock_response["body"].read.return_value = json.dumps({
            "content": [{
                "text": json.dumps({
                    "trade_decision": "BUY",
                    "confidence": 0.85,
                    "suggested_position_size": 2,
                    "suggested_stop_loss": 4945.0,
                    "suggested_take_profit": 4960.0,
                    "reasoning": "Strong oversold signal",
                    "key_factors": ["RSI < 30", "Positive MACD"],
                    "risk_assessment": "Low risk entry"
                })
            }]
        }).encode()
        
        mock_client.invoke_model.return_value = mock_response
        mock_boto3.client.return_value = mock_client
        
        client = BedrockClient()
        
        context = TradingContext(
            symbol="ES",
            current_price=4950.0,
            timestamp=datetime.utcnow(),
            rsi=28.0,
            macd=0.5,
            macd_signal=0.3,
            macd_hist=0.2,
            atr=10.0,
            sentiment_score=0.3,
        )
        
        recommendation = client.get_trade_recommendation(context)
        
        assert recommendation is not None
        assert recommendation.trade_decision == "BUY"
        assert recommendation.confidence == 0.85
        assert recommendation.suggested_position_size == 2


class TestTradeAdvisor:
    """Test trade advisor."""
    
    def test_trade_advisor_initialization_without_llm(self):
        """Test advisor initialization without LLM."""
        advisor = TradeAdvisor(enable_llm=False)
        
        assert advisor.enable_llm is False
        assert advisor.bedrock_client is None
    
    def test_enhance_signal_without_llm(self):
        """Test signal enhancement with LLM disabled."""
        advisor = TradeAdvisor(enable_llm=False)
        
        original_signal = Signal(
            action="BUY",
            confidence=0.75,
            metadata={"rsi": 30}
        )
        
        context = TradingContext(
            symbol="ES",
            current_price=4950.0,
            timestamp=datetime.utcnow(),
            rsi=30.0,
            macd=0.5,
            macd_signal=0.3,
            macd_hist=0.2,
            atr=10.0,
        )
        
        enhanced_signal, llm_rec = advisor.enhance_signal(original_signal, context)
        
        # Should return original signal unchanged
        assert enhanced_signal.action == "BUY"
        assert enhanced_signal.confidence == 0.75
        assert llm_rec is None
    
    @patch("mytrader.llm.trade_advisor.BedrockClient")
    def test_enhance_signal_consensus_mode(self, mock_bedrock):
        """Test signal enhancement in consensus mode."""
        # Mock LLM recommendation
        mock_client = MagicMock()
        mock_llm_rec = TradeRecommendation(
            trade_decision="BUY",
            confidence=0.80,
            reasoning="Agree with buy signal",
        )
        mock_client.get_trade_recommendation.return_value = mock_llm_rec
        mock_bedrock.return_value = mock_client
        
        advisor = TradeAdvisor(
            bedrock_client=mock_client,
            enable_llm=True,
            llm_override_mode=False,
            min_confidence_threshold=0.7,
        )
        
        original_signal = Signal(
            action="BUY",
            confidence=0.75,
            metadata={}
        )
        
        context = TradingContext(
            symbol="ES",
            current_price=4950.0,
            timestamp=datetime.utcnow(),
            rsi=30.0,
            macd=0.5,
            macd_signal=0.3,
            macd_hist=0.2,
            atr=10.0,
        )
        
        enhanced_signal, llm_rec = advisor.enhance_signal(original_signal, context)
        
        # Signals agree - confidence should be boosted
        assert enhanced_signal.action == "BUY"
        assert enhanced_signal.confidence > 0.75  # Boosted
        assert llm_rec is not None
        assert enhanced_signal.metadata["consensus"] is True


class TestTradeLogger:
    """Test trade logger."""
    
    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create temporary database for testing."""
        db_path = tmp_path / "test_trades.db"
        return db_path
    
    def test_trade_logger_initialization(self, temp_db):
        """Test trade logger initialization."""
        logger = TradeLogger(db_path=temp_db)
        
        assert logger.db_path.exists()
    
    def test_log_trade_entry(self, temp_db):
        """Test logging trade entry."""
        logger = TradeLogger(db_path=temp_db)
        
        context = TradingContext(
            symbol="ES",
            current_price=4950.0,
            timestamp=datetime.utcnow(),
            rsi=30.0,
            macd=0.5,
            macd_signal=0.3,
            macd_hist=0.2,
            atr=10.0,
        )
        
        recommendation = TradeRecommendation(
            trade_decision="BUY",
            confidence=0.85,
            reasoning="Test trade",
        )
        
        outcome = TradeOutcome(
            order_id=12345,
            symbol="ES",
            timestamp=datetime.utcnow(),
            action="BUY",
            quantity=2,
            entry_price=4950.0,
            entry_context=context,
        )
        
        trade_id = logger.log_trade_entry(outcome, recommendation)
        
        assert trade_id > 0
    
    def test_update_trade_exit(self, temp_db):
        """Test updating trade with exit information."""
        logger = TradeLogger(db_path=temp_db)
        
        # Log entry
        outcome = TradeOutcome(
            order_id=12345,
            symbol="ES",
            timestamp=datetime.utcnow(),
            action="BUY",
            quantity=2,
            entry_price=4950.0,
        )
        
        trade_id = logger.log_trade_entry(outcome)
        
        # Update with exit
        logger.update_trade_exit(
            order_id=12345,
            exit_price=4960.0,
            realized_pnl=500.0,
            trade_duration_minutes=15.0,
            outcome="WIN",
        )
        
        # Verify update
        trades = logger.get_recent_trades(limit=1)
        assert len(trades) == 1
        assert trades[0]["exit_price"] == 4960.0
        assert trades[0]["outcome"] == "WIN"
    
    def test_performance_summary(self, temp_db):
        """Test performance summary calculation."""
        logger = TradeLogger(db_path=temp_db)
        
        # Log some trades
        for i in range(5):
            outcome = TradeOutcome(
                order_id=10000 + i,
                symbol="ES",
                timestamp=datetime.utcnow() - timedelta(days=i),
                action="BUY",
                quantity=2,
                entry_price=4950.0,
            )
            logger.log_trade_entry(outcome)
            
            # Update with exit
            pnl = 500.0 if i % 2 == 0 else -250.0
            outcome_type = "WIN" if pnl > 0 else "LOSS"
            
            logger.update_trade_exit(
                order_id=10000 + i,
                exit_price=4960.0 if pnl > 0 else 4945.0,
                realized_pnl=pnl,
                trade_duration_minutes=15.0,
                outcome=outcome_type,
            )
        
        summary = logger.get_performance_summary(days=30)
        
        assert summary["total_trades"] == 5
        assert summary["winning_trades"] == 3
        assert summary["losing_trades"] == 2
        assert summary["win_rate"] == 0.6


class TestSentimentAggregator:
    """Test sentiment aggregator."""
    
    def test_sentiment_aggregator_without_comprehend(self):
        """Test sentiment aggregator without AWS Comprehend."""
        aggregator = SentimentAggregator(enable_comprehend=False)
        
        assert aggregator.enable_comprehend is False
        assert aggregator.comprehend_client is None
    
    def test_aggregate_sentiment_with_existing(self):
        """Test sentiment aggregation with existing sentiment."""
        aggregator = SentimentAggregator(enable_comprehend=False)
        
        aggregated = aggregator.aggregate_sentiment(
            existing_sentiment=0.5
        )
        
        assert aggregated == 0.5
    
    def test_aggregate_sentiment_multiple_sources(self):
        """Test aggregation from multiple sources."""
        aggregator = SentimentAggregator(enable_comprehend=False)
        
        # Should aggregate multiple sources (even when Comprehend disabled, still aggregates existing sentiment)
        aggregated = aggregator.aggregate_sentiment(
            news_headlines=["Market rallies on positive news"],
            social_media_posts=["Bullish sentiment"],
            existing_sentiment=0.3,
        )
        
        # Aggregated value should be in valid range
        assert -1.0 <= aggregated <= 1.0


class TestLLMEnhancedStrategy:
    """Test LLM-enhanced strategy."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features DataFrame."""
        np.random.seed(42)
        n = 50
        
        dates = pd.date_range('2024-01-01', periods=n, freq='5min')
        
        data = pd.DataFrame({
            'close': 4950.0 + np.random.randn(n) * 5,
            'RSI_14': 30.0 + np.random.randn(n) * 10,
            'MACD_12_26_9': np.random.randn(n) * 0.5,
            'MACDsignal_12_26_9': np.random.randn(n) * 0.4,
            'MACDhist_12_26_9': np.random.randn(n) * 0.2,
            'ATR_14': 10.0 + np.random.randn(n) * 2,
            'sentiment_score': np.random.randn(n) * 0.3,
        }, index=dates)
        
        return data
    
    def test_llm_enhanced_strategy_initialization(self):
        """Test strategy initialization."""
        strategy = LLMEnhancedStrategy(enable_llm=False)
        
        assert strategy.name == "llm_enhanced"
        assert strategy.enable_llm is False
        assert strategy.base_strategy is not None
    
    def test_generate_signal_without_llm(self, sample_features):
        """Test signal generation with LLM disabled."""
        strategy = LLMEnhancedStrategy(enable_llm=False)
        
        signal = strategy.generate(sample_features)
        
        assert signal.action in ["BUY", "SELL", "HOLD"]
        assert 0.0 <= signal.confidence <= 1.0
    
    def test_update_config(self):
        """Test updating strategy configuration."""
        strategy = LLMEnhancedStrategy(
            enable_llm=False,
            min_llm_confidence=0.7
        )
        
        strategy.update_config(
            min_llm_confidence=0.8,
            llm_override_mode=True
        )
        
        assert strategy.min_llm_confidence == 0.8
        assert strategy.llm_override_mode is True


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
