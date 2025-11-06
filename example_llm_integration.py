#!/usr/bin/env python3
"""Example script demonstrating AWS Bedrock LLM integration for trading.

This script shows how to:
1. Use the LLM-enhanced strategy
2. Query the LLM directly for recommendations
3. Log trades for training
4. View performance metrics
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from mytrader.features.feature_engineer import engineer_features
from mytrader.llm.bedrock_client import BedrockClient
from mytrader.llm.data_schema import TradingContext, TradeOutcome
from mytrader.llm.sentiment_aggregator import SentimentAggregator
from mytrader.llm.trade_advisor import TradeAdvisor
from mytrader.llm.trade_logger import TradeLogger
from mytrader.strategies.llm_enhanced_strategy import LLMEnhancedStrategy


def create_sample_market_data() -> pd.DataFrame:
    """Create sample market data for demonstration."""
    print("📊 Creating sample market data...")
    
    np.random.seed(42)
    n = 100
    
    # Generate trending price data
    base_price = 4950
    trend = np.linspace(0, 20, n)
    noise = np.random.randn(n) * 2
    close_prices = base_price + trend + noise
    
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(n) * 0.5,
        'high': close_prices + abs(np.random.randn(n) * 1),
        'low': close_prices - abs(np.random.randn(n) * 1),
        'close': close_prices,
        'volume': np.random.randint(10000, 15000, n)
    })
    data.index = pd.date_range('2024-01-01', periods=n, freq='5min')
    
    print(f"✅ Created {len(data)} bars of market data")
    return data


def example_1_llm_enhanced_strategy():
    """Example 1: Using LLM-enhanced strategy."""
    print("\n" + "="*70)
    print("EXAMPLE 1: LLM-Enhanced Strategy")
    print("="*70)
    
    # Create sample data
    market_data = create_sample_market_data()
    
    # Engineer features
    print("\n🔧 Engineering features...")
    features = engineer_features(market_data, None)  # No sentiment data for demo
    
    # Create LLM-enhanced strategy (with LLM disabled for demo)
    print("\n🤖 Initializing LLM-enhanced strategy...")
    strategy = LLMEnhancedStrategy(
        enable_llm=False,  # Set to True if you have AWS credentials configured
        min_llm_confidence=0.7,
        llm_override_mode=False,
    )
    
    # Generate signal
    print("\n📈 Generating trading signal...")
    signal = strategy.generate(features)
    
    print(f"\n✅ Signal Generated:")
    print(f"   Action: {signal.action}")
    print(f"   Confidence: {signal.confidence:.2f}")
    print(f"   Metadata: {signal.metadata}")
    
    print("\n💡 Tip: Set enable_llm=True and configure AWS credentials to use LLM features")


def example_2_direct_llm_query():
    """Example 2: Directly query LLM for recommendation."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Direct LLM Query (Demonstration)")
    print("="*70)
    
    # Create trading context
    print("\n📊 Building trading context...")
    context = TradingContext(
        symbol="ES",
        current_price=4950.0,
        timestamp=datetime.now(),
        rsi=28.5,  # Oversold
        macd=0.5,
        macd_signal=0.3,
        macd_hist=0.2,
        atr=10.0,
        adx=32.0,  # Strong trend
        bb_percent=0.15,  # Near lower band
        sentiment_score=0.2,  # Slightly positive
        current_position=0,
        unrealized_pnl=0.0,
        portfolio_heat=0.15,
        daily_pnl=250.0,
        win_rate=0.65,
        market_regime="mean_reverting",
        volatility_regime="normal",
    )
    
    print("\n📝 Context Summary:")
    print(f"   Price: ${context.current_price:.2f}")
    print(f"   RSI: {context.rsi:.2f} (Oversold)")
    print(f"   MACD: {context.macd:.4f}")
    print(f"   Sentiment: {context.sentiment_score:.2f}")
    print(f"   Market Regime: {context.market_regime}")
    
    print("\n💡 In production, this would query AWS Bedrock LLM:")
    print("   1. Send context to Claude 3 / Titan")
    print("   2. Receive structured JSON recommendation")
    print("   3. Get reasoning, confidence, and risk assessment")
    
    # Show what the response would look like
    print("\n📋 Example LLM Response:")
    print("""
    {
        "trade_decision": "BUY",
        "confidence": 0.85,
        "suggested_position_size": 2,
        "suggested_stop_loss": 4945.0,
        "suggested_take_profit": 4960.0,
        "reasoning": "Strong oversold signal with RSI at 28.5, bullish MACD...",
        "key_factors": [
            "RSI oversold (28.5 < 30)",
            "Bullish MACD histogram divergence",
            "Sentiment improving (0.2)",
            "ATR suggests controlled volatility"
        ],
        "risk_assessment": "Low risk entry with 2:1 reward ratio..."
    }
    """)
    
    print("\n💡 To enable: Set AWS credentials and enable_llm=True in config")


def example_3_trade_logging():
    """Example 3: Logging trades for training."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Trade Logging for LLM Training")
    print("="*70)
    
    # Initialize trade logger
    print("\n📝 Initializing trade logger...")
    logger = TradeLogger(db_path="data/demo_llm_trades.db")
    
    # Log a sample trade entry
    print("\n📊 Logging sample trade entry...")
    
    context = TradingContext(
        symbol="ES",
        current_price=4950.0,
        timestamp=datetime.now(),
        rsi=28.5,
        macd=0.5,
        macd_signal=0.3,
        macd_hist=0.2,
        atr=10.0,
    )
    
    outcome = TradeOutcome(
        order_id=10001,
        symbol="ES",
        timestamp=datetime.now(),
        action="BUY",
        quantity=2,
        entry_price=4950.0,
        entry_context=context,
    )
    
    trade_id = logger.log_trade_entry(outcome)
    print(f"✅ Logged trade entry with ID: {trade_id}")
    
    # Simulate trade exit after some time
    print("\n📊 Updating trade with exit information...")
    logger.update_trade_exit(
        order_id=10001,
        exit_price=4960.0,
        realized_pnl=500.0,
        trade_duration_minutes=15.0,
        outcome="WIN",
    )
    print("✅ Trade exit logged")
    
    # View recent trades
    print("\n📋 Recent Trades:")
    recent = logger.get_recent_trades(limit=5)
    for trade in recent:
        print(f"   {trade['timestamp']}: {trade['action']} @ ${trade['entry_price']:.2f}")
        if trade['exit_price']:
            print(f"      Exit: ${trade['exit_price']:.2f} | P&L: ${trade['realized_pnl']:.2f}")
    
    # View performance summary
    print("\n📊 Performance Summary:")
    summary = logger.get_performance_summary(days=30)
    print(f"   Total Trades: {summary['total_trades']}")
    print(f"   Win Rate: {summary['win_rate']:.2%}")
    print(f"   Total P&L: ${summary['total_pnl']:.2f}")
    
    print("\n💡 This data can be exported to S3 for LLM fine-tuning")


def example_4_sentiment_analysis():
    """Example 4: Sentiment aggregation."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Sentiment Analysis (Demonstration)")
    print("="*70)
    
    # Initialize sentiment aggregator (without AWS Comprehend for demo)
    print("\n📊 Initializing sentiment aggregator...")
    aggregator = SentimentAggregator(enable_comprehend=False)
    
    print("\n📝 In production, this would:")
    print("   1. Fetch news headlines from NewsAPI, Finnhub, etc.")
    print("   2. Analyze sentiment using AWS Comprehend")
    print("   3. Aggregate scores from multiple sources")
    print("   4. Normalize to -1.0 (bearish) to +1.0 (bullish)")
    
    # Demonstrate aggregation
    print("\n📊 Example sentiment aggregation:")
    sentiment = aggregator.aggregate_sentiment(
        news_headlines=[
            "S&P 500 rallies on strong earnings",
            "Fed signals pause in rate hikes",
        ],
        existing_sentiment=0.2,
    )
    
    print(f"\n✅ Aggregated Sentiment: {sentiment:.3f}")
    print("\n💡 To enable AWS Comprehend: Set enable_comprehend=True and configure credentials")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("AWS BEDROCK LLM INTEGRATION EXAMPLES")
    print("="*70)
    print("\nThis script demonstrates the LLM integration features.")
    print("Note: LLM features are disabled for this demo.")
    print("To enable: Configure AWS credentials and set enable_llm=True\n")
    
    try:
        # Run examples
        example_1_llm_enhanced_strategy()
        example_2_direct_llm_query()
        example_3_trade_logging()
        example_4_sentiment_analysis()
        
        print("\n" + "="*70)
        print("✅ ALL EXAMPLES COMPLETED")
        print("="*70)
        print("\n📖 For more information:")
        print("   - Read: LLM_INTEGRATION.md")
        print("   - Configure: config.example.yaml")
        print("   - Tests: pytest tests/test_llm_integration.py")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
