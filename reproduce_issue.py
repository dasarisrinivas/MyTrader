
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from mytrader.strategies.engine import StrategyEngine
from mytrader.strategies.momentum_reversal import MomentumReversalStrategy
from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
from mytrader.features.feature_engineer import engineer_features

def generate_dummy_data(n_bars=100, volatility=0.0002):
    """Generate dummy 5-second bar data."""
    base_price = 5000.0
    prices = [base_price]
    timestamps = [datetime.utcnow() - timedelta(seconds=5 * (n_bars - i)) for i in range(n_bars)]
    
    for _ in range(n_bars - 1):
        change = np.random.normal(0.0002, volatility) # Add positive trend (0.02% per bar)
        prices.append(prices[-1] * (1 + change))
        
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + 0.0001) for p in prices],
        'low': [p * (1 - 0.0001) for p in prices],
        'close': prices,
        'volume': 100
    })
    df.set_index('timestamp', inplace=True)
    return df

def test_strategies():
    # Create strategies
    strategies = [
        RsiMacdSentimentStrategy(),
        MomentumReversalStrategy(threshold=0.01) # Current high threshold
    ]
    engine = StrategyEngine(strategies)
    
    print("Testing with current threshold (0.01)...")
    
    # Generate data
    df = generate_dummy_data(n_bars=200, volatility=0.0001) # Realistic 5s volatility
    features = engineer_features(df, None)
    returns = df['close'].pct_change().dropna()
    
    # Evaluate
    signal = engine.evaluate(features, returns)
    print(f"Signal with normal volatility: {signal.action} (Conf: {signal.confidence:.4f})")
    print(f"Metadata: {signal.metadata}")
    
    # Generate high volatility data (but still likely below 1% per 5s)
    df_high = generate_dummy_data(n_bars=200, volatility=0.005) # High volatility
    features_high = engineer_features(df_high, None)
    returns_high = df_high['close'].pct_change().dropna()
    
    signal_high = engine.evaluate(features_high, returns_high)
    print(f"Signal with high volatility: {signal_high.action} (Conf: {signal_high.confidence:.4f})")
    print(f"Metadata: {signal_high.metadata}")
    
    # Test with default threshold (should now be 0.0001)
    print("\nTesting with DEFAULT threshold (should be 0.0001)...")
    strategies_default = [
        RsiMacdSentimentStrategy(),
        MomentumReversalStrategy() # Uses default
    ]
    engine_default = StrategyEngine(strategies_default)
    
    signal_default = engine_default.evaluate(features, returns)
    print(f"Signal with normal volatility (default threshold): {signal_default.action} (Conf: {signal_default.confidence:.4f})")
    print(f"Metadata: {signal_default.metadata}")

if __name__ == "__main__":
    test_strategies()
