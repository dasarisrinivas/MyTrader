#!/usr/bin/env python3
"""Test signal generation."""

import pandas as pd
import numpy as np
from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
from mytrader.features.feature_engineer import engineer_features

# Generate data
np.random.seed(42)
n_bars = 1000
trend = np.linspace(4700, 4800, n_bars)
noise = np.random.normal(0, 5, n_bars).cumsum() * 0.5
close = trend + noise

data = {
    'timestamp': pd.date_range('2024-01-01 09:30', periods=n_bars, freq='1min'),
    'open': close - np.random.uniform(0.5, 2, n_bars),
    'high': close + np.random.uniform(0.5, 3, n_bars),
    'low': close - np.random.uniform(0.5, 3, n_bars),
    'close': close,
    'volume': np.random.randint(10000, 15000, n_bars),
    'sentiment_twitter': np.random.uniform(-0.3, 0.3, n_bars),
    'sentiment_news': np.random.uniform(-0.2, 0.2, n_bars)
}

df = pd.DataFrame(data)
df.set_index('timestamp', inplace=True)

# Engineer features
df_features = engineer_features(df)

# Test strategy with very relaxed parameters
strategy = RsiMacdSentimentStrategy(
    rsi_buy=50,  # Buy when RSI < 50
    rsi_sell=50,  # Sell when RSI > 50
    sentiment_buy=-0.5,  # Very low bar
    sentiment_sell=-0.5   # Very low bar
)

# Check last 20 rows
print("Last 20 bars - Signal Analysis:")
print("=" * 100)
for i in range(-20, 0):
    row = df_features.iloc[i:i+1]
    signal = strategy.generate(row)
    
    rsi = row['RSI_14'].values[0] if 'RSI_14' in row.columns else 50
    macd = row['MACD_12_26_9'].values[0] if 'MACD_12_26_9' in row.columns else 0
    sentiment = row['sentiment_score'].values[0] if 'sentiment_score' in row.columns else 0
    
    print(f"Bar {n_bars + i}: RSI={rsi:>6.2f} | MACD={macd:>7.2f} | Sentiment={sentiment:>6.2f} | Signal={signal.action:>4} | Conf={signal.confidence:.2f}")

print("\n" + "=" * 100)
print("\nStrategy Conditions:")
print(f"BUY:  RSI < {strategy.rsi_buy} AND MACD > 0 AND Sentiment >= {strategy.sentiment_buy}")
print(f"SELL: RSI > {strategy.rsi_sell} AND MACD < 0 AND Sentiment <= {strategy.sentiment_sell}")
