#!/usr/bin/env python3
"""Generate comprehensive synthetic market data with multiple regimes."""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_trending_data(n: int, base_price: float, trend_strength: float, volatility: float) -> pd.DataFrame:
    """Generate trending market data."""
    np.random.seed(42)
    
    # Trend component
    trend = np.linspace(0, trend_strength * n, n)
    
    # Random walk component
    returns = np.random.randn(n) * volatility
    cumulative_returns = np.cumsum(returns)
    
    # Combine trend and noise
    close = base_price + trend + cumulative_returns
    
    # Generate OHLCV
    data = pd.DataFrame({
        'open': np.roll(close, 1),
        'high': close + np.abs(np.random.randn(n) * volatility * 1.5),
        'low': close - np.abs(np.random.randn(n) * volatility * 1.5),
        'close': close,
        'volume': np.random.randint(10000, 20000, n)
    })
    data.loc[0, 'open'] = base_price
    
    return data


def generate_mean_reverting_data(n: int, base_price: float, volatility: float) -> pd.DataFrame:
    """Generate mean-reverting market data."""
    np.random.seed(43)
    
    # Oscillate around base price
    oscillation = np.sin(np.linspace(0, 4 * np.pi, n)) * volatility * 5
    noise = np.random.randn(n) * volatility * 0.5
    
    close = base_price + oscillation + noise
    
    data = pd.DataFrame({
        'open': np.roll(close, 1),
        'high': close + np.abs(np.random.randn(n) * volatility * 0.8),
        'low': close - np.abs(np.random.randn(n) * volatility * 0.8),
        'close': close,
        'volume': np.random.randint(10000, 20000, n)
    })
    data.loc[0, 'open'] = base_price
    
    return data


def generate_high_volatility_data(n: int, base_price: float) -> pd.DataFrame:
    """Generate high volatility market data."""
    np.random.seed(44)
    
    # Large random moves
    returns = np.random.randn(n) * 0.015  # 1.5% moves
    close = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': np.roll(close, 1),
        'high': close * (1 + np.abs(np.random.randn(n) * 0.01)),
        'low': close * (1 - np.abs(np.random.randn(n) * 0.01)),
        'close': close,
        'volume': np.random.randint(15000, 30000, n)  # Higher volume
    })
    data.loc[0, 'open'] = base_price
    
    return data


def add_sentiment_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic sentiment data."""
    np.random.seed(45)
    n = len(df)
    
    # Sentiment correlated with price movement but with lag and noise
    returns = df['close'].pct_change().fillna(0)
    
    # Twitter sentiment - more volatile, leads price
    twitter_base = returns.rolling(5).mean().shift(-2).fillna(0)
    twitter_noise = np.random.randn(n) * 0.15
    df['sentiment_twitter'] = (twitter_base + twitter_noise).clip(-0.8, 0.8)
    
    # News sentiment - less volatile, lags price
    news_base = returns.rolling(10).mean().shift(3).fillna(0)
    news_noise = np.random.randn(n) * 0.08
    df['sentiment_news'] = (news_base + news_noise).clip(-0.6, 0.6)
    
    return df


def main():
    """Generate comprehensive market data with multiple regimes."""
    base_price = 4700.0
    bars_per_day = 390  # 6.5 hours * 60 minutes
    
    # Create different market regimes
    segments = []
    
    # 1. Strong uptrend (2 days)
    print("Generating uptrend regime...")
    uptrend = generate_trending_data(bars_per_day * 2, base_price, 0.08, 2.0)
    segments.append(uptrend)
    
    # 2. Mean reverting / sideways (2 days)
    print("Generating mean-reverting regime...")
    sideways = generate_mean_reverting_data(bars_per_day * 2, uptrend['close'].iloc[-1], 3.0)
    segments.append(sideways)
    
    # 3. Downtrend (2 days)
    print("Generating downtrend regime...")
    downtrend = generate_trending_data(bars_per_day * 2, sideways['close'].iloc[-1], -0.08, 2.5)
    segments.append(downtrend)
    
    # 4. High volatility (1 day)
    print("Generating high volatility regime...")
    volatile = generate_high_volatility_data(bars_per_day, downtrend['close'].iloc[-1])
    segments.append(volatile)
    
    # 5. Recovery uptrend (2 days)
    print("Generating recovery uptrend...")
    recovery = generate_trending_data(bars_per_day * 2, volatile['close'].iloc[-1], 0.06, 2.0)
    segments.append(recovery)
    
    # 6. Final consolidation (1 day)
    print("Generating consolidation regime...")
    consolidation = generate_mean_reverting_data(bars_per_day, recovery['close'].iloc[-1], 2.0)
    segments.append(consolidation)
    
    # Combine all segments
    print("Combining segments...")
    full_data = pd.concat(segments, ignore_index=True)
    
    # Add timestamps
    start_date = pd.Timestamp('2024-01-02 09:30:00')
    full_data.index = pd.date_range(start_date, periods=len(full_data), freq='1min')
    
    # Filter to market hours only (9:30 AM - 4:00 PM)
    full_data = full_data.between_time('09:30', '16:00')
    
    # Reset index to make timestamp a column
    full_data = full_data.reset_index()
    full_data.rename(columns={'index': 'timestamp'}, inplace=True)
    
    # Add sentiment data
    print("Adding sentiment data...")
    full_data = add_sentiment_data(full_data)
    
    # Save to CSV
    output_path = Path(__file__).parent.parent / 'data' / 'es_comprehensive_data.csv'
    output_path.parent.mkdir(exist_ok=True, parents=True)
    full_data.to_csv(output_path, index=False)
    
    print(f"\n✅ Generated {len(full_data)} bars of comprehensive market data")
    print(f"📁 Saved to: {output_path}")
    print(f"\n📊 Data Statistics:")
    print(f"   Price range: ${full_data['close'].min():.2f} - ${full_data['close'].max():.2f}")
    print(f"   Total return: {(full_data['close'].iloc[-1] / full_data['close'].iloc[0] - 1) * 100:.2f}%")
    print(f"   Avg volume: {full_data['volume'].mean():.0f}")
    print(f"   Sentiment range: [{full_data['sentiment_twitter'].min():.2f}, {full_data['sentiment_twitter'].max():.2f}]")


if __name__ == '__main__':
    main()
