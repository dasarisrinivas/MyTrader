#!/usr/bin/env python3
"""
Check if data has required technical indicators
"""
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Installing pandas...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "-q"])
    import pandas as pd

print("="*80)
print("DATA QUALITY CHECK")
print("="*80 + "\n")

# Load data
data_path = "data/es_synthetic_with_sentiment.csv"
print(f"Loading: {data_path}")
df = pd.read_csv(data_path)

print(f"\n✓ Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"✓ Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}\n")

print("Columns present:")
print("-" * 80)
for col in df.columns:
    print(f"  • {col}")

print("\n" + "="*80)
print("INDICATOR CHECK")
print("="*80 + "\n")

required_indicators = {
    "Basic OHLCV": ["open", "high", "low", "close", "volume"],
    "RSI": ["RSI_14"],
    "MACD": ["MACD_12_26_9", "MACDsignal_12_26_9", "MACDhist_12_26_9"],
    "Bollinger Bands": ["BB_upper", "BB_lower", "BB_middle", "BB_percent", "BB_width"],
    "Moving Averages": ["SMA_20", "SMA_50", "EMA_20", "EMA_50"],
    "ADX": ["ADX_14"],
    "ATR": ["ATR_14"],
    "Sentiment": ["sentiment_twitter", "sentiment_news", "sentiment_score"]
}

missing = []
present = []

for category, indicators in required_indicators.items():
    print(f"{category}:")
    for indicator in indicators:
        if indicator in df.columns:
            print(f"  ✅ {indicator}")
            present.append(indicator)
        else:
            print(f"  ❌ {indicator} - MISSING")
            missing.append(indicator)
    print()

print("="*80)
print("SUMMARY")
print("="*80 + "\n")

print(f"Present: {len(present)} indicators")
print(f"Missing: {len(missing)} indicators\n")

if missing:
    print("⚠️  PROBLEM: Missing indicators!")
    print("\nThe data only has basic OHLCV and sentiment.")
    print("Technical indicators (RSI, MACD, etc.) need to be calculated.\n")
    
    print("The backtesting engine should calculate these automatically via")
    print("feature_engineer.py, but let's verify:\n")
    
    print("Solution: The engineer_features() function should add all indicators.")
    print("Check: mytrader/features/feature_engineer.py")
    
    print("\nTo test if indicators are being calculated during backtest:")
    print("  python3 diagnose_backtest.py")
    
else:
    print("✅ All required indicators are present!")
    print("The data is properly formatted with all technical indicators.\n")
    print("You can proceed with:")
    print("  python3 test_enhanced_strategy.py")

print("\n" + "="*80 + "\n")
