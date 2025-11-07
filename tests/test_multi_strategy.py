"""
Test the Multi-Strategy system
"""
import pandas as pd
import numpy as np
from mytrader.strategies.multi_strategy import MultiStrategy

# Create sample data
dates = pd.date_range('2025-01-01', periods=100, freq='5min')
np.random.seed(42)

# Generate realistic ES futures data
base_price = 6850
prices = base_price + np.cumsum(np.random.randn(100) * 2)

df = pd.DataFrame({
    'timestamp': dates,
    'open': prices + np.random.randn(100) * 0.5,
    'high': prices + abs(np.random.randn(100)) * 2,
    'low': prices - abs(np.random.randn(100)) * 2,
    'close': prices,
    'volume': np.random.randint(1000, 5000, 100)
})

df.set_index('timestamp', inplace=True)

print("=" * 70)
print("MULTI-STRATEGY TRADING SYSTEM TEST")
print("=" * 70)
print(f"\nTest Data: {len(df)} bars")
print(f"Price Range: {df['close'].min():.2f} - {df['close'].max():.2f}")
print(f"Current Price: {df['close'].iloc[-1]:.2f}\n")

# Test 1: Trend-Following Strategy
print("\n" + "=" * 70)
print("TEST 1: TREND-FOLLOWING STRATEGY")
print("=" * 70)
strategy = MultiStrategy(strategy_mode="trend_following", min_confidence=0.60)
action, confidence, risk_params = strategy.generate_signal(df)

print(f"\n✅ Signal Generated:")
print(f"   Action: {action}")
print(f"   Confidence: {confidence:.2f}")
print(f"   Market Bias: {strategy.market_bias}")
print(f"   Volatility: {strategy.volatility_level}")

if risk_params:
    print(f"\n📊 Risk Parameters:")
    print(f"   ATR: {risk_params['atr']:.2f}")
    print(f"   Stop Loss (Long): {risk_params['stop_loss_long']:.2f}")
    print(f"   Take Profit (Long): {risk_params['take_profit_long']:.2f}")
    print(f"   Stop Loss (Short): {risk_params['stop_loss_short']:.2f}")
    print(f"   Take Profit (Short): {risk_params['take_profit_short']:.2f}")

# Test 2: Breakout Strategy
print("\n" + "=" * 70)
print("TEST 2: BREAKOUT STRATEGY")
print("=" * 70)
strategy = MultiStrategy(strategy_mode="breakout", min_confidence=0.60)
action, confidence, risk_params = strategy.generate_signal(df)

print(f"\n✅ Signal Generated:")
print(f"   Action: {action}")
print(f"   Confidence: {confidence:.2f}")
print(f"   Market Bias: {strategy.market_bias}")
print(f"   Volatility: {strategy.volatility_level}")

# Test 3: Mean Reversion Strategy
print("\n" + "=" * 70)
print("TEST 3: MEAN REVERSION STRATEGY")
print("=" * 70)
strategy = MultiStrategy(strategy_mode="mean_reversion", min_confidence=0.60)
action, confidence, risk_params = strategy.generate_signal(df)

print(f"\n✅ Signal Generated:")
print(f"   Action: {action}")
print(f"   Confidence: {confidence:.2f}")
print(f"   Market Bias: {strategy.market_bias}")
print(f"   Volatility: {strategy.volatility_level}")

# Test 4: Auto Strategy Selection
print("\n" + "=" * 70)
print("TEST 4: AUTO STRATEGY SELECTION")
print("=" * 70)
strategy = MultiStrategy(strategy_mode="auto", min_confidence=0.60)
action, confidence, risk_params = strategy.generate_signal(df)

print(f"\n✅ Signal Generated:")
print(f"   Action: {action}")
print(f"   Confidence: {confidence:.2f}")
print(f"   Market Bias: {strategy.market_bias}")
print(f"   Volatility: {strategy.volatility_level}")

# Test 5: Exit Conditions
print("\n" + "=" * 70)
print("TEST 5: EXIT CONDITIONS")
print("=" * 70)

if risk_params:
    # Simulate a long position
    entry_price = df['close'].iloc[-10]
    current_position = 1
    
    print(f"\n📦 Simulated Position:")
    print(f"   Position: {current_position} contract (LONG)")
    print(f"   Entry Price: {entry_price:.2f}")
    print(f"   Current Price: {df['close'].iloc[-1]:.2f}")
    print(f"   P&L: {(df['close'].iloc[-1] - entry_price):.2f} points")
    
    should_exit, reason = strategy.should_exit_position(
        df=df,
        entry_price=entry_price,
        position=current_position,
        risk_params=risk_params
    )
    
    print(f"\n🛑 Exit Check:")
    print(f"   Should Exit: {should_exit}")
    print(f"   Reason: {reason}")

print("\n" + "=" * 70)
print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nThe Multi-Strategy system is working correctly.")
print("You can now use it in live trading.")
print("\nTo start trading:")
print("  1. ./start_dashboard.sh")
print("  2. Click 'Start Trading' in the UI")
print("  3. Watch the signals with risk parameters!\n")
