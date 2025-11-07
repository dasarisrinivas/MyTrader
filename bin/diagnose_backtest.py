#!/usr/bin/env python3
"""
Quick diagnostic to understand why backtest has poor win rate
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Loading modules...")
try:
    import pandas as pd
    import numpy as np
    from mytrader.backtesting.engine import BacktestingEngine
    from mytrader.config import BacktestConfig, TradingConfig
    from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
    from mytrader.strategies.momentum_reversal import MomentumReversalStrategy
    print("✓ Modules loaded\n")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nInstalling required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "-q"])
    print("Please run the script again.")
    sys.exit(1)

print("="*80)
print("BACKTEST DIAGNOSTIC - Finding Win Rate Issues")
print("="*80 + "\n")

# Load data
print("1. Loading data...")
data_path = "data/es_synthetic_with_sentiment.csv"
df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")
print(f"   ✓ Loaded {len(df)} bars")
print(f"   ✓ Date range: {df.index[0]} to {df.index[-1]}")
print(f"   ✓ Columns: {', '.join(df.columns)}\n")

# Configuration
trading_config = TradingConfig(
    max_position_size=4,
    max_daily_loss=2000.0,
    max_daily_trades=20,
    initial_capital=100000.0,
    stop_loss_ticks=10.0,
    take_profit_ticks=20.0,
    tick_size=0.25,
    tick_value=12.5,
    commission_per_contract=2.4,
    contract_multiplier=50.0
)

backtest_config = BacktestConfig(
    initial_capital=100000.0,
    slippage=0.25,
    risk_free_rate=0.02
)

print("2. Testing RSI/MACD Strategy (baseline)...")
strategy1 = RsiMacdSentimentStrategy()
engine1 = BacktestingEngine([strategy1], trading_config, backtest_config)
result1 = engine1.run(df)

print(f"   Total Trades: {len(result1.trades)}")
print(f"   Win Rate: {result1.metrics.get('win_rate', 0)*100:.2f}%")
print(f"   Sharpe: {result1.metrics.get('sharpe', 0):.2f}")
print(f"   Max DD: {result1.metrics.get('max_drawdown', 0)*100:.2f}%")
print(f"   Total Return: {result1.metrics.get('total_return', 0)*100:.2f}%")
print(f"   Profit Factor: {result1.metrics.get('profit_factor', 0):.2f}\n")

# Analyze trades
if result1.trades:
    print("3. Trade Analysis:")
    wins = [t for t in result1.trades if t.get('realized', 0) > 0]
    losses = [t for t in result1.trades if t.get('realized', 0) < 0]
    
    print(f"   Winning trades: {len(wins)}")
    print(f"   Losing trades: {len(losses)}")
    
    if wins:
        avg_win = np.mean([t['realized'] for t in wins])
        print(f"   Average win: ${avg_win:.2f}")
    
    if losses:
        avg_loss = np.mean([t['realized'] for t in losses])
        print(f"   Average loss: ${avg_loss:.2f}")
    
    # Show first 10 trades
    print("\n   First 10 trades:")
    for i, trade in enumerate(result1.trades[:10]):
        pnl = trade.get('realized', 0)
        action = trade.get('action', 'UNKNOWN')
        price = trade.get('price', 0)
        qty = trade.get('qty', 0)
        status = "WIN" if pnl > 0 else "LOSS"
        print(f"   {i+1}. {action:6s} {qty:2d} @ ${price:.2f} → {status:4s} ${pnl:+8.2f}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

win_rate = result1.metrics.get('win_rate', 0)
total_return = result1.metrics.get('total_return', 0)

if win_rate < 0.3:
    print("\n⚠️  CRITICAL: Very low win rate (<30%)")
    print("   Possible causes:")
    print("   1. Stop loss too tight (10 ticks = $125)")
    print("   2. Take profit too far (20 ticks = $250)")
    print("   3. Poor entry timing (signals not aligned with price action)")
    print("   4. Strategy parameters need optimization")
    
    print("\n   Recommendations:")
    print("   • Try wider stops: 15-20 ticks instead of 10")
    print("   • Try closer targets: 15-25 ticks instead of 20")
    print("   • Optimize RSI thresholds (currently 40/60)")
    print("   • Run Bayesian optimization:")
    print("     python3 scripts/advanced_optimizer.py --trials 50")

elif win_rate < 0.5:
    print("\n⚠️  Low win rate (30-50%)")
    print("   This is marginal. Optimization recommended.")
    print("   Run: python3 scripts/advanced_optimizer.py --trials 50")

else:
    print("\n✓ Win rate acceptable (>50%)")
    print("   Consider further optimization for improvement.")

if total_return < 0:
    print(f"\n⚠️  CRITICAL: Losing strategy (Return: {total_return*100:.2f}%)")
    print("   DO NOT trade this without optimization!")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. Run optimization to find better parameters:")
print("   python3 scripts/advanced_optimizer.py \\")
print("       --data data/es_synthetic_with_sentiment.csv \\")
print("       --strategy enhanced \\")
print("       --trials 100 \\")
print("       --output reports/optimization.json")

print("\n2. Or try adjusted parameters manually:")
print("   - Increase stop loss to 15-20 ticks")
print("   - Adjust take profit to 20-30 ticks")
print("   - Optimize RSI levels (try 35/65 or 30/70)")

print("\n3. Test the enhanced regime strategy:")
print("   (This strategy adapts to market conditions)")
print("   python3 scripts/performance_analyzer.py")

print("\n" + "="*80 + "\n")
