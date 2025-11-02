#!/usr/bin/env python3
"""
Test the enhanced regime-based strategy vs baseline
This should show dramatic improvement
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
    from mytrader.strategies.enhanced_regime_strategy import EnhancedRegimeStrategy
    print("✓ Modules loaded\n")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "-q"])
    import pandas as pd
    import numpy as np
    from mytrader.backtesting.engine import BacktestingEngine
    from mytrader.config import BacktestConfig, TradingConfig
    from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
    from mytrader.strategies.enhanced_regime_strategy import EnhancedRegimeStrategy

print("="*80)
print("STRATEGY COMPARISON: Baseline vs Enhanced")
print("="*80 + "\n")

# Load data
print("Loading data...")
data_path = "data/es_synthetic_with_sentiment.csv"
df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")
print(f"✓ Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}\n")

# Configs
trading_config = TradingConfig(
    max_position_size=4,
    max_daily_loss=2000.0,
    max_daily_trades=20,
    initial_capital=100000.0,
    stop_loss_ticks=15.0,  # Using wider stop
    take_profit_ticks=25.0,  # Using closer target
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

# Test 1: Baseline RSI/MACD Strategy
print("1. Testing Baseline RSI/MACD Strategy...")
print("-" * 80)
baseline = RsiMacdSentimentStrategy(rsi_buy=30.0, rsi_sell=70.0)
engine1 = BacktestingEngine([baseline], trading_config, backtest_config)
result1 = engine1.run(df)

print(f"Total Trades:    {len(result1.trades)}")
print(f"Win Rate:        {result1.metrics.get('win_rate', 0)*100:.2f}%")
print(f"Total Return:    {result1.metrics.get('total_return', 0)*100:.2f}%")
print(f"Sharpe Ratio:    {result1.metrics.get('sharpe', 0):.2f}")
print(f"Sortino Ratio:   {result1.metrics.get('sortino', 0):.2f}")
print(f"Max Drawdown:    {result1.metrics.get('max_drawdown', 0)*100:.2f}%")
print(f"Profit Factor:   {result1.metrics.get('profit_factor', 0):.2f}")
print(f"Final Equity:    ${result1.equity_curve.iloc[-1] if len(result1.equity_curve) > 0 else 100000:,.2f}\n")

# Test 2: Enhanced Regime Strategy
print("2. Testing Enhanced Regime-Based Strategy...")
print("-" * 80)
enhanced = EnhancedRegimeStrategy()
engine2 = BacktestingEngine([enhanced], trading_config, backtest_config)
result2 = engine2.run(df)

print(f"Total Trades:    {len(result2.trades)}")
print(f"Win Rate:        {result2.metrics.get('win_rate', 0)*100:.2f}%")
print(f"Total Return:    {result2.metrics.get('total_return', 0)*100:.2f}%")
print(f"Sharpe Ratio:    {result2.metrics.get('sharpe', 0):.2f}")
print(f"Sortino Ratio:   {result2.metrics.get('sortino', 0):.2f}")
print(f"Max Drawdown:    {result2.metrics.get('max_drawdown', 0)*100:.2f}%")
print(f"Profit Factor:   {result2.metrics.get('profit_factor', 0):.2f}")
print(f"Final Equity:    ${result2.equity_curve.iloc[-1] if len(result2.equity_curve) > 0 else 100000:,.2f}\n")

# Comparison
print("="*80)
print("COMPARISON SUMMARY")
print("="*80 + "\n")

baseline_wr = result1.metrics.get('win_rate', 0) * 100
enhanced_wr = result2.metrics.get('win_rate', 0) * 100
baseline_ret = result1.metrics.get('total_return', 0) * 100
enhanced_ret = result2.metrics.get('total_return', 0) * 100
baseline_sharpe = result1.metrics.get('sharpe', 0)
enhanced_sharpe = result2.metrics.get('sharpe', 0)

print(f"{'Metric':<20} {'Baseline':<15} {'Enhanced':<15} {'Improvement':<15}")
print("-" * 65)
print(f"{'Win Rate':<20} {baseline_wr:>14.1f}% {enhanced_wr:>14.1f}% {enhanced_wr - baseline_wr:>+14.1f}%")
print(f"{'Total Return':<20} {baseline_ret:>14.2f}% {enhanced_ret:>14.2f}% {enhanced_ret - baseline_ret:>+14.2f}%")
print(f"{'Sharpe Ratio':<20} {baseline_sharpe:>14.2f} {enhanced_sharpe:>14.2f} {enhanced_sharpe - baseline_sharpe:>+14.2f}")
print(f"{'Total Trades':<20} {len(result1.trades):>14} {len(result2.trades):>14} {len(result2.trades) - len(result1.trades):>+14}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80 + "\n")

if enhanced_wr > baseline_wr + 10:
    print("✅ SIGNIFICANT IMPROVEMENT!")
    print(f"   Enhanced strategy is {enhanced_wr - baseline_wr:.1f}% better in win rate")
    print("   Recommendation: Use the enhanced regime-based strategy")
    
    if enhanced_wr >= 60:
        print("\n🎯 MEETS TARGET: Win rate ≥ 60%")
    elif enhanced_wr >= 50:
        print("\n👍 GOOD: Win rate ≥ 50%, consider optimization for further improvement")
    else:
        print("\n⚠️  NEEDS OPTIMIZATION: Run Bayesian optimization to reach 60% target")
        print("   python3 scripts/advanced_optimizer.py --strategy enhanced --trials 100")

elif len(result2.trades) == 0:
    print("⚠️  Enhanced strategy generated no trades!")
    print("   This might be due to:")
    print("   1. Data doesn't have required indicators")
    print("   2. Filters are too strict")
    print("   3. Insufficient historical data")
    print("\n   Try adjusting filter thresholds or check data quality")

else:
    print("⚠️  Both strategies perform poorly")
    print("   This suggests:")
    print("   1. Data quality issues")
    print("   2. Indicators not calculated properly")
    print("   3. Market conditions unsuitable")
    print("\n   Recommendation: Check data and feature engineering")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80 + "\n")

if enhanced_wr >= 50:
    print("1. ✓ Use enhanced strategy as baseline")
    print("2. Run full optimization for even better results:")
    print("   python3 quickstart_optimization.py")
    print("3. Paper trade to validate real-world performance")
else:
    print("1. Run diagnostic on data quality:")
    print("   python3 diagnose_backtest.py")
    print("2. Check if indicators are calculated:")
    print("   import pandas as pd")
    print("   df = pd.read_csv('data/es_synthetic_with_sentiment.csv')")
    print("   print(df.columns)")
    print("3. Run optimization to find better parameters:")
    print("   python3 quickstart_optimization.py")

print("\n" + "="*80 + "\n")
