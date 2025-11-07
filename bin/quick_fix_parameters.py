#!/usr/bin/env python3
"""
Quick parameter sweep to find better stop-loss/take-profit combinations
Tests multiple configurations to immediately improve win rate
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

print("="*80)
print("PARAMETER SWEEP - Finding Better Stop/Target Combinations")
print("="*80 + "\n")

# Load data
print("Loading data...")
data_path = "data/es_synthetic_with_sentiment.csv"
df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")
print(f"✓ Loaded {len(df)} bars\n")

backtest_config = BacktestConfig(
    initial_capital=100000.0,
    slippage=0.25,
    risk_free_rate=0.02
)

# Test combinations
stop_loss_options = [8.0, 10.0, 12.0, 15.0, 20.0]
take_profit_options = [15.0, 20.0, 25.0, 30.0, 40.0]
rsi_buy_options = [30.0, 35.0, 40.0]
rsi_sell_options = [60.0, 65.0, 70.0]

print("Testing parameter combinations...")
print(f"Stop Loss: {stop_loss_options}")
print(f"Take Profit: {take_profit_options}")
print(f"RSI Buy: {rsi_buy_options}")
print(f"RSI Sell: {rsi_sell_options}\n")

results = []

total_tests = len(stop_loss_options) * len(take_profit_options) * len(rsi_buy_options) * len(rsi_sell_options)
test_num = 0

for stop_loss in stop_loss_options:
    for take_profit in take_profit_options:
        for rsi_buy in rsi_buy_options:
            for rsi_sell in rsi_sell_options:
                test_num += 1
                
                # Skip if reward/risk ratio is too low
                if take_profit / stop_loss < 1.2:
                    continue
                
                trading_config = TradingConfig(
                    max_position_size=4,
                    max_daily_loss=2000.0,
                    max_daily_trades=20,
                    initial_capital=100000.0,
                    stop_loss_ticks=stop_loss,
                    take_profit_ticks=take_profit,
                    tick_size=0.25,
                    tick_value=12.5,
                    commission_per_contract=2.4,
                    contract_multiplier=50.0
                )
                
                strategy = RsiMacdSentimentStrategy(
                    rsi_buy=rsi_buy,
                    rsi_sell=rsi_sell
                )
                
                try:
                    engine = BacktestingEngine([strategy], trading_config, backtest_config)
                    result = engine.run(df)
                    
                    metrics = result.metrics
                    
                    results.append({
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'rsi_buy': rsi_buy,
                        'rsi_sell': rsi_sell,
                        'trades': len(result.trades),
                        'win_rate': metrics.get('win_rate', 0),
                        'total_return': metrics.get('total_return', 0),
                        'sharpe': metrics.get('sharpe', 0),
                        'max_dd': metrics.get('max_drawdown', 0),
                        'profit_factor': metrics.get('profit_factor', 0),
                        'score': metrics.get('sharpe', 0) * (1 - abs(metrics.get('max_drawdown', 1))) * (1 if len(result.trades) > 20 else 0.5)
                    })
                    
                    if test_num % 10 == 0:
                        print(f"   Tested {test_num}/{total_tests} combinations...")
                
                except Exception as e:
                    print(f"   Error with SL={stop_loss}, TP={take_profit}: {e}")

print(f"\n✓ Completed {len(results)} valid tests\n")

# Sort by composite score
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('score', ascending=False)

print("="*80)
print("TOP 10 PARAMETER COMBINATIONS")
print("="*80 + "\n")

print(f"{'Rank':<5} {'SL':<6} {'TP':<6} {'RSI':<10} {'Trades':<8} {'Win%':<8} {'Return%':<10} {'Sharpe':<8} {'MaxDD%':<8}")
print("-"*80)

for idx, row in results_df.head(10).iterrows():
    print(f"{results_df.index.get_loc(idx)+1:<5} "
          f"{row['stop_loss']:<6.1f} "
          f"{row['take_profit']:<6.1f} "
          f"{row['rsi_buy']:.0f}/{row['rsi_sell']:.0f}    "
          f"{row['trades']:<8.0f} "
          f"{row['win_rate']*100:<8.1f} "
          f"{row['total_return']*100:<10.2f} "
          f"{row['sharpe']:<8.2f} "
          f"{abs(row['max_dd'])*100:<8.2f}")

# Best configuration
best = results_df.iloc[0]

print("\n" + "="*80)
print("RECOMMENDED CONFIGURATION")
print("="*80 + "\n")

print("Update your config.yaml with these values:")
print(f"""
trading:
  stop_loss_ticks: {best['stop_loss']:.1f}      # Changed from 10.0
  take_profit_ticks: {best['take_profit']:.1f}   # Changed from 20.0

# For strategy parameters, update in code or optimization:
# RSI Buy Threshold: {best['rsi_buy']:.1f}
# RSI Sell Threshold: {best['rsi_sell']:.1f}
""")

print("Expected Performance:")
print(f"  • Win Rate: {best['win_rate']*100:.1f}% (vs current ~0.5%)")
print(f"  • Total Return: {best['total_return']*100:.2f}%")
print(f"  • Sharpe Ratio: {best['sharpe']:.2f}")
print(f"  • Max Drawdown: {abs(best['max_dd'])*100:.2f}%")
print(f"  • Profit Factor: {best['profit_factor']:.2f}")
print(f"  • Total Trades: {best['trades']:.0f}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80 + "\n")

if best['win_rate'] < 0.5:
    print("⚠️  Note: Even the best configuration has <50% win rate.")
    print("   This suggests the strategy logic itself needs improvement.")
    print("   Recommendation: Use the enhanced regime-based strategy instead.")
    print("   Run: python3 scripts/performance_analyzer.py")
else:
    print("✓ Good improvement! Win rate >50%")
    print("  Consider running full optimization for even better results:")
    print("  python3 scripts/advanced_optimizer.py --trials 100")

# Save results
output_file = "reports/parameter_sweep_results.csv"
Path("reports").mkdir(exist_ok=True)
results_df.to_csv(output_file, index=False)
print(f"\n✓ Full results saved to: {output_file}")

print("\n" + "="*80 + "\n")
