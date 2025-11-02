#!/usr/bin/env python3
"""
Test strategies on October 2025 real ES futures data.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from mytrader.config import TradingConfig, BacktestConfig
from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
from mytrader.strategies.enhanced_regime_strategy import EnhancedRegimeStrategy
from mytrader.backtesting.engine import BacktestingEngine
from mytrader.features.feature_engineer import engineer_features

def run_october_backtest():
    """Run backtest on October 2025 data."""
    
    print("\n" + "="*70)
    print("🚀 BACKTESTING ON OCTOBER 2025 REAL ES FUTURES DATA")
    print("="*70 + "\n")
    
    # Load October 2025 data
    data_file = "data/es_october_2025.csv"
    print(f"📁 Loading data from: {data_file}")
    
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    
    print(f"   ✅ Loaded {len(df)} bars")
    print(f"   📅 Period: {df.index.min()} to {df.index.max()}")
    print(f"   💰 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}\n")
    
    # Configuration
    trading_config = TradingConfig(
        initial_capital=100000.0,
        commission_per_contract=1.5,
        stop_loss_ticks=15.0,
        take_profit_ticks=25.0,
        max_position_size=1
    )
    
    backtest_config = BacktestConfig(
        initial_capital=100000.0
    )
    
    # Test both strategies
    strategies_to_test = [
        ("Baseline RSI/MACD", RsiMacdSentimentStrategy()),
        ("Enhanced Regime", EnhancedRegimeStrategy())
    ]
    
    results = []
    
    for name, strategy in strategies_to_test:
        print(f"\n{'─'*70}")
        print(f"📊 Testing: {name}")
        print(f"{'─'*70}\n")
        
        engine = BacktestingEngine(
            strategies=[strategy],
            trading_config=trading_config,
            backtest_config=backtest_config
        )
        
        backtest_result = engine.run(df)
        
        if backtest_result and backtest_result.metrics:
            metrics = backtest_result.metrics
            
            total_return = metrics.get('total_return', 0) * 100
            sharpe = metrics.get('sharpe_ratio', 0)
            max_dd = metrics.get('max_drawdown', 0) * 100
            win_rate = metrics.get('win_rate', 0) * 100
            total_trades = metrics.get('total_trades', 0)
            profit_factor = metrics.get('profit_factor', 0)
            
            print(f"\n📈 Results for {name}:")
            print(f"   Total Trades:    {total_trades}")
            print(f"   Win Rate:        {win_rate:.2f}%")
            print(f"   Total Return:    {total_return:+.2f}%")
            print(f"   Sharpe Ratio:    {sharpe:.2f}")
            print(f"   Max Drawdown:    {max_dd:.2f}%")
            print(f"   Profit Factor:   {profit_factor:.2f}")
            
            results.append({
                'strategy': name,
                'trades': total_trades,
                'win_rate': win_rate,
                'return': total_return,
                'sharpe': sharpe,
                'max_dd': max_dd,
                'profit_factor': profit_factor
            })
    
    # Summary
    print("\n" + "="*70)
    print("📊 COMPARISON SUMMARY")
    print("="*70 + "\n")
    
    print(f"{'Strategy':<20} {'Trades':<10} {'Win Rate':<12} {'Return':<12} {'Sharpe':<10}")
    print("─" * 70)
    
    for r in results:
        print(f"{r['strategy']:<20} {r['trades']:<10} {r['win_rate']:>8.2f}%  {r['return']:>+8.2f}%  {r['sharpe']:>8.2f}")
    
    print("\n" + "="*70)
    
    # Compare to targets
    print("\n🎯 TARGET METRICS:")
    print("   Win Rate:        ≥60%")
    print("   Sharpe Ratio:    ≥1.5")
    print("   Max Drawdown:    ≤15%")
    print("   Profit Factor:   ≥1.3")
    
    print("\n💡 RECOMMENDATION:")
    best = max(results, key=lambda x: x['return'])
    
    if best['win_rate'] >= 60 and best['sharpe'] >= 1.5:
        print(f"   ✅ {best['strategy']} MEETS TARGETS! Ready for optimization.")
    elif best['return'] > 0:
        print(f"   ⚠️  {best['strategy']} is profitable but below targets.")
        print(f"   → Run optimization: python3 quickstart_optimization.py")
    else:
        print(f"   ❌ All strategies losing money on this period.")
        print(f"   → Try different parameters or data period")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    run_october_backtest()
