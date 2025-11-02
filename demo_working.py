#!/usr/bin/env python3
"""Working backtest example with trades."""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from mytrader.backtesting.engine import BacktestingEngine
from mytrader.config import BacktestConfig, TradingConfig
from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
from mytrader.strategies.momentum_reversal import MomentumReversalStrategy
from mytrader.features.feature_engineer import engineer_features
from mytrader.utils.logger import configure_logging, logger


def generate_trending_data(n_bars=1000):
    """Generate synthetic trending market data."""
    np.random.seed(42)
    
    # Create trending price with noise
    trend = np.linspace(4700, 4800, n_bars)
    noise = np.random.normal(0, 5, n_bars).cumsum() * 0.5
    close = trend + noise
    
    # Generate OHLC
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
    
    return df


def main():
    configure_logging(level="INFO")
    
    print("\n" + "=" * 80)
    print("MyTrader - Working Backtest Example")
    print("=" * 80 + "\n")
    
    # Generate synthetic data
    print("📈 Generating synthetic trending market data...")
    df = generate_trending_data(n_bars=1000)
    print(f"   Generated {len(df)} bars")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Engineer features
    print("\n🔧 Engineering technical features...")
    df_with_features = engineer_features(df)
    print(f"   Added {len(df_with_features.columns) - len(df.columns)} technical indicators")
    print(f"   Total features: {len(df_with_features.columns)}")
    
    # Setup strategies with relaxed parameters
    print("⚙️  Setting up strategies...")
    strategies = [
        # More aggressive RSI parameters
        RsiMacdSentimentStrategy(
            rsi_buy=40,  # More liberal buy threshold
            rsi_sell=60,  # More liberal sell threshold
            sentiment_buy=-0.5,  # Very low sentiment threshold
            sentiment_sell=-0.5   # Very low sentiment threshold
        ),
        # More sensitive momentum strategy
        MomentumReversalStrategy(
            lookback=10,
            threshold=0.005  # Lower threshold
        )
    ]
    print(f"   Configured {len(strategies)} strategies\n")
    
    # Setup configuration
    trading_config = TradingConfig(
        initial_capital=100000.0,
        max_position_size=4,
        max_daily_loss=5000.0,  # Higher limit
        max_daily_trades=50,     # More trades allowed
        stop_loss_ticks=15.0,    # Wider stop
        take_profit_ticks=30.0,  # Wider target
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
    
    # Run backtest
    print("\n🚀 Running backtest...")
    engine = BacktestingEngine(strategies, trading_config, backtest_config)
    result = engine.run(df_with_features)
    
    # Display results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    
    metrics = result.metrics
    
    print("\n📈 Performance Metrics:")
    print(f"   Total Return:      {metrics.get('total_return', 0) * 100:>10.2f}%")
    print(f"   CAGR:              {metrics.get('cagr', 0) * 100:>10.2f}%")
    print(f"   Sharpe Ratio:      {metrics.get('sharpe', 0):>10.2f}")
    print(f"   Sortino Ratio:     {metrics.get('sortino', 0):>10.2f}")
    print(f"   Max Drawdown:      {metrics.get('max_drawdown', 0) * 100:>10.2f}%")
    print(f"   Avg Drawdown:      {metrics.get('avg_drawdown', 0) * 100:>10.2f}%")
    print(f"   Profit Factor:     {metrics.get('profit_factor', 0):>10.2f}")
    print(f"   Calmar Ratio:      {metrics.get('calmar_ratio', 0):>10.2f}")
    print(f"   Volatility:        {metrics.get('volatility', 0) * 100:>10.2f}%")
    
    if 'total_trades' in metrics and metrics['total_trades'] > 0:
        print("\n📊 Trade Statistics:")
        print(f"   Total Trades:      {metrics['total_trades']:>10}")
        print(f"   Winning Trades:    {metrics['winning_trades']:>10}")
        print(f"   Losing Trades:     {metrics['losing_trades']:>10}")
        print(f"   Win Rate:          {metrics['win_rate'] * 100:>10.2f}%")
        print(f"   Avg Win:           ${metrics['avg_win']:>10.2f}")
        print(f"   Avg Loss:          ${metrics['avg_loss']:>10.2f}")
        print(f"   Largest Win:       ${metrics['largest_win']:>10.2f}")
        print(f"   Largest Loss:      ${metrics['largest_loss']:>10.2f}")
        print(f"   Avg Trade:         ${metrics['avg_trade']:>10.2f}")
        print(f"   Expectancy:        ${metrics['expectancy']:>10.2f}")
        if 'avg_holding_hours' in metrics:
            print(f"   Avg Hold Time:     {metrics['avg_holding_hours']:>10.1f} hours")
    
    initial_capital = 100000.0
    final_equity = result.equity_curve.iloc[-1] if not result.equity_curve.empty else initial_capital
    pnl = final_equity - initial_capital
    
    print(f"\n💰 Performance Summary:")
    print(f"   Initial Capital:   ${initial_capital:>10,.2f}")
    print(f"   Final Equity:      ${final_equity:>10,.2f}")
    print(f"   Net P&L:           ${pnl:>10,.2f}")
    print(f"   Return:            {(pnl/initial_capital)*100:>10.2f}%")
    
    if result.trades:
        print(f"\n📋 Trade Log (showing first 10 and last 5):")
        print(f"   {'#':<4} {'Time':<20} {'Action':<6} {'Qty':<4} {'Price':<10} {'PnL':<12} {'Reason':<10}")
        print(f"   {'-'*80}")
        
        for i, trade in enumerate(result.trades[:10]):
            timestamp = trade.get('timestamp', 'N/A')[:19]
            action = trade.get('action', 'N/A')
            qty = trade.get('qty', 0)
            price = trade.get('price', 0)
            realized = trade.get('realized', 0.0)
            reason = trade.get('reason', '-')
            
            print(f"   {i+1:<4} {timestamp:<20} {action:<6} {qty:<4} ${price:<9.2f} ${realized:<11.2f} {reason:<10}")
        
        if len(result.trades) > 15:
            print(f"   ... ({len(result.trades) - 15} trades omitted) ...")
            
            for i, trade in enumerate(result.trades[-5:], start=len(result.trades)-4):
                timestamp = trade.get('timestamp', 'N/A')[:19]
                action = trade.get('action', 'N/A')
                qty = trade.get('qty', 0)
                price = trade.get('price', 0)
                realized = trade.get('realized', 0.0)
                reason = trade.get('reason', '-')
                
                print(f"   {i:<4} {timestamp:<20} {action:<6} {qty:<4} ${price:<9.2f} ${realized:<11.2f} {reason:<10}")
    
    print("\n" + "=" * 80)
    print("✅ Backtest completed successfully!")
    print("=" * 80 + "\n")
    
    # Save equity curve
    if not result.equity_curve.empty:
        output_path = Path("reports/working_example_equity.csv")
        output_path.parent.mkdir(exist_ok=True)
        result.equity_curve.to_csv(output_path)
        print(f"📊 Equity curve saved to: {output_path}")
    
    # Export full report
    from mytrader.backtesting.performance import export_report
    json_path = Path("reports/working_example.json")
    export_report(result.metrics, result.trades, json_path, format="json")
    print(f"📊 Full report saved to: {json_path}\n")


if __name__ == "__main__":
    main()
