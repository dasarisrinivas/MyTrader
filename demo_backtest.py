#!/usr/bin/env python3
"""Comprehensive backtest demonstration with diagnostics."""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from mytrader.backtesting.engine import BacktestingEngine
from mytrader.config import BacktestConfig, TradingConfig
from mytrader.features.feature_engineer import engineer_features
from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
from mytrader.strategies.momentum_reversal import MomentumReversalStrategy
from mytrader.strategies.engine import StrategyEngine
from mytrader.utils.logger import configure_logging, logger


def main():
    configure_logging(level="INFO")
    
    print("\n" + "=" * 80)
    print("MyTrader Backtest Demonstration")
    print("=" * 80 + "\n")
    
    # Load data
    data_path = Path("data/es_synthetic_with_sentiment.csv")
    print(f"📁 Loading data from: {data_path}")
    df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")
    print(f"   Total bars: {len(df)}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Columns: {', '.join(df.columns)}\n")
    
    # Engineer features
    print("🔧 Engineering features...")
    price_cols = ["open", "high", "low", "close", "volume"]
    sentiment_cols = [c for c in df.columns if "sentiment" in c]
    
    price_df = df[price_cols]
    sentiment_df = df[sentiment_cols] if sentiment_cols else None
    
    features = engineer_features(price_df, sentiment_df)
    print(f"   Features generated: {len(features.columns)}")
    print(f"   Feature rows: {len(features)} (after dropna)")
    print(f"   Sample indicators: {', '.join(list(features.columns)[:15])}\n")
    
    # Check strategy signals
    print("📊 Testing strategy signal generation...")
    strategies = [
        RsiMacdSentimentStrategy(rsi_buy=30, rsi_sell=70, sentiment_buy=0.0, sentiment_sell=0.0),
        MomentumReversalStrategy(lookback=20, threshold=0.01)
    ]
    
    engine = StrategyEngine(strategies)
    returns = features["close"].pct_change().dropna()
    
    # Check signals on last 10 bars
    signals = []
    for i in range(max(0, len(features) - 10), len(features)):
        window = features.iloc[:i+1]
        window_returns = returns.iloc[:i+1] if len(returns) >= i+1 else returns
        signal = engine.evaluate(window, window_returns)
        signals.append({
            'timestamp': window.index[-1],
            'action': signal.action,
            'confidence': signal.confidence
        })
    
    print("   Last 10 signals:")
    for sig in signals:
        print(f"     {sig['timestamp']}: {sig['action']:5s} (conf={sig['confidence']:.2f})")
    print()
    
    # Setup backtest configuration
    trading_config = TradingConfig(
        initial_capital=100000.0,
        max_position_size=4,
        max_daily_loss=2000.0,
        max_daily_trades=20,
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
    
    # Run backtest
    print("🚀 Running backtest...")
    backtest_engine = BacktestingEngine(strategies, trading_config, backtest_config)
    
    # Use the original dataframe for backtest
    result = backtest_engine.run(df)
    
    # Display results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    
    metrics = result.metrics
    
    print("\n📈 Performance Metrics:")
    print(f"   Total Return:      {metrics.get('total_return', 0) * 100:>8.2f}%")
    print(f"   CAGR:              {metrics.get('cagr', 0) * 100:>8.2f}%")
    print(f"   Sharpe Ratio:      {metrics.get('sharpe', 0):>8.2f}")
    print(f"   Sortino Ratio:     {metrics.get('sortino', 0):>8.2f}")
    print(f"   Max Drawdown:      {metrics.get('max_drawdown', 0) * 100:>8.2f}%")
    print(f"   Profit Factor:     {metrics.get('profit_factor', 0):>8.2f}")
    print(f"   Calmar Ratio:      {metrics.get('calmar_ratio', 0):>8.2f}")
    print(f"   Volatility:        {metrics.get('volatility', 0) * 100:>8.2f}%")
    
    if 'total_trades' in metrics:
        print("\n📊 Trade Statistics:")
        print(f"   Total Trades:      {metrics.get('total_trades', 0):>8}")
        print(f"   Winning Trades:    {metrics.get('winning_trades', 0):>8}")
        print(f"   Losing Trades:     {metrics.get('losing_trades', 0):>8}")
        print(f"   Win Rate:          {metrics.get('win_rate', 0) * 100:>8.2f}%")
        print(f"   Avg Win:           ${metrics.get('avg_win', 0):>8.2f}")
        print(f"   Avg Loss:          ${metrics.get('avg_loss', 0):>8.2f}")
        print(f"   Largest Win:       ${metrics.get('largest_win', 0):>8.2f}")
        print(f"   Largest Loss:      ${metrics.get('largest_loss', 0):>8.2f}")
        print(f"   Avg Trade:         ${metrics.get('avg_trade', 0):>8.2f}")
        print(f"   Expectancy:        ${metrics.get('expectancy', 0):>8.2f}")
    
    print(f"\n💰 Final Equity:      ${result.equity_curve.iloc[-1] if not result.equity_curve.empty else 100000:,.2f}")
    print(f"📝 Trade Count:        {len(result.trades)}")
    
    if result.trades:
        print("\n📋 Sample Trades (first 5):")
        for i, trade in enumerate(result.trades[:5]):
            print(f"   {i+1}. {trade.get('timestamp', 'N/A')}: {trade.get('action', 'N/A'):6s} "
                  f"{trade.get('qty', 0):2d} @ ${trade.get('price', 0):.2f} "
                  f"PnL: ${trade.get('realized', 0):.2f if 'realized' in trade else 0.0}")
    else:
        print("\n⚠️  No trades were generated!")
        print("   This could be due to:")
        print("     - Strategy parameters too conservative")
        print("     - Insufficient data length")
        print("     - Market conditions not meeting entry criteria")
        print("     - Risk limits preventing trades")
    
    print("\n" + "=" * 80)
    print("✅ Backtest completed successfully!")
    print("=" * 80 + "\n")
    
    # Export results
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    
    from mytrader.backtesting.performance import export_report
    json_path = output_dir / "demo_backtest.json"
    export_report(result.metrics, result.trades, json_path, format="json")
    print(f"📊 Results exported to: {json_path}\n")


if __name__ == "__main__":
    main()
