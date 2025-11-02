#!/usr/bin/env python3
"""
ML-Based Strategy Optimization Script

This script uses Optuna to find optimal strategy parameters through:
1. Walk-forward analysis for out-of-sample validation
2. Multi-objective optimization (Sharpe, Sortino, Profit Factor)
3. Hyperparameter tuning with Bayesian optimization

Usage:
    python scripts/optimize_strategy.py --data data/es_historical.csv
    python scripts/optimize_strategy.py --data data/es_historical.csv --trials 200 --strategy rsi_macd
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import optuna

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mytrader.backtesting.engine import BacktestingEngine
from mytrader.config import BacktestConfig, TradingConfig
from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
from mytrader.strategies.momentum_reversal import MomentumReversalStrategy
from mytrader.features.feature_engineer import engineer_features
from mytrader.utils.logger import configure_logging, logger


def load_and_prepare_data(data_path):
    """Load and prepare data for optimization."""
    print(f"\n📂 Loading data from: {data_path}")
    
    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
    
    # Add sentiment if not present (use neutral)
    if 'sentiment_twitter' not in df.columns:
        df['sentiment_twitter'] = 0.0
    if 'sentiment_news' not in df.columns:
        df['sentiment_news'] = 0.0
    
    print(f"   Loaded {len(df)} bars")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    # Engineer features
    print("\n🔧 Engineering features...")
    df_features = engineer_features(df)
    print(f"   Generated {len(df_features.columns)} features")
    
    return df_features


def optimize_rsi_macd_strategy(data, n_trials=100, metric='sharpe'):
    """Optimize RSI+MACD+Sentiment strategy parameters."""
    
    print("\n" + "=" * 80)
    print("Optimizing RSI+MACD+Sentiment Strategy")
    print("=" * 80)
    
    trading_config = TradingConfig(
        initial_capital=100000.0,
        max_position_size=4,
        max_daily_loss=5000.0,
        max_daily_trades=20,
        stop_loss_ticks=20.0,
        take_profit_ticks=40.0,
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
    
    def objective(trial):
        """Objective function for Optuna."""
        
        # Suggest parameters
        rsi_buy = trial.suggest_float('rsi_buy', 20.0, 50.0)
        rsi_sell = trial.suggest_float('rsi_sell', 50.0, 80.0)
        sentiment_buy = trial.suggest_float('sentiment_buy', -1.0, 0.8)
        sentiment_sell = trial.suggest_float('sentiment_sell', -0.8, 1.0)
        
        # Create strategy
        strategy = RsiMacdSentimentStrategy(
            rsi_buy=rsi_buy,
            rsi_sell=rsi_sell,
            sentiment_buy=sentiment_buy,
            sentiment_sell=sentiment_sell
        )
        
        # Run backtest
        engine = BacktestingEngine([strategy], trading_config, backtest_config)
        result = engine.run(data)
        
        # Get metrics
        metrics = result.metrics
        total_trades = metrics.get('total_trades', 0)
        
        # Penalize if no trades or too few trades
        if total_trades < 5:
            return -999.0
        
        # Return optimization metric
        if metric == 'sharpe':
            return metrics.get('sharpe', -999.0)
        elif metric == 'sortino':
            return metrics.get('sortino', -999.0)
        elif metric == 'profit_factor':
            pf = metrics.get('profit_factor', 0)
            return pf if pf != float('inf') else 10.0
        elif metric == 'calmar':
            return metrics.get('calmar_ratio', -999.0)
        else:
            return metrics.get('total_return', -1.0)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
    )
    
    print(f"\n🔍 Running {n_trials} optimization trials...")
    print(f"   Optimization metric: {metric}")
    print(f"   This may take 10-30 minutes depending on data size...\n")
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Results
    print("\n" + "=" * 80)
    print("Optimization Results")
    print("=" * 80)
    
    print(f"\n✅ Best {metric}: {study.best_value:.4f}")
    print("\n📊 Best Parameters:")
    for param, value in study.best_params.items():
        print(f"   {param:20s}: {value:.4f}")
    
    # Run final backtest with best parameters
    print("\n🚀 Running backtest with optimal parameters...")
    best_strategy = RsiMacdSentimentStrategy(**study.best_params)
    engine = BacktestingEngine([best_strategy], trading_config, backtest_config)
    result = engine.run(data)
    
    # Display full metrics
    print("\n📈 Performance Metrics:")
    for key, value in result.metrics.items():
        if isinstance(value, (int, float)):
            print(f"   {key:20s}: {value:>12.4f}")
    
    return study, result


def optimize_momentum_strategy(data, n_trials=100, metric='sharpe'):
    """Optimize Momentum Reversal strategy parameters."""
    
    print("\n" + "=" * 80)
    print("Optimizing Momentum Reversal Strategy")
    print("=" * 80)
    
    trading_config = TradingConfig(
        initial_capital=100000.0,
        max_position_size=4,
        max_daily_loss=5000.0,
        max_daily_trades=20,
        stop_loss_ticks=20.0,
        take_profit_ticks=40.0,
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
    
    def objective(trial):
        """Objective function for Optuna."""
        
        # Suggest parameters
        lookback = trial.suggest_int('lookback', 5, 30)
        threshold = trial.suggest_float('threshold', 0.001, 0.02)
        
        # Create strategy
        strategy = MomentumReversalStrategy(
            lookback=lookback,
            threshold=threshold
        )
        
        # Run backtest
        engine = BacktestingEngine([strategy], trading_config, backtest_config)
        result = engine.run(data)
        
        # Get metrics
        metrics = result.metrics
        total_trades = metrics.get('total_trades', 0)
        
        # Penalize if no trades
        if total_trades < 5:
            return -999.0
        
        # Return optimization metric
        if metric == 'sharpe':
            return metrics.get('sharpe', -999.0)
        elif metric == 'sortino':
            return metrics.get('sortino', -999.0)
        elif metric == 'profit_factor':
            pf = metrics.get('profit_factor', 0)
            return pf if pf != float('inf') else 10.0
        else:
            return metrics.get('total_return', -1.0)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    print(f"\n🔍 Running {n_trials} optimization trials...")
    print(f"   Optimization metric: {metric}\n")
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Results
    print("\n" + "=" * 80)
    print("Optimization Results")
    print("=" * 80)
    
    print(f"\n✅ Best {metric}: {study.best_value:.4f}")
    print("\n📊 Best Parameters:")
    for param, value in study.best_params.items():
        print(f"   {param:20s}: {value}")
    
    return study


def walk_forward_optimization(data, n_splits=5, train_ratio=0.7):
    """Perform walk-forward optimization for out-of-sample validation."""
    
    print("\n" + "=" * 80)
    print("Walk-Forward Optimization")
    print("=" * 80)
    
    print(f"\n📊 Configuration:")
    print(f"   Number of splits: {n_splits}")
    print(f"   Train/test ratio: {train_ratio:.0%}/{1-train_ratio:.0%}")
    
    # Split data
    total_size = len(data)
    fold_size = total_size // n_splits
    
    results = []
    
    for i in range(n_splits):
        print(f"\n{'='*80}")
        print(f"Fold {i+1}/{n_splits}")
        print(f"{'='*80}")
        
        # Define train and test periods
        start_idx = i * fold_size
        train_end_idx = start_idx + int(fold_size * train_ratio)
        test_end_idx = start_idx + fold_size
        
        train_data = data.iloc[start_idx:train_end_idx]
        test_data = data.iloc[train_end_idx:test_end_idx]
        
        print(f"\n📅 Train period: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} bars)")
        print(f"📅 Test period:  {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} bars)")
        
        # Optimize on train data (fewer trials for speed)
        print("\n🔍 Optimizing on train data...")
        study, _ = optimize_rsi_macd_strategy(train_data, n_trials=50, metric='sharpe')
        
        # Test on out-of-sample data
        print("\n📈 Testing on out-of-sample data...")
        
        trading_config = TradingConfig(
            initial_capital=100000.0,
            max_position_size=4,
            max_daily_loss=5000.0,
            max_daily_trades=20,
            stop_loss_ticks=20.0,
            take_profit_ticks=40.0,
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
        
        best_strategy = RsiMacdSentimentStrategy(**study.best_params)
        engine = BacktestingEngine([best_strategy], trading_config, backtest_config)
        test_result = engine.run(test_data)
        
        # Store results
        fold_result = {
            'fold': i + 1,
            'train_sharpe': study.best_value,
            'test_sharpe': test_result.metrics.get('sharpe', 0),
            'test_return': test_result.metrics.get('total_return', 0),
            'test_trades': test_result.metrics.get('total_trades', 0),
            'parameters': study.best_params
        }
        results.append(fold_result)
        
        print(f"\n📊 Fold {i+1} Results:")
        print(f"   Train Sharpe: {fold_result['train_sharpe']:.4f}")
        print(f"   Test Sharpe:  {fold_result['test_sharpe']:.4f}")
        print(f"   Test Return:  {fold_result['test_return']*100:.2f}%")
        print(f"   Test Trades:  {fold_result['test_trades']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Walk-Forward Summary")
    print("=" * 80)
    
    avg_train_sharpe = np.mean([r['train_sharpe'] for r in results])
    avg_test_sharpe = np.mean([r['test_sharpe'] for r in results])
    avg_test_return = np.mean([r['test_return'] for r in results])
    
    print(f"\n📊 Average Metrics:")
    print(f"   Train Sharpe:  {avg_train_sharpe:.4f}")
    print(f"   Test Sharpe:   {avg_test_sharpe:.4f}")
    print(f"   Test Return:   {avg_test_return*100:.2f}%")
    print(f"   Sharpe Ratio:  {avg_test_sharpe/avg_train_sharpe:.2%} (test/train)")
    
    if avg_test_sharpe > 0.8 * avg_train_sharpe:
        print("\n✅ Good generalization - test performance is close to train")
    else:
        print("\n⚠️  Possible overfitting - test performance significantly lower")
    
    return results


def main():
    """Main optimization workflow."""
    configure_logging(level="INFO")
    
    print("\n" + "=" * 80)
    print("MyTrader - ML Strategy Optimization")
    print("=" * 80)
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Optimize trading strategy parameters')
    parser.add_argument('--data', required=True, help='Path to historical data CSV')
    parser.add_argument('--strategy', default='rsi_macd', choices=['rsi_macd', 'momentum'],
                       help='Strategy to optimize (default: rsi_macd)')
    parser.add_argument('--trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--metric', default='sharpe', choices=['sharpe', 'sortino', 'profit_factor', 'calmar'],
                       help='Optimization metric (default: sharpe)')
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward optimization')
    parser.add_argument('--splits', type=int, default=5, help='Number of walk-forward splits')
    parser.add_argument('--output', default='reports/optimization_results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    # Load data
    data = load_and_prepare_data(args.data)
    
    # Run optimization
    if args.walk_forward:
        results = walk_forward_optimization(data, n_splits=args.splits)
        
        # Save results
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_path}")
        
    else:
        if args.strategy == 'rsi_macd':
            study, result = optimize_rsi_macd_strategy(data, args.trials, args.metric)
        else:
            study = optimize_momentum_strategy(data, args.trials, args.metric)
        
        # Save optimization study
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True)
        
        study_data = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'optimization_metric': args.metric,
            'timestamp': datetime.now().isoformat()
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(study_data, f, indent=2)
        
        print(f"\n💾 Optimization results saved to: {output_path}")
    
    print("\n✅ Optimization complete!")
    print("\nNext steps:")
    print("1. Update your config.yaml with the optimal parameters")
    print("2. Run a final backtest: python main.py backtest --data " + args.data)
    print("3. Start paper trading: python main.py live --config config.yaml")


if __name__ == "__main__":
    main()
