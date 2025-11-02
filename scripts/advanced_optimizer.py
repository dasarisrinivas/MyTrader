"""
Advanced Bayesian Optimization for Trading Strategy Parameters
Uses Optuna for hyperparameter tuning with multiple objectives
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    print("Installing optuna...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

from mytrader.backtesting.engine import BacktestingEngine
from mytrader.config import BacktestConfig, TradingConfig
from mytrader.strategies.enhanced_regime_strategy import EnhancedRegimeStrategy
from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
from mytrader.strategies.momentum_reversal import MomentumReversalStrategy
from mytrader.utils.logger import configure_logging, logger


class AdvancedStrategyOptimizer:
    """
    Multi-objective optimization for trading strategies
    Optimizes for: Sharpe ratio, Sortino ratio, Max drawdown, Profit factor
    """
    
    def __init__(
        self,
        data_path: str,
        strategy_type: str = "enhanced",
        optimization_metric: str = "risk_adjusted_return",
        n_trials: int = 100,
        n_jobs: int = 1
    ):
        self.data_path = Path(data_path)
        self.strategy_type = strategy_type
        self.optimization_metric = optimization_metric
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        
        # Load data
        logger.info(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path, parse_dates=["timestamp"], index_col="timestamp")
        logger.info(f"Loaded {len(self.data)} bars from {self.data.index[0]} to {self.data.index[-1]}")
        
        # Split data for train/validation
        split_idx = int(len(self.data) * 0.7)
        self.train_data = self.data.iloc[:split_idx]
        self.validation_data = self.data.iloc[split_idx:]
        
        logger.info(f"Train: {len(self.train_data)} bars, Validation: {len(self.validation_data)} bars")
        
        # Base configurations
        self.trading_config = TradingConfig(
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
        
        self.backtest_config = BacktestConfig(
            initial_capital=100000.0,
            slippage=0.25,
            risk_free_rate=0.02
        )
        
        self.best_params = None
        self.best_value = None
        self.best_strategy = None
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization."""
        
        # Sample parameters based on strategy type
        if self.strategy_type == "enhanced":
            strategy = self._sample_enhanced_params(trial)
        elif self.strategy_type == "rsi_macd":
            strategy = self._sample_rsi_macd_params(trial)
        else:
            strategy = self._sample_momentum_params(trial)
        
        # Sample risk management parameters
        stop_loss_ticks = trial.suggest_float("stop_loss_ticks", 5.0, 25.0)
        take_profit_ticks = trial.suggest_float("take_profit_ticks", 10.0, 50.0)
        max_position_size = trial.suggest_int("max_position_size", 1, 6)
        
        # Update configs
        trading_config = TradingConfig(
            max_position_size=max_position_size,
            max_daily_loss=self.trading_config.max_daily_loss,
            max_daily_trades=self.trading_config.max_daily_trades,
            initial_capital=self.trading_config.initial_capital,
            stop_loss_ticks=stop_loss_ticks,
            take_profit_ticks=take_profit_ticks,
            tick_size=self.trading_config.tick_size,
            tick_value=self.trading_config.tick_value,
            commission_per_contract=self.trading_config.commission_per_contract,
            contract_multiplier=self.trading_config.contract_multiplier
        )
        
        # Run backtest on training data
        try:
            engine = BacktestingEngine([strategy], trading_config, self.backtest_config)
            result = engine.run(self.train_data)
            
            metrics = result.metrics
            
            # Check if we have any trades
            if len(result.trades) < 10:
                return -999.0  # Heavily penalize strategies with too few trades
            
            # Calculate optimization metric
            score = self._calculate_optimization_score(metrics)
            
            # Report intermediate values for pruning
            trial.report(score, trial.number)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return score
            
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return -999.0
    
    def _sample_enhanced_params(self, trial: optuna.Trial) -> EnhancedRegimeStrategy:
        """Sample parameters for enhanced regime strategy."""
        return EnhancedRegimeStrategy(
            rsi_oversold_trending=trial.suggest_float("rsi_oversold_trending", 25.0, 45.0),
            rsi_overbought_trending=trial.suggest_float("rsi_overbought_trending", 55.0, 75.0),
            rsi_oversold_ranging=trial.suggest_float("rsi_oversold_ranging", 20.0, 40.0),
            rsi_overbought_ranging=trial.suggest_float("rsi_overbought_ranging", 60.0, 80.0),
            adx_trend_threshold=trial.suggest_float("adx_trend_threshold", 20.0, 35.0),
            adx_strong_trend=trial.suggest_float("adx_strong_trend", 35.0, 50.0),
            atr_percentile_low=trial.suggest_float("atr_percentile_low", 10.0, 30.0),
            atr_percentile_high=trial.suggest_float("atr_percentile_high", 70.0, 90.0),
            volume_ma_multiplier=trial.suggest_float("volume_ma_multiplier", 0.8, 1.5),
            bb_extreme_lower=trial.suggest_float("bb_extreme_lower", 0.0, 0.15),
            bb_extreme_upper=trial.suggest_float("bb_extreme_upper", 0.85, 1.0),
            min_confirmations_trend=trial.suggest_int("min_confirmations_trend", 2, 4),
            min_confirmations_range=trial.suggest_int("min_confirmations_range", 2, 3),
        )
    
    def _sample_rsi_macd_params(self, trial: optuna.Trial) -> RsiMacdSentimentStrategy:
        """Sample parameters for RSI/MACD strategy."""
        return RsiMacdSentimentStrategy(
            rsi_buy=trial.suggest_float("rsi_buy", 25.0, 45.0),
            rsi_sell=trial.suggest_float("rsi_sell", 55.0, 75.0),
            sentiment_buy=trial.suggest_float("sentiment_buy", -0.5, 0.0),
            sentiment_sell=trial.suggest_float("sentiment_sell", 0.0, 0.5),
            use_macd_crossover=trial.suggest_categorical("use_macd_crossover", [True, False])
        )
    
    def _sample_momentum_params(self, trial: optuna.Trial) -> MomentumReversalStrategy:
        """Sample parameters for momentum strategy."""
        return MomentumReversalStrategy(
            lookback=trial.suggest_int("lookback", 10, 40),
            threshold=trial.suggest_float("threshold", 0.005, 0.025)
        )
    
    def _calculate_optimization_score(self, metrics: Dict) -> float:
        """
        Calculate optimization score based on multiple metrics.
        Combines Sharpe, Sortino, Max DD, Profit Factor, Win Rate
        """
        sharpe = metrics.get("sharpe", 0)
        sortino = metrics.get("sortino", 0)
        max_dd = abs(metrics.get("max_drawdown", 1.0))
        profit_factor = metrics.get("profit_factor", 0)
        win_rate = metrics.get("win_rate", 0)
        total_trades = metrics.get("total_trades", 0)
        
        # Penalize low trade count
        if total_trades < 20:
            trade_penalty = 0.5
        elif total_trades < 50:
            trade_penalty = 0.8
        else:
            trade_penalty = 1.0
        
        if self.optimization_metric == "sharpe":
            return sharpe * trade_penalty
        
        elif self.optimization_metric == "sortino":
            return sortino * trade_penalty
        
        elif self.optimization_metric == "risk_adjusted_return":
            # Custom composite score
            # Maximize: Sharpe, Sortino, Profit Factor, Win Rate
            # Minimize: Max Drawdown
            
            sharpe_score = max(0, sharpe) * 0.3
            sortino_score = max(0, sortino) * 0.25
            dd_score = (1 - min(1, max_dd * 2)) * 0.25  # Penalize DD > 50%
            pf_score = min(1, profit_factor / 2) * 0.1
            wr_score = win_rate * 0.1
            
            composite = (sharpe_score + sortino_score + dd_score + pf_score + wr_score) * trade_penalty
            
            return composite
        
        elif self.optimization_metric == "calmar":
            calmar = metrics.get("calmar_ratio", 0)
            return calmar * trade_penalty
        
        else:
            return sharpe * trade_penalty
    
    def optimize(self) -> Dict:
        """Run Bayesian optimization."""
        logger.info(f"Starting optimization with {self.n_trials} trials")
        logger.info(f"Strategy type: {self.strategy_type}")
        logger.info(f"Optimization metric: {self.optimization_metric}")
        
        # Create study with pruner for faster optimization
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        # Get best parameters
        self.best_params = study.best_params
        self.best_value = study.best_value
        
        logger.info(f"Optimization complete!")
        logger.info(f"Best value: {self.best_value:.4f}")
        logger.info(f"Best parameters: {json.dumps(self.best_params, indent=2)}")
        
        # Validate on test set
        validation_results = self._validate_best_params()
        
        return {
            "best_params": self.best_params,
            "best_train_score": self.best_value,
            "validation_results": validation_results,
            "optimization_history": self._get_optimization_history(study),
            "timestamp": datetime.now().isoformat()
        }
    
    def _validate_best_params(self) -> Dict:
        """Validate best parameters on validation set."""
        logger.info("Validating best parameters on holdout set...")
        
        # Reconstruct best strategy
        if self.strategy_type == "enhanced":
            strategy = EnhancedRegimeStrategy(
                rsi_oversold_trending=self.best_params.get("rsi_oversold_trending", 35.0),
                rsi_overbought_trending=self.best_params.get("rsi_overbought_trending", 65.0),
                rsi_oversold_ranging=self.best_params.get("rsi_oversold_ranging", 30.0),
                rsi_overbought_ranging=self.best_params.get("rsi_overbought_ranging", 70.0),
                adx_trend_threshold=self.best_params.get("adx_trend_threshold", 25.0),
                adx_strong_trend=self.best_params.get("adx_strong_trend", 40.0),
                atr_percentile_low=self.best_params.get("atr_percentile_low", 20.0),
                atr_percentile_high=self.best_params.get("atr_percentile_high", 80.0),
                volume_ma_multiplier=self.best_params.get("volume_ma_multiplier", 1.2),
                bb_extreme_lower=self.best_params.get("bb_extreme_lower", 0.05),
                bb_extreme_upper=self.best_params.get("bb_extreme_upper", 0.95),
                min_confirmations_trend=self.best_params.get("min_confirmations_trend", 3),
                min_confirmations_range=self.best_params.get("min_confirmations_range", 2),
            )
        elif self.strategy_type == "rsi_macd":
            strategy = RsiMacdSentimentStrategy(
                rsi_buy=self.best_params.get("rsi_buy", 40.0),
                rsi_sell=self.best_params.get("rsi_sell", 60.0),
                sentiment_buy=self.best_params.get("sentiment_buy", -0.3),
                sentiment_sell=self.best_params.get("sentiment_sell", 0.3),
                use_macd_crossover=self.best_params.get("use_macd_crossover", False)
            )
        else:
            strategy = MomentumReversalStrategy(
                lookback=self.best_params.get("lookback", 20),
                threshold=self.best_params.get("threshold", 0.01)
            )
        
        # Update trading config with best params
        trading_config = TradingConfig(
            max_position_size=self.best_params.get("max_position_size", 4),
            max_daily_loss=self.trading_config.max_daily_loss,
            max_daily_trades=self.trading_config.max_daily_trades,
            initial_capital=self.trading_config.initial_capital,
            stop_loss_ticks=self.best_params.get("stop_loss_ticks", 10.0),
            take_profit_ticks=self.best_params.get("take_profit_ticks", 20.0),
            tick_size=self.trading_config.tick_size,
            tick_value=self.trading_config.tick_value,
            commission_per_contract=self.trading_config.commission_per_contract,
            contract_multiplier=self.trading_config.contract_multiplier
        )
        
        # Run on validation data
        engine = BacktestingEngine([strategy], trading_config, self.backtest_config)
        result = engine.run(self.validation_data)
        
        logger.info("Validation results:")
        for key, value in result.metrics.items():
            logger.info(f"  {key}: {value}")
        
        return {
            "metrics": result.metrics,
            "total_trades": len(result.trades),
            "final_equity": float(result.equity_curve.iloc[-1]) if len(result.equity_curve) > 0 else 100000.0
        }
    
    def _get_optimization_history(self, study: optuna.Study) -> List[Dict]:
        """Extract optimization history for analysis."""
        history = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    "trial_number": trial.number,
                    "value": trial.value,
                    "params": trial.params
                })
        return history
    
    def export_results(self, output_path: str = "reports/advanced_optimization.json"):
        """Export optimization results to file."""
        results = {
            "strategy_type": self.strategy_type,
            "optimization_metric": self.optimization_metric,
            "best_params": self.best_params,
            "best_train_score": self.best_value,
            "n_trials": self.n_trials,
            "timestamp": datetime.now().isoformat()
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results exported to {output_file}")


def main():
    """Main optimization runner."""
    configure_logging(level="INFO")
    
    import argparse
    parser = argparse.ArgumentParser(description="Advanced Strategy Optimizer")
    parser.add_argument("--data", type=str, default="data/es_synthetic_with_sentiment.csv",
                       help="Path to historical data CSV")
    parser.add_argument("--strategy", type=str, default="enhanced",
                       choices=["enhanced", "rsi_macd", "momentum"],
                       help="Strategy type to optimize")
    parser.add_argument("--metric", type=str, default="risk_adjusted_return",
                       choices=["sharpe", "sortino", "risk_adjusted_return", "calmar"],
                       help="Optimization metric")
    parser.add_argument("--trials", type=int, default=100,
                       help="Number of optimization trials")
    parser.add_argument("--jobs", type=int, default=1,
                       help="Number of parallel jobs")
    parser.add_argument("--output", type=str, default="reports/advanced_optimization.json",
                       help="Output file path")
    
    args = parser.parse_args()
    
    # Run optimization
    optimizer = AdvancedStrategyOptimizer(
        data_path=args.data,
        strategy_type=args.strategy,
        optimization_metric=args.metric,
        n_trials=args.trials,
        n_jobs=args.jobs
    )
    
    results = optimizer.optimize()
    
    # Export results
    optimizer.export_results(args.output)
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nBest Training Score: {results['best_train_score']:.4f}")
    print(f"\nValidation Metrics:")
    for key, value in results['validation_results']['metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(f"\nResults saved to: {args.output}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
