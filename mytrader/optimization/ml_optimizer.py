"""Machine learning-based strategy optimization using Optuna."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from ..strategies.base import BaseStrategy
from ..strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
from ..utils.logger import logger

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("Optuna not installed. ML optimization will not be available.")


@dataclass
class MLOptimizationResult:
    best_params: Dict[str, Any]
    best_score: float
    study: Any = None
    optimization_history: List[Dict] = None


class MLParameterOptimizer:
    """Advanced parameter optimization using Optuna's TPE sampler."""
    
    def __init__(
        self, 
        strategies: Iterable[BaseStrategy],
        n_trials: int = 100,
        n_jobs: int = 1
    ) -> None:
        if not HAS_OPTUNA:
            raise ImportError("Optuna is required for ML optimization. Install with: pip install optuna")
        
        self.strategies = list(strategies)
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.best_params = {}

    def optimize(
        self, 
        data: pd.DataFrame,
        param_space: Dict[str, tuple],
        metric: str = "sharpe"
    ) -> MLOptimizationResult:
        """
        Optimize strategy parameters using Optuna.
        
        Args:
            data: Historical data with features
            param_space: Dictionary mapping parameter names to (min, max, type) tuples
                        e.g., {"rsi_buy": (20, 40, "int"), "sentiment_threshold": (0.3, 0.7, "float")}
            metric: Optimization metric ("sharpe", "sortino", "profit_factor", "total_return")
        """
        strategy = next((s for s in self.strategies if isinstance(s, RsiMacdSentimentStrategy)), None)
        if strategy is None:
            logger.warning("No compatible strategy found for optimization")
            return MLOptimizationResult(best_params={}, best_score=-np.inf)

        def objective(trial: optuna.Trial) -> float:
            """Objective function for Optuna optimization."""
            # Sample parameters
            params = {}
            for param_name, (min_val, max_val, param_type) in param_space.items():
                if param_type == "int":
                    params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                elif param_type == "float":
                    params[param_name] = trial.suggest_float(param_name, min_val, max_val)
                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, min_val)  # min_val is list of choices
            
            # Apply parameters to strategy
            original = strategy.__dict__.copy()
            try:
                for key, value in params.items():
                    if hasattr(strategy, key):
                        setattr(strategy, key, value)
                
                # Evaluate strategy performance
                score = self._evaluate_strategy(strategy, data, metric)
                return score
            finally:
                # Restore original parameters
                for key, value in original.items():
                    setattr(strategy, key, value)

        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20)
        )
        
        # Optimize
        logger.info("Starting Optuna optimization with %d trials...", self.n_trials)
        study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs, show_progress_bar=True)
        
        # Get results
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info("Optimization complete. Best score: %.4f, Best params: %s", best_score, best_params)
        
        # Apply best parameters
        self._apply_params(best_params)
        
        # Extract history
        history = [
            {
                "trial": t.number,
                "score": t.value,
                "params": t.params,
                "state": str(t.state)
            }
            for t in study.trials
        ]
        
        return MLOptimizationResult(
            best_params=best_params,
            best_score=best_score,
            study=study,
            optimization_history=history
        )

    def _evaluate_strategy(self, strategy: BaseStrategy, data: pd.DataFrame, metric: str) -> float:
        """Evaluate strategy performance on data."""
        if len(data) < 50:
            return -np.inf
        
        returns = data["close"].pct_change().dropna()
        signals = []
        
        # Generate signals
        for i in range(50, len(data)):
            window = data.iloc[:i]
            signal = strategy.generate(window)
            signals.append(signal.action)
        
        # Calculate returns based on signals
        strategy_returns = []
        position = 0
        
        for i, signal in enumerate(signals):
            if signal == "BUY" and position <= 0:
                position = 1
            elif signal == "SELL" and position >= 0:
                position = -1
            elif signal == "HOLD":
                pass
            
            # Calculate return for this period
            if position != 0 and i + 50 < len(returns):
                strategy_returns.append(returns.iloc[i + 50] * position)
            else:
                strategy_returns.append(0)
        
        if not strategy_returns:
            return -np.inf
        
        strategy_returns = pd.Series(strategy_returns)
        
        # Calculate requested metric
        if metric == "sharpe":
            mean_ret = strategy_returns.mean()
            std_ret = strategy_returns.std()
            return float(mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else -np.inf
        elif metric == "sortino":
            mean_ret = strategy_returns.mean()
            downside = strategy_returns[strategy_returns < 0]
            downside_std = downside.std() if len(downside) > 0 else strategy_returns.std()
            return float(mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else -np.inf
        elif metric == "profit_factor":
            gains = strategy_returns[strategy_returns > 0].sum()
            losses = abs(strategy_returns[strategy_returns < 0].sum())
            return float(gains / losses) if losses > 0 else -np.inf
        elif metric == "total_return":
            return float((1 + strategy_returns).prod() - 1)
        else:
            return -np.inf

    def _apply_params(self, params: Dict[str, Any]) -> None:
        """Apply optimized parameters to strategies."""
        for strategy in self.strategies:
            for key, value in params.items():
                if hasattr(strategy, key):
                    setattr(strategy, key, value)
                    logger.debug("Applied %s=%s to %s", key, value, strategy.name)


class WalkForwardOptimizer:
    """Walk-forward optimization for strategy parameters."""
    
    def __init__(
        self,
        strategies: Iterable[BaseStrategy],
        train_window: int = 5000,
        test_window: int = 1000,
        step_size: int = 500
    ) -> None:
        self.strategies = list(strategies)
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size

    def optimize(
        self,
        data: pd.DataFrame,
        param_space: Dict[str, tuple]
    ) -> List[MLOptimizationResult]:
        """
        Perform walk-forward optimization.
        
        Returns a list of optimization results for each walk-forward period.
        """
        results = []
        
        total_length = len(data)
        n_splits = (total_length - self.train_window - self.test_window) // self.step_size
        
        logger.info("Starting walk-forward optimization with %d splits", n_splits)
        
        for i in range(n_splits):
            start_train = i * self.step_size
            end_train = start_train + self.train_window
            end_test = end_train + self.test_window
            
            if end_test > total_length:
                break
            
            train_data = data.iloc[start_train:end_train]
            test_data = data.iloc[end_train:end_test]
            
            logger.info("Walk-forward period %d: train[%d:%d] test[%d:%d]",
                       i + 1, start_train, end_train, end_train, end_test)
            
            # Optimize on training period
            optimizer = MLParameterOptimizer(self.strategies, n_trials=50)
            result = optimizer.optimize(train_data, param_space)
            
            # Evaluate on test period
            test_score = optimizer._evaluate_strategy(
                self.strategies[0], test_data, "sharpe"
            )
            
            result.best_score = test_score  # Replace with out-of-sample score
            results.append(result)
            
            logger.info("Period %d test score: %.4f", i + 1, test_score)
        
        return results
