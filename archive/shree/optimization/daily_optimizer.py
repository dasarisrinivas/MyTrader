"""
Daily Parameter Optimization Module
Runs after market close to optimize strategy parameters.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from loguru import logger

from ..strategies.base import BaseStrategy
from ..optimization.optimizer import ParameterOptimizer, OptimizationResult


def daily_optimization(
    strategies: List[BaseStrategy],
    data: pd.DataFrame,
    parameter_grid: Dict,
    output_path: str = "data/optimized_params.json",
    window_length: int = 5000,
) -> OptimizationResult:
    """
    Run daily parameter optimization and save results.
    
    This function should be called after market close (e.g., 4:00 PM ET)
    to optimize strategy parameters based on recent performance.
    
    Args:
        strategies: List of strategies to optimize
        data: Historical OHLCV data with features
        parameter_grid: Grid of parameters to search
        output_path: Path to save optimized parameters JSON
        window_length: Number of bars to use for optimization
        
    Returns:
        OptimizationResult with best parameters and score
    """
    logger.info("=" * 60)
    logger.info("DAILY PARAMETER OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info(f"Data length: {len(data)} bars")
    logger.info(f"Window length: {window_length} bars")
    logger.info(f"Parameter grid: {parameter_grid}")
    
    # Use most recent data for optimization
    if len(data) > window_length:
        data = data.tail(window_length)
        logger.info(f"Using last {window_length} bars for optimization")
    
    # Run optimization
    logger.info("Running parameter optimization...")
    optimizer = ParameterOptimizer(strategies)
    result = optimizer.optimize(data, parameter_grid)
    
    logger.info("=" * 60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Best score: {result.best_score:.4f}")
    logger.info(f"Best parameters:")
    for key, value in result.best_params.items():
        logger.info(f"  {key}: {value}")
    
    # Save results to JSON
    output = {
        "timestamp": datetime.now().isoformat(),
        "optimization_window": window_length,
        "data_points": len(data),
        "best_score": float(result.best_score),
        "best_params": result.best_params,
        "parameter_grid": parameter_grid,
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"✅ Optimized parameters saved to {output_path}")
    logger.info("=" * 60)
    
    return result


def load_optimized_params(
    path: str = "data/optimized_params.json"
) -> Optional[Dict]:
    """
    Load optimized parameters from JSON file.
    
    Args:
        path: Path to optimized parameters JSON
        
    Returns:
        Dictionary with best parameters, or None if file doesn't exist
    """
    path = Path(path)
    
    if not path.exists():
        logger.warning(f"Optimized parameters file not found: {path}")
        logger.info("Run daily_optimization() first or use default parameters")
        return None
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        logger.info("✅ Loaded optimized parameters from startup")
        logger.info(f"   Optimization date: {data.get('timestamp', 'unknown')}")
        logger.info(f"   Best score: {data.get('best_score', 'unknown')}")
        logger.info(f"   Parameters: {data.get('best_params', {})}")
        
        return data.get('best_params', {})
    
    except Exception as e:
        logger.error(f"Failed to load optimized parameters: {e}")
        return None


def apply_optimized_params(
    strategies: List[BaseStrategy],
    params: Dict
) -> None:
    """
    Apply optimized parameters to strategies.
    
    Args:
        strategies: List of strategies to update
        params: Dictionary of parameter name -> value
    """
    if not params:
        logger.info("No parameters to apply")
        return
    
    logger.info("Applying optimized parameters to strategies...")
    
    for strategy in strategies:
        applied_count = 0
        for key, value in params.items():
            if hasattr(strategy, key):
                setattr(strategy, key, value)
                applied_count += 1
                logger.debug(f"  {strategy.__class__.__name__}.{key} = {value}")
        
        if applied_count > 0:
            logger.info(f"  ✅ Applied {applied_count} parameters to {strategy.__class__.__name__}")


def should_run_optimization(current_time: datetime, optimization_hour: int = 16) -> bool:
    """
    Check if it's time to run daily optimization.
    
    Args:
        current_time: Current timestamp
        optimization_hour: Hour to run optimization (default 4 PM ET)
        
    Returns:
        True if optimization should run
    """
    # Run optimization at specified hour (after market close)
    # Only run once per day
    return current_time.hour == optimization_hour and current_time.minute < 5
