"""Adaptive optimization for strategy parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from ..strategies.base import BaseStrategy
from ..strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy


@dataclass
class OptimizationResult:
    best_params: Dict[str, float]
    best_score: float


class ParameterOptimizer:
    def __init__(self, strategies: Iterable[BaseStrategy]) -> None:
        self.strategies = list(strategies)

    def optimize(self, data: pd.DataFrame, param_grid: Dict[str, Iterable]) -> OptimizationResult:
        best_score = -np.inf
        best_params: Dict[str, float] = {}
        grid = ParameterGrid(param_grid)

        for params in grid:
            score = self._evaluate(data, params)
            if score > best_score:
                best_score = score
                best_params = dict(params)
        if best_params:
            self._apply(best_params)
        return OptimizationResult(best_params=best_params, best_score=float(best_score))

    def _evaluate(self, data: pd.DataFrame, params: Dict[str, float]) -> float:
        strategy = next((s for s in self.strategies if isinstance(s, RsiMacdSentimentStrategy)), None)
        if strategy is None:
            return -np.inf
        original = strategy.__dict__.copy()
        try:
            for key, value in params.items():
                if hasattr(strategy, key):
                    setattr(strategy, key, value)
            returns = data["close"].pct_change().dropna()
            return float(returns.mean() / returns.std()) if returns.std() > 0 else -np.inf
        finally:
            for key, value in original.items():
                setattr(strategy, key, value)

    def _apply(self, params: Dict[str, float]) -> None:
        for strategy in self.strategies:
            for key, value in params.items():
                if hasattr(strategy, key):
                    setattr(strategy, key, value)
