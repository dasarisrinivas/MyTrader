"""Strategy ensemble engine."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal


class StrategyEngine:
    def __init__(self, strategies: Iterable[BaseStrategy], sharpe_window: int = 1000) -> None:
        self.strategies: List[BaseStrategy] = list(strategies)
        self.sharpe_window = sharpe_window
        self.threshold_adjustment: float = 0.0

    def evaluate(self, features: pd.DataFrame, returns: pd.Series | None = None) -> Signal:
        window = features.tail(self.sharpe_window)
        results = [(strategy, strategy.generate(window)) for strategy in self.strategies]
        signals = [sig for _, sig in results]
        action = self._aggregate(signals)
        matching_pairs = [(strategy, sig) for strategy, sig in results if sig.action == action]
        matching_conf = [sig.confidence for _, sig in matching_pairs]
        confidence = float(np.mean(matching_conf)) if matching_conf else 0.0

        # Start with metadata from the first contributing strategy for richer context
        if matching_pairs:
            primary_strategy, primary_signal = matching_pairs[0]
            metadata = dict(primary_signal.metadata)
            metadata["signal_source"] = primary_strategy.name
        else:
            metadata = {"signal_source": "none"}

        # Add per-strategy confidence transparency
        for strategy, sig in results:
            metadata[f"{strategy.name}_confidence"] = sig.confidence
            metadata[f"{strategy.name}_action"] = sig.action

        if returns is not None and len(returns) >= self.sharpe_window:
            sharpe = self._rolling_sharpe(returns[-self.sharpe_window :])
            metadata["rolling_sharpe"] = sharpe
            self.threshold_adjustment = float(np.clip(sharpe / 5, -0.2, 0.2))
        metadata["threshold_adjustment"] = self.threshold_adjustment

        threshold = 0.5 + self.threshold_adjustment
        if action != "HOLD" and confidence < threshold:
            action = "HOLD"
            confidence = 0.0

        return Signal(action=action, confidence=confidence, metadata=metadata)

    def _aggregate(self, signals: List[Signal]) -> str:
        votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for signal in signals:
            votes[signal.action] += 1
        if votes["BUY"] > votes["SELL"] and votes["BUY"] > 0:
            return "BUY"
        if votes["SELL"] > votes["BUY"] and votes["SELL"] > 0:
            return "SELL"
        return "HOLD"

    @staticmethod
    def _rolling_sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
        excess = returns - risk_free / 252
        mean = excess.mean()
        std = excess.std(ddof=1)
        return float(mean / std * (252 ** 0.5)) if std > 0 else 0.0
