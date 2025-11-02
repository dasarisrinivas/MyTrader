"""Momentum reversal strategy."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import BaseStrategy, Signal


@dataclass
class MomentumReversalStrategy(BaseStrategy):
    name: str = "momentum_reversal"
    lookback: int = 20
    threshold: float = 0.01

    def generate(self, features: pd.DataFrame) -> Signal:
        if len(features) < 2:
            return Signal(action="HOLD", confidence=0.0, metadata={"mean_return": 0.0, "volatility": 0.0})
        recent = features.iloc[-self.lookback :]
        returns = recent["close"].pct_change().dropna()
        mean_return = returns.mean()
        volatility = returns.std()
        data = {
            "mean_return": float(mean_return),
            "volatility": float(volatility if not pd.isna(volatility) else 0),
        }
        if mean_return > self.threshold and volatility < self.threshold * 2:
            return Signal(action="BUY", confidence=float(min(1, mean_return / self.threshold)), metadata=data)
        if mean_return < -self.threshold and volatility < self.threshold * 2:
            return Signal(action="SELL", confidence=float(min(1, abs(mean_return) / self.threshold)), metadata=data)
        return Signal(action="HOLD", confidence=0.0, metadata=data)
