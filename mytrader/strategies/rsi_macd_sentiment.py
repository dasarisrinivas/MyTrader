"""RSI + MACD + sentiment blended strategy."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import BaseStrategy, Signal


@dataclass
class RsiMacdSentimentStrategy(BaseStrategy):
    name: str = "rsi_macd_sentiment"
    rsi_buy: float = 30.0
    rsi_sell: float = 70.0
    sentiment_buy: float = 0.6
    sentiment_sell: float = 0.4

    def generate(self, features: pd.DataFrame) -> Signal:
        latest = features.iloc[-1]
        rsi = latest.get("RSI_14", 50)
        macd = latest.get("MACD_12_26_9", 0)
        sentiment = latest.get("sentiment_score", 0)

        if rsi < self.rsi_buy and macd > 0 and sentiment >= self.sentiment_buy:
            return Signal(action="BUY", confidence=float(min(1, sentiment)), metadata={"rsi": float(rsi), "macd": float(macd), "sentiment": float(sentiment)})
        if rsi > self.rsi_sell and macd < 0 and sentiment <= self.sentiment_sell:
            return Signal(action="SELL", confidence=float(min(1, 1 - sentiment)), metadata={"rsi": float(rsi), "macd": float(macd), "sentiment": float(sentiment)})
        return Signal(action="HOLD", confidence=0.0, metadata={"rsi": float(rsi), "macd": float(macd), "sentiment": float(sentiment)})
