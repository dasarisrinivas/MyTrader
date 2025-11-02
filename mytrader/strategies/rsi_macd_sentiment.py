"""RSI + MACD + sentiment blended strategy with enhanced signal generation."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import BaseStrategy, Signal


@dataclass
class RsiMacdSentimentStrategy(BaseStrategy):
    name: str = "rsi_macd_sentiment"
    rsi_buy: float = 40.0  # More balanced threshold
    rsi_sell: float = 60.0  # More balanced threshold
    sentiment_buy: float = -0.3  # More realistic threshold
    sentiment_sell: float = 0.3  # More realistic threshold
    use_macd_crossover: bool = False  # Use direct MACD comparison for more signals

    def generate(self, features: pd.DataFrame) -> Signal:
        """Generate enhanced trading signal with multiple confirmation factors."""
        if len(features) < 2:
            return Signal(action="HOLD", confidence=0.0, metadata={})
            
        latest = features.iloc[-1]
        prev = features.iloc[-2]
        
        rsi = latest.get("RSI_14", 50)
        macd = latest.get("MACD_12_26_9", 0)
        macd_signal = latest.get("MACDsignal_12_26_9", 0)
        macd_hist = latest.get("MACDhist_12_26_9", 0)
        sentiment = latest.get("sentiment_score", 0)
        
        # Previous MACD for crossover detection
        prev_macd_hist = prev.get("MACDhist_12_26_9", 0)
        
        # Additional indicators for confirmation
        adx = latest.get("ADX_14", 0)
        atr = latest.get("ATR_14", 0)
        bb_percent = latest.get("BB_percent", 0.5)
        
        # Calculate confidence based on signal strength
        confidence = 0.5
        
        # BUY signal conditions
        buy_conditions = []
        if rsi < self.rsi_buy:
            buy_conditions.append(True)
            confidence += (self.rsi_buy - rsi) / 100  # Stronger signal for lower RSI
        
        if self.use_macd_crossover:
            # Bullish MACD crossover
            if macd_hist > 0 and prev_macd_hist <= 0:
                buy_conditions.append(True)
                confidence += 0.2
        else:
            if macd > 0:
                buy_conditions.append(True)
        
        if sentiment > self.sentiment_buy:
            buy_conditions.append(True)
            confidence += sentiment * 0.1 if sentiment > 0 else 0
        
        # Bollinger Band confirmation (oversold)
        if bb_percent < 0.2:
            confidence += 0.1
        
        # Trend strength confirmation
        if adx > 25:
            confidence += 0.1
        
        # Generate BUY signal
        if len(buy_conditions) >= 2:  # At least 2 conditions met
            confidence = min(0.95, confidence)
            return Signal(
                action="BUY",
                confidence=float(confidence),
                metadata={
                    "rsi": float(rsi),
                    "macd": float(macd),
                    "macd_hist": float(macd_hist),
                    "sentiment": float(sentiment),
                    "adx": float(adx),
                    "bb_percent": float(bb_percent),
                    "conditions_met": len(buy_conditions)
                }
            )
        
        # SELL signal conditions
        sell_conditions = []
        confidence = 0.5
        
        if rsi > self.rsi_sell:
            sell_conditions.append(True)
            confidence += (rsi - self.rsi_sell) / 100
        
        if self.use_macd_crossover:
            # Bearish MACD crossover
            if macd_hist < 0 and prev_macd_hist >= 0:
                sell_conditions.append(True)
                confidence += 0.2
        else:
            if macd < 0:
                sell_conditions.append(True)
        
        if sentiment < self.sentiment_sell:
            sell_conditions.append(True)
            confidence += abs(sentiment) * 0.1 if sentiment < 0 else 0
        
        # Bollinger Band confirmation (overbought)
        if bb_percent > 0.8:
            confidence += 0.1
        
        # Trend strength confirmation
        if adx > 25:
            confidence += 0.1
        
        # Generate SELL signal
        if len(sell_conditions) >= 2:  # At least 2 conditions met
            confidence = min(0.95, confidence)
            return Signal(
                action="SELL",
                confidence=float(confidence),
                metadata={
                    "rsi": float(rsi),
                    "macd": float(macd),
                    "macd_hist": float(macd_hist),
                    "sentiment": float(sentiment),
                    "adx": float(adx),
                    "bb_percent": float(bb_percent),
                    "conditions_met": len(sell_conditions)
                }
            )
        
        return Signal(
            action="HOLD",
            confidence=0.0,
            metadata={
                "rsi": float(rsi),
                "macd": float(macd),
                "sentiment": float(sentiment)
            }
        )
