"""Market regime detection for adaptive strategy selection."""
from __future__ import annotations

from enum import Enum
from typing import Tuple

import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


def detect_market_regime(features: pd.DataFrame, lookback: int = 50) -> Tuple[MarketRegime, float]:
    """
    Detect current market regime based on price action and indicators.
    
    Args:
        features: DataFrame with OHLCV and technical indicators
        lookback: Period to analyze for regime detection
        
    Returns:
        Tuple of (regime, confidence_score)
    """
    if len(features) < lookback:
        return MarketRegime.MEAN_REVERTING, 0.5
    
    recent = features.tail(lookback)
    latest = features.iloc[-1]
    
    # Calculate metrics for regime detection
    close = recent["close"]
    returns = close.pct_change().dropna()
    
    # Trend detection using multiple EMAs
    ema_21 = latest.get("EMA_21", latest["close"])
    ema_50 = latest.get("EMA_50", latest["close"])
    ema_200 = latest.get("EMA_200", latest["close"])
    
    # ADX for trend strength
    adx = latest.get("ADX_14", 20)
    
    # ATR for volatility
    atr = latest.get("ATR_14", 0)
    avg_atr = recent["ATR_14"].mean() if "ATR_14" in recent.columns else atr
    
    # Volatility metrics
    volatility = returns.std()
    avg_volatility = returns.rolling(20).std().mean()
    
    # Determine regime
    confidence = 0.6
    
    # 1. Check for trending market
    if adx > 25:  # Strong trend
        if ema_21 > ema_50 > ema_200:
            # Uptrend
            confidence = min(0.9, 0.6 + (adx - 25) / 100)
            return MarketRegime.TRENDING_UP, confidence
        elif ema_21 < ema_50 < ema_200:
            # Downtrend
            confidence = min(0.9, 0.6 + (adx - 25) / 100)
            return MarketRegime.TRENDING_DOWN, confidence
    
    # 2. Check for high volatility
    if volatility > avg_volatility * 1.5:
        confidence = min(0.85, 0.6 + (volatility / avg_volatility - 1))
        return MarketRegime.HIGH_VOLATILITY, confidence
    
    # 3. Check for low volatility
    if volatility < avg_volatility * 0.5:
        confidence = 0.7
        return MarketRegime.LOW_VOLATILITY, confidence
    
    # 4. Default to mean-reverting
    return MarketRegime.MEAN_REVERTING, 0.6


def get_regime_parameters(regime: MarketRegime) -> dict:
    """
    Get adaptive parameters based on market regime.
    
    Args:
        regime: Current market regime
        
    Returns:
        Dictionary of strategy parameters
    """
    if regime == MarketRegime.TRENDING_UP:
        return {
            "rsi_buy": 40,
            "rsi_sell": 75,
            "sentiment_buy": -0.3,
            "sentiment_sell": 0.3,
            "use_macd_crossover": True,
            "position_multiplier": 1.2,
            # Risk parameters
            "atr_multiplier_sl": 2.0,
            "atr_multiplier_tp": 4.0,
            "risk_reward_ratio": 2.0,
        }
    elif regime == MarketRegime.TRENDING_DOWN:
        return {
            "rsi_buy": 25,
            "rsi_sell": 60,
            "sentiment_buy": -0.4,
            "sentiment_sell": 0.2,
            "use_macd_crossover": True,
            "position_multiplier": 0.8,
            # Risk parameters
            "atr_multiplier_sl": 2.0,
            "atr_multiplier_tp": 4.0,
            "risk_reward_ratio": 2.0,
        }
    elif regime == MarketRegime.HIGH_VOLATILITY:
        return {
            "rsi_buy": 25,
            "rsi_sell": 75,
            "sentiment_buy": -0.2,
            "sentiment_sell": 0.2,
            "use_macd_crossover": False,
            "position_multiplier": 0.7,  # Reduce position size in high vol
            # Risk parameters - Wider stops to avoid noise
            "atr_multiplier_sl": 2.5,
            "atr_multiplier_tp": 5.0,  # Aim for larger moves
            "risk_reward_ratio": 2.0,
        }
    elif regime == MarketRegime.LOW_VOLATILITY:
        return {
            "rsi_buy": 35,
            "rsi_sell": 65,
            "sentiment_buy": -0.2,
            "sentiment_sell": 0.2,
            "use_macd_crossover": True,
            "position_multiplier": 1.0,
            # Risk parameters - Tighter stops in quiet market
            "atr_multiplier_sl": 1.5,
            "atr_multiplier_tp": 2.5,  # Smaller targets
            "risk_reward_ratio": 1.67,
        }
    else:  # MEAN_REVERTING
        return {
            "rsi_buy": 30,
            "rsi_sell": 70,
            "sentiment_buy": -0.2,
            "sentiment_sell": 0.2,
            "use_macd_crossover": False,
            "position_multiplier": 1.0,
            # Risk parameters
            "atr_multiplier_sl": 2.0,
            "atr_multiplier_tp": 3.0,
            "risk_reward_ratio": 1.5,
        }
