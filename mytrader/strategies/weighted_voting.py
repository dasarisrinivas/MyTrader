"""
Weighted Voting Entry Logic
Combines multiple strategy signals into a weighted confidence score.
"""
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class StrategyScore:
    """Individual strategy score."""
    name: str
    score: float  # -1.0 to 1.0 (negative=bearish, positive=bullish)
    confidence: float  # 0.0 to 1.0
    weight: float = 1.0  # Weight for this strategy


@dataclass
class WeightedSignal:
    """Weighted voting result."""
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0
    weighted_score: float  # Combined weighted score
    component_scores: Dict[str, float]  # Individual strategy scores
    reasoning: str  # Explanation of decision


class WeightedVotingSystem:
    """
    Combine multiple strategy signals using weighted voting.
    
    Each strategy provides:
    - Direction score (-1.0 to 1.0)
    - Confidence (0.0 to 1.0)
    
    The system calculates a weighted average and determines action.
    """
    
    def __init__(
        self,
        trend_weight: float = 0.4,
        breakout_weight: float = 0.3,
        mean_reversion_weight: float = 0.3,
        min_confidence_threshold: float = 0.70,
    ):
        """
        Initialize weighted voting system.
        
        Args:
            trend_weight: Weight for trend-following signals
            breakout_weight: Weight for breakout signals
            mean_reversion_weight: Weight for mean-reversion signals
            min_confidence_threshold: Minimum confidence to trade
        """
        self.trend_weight = trend_weight
        self.breakout_weight = breakout_weight
        self.mean_reversion_weight = mean_reversion_weight
        self.min_confidence_threshold = min_confidence_threshold
        
        # Normalize weights to sum to 1.0
        total = trend_weight + breakout_weight + mean_reversion_weight
        self.trend_weight /= total
        self.breakout_weight /= total
        self.mean_reversion_weight /= total
        
        logger.info(f"Weighted Voting System initialized:")
        logger.info(f"  Trend: {self.trend_weight:.2f}")
        logger.info(f"  Breakout: {self.breakout_weight:.2f}")
        logger.info(f"  Mean Reversion: {self.mean_reversion_weight:.2f}")
        logger.info(f"  Min Confidence: {self.min_confidence_threshold:.2f}")
    
    def calculate_weighted_signal(
        self,
        trend_score: float,
        trend_confidence: float,
        breakout_score: float,
        breakout_confidence: float,
        mean_reversion_score: float,
        mean_reversion_confidence: float,
    ) -> WeightedSignal:
        """
        Calculate weighted voting signal from component strategies.
        
        Args:
            trend_score: Trend-following direction score (-1 to 1)
            trend_confidence: Trend confidence (0 to 1)
            breakout_score: Breakout direction score (-1 to 1)
            breakout_confidence: Breakout confidence (0 to 1)
            mean_reversion_score: Mean-reversion direction score (-1 to 1)
            mean_reversion_confidence: Mean-reversion confidence (0 to 1)
            
        Returns:
            WeightedSignal with action and confidence
        """
        # Calculate weighted score
        weighted_score = (
            trend_score * trend_confidence * self.trend_weight +
            breakout_score * breakout_confidence * self.breakout_weight +
            mean_reversion_score * mean_reversion_confidence * self.mean_reversion_weight
        )
        
        # Calculate overall confidence (weighted average of confidences)
        overall_confidence = (
            trend_confidence * self.trend_weight +
            breakout_confidence * self.breakout_weight +
            mean_reversion_confidence * self.mean_reversion_weight
        )
        
        # Store component scores for logging
        component_scores = {
            "trend_score": trend_score,
            "trend_confidence": trend_confidence,
            "trend_weighted": trend_score * trend_confidence * self.trend_weight,
            "breakout_score": breakout_score,
            "breakout_confidence": breakout_confidence,
            "breakout_weighted": breakout_score * breakout_confidence * self.breakout_weight,
            "mean_reversion_score": mean_reversion_score,
            "mean_reversion_confidence": mean_reversion_confidence,
            "mean_reversion_weighted": mean_reversion_score * mean_reversion_confidence * self.mean_reversion_weight,
        }
        
        # Determine action based on weighted score
        action = "HOLD"
        reasoning_parts = []
        
        # Strong bullish signal
        if weighted_score > 0.1 and overall_confidence >= self.min_confidence_threshold:
            action = "BUY"
            reasoning_parts.append(f"Bullish weighted score: {weighted_score:.3f}")
        # Strong bearish signal
        elif weighted_score < -0.1 and overall_confidence >= self.min_confidence_threshold:
            action = "SELL"
            reasoning_parts.append(f"Bearish weighted score: {weighted_score:.3f}")
        # Weak signal or low confidence
        else:
            if overall_confidence < self.min_confidence_threshold:
                reasoning_parts.append(f"Confidence too low: {overall_confidence:.3f} < {self.min_confidence_threshold}")
            else:
                reasoning_parts.append(f"Weak signal: {weighted_score:.3f} (threshold: ±0.1)")
        
        # Add component breakdown
        reasoning_parts.append(f"Trend: {trend_score:.2f}×{trend_confidence:.2f}")
        reasoning_parts.append(f"Breakout: {breakout_score:.2f}×{breakout_confidence:.2f}")
        reasoning_parts.append(f"MeanRev: {mean_reversion_score:.2f}×{mean_reversion_confidence:.2f}")
        
        reasoning = " | ".join(reasoning_parts)
        
        return WeightedSignal(
            action=action,
            confidence=overall_confidence,
            weighted_score=weighted_score,
            component_scores=component_scores,
            reasoning=reasoning
        )
    
    def extract_strategy_scores(
        self,
        df: pd.DataFrame,
        current_position: int = 0
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Extract individual strategy scores from market data.
        
        Args:
            df: DataFrame with OHLCV and indicators
            current_position: Current position size
            
        Returns:
            Tuple of (trend_score, trend_conf, breakout_score, breakout_conf,
                     mean_rev_score, mean_rev_conf)
        """
        if len(df) < 50:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        # Calculate trend score
        trend_score, trend_conf = self._calculate_trend_score(df)
        
        # Calculate breakout score
        breakout_score, breakout_conf = self._calculate_breakout_score(df)
        
        # Calculate mean reversion score
        mean_rev_score, mean_rev_conf = self._calculate_mean_reversion_score(df)
        
        return (
            trend_score, trend_conf,
            breakout_score, breakout_conf,
            mean_rev_score, mean_rev_conf
        )
    
    def _calculate_trend_score(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate trend-following score and confidence."""
        try:
            # Use moving averages
            df_copy = df.copy()
            df_copy['ma_10'] = df_copy['close'].rolling(10).mean()
            df_copy['ma_50'] = df_copy['close'].rolling(50).mean()
            
            ma_10 = df_copy['ma_10'].iloc[-1]
            ma_50 = df_copy['ma_50'].iloc[-1]
            current_price = df_copy['close'].iloc[-1]
            
            # Score based on MA relationship
            if ma_10 > ma_50:
                # Uptrend - score based on separation
                separation = (ma_10 - ma_50) / ma_50
                score = min(1.0, separation * 100)  # Scale to -1 to 1
                confidence = min(0.9, 0.6 + separation * 50)
            else:
                # Downtrend
                separation = (ma_50 - ma_10) / ma_50
                score = -min(1.0, separation * 100)
                confidence = min(0.9, 0.6 + separation * 50)
            
            return float(score), float(confidence)
        
        except Exception as e:
            logger.error(f"Error calculating trend score: {e}")
            return 0.0, 0.0
    
    def _calculate_breakout_score(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate breakout score and confidence."""
        try:
            lookback = 20
            df_copy = df.copy()
            df_copy['rolling_high'] = df_copy['high'].rolling(lookback).max()
            df_copy['rolling_low'] = df_copy['low'].rolling(lookback).min()
            
            current_price = df_copy['close'].iloc[-1]
            prev_high = df_copy['rolling_high'].iloc[-2]
            prev_low = df_copy['rolling_low'].iloc[-2]
            
            # Breakout detection
            if current_price > prev_high:
                # Upward breakout
                breakout_pct = (current_price - prev_high) / prev_high
                score = min(1.0, breakout_pct * 200)
                confidence = min(0.9, 0.65 + breakout_pct * 100)
            elif current_price < prev_low:
                # Downward breakout
                breakout_pct = (prev_low - current_price) / prev_low
                score = -min(1.0, breakout_pct * 200)
                confidence = min(0.9, 0.65 + breakout_pct * 100)
            else:
                # No breakout
                score = 0.0
                confidence = 0.3
            
            return float(score), float(confidence)
        
        except Exception as e:
            logger.error(f"Error calculating breakout score: {e}")
            return 0.0, 0.0
    
    def _calculate_mean_reversion_score(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate mean-reversion score and confidence."""
        try:
            # Use Bollinger Bands and RSI
            df_copy = df.copy()
            df_copy['bb_middle'] = df_copy['close'].rolling(20).mean()
            df_copy['bb_std'] = df_copy['close'].rolling(20).std()
            df_copy['bb_upper'] = df_copy['bb_middle'] + (2 * df_copy['bb_std'])
            df_copy['bb_lower'] = df_copy['bb_middle'] - (2 * df_copy['bb_std'])
            
            # Calculate RSI
            delta = df_copy['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_price = df_copy['close'].iloc[-1]
            bb_upper = df_copy['bb_upper'].iloc[-1]
            bb_lower = df_copy['bb_lower'].iloc[-1]
            bb_middle = df_copy['bb_middle'].iloc[-1]
            current_rsi = rsi.iloc[-1]
            
            # Oversold (buy signal)
            if current_price < bb_lower and current_rsi < 30:
                distance_pct = (bb_lower - current_price) / bb_lower
                score = min(1.0, distance_pct * 50)
                confidence = min(0.9, 0.7 + distance_pct * 20)
            # Overbought (sell signal)
            elif current_price > bb_upper and current_rsi > 70:
                distance_pct = (current_price - bb_upper) / bb_upper
                score = -min(1.0, distance_pct * 50)
                confidence = min(0.9, 0.7 + distance_pct * 20)
            else:
                # No clear signal
                score = 0.0
                confidence = 0.3
            
            return float(score), float(confidence)
        
        except Exception as e:
            logger.error(f"Error calculating mean reversion score: {e}")
            return 0.0, 0.0
