"""Integration layer between scoring-based entry system and MES strategy.

This module provides a clean interface to use the scoring system
within the existing MesOneMinuteTrendStrategy without major refactoring.

Usage:
    evaluator = ScoringEntryEvaluator(config)
    decision = evaluator.evaluate(latest, prev, recent_bars, current_time, metadata)
    
Author: Senior Quantitative Trading Engineer - Feb 2026
"""
from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from loguru import logger

from .scoring_entry import (
    SignalScore,
    PositionSize,
    calculate_signal_score,
    should_enter_trade,
    check_risk_gates
)


@dataclass
class ScoringDecision:
    """Entry decision from scoring system."""
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0-1 scale
    reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: float = 1.0  # 0, 0.5, or 1.0
    metadata: Dict[str, Any] = None
    signal_score: Optional[SignalScore] = None


class ScoringEntryEvaluator:
    """Evaluates entry signals using the scoring system.
    
    This class acts as a bridge between the scoring module and the
    existing strategy, providing scoring-based decisions with proper
    position sizing and risk management.
    """
    
    def __init__(self, config: Any):
        """Initialize evaluator with strategy configuration.
        
        Args:
            config: OneMinuteStrategyConfig or compatible config object
        """
        self.config = config
        
        # Scoring thresholds (can be configured)
        self.min_full_size_score = getattr(config, 'scoring_full_size_threshold', 60.0)
        self.min_half_size_score = getattr(config, 'scoring_half_size_threshold', 45.0)
        
        # Risk limits
        self.max_loss_per_trade = getattr(config, 'max_loss_per_trade', 100.0)
        self.max_daily_loss = getattr(config, 'max_daily_loss', 500.0)
        self.max_open_risk = getattr(config, 'max_open_risk', 300.0)
        
        # Position sizing from config
        self.stop_atr_mult = getattr(config, 'stop_atr_multiplier', 1.5)
        self.target_risk_mult = getattr(config, 'take_profit_multiple', 2.0)
        self.min_stop_points = getattr(config, 'min_stop_points', 3.25)
        
    def evaluate(
        self,
        latest: pd.Series,
        prev: Optional[pd.Series],
        recent_bars: pd.DataFrame,
        current_time: pd.Timestamp,
        metadata: Dict[str, Any],
        daily_pnl: float = 0.0,
        open_risk: float = 0.0
    ) -> ScoringDecision:
        """Evaluate entry signal using scoring system.
        
        Args:
            latest: Current bar data (pd.Series from enriched DataFrame)
            prev: Previous bar data for slope calculations
            recent_bars: Recent 10-20 bars for momentum analysis
            current_time: Current timestamp
            metadata: Strategy metadata (trend_label, session_type, etc.)
            daily_pnl: Current daily P&L
            open_risk: Current open risk
            
        Returns:
            ScoringDecision with action, confidence, stops, and diagnostics
        """
        # Extract ATR for stop calculation
        atr_value = float(latest.get("ATR_14", 1.0))
        
        # Calculate ATR percentile
        atr_percentile = self._calculate_atr_percentile(recent_bars, atr_value)
        
        # Prepare data dict for scoring
        data = self._prepare_data_dict(latest, metadata)
        prev_data = self._prepare_data_dict(prev, metadata) if prev is not None else None
        
        # Convert timestamp
        timestamp = current_time.to_pydatetime() if hasattr(current_time, 'to_pydatetime') else current_time
        
        # Calculate signal score
        signal_score = calculate_signal_score(
            data=data,
            prev_data=prev_data,
            recent_bars=recent_bars,
            timestamp=timestamp,
            atr_percentile=atr_percentile
        )
        
        # Check risk gates (HARD constraints)
        session_type = metadata.get('session_type', 'RTH')
        session_end = self._get_session_end_time(session_type)
        
        risk_allowed, risk_reason = check_risk_gates(
            data=data,
            timestamp=timestamp,
            current_position_pnl=0.0,  # Not tracking individual position here
            daily_pnl=daily_pnl,
            open_risk=open_risk,
            max_loss_per_trade=self.max_loss_per_trade,
            max_daily_loss=self.max_daily_loss,
            max_open_risk=self.max_open_risk,
            session_end_time=session_end
        )
        
        if not risk_allowed:
            # Risk gate blocked - return HOLD
            return ScoringDecision(
                action="HOLD",
                confidence=0.0,
                reason=f"RISK_GATE: {risk_reason}",
                metadata=metadata,
                signal_score=signal_score
            )
        
        # Determine position size from score
        position_size, size_reason = should_enter_trade(
            signal_score,
            self.min_full_size_score,
            self.min_half_size_score
        )
        
        # If no trade, return HOLD
        if position_size == PositionSize.NONE:
            return ScoringDecision(
                action="HOLD",
                confidence=0.0,
                reason=size_reason,
                metadata=metadata,
                signal_score=signal_score
            )
        
        # Calculate stops and targets
        close = float(latest.get('close', 0))
        stop_loss, take_profit = self._calculate_brackets(
            direction=signal_score.direction,
            entry_price=close,
            atr=atr_value,
            position_size_multiplier=position_size.value
        )
        
        # Convert position size to confidence (0-1 scale for compatibility)
        # Full size = 0.8 confidence, Half size = 0.6 confidence
        confidence = 0.8 if position_size == PositionSize.FULL else 0.6
        
        # Build action
        action = "BUY" if signal_score.direction == "LONG" else "SELL"
        
        # Build comprehensive reason string
        reason_parts = [
            f"SCORE={signal_score.total_score:.1f}",
            f"SIZE={position_size.value:.1f}x",
            f"TOP={'+'.join(signal_score.reasons[:3])}"
        ]
        reason = " | ".join(reason_parts)
        
        # Add scoring diagnostics to metadata
        metadata_out = metadata.copy()
        metadata_out.update({
            'signal_score': signal_score.total_score,
            'position_size': position_size.value,
            'score_breakdown': signal_score.get_breakdown(),
            'score_components': [
                {'name': c.name, 'value': c.value, 'category': c.category}
                for c in signal_score.components
            ]
        })
        
        # Log comprehensive diagnostic
        logger.info(f"SCORING_ENTRY: {action} {signal_score.format_diagnostic()}")
        
        return ScoringDecision(
            action=action,
            confidence=confidence,
            reason=reason,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size.value,
            metadata=metadata_out,
            signal_score=signal_score
        )
    
    def _prepare_data_dict(self, bar: pd.Series, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data dictionary for scoring system."""
        if bar is None:
            return {}
        
        def safe_float(val, default=0.0):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            try:
                return float(val)
            except (TypeError, ValueError):
                return default
        
        return {
            # Price data
            'close': safe_float(bar.get('close', 0)),
            'open': safe_float(bar.get('open', bar.get('close', 0))),
            'high': safe_float(bar.get('high', bar.get('close', 0))),
            'low': safe_float(bar.get('low', bar.get('close', 0))),
            
            # EMAs
            'EMA_9': safe_float(bar.get('EMA_9', bar.get('close', 0))),
            'EMA_21': safe_float(bar.get('EMA_21', bar.get('close', 0))),
            'EMA_50': safe_float(bar.get('EMA_50', bar.get('EMA_21', bar.get('close', 0)))),
            
            # VWAP
            'SESSION_VWAP': safe_float(bar.get('SESSION_VWAP', bar.get('close', 0))),
            
            # Indicators
            'RSI_14': safe_float(bar.get('RSI_14', 50)),
            'MACD_hist': safe_float(bar.get('MACD_hist', 0)),
            'ADX_14': safe_float(bar.get('ADX_14', 0)),
            'ATR_14': safe_float(bar.get('ATR_14', 1.0)),
            
            # Volume
            'volume': safe_float(bar.get('volume', 0)),
            
            # Metadata from strategy
            'trend_label': metadata.get('trend_label', 'UNKNOWN'),
            'trend_label_htf': metadata.get('15m_regime', metadata.get('trend_label', 'UNKNOWN')),
        }
    
    def _calculate_atr_percentile(self, recent_bars: pd.DataFrame, current_atr: float) -> float:
        """Calculate current ATR percentile from recent bars."""
        if 'ATR_14' not in recent_bars.columns:
            return 0.5  # Default to median
        
        atr_series = recent_bars['ATR_14'].tail(200).dropna()
        if len(atr_series) < 20:
            return 0.5
        
        # Calculate percentile using linear interpolation
        try:
            percentile = (atr_series < current_atr).sum() / len(atr_series)
            return float(percentile)
        except Exception as e:
            logger.warning(f"Could not calculate ATR percentile: {e}")
            return 0.5
    
    def _get_session_end_time(self, session_type: str) -> Optional[time]:
        """Get session end time for risk gate checking."""
        if session_type == "RTH":
            # RTH ends at 15:00 CST (3 PM)
            return time(15, 0)
        elif session_type == "OVERNIGHT":
            # Overnight ends at 09:30 CST (9:30 AM)
            return time(9, 30)
        else:
            return None
    
    def _calculate_brackets(
        self,
        direction: str,
        entry_price: float,
        atr: float,
        position_size_multiplier: float = 1.0
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels.
        
        Args:
            direction: 'LONG' or 'SHORT'
            entry_price: Entry price
            atr: Current ATR value
            position_size_multiplier: Position size (0.5 or 1.0)
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        # Base stop distance
        stop_distance = max(atr * self.stop_atr_mult, self.min_stop_points)
        
        # For half-size positions, can use slightly wider stops
        if position_size_multiplier < 1.0:
            stop_distance *= 1.2
        
        # Calculate target
        target_distance = stop_distance * self.target_risk_mult
        
        if direction == "LONG":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + target_distance
        else:  # SHORT
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - target_distance
        
        return stop_loss, take_profit


def create_scoring_evaluator(config: Any) -> ScoringEntryEvaluator:
    """Factory function to create scoring evaluator.
    
    Args:
        config: Strategy configuration object
        
    Returns:
        Configured ScoringEntryEvaluator instance
    """
    return ScoringEntryEvaluator(config)
