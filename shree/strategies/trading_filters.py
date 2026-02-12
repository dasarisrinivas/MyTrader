"""Trading filters for multi-timeframe levels and entry validation.

This module provides:
1. Higher-timeframe level calculation (PDH/PDL, WH/WL, PWH/PWL)
2. Trend confirmation filters (EMA-based)
3. Volatility filters
4. Candle close validation
5. Support/Resistance proximity checks
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..utils.logger import logger


@dataclass
class PriceLevels:
    """Container for important price levels."""
    # Previous Day Levels
    pdh: Optional[float] = None  # Previous Day High
    pdl: Optional[float] = None  # Previous Day Low
    pdc: Optional[float] = None  # Previous Day Close
    
    # This Week Levels
    wh: Optional[float] = None   # This Week High (so far)
    wl: Optional[float] = None   # This Week Low (so far)
    
    # Previous Week Levels
    pwh: Optional[float] = None  # Previous Week High
    pwl: Optional[float] = None  # Previous Week Low
    
    # Session levels (RTH)
    session_high: Optional[float] = None
    session_low: Optional[float] = None
    
    # Calculated zones
    pivot: Optional[float] = None  # Daily pivot point
    r1: Optional[float] = None     # Resistance 1
    s1: Optional[float] = None     # Support 1
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def is_valid(self) -> bool:
        """Check if essential levels are available."""
        return self.pdh is not None and self.pdl is not None


@dataclass
class TradingFilterResult:
    """Result of trading filter evaluation."""
    can_trade: bool
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence_adjustment: float = 0.0
    reasons: List[str] = field(default_factory=list)
    levels: Optional[PriceLevels] = None
    
    def add_reason(self, reason: str):
        self.reasons.append(reason)


class TradingFilters:
    """
    Comprehensive trading filters for signal validation.
    
    Features:
    - Multi-timeframe level calculation (PDH/PDL, WH/WL, PWH/PWL)
    - Trend confirmation (EMA 9 > EMA 20 for longs, etc.)
    - Volatility filters (ATR-based)
    - Candle close validation
    - Support/Resistance proximity
    - Chop zone detection
    """
    
    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 20,
        atr_period: int = 14,
        min_atr_threshold: float = 0.5,  # Minimum ATR for valid volatility
        max_atr_threshold: float = 5.0,  # Maximum ATR to avoid spike entries
        chop_zone_buffer_pct: float = 0.25,  # % buffer inside PDH-PDL for chop zone
        sr_proximity_ticks: int = 8,  # Ticks proximity to S/R to avoid entry
        require_candle_close: bool = True,  # Wait for candle close
        candle_period_seconds: int = 60,  # 1-minute candles
        require_trend_alignment: bool = True,
        allow_counter_trend: bool = False,
        ema_alignment_tolerance_pct: float = 0.0002,
        counter_trend_penalty: float = 0.10,
        min_atr_percentile: Optional[float] = None,
        atr_percentile_lookback: int = 120,
        low_atr_penalty_mode: bool = False,
        low_atr_penalty: float = 0.10,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.min_atr_threshold = min_atr_threshold
        self.max_atr_threshold = max_atr_threshold
        self.chop_zone_buffer_pct = chop_zone_buffer_pct
        self.sr_proximity_ticks = sr_proximity_ticks
        self.require_candle_close = require_candle_close
        self.candle_period_seconds = candle_period_seconds
        self.require_trend_alignment = require_trend_alignment
        self.allow_counter_trend = allow_counter_trend
        self.ema_alignment_tolerance_pct = ema_alignment_tolerance_pct
        self.counter_trend_penalty = counter_trend_penalty
        self.min_atr_percentile = min_atr_percentile
        self.atr_percentile_lookback = atr_percentile_lookback
        self.low_atr_penalty_mode = low_atr_penalty_mode
        self.low_atr_penalty = low_atr_penalty
        
        # State tracking
        self._levels: Optional[PriceLevels] = None
        self._levels_updated_at: Optional[datetime] = None
        self._historical_data: Optional[pd.DataFrame] = None
        self._last_candle_close_time: Optional[datetime] = None
        
    def set_historical_data(self, df: pd.DataFrame):
        """Set historical data for level calculation."""
        self._historical_data = df.copy()
        self._calculate_levels()
        
    def _calculate_levels(self):
        """Calculate PDH/PDL, WH/WL, PWH/PWL from historical data."""
        if self._historical_data is None or len(self._historical_data) < 2:
            logger.warning("Insufficient historical data for level calculation")
            return
            
        df = self._historical_data.copy()
        
        # Ensure we have datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.index = pd.to_datetime(df['timestamp'])
            else:
                logger.warning("Cannot calculate levels: no datetime index")
                return
        
        now = datetime.now(timezone.utc)
        today = now.date()
        
        levels = PriceLevels()
        
        try:
            # Calculate Previous Day Levels
            yesterday = today - timedelta(days=1)
            yesterday_data = df[df.index.date == yesterday]
            
            if len(yesterday_data) > 0:
                levels.pdh = float(yesterday_data['high'].max())
                levels.pdl = float(yesterday_data['low'].min())
                levels.pdc = float(yesterday_data['close'].iloc[-1])
                logger.info(f"PDH={levels.pdh:.2f}, PDL={levels.pdl:.2f}, PDC={levels.pdc:.2f}")
            else:
                # Fallback: use last 24h of data
                last_24h = df[df.index >= now - timedelta(hours=24)]
                if len(last_24h) > 0:
                    levels.pdh = float(last_24h['high'].max())
                    levels.pdl = float(last_24h['low'].min())
                    levels.pdc = float(last_24h['close'].iloc[-1])
                    logger.info(f"Using 24h fallback: PDH={levels.pdh:.2f}, PDL={levels.pdl:.2f}")
            
            # Calculate This Week Levels (Monday-Friday of current week)
            week_start = today - timedelta(days=today.weekday())  # Monday
            this_week_data = df[(df.index.date >= week_start) & (df.index.date <= today)]
            
            if len(this_week_data) > 0:
                levels.wh = float(this_week_data['high'].max())
                levels.wl = float(this_week_data['low'].min())
                logger.info(f"WH={levels.wh:.2f}, WL={levels.wl:.2f}")
            
            # Calculate Previous Week Levels
            prev_week_start = week_start - timedelta(days=7)
            prev_week_end = week_start - timedelta(days=1)
            prev_week_data = df[(df.index.date >= prev_week_start) & (df.index.date <= prev_week_end)]
            
            if len(prev_week_data) > 0:
                levels.pwh = float(prev_week_data['high'].max())
                levels.pwl = float(prev_week_data['low'].min())
                logger.info(f"PWH={levels.pwh:.2f}, PWL={levels.pwl:.2f}")
            
            # Calculate Pivot Points (Classic)
            if levels.pdh and levels.pdl and levels.pdc:
                levels.pivot = (levels.pdh + levels.pdl + levels.pdc) / 3
                levels.r1 = 2 * levels.pivot - levels.pdl
                levels.s1 = 2 * levels.pivot - levels.pdh
                logger.info(f"Pivot={levels.pivot:.2f}, R1={levels.r1:.2f}, S1={levels.s1:.2f}")
            
            levels.timestamp = now
            self._levels = levels
            self._levels_updated_at = now
            
        except Exception as e:
            logger.error(f"Failed to calculate price levels: {e}")
            
    def get_levels(self) -> Optional[PriceLevels]:
        """Get current price levels."""
        return self._levels
        
    def evaluate(
        self,
        current_price: float,
        proposed_action: str,
        features: pd.DataFrame,
        tick_size: float = 0.25,
    ) -> TradingFilterResult:
        """
        Evaluate whether a trade should be allowed based on multiple filters.
        
        Args:
            current_price: Current market price
            proposed_action: 'BUY' or 'SELL'
            features: DataFrame with indicators (EMA_9, EMA_20, ATR_14, etc.)
            tick_size: Contract tick size for proximity calculations
            
        Returns:
            TradingFilterResult with can_trade flag and reasons
        """
        result = TradingFilterResult(
            can_trade=True,
            action=proposed_action,
            levels=self._levels
        )
        
        if len(features) < 2:
            result.can_trade = False
            result.add_reason("Insufficient feature data")
            return result
            
        latest = features.iloc[-1]
        price_for_tolerance = self._get_price_for_tolerance(current_price, latest)
        
        # 1. CANDLE CLOSE VALIDATION
        if self.require_candle_close:
            if not self._is_candle_closed():
                result.can_trade = False
                result.add_reason("Waiting for candle close")
                return result
        
        # 2. TREND CONFIRMATION (EMA-based with tolerance)
        ema_fast = latest.get(f'EMA_{self.ema_fast}', None)
        ema_slow = latest.get(f'EMA_{self.ema_slow}', None)
        
        if ema_fast is not None and ema_slow is not None:
            ema_diff = ema_fast - ema_slow
            tolerance = self._calculate_ema_tolerance(price_for_tolerance)
            within_tolerance = tolerance > 0 and abs(ema_diff) <= tolerance
            
            if proposed_action == "BUY":
                if ema_diff > tolerance:
                    result.confidence_adjustment += 0.05
                    result.add_reason(f"Trend aligned: EMA{self.ema_fast}({ema_fast:.2f}) > EMA{self.ema_slow}({ema_slow:.2f})")
                elif ema_diff >= 0 or within_tolerance:
                    result.add_reason(f"Trend neutral: EMA{self.ema_fast}~EMA{self.ema_slow}")
                else:
                    should_block = self.require_trend_alignment and not self.allow_counter_trend and not within_tolerance
                    if should_block:
                        result.can_trade = False
                        result.add_reason(
                            f"Trend filter: EMA{self.ema_fast}({ema_fast:.2f}) < EMA{self.ema_slow}({ema_slow:.2f})"
                        )
                        return result
                    penalty = self.counter_trend_penalty if self.allow_counter_trend or not self.require_trend_alignment else 0.0
                    if penalty:
                        result.confidence_adjustment -= penalty
                    result.add_reason("Counter-trend long penalized")
                    
            elif proposed_action == "SELL":
                if ema_diff < -tolerance:
                    result.confidence_adjustment += 0.05
                    result.add_reason(f"Trend aligned: EMA{self.ema_fast}({ema_fast:.2f}) < EMA{self.ema_slow}({ema_slow:.2f})")
                elif ema_diff <= 0 or within_tolerance:
                    result.add_reason(f"Trend neutral: EMA{self.ema_fast}~EMA{self.ema_slow}")
                else:
                    should_block = self.require_trend_alignment and not self.allow_counter_trend and not within_tolerance
                    if should_block:
                        result.can_trade = False
                        result.add_reason(
                            f"Trend filter: EMA{self.ema_fast}({ema_fast:.2f}) > EMA{self.ema_slow}({ema_slow:.2f})"
                        )
                        return result
                    penalty = self.counter_trend_penalty if self.allow_counter_trend or not self.require_trend_alignment else 0.0
                    if penalty:
                        result.confidence_adjustment -= penalty
                    result.add_reason("Counter-trend short penalized")
        
        # 3. VOLATILITY FILTER (ATR-based)
        atr_series = self._get_atr_series(features)
        atr = float(atr_series.iloc[-1]) if atr_series is not None and len(atr_series) > 0 else 0.0
        effective_min_atr = self._compute_min_atr_threshold(atr_series)
        
        if atr > 0:
            if atr < effective_min_atr:
                if self.low_atr_penalty_mode:
                    penalty = max(0.0, self.low_atr_penalty)
                    if penalty:
                        result.confidence_adjustment -= penalty
                    result.add_reason(f"Low ATR penalty: {atr:.2f} < floor {effective_min_atr:.2f}")
                else:
                    result.can_trade = False
                    result.add_reason(f"Volatility too low: ATR={atr:.2f} < {effective_min_atr:.2f}")
                    return result
                
            if atr > self.max_atr_threshold:
                result.can_trade = False
                result.add_reason(f"Volatility spike: ATR={atr:.2f} > {self.max_atr_threshold}")
                return result
        
        # 4. MULTI-TIMEFRAME LEVEL FILTERS
        if self._levels and self._levels.is_valid():
            pdh = self._levels.pdh
            pdl = self._levels.pdl
            
            # Calculate chop zone (inner X% of PDH-PDL range)
            range_size = pdh - pdl
            buffer = range_size * self.chop_zone_buffer_pct
            chop_upper = pdh - buffer
            chop_lower = pdl + buffer
            
            # Check if price is in chop zone
            if chop_lower < current_price < chop_upper:
                result.confidence_adjustment -= 0.15
                result.add_reason(f"Price in chop zone: {chop_lower:.2f} < {current_price:.2f} < {chop_upper:.2f}")
            
            # PDH/PDL breakout logic
            proximity = self.sr_proximity_ticks * tick_size
            
            if proposed_action == "BUY":
                # Avoid longs below PDL
                if current_price < pdl:
                    result.can_trade = False
                    result.add_reason(f"Long blocked: Price {current_price:.2f} below PDL {pdl:.2f}")
                    return result
                
                # Bonus for breaking above PDH
                if current_price > pdh:
                    result.confidence_adjustment += 0.10
                    result.add_reason(f"Breakout above PDH: {current_price:.2f} > {pdh:.2f}")
                
                # Avoid entry near resistance (PDH)
                if pdh - proximity < current_price < pdh:
                    result.confidence_adjustment -= 0.10
                    result.add_reason(f"Near resistance PDH: {current_price:.2f}")
                    
            elif proposed_action == "SELL":
                # Avoid shorts above PDH
                if current_price > pdh:
                    result.can_trade = False
                    result.add_reason(f"Short blocked: Price {current_price:.2f} above PDH {pdh:.2f}")
                    return result
                
                # Bonus for breaking below PDL
                if current_price < pdl:
                    result.confidence_adjustment += 0.10
                    result.add_reason(f"Breakout below PDL: {current_price:.2f} < {pdl:.2f}")
                
                # Avoid entry near support (PDL)
                if pdl < current_price < pdl + proximity:
                    result.confidence_adjustment -= 0.10
                    result.add_reason(f"Near support PDL: {current_price:.2f}")
            
            # Weekly level checks
            if self._levels.pwh and self._levels.pwl:
                if proposed_action == "BUY" and current_price < self._levels.pwl:
                    result.confidence_adjustment -= 0.20
                    result.add_reason(f"Below prev week low: {current_price:.2f} < PWL {self._levels.pwl:.2f}")
                elif proposed_action == "SELL" and current_price > self._levels.pwh:
                    result.confidence_adjustment -= 0.20
                    result.add_reason(f"Above prev week high: {current_price:.2f} > PWH {self._levels.pwh:.2f}")
        
        # 5. VOLUME CONFIRMATION (if available)
        volume = latest.get('volume', 0)
        avg_volume = features['volume'].rolling(20).mean().iloc[-1] if 'volume' in features.columns else 0
        
        if volume > 0 and avg_volume > 0:
            if volume < avg_volume * 0.5:
                result.confidence_adjustment -= 0.05
                result.add_reason(f"Low volume: {volume} < 50% avg ({avg_volume:.0f})")
            elif volume > avg_volume * 1.5:
                result.confidence_adjustment += 0.05
                result.add_reason(f"High volume: {volume} > 150% avg ({avg_volume:.0f})")
        
        return result
    
    def _get_atr_series(self, features: pd.DataFrame) -> Optional[pd.Series]:
        """Return ATR series for dynamic threshold calculations."""
        candidate_cols = [f'ATR_{self.atr_period}', 'ATR_14']
        for col in candidate_cols:
            if col in features.columns:
                series = features[col].dropna()
                if len(series) > 0:
                    return series
        return None
    
    def _compute_min_atr_threshold(self, atr_series: Optional[pd.Series]) -> float:
        """Compute adaptive ATR floor using configured percentile."""
        threshold = self.min_atr_threshold
        if atr_series is None or self.min_atr_percentile is None:
            return threshold
        
        recent = atr_series
        if self.atr_percentile_lookback > 0:
            recent = recent.tail(self.atr_percentile_lookback)
        recent = recent.dropna()
        if len(recent) == 0:
            return threshold
        
        percentile_value = self.min_atr_percentile
        if percentile_value <= 1:
            percentile_value *= 100
        percentile_value = max(0.0, min(100.0, percentile_value))
        
        try:
            dynamic_floor = float(np.percentile(recent, percentile_value))
        except Exception as exc:
            logger.debug(f"ATR percentile calculation failed: {exc}")
            return threshold
        
        if np.isnan(dynamic_floor):
            return threshold
        return max(threshold, dynamic_floor)
    
    def _get_price_for_tolerance(self, current_price: Optional[float], latest_row: pd.Series) -> float:
        """Resolve price used when computing EMA tolerance bands."""
        if current_price is not None:
            return float(current_price)
        fallback = latest_row.get('close', latest_row.get('price'))
        try:
            return float(fallback) if fallback is not None else 0.0
        except (TypeError, ValueError):
            return 0.0
    
    def _calculate_ema_tolerance(self, price: float) -> float:
        """Calculate EMA tolerance using configured percentage of price."""
        if price > 0 and self.ema_alignment_tolerance_pct > 0:
            return price * self.ema_alignment_tolerance_pct
        return 0.0
    
    def _is_candle_closed(self) -> bool:
        """Check if we're at a candle boundary (close)."""
        now = datetime.now(timezone.utc)
        seconds_since_minute = now.second
        
        # Consider candle closed if we're in first 5 seconds of a new minute
        # (for 1-minute candles)
        if self.candle_period_seconds == 60:
            if seconds_since_minute < 5:
                # Check we haven't already processed this candle
                minute_boundary = now.replace(second=0, microsecond=0)
                if self._last_candle_close_time != minute_boundary:
                    self._last_candle_close_time = minute_boundary
                    return True
                return False  # Already processed this candle
            return False
        
        # For other periods, use modulo
        seconds_in_period = now.timestamp() % self.candle_period_seconds
        return seconds_in_period < 5
    
    def update_session_levels(self, high: float, low: float):
        """Update intraday session high/low."""
        if self._levels:
            if self._levels.session_high is None or high > self._levels.session_high:
                self._levels.session_high = high
            if self._levels.session_low is None or low < self._levels.session_low:
                self._levels.session_low = low
                
    def reset_session(self):
        """Reset session levels (call at session start)."""
        if self._levels:
            self._levels.session_high = None
            self._levels.session_low = None


def calculate_enhanced_confidence(
    base_confidence: float,
    features: pd.DataFrame,
    action: str,
    price_levels: Optional[PriceLevels] = None,
    current_price: Optional[float] = None,
) -> Tuple[float, List[str]]:
    """
    Calculate enhanced confidence score with multiple factors.
    
    Factors:
    - Trend alignment (EMA)
    - RSI position (oversold/overbought)
    - Volume confirmation
    - ATR normalization
    - Multi-timeframe level proximity
    
    Returns:
        Tuple of (adjusted_confidence, list_of_adjustment_reasons)
    """
    confidence = base_confidence
    reasons = []
    
    if len(features) < 2:
        return confidence, reasons
        
    latest = features.iloc[-1]
    
    # 1. Trend Alignment (EMA 9 vs 20)
    ema_9 = latest.get('EMA_9', None)
    ema_20 = latest.get('EMA_20', None)
    
    if ema_9 is not None and ema_20 is not None:
        ema_diff_pct = (ema_9 - ema_20) / ema_20 * 100 if ema_20 != 0 else 0
        
        if action == "BUY" and ema_diff_pct > 0.1:
            confidence += 0.05
            reasons.append(f"Trend aligned (EMA diff: +{ema_diff_pct:.2f}%)")
        elif action == "SELL" and ema_diff_pct < -0.1:
            confidence += 0.05
            reasons.append(f"Trend aligned (EMA diff: {ema_diff_pct:.2f}%)")
        elif (action == "BUY" and ema_diff_pct < -0.2) or (action == "SELL" and ema_diff_pct > 0.2):
            confidence -= 0.10
            reasons.append(f"Counter-trend (EMA diff: {ema_diff_pct:.2f}%)")
    
    # 2. RSI Confirmation
    rsi = latest.get('RSI_14', 50)
    
    if action == "BUY":
        if rsi < 30:
            confidence += 0.10
            reasons.append(f"Oversold RSI: {rsi:.1f}")
        elif rsi < 40:
            confidence += 0.05
            reasons.append(f"RSI supportive: {rsi:.1f}")
        elif rsi > 70:
            confidence -= 0.15
            reasons.append(f"Overbought RSI: {rsi:.1f}")
    elif action == "SELL":
        if rsi > 70:
            confidence += 0.10
            reasons.append(f"Overbought RSI: {rsi:.1f}")
        elif rsi > 60:
            confidence += 0.05
            reasons.append(f"RSI supportive: {rsi:.1f}")
        elif rsi < 30:
            confidence -= 0.15
            reasons.append(f"Oversold RSI: {rsi:.1f}")
    
    # 3. Volume Confirmation
    volume = latest.get('volume', 0)
    if 'volume' in features.columns:
        avg_vol = features['volume'].rolling(20).mean().iloc[-1]
        if avg_vol > 0 and volume > 0:
            vol_ratio = volume / avg_vol
            if vol_ratio > 2.0:
                confidence += 0.08
                reasons.append(f"High volume: {vol_ratio:.1f}x avg")
            elif vol_ratio > 1.5:
                confidence += 0.04
                reasons.append(f"Above avg volume: {vol_ratio:.1f}x")
            elif vol_ratio < 0.5:
                confidence -= 0.05
                reasons.append(f"Low volume: {vol_ratio:.1f}x avg")
    
    # 4. ATR-based volatility check
    atr = latest.get('ATR_14', 0)
    close = latest.get('close', current_price or 0)
    
    if atr > 0 and close > 0:
        atr_pct = atr / close * 100
        if 0.1 < atr_pct < 0.5:
            confidence += 0.03
            reasons.append(f"Good volatility: ATR {atr_pct:.2f}%")
        elif atr_pct > 1.0:
            confidence -= 0.10
            reasons.append(f"High volatility: ATR {atr_pct:.2f}%")
    
    # 5. Multi-timeframe level context
    if price_levels and price_levels.is_valid() and current_price:
        pdh = price_levels.pdh
        pdl = price_levels.pdl
        range_size = pdh - pdl
        
        # Position in range (0 = at PDL, 1 = at PDH)
        if range_size > 0:
            position_in_range = (current_price - pdl) / range_size
            
            if action == "BUY":
                if position_in_range < 0.3:
                    confidence += 0.08
                    reasons.append(f"Near range low: {position_in_range:.1%}")
                elif position_in_range > 0.8:
                    confidence -= 0.10
                    reasons.append(f"Near range high: {position_in_range:.1%}")
            elif action == "SELL":
                if position_in_range > 0.7:
                    confidence += 0.08
                    reasons.append(f"Near range high: {position_in_range:.1%}")
                elif position_in_range < 0.2:
                    confidence -= 0.10
                    reasons.append(f"Near range low: {position_in_range:.1%}")
    
    # Clamp confidence to valid range
    confidence = max(0.0, min(1.0, confidence))
    
    return confidence, reasons
