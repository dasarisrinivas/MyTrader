"""Multi-timeframe candle aggregation.

Created: Jan 2, 2026
Updated: Jan 12, 2026 - Added support for 15m and 30m timeframes with trend hierarchy

Purpose: Aggregate 1-minute candles into higher timeframes (5m, 15m, 30m) for
multi-timeframe trend confirmation. This addresses the audit finding that 1-minute 
signals were too noisy and caused frequent whipsaw losses.

TREND AUTHORITY HIERARCHY (Jan 12, 2026):
- 15m candle = PRIMARY trend authority (must agree for any trade)
- 30m candle = CONFIRMATION trend (must not contradict)
- 5m candle = Alignment and momentum confirmation
- 1m candle = ENTRY TIMING ONLY (NEVER determines trend direction)

Usage:
    aggregator = MultiTimeframeCandleBuilder(base_interval=1, target_interval=5)
    aggregator.add_bar(timestamp, open, high, low, close, volume)
    
    if aggregator.has_complete_candle():
        candle_5m = aggregator.get_latest_candle()
        trend_5m = aggregator.get_trend()
    
    # For full MTF analysis:
    mtf_manager = MTFCandleManager()
    mtf_manager.add_1m_bar(timestamp, open, high, low, close, volume)
    trends = mtf_manager.get_all_trends()  # Returns trends for 5m, 15m, 30m
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logger import logger


@dataclass
class AggregatedCandle:
    """A completed aggregated candle."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bar_count: int  # Number of base bars that make up this candle
    
    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def range_size(self) -> float:
        return self.high - self.low
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


@dataclass
class MultiTimeframeState:
    """State for the multi-timeframe candle builder."""
    current_open: Optional[float] = None
    current_high: float = float("-inf")
    current_low: float = float("inf")
    current_close: Optional[float] = None
    current_volume: float = 0.0
    current_bar_count: int = 0
    current_period_start: Optional[datetime] = None


class MultiTimeframeCandleBuilder:
    """Aggregates base-interval candles (e.g., 1-min) into target-interval candles (e.g., 5-min).
    
    This enables multi-timeframe analysis where:
    - 5-min candles determine overall trend direction
    - 1-min candles are used for precise entry timing
    
    The 5-min trend filter would have prevented 4 of 5 losing trades
    in the Jan 2, 2026 audit session.
    """
    
    def __init__(
        self,
        base_interval: int = 1,
        target_interval: int = 5,
        ema_period: int = 20,
        max_history: int = 100,
    ):
        """Initialize the multi-timeframe candle builder.
        
        Args:
            base_interval: Base candle interval in minutes (typically 1)
            target_interval: Target candle interval in minutes (typically 5)
            ema_period: EMA period for trend calculation on target timeframe
            max_history: Maximum number of aggregated candles to retain
        """
        if target_interval % base_interval != 0:
            raise ValueError(
                f"Target interval ({target_interval}) must be divisible by "
                f"base interval ({base_interval})"
            )
        
        self.base_interval = base_interval
        self.target_interval = target_interval
        self.bars_per_candle = target_interval // base_interval
        self.ema_period = ema_period
        self.max_history = max_history
        
        self._state = MultiTimeframeState()
        self._candles: Deque[AggregatedCandle] = deque(maxlen=max_history)
        self._ema_values: Deque[float] = deque(maxlen=max_history)
        self._last_completed_candle: Optional[AggregatedCandle] = None
        
        logger.info(
            f"MultiTimeframeCandleBuilder initialized: {base_interval}m -> {target_interval}m "
            f"(EMA period={ema_period})"
        )
    
    def _get_period_start(self, timestamp: datetime) -> datetime:
        """Get the start of the target-interval period containing this timestamp."""
        # Align to target interval boundaries
        minute = timestamp.minute
        period_minute = (minute // self.target_interval) * self.target_interval
        return timestamp.replace(minute=period_minute, second=0, microsecond=0)
    
    def add_bar(
        self,
        timestamp: datetime,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float = 0.0,
    ) -> Optional[AggregatedCandle]:
        """Add a base-interval bar and return completed candle if any.
        
        Args:
            timestamp: Bar timestamp
            open_price: Open price
            high_price: High price
            low_price: Low price
            close_price: Close price
            volume: Volume (optional)
        
        Returns:
            AggregatedCandle if a target-interval candle was completed, else None
        """
        period_start = self._get_period_start(timestamp)
        
        # Check if we're starting a new period
        if self._state.current_period_start is None:
            # First bar ever
            self._state.current_period_start = period_start
            self._state.current_open = open_price
            self._state.current_high = high_price
            self._state.current_low = low_price
            self._state.current_close = close_price
            self._state.current_volume = volume
            self._state.current_bar_count = 1
            return None
        
        if period_start != self._state.current_period_start:
            # New period - complete the previous candle
            completed = self._complete_candle()
            
            # Start new period
            self._state.current_period_start = period_start
            self._state.current_open = open_price
            self._state.current_high = high_price
            self._state.current_low = low_price
            self._state.current_close = close_price
            self._state.current_volume = volume
            self._state.current_bar_count = 1
            
            return completed
        
        # Same period - update OHLCV
        self._state.current_high = max(self._state.current_high, high_price)
        self._state.current_low = min(self._state.current_low, low_price)
        self._state.current_close = close_price
        self._state.current_volume += volume
        self._state.current_bar_count += 1
        
        return None
    
    def _complete_candle(self) -> Optional[AggregatedCandle]:
        """Complete the current candle and add to history."""
        if self._state.current_open is None or self._state.current_period_start is None:
            return None
        
        candle = AggregatedCandle(
            timestamp=self._state.current_period_start,
            open=self._state.current_open,
            high=self._state.current_high,
            low=self._state.current_low,
            close=self._state.current_close or self._state.current_open,
            volume=self._state.current_volume,
            bar_count=self._state.current_bar_count,
        )
        
        self._candles.append(candle)
        self._last_completed_candle = candle
        
        # Update EMA
        self._update_ema(candle.close)
        
        logger.debug(
            f"Completed {self.target_interval}m candle: "
            f"O={candle.open:.2f} H={candle.high:.2f} L={candle.low:.2f} C={candle.close:.2f} "
            f"(bars={candle.bar_count})"
        )
        
        return candle
    
    def _update_ema(self, close: float) -> None:
        """Update EMA with new close price."""
        if not self._ema_values:
            self._ema_values.append(close)
            return
        
        multiplier = 2.0 / (self.ema_period + 1)
        prev_ema = self._ema_values[-1]
        new_ema = (close - prev_ema) * multiplier + prev_ema
        self._ema_values.append(new_ema)
    
    def has_complete_candle(self) -> bool:
        """Check if there's at least one completed target-interval candle."""
        return len(self._candles) > 0
    
    def get_latest_candle(self) -> Optional[AggregatedCandle]:
        """Get the most recently completed candle."""
        return self._last_completed_candle
    
    def get_candles(self, n: int = 0) -> List[AggregatedCandle]:
        """Get the last n candles (or all if n=0)."""
        if n <= 0:
            return list(self._candles)
        return list(self._candles)[-n:]
    
    def get_ema(self) -> Optional[float]:
        """Get the current EMA value."""
        return self._ema_values[-1] if self._ema_values else None
    
    def get_trend(self) -> str:
        """Determine the trend based on EMA and price action.
        
        Returns:
            'UPTREND', 'DOWNTREND', or 'NEUTRAL'
        """
        if len(self._candles) < 3 or not self._ema_values:
            return "NEUTRAL"
        
        current_ema = self._ema_values[-1]
        last_candle = self._candles[-1]
        prev_candle = self._candles[-2]
        tolerance_pct = 0.002  # Allow minor pullbacks (0.2%)
        
        # Price above EMA and making higher lows (with small tolerance)
        if last_candle.close > current_ema and last_candle.low >= prev_candle.low * (1 - tolerance_pct):
            return "UPTREND"
        
        # Price below EMA and making lower highs
        if last_candle.close < current_ema and last_candle.high < prev_candle.high:
            return "DOWNTREND"
        
        # Check EMA slope
        if len(self._ema_values) >= 3:
            ema_slope = self._ema_values[-1] - self._ema_values[-3]
            if ema_slope > 0.5:  # Rising EMA
                return "UPTREND"
            elif ema_slope < -0.5:  # Falling EMA
                return "DOWNTREND"
        
        return "NEUTRAL"
    
    def is_trend_aligned(self, action: str) -> Tuple[bool, str]:
        """Check if a trading action aligns with the higher-timeframe trend.
        
        Args:
            action: 'BUY', 'SELL', 'SCALP_BUY', or 'SCALP_SELL'
        
        Returns:
            Tuple of (is_aligned, reason_string)
        """
        trend = self.get_trend()
        is_buy = action.upper() in ("BUY", "SCALP_BUY")
        is_sell = action.upper() in ("SELL", "SCALP_SELL")
        
        if trend == "NEUTRAL":
            return True, f"{self.target_interval}m_NEUTRAL"
        
        if is_buy and trend == "UPTREND":
            return True, f"{self.target_interval}m_BUY_IN_UPTREND"
        
        if is_sell and trend == "DOWNTREND":
            return True, f"{self.target_interval}m_SELL_IN_DOWNTREND"
        
        # Counter-trend
        if is_buy and trend == "DOWNTREND":
            return False, f"COUNTER_TREND:{self.target_interval}m_BUY_IN_DOWNTREND"
        
        if is_sell and trend == "UPTREND":
            return False, f"COUNTER_TREND:{self.target_interval}m_SELL_IN_UPTREND"
        
        return True, f"{self.target_interval}m_UNKNOWN"
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert candle history to DataFrame for analysis."""
        if not self._candles:
            return pd.DataFrame()
        
        data = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "bar_count": c.bar_count,
            }
            for c in self._candles
        ]
        
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        
        # Add EMA column
        if self._ema_values:
            ema_list = list(self._ema_values)
            # Pad with NaN if needed
            while len(ema_list) < len(df):
                ema_list.insert(0, np.nan)
            df[f"EMA_{self.ema_period}"] = ema_list[-len(df):]
        
        return df
    
    def reset(self) -> None:
        """Reset all state."""
        self._state = MultiTimeframeState()
        self._candles.clear()
        self._ema_values.clear()
        self._last_completed_candle = None
        logger.info("MultiTimeframeCandleBuilder reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the aggregated candles."""
        if not self._candles:
            return {}
        
        candles = list(self._candles)
        closes = [c.close for c in candles]
        ranges = [c.range_size for c in candles]
        
        bullish_count = sum(1 for c in candles if c.is_bullish)
        bearish_count = sum(1 for c in candles if c.is_bearish)
        
        return {
            "candle_count": len(candles),
            "avg_range": np.mean(ranges) if ranges else 0,
            "max_range": max(ranges) if ranges else 0,
            "min_range": min(ranges) if ranges else 0,
            "bullish_pct": bullish_count / len(candles) if candles else 0,
            "bearish_pct": bearish_count / len(candles) if candles else 0,
            "current_ema": self.get_ema(),
            "current_trend": self.get_trend(),
            "last_close": closes[-1] if closes else None,
        }


@dataclass
class MTFTrendData:
    """Trend data for a single timeframe."""
    timeframe: str  # "5m", "15m", "30m"
    trend: str  # "UPTREND", "DOWNTREND", "NEUTRAL"
    confidence: float  # 0.0 to 1.0
    ema_value: Optional[float]
    last_close: Optional[float]
    candle_close_time: Optional[datetime]
    candle_count: int
    is_valid: bool  # Has enough data for reliable trend


class MTFCandleManager:
    """Manages multiple timeframe candle aggregators for full MTF analysis.
    
    Creates and maintains 5m, 15m, and 30m aggregators from 1m bar data.
    Provides unified interface for trend checks across all timeframes.
    
    TREND AUTHORITY HIERARCHY:
    - 15m = PRIMARY (must be bullish for longs, bearish for shorts)
    - 30m = CONFIRMATION (must not contradict 15m)
    - 5m = ALIGNMENT (momentum confirmation)
    - 1m = ENTRY TIMING ONLY (this manager doesn't track 1m trend)
    """
    
    MIN_CANDLES_FOR_TREND = {
        "5m": 3,   # Need 3 x 5m candles (15 minutes of data)
        "15m": 2,  # Need 2 x 15m candles (30 minutes of data)
        "30m": 2,  # Need 2 x 30m candles (60 minutes of data)
    }
    
    def __init__(
        self,
        ema_period_5m: int = 20,
        ema_period_15m: int = 20,
        ema_period_30m: int = 20,
        max_history: int = 100,
    ):
        """Initialize the MTF manager with all timeframe aggregators.
        
        Args:
            ema_period_5m: EMA period for 5-minute trend
            ema_period_15m: EMA period for 15-minute trend
            ema_period_30m: EMA period for 30-minute trend
            max_history: Maximum candles to retain per timeframe
        """
        self._builders: Dict[str, MultiTimeframeCandleBuilder] = {
            "5m": MultiTimeframeCandleBuilder(
                base_interval=1,
                target_interval=5,
                ema_period=ema_period_5m,
                max_history=max_history,
            ),
            "15m": MultiTimeframeCandleBuilder(
                base_interval=1,
                target_interval=15,
                ema_period=ema_period_15m,
                max_history=max_history,
            ),
            "30m": MultiTimeframeCandleBuilder(
                base_interval=1,
                target_interval=30,
                ema_period=ema_period_30m,
                max_history=max_history,
            ),
        }
        
        # Track completed candle timestamps for each timeframe
        self._last_candle_times: Dict[str, Optional[datetime]] = {
            "5m": None,
            "15m": None,
            "30m": None,
        }
        
        logger.info(
            f"MTFCandleManager initialized: 5m(ema={ema_period_5m}), "
            f"15m(ema={ema_period_15m}), 30m(ema={ema_period_30m})"
        )
    
    def add_1m_bar(
        self,
        timestamp: datetime,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float = 0.0,
    ) -> Dict[str, Optional[AggregatedCandle]]:
        """Add a 1-minute bar and return any completed candles.
        
        Args:
            timestamp: Bar timestamp
            open_price: Open price
            high_price: High price  
            low_price: Low price
            close_price: Close price
            volume: Volume
        
        Returns:
            Dict mapping timeframe to completed candle (None if not complete)
        """
        completed: Dict[str, Optional[AggregatedCandle]] = {}
        
        for tf, builder in self._builders.items():
            candle = builder.add_bar(
                timestamp=timestamp,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                volume=volume,
            )
            completed[tf] = candle
            
            if candle is not None:
                self._last_candle_times[tf] = candle.timestamp
                logger.info(
                    f"📊 {tf.upper()} CANDLE COMPLETE: "
                    f"O={candle.open:.2f} H={candle.high:.2f} "
                    f"L={candle.low:.2f} C={candle.close:.2f} | "
                    f"Trend: {builder.get_trend()}"
                )
        
        return completed
    
    def get_trend(self, timeframe: str) -> str:
        """Get trend for a specific timeframe.
        
        Args:
            timeframe: "5m", "15m", or "30m"
        
        Returns:
            "UPTREND", "DOWNTREND", or "NEUTRAL"
        """
        if timeframe not in self._builders:
            return "UNKNOWN"
        return self._builders[timeframe].get_trend()
    
    def get_all_trends(self) -> Dict[str, MTFTrendData]:
        """Get trend data for all timeframes.
        
        Returns:
            Dict mapping timeframe to MTFTrendData
        """
        trends: Dict[str, MTFTrendData] = {}
        
        for tf, builder in self._builders.items():
            min_candles = self.MIN_CANDLES_FOR_TREND.get(tf, 3)
            candle_count = len(builder._candles)
            is_valid = candle_count >= min_candles
            
            trend = builder.get_trend() if is_valid else "UNKNOWN"
            
            # Calculate confidence based on EMA slope and candle count
            confidence = self._calculate_trend_confidence(builder, is_valid)
            
            latest = builder.get_latest_candle()
            
            trends[tf] = MTFTrendData(
                timeframe=tf,
                trend=trend,
                confidence=confidence,
                ema_value=builder.get_ema(),
                last_close=latest.close if latest else None,
                candle_close_time=self._last_candle_times.get(tf),
                candle_count=candle_count,
                is_valid=is_valid,
            )
        
        return trends
    
    def _calculate_trend_confidence(
        self,
        builder: MultiTimeframeCandleBuilder,
        is_valid: bool,
    ) -> float:
        """Calculate trend confidence based on EMA slope and consistency."""
        if not is_valid or len(builder._ema_values) < 3:
            return 0.0
        
        # Calculate EMA slope (normalized)
        ema_slope = builder._ema_values[-1] - builder._ema_values[-3]
        
        # Count consistent candles
        candles = list(builder._candles)[-5:]
        trend = builder.get_trend()
        
        if trend == "UPTREND":
            consistent = sum(1 for c in candles if c.is_bullish)
        elif trend == "DOWNTREND":
            consistent = sum(1 for c in candles if c.is_bearish)
        else:
            consistent = 0
        
        # Confidence formula: EMA slope strength + consistency
        slope_factor = min(1.0, abs(ema_slope) / 5.0)  # Normalize to ~0-1
        consistency_factor = consistent / len(candles) if candles else 0
        
        confidence = (slope_factor * 0.4 + consistency_factor * 0.6)
        return min(1.0, max(0.0, confidence))
    
    def check_long_alignment(self) -> Tuple[bool, str, Dict[str, str]]:
        """Check if all timeframes support a LONG trade.
        
        RULES (enforced by trend hierarchy):
        - 15m must be UPTREND (PRIMARY authority)
        - 30m must NOT be DOWNTREND (CONFIRMATION)
        - 5m should be UPTREND or NEUTRAL (ALIGNMENT)
        
        Returns:
            Tuple of (is_aligned, reason, trend_dict)
        """
        trends = self.get_all_trends()
        trend_dict = {tf: data.trend for tf, data in trends.items()}
        
        # Check 15m PRIMARY authority
        tf_15m = trends.get("15m")
        if not tf_15m or not tf_15m.is_valid:
            return False, "15m_DATA_INSUFFICIENT", trend_dict
        if tf_15m.trend != "UPTREND":
            return False, f"15m_NOT_BULLISH({tf_15m.trend})", trend_dict
        
        # Check 30m CONFIRMATION
        tf_30m = trends.get("30m")
        if tf_30m and tf_30m.is_valid and tf_30m.trend == "DOWNTREND":
            return False, f"30m_BEARISH({tf_30m.trend})", trend_dict
        
        # Check 5m ALIGNMENT (only block if strongly bearish)
        tf_5m = trends.get("5m")
        if tf_5m and tf_5m.is_valid and tf_5m.trend == "DOWNTREND" and tf_5m.confidence > 0.6:
            return False, f"5m_STRONGLY_BEARISH({tf_5m.trend})", trend_dict
        
        return True, "LONG_ALIGNED", trend_dict
    
    def check_short_alignment(self) -> Tuple[bool, str, Dict[str, str]]:
        """Check if all timeframes support a SHORT trade.
        
        RULES (enforced by trend hierarchy):
        - 15m must be DOWNTREND (PRIMARY authority)
        - 30m must NOT be UPTREND (CONFIRMATION)
        - 5m should be DOWNTREND or NEUTRAL (ALIGNMENT)
        
        Returns:
            Tuple of (is_aligned, reason, trend_dict)
        """
        trends = self.get_all_trends()
        trend_dict = {tf: data.trend for tf, data in trends.items()}
        
        # Check 15m PRIMARY authority
        tf_15m = trends.get("15m")
        if not tf_15m or not tf_15m.is_valid:
            return False, "15m_DATA_INSUFFICIENT", trend_dict
        if tf_15m.trend != "DOWNTREND":
            return False, f"15m_NOT_BEARISH({tf_15m.trend})", trend_dict
        
        # Check 30m CONFIRMATION
        tf_30m = trends.get("30m")
        if tf_30m and tf_30m.is_valid and tf_30m.trend == "UPTREND":
            return False, f"30m_BULLISH({tf_30m.trend})", trend_dict
        
        # Check 5m ALIGNMENT (only block if strongly bullish)
        tf_5m = trends.get("5m")
        if tf_5m and tf_5m.is_valid and tf_5m.trend == "UPTREND" and tf_5m.confidence > 0.6:
            return False, f"5m_STRONGLY_BULLISH({tf_5m.trend})", trend_dict
        
        return True, "SHORT_ALIGNED", trend_dict
    
    def has_sufficient_data(self) -> bool:
        """Check if we have enough data for reliable trend analysis."""
        for tf, min_count in self.MIN_CANDLES_FOR_TREND.items():
            builder = self._builders.get(tf)
            if not builder or len(builder._candles) < min_count:
                return False
        return True
    
    def get_last_candle_time(self, timeframe: str) -> Optional[datetime]:
        """Get the timestamp of the last completed candle for a timeframe."""
        return self._last_candle_times.get(timeframe)
    
    def reset(self, reason: str = "manual") -> None:
        """Reset all aggregators."""
        for builder in self._builders.values():
            builder.reset()
        self._last_candle_times = {tf: None for tf in self._builders}
        logger.info(f"MTFCandleManager reset: {reason}")
        
        return {
            "candle_count": len(candles),
            "avg_range": np.mean(ranges) if ranges else 0,
            "max_range": max(ranges) if ranges else 0,
            "min_range": min(ranges) if ranges else 0,
            "bullish_pct": bullish_count / len(candles) if candles else 0,
            "bearish_pct": bearish_count / len(candles) if candles else 0,
            "current_ema": self.get_ema(),
            "current_trend": self.get_trend(),
            "last_close": closes[-1] if closes else None,
        }
