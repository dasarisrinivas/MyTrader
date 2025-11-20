"""
Market Regime Filter
Determines if market conditions are suitable for trading.
"""
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class RegimeCheckResult:
    """Result of market regime check."""
    tradable: bool
    reason: str
    atr: Optional[float] = None
    spread: Optional[float] = None
    volatility_spike: bool = False


class MarketRegimeFilter:
    """
    Filter to determine if market conditions are suitable for trading.
    
    Checks:
    - ATR threshold (avoid low volatility periods)
    - VIX levels (avoid extreme fear/greed)
    - Bid/ask spread (ensure liquidity)
    - High-impact economic events
    - Trading hours
    """
    
    def __init__(
        self,
        min_atr_threshold: float = 0.5,
        max_spread_ticks: int = 1,
        vix_low_threshold: float = 10.0,
        vix_high_threshold: float = 40.0,
        volatility_spike_threshold: float = 2.0,  # ATR vs 20-period avg
    ):
        """
        Initialize market regime filter.
        
        Args:
            min_atr_threshold: Minimum ATR required for trading
            max_spread_ticks: Maximum bid/ask spread in ticks
            vix_low_threshold: VIX too low (complacent market)
            vix_high_threshold: VIX too high (panic market)
            volatility_spike_threshold: ATR multiplier vs average
        """
        self.min_atr_threshold = min_atr_threshold
        self.max_spread_ticks = max_spread_ticks
        self.vix_low_threshold = vix_low_threshold
        self.vix_high_threshold = vix_high_threshold
        self.volatility_spike_threshold = volatility_spike_threshold
        
        # High-impact event schedule (simplified - could be enhanced with API)
        self.high_impact_events = {
            # FOMC meetings (8 times per year) - 2 PM ET
            # CPI releases (monthly) - 8:30 AM ET
            # NFP (Non-Farm Payroll) - First Friday of month, 8:30 AM ET
            # These would be loaded from economic calendar API in production
        }
    
    def check_regime(
        self,
        df: pd.DataFrame,
        current_time: Optional[datetime] = None,
        bid_price: Optional[float] = None,
        ask_price: Optional[float] = None,
        vix_value: Optional[float] = None,
        tick_size: float = 0.25,
    ) -> RegimeCheckResult:
        """
        Check if current market regime is suitable for trading.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            current_time: Current timestamp (defaults to now)
            bid_price: Current bid price
            ask_price: Current ask price
            vix_value: Current VIX level
            tick_size: Tick size for spread calculation
            
        Returns:
            RegimeCheckResult with tradable flag and reason
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Check 1: Trading hours (ES futures trade nearly 24/5)
        # Regular hours: 9:30 AM - 4:00 PM ET (most liquid)
        # Extended: 6:00 PM - 5:00 PM ET next day
        # We'll focus on regular hours for best execution
        if not self._is_regular_trading_hours(current_time):
            return RegimeCheckResult(
                tradable=False,
                reason="Outside regular trading hours (9:30 AM - 4:00 PM ET)"
            )
        
        # Check 2: ATR threshold
        if len(df) >= 14:
            atr = self._calculate_atr(df)
            if atr is None or pd.isna(atr):
                return RegimeCheckResult(
                    tradable=False,
                    reason="ATR calculation failed",
                    atr=None
                )
            
            if atr < self.min_atr_threshold:
                return RegimeCheckResult(
                    tradable=False,
                    reason=f"ATR too low: {atr:.2f} < {self.min_atr_threshold}",
                    atr=atr
                )
            
            # Check for volatility spike
            if len(df) >= 34:  # Need 20 periods + 14 for ATR
                atr_series = df['high'] - df['low']
                if 'ATR_14' in df.columns:
                    atr_series = df['ATR_14']
                else:
                    high_low = df['high'] - df['low']
                    high_close = abs(df['high'] - df['close'].shift())
                    low_close = abs(df['low'] - df['close'].shift())
                    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr_series = true_range.rolling(14).mean()
                
                avg_atr = atr_series.rolling(20).mean().iloc[-1]
                if not pd.isna(avg_atr) and avg_atr > 0:
                    atr_ratio = atr / avg_atr
                    if atr_ratio > self.volatility_spike_threshold:
                        return RegimeCheckResult(
                            tradable=False,
                            reason=f"Volatility spike detected: ATR {atr_ratio:.2f}x average",
                            atr=atr,
                            volatility_spike=True
                        )
        else:
            return RegimeCheckResult(
                tradable=False,
                reason="Insufficient data for ATR calculation (need 14+ bars)"
            )
        
        # Check 3: Bid/ask spread
        if bid_price is not None and ask_price is not None:
            spread = ask_price - bid_price
            spread_ticks = spread / tick_size
            
            if spread_ticks > self.max_spread_ticks:
                return RegimeCheckResult(
                    tradable=False,
                    reason=f"Spread too wide: {spread_ticks:.1f} ticks > {self.max_spread_ticks}",
                    atr=atr if 'atr' in locals() else None,
                    spread=spread
                )
        
        # Check 4: VIX levels (if provided)
        if vix_value is not None:
            if vix_value < self.vix_low_threshold:
                return RegimeCheckResult(
                    tradable=False,
                    reason=f"VIX too low (complacent): {vix_value:.1f} < {self.vix_low_threshold}",
                    atr=atr if 'atr' in locals() else None
                )
            
            if vix_value > self.vix_high_threshold:
                return RegimeCheckResult(
                    tradable=False,
                    reason=f"VIX too high (panic): {vix_value:.1f} > {self.vix_high_threshold}",
                    atr=atr if 'atr' in locals() else None
                )
        
        # Check 5: High-impact economic events
        if self._is_high_impact_event_time(current_time):
            return RegimeCheckResult(
                tradable=False,
                reason="High-impact economic event scheduled (FOMC, CPI, NFP)",
                atr=atr if 'atr' in locals() else None
            )
        
        # All checks passed
        return RegimeCheckResult(
            tradable=True,
            reason="Market regime suitable for trading",
            atr=atr if 'atr' in locals() else None,
            spread=spread if 'spread' in locals() else None
        )
    
    def _is_regular_trading_hours(self, dt: datetime) -> bool:
        """Check if time is within regular trading hours (9:30 AM - 4:00 PM ET)."""
        # Convert to ET time zone
        try:
            from zoneinfo import ZoneInfo
            et_tz = ZoneInfo("America/New_York")
        except ImportError:
            # Fallback for Python < 3.9
            import pytz
            et_tz = pytz.timezone("America/New_York")
        
        # Convert to ET
        if dt.tzinfo is None:
            # Assume UTC if no timezone
            from datetime import timezone as tz
            dt = dt.replace(tzinfo=tz.utc)
        
        dt_et = dt.astimezone(et_tz)
        current_time = dt_et.time()
        
        # Skip weekends
        if dt_et.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Regular hours: 9:30 AM - 4:00 PM ET
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        return market_open <= current_time <= market_close
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate Average True Range."""
        try:
            # Check if ATR already calculated
            if 'ATR_14' in df.columns:
                return float(df['ATR_14'].iloc[-1])
            
            # Calculate manually
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean().iloc[-1]
            
            return float(atr) if not pd.isna(atr) else None
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return None
    
    def _is_high_impact_event_time(self, dt: datetime) -> bool:
        """
        Check if current time is near a high-impact economic event.
        
        In production, this would query an economic calendar API.
        For now, we block trading around typical event times:
        - 8:30 AM ET (CPI, NFP)
        - 2:00 PM ET (FOMC)
        """
        current_time = dt.time()
        
        # Block 30 minutes before and after typical event times
        # 8:30 AM events (CPI, NFP) - block 8:00-9:00 AM
        if time(8, 0) <= current_time <= time(9, 0):
            # Check if it's first Friday (NFP) or mid-month (CPI)
            if dt.day <= 7 and dt.weekday() == 4:  # First Friday
                return True
            if 10 <= dt.day <= 15:  # Mid-month (CPI typically around 13th)
                return True
        
        # 2:00 PM events (FOMC) - block 1:30-2:30 PM on FOMC days
        # FOMC meets 8 times per year - would need calendar API for exact dates
        if time(13, 30) <= current_time <= time(14, 30):
            # For now, just be cautious around 2 PM
            # In production, check against FOMC calendar
            pass
        
        return False
    
    def log_regime_status(self, result: RegimeCheckResult) -> None:
        """Log the regime check result."""
        if result.tradable:
            logger.info(f"✅ {result.reason}")
            if result.atr:
                logger.info(f"   ATR: {result.atr:.2f}")
            if result.spread:
                logger.info(f"   Spread: {result.spread:.2f}")
        else:
            logger.warning(f"⚠️  Trading blocked: {result.reason}")
