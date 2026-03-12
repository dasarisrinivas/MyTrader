"""
Market Regime Filter
Determines if market conditions are suitable for trading.
"""
from dataclasses import dataclass
from datetime import datetime, time
from typing import List, Optional, Tuple

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
        high_impact_event_dates: Optional[List[str]] = None,
    ):
        """
        Initialize market regime filter.
        
        Args:
            min_atr_threshold: Minimum ATR required for trading
            max_spread_ticks: Maximum bid/ask spread in ticks
            vix_low_threshold: VIX too low (complacent market)
            vix_high_threshold: VIX too high (panic market)
            volatility_spike_threshold: ATR multiplier vs average
            high_impact_event_dates: Explicit list of dates (YYYY-MM-DD) to
                treat as high-impact event days, overriding heuristics.
                Load from config or env ``HIGH_IMPACT_DATES`` to block
                trading on known event days that the heuristic would miss.
                Example: ["2026-02-20", "2026-03-19"]
        """
        self.min_atr_threshold = min_atr_threshold
        self.max_spread_ticks = max_spread_ticks
        self.vix_low_threshold = vix_low_threshold
        self.vix_high_threshold = vix_high_threshold
        self.volatility_spike_threshold = volatility_spike_threshold
        
        # Manual override dates — always block 8:00-9:15 AM ET on these days
        import os as _os
        env_dates = _os.environ.get("HIGH_IMPACT_DATES", "")
        raw_dates = (high_impact_event_dates or []) + [
            d.strip() for d in env_dates.split(",") if d.strip()
        ]
        self._high_impact_override_dates: set = set()
        for d in raw_dates:
            try:
                from datetime import date as _date
                self._high_impact_override_dates.add(_date.fromisoformat(d))
            except (ValueError, TypeError):
                logger.warning(f"Ignoring invalid high-impact date: {d!r}")

        # High-impact event schedule (simplified - could be enhanced with API)
        self.high_impact_events = {
            # FOMC meetings (8 times per year) - 2 PM ET
            # CPI releases (monthly) - 8:30 AM ET
            # NFP (Non-Farm Payroll) - First Friday of month, 8:30 AM ET
            # Core PCE - Last Friday of month, 8:30 AM ET
            # GDP Advance - End of month, 8:30 AM ET
            # These are now detected heuristically + via manual overrides
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
        # For 24h operation, we no longer block outside RTH
        # Instead, we just note the session for parameter adjustments
        is_rth = self._is_regular_trading_hours(current_time)
        is_market_open = self._is_market_open(current_time)
        
        if not is_market_open:
            return RegimeCheckResult(
                tradable=False,
                reason="Market closed (maintenance window or weekend)"
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
    
    def _is_market_open(self, dt: datetime) -> bool:
        """
        Check if ES/MES market is open (24h except maintenance and weekends).
        
        CME ES/MES Schedule:
        - Sunday 5:00 PM CT to Friday 4:00 PM CT
        - Daily maintenance: 4:00 PM - 5:00 PM CT
        """
        try:
            from zoneinfo import ZoneInfo
            ct_tz = ZoneInfo("America/Chicago")
        except ImportError:
            import pytz
            ct_tz = pytz.timezone("America/Chicago")
        
        if dt.tzinfo is None:
            from datetime import timezone as tz
            dt = dt.replace(tzinfo=tz.utc)
        
        dt_ct = dt.astimezone(ct_tz)
        current_time = dt_ct.time()
        weekday = dt_ct.weekday()  # 0=Monday, 6=Sunday
        
        # Weekend: Saturday all day, Sunday before 5 PM CT
        if weekday == 5:  # Saturday
            return False
        if weekday == 6 and current_time < time(17, 0):  # Sunday before 5 PM
            return False
        
        # Friday after 4 PM CT (close for weekend)
        if weekday == 4 and current_time >= time(16, 0):
            return False
        
        # Daily maintenance: 4:00 PM - 5:00 PM CT
        if time(16, 0) <= current_time < time(17, 0):
            return False
        
        return True
    
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
        
        Blocks trading ±30 min around known high-impact release windows.
        Covers: NFP, CPI, Core PCE, GDP Advance, FOMC.
        
        FEB 20 2026 FIX: Added Core PCE (last Friday of month, 8:30 AM ET)
        and GDP Advance Estimate (end-of-month, 8:30 AM ET).  Previous
        version only checked NFP and CPI, leaving the bot exposed to the
        two most market-moving macro prints after NFP.
        
        NOTE: In production, replace with an economic-calendar API for
        exact release dates.  The heuristics below cover ~90 % of cases.
        """
        try:
            from zoneinfo import ZoneInfo
            et_tz = ZoneInfo("America/New_York")
        except ImportError:
            import pytz
            et_tz = pytz.timezone("America/New_York")

        # Ensure we're working in ET (release times are ET-based)
        if dt.tzinfo is None:
            from datetime import timezone as tz
            dt = dt.replace(tzinfo=tz.utc)
        dt_et = dt.astimezone(et_tz)
        current_time = dt_et.time()
        
        import calendar as _cal
        
        # ── Manual override dates (from config or env HIGH_IMPACT_DATES) ──
        # MAR 12 2026: Extended post-release window to 10:00 AM ET.
        # Rationale: CPI/NFP at 8:30 ET causes market whipsaw through the
        # full RTH open period (9:30 ET open + 30 min). Previous 9:15 ET
        # cutoff left the bot exposed to post-open continuation spikes.
        if self._high_impact_override_dates and dt_et.date() in self._high_impact_override_dates:
            if time(8, 0) <= current_time <= time(10, 0):
                logger.info(f"🚫 High-impact event: manual override for {dt_et.date()}")
                return True

        # ── 8:30 AM ET window (block 8:00 – 10:00 AM ET) ─────────────
        # MAR 12 2026: Extended from 9:15 → 10:00 AM ET.
        # 8:30 releases (CPI, NFP, PCE, GDP) drive volatility through the
        # 9:30 RTH open and the first 30 min of cash trading.  Entering at
        # 9:00–9:30 ET (8:00–8:30 CST) on these days is the highest-risk
        # window — OR breakout signals fire on the post-release spike which
        # then reverses sharply once the initial reaction exhausts.
        if time(8, 0) <= current_time <= time(10, 0):
            # NFP — First Friday of month, 8:30 AM ET
            if dt_et.day <= 7 and dt_et.weekday() == 4:
                logger.info("🚫 High-impact event: NFP (first Friday)")
                return True

            # CPI — Typically 10th-15th of month, 8:30 AM ET
            if 10 <= dt_et.day <= 15:
                logger.info("🚫 High-impact event: CPI window (mid-month)")
                return True

            # Core PCE — Last Friday of month, 8:30 AM ET
            # Detect last Friday: the next Friday would be in the next month
            if dt_et.weekday() == 4:  # Friday
                _, month_last_day = _cal.monthrange(dt_et.year, dt_et.month)
                if dt_et.day + 7 > month_last_day:
                    logger.info("🚫 High-impact event: Core PCE (last Friday)")
                    return True

            # GDP Advance Estimate — Typically last week of month, 8:30 AM ET
            _, month_last_day = _cal.monthrange(dt_et.year, dt_et.month)
            if dt_et.day >= month_last_day - 6:
                # Only flag weekdays in last 7 calendar days
                if dt_et.weekday() < 5:
                    logger.info("🚫 High-impact event: GDP window (end-of-month)")
                    return True
        
        # ── 2:00 PM ET window (block 1:30 – 2:45 PM ET) ─────────────
        # FOMC rate decisions — 8 times/year.  Without a calendar API we
        # cannot know the exact dates, but we can flag Wed afternoons in
        # FOMC-heavy months (Jan, Mar, May, Jun, Jul, Sep, Nov, Dec).
        if time(13, 30) <= current_time <= time(14, 45):
            fomc_months = {1, 3, 5, 6, 7, 9, 11, 12}
            if dt_et.month in fomc_months and dt_et.weekday() == 2:
                # Wednesdays in FOMC-heavy months → cautious block
                logger.info("🚫 High-impact event: possible FOMC window")
                return True
        
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
