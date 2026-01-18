"""
Continuous Futures Roll Builder
===============================

Builds continuous futures series from individual contract data:
- Handles rolling across contracts for 2+ year backtests
- Supports multiple roll rules (volume crossover, N days before expiry)
- Supports adjustment methods (back-adjust, ratio-adjust, unadjusted)
- Generates roll calendar with dates and adjustment factors
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger


class RollMethod(Enum):
    """Method for determining when to roll contracts."""
    VOLUME_CROSSOVER = "volume_crossover"    # Roll when next contract volume exceeds current
    OPEN_INTEREST = "open_interest"          # Roll when OI crosses over
    DAYS_BEFORE_EXPIRY = "days_before_expiry"  # Fixed N days before expiration
    FIRST_NOTICE = "first_notice"            # Roll on first notice date
    LAST_TRADE = "last_trade"                # Roll on last trade date minus buffer


class AdjustmentMethod(Enum):
    """Method for adjusting prices across roll points."""
    BACK_ADJUST = "back_adjust"      # Add/subtract difference (preserves dollar moves)
    RATIO_ADJUST = "ratio_adjust"    # Multiply by ratio (preserves % moves)
    UNADJUSTED = "unadjusted"        # No adjustment (gaps at rolls)


@dataclass
class RollConfig:
    """Configuration for continuous futures construction."""
    
    # Roll method
    roll_method: RollMethod = RollMethod.DAYS_BEFORE_EXPIRY
    
    # Days before expiry (if using DAYS_BEFORE_EXPIRY method)
    days_before_expiry: int = 5
    
    # Volume/OI lookback for crossover detection
    crossover_lookback_days: int = 3
    crossover_threshold: float = 1.0  # New contract must exceed old by this factor
    
    # Adjustment method
    adjustment_method: AdjustmentMethod = AdjustmentMethod.BACK_ADJUST
    
    # CME ES/MES contract specifications
    contract_months: List[str] = field(default_factory=lambda: ["H", "M", "U", "Z"])  # Mar, Jun, Sep, Dec
    
    # Roll time of day (UTC)
    roll_time_utc: str = "22:00"  # After regular session close
    
    # Whether to include both contracts during roll transition
    overlap_bars: int = 0  # Number of bars to keep from old contract after roll


@dataclass
class RollEvent:
    """Information about a single roll event."""
    roll_date: datetime
    from_contract: str
    to_contract: str
    adjustment_factor: float  # Additive for back-adjust, multiplicative for ratio
    from_price: float
    to_price: float
    method: str


class FuturesRollCalendar:
    """
    Generates roll dates for futures contracts.
    
    CME ES/MES contract months: H (Mar), M (Jun), U (Sep), Z (Dec)
    Expiration: 3rd Friday of contract month at 9:30 AM ET
    """
    
    CONTRACT_MONTHS = {
        "H": 3,   # March
        "M": 6,   # June
        "U": 9,   # September
        "Z": 12,  # December
    }
    
    def __init__(self, symbol: str = "ES"):
        self.symbol = symbol
    
    def get_expiry_date(self, year: int, month_code: str) -> datetime:
        """
        Get the expiration date for a contract.
        
        CME ES/MES: 3rd Friday of contract month at 9:30 AM ET
        """
        month = self.CONTRACT_MONTHS.get(month_code)
        if month is None:
            raise ValueError(f"Invalid month code: {month_code}")
        
        # Find 3rd Friday
        first_day = datetime(year, month, 1, tzinfo=timezone.utc)
        
        # Find first Friday
        days_to_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_to_friday)
        
        # 3rd Friday
        third_friday = first_friday + timedelta(weeks=2)
        
        # Set to 9:30 AM ET (14:30 UTC during EST, 13:30 during EDT)
        third_friday = third_friday.replace(hour=14, minute=30)
        
        return third_friday
    
    def get_contract_code(self, year: int, month_code: str) -> str:
        """Generate contract code like 'ESH24' for Mar 2024."""
        year_suffix = str(year)[-2:]  # Last 2 digits
        return f"{self.symbol}{month_code}{year_suffix}"
    
    def generate_contracts(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """
        Generate list of contracts covering the date range.
        
        Returns list of dicts with contract_code, expiry_date, start_active, end_active
        """
        contracts = []
        month_codes = ["H", "M", "U", "Z"]
        
        # Start from year before to ensure we have contracts for early dates
        start_year = start_date.year - 1
        end_year = end_date.year + 1
        
        for year in range(start_year, end_year + 1):
            for month_code in month_codes:
                expiry = self.get_expiry_date(year, month_code)
                
                contracts.append({
                    "contract_code": self.get_contract_code(year, month_code),
                    "expiry": expiry,
                    "year": year,
                    "month_code": month_code,
                    "month": self.CONTRACT_MONTHS[month_code],
                })
        
        # Sort by expiry
        contracts.sort(key=lambda x: x["expiry"])
        
        # Filter to relevant range
        relevant = []
        for i, contract in enumerate(contracts):
            # Include if expiry is after start_date and contract could be active
            # A contract is typically active from ~2 months before expiry
            active_start = contract["expiry"] - timedelta(days=90)
            
            if contract["expiry"] >= start_date and active_start <= end_date:
                relevant.append(contract)
        
        return relevant
    
    def generate_roll_dates(
        self,
        contracts: List[Dict],
        config: RollConfig
    ) -> List[Dict]:
        """
        Generate roll dates based on configuration.
        
        Returns list of dicts with roll_date, from_contract, to_contract
        """
        if len(contracts) < 2:
            return []
        
        roll_schedule = []
        
        for i in range(len(contracts) - 1):
            current = contracts[i]
            next_contract = contracts[i + 1]
            
            if config.roll_method == RollMethod.DAYS_BEFORE_EXPIRY:
                roll_date = current["expiry"] - timedelta(days=config.days_before_expiry)
            elif config.roll_method == RollMethod.FIRST_NOTICE:
                # First notice is typically 2 business days before expiry
                roll_date = current["expiry"] - timedelta(days=3)
            else:
                # Default to 5 days before expiry
                roll_date = current["expiry"] - timedelta(days=5)
            
            # Set roll time
            hour, minute = map(int, config.roll_time_utc.split(":"))
            roll_date = roll_date.replace(hour=hour, minute=minute)
            
            roll_schedule.append({
                "roll_date": roll_date,
                "from_contract": current["contract_code"],
                "to_contract": next_contract["contract_code"],
                "from_expiry": current["expiry"],
                "to_expiry": next_contract["expiry"],
            })
        
        return roll_schedule


class ContinuousFuturesBuilder:
    """
    Builds continuous futures series from individual contract data.
    
    Key capabilities:
    - Combines multiple contracts into a single continuous series
    - Applies price adjustments at roll points
    - Preserves roll information for analysis
    - Handles gaps and missing data gracefully
    """
    
    def __init__(self, config: Optional[RollConfig] = None, symbol: str = "ES"):
        self.config = config or RollConfig()
        self.symbol = symbol
        self.calendar = FuturesRollCalendar(symbol)
        self.roll_events: List[RollEvent] = []
    
    def build(
        self,
        contract_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        volume_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Build continuous futures series from multiple contracts.
        
        Args:
            contract_data: Dict mapping contract code to price DataFrame
            start_date: Start of desired date range
            end_date: End of desired date range
            volume_data: Optional separate volume data for roll detection
            
        Returns:
            Continuous futures DataFrame with adjusted prices
        """
        if not contract_data:
            raise ValueError("No contract data provided")
        
        self.roll_events = []
        
        # Generate contracts and roll schedule
        contracts = self.calendar.generate_contracts(start_date, end_date)
        roll_schedule = self.calendar.generate_roll_dates(contracts, self.config)
        
        logger.info(f"Building continuous series from {len(contract_data)} contracts")
        logger.info(f"Roll schedule: {len(roll_schedule)} rolls")
        
        # Determine actual roll dates based on method
        if self.config.roll_method == RollMethod.VOLUME_CROSSOVER and volume_data:
            roll_schedule = self._adjust_rolls_by_volume(roll_schedule, volume_data)
        
        # Build the continuous series
        if self.config.adjustment_method == AdjustmentMethod.BACK_ADJUST:
            continuous = self._build_back_adjusted(contract_data, roll_schedule, start_date, end_date)
        elif self.config.adjustment_method == AdjustmentMethod.RATIO_ADJUST:
            continuous = self._build_ratio_adjusted(contract_data, roll_schedule, start_date, end_date)
        else:
            continuous = self._build_unadjusted(contract_data, roll_schedule, start_date, end_date)
        
        # Filter to requested date range
        continuous = continuous[(continuous.index >= start_date) & (continuous.index <= end_date)]
        
        return continuous
    
    def _adjust_rolls_by_volume(
        self,
        roll_schedule: List[Dict],
        volume_data: Dict[str, pd.DataFrame]
    ) -> List[Dict]:
        """Adjust roll dates based on volume crossover."""
        adjusted_schedule = []
        
        for roll_info in roll_schedule:
            from_code = roll_info["from_contract"]
            to_code = roll_info["to_contract"]
            
            if from_code not in volume_data or to_code not in volume_data:
                # Fall back to scheduled date
                adjusted_schedule.append(roll_info)
                continue
            
            from_vol = volume_data[from_code]["volume"]
            to_vol = volume_data[to_code]["volume"]
            
            # Find volume crossover
            combined = pd.DataFrame({
                "from_vol": from_vol,
                "to_vol": to_vol
            }).dropna()
            
            if combined.empty:
                adjusted_schedule.append(roll_info)
                continue
            
            # Use rolling average to smooth
            lookback = self.config.crossover_lookback_days * 390  # ~390 1-min bars per day
            combined["from_avg"] = combined["from_vol"].rolling(lookback, min_periods=1).mean()
            combined["to_avg"] = combined["to_vol"].rolling(lookback, min_periods=1).mean()
            
            # Find crossover
            crossover = combined[
                combined["to_avg"] > combined["from_avg"] * self.config.crossover_threshold
            ]
            
            if not crossover.empty:
                crossover_date = crossover.index[0]
                
                # Must be before scheduled expiry
                if crossover_date < roll_info["from_expiry"]:
                    roll_info["roll_date"] = crossover_date
                    roll_info["roll_method"] = "volume_crossover"
            
            adjusted_schedule.append(roll_info)
        
        return adjusted_schedule
    
    def _build_back_adjusted(
        self,
        contract_data: Dict[str, pd.DataFrame],
        roll_schedule: List[Dict],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Build back-adjusted continuous series.
        
        Back-adjustment preserves dollar moves by adding/subtracting
        the price difference at each roll point.
        """
        # Start with the most recent contract and work backwards
        result_segments = []
        cumulative_adjustment = 0.0
        
        # Process rolls in reverse chronological order
        roll_schedule = sorted(roll_schedule, key=lambda x: x["roll_date"], reverse=True)
        
        for i, roll_info in enumerate(roll_schedule):
            roll_date = roll_info["roll_date"]
            to_code = roll_info["to_contract"]
            from_code = roll_info["from_contract"]
            
            if to_code not in contract_data:
                logger.warning(f"Contract {to_code} not in data, skipping")
                continue
            
            to_df = contract_data[to_code].copy()
            
            if from_code in contract_data:
                from_df = contract_data[from_code]
                
                # Find prices at roll point
                to_at_roll = to_df[to_df.index <= roll_date].tail(1)
                from_at_roll = from_df[from_df.index <= roll_date].tail(1)
                
                if not to_at_roll.empty and not from_at_roll.empty:
                    to_price = float(to_at_roll["close"].iloc[0])
                    from_price = float(from_at_roll["close"].iloc[0])
                    
                    # Adjustment to apply to older data
                    adjustment = to_price - from_price
                    
                    self.roll_events.append(RollEvent(
                        roll_date=roll_date,
                        from_contract=from_code,
                        to_contract=to_code,
                        adjustment_factor=adjustment,
                        from_price=from_price,
                        to_price=to_price,
                        method="back_adjust"
                    ))
                    
                    cumulative_adjustment += adjustment
            
            # Get data after roll (or from start if first segment)
            if i == 0:
                segment = to_df[to_df.index <= end_date]
            else:
                prev_roll_date = roll_schedule[i - 1]["roll_date"]
                segment = to_df[(to_df.index > roll_date) & (to_df.index <= prev_roll_date)]
            
            result_segments.append(segment)
        
        # Handle the oldest segment (before first roll)
        if roll_schedule:
            first_roll = roll_schedule[-1]
            from_code = first_roll["from_contract"]
            
            if from_code in contract_data:
                from_df = contract_data[from_code].copy()
                oldest_segment = from_df[from_df.index < first_roll["roll_date"]]
                
                # Apply cumulative adjustment
                if not oldest_segment.empty:
                    for col in ["open", "high", "low", "close"]:
                        oldest_segment[col] = oldest_segment[col] + cumulative_adjustment
                    if "vwap" in oldest_segment.columns:
                        oldest_segment["vwap"] = oldest_segment["vwap"] + cumulative_adjustment
                    
                    result_segments.append(oldest_segment)
        
        if not result_segments:
            return pd.DataFrame()
        
        # Combine all segments
        continuous = pd.concat(result_segments, axis=0)
        continuous = continuous[~continuous.index.duplicated(keep="first")]
        continuous = continuous.sort_index()
        
        # Mark which contract each bar belongs to
        continuous["contract"] = ""
        for roll_info in roll_schedule:
            mask = continuous.index > roll_info["roll_date"]
            continuous.loc[mask, "contract"] = roll_info["to_contract"]
        
        # Fill remaining (oldest) with first contract
        if roll_schedule:
            oldest_mask = continuous["contract"] == ""
            continuous.loc[oldest_mask, "contract"] = roll_schedule[-1]["from_contract"]
        
        return continuous
    
    def _build_ratio_adjusted(
        self,
        contract_data: Dict[str, pd.DataFrame],
        roll_schedule: List[Dict],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Build ratio-adjusted continuous series.
        
        Ratio adjustment preserves percentage moves by multiplying
        by the price ratio at each roll point.
        """
        # Similar structure to back-adjust but with multiplicative factor
        result_segments = []
        cumulative_ratio = 1.0
        
        roll_schedule = sorted(roll_schedule, key=lambda x: x["roll_date"], reverse=True)
        
        for i, roll_info in enumerate(roll_schedule):
            roll_date = roll_info["roll_date"]
            to_code = roll_info["to_contract"]
            from_code = roll_info["from_contract"]
            
            if to_code not in contract_data:
                continue
            
            to_df = contract_data[to_code].copy()
            
            if from_code in contract_data:
                from_df = contract_data[from_code]
                
                to_at_roll = to_df[to_df.index <= roll_date].tail(1)
                from_at_roll = from_df[from_df.index <= roll_date].tail(1)
                
                if not to_at_roll.empty and not from_at_roll.empty:
                    to_price = float(to_at_roll["close"].iloc[0])
                    from_price = float(from_at_roll["close"].iloc[0])
                    
                    if from_price > 0:
                        ratio = to_price / from_price
                        
                        self.roll_events.append(RollEvent(
                            roll_date=roll_date,
                            from_contract=from_code,
                            to_contract=to_code,
                            adjustment_factor=ratio,
                            from_price=from_price,
                            to_price=to_price,
                            method="ratio_adjust"
                        ))
                        
                        cumulative_ratio *= ratio
            
            if i == 0:
                segment = to_df[to_df.index <= end_date]
            else:
                prev_roll_date = roll_schedule[i - 1]["roll_date"]
                segment = to_df[(to_df.index > roll_date) & (to_df.index <= prev_roll_date)]
            
            result_segments.append(segment)
        
        # Handle oldest segment
        if roll_schedule:
            first_roll = roll_schedule[-1]
            from_code = first_roll["from_contract"]
            
            if from_code in contract_data:
                from_df = contract_data[from_code].copy()
                oldest_segment = from_df[from_df.index < first_roll["roll_date"]]
                
                if not oldest_segment.empty:
                    for col in ["open", "high", "low", "close"]:
                        oldest_segment[col] = oldest_segment[col] * cumulative_ratio
                    if "vwap" in oldest_segment.columns:
                        oldest_segment["vwap"] = oldest_segment["vwap"] * cumulative_ratio
                    
                    result_segments.append(oldest_segment)
        
        if not result_segments:
            return pd.DataFrame()
        
        continuous = pd.concat(result_segments, axis=0)
        continuous = continuous[~continuous.index.duplicated(keep="first")]
        continuous = continuous.sort_index()
        
        return continuous
    
    def _build_unadjusted(
        self,
        contract_data: Dict[str, pd.DataFrame],
        roll_schedule: List[Dict],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Build unadjusted series (gaps at roll points)."""
        segments = []
        
        roll_schedule = sorted(roll_schedule, key=lambda x: x["roll_date"])
        
        for i, roll_info in enumerate(roll_schedule):
            roll_date = roll_info["roll_date"]
            from_code = roll_info["from_contract"]
            to_code = roll_info["to_contract"]
            
            if from_code in contract_data:
                from_df = contract_data[from_code].copy()
                
                if i == 0:
                    segment = from_df[from_df.index < roll_date]
                else:
                    prev_roll = roll_schedule[i - 1]["roll_date"]
                    segment = from_df[(from_df.index >= prev_roll) & (from_df.index < roll_date)]
                
                if not segment.empty:
                    segment["contract"] = from_code
                    segments.append(segment)
            
            # Record roll event
            if from_code in contract_data and to_code in contract_data:
                from_df = contract_data[from_code]
                to_df = contract_data[to_code]
                
                from_at_roll = from_df[from_df.index <= roll_date].tail(1)
                to_at_roll = to_df[to_df.index <= roll_date].tail(1)
                
                if not from_at_roll.empty and not to_at_roll.empty:
                    self.roll_events.append(RollEvent(
                        roll_date=roll_date,
                        from_contract=from_code,
                        to_contract=to_code,
                        adjustment_factor=0.0,
                        from_price=float(from_at_roll["close"].iloc[0]),
                        to_price=float(to_at_roll["close"].iloc[0]),
                        method="unadjusted"
                    ))
        
        # Handle data after last roll
        if roll_schedule:
            last_roll = roll_schedule[-1]
            to_code = last_roll["to_contract"]
            
            if to_code in contract_data:
                to_df = contract_data[to_code].copy()
                final_segment = to_df[to_df.index >= last_roll["roll_date"]]
                
                if not final_segment.empty:
                    final_segment["contract"] = to_code
                    segments.append(final_segment)
        
        if not segments:
            return pd.DataFrame()
        
        continuous = pd.concat(segments, axis=0)
        continuous = continuous[~continuous.index.duplicated(keep="first")]
        return continuous.sort_index()
    
    def get_roll_report(self) -> pd.DataFrame:
        """Get a report of all roll events."""
        if not self.roll_events:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                "roll_date": e.roll_date,
                "from_contract": e.from_contract,
                "to_contract": e.to_contract,
                "adjustment": e.adjustment_factor,
                "from_price": e.from_price,
                "to_price": e.to_price,
                "method": e.method,
                "gap_points": abs(e.to_price - e.from_price),
                "gap_pct": abs(e.to_price - e.from_price) / e.from_price * 100 if e.from_price else 0,
            }
            for e in self.roll_events
        ])


def generate_sample_roll_calendar(
    start_year: int = 2024,
    end_year: int = 2026,
    symbol: str = "MES"
) -> pd.DataFrame:
    """Generate a sample roll calendar for documentation."""
    calendar = FuturesRollCalendar(symbol)
    config = RollConfig()
    
    start = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    end = datetime(end_year, 12, 31, tzinfo=timezone.utc)
    
    contracts = calendar.generate_contracts(start, end)
    roll_schedule = calendar.generate_roll_dates(contracts, config)
    
    return pd.DataFrame(roll_schedule)


if __name__ == "__main__":
    # Test roll calendar generation
    calendar = generate_sample_roll_calendar()
    print("Roll Calendar:")
    print(calendar.to_string())
