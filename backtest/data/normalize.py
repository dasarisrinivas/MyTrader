"""
Data Normalization Module
=========================

Normalizes downloaded data to a unified schema:
- Consistent column names and dtypes
- UTC timezone normalization
- Missing bar handling (forward-fill or gap marking)
- Multi-timeframe alignment (1m and 5m)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
from loguru import logger


class MissingBarPolicy(Enum):
    """Policy for handling missing bars."""
    FORWARD_FILL = "forward_fill"      # Fill missing with previous close, zero volume
    DROP_GAPS = "drop_gaps"            # Keep gaps, record them in metadata
    INTERPOLATE = "interpolate"        # Linear interpolation (not recommended)


class SessionType(Enum):
    """Trading session type."""
    RTH = "rth"          # Regular Trading Hours (9:30 AM - 4:00 PM ET for equities)
    ETH = "eth"          # Extended Trading Hours (includes pre/post)
    FULL = "full"        # 24-hour futures session (Sun 6PM - Fri 5PM ET)
    CUSTOM = "custom"    # Custom session hours


@dataclass
class NormalizationConfig:
    """Configuration for data normalization."""
    
    # Column schema (input -> standard)
    column_mapping: Dict[str, str] = field(default_factory=lambda: {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "vwap": "vwap",
        "average": "vwap",
        "trade_count": "trades",
        "barCount": "trades",
    })
    
    # Standard columns to output
    output_columns: List[str] = field(default_factory=lambda: [
        "open", "high", "low", "close", "volume", "vwap", "trades"
    ])
    
    # Timezone handling
    input_timezone: str = "UTC"
    output_timezone: str = "UTC"
    
    # Missing bar policy
    missing_bar_policy: MissingBarPolicy = MissingBarPolicy.FORWARD_FILL
    max_gap_minutes: int = 60  # Maximum gap to fill; larger gaps are recorded
    
    # Session configuration
    session_type: SessionType = SessionType.FULL
    
    # CME Futures session times (in US/Eastern)
    # Sunday 6:00 PM - Friday 5:00 PM with daily maintenance 5:00-6:00 PM
    futures_session_start: time = time(18, 0)   # 6:00 PM ET
    futures_session_end: time = time(17, 0)     # 5:00 PM ET next day
    maintenance_start: time = time(17, 0)       # 5:00 PM ET
    maintenance_end: time = time(18, 0)         # 6:00 PM ET
    
    # RTH for ES/MES (CME): 9:30 AM - 4:00 PM ET
    rth_start: time = time(9, 30)
    rth_end: time = time(16, 0)
    
    # Validation settings
    min_volume_threshold: int = 0  # Bars below this are flagged
    max_price_change_pct: float = 10.0  # Flag suspicious price moves


@dataclass
class GapInfo:
    """Information about a gap in data."""
    start: datetime
    end: datetime
    duration_minutes: int
    bar_count: int
    filled: bool
    reason: str


class DataNormalizer:
    """
    Normalizes raw price data to a standardized format.
    
    Handles:
    - Column name standardization
    - Timezone conversion to UTC
    - Missing bar detection and handling
    - Data validation and anomaly flagging
    - Multi-timeframe alignment
    """
    
    def __init__(self, config: Optional[NormalizationConfig] = None):
        self.config = config or NormalizationConfig()
        self.gaps: List[GapInfo] = []
        self.anomalies: List[Dict] = []
    
    def normalize(
        self,
        df: pd.DataFrame,
        bar_size_minutes: int = 1,
        symbol: str = "UNKNOWN"
    ) -> pd.DataFrame:
        """
        Normalize a DataFrame to standard schema.
        
        Args:
            df: Input DataFrame with price data
            bar_size_minutes: Expected bar size in minutes
            symbol: Symbol name for logging
            
        Returns:
            Normalized DataFrame
        """
        if df.empty:
            logger.warning(f"Empty DataFrame provided for {symbol}")
            return df
        
        result = df.copy()
        
        # Step 1: Standardize column names
        result = self._standardize_columns(result)
        
        # Step 2: Ensure datetime index in UTC
        result = self._normalize_timezone(result)
        
        # Step 3: Sort by timestamp
        result = result.sort_index()
        
        # Step 4: Remove duplicates
        result = result[~result.index.duplicated(keep="first")]
        
        # Step 5: Handle missing bars
        result = self._handle_missing_bars(result, bar_size_minutes, symbol)
        
        # Step 6: Validate data
        self._validate_data(result, symbol)
        
        # Step 7: Ensure correct dtypes
        result = self._enforce_dtypes(result)
        
        return result
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to standard names."""
        # Lowercase all columns first
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Apply mapping
        rename_map = {}
        for old, new in self.config.column_mapping.items():
            if old.lower() in df.columns:
                rename_map[old.lower()] = new
        
        df = df.rename(columns=rename_map)
        
        # Ensure required columns exist
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Add optional columns if missing
        if "vwap" not in df.columns:
            df["vwap"] = np.nan
        if "trades" not in df.columns:
            df["trades"] = 0
        
        # Keep only output columns
        output_cols = [c for c in self.config.output_columns if c in df.columns]
        return df[output_cols]
    
    def _normalize_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure index is datetime with UTC timezone."""
        # Handle index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df.index = pd.to_datetime(df["timestamp"])
                df = df.drop(columns=["timestamp"], errors="ignore")
            elif "date" in df.columns:
                df.index = pd.to_datetime(df["date"])
                df = df.drop(columns=["date"], errors="ignore")
            elif "datetime" in df.columns:
                df.index = pd.to_datetime(df["datetime"])
                df = df.drop(columns=["datetime"], errors="ignore")
            else:
                df.index = pd.to_datetime(df.index)
        
        df.index.name = "timestamp"
        
        # Convert to UTC
        if df.index.tzinfo is None:
            # Assume input timezone
            try:
                df.index = df.index.tz_localize(self.config.input_timezone)
            except Exception:
                # Already localized or ambiguous
                df.index = df.index.tz_localize(self.config.input_timezone, ambiguous="infer", nonexistent="shift_forward")
        
        df.index = df.index.tz_convert("UTC")
        
        return df
    
    def _handle_missing_bars(
        self,
        df: pd.DataFrame,
        bar_size_minutes: int,
        symbol: str
    ) -> pd.DataFrame:
        """Detect and handle missing bars based on policy."""
        if df.empty:
            return df
        
        self.gaps = []
        
        # Generate expected timestamps
        freq = f"{bar_size_minutes}T"  # e.g., "1T" for 1 minute
        expected_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq,
            tz="UTC"
        )
        
        # Find missing timestamps
        missing = expected_index.difference(df.index)
        
        if len(missing) == 0:
            return df
        
        # Filter missing based on session (exclude maintenance windows)
        valid_missing = self._filter_valid_gaps(missing)
        
        if len(valid_missing) == 0:
            return df
        
        logger.info(f"{symbol}: Found {len(valid_missing)} missing bars")
        
        # Group into continuous gaps
        gap_groups = self._group_gaps(valid_missing, bar_size_minutes)
        
        # Handle each gap based on policy
        if self.config.missing_bar_policy == MissingBarPolicy.FORWARD_FILL:
            df = self._forward_fill_gaps(df, gap_groups, bar_size_minutes)
        elif self.config.missing_bar_policy == MissingBarPolicy.DROP_GAPS:
            # Just record the gaps, don't modify data
            pass
        elif self.config.missing_bar_policy == MissingBarPolicy.INTERPOLATE:
            df = self._interpolate_gaps(df, gap_groups)
        
        return df
    
    def _filter_valid_gaps(self, missing: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Filter out gaps that occur during maintenance windows."""
        if self.config.session_type == SessionType.FULL:
            # For 24h futures, only exclude maintenance window
            valid = []
            for ts in missing:
                # Convert to Eastern for maintenance check
                ts_et = ts.tz_convert("America/New_York")
                t = ts_et.time()
                
                # Exclude 5:00 PM - 6:00 PM ET maintenance
                if not (self.config.maintenance_start <= t < self.config.maintenance_end):
                    # Also exclude weekends (futures don't trade Sat-Sun)
                    if ts_et.weekday() < 5:  # Mon-Fri
                        valid.append(ts)
                    elif ts_et.weekday() == 6 and t >= self.config.futures_session_start:
                        # Sunday after 6 PM ET
                        valid.append(ts)
            
            return pd.DatetimeIndex(valid, tz="UTC")
        
        elif self.config.session_type == SessionType.RTH:
            # Only include RTH hours
            valid = []
            for ts in missing:
                ts_et = ts.tz_convert("America/New_York")
                t = ts_et.time()
                
                if self.config.rth_start <= t < self.config.rth_end:
                    if ts_et.weekday() < 5:  # Mon-Fri
                        valid.append(ts)
            
            return pd.DatetimeIndex(valid, tz="UTC")
        
        return missing
    
    def _group_gaps(
        self,
        missing: pd.DatetimeIndex,
        bar_size_minutes: int
    ) -> List[Tuple[datetime, datetime]]:
        """Group consecutive missing timestamps into gap ranges."""
        if len(missing) == 0:
            return []
        
        groups = []
        expected_delta = timedelta(minutes=bar_size_minutes)
        
        gap_start = missing[0]
        prev = missing[0]
        
        for ts in missing[1:]:
            if ts - prev > expected_delta * 1.5:  # Allow some tolerance
                # End of gap
                groups.append((gap_start, prev))
                gap_start = ts
            prev = ts
        
        # Don't forget the last gap
        groups.append((gap_start, prev))
        
        # Record gap info
        for start, end in groups:
            duration = int((end - start).total_seconds() / 60) + bar_size_minutes
            bar_count = duration // bar_size_minutes
            
            self.gaps.append(GapInfo(
                start=start.to_pydatetime(),
                end=end.to_pydatetime(),
                duration_minutes=duration,
                bar_count=bar_count,
                filled=self.config.missing_bar_policy == MissingBarPolicy.FORWARD_FILL,
                reason="MISSING_DATA"
            ))
        
        return groups
    
    def _forward_fill_gaps(
        self,
        df: pd.DataFrame,
        gaps: List[Tuple[datetime, datetime]],
        bar_size_minutes: int
    ) -> pd.DataFrame:
        """Forward-fill gaps with previous close and zero volume."""
        freq = f"{bar_size_minutes}T"
        
        for gap_start, gap_end in gaps:
            # Only fill gaps up to max_gap_minutes
            gap_duration = (gap_end - gap_start).total_seconds() / 60 + bar_size_minutes
            
            if gap_duration > self.config.max_gap_minutes:
                logger.warning(
                    f"Gap {gap_start} to {gap_end} ({gap_duration:.0f}m) exceeds max "
                    f"({self.config.max_gap_minutes}m). Not filling."
                )
                continue
            
            # Find the bar just before the gap
            before = df[df.index < gap_start]
            if before.empty:
                continue
            
            last_bar = before.iloc[-1]
            fill_price = last_bar["close"]
            
            # Generate fill timestamps
            fill_index = pd.date_range(
                start=gap_start,
                end=gap_end,
                freq=freq,
                tz="UTC"
            )
            
            # Create fill data
            fill_data = pd.DataFrame(
                index=fill_index,
                data={
                    "open": fill_price,
                    "high": fill_price,
                    "low": fill_price,
                    "close": fill_price,
                    "volume": 0,
                    "vwap": fill_price,
                    "trades": 0,
                }
            )
            fill_data.index.name = "timestamp"
            
            # Combine
            df = pd.concat([df, fill_data])
        
        # Sort and deduplicate
        df = df[~df.index.duplicated(keep="first")]
        return df.sort_index()
    
    def _interpolate_gaps(
        self,
        df: pd.DataFrame,
        gaps: List[Tuple[datetime, datetime]]
    ) -> pd.DataFrame:
        """Linear interpolation for gaps (use with caution)."""
        logger.warning("Interpolation for gaps may introduce lookahead bias!")
        return df.interpolate(method="linear")
    
    def _validate_data(self, df: pd.DataFrame, symbol: str) -> None:
        """Validate data and flag anomalies."""
        self.anomalies = []
        
        # Check for zero/negative prices
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            invalid = df[df[col] <= 0]
            if not invalid.empty:
                for idx in invalid.index:
                    self.anomalies.append({
                        "timestamp": idx,
                        "type": "INVALID_PRICE",
                        "column": col,
                        "value": invalid.loc[idx, col],
                        "symbol": symbol
                    })
        
        # Check OHLC consistency
        invalid_ohlc = df[(df["high"] < df["low"]) | 
                         (df["high"] < df["open"]) | 
                         (df["high"] < df["close"]) |
                         (df["low"] > df["open"]) |
                         (df["low"] > df["close"])]
        
        for idx in invalid_ohlc.index:
            self.anomalies.append({
                "timestamp": idx,
                "type": "INVALID_OHLC",
                "ohlc": {
                    "open": invalid_ohlc.loc[idx, "open"],
                    "high": invalid_ohlc.loc[idx, "high"],
                    "low": invalid_ohlc.loc[idx, "low"],
                    "close": invalid_ohlc.loc[idx, "close"],
                },
                "symbol": symbol
            })
        
        # Check for suspicious price moves
        df["pct_change"] = df["close"].pct_change().abs() * 100
        suspicious = df[df["pct_change"] > self.config.max_price_change_pct]
        
        for idx in suspicious.index:
            self.anomalies.append({
                "timestamp": idx,
                "type": "LARGE_PRICE_MOVE",
                "pct_change": suspicious.loc[idx, "pct_change"],
                "symbol": symbol
            })
        
        if self.anomalies:
            logger.warning(f"{symbol}: Found {len(self.anomalies)} data anomalies")
    
    def _enforce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct data types."""
        dtype_map = {
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": int,
            "trades": int,
        }
        
        for col, dtype in dtype_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        
        # VWAP can have NaN
        if "vwap" in df.columns:
            df["vwap"] = df["vwap"].astype(float)
        
        return df
    
    def resample_to_higher_timeframe(
        self,
        df: pd.DataFrame,
        target_minutes: int,
        use_completed_only: bool = True
    ) -> pd.DataFrame:
        """
        Resample to a higher timeframe (e.g., 1m -> 5m).
        
        CRITICAL: When use_completed_only=True, each row only uses data
        that was available at or before its timestamp (no lookahead).
        
        Args:
            df: Input DataFrame at lower timeframe
            target_minutes: Target timeframe in minutes
            use_completed_only: If True, shift resampled data to avoid lookahead
            
        Returns:
            Resampled DataFrame
        """
        if df.empty:
            return df
        
        freq = f"{target_minutes}T"
        
        # Standard OHLCV resampling
        resampled = df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        
        # Handle VWAP if present
        if "vwap" in df.columns:
            # Volume-weighted average of VWAP
            vwap_sum = (df["vwap"] * df["volume"]).resample(freq).sum()
            vol_sum = df["volume"].resample(freq).sum()
            resampled["vwap"] = vwap_sum / vol_sum.replace(0, np.nan)
        
        if "trades" in df.columns:
            resampled["trades"] = df["trades"].resample(freq).sum()
        
        # Drop NaN rows (incomplete bars at start)
        resampled = resampled.dropna(subset=["close"])
        
        if use_completed_only:
            # CRITICAL FOR NO LOOKAHEAD:
            # The resampled bar at time T aggregates data from T to T+target_minutes.
            # At time T, this bar is NOT yet complete.
            # Shift the index forward so the bar is timestamped at completion time.
            resampled.index = resampled.index + timedelta(minutes=target_minutes)
            logger.debug(f"Shifted resampled data by {target_minutes} minutes for no-lookahead")
        
        return resampled
    
    def align_multi_timeframe(
        self,
        df_1m: pd.DataFrame,
        df_5m: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align 1m and 5m data, adding 5m features to 1m bars.
        
        For each 1m bar at time T, uses the LAST COMPLETED 5m bar
        (the one whose close timestamp <= T).
        
        This ensures no lookahead bias.
        """
        if df_1m.empty or df_5m.empty:
            return df_1m
        
        # Add prefix to 5m columns
        df_5m_prefixed = df_5m.add_prefix("5m_")
        
        # For each 1m bar, find the last completed 5m bar
        # Use merge_asof for efficient time-based alignment
        df_1m_reset = df_1m.reset_index()
        df_5m_reset = df_5m_prefixed.reset_index()
        
        aligned = pd.merge_asof(
            df_1m_reset.sort_values("timestamp"),
            df_5m_reset.sort_values("timestamp"),
            on="timestamp",
            direction="backward"  # Use last 5m bar at or before this time
        )
        
        aligned.set_index("timestamp", inplace=True)
        
        return aligned
    
    def get_gap_report(self) -> pd.DataFrame:
        """Get a report of all gaps found during normalization."""
        if not self.gaps:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                "start": g.start,
                "end": g.end,
                "duration_minutes": g.duration_minutes,
                "bar_count": g.bar_count,
                "filled": g.filled,
                "reason": g.reason
            }
            for g in self.gaps
        ])
    
    def get_anomaly_report(self) -> pd.DataFrame:
        """Get a report of all anomalies found during validation."""
        if not self.anomalies:
            return pd.DataFrame()
        
        return pd.DataFrame(self.anomalies)
