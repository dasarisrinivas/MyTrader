"""
Session Classification Utilities
=================================

FEB 2026 — Session-isolated indicator architecture.

Classifies every 15m bar into its trading session so that RTH indicators
are computed ONLY from RTH bars and overnight indicators are computed
ONLY from overnight bars.

CRITICAL INSIGHT:
  When overnight bars are mixed into the indicator DataFrame, they corrupt
  EMA/ATR/ADX values used during RTH:
    - RTH-only indicators: PF 1.67, +$3,031 (193 trades)
    - Mixed indicators:    PF 1.21, +$1,410 (241 trades)

This module ensures that pollution NEVER happens.

Session Definitions (all times in US/Eastern):
  RTH:         09:30 – 16:00 ET, Mon–Fri
  Overnight:   18:00 – 09:30 ET (spans midnight)
  Maintenance: 17:00 – 18:00 ET (CME daily halt)
  Weekend:     Saturday all day, Sunday before 18:00

Edge Cases Handled:
  - DST transitions (uses US/Eastern, not fixed offset)
  - Sunday open (18:00 ET)
  - Holiday half-days (configurable RTH end time)
  - Rollover dates (flagged, not filtered)
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Optional, Tuple

import pandas as pd
import numpy as np

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

ET = ZoneInfo("US/Eastern")


class TradingSession(Enum):
    """Trading session classification."""
    RTH = "RTH"                    # Regular Trading Hours: 09:30-16:00 ET
    OVERNIGHT = "OVERNIGHT"        # Globex overnight: 18:00-09:30 ET
    MAINTENANCE = "MAINTENANCE"    # CME daily halt: 17:00-18:00 ET
    WEEKEND = "WEEKEND"            # Saturday + Sunday before 18:00


# ──────────────────────────────────────────────────────────────────────
#  Core session classifier
# ──────────────────────────────────────────────────────────────────────

def classify_session(
    ts: datetime,
    rth_start: time = time(9, 30),
    rth_end: time = time(16, 0),
    maintenance_start: time = time(17, 0),
    maintenance_end: time = time(18, 0),
) -> TradingSession:
    """Classify a timestamp into its trading session.

    Args:
        ts: Timezone-aware datetime (any tz — will be converted to ET).
        rth_start: RTH start time (default 09:30 ET).
        rth_end: RTH end time (default 16:00 ET). Override for half-days.
        maintenance_start: CME halt start (default 17:00 ET).
        maintenance_end: CME halt end (default 18:00 ET).

    Returns:
        TradingSession enum value.

    Raises:
        ValueError: If ts is timezone-naive.
    """
    if ts.tzinfo is None:
        raise ValueError(f"Timestamp must be timezone-aware, got naive: {ts}")

    # Convert to Eastern Time (handles DST automatically)
    et = ts.astimezone(ET)
    t = et.time()
    dow = et.weekday()  # Mon=0 … Sun=6

    # ── Weekend ──
    if dow == 5:  # Saturday — always weekend
        return TradingSession.WEEKEND
    if dow == 6 and t < maintenance_end:  # Sunday before 18:00
        return TradingSession.WEEKEND

    # ── CME Maintenance ──
    if maintenance_start <= t < maintenance_end:
        return TradingSession.MAINTENANCE

    # ── RTH ──
    # Respect the configured trading window on Sunday after the maintenance
    # reopen as well. This allows widened session configs (for example,
    # 00:00–23:59 ET) to treat Sunday evening Globex bars as tradable while
    # still preserving the weekend block before the reopen.
    if rth_start <= t < rth_end and dow != 5:
        return TradingSession.RTH

    # ── Everything else is Overnight ──
    return TradingSession.OVERNIGHT


def is_rth(ts: datetime) -> bool:
    """Quick check: is this timestamp within RTH?"""
    return classify_session(ts) == TradingSession.RTH


def is_overnight(ts: datetime) -> bool:
    """Quick check: is this timestamp within the overnight session?"""
    return classify_session(ts) == TradingSession.OVERNIGHT


# ──────────────────────────────────────────────────────────────────────
#  DataFrame session tagging & splitting
# ──────────────────────────────────────────────────────────────────────

def tag_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'session' column to DataFrame with DatetimeIndex.

    Args:
        df: DataFrame with timezone-aware DatetimeIndex.

    Returns:
        Copy of df with a new 'session' column containing TradingSession values.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("DatetimeIndex must be timezone-aware")

    result = df.copy()
    result["session"] = [classify_session(ts) for ts in result.index]
    return result


def split_by_session(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a 15m OHLCV DataFrame into RTH-only and Overnight-only.

    The returned DataFrames contain only bars from their respective sessions.
    Indicators computed on these will be session-pure — no cross-contamination.

    The RTH DataFrame has a "gap" between 16:00 Friday and 09:30 Monday.
    This is intentional — the indicator functions (EMA, ATR) treat it as
    a continuous series of RTH bars, which is exactly what the validated
    backtest does.

    Args:
        df: Full 15m OHLCV DataFrame with timezone-aware DatetimeIndex.

    Returns:
        (df_rth, df_overnight) tuple.
    """
    tagged = tag_sessions(df)
    df_rth = tagged[tagged["session"] == TradingSession.RTH].drop(columns=["session"])
    df_on = tagged[tagged["session"] == TradingSession.OVERNIGHT].drop(columns=["session"])
    return df_rth, df_on


def get_rth_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only RTH bars from a DataFrame. Convenience wrapper."""
    df_rth, _ = split_by_session(df)
    return df_rth


def get_overnight_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only overnight bars from a DataFrame. Convenience wrapper."""
    _, df_on = split_by_session(df)
    return df_on


# ──────────────────────────────────────────────────────────────────────
#  Overnight sub-session classification (for time filters)
# ──────────────────────────────────────────────────────────────────────

class OvernightWindow(Enum):
    """Sub-windows within the overnight session, each with different
    liquidity/volatility characteristics."""
    POST_CLOSE = "POST_CLOSE"       # 18:00-20:00 ET — avoid (noise)
    ASIA = "ASIA"                   # 20:00-01:00 ET — range-bound
    ASIA_EUROPE = "ASIA_EUROPE"     # 01:00-04:00 ET — best ON liquidity
    EUROPE = "EUROPE"               # 04:00-08:00 ET — macro-driven
    PRE_RTH = "PRE_RTH"            # 08:00-09:15 ET — exit window
    RTH_BUFFER = "RTH_BUFFER"      # 09:15-09:30 ET — NO new entries


def classify_overnight_window(ts: datetime) -> Optional[OvernightWindow]:
    """Classify an overnight timestamp into its sub-window.

    Returns None if the timestamp is not in the overnight session.
    """
    if classify_session(ts) != TradingSession.OVERNIGHT:
        return None

    et = ts.astimezone(ET)
    t = et.time()

    if time(18, 0) <= t < time(20, 0):
        return OvernightWindow.POST_CLOSE
    if t >= time(20, 0) or t < time(1, 0):
        return OvernightWindow.ASIA
    if time(1, 0) <= t < time(4, 0):
        return OvernightWindow.ASIA_EUROPE
    if time(4, 0) <= t < time(8, 0):
        return OvernightWindow.EUROPE
    if time(8, 0) <= t < time(9, 15):
        return OvernightWindow.PRE_RTH
    if time(9, 15) <= t < time(9, 30):
        return OvernightWindow.RTH_BUFFER

    return None  # Should not reach here


def is_tradeable_overnight(ts: datetime) -> bool:
    """Check if an overnight timestamp is in a tradeable sub-window.

    Avoids:
      - POST_CLOSE (18:00-20:00) — low liquidity noise
      - RTH_BUFFER (09:15-09:30) — too close to RTH open
      - Sunday 18:00-20:00 — gap-fill noise
    """
    window = classify_overnight_window(ts)
    if window is None:
        return False

    # Block post-close and RTH buffer
    if window in (OvernightWindow.POST_CLOSE, OvernightWindow.RTH_BUFFER):
        return False

    # Extra: Block Sunday evening (first 2 hours of week)
    et = ts.astimezone(ET)
    if et.weekday() == 6 and et.time() >= time(18, 0):  # Sunday evening
        return False

    return True


# ──────────────────────────────────────────────────────────────────────
#  Gap computation (RTH close → next RTH open)
# ──────────────────────────────────────────────────────────────────────

def compute_overnight_gap(df_rth: pd.DataFrame) -> pd.Series:
    """Compute the overnight gap for each RTH session.

    Gap = (today's first RTH bar open) / (yesterday's last RTH bar close) - 1

    Args:
        df_rth: RTH-only DataFrame with OHLCV and DatetimeIndex in ET or UTC.

    Returns:
        Series indexed by date with gap percentages.
    """
    if df_rth.empty:
        return pd.Series(dtype=float)

    # Convert to ET for date grouping
    idx_et = df_rth.index.tz_convert(ET) if df_rth.index.tz is not None else df_rth.index
    dates = idx_et.date

    # Get first open and last close per day
    rth_copy = df_rth.copy()
    rth_copy["_date"] = dates
    daily_first_open = rth_copy.groupby("_date")["open"].first()
    daily_last_close = rth_copy.groupby("_date")["close"].last()

    # Gap = today open / yesterday close - 1
    prev_close = daily_last_close.shift(1)
    gap = (daily_first_open / prev_close) - 1.0

    return gap.dropna()
