"""Centralized timezone utilities for CST (Central Standard Time).

All timestamps in the system should use these utilities for consistency.
CST = UTC-6 (Central Standard Time, no DST adjustment for simplicity)
CDT = UTC-5 (Central Daylight Time, active Mar-Nov)

We use America/Chicago which automatically handles DST transitions.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

# Central Time Zone (handles DST automatically)
CST = ZoneInfo("America/Chicago")

# For reference: fixed offsets (use CST above for auto DST)
CST_OFFSET = timezone(timedelta(hours=-6))  # Standard time
CDT_OFFSET = timezone(timedelta(hours=-5))  # Daylight time


def now_cst() -> datetime:
    """Get current time in CST/CDT (Central Time).
    
    Returns:
        datetime: Current time with America/Chicago timezone
    """
    return datetime.now(CST)


def now_cst_str(fmt: str = "%Y-%m-%d %H:%M:%S CST") -> str:
    """Get current time as formatted string in CST.
    
    Args:
        fmt: strftime format string (default includes CST label)
        
    Returns:
        Formatted datetime string
    """
    return datetime.now(CST).strftime(fmt)


def utc_to_cst(dt: datetime) -> datetime:
    """Convert UTC datetime to CST/CDT.
    
    Args:
        dt: datetime in UTC (aware or naive)
        
    Returns:
        datetime in Central Time
    """
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(CST)


def cst_to_utc(dt: datetime) -> datetime:
    """Convert CST/CDT datetime to UTC.
    
    Args:
        dt: datetime in Central Time (aware or naive)
        
    Returns:
        datetime in UTC
    """
    if dt.tzinfo is None:
        # Assume naive datetime is CST
        dt = dt.replace(tzinfo=CST)
    return dt.astimezone(timezone.utc)


def format_cst(dt: Optional[datetime], fmt: str = "%Y-%m-%d %H:%M:%S CST") -> str:
    """Format a datetime in CST.
    
    Args:
        dt: datetime to format (UTC or CST)
        fmt: strftime format
        
    Returns:
        Formatted string in CST
    """
    if dt is None:
        return "N/A"
    if dt.tzinfo is None or dt.tzinfo == timezone.utc:
        dt = utc_to_cst(dt)
    return dt.strftime(fmt)


def today_cst() -> str:
    """Get today's date in CST as YYYY-MM-DD string."""
    return datetime.now(CST).strftime("%Y-%m-%d")


def timestamp_cst() -> str:
    """Get current timestamp for filenames/IDs (no spaces/colons)."""
    return datetime.now(CST).strftime("%Y%m%d_%H%M%S")


def iso_cst() -> str:
    """Get ISO format timestamp in CST."""
    return datetime.now(CST).isoformat()


def trading_hours_cst() -> tuple[int, int]:
    """Get current hour in CST for trading hour checks.
    
    Returns:
        Tuple of (hour, minute) in CST
    """
    now = datetime.now(CST)
    return now.hour, now.minute


def is_market_hours_cst() -> bool:
    """Check if current time is during ES futures market hours.
    
    ES futures trade nearly 24 hours:
    - Sunday 5:00 PM CST to Friday 4:00 PM CST
    - Daily break: 4:00 PM - 5:00 PM CST
    
    Returns:
        True if market is open
    """
    now = datetime.now(CST)
    hour = now.hour
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    
    # Weekend check (Saturday all day, Sunday before 5 PM)
    if weekday == 5:  # Saturday
        return False
    if weekday == 6 and hour < 17:  # Sunday before 5 PM
        return False
    
    # Daily maintenance break (4 PM - 5 PM CST)
    if hour == 16:
        return False
    
    return True


# Loguru custom time formatter
def cst_time_formatter(record):
    """Custom time formatter for loguru to use CST."""
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")
