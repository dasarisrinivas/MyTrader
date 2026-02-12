"""Session time management for trading entry modules.

Provides ``SessionWindow`` enum and ``SessionTimeManager`` for determining
which session window the market is in (pre-market, morning prime, midday, etc.)
and whether specific entry types are allowed based on the current time.

All times are in CST (Central Standard Time).
"""
from datetime import datetime, time
from enum import Enum
from typing import Optional, Tuple


class SessionWindow(Enum):
    """Trading session windows for different strategies."""
    PRE_MARKET = "PRE_MARKET"           # Before 9:30 CST
    MORNING_OPEN = "MORNING_OPEN"       # 9:30-10:00 CST (volatile, careful)
    MORNING_PRIME = "MORNING_PRIME"     # 9:30-10:45 CST (best continuation for BUY)
    MIDDAY = "MIDDAY"                   # 10:45-14:00 CST (chop, range/reversion only)
    AFTERNOON = "AFTERNOON"             # 14:00-15:00 CST (possible trend)
    CLOSE = "CLOSE"                     # 15:00-16:00 CST (position squaring)
    OVERNIGHT = "OVERNIGHT"             # After 16:00 CST


class SessionTimeManager:
    """Centralized session time logic for all entry modules.

    Session Windows (CST):
    - PRE_MARKET: Before 09:30
    - MORNING_OPEN: 09:30-10:00 (volatile open, careful)
    - MORNING_PRIME: 09:30-10:45 (best for BUY_CONTINUATION)
    - MIDDAY: 10:45-14:00 (chop zone, RANGE/MEAN_REVERSION only)
    - AFTERNOON: 14:00-15:00 (possible trend resumption)
    - CLOSE: 15:00-16:00 (position squaring)
    - OVERNIGHT: After 16:00

    Key cutoffs:
    - buy_continuation_cutoff: 10:45 CST (after this, disable BUY_CONTINUATION)
    - reversal_block_until: 10:15 CST (no reversal trades before this)
    """

    # Session boundaries (CST)
    PRE_MARKET_END = time(9, 30)
    MORNING_OPEN_END = time(10, 0)

    # MORNING_PRIME: Best window for BUY_CONTINUATION
    MORNING_PRIME_START = time(9, 30)
    MORNING_PRIME_END = time(10, 45)
    BUY_CONTINUATION_CUTOFF = time(10, 45)

    # MIDDAY: Chop zone, only RANGE/MEAN_REVERSION allowed
    MIDDAY_START = time(10, 45)
    MIDDAY_END = time(14, 0)

    # Afternoon/Close
    AFTERNOON_START = time(14, 0)
    AFTERNOON_END = time(15, 0)
    CLOSE_START = time(15, 0)
    CLOSE_END = time(16, 0)

    # Special cutoffs
    REVERSAL_BLOCK_UNTIL = time(10, 15)

    @classmethod
    def get_session_window(cls, timestamp: Optional[datetime]) -> SessionWindow:
        """Determine current session window from *timestamp*.

        Args:
            timestamp: Current datetime (should be in CST).

        Returns:
            The ``SessionWindow`` enum value for the current time.
        """
        if timestamp is None:
            return SessionWindow.MORNING_PRIME

        try:
            t = timestamp.time() if hasattr(timestamp, "time") else time(10, 0)
        except Exception:
            return SessionWindow.MORNING_PRIME

        if t < cls.PRE_MARKET_END:
            return SessionWindow.PRE_MARKET
        elif t < cls.MORNING_PRIME_END:
            return SessionWindow.MORNING_PRIME
        elif t < cls.MIDDAY_END:
            return SessionWindow.MIDDAY
        elif t < cls.AFTERNOON_END:
            return SessionWindow.AFTERNOON
        elif t < cls.CLOSE_END:
            return SessionWindow.CLOSE
        else:
            return SessionWindow.OVERNIGHT

    @classmethod
    def is_buy_continuation_allowed(cls, timestamp: Optional[datetime]) -> Tuple[bool, str]:
        """Return ``(True, reason)`` if BUY_CONTINUATION is allowed at *timestamp*."""
        if timestamp is None:
            return True, "no_timestamp"

        try:
            t = timestamp.time() if hasattr(timestamp, "time") else time(10, 0)
        except Exception:
            return True, "time_parse_error"

        if t < cls.MORNING_PRIME_START:
            return False, f"PRE_MARKET: {t.strftime('%H:%M')} < 09:30 CST"
        elif t <= cls.BUY_CONTINUATION_CUTOFF:
            return True, f"MORNING_PRIME: {t.strftime('%H:%M')} within 09:30-10:45 CST"
        else:
            return False, f"BUY_CUTOFF_REACHED: {t.strftime('%H:%M')} > 10:45 CST (RANGE/REVERSION only)"

    @classmethod
    def is_short_continuation_allowed(cls, timestamp: Optional[datetime]) -> Tuple[bool, str]:
        """Return ``(True, reason)`` if SHORT_CONTINUATION is allowed at *timestamp*."""
        if timestamp is None:
            return True, "no_timestamp"

        try:
            t = timestamp.time() if hasattr(timestamp, "time") else time(10, 0)
        except Exception:
            return True, "time_parse_error"

        if t < cls.MORNING_PRIME_START:
            return False, f"PRE_MARKET: {t.strftime('%H:%M')} < 09:30 CST"
        elif t <= cls.BUY_CONTINUATION_CUTOFF:
            return True, f"MORNING_PRIME: {t.strftime('%H:%M')} within 09:30-10:45 CST"
        else:
            return False, f"SHORT_CUTOFF_REACHED: {t.strftime('%H:%M')} > 10:45 CST (RANGE/REVERSION only)"

    @classmethod
    def is_range_reversion_allowed(cls, timestamp: Optional[datetime]) -> Tuple[bool, str]:
        """Return ``(True, reason)`` if RANGE/MEAN_REVERSION is allowed at *timestamp*."""
        if timestamp is None:
            return True, "no_timestamp"

        try:
            t = timestamp.time() if hasattr(timestamp, "time") else time(12, 0)
        except Exception:
            return True, "time_parse_error"

        if cls.MIDDAY_START <= t <= cls.MIDDAY_END:
            return True, f"MIDDAY_RANGE_WINDOW: {t.strftime('%H:%M')} within 10:45-14:00 CST"
        elif time(9, 45) <= t <= time(14, 30):
            return True, f"EXTENDED_RANGE_WINDOW: {t.strftime('%H:%M')} within 09:45-14:30 CST"
        else:
            return False, f"OUTSIDE_RANGE_WINDOW: {t.strftime('%H:%M')}"

    @classmethod
    def is_reversal_blocked(cls, timestamp: Optional[datetime]) -> Tuple[bool, str]:
        """Return ``(True, reason)`` if reversals should be blocked at *timestamp*."""
        if timestamp is None:
            return False, "no_timestamp"

        try:
            t = timestamp.time() if hasattr(timestamp, "time") else time(10, 30)
        except Exception:
            return False, "time_parse_error"

        if t < cls.REVERSAL_BLOCK_UNTIL:
            return True, f"REVERSAL_BLOCKED: {t.strftime('%H:%M')} < 10:15 CST"
        else:
            return False, f"REVERSAL_ALLOWED: {t.strftime('%H:%M')} >= 10:15 CST"
