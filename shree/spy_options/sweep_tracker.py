"""Repeat sweep detector for SPY options signal bot.

Tracks when the same (strike, right, expiry) combination triggers a volume
spike more than once within a rolling time window. Repeated sweeps are a
stronger signal than a single spike — smart money is accumulating a position.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List


@dataclass
class _SweepEvent:
    key: str        # f"{strike:.0f}{right}{expiry}"
    ts: datetime


class SweepTracker:
    """Tracks repeat sweep events per (strike, right, expiry) key.

    All methods are synchronous and fast (no I/O). Designed to be called
    inside the signal engine once per detected volume spike.
    """

    def __init__(self, window_minutes: int = 15) -> None:
        self._window = timedelta(minutes=window_minutes)
        self._events: List[_SweepEvent] = []

    def _key(self, strike: float, right: str, expiry: str) -> str:
        return f"{strike:.0f}{right}{expiry}"

    def _prune(self) -> None:
        """Remove events older than the rolling window."""
        cutoff = datetime.utcnow() - self._window
        self._events = [e for e in self._events if e.ts >= cutoff]

    def record(self, strike: float, right: str, expiry: str) -> None:
        """Record a sweep event. Call whenever a volume spike is detected."""
        self._events.append(_SweepEvent(key=self._key(strike, right, expiry), ts=datetime.utcnow()))

    def flow_score(self, strike: float, right: str, expiry: str) -> float:
        """Return a 0.0–1.0 score based on repeat sweeps in the window.

        Score:
            0.0  — seen fewer than 2 times (single occurrence, no boost)
            0.5  — seen exactly 2 times (repeat confirmation)
            1.0  — seen 3+ times (persistent accumulation)
        """
        self._prune()
        key = self._key(strike, right, expiry)
        count = sum(1 for e in self._events if e.key == key)
        if count >= 3:
            return 1.0
        if count == 2:
            return 0.5
        return 0.0

    def reset(self) -> None:
        """Clear all state — called on daily reset in manager."""
        self._events.clear()
