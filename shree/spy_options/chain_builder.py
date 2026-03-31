"""Option chain data structures and intraday volume tracking.

ChainSnapshot holds all option quotes for one expiry.
VolumeTracker detects volume spikes by comparing poll-to-poll increments
against a rolling baseline, since IB REST provides cumulative day-volume
(not tick-level flow data).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class OptionQuote:
    """Snapshot of a single option contract at a point in time."""

    conid: int
    symbol: str          # human-readable e.g. "SPY APR26 540C"
    strike: float
    right: str           # "C" = call, "P" = put
    expiry_month: str    # IB format e.g. "APR26"

    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    bid_size: int = 0
    ask_size: int = 0
    volume: int = 0      # cumulative day volume from IB field 87

    @property
    def mid(self) -> float:
        """Mid-market price."""
        if self.ask > 0 and self.bid > 0:
            return (self.bid + self.ask) / 2.0
        return self.last

    @property
    def spread(self) -> float:
        return max(0.0, self.ask - self.bid)

    @property
    def bid_ask_ratio(self) -> float:
        """bid_size / ask_size — >1 indicates buying pressure."""
        if self.ask_size > 0:
            return self.bid_size / self.ask_size
        return float(self.bid_size)

    @property
    def ask_bid_ratio(self) -> float:
        """ask_size / bid_size — >1 indicates selling pressure."""
        if self.bid_size > 0:
            return self.ask_size / self.bid_size
        return float(self.ask_size)


class VolumeTracker:
    """Tracks cumulative-to-incremental volume conversion per option conid.

    IB field 87 gives total day volume (cumulative).  To detect sweeps we need
    the delta between successive polls.  We maintain a rolling history of those
    deltas to compute a per-strike baseline.
    """

    def __init__(self, max_history: int = 20) -> None:
        self._last: Dict[int, int] = {}          # conid → last known cum-volume
        self._increments: Dict[int, List[int]] = {}  # conid → [delta0, delta1, ...]
        self._max_history = max_history

    def update(self, conid: int, current_volume: int) -> int:
        """Record the latest cumulative volume and return the poll-interval delta.

        Args:
            current_volume: Cumulative day volume from IB snapshot.

        Returns:
            Contracts traded since the last poll (≥ 0).
        """
        prev = self._last.get(conid, current_volume)
        increment = max(0, current_volume - prev)
        self._last[conid] = current_volume

        history = self._increments.setdefault(conid, [])
        history.append(increment)
        if len(history) > self._max_history:
            history.pop(0)
        return increment

    def rolling_avg(self, conid: int) -> float:
        """Rolling average of past poll deltas (excludes the most recent).

        Uses all-but-last entry as the baseline so the current spike is not
        self-normalising.
        """
        history = self._increments.get(conid, [])
        if len(history) < 2:
            return 0.0
        baseline = history[:-1]
        return sum(baseline) / len(baseline)

    def last_cumulative(self, conid: int) -> int:
        return self._last.get(conid, 0)

    def reset(self) -> None:
        """Clear all state (call at start of each trading day)."""
        self._last.clear()
        self._increments.clear()


class ChainSnapshot:
    """All option quotes for a single SPY expiry month."""

    def __init__(self, expiry_month: str) -> None:
        self.expiry_month = expiry_month
        self.calls: List[OptionQuote] = []
        self.puts: List[OptionQuote] = []
        self.timestamp: datetime = datetime.utcnow()

    # ── Aggregate metrics ─────────────────────────────────────────────────────

    @property
    def total_call_volume(self) -> int:
        return sum(q.volume for q in self.calls)

    @property
    def total_put_volume(self) -> int:
        return sum(q.volume for q in self.puts)

    @property
    def put_call_ratio(self) -> Optional[float]:
        cv = self.total_call_volume
        return self.total_put_volume / cv if cv > 0 else None

    # ── Strike lookups ────────────────────────────────────────────────────────

    def atm_strike(self, spy_price: float) -> float:
        """Return the listed strike closest to current SPY price."""
        all_strikes = sorted({q.strike for q in self.calls + self.puts})
        if not all_strikes:
            return spy_price
        return min(all_strikes, key=lambda s: abs(s - spy_price))

    def call_at(self, strike: float) -> Optional[OptionQuote]:
        for q in self.calls:
            if math.isclose(q.strike, strike, rel_tol=1e-4):
                return q
        return None

    def put_at(self, strike: float) -> Optional[OptionQuote]:
        for q in self.puts:
            if math.isclose(q.strike, strike, rel_tol=1e-4):
                return q
        return None

    def __repr__(self) -> str:
        return (
            f"ChainSnapshot({self.expiry_month} "
            f"calls={len(self.calls)} puts={len(self.puts)} "
            f"P/C={self.put_call_ratio:.2f if self.put_call_ratio else 'n/a'})"
        )
