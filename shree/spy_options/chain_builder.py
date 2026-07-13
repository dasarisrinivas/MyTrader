"""Option chain data structures, liquidity filtering, and intraday volume tracking.

ChainSnapshot holds all option quotes for one expiry.
VolumeTracker detects volume spikes by comparing poll-to-poll increments
against a rolling baseline, since IB provides cumulative day-volume.
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
    volume: int = 0      # cumulative day volume from IB

    # Greeks (from IB modelGreeks — 0.0 if unavailable)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0   # per day, negative for long options
    vega: float = 0.0
    impl_vol: float = 0.0

    # Liquidity
    open_interest: int = 0

    @property
    def mid(self) -> float:
        if self.ask > 0 and self.bid > 0:
            return (self.bid + self.ask) / 2.0
        return self.last

    @property
    def spread(self) -> float:
        return max(0.0, self.ask - self.bid)

    @property
    def spread_pct(self) -> float:
        """Bid/ask spread as percentage of mid price."""
        m = self.mid
        if m > 0:
            return self.spread / m * 100.0
        return 0.0

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


def passes_liquidity(
    quote: OptionQuote,
    min_oi: int = 1000,
    max_spread_pct: float = 8.0,
    min_volume: int = 500,
) -> bool:
    """Return True if quote passes all liquidity gates.

    Called in manager._build_chain() before adding quotes to ChainSnapshot.
    Contracts failing this are silently dropped — no signal can fire on them.
    """
    # Require a real two-sided quote. Without this, a contract with no live
    # market (bid=ask=0 — a slow tick, or IB delayed-data mode where EVERY field
    # reads 0) sails through: mid=0 → spread_pct=0, and the OI/volume gates below
    # only fire on NONZERO-but-below-floor values, so all-zero passes as if it
    # were the most liquid strike on the board. (audit 2026-07-13)
    if not (quote.bid > 0 and quote.ask > 0):
        return False
    if quote.open_interest > 0 and quote.open_interest < min_oi:
        return False
    if quote.spread_pct > max_spread_pct and quote.spread_pct > 0:
        return False
    if quote.volume > 0 and quote.volume < min_volume:
        return False
    return True


class VolumeTracker:
    """Tracks cumulative-to-incremental volume conversion per option conid.

    IB provides total day volume (cumulative). To detect sweeps we need
    the delta between successive polls. We maintain a rolling history of those
    deltas to compute a per-strike baseline.
    """

    def __init__(self, max_history: int = 20) -> None:
        self._last: Dict[int, int] = {}
        self._increments: Dict[int, List[int]] = {}
        self._max_history = max_history

    def update(self, conid: int, current_volume: int) -> int:
        """Record latest cumulative volume and return the poll-interval delta."""
        prev = self._last.get(conid, current_volume)
        increment = max(0, current_volume - prev)
        self._last[conid] = current_volume

        history = self._increments.setdefault(conid, [])
        history.append(increment)
        if len(history) > self._max_history:
            history.pop(0)
        return increment

    def rolling_avg(self, conid: int) -> float:
        """Rolling average of past poll deltas (excludes most recent)."""
        history = self._increments.get(conid, [])
        if len(history) < 2:
            return 0.0
        baseline = history[:-1]
        return sum(baseline) / len(baseline)

    def last_cumulative(self, conid: int) -> int:
        return self._last.get(conid, 0)

    def reset(self) -> None:
        self._last.clear()
        self._increments.clear()


class ChainSnapshot:
    """All option quotes for a single SPY expiry month."""

    def __init__(self, expiry_month: str, expiry_date: str = "") -> None:
        self.expiry_month = expiry_month
        self.expiry_date = expiry_date   # YYYYMMDD (e.g. "20260417")
        self.calls: List[OptionQuote] = []
        self.puts: List[OptionQuote] = []
        self.timestamp: datetime = datetime.utcnow()

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

    def atm_strike(self, spy_price: float) -> float:
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
            f"calls={len(self.calls)} puts={len(self.puts)})"
        )
