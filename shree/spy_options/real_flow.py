"""Real order-flow data for SPY — Level 2 depth + tick-by-tick time & sales.

Replaces two proxies with real data (JUL 5 2026):

  1. TAPE (time & sales) — ``reqTickByTickData(SPY, 'AllLast')``.
     Every print is classified against the prevailing NBBO:
         price >= ask  → buy aggression  (lifted the offer)
         price <= bid  → sell aggression (hit the bid)
         inside spread → neutral (ignored)
     Rolling per-minute buckets produce a tape score (-100..+100) and a
     large-print bias from block-sized trades.  This is REAL aggression —
     unlike the volume-spike-vs-rolling-average proxy in the signal engine.

  2. DEPTH (Level 2) — ``reqMktDepth(SPY, isSmartDepth=True)``.
     Aggregated SMART book, top N levels each side → depth imbalance
     (-1..+1).  Positive = resting bid size dominates (support below),
     negative = offer size dominates (supply above).

Both feeds degrade gracefully: depth requires paid entitlements
(NASDAQ TotalView / NYSE ArcaBook); if IB rejects the request or no updates
arrive, the feed marks itself unavailable and the bot runs exactly as before.

Consumption:
  * ``RealFlowFeed.snapshot()`` is called once per poll by the manager and
    injected into ``ExternalContext`` (tape_score, depth_imbalance, ...).
  * DynamicConfidence block 19 applies small, conservative adjustments —
    these weights are UNBACKTESTED (no historical tick data available) and
    are deliberately capped at ±5% pending live calibration.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

from ..utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RealFlowState:
    """Point-in-time snapshot of real order-flow metrics for SPY."""

    # Tape (time & sales)
    tape_available: bool = False
    tape_score: float = 0.0            # -100..+100  (buy − sell) / total
    tape_buy_vol: int = 0              # shares lifted at/above ask (window)
    tape_sell_vol: int = 0             # shares hit at/below bid (window)
    tape_prints: int = 0               # classified prints in window
    tape_large_bias: str = "NEUTRAL"   # BUY | SELL | NEUTRAL (block trades)
    tape_large_buy_vol: int = 0
    tape_large_sell_vol: int = 0

    # Depth (Level 2)
    depth_available: bool = False
    depth_imbalance: float = 0.0       # -1..+1  (bid − ask) / (bid + ask)
    depth_bid_qty: int = 0             # total size, top N bid levels
    depth_ask_qty: int = 0             # total size, top N ask levels


# ─────────────────────────────────────────────────────────────────────────────
# Feed
# ─────────────────────────────────────────────────────────────────────────────

class RealFlowFeed:
    """Manages SPY tick-by-tick tape and Level 2 depth subscriptions.

    Uses the existing ib_insync ``IB`` connection (no separate socket).
    Call ``await start()`` once after the gateway connects, ``snapshot()``
    every poll, and ``stop()`` on shutdown.
    """

    def __init__(
        self,
        ib,                                 # ib_insync.IB (shared connection)
        spy_contract,                       # qualified Stock('SPY', 'SMART', 'USD')
        tape_enabled: bool = True,
        depth_enabled: bool = True,
        depth_levels: int = 5,
        tape_window_minutes: int = 5,
        large_print_shares: int = 10_000,
    ) -> None:
        self._ib = ib
        self._contract = spy_contract
        self._tape_enabled = tape_enabled
        self._depth_enabled = depth_enabled
        self._depth_levels = max(1, depth_levels)
        self._window_min = max(1, tape_window_minutes)
        self._large_shares = large_print_shares

        self._tape_ticker = None
        self._depth_ticker = None

        # Per-minute tape buckets: minute_epoch → [buy, sell, large_buy, large_sell, prints]
        self._buckets: Dict[int, List[int]] = {}

        # Last processed index into ticker.tickByTicks (ib_insync appends batches)
        self._tape_update_count = 0
        self._depth_update_count = 0

        # Current top-of-book for print classification (from depth book when
        # available, else from the bid/ask on the tape ticker itself)
        self._bid: float = 0.0
        self._ask: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Subscribe to tape and depth.  Failures disable that feed only."""
        if self._tape_enabled:
            try:
                self._tape_ticker = self._ib.reqTickByTickData(
                    self._contract, "AllLast", numberOfTicks=0, ignoreSize=False
                )
                self._tape_ticker.updateEvent += self._on_tape_update
                logger.info("RealFlow: tick-by-tick tape subscribed (SPY AllLast)")
            except Exception as exc:
                logger.warning("RealFlow: tape subscription failed: {}", exc)
                self._tape_ticker = None

        if self._depth_enabled:
            try:
                self._depth_ticker = self._ib.reqMktDepth(
                    self._contract, numRows=self._depth_levels, isSmartDepth=True
                )
                self._depth_ticker.updateEvent += self._on_depth_update
                logger.info(
                    "RealFlow: L2 depth subscribed (SPY SMART, {} levels)",
                    self._depth_levels,
                )
            except Exception as exc:
                logger.warning(
                    "RealFlow: depth subscription failed (needs TotalView/ArcaBook "
                    "entitlement?): {}", exc,
                )
                self._depth_ticker = None

    def stop(self) -> None:
        try:
            if self._tape_ticker is not None:
                self._ib.cancelTickByTickData(self._contract, "AllLast")
                self._tape_ticker = None
        except Exception:
            pass
        try:
            if self._depth_ticker is not None:
                self._ib.cancelMktDepth(self._contract, isSmartDepth=True)
                self._depth_ticker = None
        except Exception:
            pass

    # ── Event handlers (called by ib_insync event loop) ──────────────────────

    def _on_depth_update(self, ticker) -> None:
        self._depth_update_count += 1
        # Track top-of-book for tape classification
        try:
            if ticker.domBids:
                self._bid = float(ticker.domBids[0].price)
            if ticker.domAsks:
                self._ask = float(ticker.domAsks[0].price)
        except Exception:
            pass

    def _on_tape_update(self, ticker) -> None:
        """Classify the newest batch of prints against the prevailing NBBO."""
        try:
            ticks = ticker.tickByTicks
            if not ticks:
                return
            self._tape_update_count += 1

            # Fallback top-of-book from the tape ticker itself (bid/ask fields
            # populate when no depth feed is running)
            bid, ask = self._bid, self._ask
            if (bid <= 0 or ask <= 0) and ticker.bid and ticker.ask:
                bid = float(ticker.bid)
                ask = float(ticker.ask)
            if bid <= 0 or ask <= 0 or ask < bid:
                return  # cannot classify without a sane book

            minute = int(datetime.now(timezone.utc).timestamp() // 60)
            bucket = self._buckets.setdefault(minute, [0, 0, 0, 0, 0])

            for t in ticks:
                px = float(getattr(t, "price", 0.0) or 0.0)
                sz = int(getattr(t, "size", 0) or 0)
                if px <= 0 or sz <= 0:
                    continue
                if px >= ask:                       # lifted the offer → buyer
                    bucket[0] += sz
                    if sz >= self._large_shares:
                        bucket[2] += sz
                    bucket[4] += 1
                elif px <= bid:                     # hit the bid → seller
                    bucket[1] += sz
                    if sz >= self._large_shares:
                        bucket[3] += sz
                    bucket[4] += 1
                # inside-spread prints are ambiguous — skipped by design

            self._prune(minute)
        except Exception as exc:                    # never break the event loop
            logger.debug("RealFlow tape handler error: {}", exc)

    def _prune(self, current_minute: int) -> None:
        cutoff = current_minute - self._window_min
        stale = [m for m in self._buckets if m < cutoff]
        for m in stale:
            del self._buckets[m]

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> RealFlowState:
        state = RealFlowState()

        # Tape
        if self._tape_ticker is not None and self._tape_update_count > 0:
            now_min = int(datetime.now(timezone.utc).timestamp() // 60)
            self._prune(now_min)
            buy = sell = lbuy = lsell = prints = 0
            for b in self._buckets.values():
                buy += b[0]; sell += b[1]; lbuy += b[2]; lsell += b[3]; prints += b[4]
            total = buy + sell
            state.tape_available = prints > 0
            state.tape_buy_vol = buy
            state.tape_sell_vol = sell
            state.tape_prints = prints
            state.tape_large_buy_vol = lbuy
            state.tape_large_sell_vol = lsell
            if total > 0:
                state.tape_score = round((buy - sell) / total * 100.0, 1)
            ltotal = lbuy + lsell
            if ltotal > 0:
                if lbuy >= ltotal * 0.65:
                    state.tape_large_bias = "BUY"
                elif lsell >= ltotal * 0.65:
                    state.tape_large_bias = "SELL"

        # Depth
        if self._depth_ticker is not None and self._depth_update_count > 0:
            try:
                bids = self._depth_ticker.domBids[: self._depth_levels]
                asks = self._depth_ticker.domAsks[: self._depth_levels]
                bid_qty = sum(int(l.size) for l in bids if l.size)
                ask_qty = sum(int(l.size) for l in asks if l.size)
                total = bid_qty + ask_qty
                if total > 0:
                    state.depth_available = True
                    state.depth_bid_qty = bid_qty
                    state.depth_ask_qty = ask_qty
                    state.depth_imbalance = round((bid_qty - ask_qty) / total, 3)
            except Exception:
                pass

        return state
