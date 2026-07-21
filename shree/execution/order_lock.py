"""Order lock and submission deduplication helpers.

Extracted from ``ib_executor.py`` to isolate the concurrency-guard and
idempotency logic from the core order-placement pipeline.

Classes
-------
OrderLockManager
    Hard order lock preventing overlapping bracket placements.
SubmissionDeduplicator
    Idempotency-signature tracking to prevent duplicate orders across
    restarts and fast signal replays.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set

from ..utils.logger import logger

if TYPE_CHECKING:
    from ib_insync import Trade


class OrderLockManager:
    """Hard order lock to prevent overlapping bracket submissions.

    Tracks which IB order-IDs belong to the current lock so they can be
    bulk-cancelled if the lock times out.

    Parameters
    ----------
    timeout_seconds : int
        Maximum seconds a lock may be held before the watchdog fires.
    cancel_trade_fn : callable
        ``TradeExecutor._cancel_trade`` — called when the watchdog needs
        to cancel stale orders.
    active_orders : dict
        Reference to ``TradeExecutor.active_orders`` (live dict).
    """

    def __init__(
        self,
        timeout_seconds: int = 300,
        cancel_trade_fn: Optional[Callable[..., Any]] = None,
        active_orders: Optional[Dict[int, "Trade"]] = None,
    ) -> None:
        self._locked: bool = False
        self._reason: Optional[str] = None
        self._engaged_at: Optional[datetime] = None
        self._parent_id: Optional[int] = None
        self._order_ids: Set[int] = set()
        self._timeout_seconds: int = timeout_seconds
        self._cancel_trade_fn = cancel_trade_fn
        self._active_orders = active_orders if active_orders is not None else {}

    # -- public queries ------------------------------------------------

    @property
    def is_locked(self) -> bool:
        self._enforce_timeout()
        return self._locked

    @property
    def reason(self) -> Optional[str]:
        self._enforce_timeout()
        return self._reason

    @property
    def age_seconds(self) -> float:
        self._enforce_timeout()
        return self._age()

    # -- mutations -----------------------------------------------------

    def engage(self, reason: str) -> None:
        """Engage the lock.  Only one lock may be held at a time."""
        self._locked = True
        self._reason = reason
        self._engaged_at = datetime.utcnow()
        self._order_ids.clear()
        self._parent_id = None
        logger.warning(f"🔒 Order lock engaged: {reason}")

    def release(self, context: str = "") -> None:
        """Release the lock and clear all tracked order-IDs."""
        if self._locked:
            logger.info(f"🔓 Order lock released ({context})")
        self._locked = False
        self._reason = None
        self._engaged_at = None
        self._parent_id = None
        self._order_ids.clear()

    def force_release(self, reason: str = "manual override", cancel_tracked: bool = True) -> None:
        """External release with optional cancellation of tracked orders."""
        if cancel_tracked:
            self._cancel_tracked_orders(reason)
        self.release(reason)

    def register_order_id(self, order_id: Optional[int]) -> None:
        """Associate an IB order-ID with the current lock."""
        if not self._locked or order_id is None:
            return
        self._order_ids.add(order_id)
        if self._parent_id is None:
            self._parent_id = order_id

    # -- internals -----------------------------------------------------

    def _age(self) -> float:
        if not self._locked or not self._engaged_at:
            return 0.0
        return (datetime.utcnow() - self._engaged_at).total_seconds()

    def _enforce_timeout(self) -> None:
        """Watchdog: automatically release a lock that exceeds the timeout."""
        if not self._locked or self._timeout_seconds <= 0:
            return
        if self._age() < self._timeout_seconds:
            return
        logger.error(
            "⏰ Order lock watchdog triggered after {:.0f}s (reason={}, parent={})",
            self._age(),
            self._reason,
            self._parent_id,
        )
        self._cancel_tracked_orders("lock_timeout")
        self.release("watchdog timeout")

    def _cancel_tracked_orders(self, reason: str) -> None:
        if not self._order_ids:
            return
        logger.warning("🧹 Canceling {} locked orders ({})", len(self._order_ids), reason)
        for order_id in list(self._order_ids):
            trade = self._active_orders.get(order_id)
            if trade is None or self._cancel_trade_fn is None:
                continue
            self._cancel_trade_fn(trade, reason, warn_if_missing=False)
        self._order_ids.clear()


class SubmissionDeduplicator:
    """Idempotency-signature tracker to prevent duplicate order submissions.

    Maintains both an in-memory LRU and an optional ``OrderTracker``
    (SQLite) backend for persistence across restarts.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g. ``"MES"``).
    ttl_seconds : int
        How long a signature remains "hot".
    order_tracker : OrderTracker | None
        Optional SQLite-backed tracker for durable dedup.
    """

    def __init__(
        self,
        symbol: str,
        ttl_seconds: int = 900,
        order_tracker: Optional["OrderTracker"] = None,
    ) -> None:
        self._symbol = symbol
        self._ttl = ttl_seconds
        self._tracker = order_tracker
        self._signatures: Dict[str, datetime] = {}
        self._signal_keys: Dict[str, datetime] = {}

    # -- fine-grained (hash-based) dedup --------------------------------

    def generate_signature(
        self,
        action: str,
        quantity: int,
        metadata: Dict[str, Any],
    ) -> str:
        """Build a SHA-256 idempotency hash from order context."""
        bar_ts = metadata.get("bar_close_timestamp") or ""
        signal_id = (
            metadata.get("signal_id")
            or metadata.get("trade_cycle_id")
            or ""
        )
        strategy = (
            metadata.get("strategy_name")
            or metadata.get("signal_source")
            or "unknown"
        )
        bucket = metadata.get("entry_price_bucket")
        bucket_str = f"{bucket:.2f}" if bucket is not None else ""
        payload = "|".join([
            self._symbol,
            action.upper(),
            str(bar_ts),
            str(signal_id),
            strategy,
            str(quantity),
            bucket_str,
        ])
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def is_duplicate(self, signature: str) -> bool:
        now = datetime.now(timezone.utc)
        ts = self._signatures.get(signature)
        if ts and (now - ts).total_seconds() < self._ttl:
            return True
        elif ts:
            del self._signatures[signature]
        # Check persistent store
        try:
            if self._tracker and self._tracker.signature_exists(signature, self._ttl):
                return True
        except Exception as exc:
            logger.debug(f"Signature lookup skipped: {exc}")
        return False

    def record(
        self,
        signature: str,
        action: str,
        quantity: int,
        metadata: Dict[str, Any],
    ) -> None:
        now = datetime.now(timezone.utc)
        self._signatures[signature] = now
        # Persist to SQLite
        try:
            if self._tracker:
                self._tracker.record_submission_signature(
                    signature=signature,
                    symbol=self._symbol,
                    action=action,
                    quantity=quantity,
                    price_bucket=metadata.get("entry_price_bucket"),
                    bar_timestamp=metadata.get("bar_close_timestamp"),
                    signal_id=(
                        metadata.get("signal_id")
                        or metadata.get("trade_cycle_id")
                    ),
                    strategy_name=(
                        metadata.get("strategy_name")
                        or metadata.get("signal_source")
                    ),
                )
        except Exception as exc:
            logger.debug(f"Signature persistence failed: {exc}")
        self._evict_stale(self._signatures)

    # -- coarse (signal-key) dedup -------------------------------------

    def build_signal_key(self, action: str, metadata: Dict[str, Any]) -> str:
        """Coarse idempotency key: symbol + bar timestamp + action."""
        ts = (
            metadata.get("bar_close_timestamp")
            or metadata.get("timestamp")
            or metadata.get("signal_id")
            or metadata.get("trade_cycle_id")
        )
        if not ts:
            ts = datetime.now(timezone.utc).isoformat()
        return f"{self._symbol}|{action.upper()}|{ts}"

    def is_duplicate_signal_key(self, key: str) -> bool:
        now = datetime.now(timezone.utc)
        ts = self._signal_keys.get(key)
        return bool(ts and (now - ts).total_seconds() < self._ttl)

    def record_signal_key(self, key: str) -> None:
        self._signal_keys[key] = datetime.now(timezone.utc)
        self._evict_stale(self._signal_keys)

    # -- housekeeping --------------------------------------------------

    def _evict_stale(self, store: Dict[str, datetime]) -> None:
        now = datetime.now(timezone.utc)
        stale = [k for k, ts in store.items() if (now - ts).total_seconds() > self._ttl]
        for k in stale:
            store.pop(k, None)
