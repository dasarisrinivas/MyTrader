"""IBKR contract factory for COMEX Gold futures (GC / MGC).

Design goals:
- Return a *qualified* ib_insync Contract so the caller never needs to
  interact with TWS symbol/exchange details.
- Cache qualified contracts to avoid repeated round-trips.
- Fail loudly and specifically when GC is requested without explicit opt-in
  (prevents accidentally trading the $100/point full-size contract).
- Expose a sync path (``build_unqualified``) usable without a live IB
  connection — needed by tests.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from ib_insync import Future, IB

from ...config.gold import GoldStrategyConfig
from ...utils.logger import logger


# Contract cache TTL — re-qualify after this many seconds (handles contract rolls)
_CACHE_TTL_SECONDS = 3600

# COMEX gold (GC/MGC): IB reports the *delivery* date in lastTradeDateOrContractMonth,
# not the actual Last Trading Day (LTD).  For April delivery, IB returns ~Apr 28 but
# the LTD is ~32 days earlier (late March).  Roll early by filtering out any contract
# whose reported delivery date is within this many days, ensuring we never get stuck
# on a contract past its LTD.
_ROLL_DAYS_BEFORE_DELIVERY = 35


class GoldContractFactory:
    """Create and qualify IBKR futures contracts for Gold.

    Usage::

        factory = GoldContractFactory(config, ib)
        contract = await factory.get_qualified_contract()
    """

    def __init__(self, config: GoldStrategyConfig, ib: Optional[IB] = None) -> None:
        self._cfg = config
        self._ib = ib
        self._cached_contract: Optional[Future] = None
        self._cache_time: float = 0.0

    def build_unqualified(self) -> Future:
        """Build an unqualified Future contract object (no IB round-trip).

        Suitable for tests or pre-flight validation.
        Uses the last-traded contract month when empty (IB selects front month).

        Raises:
            ValueError: if GC is requested but ``allow_gc`` is not set.
        """
        symbol = self._cfg.symbol
        exchange = self._cfg.exchange
        currency = self._cfg.currency

        if symbol.upper() == "GC" and not self._cfg.allow_gc:
            raise ValueError(
                "GC (Full Gold, 100 oz) requires 'gold.allow_gc: true'. "
                "Refusing to build contract without explicit opt-in."
            )

        # Leaving lastTradeDateOrContractMonth empty lets IB select the
        # front-month contract automatically.
        contract = Future(
            symbol=symbol,
            exchange=exchange,
            currency=currency,
        )
        logger.debug(
            "GoldContractFactory: built unqualified contract %s/%s/%s",
            symbol,
            exchange,
            currency,
        )
        return contract

    async def get_qualified_contract(self) -> Future:
        """Return a qualified IBKR contract, using cache when fresh.

        Raises:
            RuntimeError: if no IB instance was provided at construction.
            ValueError:   if qualification fails (bad symbol / no market data).
        """
        if self._ib is None:
            raise RuntimeError(
                "GoldContractFactory: IB instance required for contract qualification. "
                "Pass ib= at construction, or use build_unqualified() for offline use."
            )

        now = time.monotonic()
        if self._cached_contract is not None and (now - self._cache_time) < _CACHE_TTL_SECONDS:
            return self._cached_contract

        contract = self.build_unqualified()

        logger.info(
            f"GoldContractFactory: qualifying contract {contract.symbol} on {contract.exchange} …",
        )
        # Use reqContractDetailsAsync — same approach as the MES executor.
        # This returns every listed contract for the symbol/exchange pair,
        # so we can filter expired ones and pick the true front month.
        details = await self._ib.reqContractDetailsAsync(contract)

        if not details:
            raise ValueError(
                f"IB returned no contract details for symbol={contract.symbol} "
                f"exchange={contract.exchange} currency={contract.currency}.\n"
                "Possible causes:\n"
                "  1. Paper account lacks COMEX futures market data subscription.\n"
                "     → TWS/Gateway: Account → Market Data Subscriptions → add COMEX.\n"
                "  2. Wrong port: paper uses 4002, live uses 4001.\n"
                "  3. Symbol or exchange typo in config."
            )

        now_utc = datetime.now(timezone.utc)

        def _expiry_dt(d) -> Optional[datetime]:
            raw = getattr(d.contract, "lastTradeDateOrContractMonth", "") or ""
            try:
                if len(raw) >= 8:
                    return datetime.strptime(raw[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
                if len(raw) == 6:
                    from calendar import monthrange as _mr
                    y, mo = int(raw[:4]), int(raw[4:6])
                    return datetime(y, mo, _mr(y, mo)[1], 23, 59, 59, tzinfo=timezone.utc)
            except (ValueError, OverflowError):
                pass
            return None

        # Filter out contracts whose delivery date is within _ROLL_DAYS_BEFORE_DELIVERY days.
        # IB reports the delivery date (not LTD) for COMEX gold; for MGC the LTD is ~32
        # days before delivery, so a 35-day buffer ensures we never trade past LTD.
        roll_cutoff = now_utc + timedelta(days=_ROLL_DAYS_BEFORE_DELIVERY)
        valid = [d for d in details if (_expiry_dt(d) or now_utc) >= roll_cutoff]
        if not valid:
            # Fallback: relax to standard expiry filter (avoids total blackout if
            # all listed contracts are within the buffer, e.g. during data issues)
            logger.warning(
                "GoldContractFactory: no contracts outside {}-day roll window — "
                "falling back to standard expiry filter",
                _ROLL_DAYS_BEFORE_DELIVERY,
            )
            valid = [d for d in details if (_expiry_dt(d) or now_utc) >= now_utc]
        if not valid:
            valid = details   # All appear expired — fall back to nearest

        valid.sort(key=lambda d: d.contract.lastTradeDateOrContractMonth)
        result: Future = valid[0].contract

        self._cached_contract = result
        self._cache_time = now

        logger.info(
            "GoldContractFactory: front-month selected — conId={} localSymbol={} expiry={}",
            result.conId,
            result.localSymbol or result.symbol,
            result.lastTradeDateOrContractMonth,
        )
        return result

    def invalidate_cache(self) -> None:
        """Force re-qualification on next call (e.g., after contract roll)."""
        self._cached_contract = None
        self._cache_time = 0.0
        logger.info("GoldContractFactory: contract cache invalidated")

