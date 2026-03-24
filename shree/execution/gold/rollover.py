"""COMEX Gold futures contract roll guard.

COMEX Gold (GC / MGC) contracts expire on the third-to-last business day
of the contract month.  Liquidity migrates to the next front month several
days earlier — typically around the 20th of the prior month.

This module provides:

1. ``ContractRollMonitor`` — inspects the qualified contract's expiry date
   and emits a warning (or blocks trading) when we are within the
   ``days_before_expiry_warn`` window.

2. ``should_roll(contract, today)`` — pure function, no IB dependency.
   Returns True when the contract is within the configured roll window.
   Used in tests and for pre-flight checks.

Usage in GoldTradingManager::

    roll_monitor = ContractRollMonitor(cfg)

    # At startup (after contract qualification):
    roll_monitor.check_and_warn(contract)

    # In the main loop (once per bar or once per hour):
    if roll_monitor.needs_roll(contract):
        factory.invalidate_cache()
        contract = await factory.get_qualified_contract()
        roll_monitor.reset()
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

from ib_insync import Future

from ...config.gold import GoldStrategyConfig
from ...utils.logger import logger


# Default number of calendar days before last-trade-date to start warning.
# Gold liquidity typically migrates ~5 days before expiry.
_DEFAULT_WARN_DAYS = 5
# Default number of calendar days before expiry to hard-block trading
_DEFAULT_BLOCK_DAYS = 2


class ContractRollMonitor:
    """Track the active contract's expiry and signal when a roll is needed.

    No IB connection is required — it reads the ``lastTradeDateOrContractMonth``
    field that IB sets after ``qualifyContracts``.
    """

    def __init__(
        self,
        cfg: GoldStrategyConfig,
        warn_days: int = _DEFAULT_WARN_DAYS,
        block_days: int = _DEFAULT_BLOCK_DAYS,
    ) -> None:
        self._cfg = cfg
        self._warn_days = warn_days
        self._block_days = block_days
        self._last_notified_expiry: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def check_and_warn(self, contract: Future) -> None:
        """Log a warning if the contract is approaching expiry."""
        expiry = self._parse_expiry(contract)
        if expiry is None:
            return
        today = date.today()
        days_left = (expiry - today).days
        if days_left <= self._warn_days:
            logger.warning(
                "ContractRollMonitor: %s expires %s (%d days) — "
                "consider rolling to next front month",
                contract.localSymbol or contract.symbol,
                expiry,
                days_left,
            )

    def needs_roll(self, contract: Future, today: Optional[date] = None) -> bool:
        """Return True when the contract is within the hard-block window.

        When this returns True, the manager should invalidate the contract
        cache and re-qualify to get the next front month.
        """
        expiry = self._parse_expiry(contract)
        if expiry is None:
            return False
        today = today or date.today()
        days_left = (expiry - today).days
        if days_left <= self._block_days:
            if self._last_notified_expiry != str(expiry):
                logger.warning(
                    "ContractRollMonitor: %s has %d day(s) until expiry — triggering roll",
                    contract.localSymbol or contract.symbol,
                    days_left,
                )
                self._last_notified_expiry = str(expiry)
            return True
        return False

    def reset(self) -> None:
        """Call after a successful roll so next expiry triggers fresh logging."""
        self._last_notified_expiry = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_expiry(contract: Future) -> Optional[date]:
        """Parse IB's lastTradeDateOrContractMonth field (YYYYMMDD or YYYYMM)."""
        raw: str = contract.lastTradeDateOrContractMonth or ""
        raw = raw.strip()
        if not raw:
            return None
        try:
            if len(raw) == 8:   # YYYYMMDD
                return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
            if len(raw) == 6:   # YYYYMM — approximate: last calendar day of month
                year, month = int(raw[:4]), int(raw[4:6])
                # First day of next month minus one day
                if month == 12:
                    return date(year + 1, 1, 1) - timedelta(days=1)
                return date(year, month + 1, 1) - timedelta(days=1)
        except (ValueError, OverflowError) as exc:
            logger.debug("ContractRollMonitor: could not parse expiry {!r}: {}", raw, exc)
        return None


# ── Pure function for testing / pre-flight ────────────────────────────────────

def should_roll(
    contract: Future,
    today: Optional[date] = None,
    warn_days: int = _DEFAULT_WARN_DAYS,
) -> bool:
    """Return True if the contract is within warn_days of expiry.

    This is a stateless helper — no side effects, no logging.
    """
    today = today or date.today()
    expiry = ContractRollMonitor._parse_expiry(contract)
    if expiry is None:
        return False
    return (expiry - today).days <= warn_days
