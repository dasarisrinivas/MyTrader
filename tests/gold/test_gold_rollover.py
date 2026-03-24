"""Tests for ContractRollMonitor and should_roll."""
from __future__ import annotations

from datetime import date

import pytest
from ib_insync import Future

from shree.config.gold import GoldStrategyConfig
from shree.execution.gold.rollover import ContractRollMonitor, should_roll


def _make_contract(expiry: str) -> Future:
    """Build a minimal Future with lastTradeDateOrContractMonth set."""
    c = Future(symbol="MGC", exchange="COMEX", currency="USD")
    c.lastTradeDateOrContractMonth = expiry
    return c


class TestShouldRoll:
    def test_not_rolling_when_far_away(self) -> None:
        c = _make_contract("20260620")
        today = date(2026, 6, 1)   # 19 days before expiry
        assert not should_roll(c, today=today, warn_days=7)

    def test_rolling_within_warn_window(self) -> None:
        c = _make_contract("20260620")
        today = date(2026, 6, 15)  # 5 days before expiry
        assert should_roll(c, today=today, warn_days=7)

    def test_rolling_on_expiry_day(self) -> None:
        c = _make_contract("20260620")
        assert should_roll(c, today=date(2026, 6, 20), warn_days=7)

    def test_yyyymm_format(self) -> None:
        """6-digit format: YYYYMM → approximate last day of month."""
        c = _make_contract("202606")   # ~June 30
        today = date(2026, 6, 25)      # within 7 days of June 30
        assert should_roll(c, today=today, warn_days=7)

    def test_empty_expiry_never_rolls(self) -> None:
        c = _make_contract("")
        assert not should_roll(c)

    def test_invalid_expiry_never_rolls(self) -> None:
        c = _make_contract("BADDATA")
        assert not should_roll(c)


class TestContractRollMonitor:
    def _monitor(self) -> ContractRollMonitor:
        return ContractRollMonitor(GoldStrategyConfig(), warn_days=7, block_days=2)

    def test_needs_roll_false_when_far(self) -> None:
        monitor = self._monitor()
        c = _make_contract("20260630")
        assert not monitor.needs_roll(c, today=date(2026, 6, 1))

    def test_needs_roll_true_within_block_window(self) -> None:
        monitor = self._monitor()
        c = _make_contract("20260620")
        assert monitor.needs_roll(c, today=date(2026, 6, 19))

    def test_needs_roll_false_outside_block_window(self) -> None:
        monitor = self._monitor()
        c = _make_contract("20260620")
        assert not monitor.needs_roll(c, today=date(2026, 6, 10))  # 10 days away

    def test_reset_clears_notified_expiry(self) -> None:
        monitor = self._monitor()
        c = _make_contract("20260620")
        monitor.needs_roll(c, today=date(2026, 6, 19))
        assert monitor._last_notified_expiry is not None
        monitor.reset()
        assert monitor._last_notified_expiry is None

    def test_check_and_warn_does_not_raise(self) -> None:
        """check_and_warn must not raise even with a far-future expiry."""
        monitor = self._monitor()
        c = _make_contract("20261231")
        monitor.check_and_warn(c)  # Should log nothing / log warning silently

    def test_parse_expiry_december_rollover(self) -> None:
        """YYYYMM=202612 → December 31 2026."""
        c = _make_contract("202612")
        expiry = ContractRollMonitor._parse_expiry(c)
        assert expiry == date(2026, 12, 31)
