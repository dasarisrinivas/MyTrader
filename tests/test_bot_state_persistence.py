"""
tests/test_bot_state_persistence.py

Tests for shree/utils/bot_state.py — Fix #14 persistence across restarts.

MAR 13 2026: The consecutive-loss counter must survive bot restarts so that
the 3-loss cooldown can accumulate across same-day restarts.

MAR 16 2026 (Fix #6): Added realized_pnl_today persistence tests. The daily
P&L must survive restarts so the $250 daily loss cap is enforced even after
mid-day restarts.
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_state(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


CT_TODAY = "2026-03-13"  # Fixed for all tests

# Patch now_cst to return a stable CT date
def _mock_now_cst():
    from datetime import date
    class _FakeDT:
        def date(self):
            class _D:
                def isoformat(self_inner):
                    return CT_TODAY
            return _D()
    return _FakeDT()


# ── load_bot_state ────────────────────────────────────────────────────────────

class TestLoadBotState:

    def test_missing_file_returns_defaults(self, tmp_path):
        from shree.utils.bot_state import load_bot_state
        path = tmp_path / "bot_state.json"
        count, cooldown, pnl = load_bot_state(path=path)
        assert count == 0
        assert cooldown is None
        assert pnl == 0.0

    def test_corrupt_json_returns_defaults(self, tmp_path):
        from shree.utils.bot_state import load_bot_state
        path = tmp_path / "bot_state.json"
        path.write_text("{ not valid json }")
        count, cooldown, pnl = load_bot_state(path=path)
        assert count == 0
        assert cooldown is None
        assert pnl == 0.0

    def test_valid_state_loaded(self, tmp_path):
        from shree.utils.bot_state import load_bot_state
        path = tmp_path / "bot_state.json"
        _write_state(path, {
            "consecutive_loss_count": 2,
            "extra_cooldown_until": None,
            "realized_pnl_today": -75.50,
            "last_trade_date": CT_TODAY,
            "written_at": datetime.now(timezone.utc).isoformat(),
        })
        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            count, cooldown, pnl = load_bot_state(path=path)
        assert count == 2
        assert cooldown is None
        assert pnl == -75.50

    def test_active_cooldown_restored(self, tmp_path):
        from shree.utils.bot_state import load_bot_state
        future = datetime.now(timezone.utc) + timedelta(minutes=20)
        path = tmp_path / "bot_state.json"
        _write_state(path, {
            "consecutive_loss_count": 3,
            "extra_cooldown_until": future.isoformat(),
            "last_trade_date": CT_TODAY,
            "written_at": datetime.now(timezone.utc).isoformat(),
        })
        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            count, cooldown, pnl = load_bot_state(path=path)
        assert count == 3
        assert cooldown is not None
        assert cooldown > datetime.now(timezone.utc)

    def test_expired_cooldown_not_restored(self, tmp_path):
        from shree.utils.bot_state import load_bot_state
        # Expired 10 minutes ago — within 4h window but in the past
        expired = datetime.now(timezone.utc) - timedelta(minutes=10)
        path = tmp_path / "bot_state.json"
        _write_state(path, {
            "consecutive_loss_count": 3,
            "extra_cooldown_until": expired.isoformat(),
            "last_trade_date": CT_TODAY,
            "written_at": datetime.now(timezone.utc).isoformat(),
        })
        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            count, cooldown, pnl = load_bot_state(path=path)
        assert count == 3       # loss count preserved
        assert cooldown is None  # expired cooldown discarded

    def test_stale_cooldown_older_than_4h_discarded(self, tmp_path):
        from shree.utils.bot_state import load_bot_state
        stale = datetime.now(timezone.utc) - timedelta(hours=5)
        path = tmp_path / "bot_state.json"
        _write_state(path, {
            "consecutive_loss_count": 1,
            "extra_cooldown_until": stale.isoformat(),
            "last_trade_date": CT_TODAY,
            "written_at": datetime.now(timezone.utc).isoformat(),
        })
        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            count, cooldown, pnl = load_bot_state(path=path)
        assert cooldown is None

    def test_day_rollover_resets_loss_count(self, tmp_path):
        from shree.utils.bot_state import load_bot_state
        path = tmp_path / "bot_state.json"
        _write_state(path, {
            "consecutive_loss_count": 2,
            "extra_cooldown_until": None,
            "realized_pnl_today": -120.0,
            "last_trade_date": "2026-03-12",   # yesterday
            "written_at": datetime.now(timezone.utc).isoformat(),
        })
        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            count, cooldown, pnl = load_bot_state(path=path)
        assert count == 0   # rolled over to 0
        assert cooldown is None
        assert pnl == 0.0   # daily P&L also reset on rollover

    def test_same_day_preserves_loss_count(self, tmp_path):
        from shree.utils.bot_state import load_bot_state
        path = tmp_path / "bot_state.json"
        _write_state(path, {
            "consecutive_loss_count": 2,
            "extra_cooldown_until": None,
            "realized_pnl_today": -55.0,
            "last_trade_date": CT_TODAY,   # same day
            "written_at": datetime.now(timezone.utc).isoformat(),
        })
        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            count, cooldown, pnl = load_bot_state(path=path)
        assert count == 2   # preserved
        assert pnl == -55.0  # preserved

    def test_cooldown_survives_restart_same_day(self, tmp_path):
        """Simulate: 3 losses fire cooldown, bot restarts mid-cooldown."""
        from shree.utils.bot_state import load_bot_state
        cooldown_end = datetime.now(timezone.utc) + timedelta(minutes=15)
        path = tmp_path / "bot_state.json"
        _write_state(path, {
            "consecutive_loss_count": 3,
            "extra_cooldown_until": cooldown_end.isoformat(),
            "last_trade_date": CT_TODAY,
            "written_at": datetime.now(timezone.utc).isoformat(),
        })
        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            count, cooldown, pnl = load_bot_state(path=path)
        assert count == 3
        assert cooldown is not None
        remaining = (cooldown - datetime.now(timezone.utc)).total_seconds() / 60
        assert 14 < remaining <= 15   # ~15 min remaining


# ── save_bot_state ────────────────────────────────────────────────────────────

class TestSaveBotState:

    def test_round_trip_no_cooldown(self, tmp_path):
        from shree.utils.bot_state import save_bot_state, load_bot_state
        path = tmp_path / "bot_state.json"
        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            save_bot_state(consecutive_loss_count=2, extra_cooldown_until=None, realized_pnl_today=-45.0, path=path)
            count, cooldown, pnl = load_bot_state(path=path)
        assert count == 2
        assert cooldown is None
        assert pnl == -45.0

    def test_round_trip_with_active_cooldown(self, tmp_path):
        from shree.utils.bot_state import save_bot_state, load_bot_state
        future = datetime.now(timezone.utc) + timedelta(minutes=25)
        path = tmp_path / "bot_state.json"
        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            save_bot_state(consecutive_loss_count=3, extra_cooldown_until=future, realized_pnl_today=-100.0, path=path)
            count, cooldown, pnl = load_bot_state(path=path)
        assert count == 3
        assert cooldown is not None
        assert abs((cooldown - future).total_seconds()) < 2   # within 2s rounding
        assert pnl == -100.0

    def test_save_creates_directory(self, tmp_path):
        from shree.utils.bot_state import save_bot_state
        nested = tmp_path / "deep" / "nested" / "bot_state.json"
        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            save_bot_state(0, None, path=nested)
        assert nested.exists()

    def test_save_is_valid_json(self, tmp_path):
        from shree.utils.bot_state import save_bot_state
        path = tmp_path / "bot_state.json"
        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            save_bot_state(1, None, realized_pnl_today=-30.0, path=path)
        data = json.loads(path.read_text())
        assert "consecutive_loss_count" in data
        assert "extra_cooldown_until" in data
        assert "realized_pnl_today" in data
        assert "last_trade_date" in data
        assert "written_at" in data
        assert data["realized_pnl_today"] == -30.0

    def test_reset_to_zero_overwrites_previous(self, tmp_path):
        from shree.utils.bot_state import save_bot_state, load_bot_state
        path = tmp_path / "bot_state.json"
        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            save_bot_state(3, None, realized_pnl_today=-100.0, path=path)
            save_bot_state(0, None, realized_pnl_today=-60.0, path=path)   # win reduces loss count but P&L stays
            count, cooldown, pnl = load_bot_state(path=path)
        assert count == 0
        assert cooldown is None
        assert pnl == -60.0


# ── Integration: simulated restart sequence ───────────────────────────────────

class TestRestartSimulation:

    def test_three_losses_survive_two_restarts(self, tmp_path):
        """
        Simulate the Mar 12 scenario:
          Loss 1 → save (count=1)
          Restart → load (count=1)
          Loss 2 → save (count=2)
          Restart → load (count=2)
          Loss 3 → save (count=3, cooldown set)
          Restart → load (count=3, cooldown active) → entries blocked
        """
        from shree.utils.bot_state import save_bot_state, load_bot_state
        path = tmp_path / "bot_state.json"

        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            # Loss 1
            save_bot_state(1, None, realized_pnl_today=-40.0, path=path)
            # Restart 1
            count, cd, pnl = load_bot_state(path=path)
            assert count == 1 and cd is None and pnl == -40.0

            # Loss 2
            save_bot_state(2, None, realized_pnl_today=-80.0, path=path)
            # Restart 2
            count, cd, pnl = load_bot_state(path=path)
            assert count == 2 and cd is None and pnl == -80.0

            # Loss 3 — cooldown fires
            cooldown_end = datetime.now(timezone.utc) + timedelta(minutes=30)
            save_bot_state(3, cooldown_end, realized_pnl_today=-120.0, path=path)
            # Restart 3
            count, cd, pnl = load_bot_state(path=path)
            assert count == 3
            assert cd is not None
            assert cd > datetime.now(timezone.utc)   # still blocking
            assert pnl == -120.0

    def test_win_resets_and_survives_restart(self, tmp_path):
        from shree.utils.bot_state import save_bot_state, load_bot_state
        path = tmp_path / "bot_state.json"

        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            save_bot_state(2, None, realized_pnl_today=-80.0, path=path)
            # Win → reset count, but P&L updated
            save_bot_state(0, None, realized_pnl_today=-40.0, path=path)
            # Restart
            count, cd, pnl = load_bot_state(path=path)
        assert count == 0
        assert cd is None
        assert pnl == -40.0  # P&L preserved (win reduced the net loss)


# ── Fix #6: Daily P&L persistence ────────────────────────────────────────────

class TestDailyPnlPersistence:
    """MAR 16 2026 Fix #6: Daily P&L must survive bot restarts."""

    def test_pnl_missing_from_old_state_defaults_zero(self, tmp_path):
        """Old bot_state.json without realized_pnl_today should default to 0.0."""
        from shree.utils.bot_state import load_bot_state
        path = tmp_path / "bot_state.json"
        _write_state(path, {
            "consecutive_loss_count": 1,
            "extra_cooldown_until": None,
            # no realized_pnl_today key — old format
            "last_trade_date": CT_TODAY,
            "written_at": datetime.now(timezone.utc).isoformat(),
        })
        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            count, cooldown, pnl = load_bot_state(path=path)
        assert count == 1
        assert pnl == 0.0  # default for missing field

    def test_pnl_accumulates_across_restarts(self, tmp_path):
        """Simulate: 3 losses across 2 restarts, P&L accumulates."""
        from shree.utils.bot_state import save_bot_state, load_bot_state
        path = tmp_path / "bot_state.json"

        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            # Loss 1: -$40
            save_bot_state(1, None, realized_pnl_today=-40.0, path=path)
            # Restart
            _, _, pnl = load_bot_state(path=path)
            assert pnl == -40.0

            # Loss 2: now -$90 total
            save_bot_state(2, None, realized_pnl_today=-90.0, path=path)
            # Restart
            _, _, pnl = load_bot_state(path=path)
            assert pnl == -90.0

            # Loss 3: now -$150 total
            save_bot_state(3, None, realized_pnl_today=-150.0, path=path)
            # Restart
            _, _, pnl = load_bot_state(path=path)
            assert pnl == -150.0

    def test_pnl_resets_on_day_rollover(self, tmp_path):
        """P&L from yesterday should reset to 0.0 on new day."""
        from shree.utils.bot_state import load_bot_state
        path = tmp_path / "bot_state.json"
        _write_state(path, {
            "consecutive_loss_count": 2,
            "extra_cooldown_until": None,
            "realized_pnl_today": -200.0,  # yesterday's losses
            "last_trade_date": "2026-03-12",  # different day
            "written_at": datetime.now(timezone.utc).isoformat(),
        })
        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            count, cooldown, pnl = load_bot_state(path=path)
        assert count == 0    # rolled over
        assert pnl == 0.0    # rolled over

    def test_pnl_survives_restart_same_day(self, tmp_path):
        """Daily P&L preserved across same-day restart."""
        from shree.utils.bot_state import save_bot_state, load_bot_state
        path = tmp_path / "bot_state.json"

        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            save_bot_state(1, None, realized_pnl_today=-55.0, path=path)
            count, _, pnl = load_bot_state(path=path)
        assert pnl == -55.0

    def test_pnl_default_zero_when_not_passed(self, tmp_path):
        """save_bot_state without realized_pnl_today uses default 0.0."""
        from shree.utils.bot_state import save_bot_state, load_bot_state
        path = tmp_path / "bot_state.json"

        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            save_bot_state(0, None, path=path)  # no realized_pnl_today
            _, _, pnl = load_bot_state(path=path)
        assert pnl == 0.0

    def test_pnl_rounds_to_two_decimals(self, tmp_path):
        """P&L should be rounded to 2 decimal places."""
        from shree.utils.bot_state import save_bot_state
        path = tmp_path / "bot_state.json"

        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            save_bot_state(0, None, realized_pnl_today=-55.123456, path=path)
        data = json.loads(path.read_text())
        assert data["realized_pnl_today"] == -55.12


# ── Fix #6b: Trade P&L accumulation into tracker.daily_pnl ──────────────────

class TestDailyPnlAccumulation:
    """MAR 17 2026 Fix #6b: _notify_position_closed must update tracker.daily_pnl
    so that bot_state persists the correct cumulative daily P&L, not a stale value.
    """

    def _make_manager_stub(self, initial_daily_pnl=0.0):
        """Create a minimal LTM-like object with the fields _notify_position_closed needs."""

        class FakeTracker:
            def __init__(self):
                self.daily_pnl = initial_daily_pnl
                self.total_realized_pnl = initial_daily_pnl

        class FakeSettings:
            class trading:
                ft_consecutive_loss_trigger = 3
                cooldown_on_consecutive_losses_minutes = 30

        _tracker = FakeTracker()

        class FakeManager:
            def __init__(self):
                self.tracker = _tracker
                self.settings = FakeSettings()
                self._consecutive_loss_count = 0
                self._extra_cooldown_until = None
                self.signal_processor = None  # skip signal_processor notify

            def _get_daily_pnl_for_persist(self):
                return float(getattr(self.tracker, "daily_pnl", 0.0))

        return FakeManager()

    def test_loss_updates_daily_pnl(self):
        """A trade loss should be accumulated into tracker.daily_pnl."""
        from shree.execution.live_trading_manager import LiveTradingManager
        mgr = self._make_manager_stub(initial_daily_pnl=-55.0)

        # Call _notify_position_closed with a -$40 loss
        with patch("shree.utils.bot_state.save_bot_state"):
            LiveTradingManager._notify_position_closed(mgr, "SL", "SHORT", -40.0)

        assert mgr.tracker.daily_pnl == pytest.approx(-95.0)
        assert mgr.tracker.total_realized_pnl == pytest.approx(-95.0)

    def test_win_updates_daily_pnl(self):
        """A trade win should be accumulated into tracker.daily_pnl."""
        from shree.execution.live_trading_manager import LiveTradingManager
        mgr = self._make_manager_stub(initial_daily_pnl=-55.0)

        with patch("shree.utils.bot_state.save_bot_state"):
            LiveTradingManager._notify_position_closed(mgr, "TP", "LONG", 40.0)

        assert mgr.tracker.daily_pnl == pytest.approx(-15.0)

    def test_zero_pnl_skipped(self):
        """pnl=0.0 (IB bug) should NOT touch tracker.daily_pnl."""
        from shree.execution.live_trading_manager import LiveTradingManager
        mgr = self._make_manager_stub(initial_daily_pnl=-55.0)

        LiveTradingManager._notify_position_closed(mgr, "SL", "LONG", 0.0)

        assert mgr.tracker.daily_pnl == pytest.approx(-55.0)  # unchanged

    def test_persisted_value_includes_latest_trade(self, tmp_path):
        """The value passed to save_bot_state should include the latest trade P&L."""
        from shree.execution.live_trading_manager import LiveTradingManager
        saved_calls = []

        def capture_save(**kwargs):
            saved_calls.append(kwargs.copy())

        mgr = self._make_manager_stub(initial_daily_pnl=-55.0)

        with patch("shree.utils.bot_state.save_bot_state", side_effect=capture_save):
            LiveTradingManager._notify_position_closed(mgr, "SL", "SHORT", -40.0)

        # save_bot_state is called inside _notify_position_closed with positional+keyword args
        assert len(saved_calls) >= 1
        last_call = saved_calls[-1]
        assert last_call["realized_pnl_today"] == pytest.approx(-95.0)

    def test_multiple_trades_accumulate(self):
        """Multiple trades in a day should all accumulate."""
        from shree.execution.live_trading_manager import LiveTradingManager
        mgr = self._make_manager_stub(initial_daily_pnl=0.0)

        with patch("shree.utils.bot_state.save_bot_state"):
            LiveTradingManager._notify_position_closed(mgr, "SL", "LONG", -40.0)
            LiveTradingManager._notify_position_closed(mgr, "TP", "LONG", 55.0)
            LiveTradingManager._notify_position_closed(mgr, "SL", "SHORT", -30.0)

        assert mgr.tracker.daily_pnl == pytest.approx(-15.0)  # -40 + 55 - 30


# ── Fix #6b: LivePerformanceTracker CST day rollover ─────────────────────────

class TestTrackerCstDayRollover:
    """MAR 17 2026: Tracker must use CST for day rollover, not UTC."""

    def test_tracker_init_uses_cst(self):
        """last_reset_date should be set from CST, not UTC."""
        from shree.monitoring.live_tracker import LivePerformanceTracker
        from shree.utils.timezone_utils import now_cst
        tracker = LivePerformanceTracker(initial_capital=5000.0)
        assert tracker.last_reset_date == now_cst().date()

    def test_day_rollover_uses_cst(self):
        """update_equity should roll over daily_pnl based on CST date, not UTC date."""
        from shree.monitoring.live_tracker import LivePerformanceTracker
        from datetime import date

        tracker = LivePerformanceTracker(initial_capital=5000.0)
        tracker.daily_pnl = -55.0
        tracker.last_reset_date = date(2026, 3, 16)  # yesterday in CST

        # Mock now_cst to return Mar 17 CST
        def _mock_cst():
            class _D:
                def date(self):
                    return date(2026, 3, 17)
            return _D()

        with patch("shree.monitoring.live_tracker.now_cst", _mock_cst):
            tracker.update_equity(6750.0, realized_pnl=0.0)

        assert tracker.daily_pnl == 0.0  # rolled over
        assert tracker.last_reset_date == date(2026, 3, 17)

    def test_no_rollover_same_cst_day(self):
        """daily_pnl should NOT reset if still the same CST day."""
        from shree.monitoring.live_tracker import LivePerformanceTracker
        from datetime import date

        tracker = LivePerformanceTracker(initial_capital=5000.0)
        tracker.daily_pnl = -55.0
        tracker.last_reset_date = date(2026, 3, 17)

        def _mock_cst():
            class _D:
                def date(self):
                    return date(2026, 3, 17)
            return _D()

        with patch("shree.monitoring.live_tracker.now_cst", _mock_cst):
            tracker.update_equity(6750.0, realized_pnl=0.0)

        assert tracker.daily_pnl == -55.0  # NOT rolled over
