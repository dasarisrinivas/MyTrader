"""
tests/test_bot_state_persistence.py

Tests for shree/utils/bot_state.py — Fix #14 persistence across restarts.

MAR 13 2026: The consecutive-loss counter must survive bot restarts so that
the 3-loss cooldown can accumulate across same-day restarts.
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
        count, cooldown = load_bot_state(path=path)
        assert count == 0
        assert cooldown is None

    def test_corrupt_json_returns_defaults(self, tmp_path):
        from shree.utils.bot_state import load_bot_state
        path = tmp_path / "bot_state.json"
        path.write_text("{ not valid json }")
        count, cooldown = load_bot_state(path=path)
        assert count == 0
        assert cooldown is None

    def test_valid_state_loaded(self, tmp_path):
        from shree.utils.bot_state import load_bot_state
        path = tmp_path / "bot_state.json"
        _write_state(path, {
            "consecutive_loss_count": 2,
            "extra_cooldown_until": None,
            "last_trade_date": CT_TODAY,
            "written_at": datetime.now(timezone.utc).isoformat(),
        })
        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            count, cooldown = load_bot_state(path=path)
        assert count == 2
        assert cooldown is None

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
            count, cooldown = load_bot_state(path=path)
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
            count, cooldown = load_bot_state(path=path)
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
            count, cooldown = load_bot_state(path=path)
        assert cooldown is None

    def test_day_rollover_resets_loss_count(self, tmp_path):
        from shree.utils.bot_state import load_bot_state
        path = tmp_path / "bot_state.json"
        _write_state(path, {
            "consecutive_loss_count": 2,
            "extra_cooldown_until": None,
            "last_trade_date": "2026-03-12",   # yesterday
            "written_at": datetime.now(timezone.utc).isoformat(),
        })
        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            count, cooldown = load_bot_state(path=path)
        assert count == 0   # rolled over to 0
        assert cooldown is None

    def test_same_day_preserves_loss_count(self, tmp_path):
        from shree.utils.bot_state import load_bot_state
        path = tmp_path / "bot_state.json"
        _write_state(path, {
            "consecutive_loss_count": 2,
            "extra_cooldown_until": None,
            "last_trade_date": CT_TODAY,   # same day
            "written_at": datetime.now(timezone.utc).isoformat(),
        })
        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            count, cooldown = load_bot_state(path=path)
        assert count == 2   # preserved

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
            count, cooldown = load_bot_state(path=path)
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
            save_bot_state(consecutive_loss_count=2, extra_cooldown_until=None, path=path)
            count, cooldown = load_bot_state(path=path)
        assert count == 2
        assert cooldown is None

    def test_round_trip_with_active_cooldown(self, tmp_path):
        from shree.utils.bot_state import save_bot_state, load_bot_state
        future = datetime.now(timezone.utc) + timedelta(minutes=25)
        path = tmp_path / "bot_state.json"
        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            save_bot_state(consecutive_loss_count=3, extra_cooldown_until=future, path=path)
            count, cooldown = load_bot_state(path=path)
        assert count == 3
        assert cooldown is not None
        assert abs((cooldown - future).total_seconds()) < 2   # within 2s rounding

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
            save_bot_state(1, None, path=path)
        data = json.loads(path.read_text())
        assert "consecutive_loss_count" in data
        assert "extra_cooldown_until" in data
        assert "last_trade_date" in data
        assert "written_at" in data

    def test_reset_to_zero_overwrites_previous(self, tmp_path):
        from shree.utils.bot_state import save_bot_state, load_bot_state
        path = tmp_path / "bot_state.json"
        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            save_bot_state(3, None, path=path)
            save_bot_state(0, None, path=path)   # reset after a win
            count, cooldown = load_bot_state(path=path)
        assert count == 0
        assert cooldown is None


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
            save_bot_state(1, None, path=path)
            # Restart 1
            count, cd = load_bot_state(path=path)
            assert count == 1 and cd is None

            # Loss 2
            save_bot_state(2, None, path=path)
            # Restart 2
            count, cd = load_bot_state(path=path)
            assert count == 2 and cd is None

            # Loss 3 — cooldown fires
            cooldown_end = datetime.now(timezone.utc) + timedelta(minutes=30)
            save_bot_state(3, cooldown_end, path=path)
            # Restart 3
            count, cd = load_bot_state(path=path)
            assert count == 3
            assert cd is not None
            assert cd > datetime.now(timezone.utc)   # still blocking

    def test_win_resets_and_survives_restart(self, tmp_path):
        from shree.utils.bot_state import save_bot_state, load_bot_state
        path = tmp_path / "bot_state.json"

        with patch("shree.utils.bot_state.now_cst", _mock_now_cst):
            save_bot_state(2, None, path=path)
            # Win → reset
            save_bot_state(0, None, path=path)
            # Restart
            count, cd = load_bot_state(path=path)
        assert count == 0
        assert cd is None
