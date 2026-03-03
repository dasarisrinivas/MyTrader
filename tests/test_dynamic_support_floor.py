"""Tests for DynamicSupportFloor — auto-computed from market data."""
import pytest
from datetime import datetime

from shree.risk.dynamic_support import DynamicSupportFloor, DynamicSupportFloorConfig


# ─── Helpers ──────────────────────────────────────────────────────────

def _make_historical(pdl=6870.75, weekly_low=6828.50, pdh=6961.25, weekly_high=6988.0):
    """Build a _historical_context dict like LiveTradingManager creates."""
    return {
        "previous_day": {"high": pdh, "low": pdl, "close": 6920.0, "open": 6900.0},
        "weekly": {"high": weekly_high, "low": weekly_low},
    }


# ─── Construction ────────────────────────────────────────────────────

class TestDynamicSupportFloorInit:
    def test_default_config(self):
        floor = DynamicSupportFloor()
        assert floor.config.buffer_points == 5.0
        assert floor.config.min_sources == 1
        assert floor.get_floor() is None  # No data yet

    def test_custom_config(self):
        cfg = DynamicSupportFloorConfig(buffer_points=10.0, min_sources=2)
        floor = DynamicSupportFloor(cfg)
        assert floor.config.buffer_points == 10.0
        assert floor.config.min_sources == 2


# ─── Historical context updates ──────────────────────────────────────

class TestHistoricalContextUpdate:
    def test_pdl_and_weekly_low(self):
        """Floor should be min(PDL, WL) - buffer."""
        floor = DynamicSupportFloor()
        ctx = _make_historical(pdl=6870.75, weekly_low=6828.50)
        floor.update_from_historical_context(ctx)

        # Weekly low (6828.50) is lowest → floor = 6828.50 - 5.0 = 6823.50
        assert floor.get_floor() == 6823.50

    def test_pdl_is_lowest(self):
        """When PDL < weekly low, PDL drives the floor."""
        floor = DynamicSupportFloor()
        ctx = _make_historical(pdl=6800.0, weekly_low=6828.50)
        floor.update_from_historical_context(ctx)

        assert floor.get_floor() == 6795.0  # 6800 - 5

    def test_custom_buffer(self):
        """Buffer should be subtracted from the lowest level."""
        cfg = DynamicSupportFloorConfig(buffer_points=10.0)
        floor = DynamicSupportFloor(cfg)
        ctx = _make_historical(pdl=6870.75, weekly_low=6828.50)
        floor.update_from_historical_context(ctx)

        assert floor.get_floor() == 6818.50  # 6828.50 - 10.0

    def test_empty_context(self):
        """Empty context should not crash or set a floor."""
        floor = DynamicSupportFloor()
        floor.update_from_historical_context({})
        assert floor.get_floor() is None

    def test_none_context(self):
        floor = DynamicSupportFloor()
        floor.update_from_historical_context(None)
        assert floor.get_floor() is None

    def test_missing_previous_day(self):
        floor = DynamicSupportFloor()
        floor.update_from_historical_context({"weekly": {"low": 6828.50}})
        assert floor.get_floor() == 6823.50  # Only WL available

    def test_missing_weekly(self):
        floor = DynamicSupportFloor()
        floor.update_from_historical_context({"previous_day": {"low": 6870.75}})
        assert floor.get_floor() == 6865.75  # Only PDL available


# ─── OR level updates ────────────────────────────────────────────────

class TestORLevelUpdate:
    def test_or_low_becomes_lowest(self):
        """When OR low is below PDL and WL, it drives the floor."""
        floor = DynamicSupportFloor()
        ctx = _make_historical(pdl=6870.75, weekly_low=6828.50)
        floor.update_from_historical_context(ctx)

        # OR low is below weekly low
        floor.update_or_levels(or_high=6860.0, or_low=6810.0)
        assert floor.get_floor() == 6805.0  # 6810 - 5

    def test_or_low_not_lowest(self):
        """When OR low is above weekly low, weekly low still drives."""
        floor = DynamicSupportFloor()
        ctx = _make_historical(pdl=6870.75, weekly_low=6828.50)
        floor.update_from_historical_context(ctx)

        floor.update_or_levels(or_high=6900.0, or_low=6850.0)
        # Weekly low (6828.50) is still lowest
        assert floor.get_floor() == 6823.50

    def test_or_low_zero_ignored(self):
        """OR low of 0 should be ignored."""
        floor = DynamicSupportFloor()
        ctx = _make_historical(pdl=6870.75, weekly_low=6828.50)
        floor.update_from_historical_context(ctx)

        floor.update_or_levels(or_high=6900.0, or_low=0.0)
        assert floor.get_floor() == 6823.50  # Unchanged

    def test_or_only(self):
        """If only OR low is available, it should still work."""
        floor = DynamicSupportFloor()
        floor.update_or_levels(or_high=6860.0, or_low=6840.0)
        assert floor.get_floor() == 6835.0  # 6840 - 5


# ─── Source toggling ──────────────────────────────────────────────────

class TestSourceToggling:
    def test_disable_pdl(self):
        """Disabling PDL should exclude it from computation."""
        cfg = DynamicSupportFloorConfig(use_pdl=False)
        floor = DynamicSupportFloor(cfg)
        ctx = _make_historical(pdl=6800.0, weekly_low=6828.50)
        floor.update_from_historical_context(ctx)

        # PDL (6800) is lowest but disabled → floor uses WL
        assert floor.get_floor() == 6823.50  # 6828.50 - 5

    def test_disable_weekly_low(self):
        cfg = DynamicSupportFloorConfig(use_weekly_low=False)
        floor = DynamicSupportFloor(cfg)
        ctx = _make_historical(pdl=6870.75, weekly_low=6828.50)
        floor.update_from_historical_context(ctx)

        # WL disabled → floor uses PDL
        assert floor.get_floor() == 6865.75  # 6870.75 - 5

    def test_disable_or_low(self):
        cfg = DynamicSupportFloorConfig(use_or_low=False)
        floor = DynamicSupportFloor(cfg)
        ctx = _make_historical(pdl=6870.75, weekly_low=6828.50)
        floor.update_from_historical_context(ctx)
        floor.update_or_levels(or_high=6860.0, or_low=6800.0)

        # OR_L (6800) is lowest but disabled → floor uses WL
        assert floor.get_floor() == 6823.50

    def test_all_sources_disabled(self):
        cfg = DynamicSupportFloorConfig(use_pdl=False, use_weekly_low=False, use_or_low=False)
        floor = DynamicSupportFloor(cfg)
        ctx = _make_historical(pdl=6870.75, weekly_low=6828.50)
        floor.update_from_historical_context(ctx)
        assert floor.get_floor() is None


# ─── min_sources ──────────────────────────────────────────────────────

class TestMinSources:
    def test_min_sources_2_with_1_available(self):
        """If min_sources=2 but only 1 level, floor should be None."""
        cfg = DynamicSupportFloorConfig(min_sources=2)
        floor = DynamicSupportFloor(cfg)
        floor.update_or_levels(or_high=6860.0, or_low=6840.0)
        assert floor.get_floor() is None

    def test_min_sources_2_with_2_available(self):
        cfg = DynamicSupportFloorConfig(min_sources=2)
        floor = DynamicSupportFloor(cfg)
        ctx = _make_historical(pdl=6870.75, weekly_low=6828.50)
        floor.update_from_historical_context(ctx)
        # PDL + WL = 2 sources
        assert floor.get_floor() == 6823.50

    def test_min_sources_3_with_2_available(self):
        cfg = DynamicSupportFloorConfig(min_sources=3)
        floor = DynamicSupportFloor(cfg)
        ctx = _make_historical(pdl=6870.75, weekly_low=6828.50)
        floor.update_from_historical_context(ctx)
        assert floor.get_floor() is None  # Only 2 of 3 needed

    def test_min_sources_3_with_3_available(self):
        cfg = DynamicSupportFloorConfig(min_sources=3)
        floor = DynamicSupportFloor(cfg)
        ctx = _make_historical(pdl=6870.75, weekly_low=6828.50)
        floor.update_from_historical_context(ctx)
        floor.update_or_levels(or_high=6880.0, or_low=6850.0)
        assert floor.get_floor() == 6823.50  # WL is still lowest


# ─── Reset ────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_floor(self):
        floor = DynamicSupportFloor()
        ctx = _make_historical(pdl=6870.75, weekly_low=6828.50)
        floor.update_from_historical_context(ctx)
        assert floor.get_floor() is not None

        floor.reset()
        assert floor.get_floor() is None

    def test_reset_then_update(self):
        """After reset, new data should work normally."""
        floor = DynamicSupportFloor()
        ctx = _make_historical(pdl=6870.75, weekly_low=6828.50)
        floor.update_from_historical_context(ctx)
        floor.reset()

        new_ctx = _make_historical(pdl=6750.0, weekly_low=6700.0)
        floor.update_from_historical_context(new_ctx)
        assert floor.get_floor() == 6695.0  # 6700 - 5


# ─── Direct setters ──────────────────────────────────────────────────

class TestDirectSetters:
    def test_update_pdl_directly(self):
        floor = DynamicSupportFloor()
        floor.update_pdl(6800.0)
        assert floor.get_floor() == 6795.0

    def test_update_weekly_low_directly(self):
        floor = DynamicSupportFloor()
        floor.update_weekly_low(6750.0)
        assert floor.get_floor() == 6745.0

    def test_zero_pdl_ignored(self):
        floor = DynamicSupportFloor()
        floor.update_pdl(0.0)
        assert floor.get_floor() is None

    def test_negative_pdl_ignored(self):
        floor = DynamicSupportFloor()
        floor.update_pdl(-100.0)
        assert floor.get_floor() is None


# ─── Diagnostics ──────────────────────────────────────────────────────

class TestDiagnostics:
    def test_diagnostics_no_data(self):
        floor = DynamicSupportFloor()
        diag = floor.get_diagnostics()
        assert diag["floor"] is None
        assert diag["pdl"] == 0.0
        assert diag["weekly_low"] == 0.0
        assert diag["or_low"] == 0.0
        assert diag["last_update"] is None

    def test_diagnostics_with_data(self):
        floor = DynamicSupportFloor()
        ctx = _make_historical(pdl=6870.75, weekly_low=6828.50)
        floor.update_from_historical_context(ctx)
        floor.update_or_levels(or_high=6880.0, or_low=6850.0)

        diag = floor.get_diagnostics()
        assert diag["floor"] == 6823.50
        assert diag["pdl"] == 6870.75
        assert diag["weekly_low"] == 6828.50
        assert diag["or_low"] == 6850.0
        assert diag["buffer_points"] == 5.0
        assert diag["last_update"] is not None


# ─── Floor update behavior ───────────────────────────────────────────

class TestFloorUpdateBehavior:
    def test_floor_updates_when_new_data_lower(self):
        """Floor should update when new data provides a lower level."""
        floor = DynamicSupportFloor()
        ctx = _make_historical(pdl=6870.75, weekly_low=6828.50)
        floor.update_from_historical_context(ctx)
        assert floor.get_floor() == 6823.50

        # OR low comes in lower
        floor.update_or_levels(or_high=6860.0, or_low=6810.0)
        assert floor.get_floor() == 6805.0  # Now driven by OR low

    def test_floor_stable_when_new_data_higher(self):
        """Floor should not move up just because OR low is higher."""
        floor = DynamicSupportFloor()
        ctx = _make_historical(pdl=6870.75, weekly_low=6828.50)
        floor.update_from_historical_context(ctx)
        assert floor.get_floor() == 6823.50

        # OR low is above weekly low — floor stays
        floor.update_or_levels(or_high=6900.0, or_low=6860.0)
        assert floor.get_floor() == 6823.50  # Still driven by WL

    def test_today_scenario_mar2(self):
        """Real scenario: Feb 26 PDL=6870.75, WL=6828.50, price=6821.5.
        
        Old hardcoded 6860 was 39pts above current price — useless.
        Dynamic: 6828.50 - 5 = 6823.50, which is ~2 pts above price.
        This correctly acts as a tight warning level without being stale.
        """
        floor = DynamicSupportFloor()
        ctx = _make_historical(pdl=6870.75, weekly_low=6828.50)
        floor.update_from_historical_context(ctx)

        floor_level = floor.get_floor()
        current_price = 6821.5

        # Floor should be close to (but slightly above) current price
        assert floor_level == 6823.50
        assert floor_level > current_price  # Correctly signals danger zone
        assert floor_level - current_price < 5.0  # Very close — appropriate
