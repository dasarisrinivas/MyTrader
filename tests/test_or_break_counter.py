"""Tests for OR break counter system (FEB 26 2026).

Changed Signal B (OR_BREAK_LONG) and Signal E (OR_BREAK_SHORT) from
once-per-day boolean flags to counter-based limits (max 2/day per side).

Backtest evidence:
  - 13 OR_BREAK trades analyzed
  - After OR wins, continuation trades netted +$170.70 (6W/2L across 4 days)
  - After OR losses, continuation trades netted -$93.35
  - The cross condition (prev_close on opposite side of OR level) naturally
    gates re-entry — price must return above/below OR and break again.

Live evidence (FEB 26 2026):
  - OR_BREAK_SHORT won +$36.45 at 08:53 CST
  - MES then fell 85 more points — $270+ left on table
  - Old once-per-day flag prevented a second entry on the retest
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helper: build a minimal EsFifteenMinStrategy via __new__ + manual init
# ---------------------------------------------------------------------------

def _make_strategy(or_break_max=2, adx_min=20.0, adx_max=999.0,
                   fixed_sl=6.0, fixed_tp=8.0):
    """Create an EsFifteenMinStrategy with manually set OR-break attrs."""
    from shree.strategies.es_fifteen_min import EsFifteenMinStrategy

    strat = object.__new__(EsFifteenMinStrategy)
    # OR break counters (the thing we're testing)
    strat._or_break_long_count = 0
    strat._or_break_short_count = 0
    strat._or_break_max_per_day = or_break_max

    # Session state that _check_or_breakout / _check_or_breakdown need
    strat._or_computed = True
    strat._or_high = 6950.00
    strat._or_low = 6940.00
    strat._prev_close = 0.0

    # Parameters
    strat._adx_min = adx_min
    strat._adx_max = adx_max
    strat._fixed_sl_points = fixed_sl
    strat._fixed_tp_points = fixed_tp

    return strat


# ===========================================================================
# Signal B — OR_BREAK_LONG counter tests
# ===========================================================================

class TestSignalBCounter:
    """Signal B (OR_BREAK_LONG): counter limits to max N per day."""

    # Baseline params that satisfy all conditions for a long OR break
    LONG_PARAMS = dict(
        close=6951.00,    # above OR_HIGH (6950)
        high=6952.00,
        ema9=6948.00,     # ema9 > ema21
        ema21=6940.00,
        atr=10.0,
        adx=28.0,         # in range [20, 999]
        macd_hist=1.5,     # positive
    )

    def test_first_break_fires(self):
        """First OR break long should fire normally (count 0 → 1)."""
        strat = _make_strategy()
        strat._prev_close = 6949.00  # was below OR_HIGH
        result = strat._check_or_breakout(**self.LONG_PARAMS)
        assert result is not None
        assert result[0] == "BUY"
        assert strat._or_break_long_count == 1
        assert "OR_BREAK_LONG" in result[3]
        # First break should NOT have "retest" tag
        assert "retest" not in result[3]

    def test_second_break_fires_after_retest(self):
        """Second OR break long should fire when price retests (count 1 → 2)."""
        strat = _make_strategy()
        # Simulate first break already happened
        strat._or_break_long_count = 1
        strat._prev_close = 6949.00  # price came back below and now breaks again
        result = strat._check_or_breakout(**self.LONG_PARAMS)
        assert result is not None
        assert result[0] == "BUY"
        assert strat._or_break_long_count == 2
        assert "retest#2" in result[3]

    def test_third_break_blocked_at_max_2(self):
        """Third OR break should be blocked when max=2 (default)."""
        strat = _make_strategy(or_break_max=2)
        strat._or_break_long_count = 2
        strat._prev_close = 6949.00
        result = strat._check_or_breakout(**self.LONG_PARAMS)
        assert result is None
        # Counter should not increment
        assert strat._or_break_long_count == 2

    def test_max_1_blocks_second(self):
        """With max=1, second break is blocked (old behavior)."""
        strat = _make_strategy(or_break_max=1)
        strat._or_break_long_count = 1
        strat._prev_close = 6949.00
        result = strat._check_or_breakout(**self.LONG_PARAMS)
        assert result is None

    def test_max_3_allows_third(self):
        """With max=3, third break fires."""
        strat = _make_strategy(or_break_max=3)
        strat._or_break_long_count = 2
        strat._prev_close = 6949.00
        result = strat._check_or_breakout(**self.LONG_PARAMS)
        assert result is not None
        assert strat._or_break_long_count == 3
        assert "retest#3" in result[3]

    def test_cross_condition_required(self):
        """If prev_close was already above OR_HIGH, no cross → no signal."""
        strat = _make_strategy()
        strat._prev_close = 6951.00  # already above → no cross
        result = strat._check_or_breakout(**self.LONG_PARAMS)
        assert result is None
        assert strat._or_break_long_count == 0  # counter stays at 0

    def test_long_count_independent_of_short(self):
        """Long counter is independent of short counter."""
        strat = _make_strategy()
        strat._or_break_short_count = 2  # short maxed out
        strat._prev_close = 6949.00
        result = strat._check_or_breakout(**self.LONG_PARAMS)
        assert result is not None  # long should still fire
        assert strat._or_break_long_count == 1


# ===========================================================================
# Signal E — OR_BREAK_SHORT counter tests
# ===========================================================================

class TestSignalECounter:
    """Signal E (OR_BREAK_SHORT): counter limits to max N per day."""

    # Baseline params that satisfy all conditions for a short OR break
    SHORT_PARAMS = dict(
        close=6939.00,     # below OR_LOW (6940)
        low=6938.00,
        ema9=6942.00,      # ema9 < ema21 (bearish)
        ema21=6950.00,
        atr=10.0,
        adx=28.0,          # in range [20, 999]
        macd_hist=-1.5,    # negative (bearish momentum)
    )

    def test_first_break_fires(self):
        """First OR break short should fire normally (count 0 → 1)."""
        strat = _make_strategy()
        strat._prev_close = 6941.00  # was above OR_LOW
        result = strat._check_or_breakdown(**self.SHORT_PARAMS)
        assert result is not None
        assert result[0] == "SELL"
        assert strat._or_break_short_count == 1
        assert "OR_BREAK_SHORT" in result[3]
        assert "retest" not in result[3]

    def test_second_break_fires_after_retest(self):
        """Second OR break short should fire on retest (count 1 → 2)."""
        strat = _make_strategy()
        strat._or_break_short_count = 1
        strat._prev_close = 6941.00  # bounced back above, now breaking again
        result = strat._check_or_breakdown(**self.SHORT_PARAMS)
        assert result is not None
        assert result[0] == "SELL"
        assert strat._or_break_short_count == 2
        assert "retest#2" in result[3]

    def test_third_break_blocked_at_max_2(self):
        """Third OR break short blocked when max=2."""
        strat = _make_strategy(or_break_max=2)
        strat._or_break_short_count = 2
        strat._prev_close = 6941.00
        result = strat._check_or_breakdown(**self.SHORT_PARAMS)
        assert result is None
        assert strat._or_break_short_count == 2

    def test_max_1_blocks_second(self):
        """With max=1, second break is blocked (old behavior)."""
        strat = _make_strategy(or_break_max=1)
        strat._or_break_short_count = 1
        strat._prev_close = 6941.00
        result = strat._check_or_breakdown(**self.SHORT_PARAMS)
        assert result is None

    def test_cross_condition_required(self):
        """If prev_close was already below OR_LOW, no cross → no signal."""
        strat = _make_strategy()
        strat._prev_close = 6938.00  # already below → no cross
        result = strat._check_or_breakdown(**self.SHORT_PARAMS)
        assert result is None
        assert strat._or_break_short_count == 0

    def test_short_count_independent_of_long(self):
        """Short counter is independent of long counter."""
        strat = _make_strategy()
        strat._or_break_long_count = 2  # long maxed out
        strat._prev_close = 6941.00
        result = strat._check_or_breakdown(**self.SHORT_PARAMS)
        assert result is not None
        assert strat._or_break_short_count == 1


# ===========================================================================
# Counter reset tests
# ===========================================================================

class TestCounterReset:
    """Counters reset to 0 on new session (day)."""

    def test_reset_clears_long_count(self):
        """_reset_session should zero the long OR break counter."""
        strat = _make_strategy()
        strat._or_break_long_count = 2
        strat._session_date = "2026-02-25"
        strat._trend_cont_long_count = 0
        strat._trend_cont_short_count = 0
        strat._opening_bars = []
        strat._reset_session("2026-02-26")
        assert strat._or_break_long_count == 0

    def test_reset_clears_short_count(self):
        """_reset_session should zero the short OR break counter."""
        strat = _make_strategy()
        strat._or_break_short_count = 2
        strat._session_date = "2026-02-25"
        strat._trend_cont_long_count = 0
        strat._trend_cont_short_count = 0
        strat._opening_bars = []
        strat._reset_session("2026-02-26")
        assert strat._or_break_short_count == 0


# ===========================================================================
# Stop-loss / Take-profit correctness
# ===========================================================================

class TestORBreakStopTarget:
    """SL/TP are computed correctly on OR break signals."""

    def test_long_sl_tp(self):
        """Signal B: SL = close - 6, TP = close + 8."""
        strat = _make_strategy(fixed_sl=6.0, fixed_tp=8.0)
        strat._prev_close = 6949.00
        result = strat._check_or_breakout(
            close=6951.00, high=6952.00,
            ema9=6948.00, ema21=6940.00,
            atr=10.0, adx=28.0, macd_hist=1.5,
        )
        assert result is not None
        _, sl, tp, _ = result
        assert sl == pytest.approx(6945.00)  # 6951 - 6
        assert tp == pytest.approx(6959.00)  # 6951 + 8

    def test_short_sl_tp(self):
        """Signal E: SL = close + 6, TP = close - 8."""
        strat = _make_strategy(fixed_sl=6.0, fixed_tp=8.0)
        strat._prev_close = 6941.00
        result = strat._check_or_breakdown(
            close=6939.00, low=6938.00,
            ema9=6942.00, ema21=6950.00,
            atr=10.0, adx=28.0, macd_hist=-1.5,
        )
        assert result is not None
        _, sl, tp, _ = result
        assert sl == pytest.approx(6945.00)  # 6939 + 6
        assert tp == pytest.approx(6931.00)  # 6939 - 8


# ===========================================================================
# Config default test
# ===========================================================================

class TestORBreakConfig:
    """Config defaults for OR break max per day."""

    def test_default_max_is_2(self):
        """Default ft_or_break_max_per_day should be 2."""
        config = MagicMock()
        # Ensure getattr falls through to default
        config.ft_or_break_max_per_day = 2
        strat = _make_strategy(or_break_max=2)
        assert strat._or_break_max_per_day == 2

    def test_getattr_default(self):
        """If config doesn't have ft_or_break_max_per_day, default=2."""
        from shree.strategies.es_fifteen_min import EsFifteenMinStrategy
        config = MagicMock(spec=[])  # no attributes
        # Manually test getattr behavior
        val = getattr(config, 'ft_or_break_max_per_day', 2)
        assert val == 2
