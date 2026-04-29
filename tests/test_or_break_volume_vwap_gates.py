"""Tests for APR 29 2026 OR breakout gates: volume_ratio + VWAP confirmation.

Validates:
  - volume_ratio < threshold blocks Signal B and Signal E
  - close vs VWAP_daily blocks Signal B (close <= VWAP) and E (close >= VWAP)
  - both gates can be disabled via config
  - non-breakout signals are not affected (we don't pass volume/vwap to them)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_strategy(min_vol_ratio=0.7, vwap_filter=True):
    """Build a minimal EsFifteenMinStrategy with new gates configured."""
    from shree.strategies.es_fifteen_min import EsFifteenMinStrategy

    strat = object.__new__(EsFifteenMinStrategy)
    strat._or_break_long_count = 0
    strat._or_break_short_count = 0
    strat._or_break_max_per_day = 2

    strat._or_computed = True
    strat._or_high = 6950.00
    strat._or_low = 6940.00
    strat._prev_close = 0.0

    strat._adx_min = 20.0
    strat._adx_max = 999.0
    strat._fixed_sl_points = 6.0
    strat._fixed_tp_points = 8.0

    strat._or_break_sl_atr_mult = 0.75
    strat._or_break_sl_floor = 6.0
    strat._or_break_sl_ceiling = 12.0
    strat._or_break_rr_ratio = 1.33

    strat._or_break_short_rsi_min = 40.0
    strat._or_break_long_rsi_max = 60.0

    strat._or_break_max_chase_atr = 1.0

    # The new gates under test
    strat._or_break_min_vol_ratio = min_vol_ratio
    strat._or_break_vwap_filter_enabled = vwap_filter

    strat._save_counters = lambda: None
    return strat


# Baseline params that satisfy all *other* conditions for a long OR break.
_LONG_PARAMS = dict(
    close=6951.00,    # above OR_HIGH (6950)
    high=6952.00,
    ema9=6948.00,
    ema21=6940.00,
    atr=10.0,
    adx=28.0,
    macd_hist=1.5,
    rsi=55.0,
)
_SHORT_PARAMS = dict(
    close=6939.00,    # below OR_LOW (6940)
    low=6938.00,
    ema9=6940.00,
    ema21=6948.00,
    atr=10.0,
    adx=28.0,
    macd_hist=-1.5,
    rsi=45.0,
)


class TestVolumeRatioGate:
    def test_long_blocked_when_vol_ratio_below_threshold(self):
        strat = _make_strategy(min_vol_ratio=0.7, vwap_filter=False)
        strat._prev_close = 6949.00
        result = strat._check_or_breakout(
            **_LONG_PARAMS, volume_ratio=0.5, vwap_daily=0.0
        )
        assert result is None, "Should block long OR break with vol_ratio=0.5 < 0.7"
        assert strat._or_break_long_count == 0  # counter not incremented

    def test_long_fires_when_vol_ratio_at_threshold(self):
        strat = _make_strategy(min_vol_ratio=0.7, vwap_filter=False)
        strat._prev_close = 6949.00
        result = strat._check_or_breakout(
            **_LONG_PARAMS, volume_ratio=0.7, vwap_daily=0.0
        )
        assert result is not None and result[0] == "BUY"

    def test_short_blocked_when_vol_ratio_below_threshold(self):
        strat = _make_strategy(min_vol_ratio=0.7, vwap_filter=False)
        strat._prev_close = 6941.00
        result = strat._check_or_breakdown(
            **_SHORT_PARAMS, volume_ratio=0.3, vwap_daily=0.0
        )
        assert result is None
        assert strat._or_break_short_count == 0

    def test_short_fires_when_vol_ratio_above_threshold(self):
        strat = _make_strategy(min_vol_ratio=0.7, vwap_filter=False)
        strat._prev_close = 6941.00
        result = strat._check_or_breakdown(
            **_SHORT_PARAMS, volume_ratio=1.2, vwap_daily=0.0
        )
        assert result is not None and result[0] == "SELL"

    def test_zero_threshold_disables_gate(self):
        """min_vol_ratio=0 means 'always pass' — even tiny vol gets through."""
        strat = _make_strategy(min_vol_ratio=0.0, vwap_filter=False)
        strat._prev_close = 6949.00
        result = strat._check_or_breakout(
            **_LONG_PARAMS, volume_ratio=0.01, vwap_daily=0.0
        )
        assert result is not None, "vol_ratio gate should be disabled at threshold=0"


class TestVwapGate:
    def test_long_blocked_when_close_below_vwap(self):
        strat = _make_strategy(min_vol_ratio=0.0, vwap_filter=True)
        strat._prev_close = 6949.00
        # close=6951, vwap=6960 → close below VWAP, should block long break
        result = strat._check_or_breakout(
            **_LONG_PARAMS, volume_ratio=1.0, vwap_daily=6960.0
        )
        assert result is None, "Long break must be blocked when close <= VWAP"

    def test_long_fires_when_close_above_vwap(self):
        strat = _make_strategy(min_vol_ratio=0.0, vwap_filter=True)
        strat._prev_close = 6949.00
        result = strat._check_or_breakout(
            **_LONG_PARAMS, volume_ratio=1.0, vwap_daily=6940.0
        )
        assert result is not None and result[0] == "BUY"

    def test_short_blocked_when_close_above_vwap(self):
        strat = _make_strategy(min_vol_ratio=0.0, vwap_filter=True)
        strat._prev_close = 6941.00
        # close=6939, vwap=6930 → close above VWAP, should block short break
        result = strat._check_or_breakdown(
            **_SHORT_PARAMS, volume_ratio=1.0, vwap_daily=6930.0
        )
        assert result is None

    def test_short_fires_when_close_below_vwap(self):
        strat = _make_strategy(min_vol_ratio=0.0, vwap_filter=True)
        strat._prev_close = 6941.00
        result = strat._check_or_breakdown(
            **_SHORT_PARAMS, volume_ratio=1.0, vwap_daily=6960.0
        )
        assert result is not None and result[0] == "SELL"

    def test_filter_disabled_lets_close_below_vwap_long_through(self):
        strat = _make_strategy(min_vol_ratio=0.0, vwap_filter=False)
        strat._prev_close = 6949.00
        result = strat._check_or_breakout(
            **_LONG_PARAMS, volume_ratio=1.0, vwap_daily=6960.0
        )
        assert result is not None, "VWAP gate disabled — should fire even with close below VWAP"

    def test_zero_vwap_treated_as_missing(self):
        """vwap_daily=0 indicates missing data — must not block."""
        strat = _make_strategy(min_vol_ratio=0.0, vwap_filter=True)
        strat._prev_close = 6949.00
        result = strat._check_or_breakout(
            **_LONG_PARAMS, volume_ratio=1.0, vwap_daily=0.0
        )
        assert result is not None, "Missing VWAP (=0) should not gate the signal"


class TestDefaultsArePassThrough:
    """Backward compat: callers that don't pass volume/vwap should still work."""

    def test_long_fires_with_no_vol_or_vwap_passed(self):
        strat = _make_strategy(min_vol_ratio=0.0, vwap_filter=False)
        strat._prev_close = 6949.00
        result = strat._check_or_breakout(**_LONG_PARAMS)
        assert result is not None
