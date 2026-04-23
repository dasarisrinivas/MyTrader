"""Unit tests for StructureThrottle.

Covers:
- First entry in fresh leg → allowed
- Re-entry within zone-lock → blocked
- Re-entry after regime flip → allowed (leg flushed)
- Per-leg cap enforced
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shree.spy_options.rules_v2.config import ThrottleConfig  # noqa: E402
from shree.spy_options.rules_v2.regime import (  # noqa: E402
    RANGE_BOUND,
    TRANSITION,
    TREND_DOWN,
    TREND_UP,
    RegimeV2Context,
)
from shree.spy_options.rules_v2.throttle import StructureThrottle  # noqa: E402


def _ctx(regime: str, vwap: float = 710.0) -> RegimeV2Context:
    return RegimeV2Context(
        regime=regime, vwap=vwap, vwap_slope=0.0, atr_ratio=1.0,
        pivots_recent=0, has_hhhl=False, has_lhll=False,
        vwap_crosses_30m=0, spy_vs_vwap=0.0, timestamp=datetime.utcnow(),
    )


def test_first_entry_in_fresh_leg_allowed():
    t = StructureThrottle()
    r = t.check_allow("P", 706.00, _ctx(TREND_DOWN))
    assert r.allowed


def test_zone_lock_blocks_reentry_same_leg():
    t = StructureThrottle(ThrottleConfig(zone_lock_pct=0.0010))
    ctx = _ctx(TREND_DOWN)
    r1 = t.check_allow("P", 706.00, ctx)
    assert r1.allowed
    t.record_entry("P", 706.00)
    # 706.50 is within 0.10% of 706.00 — should block
    r2 = t.check_allow("P", 706.50, ctx)
    assert not r2.allowed
    assert "zone-lock" in r2.reason


def test_zone_lock_allows_outside_band():
    t = StructureThrottle(ThrottleConfig(zone_lock_pct=0.0010))
    ctx = _ctx(TREND_DOWN)
    t.check_allow("P", 706.00, ctx)
    t.record_entry("P", 706.00)
    # 704.50 is 0.21% below — outside band
    r = t.check_allow("P", 704.50, ctx)
    assert r.allowed


def test_regime_flip_to_transition_flushes_leg():
    t = StructureThrottle()
    ctx_down = _ctx(TREND_DOWN)
    t.check_allow("P", 706.00, ctx_down)
    t.record_entry("P", 706.00)
    # Flip to TRANSITION
    ctx_trans = _ctx(TRANSITION)
    r = t.check_allow("P", 706.05, ctx_trans)
    # After flush, zone state is cleared — re-entry OK
    assert r.allowed


def test_regime_flip_to_opposite_trend_flushes_leg():
    t = StructureThrottle()
    ctx_down = _ctx(TREND_DOWN)
    t.check_allow("P", 706.00, ctx_down)
    t.record_entry("P", 706.00)
    ctx_up = _ctx(TREND_UP)
    r = t.check_allow("C", 706.05, ctx_up)
    assert r.allowed


def test_leg_cap_enforced():
    cfg = ThrottleConfig(max_entries_per_leg=2, zone_lock_pct=0.0)
    t = StructureThrottle(cfg)
    ctx = _ctx(TREND_DOWN)
    # Use identical prices so _update_pivot doesn't advance the leg counter
    # (any new lower low in TREND_DOWN creates a fresh leg, which would
    # reset the per-leg entry count and invalidate this test).
    t.check_allow("P", 706.00, ctx); t.record_entry("P", 706.00)
    t.check_allow("P", 706.00, ctx); t.record_entry("P", 706.00)
    r = t.check_allow("P", 706.00, ctx)
    assert not r.allowed
    assert "leg cap" in r.reason
