"""Unit tests for ExpectedMoveGate.

Covers:
- Skip-on-missing-data (leg_iv=None, leg_mid=None, BOTH direction).
- Reject case: leg debit too rich vs IV-implied EM.
- Pass case: EM clears the leg debit + cushion.
- Engine integration: gate slots between entry_gate and throttle when
  feature flag is on, and skips entirely when flag is off.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shree.spy_options.rules_v2.config import (  # noqa: E402
    EntryGateConfig,
    ExpectedMoveGateConfig,
    OrbGateConfig,
    PcRatioAlignmentConfig,
    RulesV2Config,
    ThrottleConfig,
)
from shree.spy_options.rules_v2.engine import EngineInputs, RulesV2Engine  # noqa: E402
from shree.spy_options.rules_v2.expected_move_gate import (  # noqa: E402
    ExpectedMoveGate,
)
from shree.spy_options.rules_v2.regime import (  # noqa: E402
    RANGE_BOUND,
    TREND_UP,
    RegimeV2Context,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _bars(start=710.0, delta=0.10, n=30):
    """Mild uptrend bars sufficient to satisfy entry_gate basics."""
    out = []
    t = datetime(2026, 4, 21, 12, 0)
    price = start
    for _ in range(n):
        o = price
        c = price + delta
        out.append({
            "date": t, "open": o,
            "high": max(o, c) + 0.05, "low": min(o, c) - 0.05,
            "close": c, "volume": 100000,
        })
        t += timedelta(minutes=5)
        price = c
    return out


def _ctx(regime: str, vwap: float) -> RegimeV2Context:
    return RegimeV2Context(
        regime=regime, vwap=vwap, vwap_slope=0.0001, atr_ratio=1.05,
        pivots_recent=1, has_hhhl=True, has_lhll=False,
        vwap_crosses_30m=0, spy_vs_vwap=1.0, timestamp=datetime.utcnow(),
    )


# ── Direct gate tests ──────────────────────────────────────────────────────


def test_skip_when_leg_iv_missing():
    gate = ExpectedMoveGate()
    r = gate.check(spy_price=710.0, leg_iv=None, leg_mid=2.50, dte=2, direction="C")
    assert r.allowed
    assert "no leg IV" in r.reason


def test_skip_when_leg_mid_missing():
    gate = ExpectedMoveGate()
    r = gate.check(spy_price=710.0, leg_iv=0.18, leg_mid=None, dte=2, direction="C")
    assert r.allowed
    assert "no leg mid" in r.reason


def test_skip_when_leg_iv_zero():
    gate = ExpectedMoveGate()
    r = gate.check(spy_price=710.0, leg_iv=0.0, leg_mid=2.50, dte=2, direction="C")
    assert r.allowed


def test_skip_for_both_direction():
    gate = ExpectedMoveGate()
    r = gate.check(spy_price=710.0, leg_iv=0.18, leg_mid=2.50, dte=2, direction="BOTH")
    assert r.allowed
    assert "BOTH" in r.reason


def test_skip_when_spy_price_zero():
    gate = ExpectedMoveGate()
    r = gate.check(spy_price=0.0, leg_iv=0.18, leg_mid=2.50, dte=2, direction="C")
    assert r.allowed


def test_reject_when_em_below_threshold():
    """SPY 710, leg IV 12%, dte=1 → EM ≈ 4.46.
    leg_mid 4.00 → threshold 4.80 → reject."""
    gate = ExpectedMoveGate(ExpectedMoveGateConfig(em_multiplier=1.2))
    r = gate.check(spy_price=710.0, leg_iv=0.12, leg_mid=4.00, dte=1, direction="C")
    assert not r.allowed
    assert "EM" in r.reason
    assert r.expected_move is not None
    # Cross-check the math
    expected_em = 710.0 * 0.12 * math.sqrt(1 / 365.0)
    assert abs(r.expected_move - expected_em) < 1e-6


def test_pass_when_em_above_threshold():
    """SPY 710, leg IV 18%, dte=2 → EM ≈ 9.41.
    leg_mid 2.80 → threshold 3.36 → pass."""
    gate = ExpectedMoveGate(ExpectedMoveGateConfig(em_multiplier=1.2))
    r = gate.check(spy_price=710.0, leg_iv=0.18, leg_mid=2.80, dte=2, direction="C")
    assert r.allowed
    assert r.expected_move is not None
    assert r.leg_mid == 2.80


def test_dte_zero_floored_to_one_day():
    """dte=0 (intraday 0DTE) is floored to 1 day to avoid sqrt(0)."""
    gate = ExpectedMoveGate()
    r = gate.check(spy_price=710.0, leg_iv=0.18, leg_mid=2.80, dte=0, direction="P")
    # With dte floored to 1: EM ≈ 6.69, threshold = 3.36 → pass
    assert r.allowed
    assert r.expected_move is not None
    assert r.expected_move > 0


def test_multiplier_is_respected():
    """Same EM/leg_mid pair flips with a stricter multiplier."""
    spy, iv, mid, dte = 710.0, 0.15, 4.00, 2
    em = spy * iv * math.sqrt(dte / 365.0)  # ≈ 7.88
    # 1.2x → threshold 4.80 → pass (em 7.88 >= 4.80)
    g_loose = ExpectedMoveGate(ExpectedMoveGateConfig(em_multiplier=1.2))
    r_loose = g_loose.check(spy, iv, mid, dte, "C")
    assert r_loose.allowed
    # 2.5x → threshold 10.00 → reject (em 7.88 < 10.00)
    g_tight = ExpectedMoveGate(ExpectedMoveGateConfig(em_multiplier=2.5))
    r_tight = g_tight.check(spy, iv, mid, dte, "C")
    assert not r_tight.allowed


# ── Engine integration tests ───────────────────────────────────────────────


def _engine_cfg(em_enabled: bool) -> RulesV2Config:
    """Build a RulesV2Config that disables every other gate so we can
    isolate the EM gate's pass/block behaviour in engine.filter()."""
    cfg = RulesV2Config(
        enabled=True,
        regime_v2_enabled=False,
        orb_gate_enabled=False,
        continuation_enabled=False,
        pc_ratio_alignment_enabled=False,
        throttle_structure_enabled=False,
        entry_gate_enabled=False,           # bypass confidence/RSI checks
        exit_rules_enabled=False,
        strike_selector_enabled=False,
        time_of_day_enabled=False,
        expected_move_gate_enabled=em_enabled,
    )
    return cfg


def _inputs(bars):
    return EngineInputs(
        bars=bars, spy_price=710.0, vix=18.0, iv_rank=30.0,
        orb_high=712.0, orb_low=708.0, rsi_5m=55.0, vwap=710.0,
        option_quotes=None, now=datetime.utcnow(),
    )


def test_engine_skips_em_gate_when_disabled():
    eng = RulesV2Engine(_engine_cfg(em_enabled=False))
    bars = _bars()
    # Pricing that WOULD reject if the gate were on
    decision = eng.filter(
        signal_type="ORB_BREAKOUT", direction="C", price=710.0,
        confidence=0.80, regime=_ctx(TREND_UP, 710.0),
        inputs=_inputs(bars),
        leg_mid=4.00, leg_iv=0.12, dte=1,   # EM 4.46 < 1.2*4.00=4.80
    )
    assert decision.allowed
    assert decision.rule == "pass"


def test_engine_blocks_when_em_gate_rejects():
    eng = RulesV2Engine(_engine_cfg(em_enabled=True))
    bars = _bars()
    decision = eng.filter(
        signal_type="ORB_BREAKOUT", direction="C", price=710.0,
        confidence=0.80, regime=_ctx(TREND_UP, 710.0),
        inputs=_inputs(bars),
        leg_mid=4.00, leg_iv=0.12, dte=1,
    )
    assert not decision.allowed
    assert decision.rule == "expected_move_gate"
    assert "EM" in decision.reason


def test_engine_passes_when_em_gate_satisfied():
    eng = RulesV2Engine(_engine_cfg(em_enabled=True))
    bars = _bars()
    decision = eng.filter(
        signal_type="ORB_BREAKOUT", direction="C", price=710.0,
        confidence=0.80, regime=_ctx(TREND_UP, 710.0),
        inputs=_inputs(bars),
        leg_mid=2.80, leg_iv=0.18, dte=2,   # EM 9.41 >> 3.36
    )
    assert decision.allowed


def test_engine_skips_em_gate_when_pricing_missing():
    """Continuation path passes no leg_mid/leg_iv — must not be blocked."""
    eng = RulesV2Engine(_engine_cfg(em_enabled=True))
    bars = _bars()
    decision = eng.filter(
        signal_type="TREND_CONTINUATION", direction="C", price=710.0,
        confidence=0.82, regime=_ctx(TREND_UP, 710.0),
        inputs=_inputs(bars),
        # No leg_mid / leg_iv / dte
    )
    assert decision.allowed
