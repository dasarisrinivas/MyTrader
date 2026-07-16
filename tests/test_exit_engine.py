"""Unit tests for the Exit Engine v2 (shree/spy_options/exit_engine.py).

Pure-function tests — no IB, no manager, no I/O. Each test builds snapshots
explicitly and feeds them through ExitEngine.evaluate, asserting both the
action and (where relevant) the score/factor breakdown.
"""

from __future__ import annotations

import dataclasses

import pytest

from shree.config.spy_options import SpyOptionsExitEngineConfig
from shree.spy_options.exit_engine import (
    FULL_EXIT,
    HOLD,
    PARTIAL_EXIT,
    TRAIL_STOP,
    STAGE_EXIT_PENDING,
    ExitEngine,
    ExitSnapshot,
    PositionExitState,
)


def make_snap(**over) -> ExitSnapshot:
    """A healthy bullish TREND_CONTINUATION position, 20 min in, no stress."""
    base = dict(
        direction="BULLISH",
        spy_price=752.0,
        entry_spy=751.0,
        last_bar_ts="2026-07-14T10:30:00",
        last_close=752.0,
        last_bar_range=0.5,
        vwap=750.5,
        ema9=751.5,
        ema21=751.0,
        ema9_slope=0.05,
        rsi_5m=62.0,
        atr14=0.6,
        vwap_band="ABOVE_1SD",
        regime="TREND_UP",
        minutes_held=20.0,
        max_hold_min=90.0,
        dte=2,
        is_late_0dte=False,
        vix=17.0,
        tape_score=10.0,
        delta_now=None,
        unrealized_r=0.3,
        premium_loss_pct_est=0.0,
        spy_adverse_pct=0.0,
        entry_vwap_band="ABOVE_1SD",
        entry_regime="TREND_UP",
        entry_tier="HIGH",
        entry_confidence=0.82,
        entry_rsi=63.0,
        entry_delta=0.40,
        iv_stop_pct=15.0,
    )
    base.update(over)
    return ExitSnapshot(**base)


def bar(snap: ExitSnapshot, ts: str, **over) -> ExitSnapshot:
    """New closed bar variant of a snapshot."""
    return dataclasses.replace(snap, last_bar_ts=ts, **over)


@pytest.fixture()
def engine() -> ExitEngine:
    return ExitEngine(SpyOptionsExitEngineConfig())


@pytest.fixture()
def state() -> PositionExitState:
    return PositionExitState()


# ── Grace period ─────────────────────────────────────────────────────────────

def test_grace_blocks_soft_exit_at_2min(engine, state):
    """The Jul-13 749P scenario: band demotion 2 minutes after entry → HOLD."""
    snap = make_snap(
        direction="BEARISH", entry_vwap_band="BELOW_2SD", vwap_band="INSIDE_1SD",
        regime="TREND_DOWN", entry_regime="TREND_DOWN", ema9_slope=-0.05,
        minutes_held=2.0, unrealized_r=0.0, rsi_5m=40.0, entry_rsi=35.0,
        last_close=750.4, vwap=750.5, ema9=750.8, ema21=751.2,
    )
    dec = engine.evaluate(state, snap)
    assert dec.action == HOLD
    assert dec.reason == "grace"


def test_catastrophic_pierces_grace(engine, state):
    dec = engine.evaluate(state, make_snap(minutes_held=1.0, spy_adverse_pct=1.2))
    assert dec.action == FULL_EXIT
    assert "catastrophic" in dec.reason


def test_premium_catastrophic_pierces_grace(engine, state):
    dec = engine.evaluate(
        state, make_snap(minutes_held=1.0, premium_loss_pct_est=26.0, iv_stop_pct=15.0)
    )
    assert dec.action == FULL_EXIT
    assert "premium" in dec.reason


# ── Hysteresis / confirmation ────────────────────────────────────────────────

def test_single_band_demotion_scores_zero(engine, state):
    """ABOVE_1SD→INSIDE_1SD on ONE closed bar contributes nothing."""
    snap = make_snap(vwap_band="INSIDE_1SD", last_close=751.8, ema9_slope=-0.01)
    dec = engine.evaluate(state, snap)
    assert dec.action == HOLD
    assert all(name != "vwap_band_decay" for name, _ in dec.factors)


def test_band_decay_needs_three_closes_and_slope(engine, state):
    snap = make_snap(vwap_band="INSIDE_1SD", ema9_slope=-0.02, last_close=751.8)
    for i in range(3):
        dec = engine.evaluate(state, bar(snap, f"2026-07-14T10:{35+5*i}:00"))
    assert ("vwap_band_decay", 10) in dec.factors
    # 10 points alone is far below any threshold.
    assert dec.action == HOLD


def test_regime_flip_single_eval_no_exit(engine, state):
    snap = make_snap(regime="TREND_DOWN", ema9_slope=-0.05)
    dec = engine.evaluate(state, snap)
    assert dec.action == HOLD
    assert all(n != "regime_flip_confirmed" for n, _ in dec.factors)


def test_regime_flip_three_evals_scores(engine, state):
    snap = make_snap(regime="TREND_DOWN", ema9_slope=-0.05)
    for _ in range(3):
        dec = engine.evaluate(state, snap)
    assert ("regime_flip_confirmed", 20) in dec.factors


def test_ema9_retest_alone_never_exits(engine, state):
    """Trend-pullback tolerance: EMA9 cross alone caps at 15 — always HOLD."""
    snap = make_snap(last_close=751.3, ema9=751.5)  # close below EMA9
    for i in range(5):
        dec = engine.evaluate(state, bar(snap, f"2026-07-14T10:{35+5*i}:00"))
        assert dec.action == HOLD


# ── Threshold, EXIT_PENDING, de-escalation ───────────────────────────────────

def _stress_bear_stack(engine, state):
    """Drive a bullish position into a confirmed multi-factor breakdown:
    VWAP cross + EMA9 + EMA21 + confirmed regime flip = 25+15+20+20 = 80."""
    snap = make_snap(
        regime="TREND_DOWN", ema9_slope=-0.08,
        last_close=749.9, vwap=750.5, ema9=750.8, ema21=750.6,
        vwap_band="BELOW_1SD", unrealized_r=-0.2, rsi_5m=44.0,
    )
    decs = []
    for i in range(3):
        decs.append(engine.evaluate(state, bar(snap, f"2026-07-14T11:{10+5*i}:00")))
    return snap, decs


def test_confirmed_breakdown_exits(engine, state):
    snap, decs = _stress_bear_stack(engine, state)
    # Score 80+ ≥ thr+15 → strong one-shot exit by the 2nd/3rd evaluation.
    assert decs[-1].action == FULL_EXIT
    assert decs[-1].score >= decs[-1].threshold + 15


def test_moderate_score_needs_second_eval(engine, state):
    """Score in [thr, thr+15) → EXIT_PENDING first, FULL_EXIT on the next."""
    cfg = SpyOptionsExitEngineConfig(base_threshold=55, one_shot_margin=15)
    eng = ExitEngine(cfg)
    # ema21 break (20) + regime flip (20) + ema9 (15) = 55, exactly thr.
    snap = make_snap(
        regime="TREND_DOWN", ema9_slope=-0.08, vwap=749.0,  # vwap far below: no cross
        last_close=750.0, ema9=750.6, ema21=750.4, unrealized_r=-0.1,
        entry_tier="HIGH", entry_confidence=0.80,
    )
    d1 = eng.evaluate(state, bar(snap, "2026-07-14T11:10:00"))
    d2 = eng.evaluate(state, bar(snap, "2026-07-14T11:15:00"))
    d3 = eng.evaluate(state, bar(snap, "2026-07-14T11:20:00"))  # score first hits 55
    d4 = eng.evaluate(state, bar(snap, "2026-07-14T11:25:00"))  # confirmation
    assert d1.action == HOLD and d2.action == HOLD
    assert d3.stage == STAGE_EXIT_PENDING and d3.action == HOLD
    assert d4.action == FULL_EXIT and "2nd consecutive" in d4.reason


def test_deescalation_hysteresis(engine, state):
    """Arm EXIT_PENDING, then evidence fades → back to MANAGE, no exit."""
    snap_bad = make_snap(
        regime="TREND_DOWN", ema9_slope=-0.08, vwap=749.0,
        last_close=750.0, ema9=750.6, ema21=750.4, unrealized_r=-0.1,
    )
    d1 = engine.evaluate(state, bar(snap_bad, "2026-07-14T11:10:00"))
    d2 = engine.evaluate(state, bar(snap_bad, "2026-07-14T11:15:00"))
    del d1, d2
    # Recovery bar: back above everything, regime restored.
    snap_ok = make_snap(last_close=752.4)
    d3 = engine.evaluate(state, bar(snap_ok, "2026-07-14T11:20:00"))
    assert d3.action == HOLD
    assert not state.pending


# ── Profit ladder ────────────────────────────────────────────────────────────

def test_partial_at_1R_once_with_BE(engine, state):
    snap = make_snap(unrealized_r=1.05)
    d1 = engine.evaluate(state, snap)
    assert d1.action == PARTIAL_EXIT
    assert d1.fraction == pytest.approx(0.5)
    assert d1.new_stop_r == pytest.approx(0.0)      # breakeven
    # Idempotent: second evaluation does NOT partial again.
    d2 = engine.evaluate(state, snap)
    assert d2.action == HOLD


def test_trail_locks_half_hwm_gain(engine, state):
    engine.evaluate(state, make_snap(unrealized_r=1.05))            # partial+BE
    engine.evaluate(state, make_snap(unrealized_r=2.0))             # HWM → 2.0
    dec = engine.evaluate(state, make_snap(unrealized_r=1.4))       # gave back 0.6R
    assert dec.action == TRAIL_STOP
    assert dec.new_stop_r == pytest.approx(1.0)                     # 2.0 × 0.5


# ── Priorities ───────────────────────────────────────────────────────────────

def test_time_ceiling_beats_everything(engine, state):
    dec = engine.evaluate(state, make_snap(minutes_held=91.0, unrealized_r=2.5))
    assert dec.action == FULL_EXIT
    assert "max hold" in dec.reason


# ── Symmetry / adaptivity / determinism ──────────────────────────────────────

def test_put_call_mirror_symmetry(engine):
    """A mirrored bearish scenario must produce the identical score."""
    bull_state, bear_state = PositionExitState(), PositionExitState()
    bull = make_snap(
        regime="TREND_DOWN", ema9_slope=-0.08, last_close=749.9,
        vwap=750.5, ema9=750.8, ema21=750.6, vwap_band="BELOW_1SD",
        unrealized_r=-0.2, rsi_5m=44.0,
    )
    bear = make_snap(
        direction="BEARISH", entry_vwap_band="BELOW_1SD",
        entry_regime="TREND_DOWN", entry_rsi=37.0, entry_delta=-0.40,
        regime="TREND_UP", ema9_slope=0.08, last_close=751.1,
        vwap=750.5, ema9=750.2, ema21=750.4, vwap_band="ABOVE_1SD",
        unrealized_r=-0.2, rsi_5m=56.0, tape_score=-10.0,  # mirrored tape
    )
    e2 = ExitEngine(SpyOptionsExitEngineConfig())
    for i in range(3):
        db = engine.evaluate(bull_state, bar(bull, f"2026-07-14T11:{10+5*i}:00"))
        dp = e2.evaluate(bear_state, bar(bear, f"2026-07-14T11:{10+5*i}:00"))
    assert db.score == dp.score
    assert sorted(n for n, _ in db.factors) == sorted(n for n, _ in dp.factors)


def test_adaptive_threshold_extreme_tier_higher(engine, state):
    hi = engine.evaluate(state, make_snap(entry_tier="EXTREME", entry_confidence=0.9))
    state2 = PositionExitState()
    lo = engine.evaluate(state2, make_snap(is_late_0dte=True, dte=0, vix=25.0,
                                           minutes_held=40.0, max_hold_min=45.0))
    assert hi.threshold > lo.threshold
    assert lo.threshold >= 40 and hi.threshold <= 80


def test_determinism_same_inputs_same_output(engine):
    s1, s2 = PositionExitState(), PositionExitState()
    e2 = ExitEngine(SpyOptionsExitEngineConfig())
    snap = make_snap(regime="TREND_DOWN", ema9_slope=-0.05, last_close=750.2,
                     ema21=750.4)
    for i in range(4):
        b = bar(snap, f"2026-07-14T12:{10+5*i}:00")
        d1 = engine.evaluate(s1, b)
        d2 = e2.evaluate(s2, b)
        assert (d1.action, d1.score, d1.threshold, d1.factors) == (
            d2.action, d2.score, d2.threshold, d2.factors
        )
