"""Phase 3 fixes for defects D1–D4 (2026-08-04 forensic audit).

Each test names the audit finding it protects. These are regression locks: if
someone re-introduces a second confidence gate, moves the V2 recorder back
downstream, drops the dedup markers, or strips the rejection context, the
corresponding test fails.
"""
from __future__ import annotations

import inspect
import json

import pytest

from shree.spy_options import manager as mgr
from shree.spy_options import signal_engine as se
from shree.spy_options.rules_v2.config import EntryGateConfig
from shree.spy_options.rules_v2.entry_gate import EntryGate
from shree.spy_options.rules_v2.regime import RegimeV2Context


def _regime(vwap=0.0):
    from datetime import datetime
    return RegimeV2Context(
        regime="TREND_UP", vwap=vwap, vwap_slope=0.0, atr_ratio=1.0,
        pivots_recent=0, has_hhhl=False, has_lhll=False, vwap_crosses_30m=0,
        spy_vs_vwap=0.0, timestamp=datetime.utcnow(), reasons=[])


# ── D1: exactly one confidence decision in production ───────────────────────

def test_d1_entry_gate_confidence_floor_is_disabled():
    assert EntryGateConfig().confidence_floor_enabled is False


def test_d1_low_confidence_no_longer_blocked_by_rules_v2():
    """91 of 125 entry_gate rejections on 2026-08-04 were this floor."""
    g = EntryGate(EntryGateConfig())
    r = g.check(direction="C", bars=[], spy_price=0.0, confidence=0.09,
                regime=_regime(), signal_type="CALL_SWEEP")
    assert "confidence" not in r.reason.lower(), r.reason


def test_d1_floor_still_works_when_explicitly_re_enabled():
    """Revert path must remain functional (one-line rollback)."""
    g = EntryGate(EntryGateConfig(confidence_floor_enabled=True))
    r = g.check(direction="C", bars=[], spy_price=0.0, confidence=0.09,
                regime=_regime(), signal_type="CALL_SWEEP")
    assert not r.allowed and "confidence" in r.reason


def test_d1_only_one_confidence_gate_remains_active():
    """Engine gate passive AND rules_v2 floor off => zero active gates."""
    from shree.utils.settings_loader import load_settings
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_settings(os.path.join(root, "config.yaml")).spy_options.signals
    assert cfg.confidence_gate_enabled is False
    assert EntryGateConfig().confidence_floor_enabled is False


# ── D2: V2 evaluates every qualified signal, not just dispatched ones ───────

def test_d2_v2_recorder_runs_before_rules_v2_and_dispatch():
    """AUG 6 2026: the recorder moved from manager._poll into
    signal_engine.evaluate(). The Phase 3 placement was still downstream of the
    directional-conflict filter, which cost 6 of 7 qualified CALL_SWEEP on
    2026-08-05. Being inside evaluate() is strictly earlier than rules_v2 and
    dispatch, so the original ordering guarantee still holds — see
    tests/test_fixes_20260806.py for the finer-grained assertions."""
    src = inspect.getsource(se.SignalEngine.evaluate)
    assert "_v2.record(_s)" in src, "V2 recorder missing from evaluate()"
    mgr_src = inspect.getsource(mgr.SpyOptionsManager._poll)
    assert "v2_shadow_gate.record(" not in mgr_src, \
        "manager must not also record — that would double-log"


def test_d2_downstream_recorder_removed():
    """The old post-dispatch record() call must be gone, or V2 double-logs."""
    src = inspect.getsource(mgr.SpyOptionsManager._dispatch_signals)
    assert "v2_shadow_gate.record(" not in src


def test_d2_comparison_notify_still_present():
    """V1-vs-V2 comparison needs V1's decision, so it stays in dispatch."""
    src = inspect.getsource(mgr.SpyOptionsManager._dispatch_signals)
    assert "v2_shadow_gate.evaluate(sig)" in src


# ── D3: shadow ledger carries dedup markers ─────────────────────────────────

class _Sig:
    signal_type = se.SignalType.CALL_SWEEP
    right = "C"
    strike = 640.0
    expiry = "AUG26"
    expiry_date = "20260807"
    confidence = 0.42
    confidence_tier = "MEDIUM"
    regime = "TREND_UP"
    spy_price = 638.5
    dte = 1


def _rows(tmp_path):
    return [json.loads(l) for l in
            (tmp_path / "logs" / "blocked_signals.jsonl").read_text().splitlines()]


def test_d3_repeat_emissions_are_marked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    se._EMISSION_SEQ.clear(); se._EMISSION_DAY.clear()
    for _ in range(5):
        se.log_blocked_signal(_Sig(), "rules_v2:entry_gate", "x")
    rows = _rows(tmp_path)
    assert len(rows) == 5, "rows must NOT be dropped — repeat rate is signal"
    assert [r["emission_seq"] for r in rows] == [1, 2, 3, 4, 5]
    assert sum(r["is_first_emission"] for r in rows) == 1


def test_d3_dedup_yields_unity_duplicate_factor(tmp_path, monkeypatch):
    """The 17.7x inflation is removed by filtering to first emissions."""
    monkeypatch.chdir(tmp_path)
    se._EMISSION_SEQ.clear(); se._EMISSION_DAY.clear()
    for _ in range(18):
        se.log_blocked_signal(_Sig(), "rules_v2:entry_gate", "x")
    rows = _rows(tmp_path)
    uniq = [r for r in rows if r["is_first_emission"]]
    assert len(rows) / len(uniq) == 18.0      # raw inflation
    assert len(uniq) == 1                      # deduped == unique opportunity


def test_d3_distinct_contracts_are_distinct_opportunities(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    se._EMISSION_SEQ.clear(); se._EMISSION_DAY.clear()

    class Other(_Sig):
        strike = 641.0
    se.log_blocked_signal(_Sig(), "rules_v2:entry_gate", "x")
    se.log_blocked_signal(Other(), "rules_v2:entry_gate", "x")
    rows = _rows(tmp_path)
    assert all(r["is_first_emission"] for r in rows)
    assert len({r["opportunity_key"] for r in rows}) == 2


# ── D4: rejection context is reconstructible ────────────────────────────────

def test_d4_rejection_row_carries_full_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    se._EMISSION_SEQ.clear(); se._EMISSION_DAY.clear()
    se.set_rejection_context(regime_v2="TREND_UP", atr_ratio=1.31,
                             rsi_5m=68.2, vwap=765.4, spy_vs_vwap=0.0065,
                             vwap_band_position="ABOVE_1SD")
    se.log_blocked_signal(_Sig(), "rules_v2:entry_gate", "exhaustion")
    r = _rows(tmp_path)[0]
    for f in ("regime_v2", "atr_ratio", "rsi_5m", "vwap", "spy_vs_vwap",
              "vwap_band_position", "delta", "iv_rank", "confidence",
              "intraday_pc_ratio", "flow_confirmation_score"):
        assert f in r, f"missing rejection-context field: {f}"
    assert r["atr_ratio"] == 1.31 and r["rsi_5m"] == 68.2


def test_d4_context_setter_never_raises():
    se.set_rejection_context(bad=object(), none=None)   # must not raise


def test_d4_logging_still_never_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class Broken:
        signal_type = se.SignalType.CALL_SWEEP
        def __getattr__(self, n):
            raise RuntimeError("boom")

    se.log_blocked_signal(Broken(), "rules_v2:entry_gate", "x")
