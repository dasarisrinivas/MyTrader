"""Confidence gate demoted to a passive metric (AUG 3 2026).

Locks the three invariants of the change so a future edit cannot silently
undo them:

  1. The gate is OFF in config.yaml, but `min_confidence` is RETAINED as the
     reference threshold for the counterfactual shadow log.
  2. The dataclass DEFAULT stays True — reverting is a one-line config flip,
     and any code path that builds the config without YAML keeps old behaviour.
  3. VWAP_REVERSION is suppressed STRUCTURALLY, not by confidence. Before this
     change it was held back only by `vrev_confidence` (0.60) sitting below
     `min_confidence` (0.77) — so turning the gate off would have promoted a
     shadow-incubating strategy straight to live, bypassing the scorecard.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from shree.config.spy_options import SpyOptionsSignalConfig
from shree.spy_options.signal_engine import (
    SignalType,
    _SHADOW_ONLY_FAMILIES,
    log_blocked_signal,
)
from shree.utils.settings_loader import load_settings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def signals_cfg():
    return load_settings(os.path.join(REPO, "config.yaml")).spy_options.signals


# ── 1. the gate is off, the threshold is retained ────────────────────────────

def test_gate_disabled_in_config(signals_cfg):
    assert signals_cfg.confidence_gate_enabled is False


def test_min_confidence_retained_as_reference(signals_cfg):
    """Threshold must survive the switch — the shadow ledger keys off it, so
    zeroing it would break continuity of the 30-session experiment."""
    assert signals_cfg.min_confidence == pytest.approx(0.77)


# ── 2. revert safety ─────────────────────────────────────────────────────────

def test_dataclass_default_preserves_old_behaviour():
    assert SpyOptionsSignalConfig().confidence_gate_enabled is True


def test_gate_flag_is_a_real_field():
    cfg = SpyOptionsSignalConfig(confidence_gate_enabled=False)
    assert cfg.confidence_gate_enabled is False


# ── 3. shadow-incubating families are confidence-independent ─────────────────

def test_vwap_reversion_is_structurally_shadow_only():
    assert SignalType.VWAP_REVERSION in _SHADOW_ONLY_FAMILIES


def test_shadow_suppression_does_not_depend_on_confidence(signals_cfg):
    """Documents the coupling this change removed: vrev confidence is still
    below the threshold, but that is no longer what keeps it out of the
    executor. If someone raises vrev_confidence above min_confidence, the
    structural suppression must still hold."""
    assert signals_cfg.vrev_confidence < signals_cfg.min_confidence
    assert SignalType.VWAP_REVERSION in _SHADOW_ONLY_FAMILIES


def test_tradeable_families_are_not_suppressed():
    for fam in (SignalType.CALL_SWEEP, SignalType.PUT_SWEEP,
                SignalType.ORB_BREAKOUT, SignalType.PC_RATIO_EXTREME,
                SignalType.TREND_CONTINUATION):
        assert fam not in _SHADOW_ONLY_FAMILIES


# ── 4. the counterfactual ledger keeps recording ─────────────────────────────

class _FakeSig:
    signal_type = SignalType.CALL_SWEEP
    right = "C"
    strike = 640.0
    expiry = "AUG26"
    expiry_date = "20260807"
    confidence = 0.42
    confidence_tier = "MEDIUM"
    regime = "RANGE_BOUND"
    spy_price = 638.5
    dte = 1


def test_shadow_gate_label_is_logged(tmp_path, monkeypatch):
    """`confidence_threshold_shadow` rows are what make the experiment
    continuous across the switch — without them the rejected arm goes dark."""
    monkeypatch.chdir(tmp_path)
    log_blocked_signal(_FakeSig(), "confidence_threshold_shadow",
                       "conf 0.42 < min 0.77 (gate DISABLED)")
    rec = json.loads((tmp_path / "logs" / "blocked_signals.jsonl").read_text().strip())
    assert rec["gate"] == "confidence_threshold_shadow"
    assert rec["confidence"] == pytest.approx(0.42)
    # expiry_date is what makes a blocked signal replayable at all
    assert rec["expiry_date"] == "20260807"


def test_log_blocked_signal_never_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class Broken:
        signal_type = SignalType.CALL_SWEEP
        def __getattr__(self, name):
            raise RuntimeError("boom")

    log_blocked_signal(Broken(), "confidence_threshold_shadow", "x")  # must not raise
