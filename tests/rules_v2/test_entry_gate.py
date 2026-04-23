"""Unit tests for EntryGate.

Covers:
- Confidence floor blocks low-confidence signals.
- VWAP-extension blocks over-extended fresh entries …
- … but does NOT block TREND_CONTINUATION signals.
- RSI exhaustion blocks fresh entries …
- … but does NOT block TREND_CONTINUATION signals.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shree.spy_options.rules_v2.config import EntryGateConfig  # noqa: E402
from shree.spy_options.rules_v2.entry_gate import EntryGate  # noqa: E402
from shree.spy_options.rules_v2.regime import (  # noqa: E402
    TREND_DOWN,
    RegimeV2Context,
)


def _bars(start=710.0, delta=-0.25, n=30):
    out = []
    t = datetime(2026, 4, 21, 12, 0)
    price = start
    for _ in range(n):
        o = price
        c = price + delta
        out.append({
            "date": t, "open": o,
            "high": max(o, c) + 0.15, "low": min(o, c) - 0.15,
            "close": c, "volume": 100000,
        })
        t += timedelta(minutes=5)
        price = c
    return out


def _ctx(regime: str, vwap: float) -> RegimeV2Context:
    return RegimeV2Context(
        regime=regime, vwap=vwap, vwap_slope=-0.0002, atr_ratio=1.1,
        pivots_recent=0, has_hhhl=False, has_lhll=True,
        vwap_crosses_30m=0, spy_vs_vwap=-2.0, timestamp=datetime.utcnow(),
    )


def test_low_confidence_blocks():
    gate = EntryGate()
    bars = _bars()
    r = gate.check(
        direction="P", bars=bars, spy_price=705.0,
        confidence=0.50,                     # below 0.70 default
        regime=_ctx(TREND_DOWN, 709.0),
    )
    assert not r.allowed
    assert "confidence" in r.reason


def test_vwap_extension_blocks_fresh_put():
    gate = EntryGate(EntryGateConfig(min_confidence=0.60, max_vwap_extension_pct=0.0025))
    bars = _bars()
    # Price 705 vs VWAP 710 = -0.7% below — way past 0.25% threshold
    r = gate.check(
        direction="P", bars=bars, spy_price=705.0,
        confidence=0.80, regime=_ctx(TREND_DOWN, 710.0),
        signal_type="ORB_BREAKOUT",
    )
    assert not r.allowed
    assert "extended" in r.reason


def test_vwap_extension_does_not_block_continuation():
    gate = EntryGate(EntryGateConfig(min_confidence=0.60, max_vwap_extension_pct=0.0025))
    bars = _bars()
    r = gate.check(
        direction="P", bars=bars, spy_price=705.0,
        confidence=0.80, regime=_ctx(TREND_DOWN, 710.0),
        signal_type="TREND_CONTINUATION",
    )
    assert r.allowed, f"continuation must bypass VWAP-extension check (got: {r.reason})"


def test_rsi_exhaustion_does_not_block_continuation():
    gate = EntryGate(EntryGateConfig(min_confidence=0.60, rsi_lower=30.0))
    # Strong downtrend pushes RSI < 30 → fresh-short entry would be blocked,
    # but TREND_CONTINUATION should pass through.
    bars = _bars(start=710.0, delta=-0.50, n=30)
    r_cont = gate.check(
        direction="P", bars=bars, spy_price=bars[-1]["close"],
        confidence=0.80, regime=_ctx(TREND_DOWN, 715.0),
        signal_type="TREND_CONTINUATION",
    )
    assert r_cont.allowed, r_cont.reason
