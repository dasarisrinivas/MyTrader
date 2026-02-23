"""Tests for VX scaling exemption on OR breakout signals (FEB 23 2026).

Validates that OR breakout/breakdown signals are exempt from VX-based
confidence scaling, while pullback signals still receive VX scaling.

Root cause: 2026-02-23 09:45 OR_BREAK_SHORT at 6865.75 was blocked
because VX 0.724× dropped confidence from 0.70→0.507 (below 0.50
threshold).  TP hit within 15 min → missed +$40 profit.

OR breakout signals thrive in elevated VX — penalizing them for
volatility is self-defeating.  Pullback signals still need VX scaling
because wider noise bands = higher stop-out risk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from shree.execution.components.signal_processor import (
    SignalGenerationResult,
    SignalProcessor,
)


# ---------------------------------------------------------------------------
# Minimal stubs (same pattern as test_chop_regime_guard.py)
# ---------------------------------------------------------------------------

@dataclass
class _StubStatus:
    last_signal: str = ""
    signal_confidence: float = 0.0
    hybrid_market_trend: Optional[str] = None
    hybrid_volatility_regime: Optional[str] = None


@dataclass
class _StubContextManager:
    def refresh_hybrid_context(self, pipeline_result):
        pass


def _make_signal(action="SELL", confidence=0.70, reason="OR_BREAK_SHORT | ADX=24 | OR_L=6867.50"):
    return SimpleNamespace(
        action=action,
        confidence=confidence,
        metadata={"reason": reason},
    )


def _make_features():
    return pd.DataFrame({
        "high": [6920.0, 6890.0, 6871.8],
        "low": [6910.0, 6870.0, 6860.0],
        "close": [6918.0, 6877.0, 6865.75],
        "open": [6915.0, 6890.0, 6873.0],
        "volume": [100, 110, 95],
    })


def _make_stub_manager(hybrid_trend=None):
    manager = MagicMock()
    manager.status = _StubStatus(hybrid_market_trend=hybrid_trend)
    manager.context_manager = _StubContextManager()
    manager._use_hybrid_pipeline = False
    manager.hybrid_pipeline = None
    manager._current_pipeline_result = None
    manager.executor = None
    return manager


def _make_processor(manager=None, engine=None, vx_multiplier=0.724):
    """Create a SignalProcessor with mocked VX multiplier."""
    proc = object.__new__(SignalProcessor)
    proc.manager = manager or _make_stub_manager()
    proc.engine = engine or MagicMock()
    proc.hybrid_pipeline = None
    proc._multi_source_enabled = False
    proc._stocktwits_enabled = False
    proc._vx_feed_enabled = True
    proc._vx_feed = MagicMock()
    proc.settings = MagicMock()
    proc._emergency_generator = MagicMock()
    # Mock the _sentiment helper with configurable VX multiplier
    sentiment_mock = MagicMock()
    sentiment_mock.get_vx_multiplier.return_value = vx_multiplier
    proc._sentiment = sentiment_mock
    return proc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVxOrBreakoutExemption:
    """OR breakout signals should be exempt from VX scaling."""

    @pytest.mark.asyncio
    async def test_or_break_short_exempt_from_vx_scaling(self):
        """OR_BREAK_SHORT with VX 0.724× → confidence stays at 0.70."""
        manager = _make_stub_manager()
        engine = MagicMock()
        signal = _make_signal(
            action="SELL",
            confidence=0.70,
            reason="OR_BREAK_SHORT | ADX=24 | OR_L=6867.50",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_multiplier=0.724)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6865.75,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "SELL"
        # Confidence should NOT be reduced by VX multiplier
        assert result.signal.confidence == 0.70
        # Should record exemption in overlays
        overlays = result.signal.metadata.get("confidence_overlays", {})
        assert "vx_multiplier_exempt" in overlays
        assert overlays["vx_multiplier_exempt"] == 0.724
        # Should NOT have regular vx_multiplier key
        assert "vx_multiplier" not in overlays

    @pytest.mark.asyncio
    async def test_or_break_long_exempt_from_vx_scaling(self):
        """OR_BREAK_LONG with VX 0.724× → confidence stays at 0.70."""
        manager = _make_stub_manager()
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="OR_BREAK_LONG | ADX=22 | OR_H=6871.75",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_multiplier=0.724)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6875.00,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "BUY"
        assert result.signal.confidence == 0.70
        overlays = result.signal.metadata.get("confidence_overlays", {})
        assert "vx_multiplier_exempt" in overlays

    @pytest.mark.asyncio
    async def test_pullback_signal_still_gets_vx_scaling(self):
        """EMA21_PB_LONG with VX 0.724× → confidence IS reduced."""
        manager = _make_stub_manager()
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="EMA21_PB_LONG | ADX=22 | RSI=52",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_multiplier=0.724)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6890.00,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "BUY"
        # Confidence SHOULD be reduced: 0.70 × 0.724 ≈ 0.507
        assert abs(result.signal.confidence - 0.70 * 0.724) < 0.001
        overlays = result.signal.metadata.get("confidence_overlays", {})
        assert "vx_multiplier" in overlays
        assert overlays["vx_multiplier"] == 0.724
        assert "vx_multiplier_exempt" not in overlays

    @pytest.mark.asyncio
    async def test_ema9_pullback_still_gets_vx_scaling(self):
        """EMA9_PB_LONG with VX 0.80× → confidence IS reduced."""
        manager = _make_stub_manager()
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="EMA9_PB_LONG | ADX=20 | RSI=55 | ATR=10.5",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_multiplier=0.80)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6895.00,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "BUY"
        # 0.70 × 0.80 = 0.56
        assert abs(result.signal.confidence - 0.56) < 0.001
        overlays = result.signal.metadata.get("confidence_overlays", {})
        assert "vx_multiplier" in overlays

    @pytest.mark.asyncio
    async def test_vx_multiplier_1_no_change_for_any_signal(self):
        """VX multiplier at 1.0 (normal) → no adjustment for any signal type."""
        manager = _make_stub_manager()
        engine = MagicMock()
        signal = _make_signal(
            action="SELL",
            confidence=0.70,
            reason="OR_BREAK_SHORT | ADX=24 | OR_L=6867.50",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_multiplier=1.0)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6865.75,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.confidence == 0.70
        overlays = result.signal.metadata.get("confidence_overlays", {})
        # Neither key should exist when multiplier is 1.0
        assert "vx_multiplier" not in overlays
        assert "vx_multiplier_exempt" not in overlays

    @pytest.mark.asyncio
    async def test_trend_continuation_still_gets_vx_scaling(self):
        """TREND_CONT_LONG is not OR breakout → VX scaling applies."""
        manager = _make_stub_manager()
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="TREND_CONT_LONG | ADX=28 | RSI=60",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_multiplier=0.724)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6910.00,
            structural_metrics=None,
        )

        assert result is not None
        # 0.70 × 0.724 ≈ 0.507
        assert abs(result.signal.confidence - 0.70 * 0.724) < 0.001
        overlays = result.signal.metadata.get("confidence_overlays", {})
        assert "vx_multiplier" in overlays

    @pytest.mark.asyncio
    async def test_today_scenario_would_pass_with_fix(self):
        """Replay exact 09:45 scenario: OR_BREAK_SHORT, VX=0.724, hybrid oppose.
        
        Before fix: 0.70 × 0.724 = 0.507 → -0.10 hybrid = 0.407 → BLOCKED
        After fix:  0.70 (exempt) → -0.10 hybrid = 0.60 → PASSES
        """
        manager = _make_stub_manager(hybrid_trend="CHOP")
        # Enable hybrid pipeline to test the full overlay chain
        manager._use_hybrid_pipeline = True
        hybrid_pipeline = AsyncMock()
        hybrid_signal = SimpleNamespace(
            action="BUY",  # Hybrid opposes the SELL
            confidence=0.47,
            metadata={"market_trend": "CHOP", "volatility_regime": "MEDIUM"},
        )
        pipeline_result = MagicMock()
        pipeline_result.rule_engine = MagicMock()
        pipeline_result.rule_engine.market_trend = "CHOP"
        pipeline_result.rule_engine.volatility_regime = "MEDIUM"
        hybrid_pipeline.process = AsyncMock(return_value=(hybrid_signal, pipeline_result))

        engine = MagicMock()
        signal = _make_signal(
            action="SELL",
            confidence=0.70,
            reason="OR_BREAK_SHORT | ADX=24 | OR_L=6867.50",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_multiplier=0.724)
        proc.hybrid_pipeline = hybrid_pipeline

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6865.75,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "SELL"
        # VX exempt → 0.70, then hybrid oppose → -0.10 = 0.60
        assert result.signal.confidence >= 0.50, (
            f"Expected conf >= 0.50, got {result.signal.confidence:.3f}. "
            f"The OR breakout should pass the threshold after VX exemption."
        )
        # Verify it's approximately 0.60 (0.70 - 0.10)
        assert abs(result.signal.confidence - 0.60) < 0.02
