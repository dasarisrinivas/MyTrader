"""Tests for VX additive signal-type-aware scaling (MAR 3 2026).

Validates that VX scaling uses additive tiers instead of multiplicative,
and that different signal types get appropriate VX adjustments:
  - OR breakouts/breakdowns get favorable treatment (thrive in elevated VX)
  - Pullback signals get cautious treatment (wider noise in high VX)
  - Trend continuation signals get context-dependent treatment

History: Replaced the old multiplicative VX system (conf * 0.7) which was
the #1 signal killer. The new additive system uses VX price tiers with
signal-type awareness.

VX tiers (MES median VX is ~18-22):
  VX < 16:  low vol - slight reduce (-0.03 pullback, -0.08 continuation)
  16-22:    normal MES baseline - neutral (0.0)
  22-28:    elevated - breakout +0.03, continuation +0.05, pullback -0.05
  28-35:    high fear - breakout 0.0, others -0.08
  35+:      extreme - breakout -0.03, others -0.15
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from shree.execution.components.signal_processor import (
    SignalGenerationResult,
    SignalProcessor,
)


# ---------------------------------------------------------------------------
# Minimal stubs
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


def _make_processor(manager=None, engine=None, vx_price=None):
    """Create a SignalProcessor with mocked VX price for additive tiers.

    Args:
        vx_price: The raw VX futures price (e.g. 24.5 means VX at 24.5).
                  None means VX feed unavailable.
    """
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

    # Mock the _sentiment helper with VX feed that returns raw price
    sentiment_mock = MagicMock()
    sentiment_mock._vx_feed_enabled = (vx_price is not None)
    if vx_price is not None:
        vx_feed_mock = MagicMock()
        vx_feed_mock.get_vx_price.return_value = vx_price
        sentiment_mock._vx_feed = vx_feed_mock
    else:
        sentiment_mock._vx_feed = None
    sentiment_mock.get_vx_multiplier.return_value = 1.0
    proc._sentiment = sentiment_mock
    return proc


# ---------------------------------------------------------------------------
# Tests - Additive VX Tiers
# ---------------------------------------------------------------------------


class TestVxOrBreakoutExemption:
    """OR breakout signals should get favorable VX treatment (additive tiers)."""

    @pytest.mark.asyncio
    async def test_or_break_short_boosted_in_elevated_vx(self):
        """OR_BREAK_SHORT with VX=24.5 (elevated) -> gets +0.03 BOOST."""
        manager = _make_stub_manager()
        engine = MagicMock()
        signal = _make_signal(
            action="SELL",
            confidence=0.70,
            reason="OR_BREAK_SHORT | ADX=24 | OR_L=6867.50",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_price=24.5)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6865.75,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "SELL"
        # VX 22-28 + breakout -> +0.03 boost: 0.70 + 0.03 = 0.73
        assert abs(result.signal.confidence - 0.73) < 0.01

    @pytest.mark.asyncio
    async def test_or_break_long_boosted_in_elevated_vx(self):
        """OR_BREAK_LONG with VX=24.5 (elevated) -> gets +0.03 BOOST."""
        manager = _make_stub_manager()
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="OR_BREAK_LONG | ADX=22 | OR_H=6871.75",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_price=24.5)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6875.00,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "BUY"
        assert abs(result.signal.confidence - 0.73) < 0.01

    @pytest.mark.asyncio
    async def test_pullback_penalized_in_elevated_vx(self):
        """EMA21_PB_LONG with VX=24.5 (elevated) -> confidence IS reduced."""
        manager = _make_stub_manager()
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="EMA21_PB_LONG | ADX=22 | RSI=52",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_price=24.5)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6890.00,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "BUY"
        # VX 22-28 + pullback -> -0.05: 0.70 - 0.05 = 0.65
        assert abs(result.signal.confidence - 0.65) < 0.01

    @pytest.mark.asyncio
    async def test_ema9_pullback_penalized_in_high_fear(self):
        """EMA9_PB_LONG with VX=30 (high fear) -> -0.08 reduction."""
        manager = _make_stub_manager()
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="EMA9_PB_LONG | ADX=20 | RSI=55 | ATR=10.5",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_price=30.0)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6895.00,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "BUY"
        # VX 28-35 + pullback -> -0.08: 0.70 - 0.08 = 0.62
        assert abs(result.signal.confidence - 0.62) < 0.01

    @pytest.mark.asyncio
    async def test_vx_normal_range_no_change(self):
        """VX=19 (normal 16-22 range) -> no adjustment for any signal type."""
        manager = _make_stub_manager()
        engine = MagicMock()
        signal = _make_signal(
            action="SELL",
            confidence=0.70,
            reason="OR_BREAK_SHORT | ADX=24 | OR_L=6867.50",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_price=19.0)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6865.75,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.confidence == 0.70

    @pytest.mark.asyncio
    async def test_trend_continuation_penalized_in_high_fear(self):
        """TREND_CONT_LONG with VX=30 (high fear) -> VX scaling applies."""
        manager = _make_stub_manager()
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="TREND_CONT_LONG | ADX=28 | RSI=60",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_price=30.0)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6910.00,
            structural_metrics=None,
        )

        assert result is not None
        # VX 28-35 + trend_cont -> -0.08: 0.70 - 0.08 = 0.62
        assert abs(result.signal.confidence - 0.62) < 0.01

    @pytest.mark.asyncio
    async def test_today_scenario_or_breakout_blocked_in_chop(self):
        """Replay 09:45 scenario: OR_BREAK_SHORT, VX=24.5, hybrid oppose.

        Before MAR 24 fix: OR breakout was exempt from CHOP guard.
        VX additive +0.03 → 0.73, hybrid oppose -0.094 → 0.636. Passed.

        After MAR 24 fix: OR breakouts in CHOP are blocked. Evidence:
        5/5 executed OR breakout fills were SL_HIT losers (−$180.87).
        The CHOP guard now correctly blocks this signal.
        """
        manager = _make_stub_manager(hybrid_trend="CHOP")
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

        proc = _make_processor(manager=manager, engine=engine, vx_price=24.5)
        proc.hybrid_pipeline = hybrid_pipeline

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6865.75,
            structural_metrics=None,
        )

        assert result is not None
        # MAR 24 2026: OR breakout now blocked in CHOP
        assert result.signal.action == "HOLD"
        assert result.signal.confidence == 0.0
        assert result.filters_passed is False
        assert result.signal.metadata["chop_guard"]["block_type"] == "or_breakout"

    @pytest.mark.asyncio
    async def test_pullback_hybrid_oppose_meaningful_conf_gets_full_negative_tier(self):
        """EMA21_PB_LONG with hybrid SELL @ 0.50 should get a -0.10 dampen.

        This is the Mar 17 Trade 7180 protection: pullback in elevated VX gets
        reduced confidence, then a meaningful hybrid disagreement pushes it below
        the live 0.60 threshold.
        """
        manager = _make_stub_manager(hybrid_trend="MICRO_UP")
        manager._use_hybrid_pipeline = True

        hybrid_pipeline = AsyncMock()
        hybrid_signal = SimpleNamespace(
            action="SELL",
            confidence=0.50,
            metadata={"market_trend": "MICRO_UP", "volatility_regime": "MEDIUM"},
        )
        pipeline_result = MagicMock()
        pipeline_result.rule_engine = MagicMock()
        pipeline_result.rule_engine.market_trend = "MICRO_UP"
        pipeline_result.rule_engine.volatility_regime = "MEDIUM"
        hybrid_pipeline.process = AsyncMock(return_value=(hybrid_signal, pipeline_result))

        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="EMA21_PB_LONG | ADX=32 | RSI=57 | MACD_H=-0.63 | ATR=9.5",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_price=22.6)
        proc.hybrid_pipeline = hybrid_pipeline

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6782.0,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "BUY"
        # 0.70 - 0.05 VX - 0.10 hybrid = 0.55
        assert abs(result.signal.confidence - 0.55) < 0.01
        assert result.signal.metadata["hybrid_advisory"]["action"] == "SELL"

    @pytest.mark.asyncio
    async def test_pullback_hybrid_oppose_low_conf_stays_light_touch(self):
        """Hybrid oppose below 0.45 should remain capped at -0.05."""
        manager = _make_stub_manager(hybrid_trend="MICRO_UP")
        manager._use_hybrid_pipeline = True

        hybrid_pipeline = AsyncMock()
        hybrid_signal = SimpleNamespace(
            action="SELL",
            confidence=0.40,
            metadata={"market_trend": "MICRO_UP", "volatility_regime": "MEDIUM"},
        )
        pipeline_result = MagicMock()
        pipeline_result.rule_engine = MagicMock()
        pipeline_result.rule_engine.market_trend = "MICRO_UP"
        pipeline_result.rule_engine.volatility_regime = "MEDIUM"
        hybrid_pipeline.process = AsyncMock(return_value=(hybrid_signal, pipeline_result))

        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="EMA21_PB_LONG | ADX=32 | RSI=57 | MACD_H=-0.63 | ATR=9.5",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_price=22.6)
        proc.hybrid_pipeline = hybrid_pipeline

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6782.0,
            structural_metrics=None,
        )

        assert result is not None
        # 0.70 - 0.05 VX - 0.05 hybrid = 0.60
        assert abs(result.signal.confidence - 0.60) < 0.01

    @pytest.mark.asyncio
    async def test_extreme_vx_breakout_mild_reduction(self):
        """OR_BREAK_SHORT with VX=38 (extreme) -> only -0.03 reduction."""
        manager = _make_stub_manager()
        engine = MagicMock()
        signal = _make_signal(
            action="SELL",
            confidence=0.70,
            reason="OR_BREAK_SHORT | ADX=24 | OR_L=6867.50",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_price=38.0)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6865.75,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "SELL"
        # VX >= 35 + breakout -> -0.03: 0.70 - 0.03 = 0.67
        assert abs(result.signal.confidence - 0.67) < 0.01

    @pytest.mark.asyncio
    async def test_extreme_vx_pullback_large_reduction(self):
        """EMA21_PB_LONG with VX=38 (extreme) -> -0.15 reduction."""
        manager = _make_stub_manager()
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="EMA21_PB_LONG | ADX=25 | RSI=55",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_price=38.0)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6890.00,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "BUY"
        # VX >= 35 + pullback -> -0.15: 0.70 - 0.15 = 0.55
        assert abs(result.signal.confidence - 0.55) < 0.01

    @pytest.mark.asyncio
    async def test_elevated_vx_trend_continuation_boosted(self):
        """TREND_CONT_LONG with VX=25 (elevated) -> +0.05 BOOST."""
        manager = _make_stub_manager()
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="TREND_CONT_LONG | ADX=30 | RSI=58",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_price=25.0)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6910.00,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "BUY"
        # VX 22-28 + trend_cont -> +0.05 boost: 0.70 + 0.05 = 0.75
        assert abs(result.signal.confidence - 0.75) < 0.01

    @pytest.mark.asyncio
    async def test_vx_feed_unavailable_no_adjustment(self):
        """When VX feed is unavailable, no VX adjustment is applied."""
        manager = _make_stub_manager()
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="EMA21_PB_LONG | ADX=25 | RSI=55",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_price=None)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6890.00,
            structural_metrics=None,
        )

        assert result is not None
        # No VX feed -> confidence unchanged
        assert result.signal.confidence == 0.70

    @pytest.mark.asyncio
    async def test_high_fear_breakout_neutral(self):
        """OR_BREAK_LONG with VX=32 (high fear) -> breakout gets 0.0 (neutral)."""
        manager = _make_stub_manager()
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="OR_BREAK_LONG | ADX=22 | OR_H=6871.75",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine, vx_price=32.0)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6875.00,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "BUY"
        # VX 28-35 + breakout -> 0.0 (neutral): confidence stays at 0.70
        assert result.signal.confidence == 0.70
