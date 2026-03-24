"""Tests for the CHOP regime guard (block-all with optional exception framework).

History:
  FEB 22 2026: Original binary block for all pullbacks in CHOP.
  FEB 24 2026: Extended to block TREND_CONT in CHOP.
  MAR 2 2026a: Direction-aware redesign (LONGs dampen, SHORTs block).
  MAR 2 2026b: REVERTED to block-all after 1-year backtest (25,646 bars,
               226 CHOP-blocked trades) showed LONGs also net losers:
                 LONGs:  26.5% WR, -$146.25, PF 0.90
                 SHORTs: 25.0% WR, -$421.25, PF 0.79
               CIs overlap — no statistically significant direction edge.
               Block-all ($0) > direction-aware (-$146.25) > blind (-$567.50).

  Optional exception framework (ENABLE_CHOP_EXCEPTION env var, OFF by default)
  allows LONG-only exceptions through a 5-gate filter:
    1. Direction == LONG
    2. ADX >= 25
    3. Daily sentiment bias == BULLISH
    4. Confidence >= 0.70
    5. ATR expanding (current ATR > ATR 5 bars ago)
"""
from __future__ import annotations

import os
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


@dataclass
class _StubRuleEngine:
    filters_passed: List[str] = field(default_factory=list)
    filters_warned: List[str] = field(default_factory=list)
    filters_blocked: List[str] = field(default_factory=list)
    indicators: Dict[str, Any] = field(default_factory=dict)
    market_trend: str = "CHOP"
    volatility_regime: str = "MEDIUM"


@dataclass
class _StubPipelineResult:
    rule_engine: _StubRuleEngine = field(default_factory=_StubRuleEngine)
    stop_loss: float = 0.0
    take_profit: float = 0.0


def _make_signal(action="BUY", confidence=0.70, reason="EMA9_PB_LONG | ADX=23 | RSI=58 | ATR=14.3"):
    return SimpleNamespace(
        action=action,
        confidence=confidence,
        metadata={"reason": reason},
    )


def _make_features(atr_expanding=False):
    if atr_expanding:
        atr_values = [8.0, 8.5, 9.0, 9.5, 10.0, 11.0]
    else:
        atr_values = [12.0, 11.5, 11.0, 10.5, 10.0, 9.5]
    return pd.DataFrame({
        "high": [6920.0, 6925.0, 6930.0, 6928.0, 6935.0, 6940.0],
        "low": [6910.0, 6915.0, 6918.0, 6916.0, 6920.0, 6925.0],
        "close": [6918.0, 6922.0, 6920.25, 6926.0, 6932.0, 6938.0],
        "open": [6915.0, 6918.0, 6920.0, 6921.0, 6925.0, 6930.0],
        "volume": [100, 110, 95, 105, 115, 120],
        "ATR_14": atr_values,
    })


def _make_stub_manager(hybrid_trend="CHOP", sentiment_bias="NEUTRAL"):
    manager = MagicMock()
    manager.status = _StubStatus(hybrid_market_trend=hybrid_trend)
    manager.context_manager = _StubContextManager()
    manager._use_hybrid_pipeline = True
    manager.hybrid_pipeline = None
    manager._current_pipeline_result = None
    manager.executor = None
    manager._last_sentiment_bias = sentiment_bias
    return manager


def _make_processor(manager=None, engine=None, hybrid_pipeline=None):
    proc = object.__new__(SignalProcessor)
    proc.manager = manager or _make_stub_manager()
    proc.engine = engine or MagicMock()
    proc.hybrid_pipeline = hybrid_pipeline
    proc._multi_source_enabled = False
    proc._stocktwits_enabled = False
    proc._vx_feed_enabled = False
    proc._vx_feed = None
    proc.settings = MagicMock()
    proc._emergency_generator = MagicMock()
    sentiment_mock = MagicMock()
    sentiment_mock.get_vx_multiplier.return_value = 1.0
    proc._sentiment = sentiment_mock
    return proc


# ---------------------------------------------------------------------------
# Block-All Tests (default, ENABLE_CHOP_EXCEPTION=off)
# ---------------------------------------------------------------------------

class TestChopRegimeGuardBlockAll:
    """Default: block ALL pullback/trend_cont in CHOP."""

    @pytest.mark.asyncio
    async def test_blocks_ema9_pullback_long_in_chop(self):
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(action="BUY", confidence=0.70, reason="EMA9_PB_LONG | ADX=23 | RSI=58 | ATR=14.3")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        with patch.dict(os.environ, {"ENABLE_CHOP_EXCEPTION": ""}, clear=False):
            result = await proc._generate_strategy_first_signal(features=_make_features(), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "HOLD"
        assert result.signal.confidence == 0.0
        assert result.filters_passed is False
        assert "chop_regime_block" in result.filters_applied

    @pytest.mark.asyncio
    async def test_blocks_ema21_pullback_long_in_chop(self):
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(action="BUY", confidence=0.70, reason="EMA21_PB_LONG | ADX=25 | RSI=55 | MACD_H=1.2 | ATR=12.0")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        with patch.dict(os.environ, {"ENABLE_CHOP_EXCEPTION": ""}, clear=False):
            result = await proc._generate_strategy_first_signal(features=_make_features(), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "HOLD"
        assert result.signal.confidence == 0.0
        assert result.filters_passed is False

    @pytest.mark.asyncio
    async def test_blocks_short_pullback_in_chop(self):
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(action="SELL", confidence=0.70, reason="EMA21_PB_SHORT | ADX=25 | RSI=35 | MACD_H=-1.5 | ATR=12.0")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        result = await proc._generate_strategy_first_signal(features=_make_features(), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "HOLD"
        assert result.filters_passed is False

    @pytest.mark.asyncio
    async def test_blocks_scalp_buy_in_chop(self):
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(action="SCALP_BUY", confidence=0.65, reason="EMA21_PB_LONG | ADX=28 | RSI=52 | ATR=11.0")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        with patch.dict(os.environ, {"ENABLE_CHOP_EXCEPTION": ""}, clear=False):
            result = await proc._generate_strategy_first_signal(features=_make_features(), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "HOLD"
        assert result.signal.confidence == 0.0
        assert result.filters_passed is False

    @pytest.mark.asyncio
    async def test_blocks_scalp_sell_in_chop(self):
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(action="SCALP_SELL", confidence=0.65, reason="EMA21_PB_SHORT | ADX=28 | RSI=42 | ATR=11.0")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        result = await proc._generate_strategy_first_signal(features=_make_features(), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "HOLD"
        assert result.signal.confidence == 0.0
        assert result.filters_passed is False

    @pytest.mark.asyncio
    async def test_blocks_trend_cont_long_in_chop(self):
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(action="BUY", confidence=0.70, reason="TREND_CONT_LONG | ADX=18 | RSI=65 | MACD_H=3.53 | e9=6887.5 | #1")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        from datetime import datetime
        mock_time = datetime(2026, 3, 2, 9, 30)
        with patch("shree.execution.components.signal_processor.now_cst", return_value=mock_time), \
             patch.dict(os.environ, {"ENABLE_CHOP_EXCEPTION": ""}, clear=False):
            result = await proc._generate_strategy_first_signal(features=_make_features(), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "HOLD"
        assert result.signal.confidence == 0.0
        assert result.filters_passed is False

    @pytest.mark.asyncio
    async def test_blocks_trend_cont_short_in_chop(self):
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(action="SELL", confidence=0.70, reason="TREND_CONT_SHORT | ADX=20 | RSI=35 | MACD_H=-2.1 | e9=6910 | #1")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        result = await proc._generate_strategy_first_signal(features=_make_features(), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "HOLD"
        assert result.filters_passed is False
        assert result.signal.metadata["chop_guard"]["block_type"] == "trend_cont"

    @pytest.mark.asyncio
    async def test_blocks_or_breakout_in_chop(self):
        """MAR 24 2026: OR breakouts now blocked in CHOP.
        Evidence: 5/5 executed OR breakout fills were SL_HIT losers (−$180.87).
        Breakouts in CHOP are failed breakouts — price crosses OR level but
        cannot sustain, reverting into the range."""
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(action="BUY", confidence=0.70, reason="OR_BREAK_LONG | ADX=25 | OR_H=6930.00")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        result = await proc._generate_strategy_first_signal(features=_make_features(), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "HOLD"
        assert result.signal.confidence == 0.0
        assert result.filters_passed is False
        assert result.signal.metadata["chop_guard"]["block_type"] == "or_breakout"

    @pytest.mark.asyncio
    async def test_blocks_or_breakdown_short_in_chop(self):
        """MAR 24 2026: OR breakdowns also blocked in CHOP."""
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(action="SELL", confidence=0.70, reason="OR_BREAK_SHORT | ADX=25 | OR_L=6900.00")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        result = await proc._generate_strategy_first_signal(features=_make_features(), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "HOLD"
        assert result.signal.confidence == 0.0
        assert result.filters_passed is False
        assert result.signal.metadata["chop_guard"]["block_type"] == "or_breakout"

    @pytest.mark.asyncio
    async def test_allows_pullback_in_uptrend(self):
        manager = _make_stub_manager(hybrid_trend="UPTREND")
        engine = MagicMock()
        signal = _make_signal(action="BUY", confidence=0.70, reason="EMA9_PB_LONG | ADX=30 | RSI=58 | ATR=10.0")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        result = await proc._generate_strategy_first_signal(features=_make_features(), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "BUY"
        assert result.signal.confidence > 0.0

    @pytest.mark.asyncio
    async def test_allows_pullback_when_no_hybrid_trend(self):
        manager = _make_stub_manager(hybrid_trend=None)
        engine = MagicMock()
        signal = _make_signal(action="BUY", confidence=0.70, reason="EMA9_PB_LONG | ADX=25 | RSI=55 | ATR=12.0")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        result = await proc._generate_strategy_first_signal(features=_make_features(), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "BUY"
        assert result.signal.confidence > 0.0

    @pytest.mark.asyncio
    async def test_hold_signals_not_affected(self):
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(action="HOLD", confidence=0.0, reason="No setup")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        result = await proc._generate_strategy_first_signal(features=_make_features(), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "HOLD"
        assert result.filters_passed is True

    @pytest.mark.asyncio
    async def test_block_metadata_preserved(self):
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(action="BUY", confidence=0.72, reason="EMA9_PB_LONG | ADX=23 | RSI=58 | ATR=14.3")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        with patch.dict(os.environ, {"ENABLE_CHOP_EXCEPTION": ""}, clear=False):
            result = await proc._generate_strategy_first_signal(features=_make_features(), returns=None, current_price=6920.25, structural_metrics=None)
        meta = result.signal.metadata
        assert "chop_guard" in meta
        assert meta["chop_guard"]["original_action"] == "BUY"
        assert meta["chop_guard"]["original_confidence"] == 0.72
        assert meta["chop_guard"]["hybrid_trend"] == "CHOP"
        assert result.signal.action == "HOLD"
        assert result.signal.confidence == 0.0

    @pytest.mark.asyncio
    async def test_allows_non_categorized_signal_in_chop(self):
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(action="BUY", confidence=0.65, reason="F:stack(e9=6910,e21=6901,e50=6894)")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        result = await proc._generate_strategy_first_signal(features=_make_features(), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "BUY"


# ---------------------------------------------------------------------------
# Exception Framework Tests (ENABLE_CHOP_EXCEPTION=1)
# ---------------------------------------------------------------------------

class TestChopExceptionFramework:
    """Optional CHOP exception: LONG-only, 5-gate filter, OFF by default."""

    @pytest.mark.asyncio
    async def test_exception_allows_long_when_all_gates_pass(self):
        """All 5 gates satisfied → LONG allowed with -0.05 dampen."""
        manager = _make_stub_manager(hybrid_trend="CHOP", sentiment_bias="BULLISH")
        engine = MagicMock()
        signal = _make_signal(action="BUY", confidence=0.75, reason="EMA21_PB_LONG | ADX=28 | RSI=55 | MACD_H=1.2 | ATR=12.0")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        with patch.dict(os.environ, {"ENABLE_CHOP_EXCEPTION": "1"}, clear=False):
            result = await proc._generate_strategy_first_signal(features=_make_features(atr_expanding=True), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "BUY"
        assert result.signal.confidence == pytest.approx(0.70, abs=0.01)
        assert "chop_exception_pass" in result.filters_applied
        meta = result.signal.metadata
        assert meta["chop_guard"]["exception_activated"] is True
        assert meta["chop_guard"]["exception_adx"] == 28.0
        assert meta["chop_guard"]["exception_bias"] == "BULLISH"

    @pytest.mark.asyncio
    async def test_exception_blocks_short_even_when_enabled(self):
        """SHORT → BLOCKED even with exception enabled (LONGs only)."""
        manager = _make_stub_manager(hybrid_trend="CHOP", sentiment_bias="BULLISH")
        engine = MagicMock()
        signal = _make_signal(action="SELL", confidence=0.75, reason="EMA21_PB_SHORT | ADX=28 | RSI=35 | ATR=12.0")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        with patch.dict(os.environ, {"ENABLE_CHOP_EXCEPTION": "1"}, clear=False):
            result = await proc._generate_strategy_first_signal(features=_make_features(atr_expanding=True), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "HOLD"
        assert result.signal.confidence == 0.0

    @pytest.mark.asyncio
    async def test_exception_blocks_when_adx_below_25(self):
        """ADX < 25 → gate 1 fails → BLOCKED."""
        manager = _make_stub_manager(hybrid_trend="CHOP", sentiment_bias="BULLISH")
        engine = MagicMock()
        signal = _make_signal(action="BUY", confidence=0.75, reason="EMA21_PB_LONG | ADX=22 | RSI=55 | ATR=12.0")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        with patch.dict(os.environ, {"ENABLE_CHOP_EXCEPTION": "1"}, clear=False):
            result = await proc._generate_strategy_first_signal(features=_make_features(atr_expanding=True), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "HOLD"
        assert result.signal.confidence == 0.0

    @pytest.mark.asyncio
    async def test_exception_blocks_when_bias_not_bullish(self):
        """NEUTRAL bias → gate 2 fails → BLOCKED."""
        manager = _make_stub_manager(hybrid_trend="CHOP", sentiment_bias="NEUTRAL")
        engine = MagicMock()
        signal = _make_signal(action="BUY", confidence=0.75, reason="EMA21_PB_LONG | ADX=28 | RSI=55 | ATR=12.0")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        with patch.dict(os.environ, {"ENABLE_CHOP_EXCEPTION": "1"}, clear=False):
            result = await proc._generate_strategy_first_signal(features=_make_features(atr_expanding=True), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "HOLD"
        assert result.signal.confidence == 0.0

    @pytest.mark.asyncio
    async def test_exception_blocks_when_confidence_below_070(self):
        """conf < 0.70 → gate 3 fails → BLOCKED."""
        manager = _make_stub_manager(hybrid_trend="CHOP", sentiment_bias="BULLISH")
        engine = MagicMock()
        signal = _make_signal(action="BUY", confidence=0.65, reason="EMA21_PB_LONG | ADX=28 | RSI=55 | ATR=12.0")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        with patch.dict(os.environ, {"ENABLE_CHOP_EXCEPTION": "1"}, clear=False):
            result = await proc._generate_strategy_first_signal(features=_make_features(atr_expanding=True), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "HOLD"
        assert result.signal.confidence == 0.0

    @pytest.mark.asyncio
    async def test_exception_blocks_when_atr_not_expanding(self):
        """ATR declining → gate 4 fails → BLOCKED."""
        manager = _make_stub_manager(hybrid_trend="CHOP", sentiment_bias="BULLISH")
        engine = MagicMock()
        signal = _make_signal(action="BUY", confidence=0.75, reason="EMA21_PB_LONG | ADX=28 | RSI=55 | ATR=12.0")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        with patch.dict(os.environ, {"ENABLE_CHOP_EXCEPTION": "1"}, clear=False):
            result = await proc._generate_strategy_first_signal(features=_make_features(atr_expanding=False), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "HOLD"
        assert result.signal.confidence == 0.0

    @pytest.mark.asyncio
    async def test_exception_off_by_default(self):
        """No env var → block all even when gates would pass."""
        manager = _make_stub_manager(hybrid_trend="CHOP", sentiment_bias="BULLISH")
        engine = MagicMock()
        signal = _make_signal(action="BUY", confidence=0.75, reason="EMA21_PB_LONG | ADX=28 | RSI=55 | ATR=12.0")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        env = os.environ.copy()
        env.pop("ENABLE_CHOP_EXCEPTION", None)
        with patch.dict(os.environ, env, clear=True):
            result = await proc._generate_strategy_first_signal(features=_make_features(atr_expanding=True), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "HOLD"
        assert result.signal.confidence == 0.0
        assert result.filters_passed is False

    @pytest.mark.asyncio
    async def test_exception_trend_cont_long_passes_all_gates(self):
        """TREND_CONT_LONG with all gates → allowed when exception on."""
        manager = _make_stub_manager(hybrid_trend="CHOP", sentiment_bias="BULLISH")
        engine = MagicMock()
        signal = _make_signal(action="BUY", confidence=0.72, reason="TREND_CONT_LONG | ADX=30 | RSI=60 | MACD_H=5.0 | e9=6900 | #1")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        from datetime import datetime
        mock_time = datetime(2026, 3, 2, 9, 30)
        with patch("shree.execution.components.signal_processor.now_cst", return_value=mock_time), \
             patch.dict(os.environ, {"ENABLE_CHOP_EXCEPTION": "1"}, clear=False):
            result = await proc._generate_strategy_first_signal(features=_make_features(atr_expanding=True), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "BUY"
        assert result.signal.confidence == pytest.approx(0.67, abs=0.01)
        meta = result.signal.metadata
        assert meta["chop_guard"]["exception_activated"] is True
        assert meta["chop_guard"]["block_type"] == "trend_cont"

    @pytest.mark.asyncio
    async def test_exception_does_not_apply_to_or_breakout(self):
        """MAR 24 2026: OR breakout in CHOP is blocked even with exception
        enabled and all gates passing. The 4-gate exception framework is
        for directional pullbacks, not range-bound breakouts."""
        manager = _make_stub_manager(hybrid_trend="CHOP", sentiment_bias="BULLISH")
        engine = MagicMock()
        signal = _make_signal(action="BUY", confidence=0.75, reason="OR_BREAK_LONG | ADX=28 | OR_H=6930.00")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        with patch.dict(os.environ, {"ENABLE_CHOP_EXCEPTION": "1"}, clear=False):
            result = await proc._generate_strategy_first_signal(features=_make_features(atr_expanding=True), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "HOLD"
        assert result.signal.confidence == 0.0
        assert result.filters_passed is False
        assert result.signal.metadata["chop_guard"]["block_type"] == "or_breakout"

    @pytest.mark.asyncio
    async def test_exception_bearish_bias_blocks(self):
        """BEARISH bias → gate 2 fails → BLOCKED."""
        manager = _make_stub_manager(hybrid_trend="CHOP", sentiment_bias="BEARISH")
        engine = MagicMock()
        signal = _make_signal(action="BUY", confidence=0.75, reason="EMA21_PB_LONG | ADX=28 | RSI=55 | ATR=12.0")
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)
        with patch.dict(os.environ, {"ENABLE_CHOP_EXCEPTION": "1"}, clear=False):
            result = await proc._generate_strategy_first_signal(features=_make_features(atr_expanding=True), returns=None, current_price=6920.25, structural_metrics=None)
        assert result.signal.action == "HOLD"
        assert result.signal.confidence == 0.0
