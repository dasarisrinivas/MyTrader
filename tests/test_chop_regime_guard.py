"""Tests for the CHOP regime guard (FEB 22 2026).

Validates that pullback signals (EMA21_PB, EMA9_PB) are blocked when
the hybrid pipeline detects a CHOP (range-bound) market regime, while
OR breakout signals are allowed through.

Root cause: Friday 2026-02-20 loss — EMA9_PB_LONG fired in CHOP regime
with 30% RAG win rate. Pullbacks need trending markets.
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
# Minimal stubs to instantiate SignalProcessor without IBKR / full app
# ---------------------------------------------------------------------------

@dataclass
class _StubStatus:
    """Mimics the status attributes the CHOP guard reads."""
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


def _make_features():
    """Build a minimal DataFrame with columns the signal processor expects."""
    return pd.DataFrame({
        "high": [6920.0, 6925.0, 6930.0],
        "low": [6910.0, 6915.0, 6918.0],
        "close": [6918.0, 6922.0, 6920.25],
        "open": [6915.0, 6918.0, 6920.0],
        "volume": [100, 110, 95],
    })


def _make_stub_manager(hybrid_trend="CHOP"):
    """Create a minimal manager mock with status.hybrid_market_trend set."""
    manager = MagicMock()
    manager.status = _StubStatus(hybrid_market_trend=hybrid_trend)
    manager.context_manager = _StubContextManager()
    manager._use_hybrid_pipeline = True
    manager.hybrid_pipeline = None
    manager._current_pipeline_result = None
    manager.executor = None
    return manager


def _make_processor(manager=None, engine=None, hybrid_pipeline=None):
    """Create a SignalProcessor with mocked dependencies."""
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
    # Mock the _sentiment helper so _get_vx_multiplier() works
    sentiment_mock = MagicMock()
    sentiment_mock.get_vx_multiplier.return_value = 1.0
    proc._sentiment = sentiment_mock
    return proc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChopRegimeGuard:
    """CHOP regime guard should block pullback signals in range-bound markets."""

    @pytest.mark.asyncio
    async def test_blocks_ema9_pullback_in_chop(self):
        """EMA9_PB_LONG in CHOP regime → blocked (HOLD)."""
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="EMA9_PB_LONG | ADX=23 | RSI=58 | ATR=14.3",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6920.25,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "HOLD"
        assert result.signal.confidence == 0.0
        assert result.filters_passed is False
        assert "chop_regime_block" in result.filters_applied
        assert "CHOP_REGIME_BLOCK" in result.signal.metadata.get("reason", "")

    @pytest.mark.asyncio
    async def test_blocks_ema21_pullback_in_chop(self):
        """EMA21_PB_LONG in CHOP regime → blocked."""
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="EMA21_PB_LONG | ADX=25 | RSI=55 | MACD_H=1.2 | ATR=12.0",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6920.25,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "HOLD"
        assert result.filters_passed is False

    @pytest.mark.asyncio
    async def test_blocks_short_pullback_in_chop(self):
        """EMA21_PB_SHORT in CHOP regime → blocked."""
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(
            action="SELL",
            confidence=0.70,
            reason="EMA21_PB_SHORT | ADX=25 | RSI=35 | MACD_H=-1.5 | ATR=12.0",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6920.25,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "HOLD"
        assert result.filters_passed is False

    @pytest.mark.asyncio
    async def test_allows_or_breakout_in_chop(self):
        """OR_BREAK_LONG in CHOP regime → ALLOWED (breakouts are exempt)."""
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="OR_BREAK_LONG | ADX=25 | OR_H=6930.00",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6920.25,
            structural_metrics=None,
        )

        assert result is not None
        # OR breakout should NOT be blocked — no "_PB_" in reason
        assert result.signal.action == "BUY"
        assert result.signal.confidence > 0.0

    @pytest.mark.asyncio
    async def test_allows_pullback_in_uptrend(self):
        """EMA9_PB_LONG in UPTREND regime → ALLOWED (pullbacks need trends)."""
        manager = _make_stub_manager(hybrid_trend="UPTREND")
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="EMA9_PB_LONG | ADX=30 | RSI=58 | ATR=10.0",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6920.25,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "BUY"
        assert result.signal.confidence > 0.0

    @pytest.mark.asyncio
    async def test_allows_pullback_when_no_hybrid_trend(self):
        """Pullback with no hybrid trend data → ALLOWED (fail open)."""
        manager = _make_stub_manager(hybrid_trend=None)
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="EMA9_PB_LONG | ADX=25 | RSI=55 | ATR=12.0",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6920.25,
            structural_metrics=None,
        )

        assert result is not None
        # No hybrid data — fail open, let trade through
        assert result.signal.action == "BUY"
        assert result.signal.confidence > 0.0

    @pytest.mark.asyncio
    async def test_hold_signals_not_affected(self):
        """HOLD signal should pass through regardless of CHOP regime."""
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(action="HOLD", confidence=0.0, reason="No setup")
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6920.25,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "HOLD"
        # HOLD returns early before CHOP guard — filters_passed should be True
        assert result.filters_passed is True

    @pytest.mark.asyncio
    async def test_chop_guard_metadata_preserved(self):
        """Blocked signal should carry original info in metadata."""
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.72,
            reason="EMA9_PB_LONG | ADX=23 | RSI=58 | ATR=14.3",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6920.25,
            structural_metrics=None,
        )

        meta = result.signal.metadata
        assert "chop_guard" in meta
        assert meta["chop_guard"]["original_action"] == "BUY"
        assert meta["chop_guard"]["original_confidence"] == 0.72
        assert meta["chop_guard"]["hybrid_trend"] == "CHOP"

    @pytest.mark.asyncio
    async def test_allows_trend_continuation_in_chop(self):
        """Trend continuation signals don't have _PB_ → should pass through CHOP guard."""
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        # Signal F: Trend continuation — reason starts with "F:" not "_PB_"
        signal = _make_signal(
            action="BUY",
            confidence=0.65,
            reason="F:stack(e9=6910,e21=6901,e50=6894)",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6920.25,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.action == "BUY"
