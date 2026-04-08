"""Tests for Apr 8 2026 loss-prevention guards.

Three guards added after the Apr 8 MES losses:
1. RAG min sample guard — require n>=5 before allowing confidence boost
2. Same-day OR breakout reversal guard — block OR_BREAK after prior OR_BREAK loss
3. Price-confirms-breakout guard — validate live price is on breakout side

See root cause analysis in commit message for details.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from shree.execution.components.signal_processor import (
    SignalGenerationResult,
    SignalProcessor,
)


# ---------------------------------------------------------------------------
# Stubs
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


def _make_signal(action="BUY", confidence=0.70, reason="OR_BREAK_LONG | ADX=21 | OR_H=6824.25"):
    return SimpleNamespace(
        action=action,
        confidence=confidence,
        metadata={"reason": reason},
    )


def _make_features():
    return pd.DataFrame({
        "high": [6920.0, 6890.0, 6830.0],
        "low": [6910.0, 6870.0, 6820.0],
        "close": [6918.0, 6877.0, 6829.25],
        "open": [6915.0, 6890.0, 6825.0],
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
    manager._open_trade_context = None
    return manager


def _make_processor(manager=None, engine=None):
    """Create a SignalProcessor with mocked internals."""
    proc = object.__new__(SignalProcessor)
    proc.manager = manager or _make_stub_manager()
    proc.engine = engine or MagicMock()
    proc.hybrid_pipeline = None
    proc._multi_source_enabled = False
    proc._stocktwits_enabled = False
    proc._vx_feed_enabled = False
    proc._vx_feed = None
    proc.settings = MagicMock()
    proc._emergency_generator = MagicMock()
    proc._original_signal_reason = ""

    # APR 8 2026: OR break reversal guard state
    proc._or_break_loss_directions = []
    proc._or_break_loss_date = None

    # Mock the _sentiment helper
    sentiment_mock = MagicMock()
    sentiment_mock._vx_feed_enabled = False
    sentiment_mock._vx_feed = None
    sentiment_mock.get_vx_multiplier.return_value = 1.0
    proc._sentiment = sentiment_mock
    return proc


# ---------------------------------------------------------------------------
# 1. RAG min sample guard
# ---------------------------------------------------------------------------

class TestRagMinSampleGuard:
    """RAG agreement boost requires n>=5 samples."""

    @pytest.mark.asyncio
    async def test_rag_n2_no_boost(self):
        """With only n=2 RAG samples, even 100% win rate should NOT boost."""
        manager = _make_stub_manager()
        signal = _make_signal(
            action="SELL",
            confidence=0.70,
            reason="OR_BREAK_SHORT | ADX=33 | OR_L=6815.00",
        )
        engine = MagicMock()
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)

        # Mock hybrid pipeline that AGREES with SELL
        rag_retrieval = SimpleNamespace(
            similar_trade_count=2,
            weighted_win_rate=1.0,  # 100% win rate
        )
        pipeline_result = SimpleNamespace(
            rule_engine=SimpleNamespace(
                filters_passed=[],
                indicators={"market_trend": "DOWN"},
                market_trend="DOWN",
                volatility_regime="MEDIUM",
            ),
            rag_retrieval=rag_retrieval,
            stop_loss=6.0,
            take_profit=8.0,
            position_size=1.0,
        )
        hybrid_signal = SimpleNamespace(action="SELL", confidence=0.53)

        # Inject hybrid pipeline mock
        proc.hybrid_pipeline = MagicMock()
        proc.hybrid_pipeline.process = AsyncMock(
            return_value=(hybrid_signal, pipeline_result)
        )
        proc.manager._use_hybrid_pipeline = True
        proc.manager.executor = AsyncMock()
        proc.manager.executor.get_current_position = AsyncMock(return_value=None)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6814.50,
            structural_metrics=None,
        )

        assert result is not None
        # With n=2, no boost should be applied.
        # Confidence should remain at base 0.70 (no sentiment overlay in test)
        assert result.signal.confidence == pytest.approx(0.70, abs=0.01)
        assert result.signal.action == "SELL"

    @pytest.mark.asyncio
    async def test_rag_n5_gets_boost(self):
        """With n=5 RAG samples, agreement boost IS applied."""
        manager = _make_stub_manager()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="OR_BREAK_LONG | ADX=21 | OR_H=6824.25",
        )
        engine = MagicMock()
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)

        rag_retrieval = SimpleNamespace(
            similar_trade_count=5,
            weighted_win_rate=0.60,
        )
        pipeline_result = SimpleNamespace(
            rule_engine=SimpleNamespace(
                filters_passed=[],
                indicators={"market_trend": "UP"},
                market_trend="UP",
                volatility_regime="MEDIUM",
            ),
            rag_retrieval=rag_retrieval,
            stop_loss=6.0,
            take_profit=8.0,
            position_size=1.0,
        )
        hybrid_signal = SimpleNamespace(action="BUY", confidence=0.53)

        proc.hybrid_pipeline = MagicMock()
        proc.hybrid_pipeline.process = AsyncMock(
            return_value=(hybrid_signal, pipeline_result)
        )
        proc.manager._use_hybrid_pipeline = True
        proc.manager.executor = AsyncMock()
        proc.manager.executor.get_current_position = AsyncMock(return_value=None)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6829.25,
            structural_metrics=None,
        )

        assert result is not None
        # With n=5, boost should be applied: min(0.15, 0.53 * 0.60 * 0.4) = 0.127
        # Final = 0.70 + 0.127 = 0.827
        assert result.signal.confidence > 0.70
        assert result.signal.action == "BUY"

    @pytest.mark.asyncio
    async def test_rag_n4_no_boost(self):
        """With n=4 samples (below threshold of 5), no boost."""
        manager = _make_stub_manager()
        signal = _make_signal(
            action="SELL",
            confidence=0.70,
            reason="OR_BREAK_SHORT | ADX=30 | OR_L=6800.00",
        )
        engine = MagicMock()
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)

        rag_retrieval = SimpleNamespace(
            similar_trade_count=4,
            weighted_win_rate=0.80,
        )
        pipeline_result = SimpleNamespace(
            rule_engine=SimpleNamespace(
                filters_passed=[],
                indicators={"market_trend": "DOWN"},
                market_trend="DOWN",
                volatility_regime="MEDIUM",
            ),
            rag_retrieval=rag_retrieval,
            stop_loss=6.0,
            take_profit=8.0,
            position_size=1.0,
        )
        hybrid_signal = SimpleNamespace(action="SELL", confidence=0.53)

        proc.hybrid_pipeline = MagicMock()
        proc.hybrid_pipeline.process = AsyncMock(
            return_value=(hybrid_signal, pipeline_result)
        )
        proc.manager._use_hybrid_pipeline = True
        proc.manager.executor = AsyncMock()
        proc.manager.executor.get_current_position = AsyncMock(return_value=None)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6798.00,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.confidence == pytest.approx(0.70, abs=0.01)


# ---------------------------------------------------------------------------
# 2. Same-day OR breakout reversal guard
# ---------------------------------------------------------------------------

class TestOrBreakReversalGuard:
    """After an OR_BREAK loss, block subsequent OR_BREAK signals same day."""

    @pytest.mark.asyncio
    async def test_or_break_blocked_after_opposite_loss(self):
        """SHORT OR_BREAK loss → BUY OR_BREAK should be BLOCKED."""
        manager = _make_stub_manager()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="OR_BREAK_LONG | ADX=21 | OR_H=6824.25",
        )
        engine = MagicMock()
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)

        # Simulate prior SHORT OR_BREAK loss earlier today
        from shree.utils.timezone_utils import now_cst
        proc._or_break_loss_date = now_cst().strftime("%Y-%m-%d")
        proc._or_break_loss_directions = ["SHORT"]

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6829.25,
            structural_metrics=None,
        )

        assert result is not None
        # Confidence should be zeroed out
        assert result.signal.confidence == 0.0
        assert result.filters_passed is False

    @pytest.mark.asyncio
    async def test_or_break_blocked_after_same_direction_loss(self):
        """LONG OR_BREAK loss → another LONG OR_BREAK should also be BLOCKED."""
        manager = _make_stub_manager()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="OR_BREAK_LONG | ADX=25 | OR_H=6830.00",
        )
        engine = MagicMock()
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)

        from shree.utils.timezone_utils import now_cst
        proc._or_break_loss_date = now_cst().strftime("%Y-%m-%d")
        proc._or_break_loss_directions = ["LONG"]

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6835.00,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.confidence == 0.0
        assert result.filters_passed is False

    @pytest.mark.asyncio
    async def test_or_break_allowed_no_prior_loss(self):
        """Without any prior OR_BREAK loss, signal should pass normally."""
        manager = _make_stub_manager()
        signal = _make_signal(
            action="SELL",
            confidence=0.70,
            reason="OR_BREAK_SHORT | ADX=33 | OR_L=6815.00",
        )
        engine = MagicMock()
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)

        # No prior losses
        proc._or_break_loss_directions = []
        proc._or_break_loss_date = None

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6814.50,
            structural_metrics=None,
        )

        assert result is not None
        # Should have base confidence (may have VX adjustment but no block)
        assert result.signal.confidence > 0
        assert result.signal.action == "SELL"

    @pytest.mark.asyncio
    async def test_or_break_allowed_on_new_day(self):
        """Prior day's OR_BREAK loss should NOT block today's signal."""
        manager = _make_stub_manager()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="OR_BREAK_LONG | ADX=22 | OR_H=6820.00",
        )
        engine = MagicMock()
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)

        # Loss was yesterday
        proc._or_break_loss_date = "2020-01-01"
        proc._or_break_loss_directions = ["SHORT"]

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6825.00,
            structural_metrics=None,
        )

        assert result is not None
        assert result.signal.confidence > 0
        assert result.signal.action == "BUY"

    @pytest.mark.asyncio
    async def test_non_or_break_allowed_after_or_loss(self):
        """Non-OR_BREAK signals (e.g. EMA pullback) should NOT be blocked
        by prior OR_BREAK losses."""
        manager = _make_stub_manager()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="EMA21_PB_LONG | RSI=42",
        )
        engine = MagicMock()
        engine.evaluate.return_value = signal
        proc = _make_processor(manager=manager, engine=engine)

        from shree.utils.timezone_utils import now_cst
        proc._or_break_loss_date = now_cst().strftime("%Y-%m-%d")
        proc._or_break_loss_directions = ["SHORT"]

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6825.00,
            structural_metrics=None,
        )

        assert result is not None
        # EMA pullback should pass — not an OR_BREAK
        assert result.signal.confidence > 0
        assert result.signal.action == "BUY"

    def test_notify_position_closed_tracks_or_loss(self):
        """notify_position_closed should record OR_BREAK losses."""
        manager = _make_stub_manager()
        manager._open_trade_context = {
            "signal_type": "OR_BREAK_SHORT",
            "action": "SELL",
        }
        proc = _make_processor(manager=manager)

        # Mock the MTF gate delegation
        proc._mtf = MagicMock()

        proc.notify_position_closed(
            close_reason="SL",
            direction="SHORT",
            pnl=-35.62,
        )

        assert "SHORT" in proc._or_break_loss_directions

    def test_notify_position_closed_ignores_non_or_break(self):
        """notify_position_closed should NOT track non-OR_BREAK losses."""
        manager = _make_stub_manager()
        manager._open_trade_context = {
            "signal_type": "EMA21_PB_LONG",
            "action": "BUY",
        }
        proc = _make_processor(manager=manager)
        proc._mtf = MagicMock()

        proc.notify_position_closed(
            close_reason="SL",
            direction="LONG",
            pnl=-25.00,
        )

        assert proc._or_break_loss_directions == []

    def test_notify_position_closed_ignores_wins(self):
        """notify_position_closed should NOT track OR_BREAK wins."""
        manager = _make_stub_manager()
        manager._open_trade_context = {
            "signal_type": "OR_BREAK_LONG",
            "action": "BUY",
        }
        proc = _make_processor(manager=manager)
        proc._mtf = MagicMock()

        proc.notify_position_closed(
            close_reason="TP",
            direction="LONG",
            pnl=40.00,
        )

        assert proc._or_break_loss_directions == []
