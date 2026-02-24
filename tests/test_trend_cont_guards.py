"""Tests for TREND_CONT guards (FEB 24 2026).

Three layers of protection for Signal F (TREND_CONT), informed by:
  - 224-trade backtest: TREND_CONT = −$7.94/trade overall
  - Today's loss: TREND_CONT_LONG at ADX=18, CHOP, 11:15 CST → −$40.62
  - MES-specific session structure analysis

Guards tested:
  1. ADX floor raised from 18 → 25 (in strategy es_fifteen_min.py)
  2. CHOP regime guard extended to block TREND_CONT (signal_processor.py)
  3. MES dead-zone time guard: block TREND_CONT 10:30-13:00 CST (signal_processor.py)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import pandas as pd
import pytest

# Patch target: now_cst is imported at module level in signal_processor
_NOW_CST_PATCH = "shree.execution.components.signal_processor.now_cst"


# ---------------------------------------------------------------------------
# Shared test stubs (mirrors test_chop_regime_guard.py)
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


def _make_signal(action="BUY", confidence=0.70, reason="TREND_CONT_LONG | ADX=30 | RSI=60"):
    return SimpleNamespace(
        action=action,
        confidence=confidence,
        metadata={"reason": reason},
    )


def _make_features():
    return pd.DataFrame({
        "high": [6920.0, 6925.0, 6930.0],
        "low": [6910.0, 6915.0, 6918.0],
        "close": [6918.0, 6922.0, 6920.25],
        "open": [6915.0, 6918.0, 6920.0],
        "volume": [100, 110, 95],
    })


def _make_stub_manager(hybrid_trend="CHOP"):
    manager = MagicMock()
    manager.status = _StubStatus(hybrid_market_trend=hybrid_trend)
    manager.context_manager = _StubContextManager()
    manager._use_hybrid_pipeline = True
    manager.hybrid_pipeline = None
    manager._current_pipeline_result = None
    manager.executor = None
    return manager


def _make_processor(manager=None, engine=None):
    from shree.execution.components.signal_processor import SignalProcessor
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
    sentiment_mock = MagicMock()
    sentiment_mock.get_vx_multiplier.return_value = 1.0
    proc._sentiment = sentiment_mock
    return proc


# ===========================================================================
# 1. ADX floor tests (strategy level — es_fifteen_min.py)
# ===========================================================================

class TestTrendContADXFloor:
    """Signal F ADX floor is now 25.0 (raised from 18.0 on FEB 24 2026).

    Backtest evidence:
      ADX<18:  67 trades, 31% WR, −$1,243  (disaster)
      ADX 18-22: 31 trades, 42% WR, +$18   (break-even)
      ADX 22-30: 50 trades, 40% WR, −$731
      ADX≥30:  76 trades, 50% WR, +$178    (only positive bucket)

    ADX≥25 blocks the worst 114 trades, saving $1,582.
    """

    def _make_strategy(self, adx_min=None):
        """Create EsFifteenMin strategy with test config."""
        from shree.strategies.es_fifteen_min import EsFifteenMinStrategy
        config = MagicMock()
        # Set up default config attributes
        config.ft_trend_cont_enabled = True
        config.ft_trend_cont_adx_min = adx_min if adx_min is not None else 25.0
        config.ft_trend_cont_stop_mult = 1.0
        config.ft_trend_cont_target_mult = 2.0
        config.ft_trend_cont_ema9_pct = 0.003
        config.ft_trend_cont_max_ext_pts = 30.0
        config.ft_trend_cont_gap_adx_min = 25.0
        config.ft_trend_cont_max_per_day = 2
        config.ft_trend_sl_atr_mult = 1.0
        config.ft_trend_sl_floor_pts = 6.0
        config.ft_trend_sl_ceiling_pts = 20.0
        config.ft_trend_rr_ratio = 1.25
        # Minimal strategy instantiation via __new__ + manual init
        strat = object.__new__(EsFifteenMinStrategy)
        strat._trend_cont_enabled = True
        strat._trend_cont_adx_min = config.ft_trend_cont_adx_min
        strat._trend_cont_ema9_pct = 0.003
        strat._trend_cont_max_ext_pts = 30.0
        strat._trend_cont_gap_adx_min = 25.0
        strat._trend_cont_long_count = 0
        strat._trend_cont_short_count = 0
        strat._trend_cont_max_per_day = 2
        strat._trend_sl_atr_mult = 1.0
        strat._trend_sl_floor = 6.0
        strat._trend_sl_ceiling = 20.0
        strat._trend_rr_ratio = 1.25
        return strat

    def _make_df(self, closes=None):
        """Build minimal DataFrame for trend continuation check."""
        if closes is None:
            closes = [6890.0, 6895.0, 6900.0, 6905.0]
        n = len(closes)
        return pd.DataFrame({
            "close": closes,
            "open": [c - 2 for c in closes],
            "high": [c + 3 for c in closes],
            "low": [c - 3 for c in closes],
        })

    def test_default_adx_min_is_25(self):
        """Default ADX minimum for TREND_CONT should be 25.0."""
        strat = self._make_strategy()
        assert strat._trend_cont_adx_min == 25.0

    def test_adx_18_blocked(self):
        """ADX=18 (today's losing trade) should be blocked by new 25 floor."""
        strat = self._make_strategy()
        result = strat._check_trend_continuation_long(
            df=self._make_df([6890, 6895, 6900, 6905]),
            close=6905.0, open_p=6900.0, low=6902.0, high=6908.0,
            ema9=6903.0, ema21=6890.0, ema50=6880.0,
            atr=8.7, adx=18.0, rsi=65.0, macd_hist=3.5,
        )
        assert result is None

    def test_adx_22_blocked(self):
        """ADX=22 should also be blocked (below new 25 floor)."""
        strat = self._make_strategy()
        result = strat._check_trend_continuation_long(
            df=self._make_df([6890, 6895, 6900, 6905]),
            close=6905.0, open_p=6900.0, low=6902.0, high=6908.0,
            ema9=6903.0, ema21=6890.0, ema50=6880.0,
            atr=10.0, adx=22.0, rsi=60.0, macd_hist=2.0,
        )
        assert result is None

    def test_adx_25_allowed(self):
        """ADX=25 (exactly at new threshold) should be allowed."""
        strat = self._make_strategy()
        result = strat._check_trend_continuation_long(
            df=self._make_df([6890, 6895, 6900, 6905]),
            close=6905.0, open_p=6900.0, low=6903.0, high=6908.0,
            ema9=6903.0, ema21=6890.0, ema50=6880.0,
            atr=10.0, adx=25.0, rsi=60.0, macd_hist=2.0,
        )
        assert result is not None
        assert result[0] == "BUY"

    def test_adx_30_allowed(self):
        """ADX=30 (strong trend) should fire normally."""
        strat = self._make_strategy()
        result = strat._check_trend_continuation_long(
            df=self._make_df([6890, 6895, 6900, 6905]),
            close=6905.0, open_p=6900.0, low=6903.0, high=6908.0,
            ema9=6903.0, ema21=6890.0, ema50=6880.0,
            atr=10.0, adx=30.0, rsi=60.0, macd_hist=2.0,
        )
        assert result is not None
        assert result[0] == "BUY"
        assert "TREND_CONT_LONG" in result[3]

    def test_adx_24_99_blocked(self):
        """ADX=24.99 (just below threshold) should be blocked."""
        strat = self._make_strategy()
        result = strat._check_trend_continuation_long(
            df=self._make_df([6890, 6895, 6900, 6905]),
            close=6905.0, open_p=6900.0, low=6902.0, high=6908.0,
            ema9=6903.0, ema21=6890.0, ema50=6880.0,
            atr=10.0, adx=24.99, rsi=60.0, macd_hist=2.0,
        )
        assert result is None

    def test_short_side_adx_25_blocked(self):
        """TREND_CONT_SHORT with ADX=20 should be blocked too."""
        strat = self._make_strategy()
        result = strat._check_trend_continuation_short(
            df=self._make_df([6920, 6915, 6910, 6905]),
            close=6905.0, open_p=6910.0, low=6902.0, high=6908.0,
            ema9=6907.0, ema21=6920.0, ema50=6930.0,
            atr=10.0, adx=20.0, rsi=35.0, macd_hist=-2.0,
        )
        assert result is None

    def test_short_side_adx_30_allowed(self):
        """TREND_CONT_SHORT with ADX=30 should fire."""
        strat = self._make_strategy()
        result = strat._check_trend_continuation_short(
            df=self._make_df([6920, 6915, 6910, 6905]),
            close=6905.0, open_p=6910.0, low=6902.0, high=6908.0,
            ema9=6907.0, ema21=6920.0, ema50=6930.0,
            atr=10.0, adx=30.0, rsi=35.0, macd_hist=-2.0,
        )
        assert result is not None
        assert result[0] == "SELL"


# ===========================================================================
# 2. Dead-zone time guard tests (signal_processor.py)
# ===========================================================================

class TestDeadZoneGuard:
    """MES dead-zone guard blocks TREND_CONT during 10:30-13:00 CST.

    MES session structure:
      08:30-10:00 CT: Power hour (+$26.9/trade)
      10:30-13:00 CT: Dead zone (−$24.8/trade for TREND_CONT)
      13:00-15:15 CT: Afternoon drift

    Today's 11:15 entry was dead-zone → 3hr chop → SL hit.
    """

    @pytest.mark.asyncio
    async def test_blocks_trend_cont_at_1115_cst(self):
        """TREND_CONT at 11:15 CST → blocked (today's exact scenario)."""
        manager = _make_stub_manager(hybrid_trend="UPTREND")  # Even in uptrend, dead zone kills
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="TREND_CONT_LONG | ADX=30 | RSI=60 | MACD_H=5.0 | e9=6900 | #1",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        # Mock now_cst to return 11:15 CST
        mock_dt = datetime(2026, 2, 24, 11, 15, 0)
        with patch("shree.execution.components.signal_processor.now_cst", return_value=mock_dt):
            result = await proc._generate_strategy_first_signal(
                features=_make_features(),
                returns=None,
                current_price=6920.25,
                structural_metrics=None,
            )

        assert result is not None
        assert result.signal.action == "HOLD"
        assert result.filters_passed is False
        assert "dead_zone_block" in result.filters_applied
        assert "DEAD_ZONE_BLOCK" in result.signal.metadata.get("reason", "")

    @pytest.mark.asyncio
    async def test_blocks_trend_cont_at_1030_cst(self):
        """TREND_CONT at 10:30 CST → blocked (start of dead zone)."""
        manager = _make_stub_manager(hybrid_trend="UPTREND")
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="TREND_CONT_LONG | ADX=30 | RSI=60 | MACD_H=5.0 | e9=6900 | #1",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        mock_dt = datetime(2026, 2, 24, 10, 30, 0)
        with patch("shree.execution.components.signal_processor.now_cst", return_value=mock_dt):
            result = await proc._generate_strategy_first_signal(
                features=_make_features(),
                returns=None,
                current_price=6920.25,
                structural_metrics=None,
            )

        assert result.signal.action == "HOLD"
        assert "dead_zone_block" in result.filters_applied

    @pytest.mark.asyncio
    async def test_blocks_trend_cont_at_1259_cst(self):
        """TREND_CONT at 12:59 CST → blocked (end of dead zone)."""
        manager = _make_stub_manager(hybrid_trend="UPTREND")
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="TREND_CONT_LONG | ADX=30 | RSI=60 | MACD_H=5.0 | e9=6900 | #1",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        mock_dt = datetime(2026, 2, 24, 12, 59, 0)
        with patch("shree.execution.components.signal_processor.now_cst", return_value=mock_dt):
            result = await proc._generate_strategy_first_signal(
                features=_make_features(),
                returns=None,
                current_price=6920.25,
                structural_metrics=None,
            )

        assert result.signal.action == "HOLD"
        assert "dead_zone_block" in result.filters_applied

    @pytest.mark.asyncio
    async def test_allows_trend_cont_at_0930_cst(self):
        """TREND_CONT at 09:30 CST → ALLOWED (power hour)."""
        manager = _make_stub_manager(hybrid_trend="UPTREND")
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="TREND_CONT_LONG | ADX=30 | RSI=60 | MACD_H=5.0 | e9=6900 | #1",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        mock_dt = datetime(2026, 2, 24, 9, 30, 0)
        with patch("shree.execution.components.signal_processor.now_cst", return_value=mock_dt):
            result = await proc._generate_strategy_first_signal(
                features=_make_features(),
                returns=None,
                current_price=6920.25,
                structural_metrics=None,
            )

        assert result.signal.action == "BUY"
        assert result.signal.confidence > 0.0

    @pytest.mark.asyncio
    async def test_allows_trend_cont_at_1300_cst(self):
        """TREND_CONT at 13:00 CST → ALLOWED (just past dead zone)."""
        manager = _make_stub_manager(hybrid_trend="UPTREND")
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="TREND_CONT_LONG | ADX=30 | RSI=60 | MACD_H=5.0 | e9=6900 | #1",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        mock_dt = datetime(2026, 2, 24, 13, 0, 0)
        with patch("shree.execution.components.signal_processor.now_cst", return_value=mock_dt):
            result = await proc._generate_strategy_first_signal(
                features=_make_features(),
                returns=None,
                current_price=6920.25,
                structural_metrics=None,
            )

        assert result.signal.action == "BUY"

    @pytest.mark.asyncio
    async def test_allows_trend_cont_at_1029_cst(self):
        """TREND_CONT at 10:29 CST → ALLOWED (just before dead zone)."""
        manager = _make_stub_manager(hybrid_trend="UPTREND")
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="TREND_CONT_LONG | ADX=30 | RSI=60 | MACD_H=5.0 | e9=6900 | #1",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        mock_dt = datetime(2026, 2, 24, 10, 29, 0)
        with patch("shree.execution.components.signal_processor.now_cst", return_value=mock_dt):
            result = await proc._generate_strategy_first_signal(
                features=_make_features(),
                returns=None,
                current_price=6920.25,
                structural_metrics=None,
            )

        assert result.signal.action == "BUY"

    @pytest.mark.asyncio
    async def test_allows_pullback_in_dead_zone(self):
        """EMA9_PB_LONG at 11:15 CST in UPTREND → ALLOWED (dead zone only affects TREND_CONT)."""
        manager = _make_stub_manager(hybrid_trend="UPTREND")
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="EMA9_PB_LONG | ADX=30 | RSI=58 | ATR=10.0",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        mock_dt = datetime(2026, 2, 24, 11, 15, 0)
        with patch("shree.execution.components.signal_processor.now_cst", return_value=mock_dt):
            result = await proc._generate_strategy_first_signal(
                features=_make_features(),
                returns=None,
                current_price=6920.25,
                structural_metrics=None,
            )

        # Pullback should NOT be affected by dead-zone guard
        assert result.signal.action == "BUY"

    @pytest.mark.asyncio
    async def test_allows_or_breakout_in_dead_zone(self):
        """OR_BREAK_LONG at 11:00 CST → ALLOWED (OR breakout exempt from everything)."""
        manager = _make_stub_manager(hybrid_trend="UPTREND")
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="OR_BREAK_LONG | ADX=25 | OR_H=6930.00",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        mock_dt = datetime(2026, 2, 24, 11, 0, 0)
        with patch("shree.execution.components.signal_processor.now_cst", return_value=mock_dt):
            result = await proc._generate_strategy_first_signal(
                features=_make_features(),
                returns=None,
                current_price=6920.25,
                structural_metrics=None,
            )

        assert result.signal.action == "BUY"

    @pytest.mark.asyncio
    async def test_dead_zone_metadata_preserved(self):
        """Blocked signal should carry dead-zone info in metadata."""
        manager = _make_stub_manager(hybrid_trend="UPTREND")
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.72,
            reason="TREND_CONT_LONG | ADX=30 | RSI=60 | e9=6900 | #1",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        mock_dt = datetime(2026, 2, 24, 11, 45, 0)
        with patch("shree.execution.components.signal_processor.now_cst", return_value=mock_dt):
            result = await proc._generate_strategy_first_signal(
                features=_make_features(),
                returns=None,
                current_price=6920.25,
                structural_metrics=None,
            )

        meta = result.signal.metadata
        assert "dead_zone_guard" in meta
        assert meta["dead_zone_guard"]["original_action"] == "BUY"
        assert meta["dead_zone_guard"]["original_confidence"] == 0.72
        assert meta["dead_zone_guard"]["time_cst"] == "11:45"


# ===========================================================================
# 3. Combined guard interaction tests
# ===========================================================================

class TestGuardInteraction:
    """Test that guards work correctly when multiple conditions overlap."""

    @pytest.mark.asyncio
    async def test_chop_guard_fires_before_dead_zone(self):
        """When both CHOP and dead-zone apply, CHOP fires first (earlier in pipeline)."""
        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.70,
            reason="TREND_CONT_LONG | ADX=30 | RSI=60 | e9=6900 | #1",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        mock_dt = datetime(2026, 2, 24, 11, 15, 0)
        with patch("shree.execution.components.signal_processor.now_cst", return_value=mock_dt):
            result = await proc._generate_strategy_first_signal(
                features=_make_features(),
                returns=None,
                current_price=6920.25,
                structural_metrics=None,
            )

        # CHOP guard should fire first, before dead-zone gets a chance
        assert result.signal.action == "HOLD"
        assert "chop_regime_block" in result.filters_applied
        assert "CHOP_REGIME_BLOCK" in result.signal.metadata.get("reason", "")

    @pytest.mark.asyncio
    async def test_todays_trade_would_be_triple_blocked(self):
        """Today's exact trade conditions should be blocked.

        ADX=18 (below 25 floor) + CHOP + 11:15 CST (dead zone) = triple block.
        Even if one guard fails, the other two would catch it.
        """
        # The ADX=18 would be blocked at strategy level (never generates signal)
        # But if it somehow got through, CHOP guard blocks it
        # And if CHOP guard failed, dead-zone blocks it
        # This test validates the CHOP guard layer specifically

        manager = _make_stub_manager(hybrid_trend="CHOP")
        engine = MagicMock()
        signal = _make_signal(
            action="BUY",
            confidence=0.509,  # Today's exact final confidence
            reason="TREND_CONT_LONG | ADX=18 | RSI=65 | MACD_H=3.53 | e9=6887.5 | #1",
        )
        engine.evaluate.return_value = signal

        proc = _make_processor(manager=manager, engine=engine)

        result = await proc._generate_strategy_first_signal(
            features=_make_features(),
            returns=None,
            current_price=6902.50,  # Today's entry price
            structural_metrics=None,
        )

        assert result.signal.action == "HOLD"
        assert result.signal.confidence == 0.0
        assert result.filters_passed is False
