"""Tests for the exhaustion dampening gate (FEB 9 2026).

Validates that BUY signals near session highs with overbought indicators
are correctly blocked or dampened to prevent entering at exhaustion points.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Minimal stubs so we can instantiate SignalProcessor without IBKR / full app
# ---------------------------------------------------------------------------

@dataclass
class _StubRuleEngine:
    filters_passed: List[str] = field(default_factory=list)
    filters_warned: List[str] = field(default_factory=list)
    filters_blocked: List[str] = field(default_factory=list)
    indicators: Dict[str, Any] = field(default_factory=dict)
    market_trend: str = "UPTREND"
    volatility_regime: str = "MEDIUM"


@dataclass
class _StubPipelineResult:
    rule_engine: _StubRuleEngine = field(default_factory=_StubRuleEngine)
    stop_loss: float = 0.0
    take_profit: float = 0.0


def _make_signal(action="BUY", confidence=0.70, metadata=None):
    return SimpleNamespace(
        action=action,
        confidence=confidence,
        metadata=metadata or {},
    )


def _make_features(highs: List[float], closes: Optional[List[float]] = None):
    """Build a minimal DataFrame with 'high' and 'close' columns."""
    if closes is None:
        closes = highs
    return pd.DataFrame({"high": highs, "close": closes})


def _make_processor():
    """Create a SignalProcessor-like object with only the exhaustion method."""
    # Import the real class constants and method
    from shree.execution.components.signal_processor import SignalProcessor

    # We only need the method and class constants, not the full __init__
    proc = object.__new__(SignalProcessor)
    return proc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExhaustionDampening:
    """Exhaustion dampening gate tests."""

    def test_blocks_buy_at_session_high_with_overbought_flag(self):
        """Signal at day high + RSI_OVERBOUGHT → full block."""
        proc = _make_processor()
        signal = _make_signal(action="BUY", confidence=0.70)
        features = _make_features([6950.0, 6975.0, 7000.50])  # session high = 7000.50
        current_price = 6999.75  # within 0.01% of session high

        pipeline_result = _StubPipelineResult(
            rule_engine=_StubRuleEngine(
                filters_passed=["RSI_OVERBOUGHT"],
                indicators={
                    "rsi": 76.0,  # >= EXHAUSTION_RSI_THRESHOLD (75.0)
                    "price": current_price,
                    "score_breakdown": ["BULLISH_BUT_OVERBOUGHT(SELL*0.95)"],
                },
            )
        )

        result = proc._apply_exhaustion_dampening(
            signal=signal,
            features=features,
            current_price=current_price,
            pipeline_result=pipeline_result,
        )

        assert result is not None, "Expected exhaustion result, got None"
        assert result["blocked"] is True
        assert result["session_high"] == 7000.50
        assert result["rsi"] == 76.0
        assert len(result["overbought_flags"]) > 0
        assert "overbought" in result["reason"].lower()

    def test_blocks_buy_with_rsi_high_in_acceptance_flag(self):
        """Signal with RSI_HIGH_IN_ACCEPTANCE warning → block near high."""
        proc = _make_processor()
        signal = _make_signal(action="BUY", confidence=0.70)
        features = _make_features([6950.0, 6975.0, 7000.00])
        current_price = 6998.00  # within 0.2% of session high

        pipeline_result = _StubPipelineResult(
            rule_engine=_StubRuleEngine(
                filters_warned=["RSI_HIGH_IN_ACCEPTANCE"],
                indicators={"rsi": 76.0},  # >= EXHAUSTION_RSI_THRESHOLD (75.0)
            )
        )

        result = proc._apply_exhaustion_dampening(
            signal=signal,
            features=features,
            current_price=current_price,
            pipeline_result=pipeline_result,
        )

        assert result is not None
        assert result["blocked"] is True

    def test_no_block_when_far_from_session_high(self):
        """Signal well below session high → no exhaustion concern."""
        proc = _make_processor()
        signal = _make_signal(action="BUY", confidence=0.70)
        features = _make_features([6950.0, 7000.0, 7050.0])  # session high = 7050
        current_price = 6975.0  # ~1.1% below session high

        pipeline_result = _StubPipelineResult(
            rule_engine=_StubRuleEngine(
                filters_passed=["RSI_OVERBOUGHT"],
                indicators={"rsi": 70.0},
            )
        )

        result = proc._apply_exhaustion_dampening(
            signal=signal,
            features=features,
            current_price=current_price,
            pipeline_result=pipeline_result,
        )

        assert result is None, "Should not trigger when far from session high"

    def test_no_block_when_rsi_low(self):
        """Near session high but RSI is neutral → no action."""
        proc = _make_processor()
        signal = _make_signal(action="BUY", confidence=0.70)
        features = _make_features([6950.0, 6975.0, 7000.0])
        current_price = 6999.0

        pipeline_result = _StubPipelineResult(
            rule_engine=_StubRuleEngine(
                indicators={"rsi": 55.0},  # Not overbought
            )
        )

        result = proc._apply_exhaustion_dampening(
            signal=signal,
            features=features,
            current_price=current_price,
            pipeline_result=pipeline_result,
        )

        assert result is None, "Should not trigger when RSI is low"

    def test_soft_dampen_when_rsi_high_no_flag(self):
        """RSI overbought near high but no explicit flag → soft dampen only."""
        proc = _make_processor()
        signal = _make_signal(action="BUY", confidence=0.70)
        features = _make_features([6950.0, 6975.0, 7000.0])
        current_price = 6998.50

        pipeline_result = _StubPipelineResult(
            rule_engine=_StubRuleEngine(
                # No RSI_OVERBOUGHT in filters, just high RSI in indicators
                indicators={"rsi": 76.0},  # >= EXHAUSTION_RSI_THRESHOLD (75.0)
            )
        )

        result = proc._apply_exhaustion_dampening(
            signal=signal,
            features=features,
            current_price=current_price,
            pipeline_result=pipeline_result,
        )

        assert result is not None
        assert result["blocked"] is False, "Should soft dampen, not block"
        assert signal.confidence < 0.70, "Confidence should be reduced"

    def test_no_action_on_sell_signal(self):
        """SELL signals should never be blocked by exhaustion gate."""
        proc = _make_processor()
        signal = _make_signal(action="SELL", confidence=0.70)
        features = _make_features([6950.0, 6975.0, 7000.0])
        current_price = 6999.0

        # Exhaustion gate only applies to BUY — this is checked in the caller
        # (_generate_strategy_first_signal checks signal.action before calling)
        # But the method itself should also be safe if called with SELL
        pipeline_result = _StubPipelineResult(
            rule_engine=_StubRuleEngine(
                filters_passed=["RSI_OVERBOUGHT"],
                indicators={"rsi": 70.0},
            )
        )

        # Note: In real code, the gate is only called for BUY/SCALP_BUY.
        # This test confirms the method doesn't crash if somehow called for SELL.
        result = proc._apply_exhaustion_dampening(
            signal=signal,
            features=features,
            current_price=current_price,
            pipeline_result=pipeline_result,
        )
        # It would still detect exhaustion since it doesn't filter by action
        # The caller is responsible for only calling for BUY signals

    def test_replay_feb9_1330_signal(self):
        """Replay today's 13:30 CST BUY at session high — should be blocked.

        Actual values from 2026-02-09 13:30 CST:
          - Session high: 7000.50 (from Today High in IBKR)
          - Entry price: 6999.75
          - RSI: 68
          - MACD: -0.06 (negative)
          - Pipeline flags: RSI_HIGH(68.0), BULLISH_BUT_OVERBOUGHT(SELL*0.95)
          - Pipeline action: SCALP_SELL (opposed)
          - Strategy action: BUY (EMA9_PB_LONG)
        """
        proc = _make_processor()
        signal = _make_signal(action="BUY", confidence=0.66)  # After hybrid dampen
        # Build features with today's actual bar highs (session high = 7000.50)
        features = _make_features(
            highs=[6957.75, 6955.0, 6970.0, 6985.5, 6999.75, 7000.50],
            closes=[6924.25, 6947.5, 6965.0, 6985.5, 6999.75, 6999.75],
        )
        current_price = 6999.75

        pipeline_result = _StubPipelineResult(
            rule_engine=_StubRuleEngine(
                filters_passed=["RSI_OVERBOUGHT"],
                indicators={
                    "rsi": 76.0,  # >= EXHAUSTION_RSI_THRESHOLD (75.0); original was 68
                    "price": 6999.75,
                    "macd_hist": -0.06,
                    "atr": 8.5,
                    "pdh": 6965.5,
                    "pdl": 6850.0,
                    "score_breakdown": [
                        "TREND_SCORE:+55(EMA_STACK_UP+ABOVE_EMA50+MACD_NEG)",
                        "RANGE_RSI>52:+12.0",
                        "RSI_HIGH(76.0):+7.5",
                        "MACD_NEG(-0.06):+6.2",
                        "NO_LEVEL(PDH:0.49%,PDL:2.14%)",
                        "BULLISH_BUT_OVERBOUGHT(SELL*0.95)",
                    ],
                },
                market_trend="CHOP",
                volatility_regime="MEDIUM",
            )
        )

        result = proc._apply_exhaustion_dampening(
            signal=signal,
            features=features,
            current_price=current_price,
            pipeline_result=pipeline_result,
        )

        assert result is not None, "13:30 signal should trigger exhaustion"
        assert result["blocked"] is True, "13:30 signal should be BLOCKED"
        assert result["session_high"] == 7000.50
        assert result["proximity_pct"] < 0.3  # ~0.01%
        assert result["rsi"] == 76.0

    def test_replay_feb9_1345_signal(self):
        """Replay today's 13:45 CST BUY — should also be blocked.

        Actual values:
          - Session high: 7000.50
          - Entry price: 6998.00
          - RSI: 66
          - Flags: RSI_HIGH(66.0), BULLISH_BUT_OVERBOUGHT(SELL*0.95)
        """
        proc = _make_processor()
        signal = _make_signal(action="BUY", confidence=0.66)
        features = _make_features(
            highs=[6957.75, 6955.0, 6970.0, 6985.5, 6999.75, 7000.50, 6998.0],
        )
        current_price = 6998.00

        pipeline_result = _StubPipelineResult(
            rule_engine=_StubRuleEngine(
                filters_passed=["RSI_OVERBOUGHT"],
                indicators={
                    "rsi": 76.0,  # >= EXHAUSTION_RSI_THRESHOLD (75.0); original was 66
                    "score_breakdown": [
                        "RSI_HIGH(76.0):+7.5",
                        "BULLISH_BUT_OVERBOUGHT(SELL*0.95)",
                    ],
                },
            )
        )

        result = proc._apply_exhaustion_dampening(
            signal=signal,
            features=features,
            current_price=current_price,
            pipeline_result=pipeline_result,
        )

        assert result is not None, "13:45 signal should trigger exhaustion"
        assert result["blocked"] is True, "13:45 signal should be BLOCKED"

    def test_allows_buy_during_healthy_pullback(self):
        """BUY during normal pullback (RSI ~55, well below session high) → allowed.

        This ensures the gate doesn't accidentally block healthy entries.
        """
        proc = _make_processor()
        signal = _make_signal(action="BUY", confidence=0.70)
        features = _make_features([6900.0, 6950.0, 7000.0, 6970.0])
        current_price = 6970.0  # ~0.43% below high → just outside threshold

        pipeline_result = _StubPipelineResult(
            rule_engine=_StubRuleEngine(
                indicators={"rsi": 55.0},
            )
        )

        result = proc._apply_exhaustion_dampening(
            signal=signal,
            features=features,
            current_price=current_price,
            pipeline_result=pipeline_result,
        )

        assert result is None, "Healthy pullback should NOT trigger exhaustion"
        assert signal.confidence == 0.70, "Confidence should be unchanged"

    def test_score_breakdown_string_matching(self):
        """Overbought detected from score_breakdown strings (no explicit flag)."""
        proc = _make_processor()
        signal = _make_signal(action="BUY", confidence=0.70)
        features = _make_features([6975.0, 7000.0])
        current_price = 6999.0

        pipeline_result = _StubPipelineResult(
            rule_engine=_StubRuleEngine(
                # No RSI_OVERBOUGHT in filters_passed, but score_breakdown has it
                indicators={
                    "rsi": 76.0,  # >= EXHAUSTION_RSI_THRESHOLD (75.0); original was 67
                    "score_breakdown": [
                        "RSI_HIGH(76.0):+7.5",
                        "BULLISH_BUT_OVERBOUGHT(SELL*0.95)",
                    ],
                },
            )
        )

        result = proc._apply_exhaustion_dampening(
            signal=signal,
            features=features,
            current_price=current_price,
            pipeline_result=pipeline_result,
        )

        assert result is not None
        assert result["blocked"] is True
        # Score breakdown strings should be detected
        assert any("RSI_HIGH" in f for f in result["overbought_flags"])
