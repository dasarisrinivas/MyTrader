"""Tests for level confirmation gating and pipeline integration confidence."""
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

from mytrader.rag.hybrid_rag_pipeline import (
    HybridRAGPipeline,
    HybridPipelineResult,
    RAGRetrievalResult,
    RuleEngineResult,
    TradeAction,
)
from mytrader.rag.pipeline_integration import HybridPipelineIntegration
from mytrader.hybrid.multi_factor_scorer import FactorScores, MultiFactorDecision
from mytrader.utils.hold_reason import HoldReason


def test_level_confirmation_no_hold_when_far_from_levels():
    pipeline = HybridRAGPipeline(config={})
    action, confidence, _, hold_reason, _ = pipeline._apply_level_confirmation(
        final_action=TradeAction.BUY,
        final_confidence=70.0,
        final_reasoning="rule confidence",
        indicators={"price": 100.0, "close": 100.0, "pdh": 130.0, "pdl": 70.0, "atr": 2.0},
    )
    assert action == TradeAction.BUY
    assert confidence == 70.0
    assert hold_reason is None
    assert pipeline._level_confirm_wait["BUY"] == 0


def test_level_confirmation_hold_near_pdh():
    pipeline = HybridRAGPipeline(config={"level_confirmation_settings": {"max_wait_candles": 3}})
    indicators = {"price": 99.8, "close": 99.8, "pdh": 100.0, "pdl": 90.0, "atr": 1.0}
    action, confidence, reasoning, hold_reason, _ = pipeline._apply_level_confirmation(
        final_action=TradeAction.BUY,
        final_confidence=72.0,
        final_reasoning="testing",
        indicators=indicators,
    )
    assert action == TradeAction.HOLD
    assert confidence == 0.0
    assert hold_reason is not None
    assert hold_reason.reason_code == "LEVEL_CONFIRMATION"
    assert "waiting for" in reasoning
    assert pipeline._level_confirm_wait["BUY"] == 1


def test_level_confirmation_timeout_allows_trade_with_penalty():
    pipeline = HybridRAGPipeline(
        config={
            "level_confirmation_settings": {
                "max_wait_candles": 2,
                "timeout_mode": "SOFT_PENALTY",
                "timeout_penalty": 0.12,
            }
        }
    )
    indicators = {"price": 99.8, "close": 99.8, "pdh": 100.0, "pdl": 90.0, "atr": 1.0}
    # First pass should hold
    action, _, _, hold_reason, _ = pipeline._apply_level_confirmation(
        final_action=TradeAction.BUY,
        final_confidence=60.0,
        final_reasoning="testing",
        indicators=indicators,
    )
    assert action == TradeAction.HOLD
    assert hold_reason is not None
    # Second pass hits timeout and applies penalty while allowing the trade
    action, confidence, reasoning, hold_reason, _ = pipeline._apply_level_confirmation(
        final_action=TradeAction.BUY,
        final_confidence=60.0,
        final_reasoning="testing",
        indicators=indicators,
    )
    assert action == TradeAction.BUY
    assert hold_reason is None
    assert confidence == pytest.approx(48.0)  # 60 - (0.12 * 100)
    assert "timeout" in reasoning.lower()
    assert pipeline._level_confirm_wait["BUY"] == 0


class _FakePipeline:
    def __init__(self, result: HybridPipelineResult):
        self.result = result
        self.calls = 0

    def process(self, market_data):
        self.calls += 1
        return self.result


class _FakeScorer:
    def __init__(self, decision: MultiFactorDecision):
        self.decision = decision

    def score(self, *args, **kwargs):
        return self.decision


class _FakeTradeLogger:
    def get_recent_market_metrics(self, limit: int = 30):
        return []

    def record_market_metrics(self, payload):
        self.payload = payload


def test_pipeline_integration_uses_pipeline_confidence_and_hold_reason():
    features = pd.DataFrame(
        [
            {
                "close": 100.0,
                "open": 99.5,
                "high": 101.0,
                "low": 99.0,
                "EMA_9": 100.0,
                "EMA_20": 99.8,
                "SMA_50": 99.5,
                "RSI_14": 55.0,
                "MACDhist_12_26_9": 0.1,
                "ATR_14": 1.0,
                "PDH": 101.0,
                "PDL": 98.0,
            }
        ],
        index=pd.date_range("2024-01-01", periods=1, freq="T"),
    )
    hold_reason = HoldReason(
        gate="hybrid_pipeline.rule_engine",
        reason_code="LEVEL_CONFIRMATION",
        reason_detail="waiting",
        context={"test": True},
    )
    rule = RuleEngineResult(
        signal=TradeAction.BUY,
        score=72.0,
        indicators={"price": 100.0, "close": 100.0, "atr": 1.0, "pdh": 101.0, "pdl": 99.0},
    )
    pipeline_result = HybridPipelineResult(
        final_action=TradeAction.HOLD,
        final_confidence=72.0,
        final_reasoning="waiting",
        rule_engine=rule,
        rag_retrieval=RAGRetrievalResult(),
        hold_reason=hold_reason,
        entry_price=100.0,
        stop_loss=1.5,
        take_profit=2.0,
        position_size=1.0,
        timestamp="2024-01-01T00:00:00Z",
        processing_time_ms=10.0,
    )
    decision = MultiFactorDecision(
        action="HOLD",
        confidence=0.5,
        scores=FactorScores(),
        reasoning=[],
        confidence_band="B",
        metadata={},
    )

    integration = HybridPipelineIntegration.__new__(HybridPipelineIntegration)
    integration.enabled = True
    integration.settings = SimpleNamespace(trading=SimpleNamespace(tick_size=0.25))
    integration.context_bus = None
    integration.pipeline = _FakePipeline(pipeline_result)
    integration.multi_factor_scorer = _FakeScorer(decision)
    integration.trade_logger = _FakeTradeLogger()
    integration.mistake_analyzer = None
    integration.storage = None
    integration.embedding_builder = None
    integration.rag_data_path = Path("rag_data")
    integration.daily_updater = None
    integration._symbol = "ES"
    integration._timeframe = "1m"
    integration._current_trade_id = None

    signal, result = integration.process_sync(features, current_price=100.0)
    assert signal.action == "HOLD"
    assert signal.confidence == pytest.approx(0.72)
    assert signal.metadata["pipeline_final_confidence"] == pytest.approx(72.0)
    assert signal.metadata["hold_reason"]["reason_code"] == "LEVEL_CONFIRMATION"
    assert signal.metadata["hold_reason"]["gate"] == hold_reason.gate
