import math
from unittest.mock import AsyncMock

import pytest

from mytrader.config import Settings
from mytrader.execution.live_trading_manager import LiveTradingManager
from mytrader.rag.pipeline_integration import HybridPipelineIntegration
from mytrader.rag.hybrid_rag_pipeline import LLMDecisionMaker, RuleEngineResult, RAGRetrievalResult, TradeAction


def test_safe_float_handles_edge_cases():
    helper = HybridPipelineIntegration._safe_float
    assert helper(1.25) == pytest.approx(1.25)
    assert helper(" 2.5 ") == pytest.approx(2.5)
    assert helper("nan", default=7.0) == 7.0
    assert helper("", default=-1.0) == -1.0
    assert helper(None, default=0.0) == 0.0
    assert helper("abc", default=3.0) == 3.0
    assert helper(math.inf, default=5.0) == 5.0


@pytest.mark.asyncio
async def test_hybrid_exception_forces_hold(monkeypatch):
    settings = Settings()
    manager = LiveTradingManager(settings, simulation_mode=True)
    manager._current_cycle_id = "test-cycle"
    manager._cycle_context["test-cycle"] = {"reason_codes": set()}
    manager._broadcast_signal = AsyncMock()
    manager._broadcast_status = AsyncMock()

    await manager._handle_hybrid_pipeline_failure(4500.0, RuntimeError("boom"))

    assert manager.status.last_signal == "HOLD"
    assert manager.status.signal_confidence == 0.0
    assert "HYBRID_PIPELINE_ERROR" in manager._cycle_context["test-cycle"]["reason_codes"]
    manager._broadcast_signal.assert_awaited()
    manager._broadcast_status.assert_awaited()


def test_uncertainty_band_blocks_llm_call():
    maker = LLMDecisionMaker(llm_client=None, config={"uncertainty_band": (0.35, 0.65)})
    rule = RuleEngineResult(signal=TradeAction.BUY, score=90)
    rag = RAGRetrievalResult(weighted_win_rate=0.5)
    allowed, reason = maker._should_invoke_llm(rule, rag, {"candle_timestamp": "2025-01-01T10:00:00"})
    assert not allowed
    assert reason == "outside_uncertainty_band"
