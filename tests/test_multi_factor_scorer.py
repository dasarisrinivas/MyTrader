"""Tests for MultiFactorScorer."""
from mytrader.hybrid.multi_factor_scorer import MultiFactorScorer
from mytrader.rag.hybrid_rag_pipeline import (
    HybridPipelineResult,
    RuleEngineResult,
    RAGRetrievalResult,
    LLMDecisionResult,
    TradeAction,
)


def build_pipeline_result():
    rule = RuleEngineResult(
        signal=TradeAction.BUY,
        score=72.0,
        market_trend="UPTREND",
        volatility_regime="LOW",
    )
    rag = RAGRetrievalResult(
        historical_win_rate=0.65,
        similar_trade_count=4,
    )
    llm = LLMDecisionResult(
        action=TradeAction.BUY,
        confidence=78.0,
        reasoning="Favorable setup",
    )
    return HybridPipelineResult(
        final_action=TradeAction.BUY,
        final_confidence=75.0,
        final_reasoning="Rule + LLM alignment",
        rule_engine=rule,
        rag_retrieval=rag,
        llm_decision=llm,
    )


def test_multi_factor_scorer_blends_context():
    scorer = MultiFactorScorer()
    pipeline_result = build_pipeline_result()
    decision = scorer.score(
        pipeline_result,
        news_context={"sentiment_score": 0.3, "bias": "BULLISH"},
        macro_context={"regime_bias": "BULLISH"},
        risk_context={"active_drawdown_pct": -0.05},
    )
    assert decision.action == "BUY"
    assert decision.confidence > 0.55
    assert decision.scores.news > 0.5


def test_multi_factor_scorer_handles_conflicts():
    scorer = MultiFactorScorer()
    pipeline_result = build_pipeline_result()
    pipeline_result.final_action = TradeAction.SELL
    pipeline_result.rule_engine.signal = TradeAction.SELL
    decision = scorer.score(
        pipeline_result,
        news_context={"sentiment_score": 0.2, "bias": "BULLISH"},
        macro_context={"regime_bias": "BEARISH"},
    )
    # conflicting news bias should reduce confidence for SELL
    assert decision.confidence < 0.6
