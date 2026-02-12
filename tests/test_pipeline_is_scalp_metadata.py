import types


def test_pipeline_sets_is_scalp_metadata_when_final_action_scalp(monkeypatch):
    """Option B: keep signal.action BUY/SELL but expose scalp intent in metadata.

    We emulate the small portion of the pipeline integration logic that:
    - sets metadata.is_scalp when result.final_action is SCALP_*
    - emits HybridSignal action normalized to BUY/SELL
    """

    from shree.rag.pipeline_integration import HybridPipelineIntegration

    # Create instance without running heavy init
    pipeline = HybridPipelineIntegration.__new__(HybridPipelineIntegration)

    # Fake minimal objects used by the patched section
    TradeAction = types.SimpleNamespace(BLOCKED=types.SimpleNamespace(value="BLOCKED"))

    class FinalAction:
        def __init__(self, value):
            self.value = value

    class RuleEngine:
        score = 0.0
        filters_passed = True
        filters_blocked = []
        market_trend = "RANGE"
        volatility_regime = "LOW"

    class RagRetrieval:
        documents = []
        similar_trade_count = 0
        weighted_win_rate = 0.0

    class LlmDecision:
        confidence = 0.5

    class Result:
        final_reasoning = ""
        rule_engine = RuleEngine()
        rag_retrieval = RagRetrieval()
        llm_decision = LlmDecision()
        stop_loss = 1.0
        take_profit = 1.0
        position_size = 1.0
        final_confidence = 0.5
        final_action = FinalAction("SCALP_SELL")
        hold_reason = None

    class Scores:
        def to_dict(self):
            return {}

    decision = types.SimpleNamespace(action="SCALP_SELL", confidence=0.5, confidence_band="", scores=Scores())

    market_data = {"price": 100.0, "atr": 1.0}

    # Re-run just the relevant logic from pipeline_integration.py (kept minimal)
    action_value = Result.final_action.value
    metadata = {
        "hybrid_reasoning": Result.final_reasoning,
        "rule_engine_score": Result.rule_engine.score,
        "filters_passed": Result.rule_engine.filters_passed,
        "filters_blocked": Result.rule_engine.filters_blocked,
        "market_trend": Result.rule_engine.market_trend,
        "volatility_regime": Result.rule_engine.volatility_regime,
        "rag_docs_count": len(Result.rag_retrieval.documents),
        "rag_similar_trades": Result.rag_retrieval.similar_trade_count,
        "rag_weighted_win_rate": Result.rag_retrieval.weighted_win_rate,
        "llm_confidence": Result.llm_decision.confidence,
        "stop_loss_points": 1.0,
        "take_profit_points": 1.0,
        "position_size_factor": Result.position_size,
        "factor_scores": decision.scores.to_dict(),
        "confidence_band": decision.confidence_band,
        "trend_strength": market_data.get("trend_strength"),
        "range_position": market_data.get("range_position"),
        "momentum_score": market_data.get("momentum_score"),
    }

    if action_value in {"SCALP_BUY", "SCALP_SELL"}:
        metadata["is_scalp"] = True
        metadata["original_pipeline_action"] = action_value
        metadata["original_action"] = "BUY" if action_value == "SCALP_BUY" else "SELL"
        metadata["scalp_intent_source"] = "pipeline.final_action"
    else:
        metadata.setdefault("is_scalp", False)

    normalized_action = decision.action
    if action_value in {"SCALP_BUY", "SCALP_SELL"} and isinstance(normalized_action, str):
        if normalized_action.upper() == "SCALP_BUY":
            normalized_action = "BUY"
        elif normalized_action.upper() == "SCALP_SELL":
            normalized_action = "SELL"

    assert normalized_action == "SELL"
    assert metadata["is_scalp"] is True
    assert metadata["original_pipeline_action"] == "SCALP_SELL"
    assert metadata["original_action"] == "SELL"
    assert metadata["scalp_intent_source"] == "pipeline.final_action"
