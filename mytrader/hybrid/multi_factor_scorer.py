"""Multi-factor scoring that fuses technical, RAG, and external context."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from mytrader.rag.hybrid_rag_pipeline import HybridPipelineResult


@dataclass
class FactorScores:
    """Normalized factor scores (0-1) for explainability."""

    technical: float = 0.5
    llm: float = 0.5
    rag: float = 0.5
    news: float = 0.5
    macro: float = 0.5
    risk: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class MultiFactorDecision:
    """Final fused decision with supporting metadata."""

    action: str
    confidence: float
    scores: FactorScores
    reasoning: List[str]
    confidence_band: str
    metadata: Dict[str, Any]


class MultiFactorScorer:
    """Blend rule-engine, RAG, news, macro, and risk signals into one score."""

    DEFAULT_WEIGHTS = {
        "technical": 0.30,
        "llm": 0.20,
        "rag": 0.20,
        "news": 0.10,
        "macro": 0.10,
        "risk": 0.10,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        weights = weights or {}
        merged = {**self.DEFAULT_WEIGHTS, **weights}
        total = sum(merged.values())
        if total <= 0:
            total = 1.0
        self.weights = {k: v / total for k, v in merged.items()}
        logger.info(f"MultiFactorScorer initialized with weights={self.weights}")

    def score(
        self,
        pipeline_result: "HybridPipelineResult",
        news_context: Optional[Dict[str, Any]] = None,
        macro_context: Optional[Dict[str, Any]] = None,
        risk_context: Optional[Dict[str, Any]] = None,
        market_metrics: Optional[Dict[str, Any]] = None,
    ) -> MultiFactorDecision:
        """Produce a fused decision."""
        news_context = news_context or {}
        macro_context = macro_context or {}
        risk_context = risk_context or {}
        market_metrics = market_metrics or {}

        scores = FactorScores(
            technical=self._normalize(pipeline_result.rule_engine.score, 100),
            llm=self._extract_llm_score(pipeline_result),
            rag=self._extract_rag_score(pipeline_result),
            news=self._extract_news_score(news_context),
            macro=self._extract_macro_score(macro_context),
            risk=self._extract_risk_score(risk_context),
        )
        logger.debug(
            "🔍 Factor inputs technical={:.2f} rag={:.2f} llm={:.2f} news={:.2f} macro={:.2f} risk={:.2f}",
            scores.technical,
            scores.rag,
            scores.llm,
            scores.news,
            scores.macro,
            scores.risk,
        )
        structural_trend = market_metrics.get("trend_strength")
        if structural_trend:
            scores.technical = float(max(0.0, min(1.0, scores.technical + structural_trend)))

        reasoning: List[str] = []
        reasoning.append(
            f"Rule engine score {pipeline_result.rule_engine.score:.1f}/100 "
            f"(trend={pipeline_result.rule_engine.market_trend}, "
            f"vol={pipeline_result.rule_engine.volatility_regime})"
        )
        if pipeline_result.llm_decision:
            reasoning.append(
                f"LLM decision {pipeline_result.llm_decision.action.value} "
                f"@ {pipeline_result.llm_decision.confidence:.0f}"
            )
        if news_context:
            reasoning.append(
                f"News sentiment {news_context.get('sentiment_score', 0):+.2f} "
                f"bias={news_context.get('bias', 'NEUTRAL')}"
            )
        if macro_context:
            reasoning.append(
                f"Macro regime {macro_context.get('regime', 'N/A')} "
                f"risk_index={macro_context.get('risk_index', 0)}"
            )

        final_confidence = sum(
            getattr(scores, key) * weight for key, weight in self.weights.items()
        )
        pipeline_conf = pipeline_result.final_confidence
        if pipeline_conf > 1.0:
            pipeline_conf = pipeline_conf / 100.0
        pipeline_conf = max(0.0, min(1.0, pipeline_conf))
        logger.debug(
            "🎯 Confidence breakdown tech={:.2f} rag={:.2f} llm={:.2f} news={:.2f} macro={:.2f} risk={:.2f}",
            scores.technical,
            scores.rag,
            scores.llm,
            scores.news,
            scores.macro,
            scores.risk,
        )
        # Blend in the pipeline's native confidence to avoid chronic HOLD bias
        final_confidence = 0.35 * pipeline_conf + 0.65 * final_confidence
        final_confidence = max(final_confidence, pipeline_conf * 0.55)
        final_confidence = max(0.0, min(1.0, final_confidence))
        logger.debug("📊 Final confidence: {:.2f} (pipeline={:.2f})", final_confidence, pipeline_conf)
        if final_confidence == 0:
            logger.error("🚨 ZERO CONFIDENCE DETECTED - COMPONENT BREAKDOWN:")
            logger.error("   Technical score: {:.3f}", scores.technical)
            logger.error("   LLM score: {:.3f}", scores.llm)
            logger.error("   RAG score: {:.3f}", scores.rag)
            logger.error("   News score: {:.3f}", scores.news)
            logger.error("   Macro score: {:.3f}", scores.macro)
            logger.error("   Risk score: {:.3f}", scores.risk)
            logger.error("   Pipeline conf: {:.3f}", pipeline_conf)
            logger.error("   Weights: {}", self.weights)
            final_confidence = max(0.15, final_confidence)
            logger.warning("🔧 Applied confidence floor: {:.3f}", final_confidence)
        action = self._action_value(pipeline_result.final_action)

        # Apply directional nudges from news/macro context
        action, final_confidence = self._apply_bias_adjustments(
            action,
            final_confidence,
            pipeline_result,
            news_context,
            macro_context,
            reasoning,
            market_metrics,
        )

        confidence_band = self._band(final_confidence)

        metadata = {
            "factor_scores": scores.to_dict(),
            "weights": self.weights,
            "final_reasoning": pipeline_result.final_reasoning,
            "news_context": news_context,
            "macro_context": macro_context,
            "risk_context": risk_context,
        }

        return MultiFactorDecision(
            action=action,
            confidence=max(0.0, min(1.0, final_confidence)),
            scores=scores,
            reasoning=reasoning,
            confidence_band=confidence_band,
            metadata=metadata,
        )

    def _normalize(self, value: float, max_value: float) -> float:
        if max_value == 0:
            return 0.5
        normalized = max(0.0, min(1.0, value / max_value))
        return normalized

    def _extract_llm_score(self, pipeline_result: HybridPipelineResult) -> float:
        if pipeline_result.llm_decision:
            return self._normalize(pipeline_result.llm_decision.confidence, 100)
        return 0.5

    def _extract_rag_score(self, pipeline_result: HybridPipelineResult) -> float:
        rag = pipeline_result.rag_retrieval
        if rag.similar_trade_count == 0:
            return 0.5
        return float(max(0.0, min(1.0, rag.historical_win_rate)))

    def _extract_news_score(self, news_context: Dict[str, Any]) -> float:
        sentiment = news_context.get("sentiment_score")
        if sentiment is None:
            return 0.5
        return max(0.0, min(1.0, (sentiment + 1) / 2))

    def _extract_macro_score(self, macro_context: Dict[str, Any]) -> float:
        risk_index = macro_context.get("risk_index")
        if risk_index is not None:
            return 1.0 - max(0.0, min(1.0, risk_index))
        bias = macro_context.get("regime_bias")
        if bias == "BULLISH":
            return 0.65
        if bias == "BEARISH":
            return 0.35
        return 0.5

    def _extract_risk_score(self, risk_context: Dict[str, Any]) -> float:
        drawdown = risk_context.get("active_drawdown_pct")
        if drawdown is None:
            return 0.5
        drawdown = min(0.5, max(0.0, abs(drawdown)))
        return max(0.0, 0.8 - drawdown * 1.2)

    def _apply_bias_adjustments(
        self,
        action: str,
        confidence: float,
        pipeline_result: HybridPipelineResult,
        news_context: Dict[str, Any],
        macro_context: Dict[str, Any],
        reasoning: List[str],
        market_metrics: Dict[str, Any],
    ) -> tuple[str, float]:
        bias = news_context.get("bias")
        if bias and bias != "NEUTRAL":
            penalty = 0.1
            if bias == "BULLISH" and action.startswith("S"):
                reasoning.append(f"News bias conflicts with SELL signal (-{penalty:.2f})")
                confidence -= penalty
            elif bias == "BEARISH" and action.startswith("B"):
                reasoning.append(f"News bias conflicts with BUY signal (-{penalty:.2f})")
                confidence -= penalty
        macro_bias = macro_context.get("regime_bias")
        if macro_bias and macro_bias != "NEUTRAL":
            adjust = 0.04
            if macro_bias == "BULLISH" and action.startswith("B"):
                confidence += adjust
                reasoning.append("Macro bias supports BUY (+0.04)")
            elif macro_bias == "BEARISH" and action.startswith("S"):
                confidence += adjust
                reasoning.append("Macro bias supports SELL (+0.04)")
            else:
                penalty = adjust + 0.01
                confidence -= penalty
                reasoning.append(f"Macro bias contradicts trade (-{penalty:.2f})")
        structural_trend = market_metrics.get("trend_strength")
        if structural_trend:
            adjustment = min(0.05, abs(structural_trend))
            if structural_trend > 0 and action.startswith("B"):
                confidence += adjustment
                reasoning.append(f"Trend structure supports BUY (+{adjustment:.2f})")
            elif structural_trend < 0 and action.startswith("S"):
                confidence += adjustment
                reasoning.append(f"Trend structure supports SELL (+{adjustment:.2f})")
            else:
                confidence -= adjustment / 2
                reasoning.append("Trend memory contradicts signal")
        range_position = market_metrics.get("range_position")
        if range_position is not None:
            if action.startswith("B") and range_position > 0.9:
                confidence -= 0.03
                reasoning.append("Price extended near resistance (-0.03)")
            elif action.startswith("S") and range_position < 0.1:
                confidence -= 0.03
                reasoning.append("Price extended near support (-0.03)")
        if self._action_value(pipeline_result.final_action) == "BLOCKED":
            reasoning.append("Rule engine blocked trade, forcing HOLD")
            return "HOLD", 0.0
        return action, confidence

    def _band(self, confidence: float) -> str:
        if confidence >= 0.75:
            return "A"
        if confidence >= 0.6:
            return "B"
        if confidence >= 0.5:
            return "C"
        return "D"

    @staticmethod
    def _action_value(action: Any) -> str:
        """Return string value from TradeAction or plain string."""
        return action.value if hasattr(action, "value") else str(action)


__all__ = ["MultiFactorScorer", "FactorScores", "MultiFactorDecision"]
