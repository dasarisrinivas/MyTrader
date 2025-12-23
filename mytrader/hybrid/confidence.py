"""Modified Confidence Scorer for MyTrader.

This module provides an updated ``ConfidenceScorer`` that relaxes the default
confidence thresholds used in the hybrid decision engine.  Recent live
observations showed that the original configuration (confidence threshold
``0.60``, minimum technical score ``0.50``, minimum model confidence
``0.40``) resulted in no trades being executed because most scores hovered
just below the required levels.  The updated defaults lower these
requirements to encourage more trading opportunities while still enforcing
meaningful consensus across technical, model and RAG components.

Key changes:

* ``confidence_threshold`` is reduced to ``0.55``.  Trades will be allowed
  when the weighted combination of technical, model and RAG scores reaches
  55 % rather than 60 % confidence.
* ``min_technical_score`` is reduced to ``0.45``.  The deterministic
  engine can now flag slightly weaker signals as candidates, allowing the
  LLM/RAG layer to provide confirmation.
* ``min_model_confidence`` is reduced to ``0.35``.  LLM recommendations
  often produce moderate confidence values, and this change prevents
  discarding otherwise viable trades when the LLM confidence is slightly
  below 40 %.

These adjustments can be further tuned via configuration or by calling
``update_weights`` and ``update_config`` on the hybrid decision engine.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mytrader.hybrid.d_engine import DEngineSignal
from mytrader.hybrid.h_engine import HEngineAdvisory
from mytrader.utils.logger import logger


@dataclass
class ConfidenceResult:
    """Result of confidence scoring.

    ``final_confidence`` reflects the weighted combination of technical,
    model and RAG scores.  ``should_trade`` indicates whether the final
    confidence meets the threshold and whether consensus requirements
    are satisfied.  Reasons provide human‑readable explanations for
    downstream analysis and logging.
    """

    final_confidence: float  # 0.0–1.0 weighted combination
    should_trade: bool       # True if confidence >= threshold
    technical_score: float = 0.0
    model_confidence: float = 0.0
    rag_similarity: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)
    action: str = "HOLD"
    position_size_pct: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_confidence": self.final_confidence,
            "should_trade": self.should_trade,
            "technical_score": self.technical_score,
            "model_confidence": self.model_confidence,
            "rag_similarity": self.rag_similarity,
            "weights": self.weights,
            "action": self.action,
            "position_size_pct": self.position_size_pct,
            "timestamp": self.timestamp.isoformat(),
            "reasons": self.reasons,
        }


class ConfidenceScorer:
    """Combines technical, model, and RAG scores with configurable weights.

    The final confidence is computed as a weighted sum of the component
    scores.  A trade is only considered if the final confidence exceeds
    ``confidence_threshold`` and, if configured, the D‑engine and H‑engine
    agree on the action (consensus mode).  Minimum individual component
    thresholds (technical and model confidence) must also be satisfied.
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "technical": 0.5,
        "model": 0.3,
        "rag": 0.2,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.55,
        require_consensus: bool = True,
        min_technical_score: float = 0.45,
        min_model_confidence: float = 0.35,
    ) -> None:
        """Initialize the confidence scorer with adjustable parameters.

        Args:
            weights: Optional dictionary overriding the default weightings for
                technical, model and RAG scores.  Weights are normalized to
                sum to 1.0.
            confidence_threshold: Minimum final confidence required to permit
                a trade.  Defaults to 0.55 (55 %).
            require_consensus: If ``True``, the D‑engine and H‑engine must
                agree on the direction (BUY/SELL) for a trade to proceed.
            min_technical_score: Minimum technical score from the D‑engine to
                consider a signal.  Defaults to 0.45.
            min_model_confidence: Minimum confidence from the H‑engine (LLM)
                to consider a recommendation.  Defaults to 0.35.
        """
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self.confidence_threshold = confidence_threshold
        self.require_consensus = require_consensus
        self.min_technical_score = min_technical_score
        self.min_model_confidence = min_model_confidence

        # Normalize weight values to sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info(
            f"ConfidenceScorer initialized: weights={self.weights}, "
            f"threshold={self.confidence_threshold:.2f}, consensus={self.require_consensus}, "
            f"min_tech={self.min_technical_score:.2f}, min_model={self.min_model_confidence:.2f}"
        )

    def calculate(
        self,
        d_signal: DEngineSignal,
        h_advisory: Optional[HEngineAdvisory] = None,
    ) -> ConfidenceResult:
        """Calculate the final confidence and trading decision.

        Combines the deterministic engine signal and (optional) heuristic/LLM
        advisory into a unified confidence score.  Returns a ``ConfidenceResult``
        containing the final confidence, recommended action and reasons.

        Args:
            d_signal: Signal from the deterministic engine.
            h_advisory: Optional advisory from the heuristic/LLM engine.

        Returns:
            ConfidenceResult with final confidence and decision details.
        """
        reasons: List[str] = []

        # Component scores
        technical_score = d_signal.technical_score

        if h_advisory:
            model_confidence = h_advisory.model_confidence
            rag_similarity = h_advisory.rag_similarity_score
        else:
            # Neutral scores when no H‑engine call was made
            model_confidence = 0.5
            rag_similarity = 0.5
            reasons.append("H-engine not invoked - using neutral scores")

        # Weighted combination
        final_confidence = (
            technical_score * self.weights["technical"] +
            model_confidence * self.weights["model"] +
            rag_similarity * self.weights["rag"]
        )

        # Initial action is the D‑engine action
        action = d_signal.action
        position_size_pct = 1.0

        # Map H‑engine recommendation into BUY/SELL/HOLD if available
        if h_advisory and h_advisory.recommendation:
            h_action = (
                "BUY" if h_advisory.recommendation == "LONG" else
                "SELL" if h_advisory.recommendation == "SHORT" else
                "HOLD"
            )

            if self.require_consensus:
                if action != h_action:
                    reasons.append(f"No consensus: D={action}, H={h_action}")
                    action = "HOLD"
                else:
                    reasons.append(f"Consensus: D={action}, H={h_action}")

            # Apply suggested position size from H‑engine
            position_size_pct = h_advisory.suggested_position_size_pct

        # Determine if trade should be executed
        should_trade = True
        if technical_score < self.min_technical_score:
            should_trade = False
            reasons.append(
                f"Technical below min: {technical_score:.2f} < {self.min_technical_score:.2f}"
            )
        if h_advisory and model_confidence < self.min_model_confidence:
            should_trade = False
            reasons.append(
                f"Model confidence below min: {model_confidence:.2f} < {self.min_model_confidence:.2f}"
            )
        if final_confidence < self.confidence_threshold:
            should_trade = False
            reasons.append(
                f"Final confidence below threshold: {final_confidence:.2f} < {self.confidence_threshold:.2f}"
            )
        if action == "HOLD":
            should_trade = False
            reasons.append("Action is HOLD")

        if should_trade:
            reasons.append(
                f"Trade approved: {action} at {final_confidence:.2f} confidence"
            )

        return ConfidenceResult(
            final_confidence=final_confidence,
            should_trade=should_trade,
            technical_score=technical_score,
            model_confidence=model_confidence,
            rag_similarity=rag_similarity,
            weights=self.weights,
            action=action if should_trade else "HOLD",
            position_size_pct=position_size_pct if should_trade else 0.0,
            reasons=reasons,
        )

    def update_weights(self, weights: Dict[str, float]) -> None:
        """Update and normalise weight parameters on the fly."""
        self.weights.update(weights)
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        logger.info(f"Weights updated: {self.weights}")