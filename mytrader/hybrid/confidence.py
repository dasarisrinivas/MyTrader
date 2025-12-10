"""Confidence Scorer - Combines technical, model, and RAG scores.

Implements configurable weighted scoring for final trading decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..utils.logger import logger
from .d_engine import DEngineSignal
from .h_engine import HEngineAdvisory


@dataclass
class ConfidenceResult:
    """Result of confidence scoring."""
    
    # Final computed values
    final_confidence: float  # 0.0-1.0 weighted combination
    should_trade: bool       # True if confidence >= threshold
    
    # Component scores
    technical_score: float = 0.0
    model_confidence: float = 0.0
    rag_similarity: float = 0.0
    
    # Weights used
    weights: Dict[str, float] = field(default_factory=dict)
    
    # Action
    action: str = "HOLD"
    position_size_pct: float = 1.0
    
    # Metadata
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
    
    Formula:
    final_confidence = (
        technical_score * technical_weight +
        model_confidence * model_weight +
        rag_similarity * rag_weight
    )
    
    Trade only if:
    - final_confidence >= confidence_threshold
    - D-engine action matches H-engine recommendation (consensus mode)
    """
    
    DEFAULT_WEIGHTS = {
        "technical": 0.5,
        "model": 0.3,
        "rag": 0.2,
    }
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.60,
        require_consensus: bool = True,
        min_technical_score: float = 0.50,
        min_model_confidence: float = 0.40,
    ):
        """Initialize confidence scorer.
        
        Args:
            weights: Weight dictionary (technical, model, rag)
            confidence_threshold: Minimum final confidence to trade
            require_consensus: Require D-engine and H-engine agreement
            min_technical_score: Minimum D-engine score to consider
            min_model_confidence: Minimum LLM confidence to consider
        """
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self.confidence_threshold = confidence_threshold
        self.require_consensus = require_consensus
        self.min_technical_score = min_technical_score
        self.min_model_confidence = min_model_confidence
        
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        
        logger.info(
            f"ConfidenceScorer initialized: weights={self.weights}, "
            f"threshold={confidence_threshold}, consensus={require_consensus}"
        )
    
    def calculate(
        self,
        d_signal: DEngineSignal,
        h_advisory: Optional[HEngineAdvisory] = None,
    ) -> ConfidenceResult:
        """Calculate final confidence and trading decision.
        
        Args:
            d_signal: Signal from D-engine
            h_advisory: Optional advisory from H-engine
            
        Returns:
            ConfidenceResult with final confidence and decision
        """
        reasons = []
        
        # Get component scores
        technical_score = d_signal.technical_score
        
        # Model confidence (default to 0.5 if no H-engine)
        if h_advisory:
            model_confidence = h_advisory.model_confidence
            rag_similarity = h_advisory.rag_similarity_score
        else:
            model_confidence = 0.5  # Neutral when no LLM
            rag_similarity = 0.5   # Neutral when no RAG
            reasons.append("H-engine not invoked - using neutral scores")
        
        # Calculate weighted confidence
        final_confidence = (
            technical_score * self.weights["technical"] +
            model_confidence * self.weights["model"] +
            rag_similarity * self.weights["rag"]
        )
        
        # Determine action
        action = d_signal.action
        position_size_pct = 1.0
        
        # Check H-engine recommendation if available
        if h_advisory and h_advisory.recommendation:
            # Map H-engine recommendation to action
            h_action = "BUY" if h_advisory.recommendation == "LONG" else (
                "SELL" if h_advisory.recommendation == "SHORT" else "HOLD"
            )
            
            # Check consensus
            if self.require_consensus:
                if action != h_action:
                    reasons.append(f"No consensus: D={action}, H={h_action}")
                    action = "HOLD"
                else:
                    reasons.append(f"Consensus: D={action}, H={h_action}")
            
            # Apply position size suggestion
            position_size_pct = h_advisory.suggested_position_size_pct
        
        # Check minimum thresholds
        should_trade = True
        
        if technical_score < self.min_technical_score:
            should_trade = False
            reasons.append(f"Technical below min: {technical_score:.2f} < {self.min_technical_score}")
        
        if h_advisory and model_confidence < self.min_model_confidence:
            should_trade = False
            reasons.append(f"Model confidence below min: {model_confidence:.2f} < {self.min_model_confidence}")
        
        if final_confidence < self.confidence_threshold:
            should_trade = False
            reasons.append(f"Final confidence below threshold: {final_confidence:.2f} < {self.confidence_threshold}")
        
        if action == "HOLD":
            should_trade = False
            reasons.append("Action is HOLD")
        
        if should_trade:
            reasons.append(f"Trade approved: {action} at {final_confidence:.2f} confidence")
        
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
    
    def update_weights(self, weights: Dict[str, float]):
        """Update weights dynamically.
        
        Args:
            weights: New weight values
        """
        self.weights.update(weights)
        
        # Normalize
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        
        logger.info(f"Weights updated: {self.weights}")
