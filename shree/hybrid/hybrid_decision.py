"""Hybrid Decision Engine - Main orchestrator for D-engine + H-engine.

This is the main entry point for the hybrid trading architecture.
It coordinates:
1. D-engine (deterministic) for real-time decisions
2. H-engine (LLM + RAG) for event-triggered confirmation
3. Confidence scoring for final decision
4. Safety management for production guards
5. Decision logging for auditability
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd

from ..utils.logger import logger
from .d_engine import DeterministicEngine, DEngineSignal
from .h_engine import HeuristicEngine, HEngineAdvisory
from .confidence import ConfidenceScorer, ConfidenceResult
from .safety import SafetyManager, SafetyCheck
from .decision_logger import DecisionLogger


@dataclass
class HybridDecision:
    """Final decision from the hybrid engine."""
    
    # Final action
    action: str  # "BUY", "SELL", "HOLD"
    should_execute: bool
    
    # Confidence
    final_confidence: float
    position_size_pct: float
    
    # Prices
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Components
    d_signal: Optional[DEngineSignal] = None
    h_advisory: Optional[HEngineAdvisory] = None
    confidence_result: Optional[ConfidenceResult] = None
    safety_check: Optional[SafetyCheck] = None
    
    # Execution tracking
    order_id: Optional[str] = None
    execution_status: str = "pending"
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "should_execute": self.should_execute,
            "final_confidence": self.final_confidence,
            "position_size_pct": self.position_size_pct,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "timestamp": self.timestamp.isoformat(),
            "reasons": self.reasons,
            "order_id": self.order_id,
            "execution_status": self.execution_status,
        }


class HybridDecisionEngine:
    """Main orchestrator for hybrid trading decisions.
    
    Flow:
    1. D-engine evaluates on every candle close
    2. If D-engine produces candidate (score >= threshold), invoke H-engine
    3. H-engine queries RAG + calls LLM for confirmation
    4. Confidence scorer combines all scores
    5. Safety manager validates against limits
    6. If all checks pass, execute trade
    7. Log everything for audit
    
    Key principles:
    - D-engine ALWAYS runs (fast, deterministic)
    - H-engine runs ONLY on candidates (cost control)
    - LLM provides bias/confirmation, NOT overrides
    - All decisions logged for analysis
    """
    
    def __init__(
        self,
        d_engine: Optional[DeterministicEngine] = None,
        h_engine: Optional[HeuristicEngine] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        safety_manager: Optional[SafetyManager] = None,
        decision_logger: Optional[DecisionLogger] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize hybrid decision engine.
        
        Args:
            d_engine: Deterministic engine (creates default if None)
            h_engine: Heuristic engine (creates default if None)
            confidence_scorer: Confidence scorer (creates default if None)
            safety_manager: Safety manager (creates default if None)
            decision_logger: Decision logger (creates default if None)
            config: Configuration overrides
        """
        self.config = config or {}
        
        # Initialize components
        self.d_engine = d_engine or DeterministicEngine(
            config=self.config.get("d_engine")
        )
        
        self.h_engine = h_engine or HeuristicEngine(
            config=self.config.get("h_engine")
        )
        
        self.confidence_scorer = confidence_scorer or ConfidenceScorer(
            weights=self.config.get("confidence", {}).get("weights"),
            confidence_threshold=self.config.get("confidence", {}).get("threshold", 0.60),
        )
        
        self.safety_manager = safety_manager or SafetyManager(
            cooldown_minutes=self.config.get("safety", {}).get("order_cooldown_minutes", 5),
            max_orders_per_window=self.config.get("safety", {}).get("max_orders_per_15min", 3),
            dry_run=self.config.get("dry_run", False),
        )
        
        self.decision_logger = decision_logger or DecisionLogger()
        
        # Statistics
        self._total_evaluations = 0
        self._candidates_generated = 0
        self._h_engine_calls = 0
        self._trades_executed = 0
        
        logger.info("HybridDecisionEngine initialized")
    
    def set_price_levels(
        self,
        pdh: Optional[float] = None,
        pdl: Optional[float] = None,
        week_high: Optional[float] = None,
        week_low: Optional[float] = None,
    ):
        """Update price levels for D-engine.
        
        Args:
            pdh: Previous day high
            pdl: Previous day low
            week_high: Current week high
            week_low: Current week low
        """
        self.d_engine.set_levels(pdh, pdl, week_high, week_low)
    
    def evaluate(
        self,
        features: pd.DataFrame,
        current_price: float,
        candle_time: datetime,
        force_h_engine: bool = False,
    ) -> HybridDecision:
        """Evaluate market data and produce trading decision.
        
        This is the main entry point, called on each candle close.
        
        Args:
            features: DataFrame with OHLCV and indicators
            current_price: Current market price
            candle_time: Candle timestamp
            force_h_engine: If True, always invoke H-engine (for testing)
            
        Returns:
            HybridDecision with action, confidence, and metadata
        """
        self._total_evaluations += 1
        reasons = []
        
        # Step 1: D-engine evaluation (ALWAYS runs)
        d_signal = self.d_engine.evaluate(
            features=features,
            current_price=current_price,
            candle_time=candle_time,
        )
        
        reasons.extend(d_signal.reasons)
        
        # Step 2: Check if D-engine produced a candidate
        h_advisory = None
        
        if d_signal.is_candidate or force_h_engine:
            self._candidates_generated += 1
            
            # Step 3: Invoke H-engine for LLM + RAG confirmation
            h_advisory = self.h_engine.evaluate(d_signal)
            self._h_engine_calls += 1
            
            if h_advisory:
                reasons.append(f"H-engine: {h_advisory.recommendation} (conf={h_advisory.model_confidence:.2f})")
                if h_advisory.explanation:
                    reasons.append(f"LLM: {h_advisory.explanation[:100]}")
        else:
            reasons.append("Not a candidate - H-engine skipped")
        
        # Step 4: Calculate final confidence
        confidence_result = self.confidence_scorer.calculate(
            d_signal=d_signal,
            h_advisory=h_advisory,
        )
        
        reasons.extend(confidence_result.reasons)
        
        # Step 5: Safety checks
        safety_check = self.safety_manager.check_all()
        
        if not safety_check.is_safe:
            reasons.append(f"Safety blocked: {safety_check.reason}")
        
        # Step 6: Final decision
        should_execute = (
            confidence_result.should_trade and
            safety_check.is_safe and
            not self.safety_manager.dry_run
        )
        
        # Build decision
        decision = HybridDecision(
            action=confidence_result.action,
            should_execute=should_execute,
            final_confidence=confidence_result.final_confidence,
            position_size_pct=confidence_result.position_size_pct,
            entry_price=d_signal.entry_price,
            stop_loss=d_signal.stop_loss,
            take_profit=d_signal.take_profit,
            d_signal=d_signal,
            h_advisory=h_advisory,
            confidence_result=confidence_result,
            safety_check=safety_check,
            reasons=reasons,
        )
        
        # Step 7: Log decision
        self.decision_logger.log_decision(
            d_signal=d_signal,
            h_advisory=h_advisory,
            confidence_result=confidence_result,
            safety_check=safety_check,
        )
        
        # Log summary
        if should_execute:
            logger.info(
                f"🟢 TRADE DECISION: {decision.action} @ {current_price:.2f} "
                f"(conf={decision.final_confidence:.2f}, size={decision.position_size_pct:.0%})"
            )
        elif d_signal.is_candidate:
            logger.info(
                f"🟡 NO TRADE (candidate): {d_signal.action} blocked - "
                f"{safety_check.reason if not safety_check.is_safe else 'confidence/consensus'}"
            )
        else:
            logger.debug(f"⚪ No action: {d_signal.action} (score={d_signal.technical_score:.2f})")
        
        return decision
    
    def record_execution(
        self,
        decision: HybridDecision,
        order_id: str,
        status: str,
        fill_price: Optional[float] = None,
    ):
        """Record trade execution result.
        
        Args:
            decision: The decision that was executed
            order_id: Order ID
            status: Execution status
            fill_price: Fill price if executed
        """
        decision.order_id = order_id
        decision.execution_status = status
        
        if status in ("filled", "partially_filled"):
            self._trades_executed += 1
            
            # Record in safety manager
            self.safety_manager.record_trade(
                action=decision.action,
                quantity=int(decision.position_size_pct * 1),  # Simplified
                price=fill_price or decision.entry_price,
                order_id=order_id,
            )
        
        logger.info(f"Execution recorded: {order_id} - {status}")
    
    def update_pnl(self, realized_pnl: float, unrealized_pnl: float = 0.0):
        """Update P&L in safety manager.
        
        Args:
            realized_pnl: Realized P&L
            unrealized_pnl: Unrealized P&L
        """
        self.safety_manager.update_pnl(realized_pnl, unrealized_pnl)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_evaluations": self._total_evaluations,
            "candidates_generated": self._candidates_generated,
            "h_engine_calls": self._h_engine_calls,
            "trades_executed": self._trades_executed,
            "candidate_rate": self._candidates_generated / self._total_evaluations if self._total_evaluations > 0 else 0,
            "h_engine_stats": self.h_engine.get_call_stats(),
            "safety_stats": self.safety_manager.get_stats(),
            "decision_stats": self.decision_logger.get_stats(),
        }
    
    def emergency_stop(self, reason: str):
        """Trigger emergency stop.
        
        Args:
            reason: Reason for emergency stop
        """
        self.safety_manager.trigger_emergency_stop(reason)
    
    def is_dry_run(self) -> bool:
        """Check if operating in dry-run mode."""
        return self.safety_manager.dry_run
