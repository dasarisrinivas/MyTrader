"""Hybrid Decision Engine - D-engine (deterministic) + H-engine (LLM/RAG).

This module implements a two-tier trading decision architecture:
1. D-engine: Fast, deterministic rules for real-time execution
2. H-engine: Event-triggered LLM + RAG for higher-level confirmation

The hybrid approach ensures:
- Fast execution for time-sensitive decisions
- AI-enhanced confirmation for high-stakes trades
- Cost control through event-driven LLM calls
- Full auditability of all decisions
"""

from .hybrid_decision import HybridDecisionEngine, HybridDecision
from .d_engine import DeterministicEngine, DEngineSignal
from .h_engine import HeuristicEngine, HEngineAdvisory
from .confidence import ConfidenceScorer, ConfidenceResult
from .safety import SafetyManager, SafetyCheck
from .decision_logger import DecisionLogger

__all__ = [
    "HybridDecisionEngine",
    "HybridDecision",
    "DeterministicEngine",
    "DEngineSignal",
    "HeuristicEngine",
    "HEngineAdvisory",
    "ConfidenceScorer",
    "ConfidenceResult",
    "SafetyManager",
    "SafetyCheck",
    "DecisionLogger",
]
