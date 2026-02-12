"""AWS Bedrock LLM integration for intelligent trading decisions."""
from __future__ import annotations

__all__ = [
    "BedrockClient",
    "TradeAdvisor", 
    "TradeRecommendation",
    "PerformanceAnalyzer",
    "AdaptiveLearningEngine",
    "ConfigurationManager",
    "WeeklyReviewEngine",
    "AutonomousTradingOrchestrator",
    "SafetyConstraints",
    "StrategyAdjustment",
    "RAGEngine",
    # Hybrid Bedrock Architecture
    "HybridBedrockClient",
    "EventDetector",
    "RAGContextBuilder",
    "BedrockSQLiteManager",
]

# Hybrid Bedrock Architecture imports
from .bedrock_hybrid_client import HybridBedrockClient
from .event_detector import EventDetector
from .rag_context_builder import RAGContextBuilder
from .sqlite_manager import BedrockSQLiteManager
