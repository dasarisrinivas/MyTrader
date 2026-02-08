"""LLM, RAG, Hybrid pipeline, and AWS Agents configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os
from datetime import time

@dataclass
class LLMConfig:
    """Configuration for AWS Bedrock LLM integration."""
    enabled: bool = field(default_factory=lambda: os.environ.get("LLM_ENABLED", "False").lower() == "true")
    model_id: str = field(default_factory=lambda: os.environ.get("LLM_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"))
    region_name: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1"))
    max_tokens: int = 2048
    temperature: float = 0.3
    min_confidence_threshold: float = 0.7
    override_mode: bool = False
    call_interval_seconds: int = 60
    enable_sentiment: bool = True
    sentiment_region: str = "us-east-1"
    s3_bucket: str = field(default_factory=lambda: os.environ.get("S3_BUCKET", ""))
    s3_prefix: str = "llm-training-data"
    retrain_interval_days: int = 7
    min_training_trades: int = 100
    trade_log_db_path: str = "data/llm_trades.db"
    
    use_background_thread: bool = True
    cache_timeout_seconds: int = 300


@dataclass
class RAGConfig:
    """Configuration for Retrieval-Augmented Generation (RAG)."""
    enabled: bool = field(default_factory=lambda: os.environ.get("RAG_ENABLED", "False").lower() == "true")
    backend: str = field(default_factory=lambda: os.environ.get("RAG_BACKEND", "local_faiss").lower())
    opensearch_enabled: bool = field(default_factory=lambda: os.environ.get("OPENSEARCH_ENABLED", "False").lower() in {"1", "true", "yes"})
    embedding_model_id: str = "amazon.titan-embed-text-v1"
    region_name: str = "us-east-1"
    vector_store_path: str = "data/rag_index"
    embedding_dimension: int = 1536
    top_k_results: int = 3
    score_threshold: float = 0.5
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    batch_size: int = 10
    knowledge_base_path: str = "data/knowledge_base"
    local_store_path: str = "rag_data/local_kb/local_kb.sqlite"
    kb_cache_ttl_seconds: int = 120
    min_similar_trades: int = field(default_factory=lambda: int(os.environ.get("MIN_SIMILAR_TRADES", "2")))
    min_win_rate: float = field(default_factory=lambda: float(os.environ.get("MIN_WIN_RATE", "0.15")))
    min_weighted_win_rate: float = field(default_factory=lambda: float(os.environ.get("MIN_WEIGHTED_WIN_RATE", "0.45")))
    min_weighted_win_rate_soft_floor: float = field(
        default_factory=lambda: float(
            os.environ.get(
                "MIN_WEIGHTED_WIN_RATE_SOFT_FLOOR",
                os.environ.get("MIN_WEIGHTED_WIN_RATE", "0.45"),
            )
        )
    )
    min_similar_trades_for_full_threshold: int = field(
        default_factory=lambda: int(os.environ.get("MIN_SIMILAR_TRADES_FOR_FULL_THRESHOLD", "0"))
    )
    min_sample_for_hard_block: int = field(
        default_factory=lambda: int(os.environ.get("RAG_MIN_SAMPLE_FOR_HARD_BLOCK", "30"))
    )
    soft_penalty_when_below: float = field(
        default_factory=lambda: float(os.environ.get("RAG_SOFT_PENALTY_WHEN_BELOW", "0.10"))
    )
    hard_block_when_below: bool = field(
        default_factory=lambda: os.environ.get("RAG_HARD_BLOCK_WHEN_BELOW", "False").lower() in {"1", "true", "yes"}
    )
    regime_mode: str = field(default_factory=lambda: os.environ.get("RAG_REGIME_MODE", "relaxed"))

    def __post_init__(self) -> None:
        self.backend = (self.backend or "off").lower()
        if self.min_weighted_win_rate_soft_floor > self.min_weighted_win_rate:
            self.min_weighted_win_rate_soft_floor = self.min_weighted_win_rate




@dataclass
class HybridConfig:
    """Configuration for Hybrid RAG + LLM Pipeline (3-layer decision system)."""
    # Master enable/disable
    enabled: bool = field(default_factory=lambda: os.environ.get("HYBRID_ENABLED", "True").lower() == "true")
    # Level confirmation gate
    level_confirmation_enabled: bool = True
    level_confirm_proximity_pct: float = 0.30
    level_confirm_buffer_atr_mult: float = 0.10
    level_confirm_min_buffer_points: float = 0.0
    level_confirm_max_wait_candles: int = 6
    level_confirm_timeout_mode: str = "SOFT_PENALTY"  # or "DISABLE"
    level_confirm_timeout_penalty: float = 0.12
    
    # D-Engine (Deterministic Rules) settings
    candidate_threshold: float = 0.55  # Minimum D-engine score to proceed to RAG
    atr_min: float = 0.15  # Minimum ATR threshold (lowered for low-vol markets)
    atr_max: float = 20.0  # Maximum ATR threshold (Increased for ES volatility)
    chop_ema_spread_min_pct: float = 0.0005  # EMA spread % threshold for CHOP_RANGE filter
    
    # H-Engine (LLM + RAG) settings  
    max_calls_per_hour: int = 10
    min_interval_seconds: int = 60
    top_k: int = 5  # RAG retrieval count
    cache_ttl_seconds: int = 300
    cooldown_minutes: int = 15
    allow_legacy_fallback: bool = False
    llm_uncertainty_band_low: float = 0.35
    llm_uncertainty_band_high: float = 0.65
    llm_call_cooldown_seconds: int = 60
    llm_response_cache_ttl_seconds: int = 900
    
    # Confidence thresholds
    min_confidence_threshold: float = 0.60
    signal_threshold: int = 40
    oversold_extension_rsi_min: float = 40.0
    no_signal_allow_weak_signals: bool = False
    no_signal_allow_chop_bias: bool = False
    min_confidence_for_trade: int = 25  # ADDED: Minimum confidence % for trade execution (25 = 25%)
    
    # RAG data paths
    rag_data_path: str = "rag_data"


@dataclass
class AWSAgentsConfig:
    """Configuration for AWS Bedrock Agents (multi-agent decision system)."""
    # Master enable/disable
    enabled: bool = field(default_factory=lambda: os.environ.get("AWS_AGENTS_ENABLED", "False").lower() == "true")
    block_on_wait: bool = True
    wait_override_confidence: float = 0.75
    
    # Configuration source
    use_deployed_config: bool = True  # Auto-load from deployed_resources.yaml
    config_path: str = "aws/config/deployed_resources.yaml"
    
    # Manual configuration (used if use_deployed_config=false)
    region_name: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1"))
    
    # Agent IDs (filled automatically from deployed_resources.yaml)
    data_agent_id: str = ""
    data_agent_alias: str = ""
    decision_agent_id: str = ""
    decision_agent_alias: str = ""
    risk_agent_id: str = ""
    risk_agent_alias: str = ""
    learning_agent_id: str = ""
    learning_agent_alias: str = ""
    
    # Knowledge Base
    knowledge_base_id: str = ""



