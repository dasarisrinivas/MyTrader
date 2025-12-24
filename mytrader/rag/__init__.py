"""RAG (Retrieval-Augmented Generation) module for MyTrader.

This module provides the infrastructure for:
- S3-based RAG document storage and retrieval
- FAISS vector embeddings (stored in S3)
- Daily market summary generation
- Trade logging with full metadata
- Mistake analysis for losing trades
- 3-layer hybrid pipeline (Rules → RAG → LLM)
- Integration with live trading manager

All data is stored in AWS S3:
- Bucket: rag-bot-storage
- Prefix: spy-futures-bot/
"""

from mytrader.rag.s3_storage import (
    S3Storage,
    S3StorageWithCache,
    S3StorageError,
    get_s3_storage,
    save_to_s3,
    read_from_s3,
)
from mytrader.rag.rag_storage_manager import (
    RAGStorageManager,
    TradeRecord,
    get_rag_storage,
)
from mytrader.rag.embedding_builder import (
    EmbeddingBuilder,
    create_embedding_builder,
)
from mytrader.rag.rag_daily_updater import (
    RAGDailyUpdater,
    create_daily_updater,
)
from mytrader.rag.hybrid_rag_pipeline import (
    HybridRAGPipeline,
    RuleEngine,
    RAGRetriever,
    LLMDecisionMaker,
    TradeAction,
    HybridPipelineResult,
    create_hybrid_pipeline,
)
from mytrader.rag.trade_logger import (
    TradeLogger,
    get_trade_logger,
)
from mytrader.rag.mistake_analyzer import (
    MistakeAnalyzer,
    get_mistake_analyzer,
)
from mytrader.rag.pipeline_integration import (
    HybridPipelineIntegration,
    HybridSignal,
    create_hybrid_integration,
)

__all__ = [
    # S3 Storage
    "S3Storage",
    "S3StorageWithCache",
    "S3StorageError",
    "get_s3_storage",
    "save_to_s3",
    "read_from_s3",
    # RAG Storage Manager
    "RAGStorageManager",
    "TradeRecord",
    "get_rag_storage",
    # Embeddings
    "EmbeddingBuilder",
    "create_embedding_builder",
    # Daily updater
    "RAGDailyUpdater",
    "create_daily_updater",
    # Pipeline
    "HybridRAGPipeline",
    "RuleEngine",
    "RAGRetriever",
    "LLMDecisionMaker",
    "TradeAction",
    "HybridPipelineResult",
    "create_hybrid_pipeline",
    # Trade logging
    "TradeLogger",
    "get_trade_logger",
    # Mistake analysis
    "MistakeAnalyzer",
    "get_mistake_analyzer",
    # Integration
    "HybridPipelineIntegration",
    "HybridSignal",
    "create_hybrid_integration",
]
