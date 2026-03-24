"""Configuration package — split from the monolithic config.py.

Provides backward-compatible re-exports so existing imports like::

    from shree.config import Settings, TradingConfig

continue to work unchanged.

Submodules
----------
data_sources    DataSourceConfig
strategy        EntryFilterConfig, OneMinuteStrategyConfig, ThirtyMinuteStrategyConfig, StrategyConfig
risk            RiskGateConfig, TradingConfig
backtest        BacktestConfig, OptimizationConfig
llm_rag         LLMConfig, RAGConfig, HybridConfig, AWSAgentsConfig
integrations    StockwitsSentimentConfig, MultiSourceSentimentConfig, VixFeedConfig, TelegramConfig
misc            LearningConfig, FeatureFlagsConfig, ObservabilityConfig
settings        Settings (root aggregator)
"""

from .data_sources import DataSourceConfig  # noqa: F401
from .strategy import (  # noqa: F401
    EntryFilterConfig,
    OneMinuteStrategyConfig,
    ThirtyMinuteStrategyConfig,
    StrategyConfig,
)
from .risk import RiskGateConfig, TradingConfig  # noqa: F401
from .backtest import BacktestConfig, OptimizationConfig  # noqa: F401
from .llm_rag import LLMConfig, RAGConfig, HybridConfig, AWSAgentsConfig  # noqa: F401
from .integrations import (  # noqa: F401
    StockwitsSentimentConfig,
    MultiSourceSentimentConfig,
    VixFeedThresholds,
    VixFeedConfig,
    TelegramConfig,
)
from .misc import LearningConfig, FeatureFlagsConfig, ObservabilityConfig, DynamicSupportConfig  # noqa: F401
from .gold import GoldStrategyConfig  # noqa: F401
from .settings import Settings  # noqa: F401

__all__ = [
    "DataSourceConfig",
    "EntryFilterConfig",
    "OneMinuteStrategyConfig",
    "ThirtyMinuteStrategyConfig",
    "StrategyConfig",
    "RiskGateConfig",
    "TradingConfig",
    "BacktestConfig",
    "OptimizationConfig",
    "LLMConfig",
    "RAGConfig",
    "HybridConfig",
    "AWSAgentsConfig",
    "StockwitsSentimentConfig",
    "MultiSourceSentimentConfig",
    "VixFeedThresholds",
    "VixFeedConfig",
    "TelegramConfig",
    "LearningConfig",
    "FeatureFlagsConfig",
    "ObservabilityConfig",
    "DynamicSupportConfig",
    "GoldStrategyConfig",
    "Settings",
]
