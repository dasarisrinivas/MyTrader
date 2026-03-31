"""Root Settings dataclass — aggregates all sub-config dataclasses.

This is the single object that gets loaded from YAML by settings_loader.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import logging

from .data_sources import DataSourceConfig
from .strategy import (
    EntryFilterConfig,
    OneMinuteStrategyConfig,
    ThirtyMinuteStrategyConfig,
    StrategyConfig,
)
from .risk import RiskGateConfig, TradingConfig
from .backtest import BacktestConfig, OptimizationConfig
from .llm_rag import LLMConfig, RAGConfig, HybridConfig, AWSAgentsConfig
from .integrations import (
    StockwitsSentimentConfig,
    MultiSourceSentimentConfig,
    VixFeedConfig,
    TelegramConfig,
)
from .misc import LearningConfig, FeatureFlagsConfig, ObservabilityConfig, DynamicSupportConfig
from .gold import GoldStrategyConfig
from .spy_options import SpyOptionsConfig

@dataclass
class Settings:
    data: DataSourceConfig = field(default_factory=DataSourceConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    one_minute: OneMinuteStrategyConfig = field(default_factory=OneMinuteStrategyConfig)
    risk_gate: RiskGateConfig = field(default_factory=RiskGateConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    strategies: List[StrategyConfig] = field(default_factory=list)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    stocktwits_sentiment: StockwitsSentimentConfig = field(default_factory=StockwitsSentimentConfig)
    multi_source_sentiment: MultiSourceSentimentConfig = field(default_factory=MultiSourceSentimentConfig)
    vix_feed: VixFeedConfig = field(default_factory=VixFeedConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    aws_agents: AWSAgentsConfig = field(default_factory=AWSAgentsConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    features: FeatureFlagsConfig = field(default_factory=FeatureFlagsConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    dynamic_support: DynamicSupportConfig = field(default_factory=DynamicSupportConfig)
    gold: GoldStrategyConfig = field(default_factory=GoldStrategyConfig)
    spy_options: SpyOptionsConfig = field(default_factory=SpyOptionsConfig)

    def validate(self) -> None:
        import logging
        logger = logging.getLogger(__name__)
        
        if self.trading.initial_capital <= 0:
            raise ValueError("initial capital must be positive")
        if self.trading.max_position_size <= 0:
            raise ValueError("max position size must be positive")
        if self.trading.tick_size <= 0:
            raise ValueError("tick size must be positive")
        if self.backtest.slippage < 0:
            raise ValueError("slippage cannot be negative")
        if self.trading.max_contracts_limit > 5:
             # Enforce hard cap in code even if env var tries to override
             self.trading.max_contracts_limit = 5
        
        # ============================================================
        # CRITICAL: Consolidate risk limits to use MOST CONSERVATIVE values
        # This prevents dangerous inconsistencies between RiskGateConfig 
        # and TradingConfig that could allow excessive risk.
        # See review.md: "Inconsistent Risk Limits"
        # ============================================================
        
        # Max contracts: use minimum of all sources
        risk_gate_contracts = self.risk_gate.max_contracts
        trading_max_pos = self.trading.max_position_size
        trading_contracts_limit = self.trading.max_contracts_limit
        
        conservative_max_contracts = min(
            risk_gate_contracts,
            trading_max_pos, 
            trading_contracts_limit
        )
        
        if conservative_max_contracts != trading_max_pos or conservative_max_contracts != trading_contracts_limit:
            logger.warning(
                f"RISK CONSOLIDATION: Max contracts mismatch detected. "
                f"RiskGate={risk_gate_contracts}, TradingConfig.max_position_size={trading_max_pos}, "
                f"TradingConfig.max_contracts_limit={trading_contracts_limit}. "
                f"Using most conservative: {conservative_max_contracts}"
            )
            self.trading.max_position_size = conservative_max_contracts
            self.trading.max_contracts_limit = conservative_max_contracts
        
        # Daily loss limit: use minimum (more conservative) value
        risk_gate_daily_loss = self.risk_gate.daily_max_loss_usd
        trading_daily_loss = self.trading.max_daily_loss
        
        conservative_daily_loss = min(risk_gate_daily_loss, trading_daily_loss)
        
        if risk_gate_daily_loss != trading_daily_loss:
            logger.warning(
                f"RISK CONSOLIDATION: Daily loss limit mismatch detected. "
                f"RiskGate=${risk_gate_daily_loss}, TradingConfig=${trading_daily_loss}. "
                f"Using most conservative: ${conservative_daily_loss}"
            )
            self.trading.max_daily_loss = conservative_daily_loss
        
        # Log final consolidated values for audit trail
        logger.info(
            f"RISK LIMITS CONSOLIDATED: max_contracts={conservative_max_contracts}, "
            f"daily_max_loss=${conservative_daily_loss}"
        )

