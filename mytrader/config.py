"""Application configuration and settings management."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os


@dataclass
class DataSourceConfig:
    tradingview_webhook_url: Optional[str] = field(default_factory=lambda: os.environ.get("TRADINGVIEW_WEBHOOK_URL"))
    tradingview_symbol: str = field(default_factory=lambda: os.environ.get("TRADINGVIEW_SYMBOL", "ES"))
    tradingview_interval: str = field(default_factory=lambda: os.environ.get("TRADINGVIEW_INTERVAL", "1m"))

    ibkr_host: str = field(default_factory=lambda: os.environ.get("IBKR_HOST", "127.0.0.1"))
    ibkr_port: int = field(default_factory=lambda: int(os.environ.get("IBKR_PORT", "4002")))
    ibkr_client_id: int = field(default_factory=lambda: int(os.environ.get("IBKR_CLIENT_ID", "1")))
    ibkr_symbol: str = field(default_factory=lambda: os.environ.get("IBKR_SYMBOL", "ES"))
    ibkr_exchange: str = field(default_factory=lambda: os.environ.get("IBKR_EXCHANGE", "CME"))
    ibkr_currency: str = field(default_factory=lambda: os.environ.get("IBKR_CURRENCY", "USD"))

    twitter_bearer_token: Optional[str] = field(default_factory=lambda: os.environ.get("TWITTER_BEARER_TOKEN"))
    news_api_keys: List[str] = field(default_factory=lambda: os.environ.get("NEWS_API_KEYS", "").split(",") if os.environ.get("NEWS_API_KEYS") else [])
    sentiment_refresh_interval: int = 60  # seconds


@dataclass
class TradingConfig:
    max_position_size: int = field(default_factory=lambda: int(os.environ.get("MAX_POSITION_SIZE", "5")))
    contracts_per_order: int = 1
    max_daily_loss: float = field(default_factory=lambda: float(os.environ.get("MAX_DAILY_LOSS", "2000.0")))
    max_loss_per_trade: float = field(default_factory=lambda: float(os.environ.get("MAX_LOSS_PER_TRADE", "1250.0")))
    max_daily_trades: int = 20
    initial_capital: float = field(default_factory=lambda: float(os.environ.get("INITIAL_CAPITAL", "100000.0")))
    contract_multiplier: float = 50.0
    stop_loss_ticks: float = 10.0
    take_profit_ticks: float = 20.0
    tick_size: float = 0.25
    tick_value: float = 12.5
    commission_per_contract: float = 2.4
    
    # Position sizing method: "fixed_fraction" or "kelly"
    position_sizing_method: str = "fixed_fraction"
    # Risk percentage per trade for fixed fractional sizing (0.005 = 0.5%, 0.01 = 1%)
    risk_per_trade_pct: float = 0.005
    
    # Safety parameters
    disaster_stop_pct: float = 0.007
    max_trade_duration_minutes: int = 60
    trade_cooldown_minutes: int = 5
    
    # Indicator warm-up
    min_bars_for_signals: int = 200
    
    # Market regime filter thresholds
    min_atr_threshold: float = 0.5
    max_spread_ticks: int = 1
    max_loop_latency_seconds: float = 3.0
    
    # Weighted voting thresholds
    min_weighted_confidence: float = 0.70
    confidence_threshold: float = field(default_factory=lambda: float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7")))

    # Hard Safety Constraints
    max_contracts_limit: int = field(default_factory=lambda: int(os.environ.get("MAX_CONTRACTS", "5")))
    margin_limit_pct: float = field(default_factory=lambda: float(os.environ.get("MARGIN_LIMIT_PCT", "0.80")))
    decision_min_interval_seconds: int = field(default_factory=lambda: int(os.environ.get("DECISION_MIN_INTERVAL_SECONDS", "30")))
    order_retry_limit: int = field(default_factory=lambda: int(os.environ.get("ORDER_RETRY_LIMIT", "3")))
    contract_month_offset: int = field(default_factory=lambda: int(os.environ.get("CONTRACT_MONTH_OFFSET", "0")))


@dataclass
class BacktestConfig:
    data_path: Path = Path("data/historical_spy_es.parquet")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = 100_000.0
    slippage: float = 0.25
    risk_free_rate: float = 0.02


@dataclass
class OptimizationConfig:
    window_length: int = 5000
    retrain_interval: int = 60 * 60
    parameter_grid: dict = field(default_factory=lambda: {
        "rsi_period": [14, 21, 28],
        "macd_fast": [12, 16, 20],
        "macd_slow": [26, 30, 35],
        "sentiment_threshold": [0.4, 0.5, 0.6],
        "momentum_lookback": [20, 30, 40],
    })
    enable_daily_optimization: bool = True
    optimized_params_path: str = "data/optimized_params.json"
    optimization_hour: int = 16


@dataclass
class StrategyConfig:
    name: str = "rsi_macd_sentiment"
    enabled: bool = True
    params: dict = field(default_factory=dict)


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
    min_weighted_win_rate: float = field(default_factory=lambda: float(os.environ.get("MIN_WEIGHTED_WIN_RATE", "0.45")))

    def __post_init__(self) -> None:
        self.backend = (self.backend or "off").lower()


@dataclass
class TelegramConfig:
    """Configuration for Telegram notifications."""
    enabled: bool = field(default_factory=lambda: os.environ.get("TELEGRAM_ENABLED", "False").lower() == "true")
    bot_token: str = field(default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    chat_id: str = field(default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID", ""))
    notify_on_trade: bool = True
    notify_on_signal: bool = False
    notify_on_error: bool = True


@dataclass
class HybridConfig:
    """Configuration for Hybrid RAG + LLM Pipeline (3-layer decision system)."""
    # Master enable/disable
    enabled: bool = field(default_factory=lambda: os.environ.get("HYBRID_ENABLED", "True").lower() == "true")
    
    # D-Engine (Deterministic Rules) settings
    candidate_threshold: float = 0.55  # Minimum D-engine score to proceed to RAG
    atr_min: float = 0.15  # Minimum ATR threshold (lowered for low-vol markets)
    atr_max: float = 5.0   # Maximum ATR threshold
    
    # H-Engine (LLM + RAG) settings  
    max_calls_per_hour: int = 10
    min_interval_seconds: int = 60
    top_k: int = 5  # RAG retrieval count
    cache_ttl_seconds: int = 300
    cooldown_minutes: int = 15
    
    # Confidence thresholds
    min_confidence_threshold: float = 0.60
    signal_threshold: int = 40
    
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


@dataclass
class LearningConfig:
    """Local learning/ingestion hooks."""
    enabled: bool = True
    outcomes_dir: str = "rag_data/training/trade_outcomes"
    history_dir: str = "rag_data/history_snapshots"
    history_days: int = 45
    ingest_window_days: int = 60
    reason_code_retention_days: int = 90


@dataclass
class FeatureFlagsConfig:
    """Feature flags to safely roll out guardrails."""
    enforce_entry_risk_checks: bool = field(default_factory=lambda: os.environ.get("FF_ENTRY_RISK_GUARDS", "true").lower() not in {"0", "false", "no"})
    enforce_wait_blocking: bool = field(default_factory=lambda: os.environ.get("FF_WAIT_BLOCKING", "true").lower() not in {"0", "false", "no"})
    enforce_reduce_only_exits: bool = field(default_factory=lambda: os.environ.get("FF_EXIT_GUARDS", "true").lower() not in {"0", "false", "no"})
    enable_learning_hooks: bool = field(default_factory=lambda: os.environ.get("FF_LEARNING_HOOKS", "true").lower() not in {"0", "false", "no"})


@dataclass
class Settings:
    data: DataSourceConfig = field(default_factory=DataSourceConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    strategies: List[StrategyConfig] = field(default_factory=list)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    aws_agents: AWSAgentsConfig = field(default_factory=AWSAgentsConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    features: FeatureFlagsConfig = field(default_factory=FeatureFlagsConfig)

    def validate(self) -> None:
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
