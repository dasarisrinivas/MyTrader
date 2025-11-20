"""Application configuration and settings management."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataSourceConfig:
    tradingview_webhook_url: Optional[str] = None
    tradingview_symbol: str = "ES"  # E-mini S&P 500 futures ticker on TradingView
    tradingview_interval: str = "1m"

    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 4002  # IB Gateway default (7497 for TWS)
    ibkr_client_id: int = 1
    ibkr_symbol: str = "ES"
    ibkr_exchange: str = "CME"  # CME exchange (not GLOBEX) for ES futures
    ibkr_currency: str = "USD"

    twitter_bearer_token: Optional[str] = None
    news_api_keys: List[str] = field(default_factory=list)
    sentiment_refresh_interval: int = 60  # seconds


@dataclass
class TradingConfig:
    max_position_size: int = 5  # Maximum total contracts
    contracts_per_order: int = 1  # Contracts per single order
    max_daily_loss: float = 2000.0
    max_daily_trades: int = 20
    initial_capital: float = 100_000.0
    contract_multiplier: float = 50.0  # ES futures multiplier
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
    disaster_stop_pct: float = 0.007  # 0.7% emergency stop
    max_trade_duration_minutes: int = 60  # Exit if trade open > 60 minutes
    trade_cooldown_minutes: int = 5  # Wait 5 minutes after each trade
    
    # Indicator warm-up
    min_bars_for_signals: int = 200  # Increased from 15 to 200
    
    # Market regime filter thresholds
    min_atr_threshold: float = 0.5  # Minimum ATR for trading
    max_spread_ticks: int = 1  # Maximum bid/ask spread in ticks
    max_loop_latency_seconds: float = 3.0  # Max acceptable loop iteration time
    
    # Weighted voting thresholds
    min_weighted_confidence: float = 0.70  # Minimum confidence for weighted entry


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
    retrain_interval: int = 60 * 60  # seconds (only used for legacy mode)
    parameter_grid: dict = field(default_factory=lambda: {
        "rsi_period": [14, 21, 28],
        "macd_fast": [12, 16, 20],
        "macd_slow": [26, 30, 35],
        "sentiment_threshold": [0.4, 0.5, 0.6],
        "momentum_lookback": [20, 30, 40],
    })
    # Daily optimization settings (run after market close)
    enable_daily_optimization: bool = True
    optimized_params_path: str = "data/optimized_params.json"
    optimization_hour: int = 16  # Run at 4 PM ET (after market close)


@dataclass
class StrategyConfig:
    name: str = "rsi_macd_sentiment"
    enabled: bool = True
    params: dict = field(default_factory=dict)


@dataclass
class LLMConfig:
    """Configuration for AWS Bedrock LLM integration."""
    enabled: bool = False
    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    region_name: str = "us-east-1"
    max_tokens: int = 2048
    temperature: float = 0.3
    min_confidence_threshold: float = 0.7
    override_mode: bool = False  # MUST be False - LLM is commentary only
    call_interval_seconds: int = 60  # Minimum seconds between LLM calls
    enable_sentiment: bool = True
    sentiment_region: str = "us-east-1"
    s3_bucket: str = ""
    s3_prefix: str = "llm-training-data"
    retrain_interval_days: int = 7
    min_training_trades: int = 100
    trade_log_db_path: str = "data/llm_trades.db"
    
    # LLM runs in background worker thread
    use_background_thread: bool = True
    cache_timeout_seconds: int = 300  # Cache LLM output for 5 minutes


@dataclass
class RAGConfig:
    """Configuration for Retrieval-Augmented Generation (RAG)."""
    enabled: bool = False
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


@dataclass
class Settings:
    data: DataSourceConfig = field(default_factory=DataSourceConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    strategies: List[StrategyConfig] = field(default_factory=list)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)

    def validate(self) -> None:
        if self.trading.initial_capital <= 0:
            raise ValueError("initial capital must be positive")
        if self.trading.max_position_size <= 0:
            raise ValueError("max position size must be positive")
        if self.trading.tick_size <= 0:
            raise ValueError("tick size must be positive")
        if self.backtest.slippage < 0:
            raise ValueError("slippage cannot be negative")
