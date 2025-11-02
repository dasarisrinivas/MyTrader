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
    ibkr_port: int = 7497
    ibkr_client_id: int = 1
    ibkr_symbol: str = "ES"
    ibkr_exchange: str = "GLOBEX"
    ibkr_currency: str = "USD"

    twitter_bearer_token: Optional[str] = None
    news_api_keys: List[str] = field(default_factory=list)
    sentiment_refresh_interval: int = 60  # seconds


@dataclass
class TradingConfig:
    max_position_size: int = 4
    max_daily_loss: float = 2000.0
    max_daily_trades: int = 20
    initial_capital: float = 100_000.0
    contract_multiplier: float = 50.0  # ES futures multiplier
    stop_loss_ticks: float = 10.0
    take_profit_ticks: float = 20.0
    tick_size: float = 0.25
    tick_value: float = 12.5
    commission_per_contract: float = 2.4


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
    retrain_interval: int = 60 * 60  # seconds
    parameter_grid: dict = field(default_factory=lambda: {
        "rsi_period": [14, 21, 28],
        "macd_fast": [12, 16, 20],
        "macd_slow": [26, 30, 35],
        "sentiment_threshold": [0.4, 0.5, 0.6],
        "momentum_lookback": [20, 30, 40],
    })


@dataclass
class StrategyConfig:
    name: str = "rsi_macd_sentiment"
    enabled: bool = True
    params: dict = field(default_factory=dict)


@dataclass
class Settings:
    data: DataSourceConfig = field(default_factory=DataSourceConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    strategies: List[StrategyConfig] = field(default_factory=list)

    def validate(self) -> None:
        if self.trading.initial_capital <= 0:
            raise ValueError("initial capital must be positive")
        if self.trading.max_position_size <= 0:
            raise ValueError("max position size must be positive")
        if self.trading.tick_size <= 0:
            raise ValueError("tick size must be positive")
        if self.backtest.slippage < 0:
            raise ValueError("slippage cannot be negative")
