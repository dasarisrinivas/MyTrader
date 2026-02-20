"""Data source configuration — IBKR, TradingView, sentiment API keys."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os
from datetime import time

@dataclass
class DataSourceConfig:
    tradingview_webhook_url: Optional[str] = field(default_factory=lambda: os.environ.get("TRADINGVIEW_WEBHOOK_URL"))
    tradingview_symbol: str = field(default_factory=lambda: os.environ.get("TRADINGVIEW_SYMBOL", "MES"))
    tradingview_interval: str = field(default_factory=lambda: os.environ.get("TRADINGVIEW_INTERVAL", "1m"))

    ibkr_host: str = field(default_factory=lambda: os.environ.get("IBKR_HOST", "127.0.0.1"))
    ibkr_port: int = field(default_factory=lambda: int(os.environ.get("IBKR_PORT", "4001")))
    ibkr_client_id: int = field(default_factory=lambda: int(os.environ.get("IBKR_CLIENT_ID", "1")))
    ibkr_symbol: str = field(default_factory=lambda: os.environ.get("IBKR_SYMBOL", "MES"))
    ibkr_exchange: str = field(default_factory=lambda: os.environ.get("IBKR_EXCHANGE", "CME"))
    ibkr_currency: str = field(default_factory=lambda: os.environ.get("IBKR_CURRENCY", "USD"))

    twitter_bearer_token: Optional[str] = field(default_factory=lambda: os.environ.get("TWITTER_BEARER_TOKEN"))
    news_api_keys: List[str] = field(default_factory=lambda: os.environ.get("NEWS_API_KEYS", "").split(",") if os.environ.get("NEWS_API_KEYS") else [])
    sentiment_refresh_interval: int = 60  # seconds



