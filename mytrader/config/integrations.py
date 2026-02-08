"""External integration configs — sentiment, VIX feed, Telegram notifications."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os
from datetime import time

@dataclass
class StockwitsSentimentConfig:
    """Configuration for Stocktwits sentiment integration (legacy single-source)."""
    enabled: bool = field(
        default_factory=lambda: os.environ.get("STOCKTWITS_SENTIMENT_ENABLED", "True").lower() in {"1", "true", "yes"}
    )
    # Symbols to fetch sentiment for
    symbols: List[str] = field(default_factory=lambda: ["ES_F", "SPY"])
    # Cache refresh interval (minimum time between API calls)
    refresh_interval_seconds: int = field(
        default_factory=lambda: int(os.environ.get("STOCKTWITS_REFRESH_INTERVAL_SECONDS", "300"))
    )
    # HTTP request timeout
    request_timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("STOCKTWITS_REQUEST_TIMEOUT", "3.0"))
    )
    # ============================================================
    # RTH (Regular Trading Hours) thresholds - 8:30 AM - 3:00 PM CT
    # Higher liquidity = more relaxed thresholds
    # ============================================================
    # Sentiment threshold to block contradictory entries
    # e.g., block long if sentiment < -0.4, block short if sentiment > 0.4
    entry_block_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_ENTRY_BLOCK_THRESHOLD", "0.4"))
    )
    # Weaker threshold for reducing confidence (but not blocking)
    weak_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_WEAK_THRESHOLD", "0.2"))
    )
    # Threshold to trigger protective actions for existing positions
    protect_position_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_PROTECT_POSITION_THRESHOLD", "0.6"))
    )
    # ============================================================
    # Low Volume Session thresholds (Evening & Overnight)
    # Lower liquidity = STRICTER thresholds to avoid bad fills
    # Evening: 5:00 PM - 11:00 PM CT
    # Overnight: 11:00 PM - 3:00 AM CT
    # ============================================================
    low_volume_entry_block_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_LOW_VOLUME_ENTRY_BLOCK_THRESHOLD", "0.25"))
    )
    low_volume_weak_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_LOW_VOLUME_WEAK_THRESHOLD", "0.10"))
    )
    low_volume_protect_position_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_LOW_VOLUME_PROTECT_POSITION_THRESHOLD", "0.40"))
    )
    # ============================================================
    # Confidence modifiers
    # ============================================================
    # Confidence modifier when sentiment is mildly contradictory
    mild_contradiction_confidence_mult: float = 0.8  # 20% reduction
    # Confidence boost when sentiment agrees with signal
    agreement_confidence_mult: float = 1.1  # 10% boost
    # Additional penalty for low volume sessions (stacks with above)
    low_volume_confidence_penalty: float = 0.9  # Extra 10% reduction in low volume


@dataclass
class MultiSourceSentimentConfig:
    """Configuration for multi-source sentiment aggregation (Stocktwits + Reddit + Twitter).
    
    This replaces StockwitsSentimentConfig with a more comprehensive multi-source approach.
    """
    enabled: bool = field(
        default_factory=lambda: os.environ.get("MULTI_SOURCE_SENTIMENT_ENABLED", "True").lower() in {"1", "true", "yes"}
    )
    
    # ============================================================
    # Source weights (must sum to 1.0)
    # ============================================================
    stocktwits_weight: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_STOCKTWITS_WEIGHT", "0.4"))
    )
    reddit_weight: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_REDDIT_WEIGHT", "0.3"))
    )
    twitter_weight: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_TWITTER_WEIGHT", "0.3"))
    )
    
    # ============================================================
    # API Configuration
    # ============================================================
    refresh_interval_seconds: int = field(
        default_factory=lambda: int(os.environ.get("SENTIMENT_REFRESH_INTERVAL_SECONDS", "300"))
    )
    request_timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_REQUEST_TIMEOUT", "3.0"))
    )
    
    # Stocktwits symbols
    stocktwits_symbols: List[str] = field(default_factory=lambda: ["ES_F", "SPY"])
    
    # Reddit configuration
    reddit_subreddits: List[str] = field(
        default_factory=lambda: ["wallstreetbets", "stocks", "options"]
    )
    reddit_search_terms: List[str] = field(
        default_factory=lambda: ["SPY", "SPX", "ES", "S&P 500", "futures"]
    )
    
    # Twitter search terms
    twitter_search_terms: List[str] = field(
        default_factory=lambda: ["$SPY", "SPX", "ES futures", "S&P 500"]
    )
    
    # ============================================================
    # RTH Thresholds (Regular Trading Hours: 8:30 AM - 3:00 PM CT)
    # ============================================================
    entry_block_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_ENTRY_BLOCK_THRESHOLD", "0.4"))
    )
    weak_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_WEAK_THRESHOLD", "0.2"))
    )
    strong_exit_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_STRONG_EXIT_THRESHOLD", "0.6"))
    )
    
    # ============================================================
    # Low Volume Session Thresholds (Evening & Overnight)
    # ============================================================
    low_volume_entry_block_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_LOW_VOLUME_ENTRY_BLOCK_THRESHOLD", "0.25"))
    )
    low_volume_weak_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_LOW_VOLUME_WEAK_THRESHOLD", "0.10"))
    )
    
    # ============================================================
    # Confidence modifiers
    # ============================================================
    mild_contradiction_confidence_mult: float = 0.7  # 30% reduction for reduce_size
    agreement_confidence_mult: float = 1.1  # 10% boost
    low_volume_confidence_penalty: float = 0.9  # Extra 10% penalty


@dataclass
class VixFeedThresholds:
    """Thresholds for VX-based volatility multiplier."""
    extreme: float = 30.0  # VX >= 30: 0.4x multiplier
    elevated: float = 20.0  # VX >= 20: 0.7x multiplier


@dataclass
class VixFeedConfig:
    """Configuration for VX Futures Feed from IBKR.
    
    Provides real-time VIX futures data for volatility-based position sizing.
    When VIX is elevated, position sizes are automatically reduced.
    """
    enabled: bool = field(
        default_factory=lambda: os.environ.get("VIX_FEED_ENABLED", "True").lower() in {"1", "true", "yes"}
    )
    
    # IBKR Connection settings
    ib_host: str = field(
        default_factory=lambda: os.environ.get("VIX_FEED_IB_HOST", "127.0.0.1")
    )
    ib_port: int = field(
        default_factory=lambda: int(os.environ.get("VIX_FEED_IB_PORT", "7497"))
    )
    client_id: int = field(
        default_factory=lambda: int(os.environ.get("VIX_FEED_CLIENT_ID", "71"))
    )
    market_data_type: int = field(
        default_factory=lambda: int(os.environ.get("VIX_FEED_MARKET_DATA_TYPE", "1"))  # 1=live, 3=delayed
    )
    
    # Stale data handling
    stale_seconds: int = field(
        default_factory=lambda: int(os.environ.get("VIX_FEED_STALE_SECONDS", "120"))
    )
    conservative_on_stale: bool = field(
        default_factory=lambda: os.environ.get("VIX_FEED_CONSERVATIVE_ON_STALE", "False").lower() in {"1", "true", "yes"}
    )
    
    # Volatility multiplier thresholds
    thresholds: VixFeedThresholds = field(default_factory=VixFeedThresholds)
    
    # Reconnection settings
    max_retries: int = field(
        default_factory=lambda: int(os.environ.get("VIX_FEED_MAX_RETRIES", "5"))
    )
    base_delay: float = field(
        default_factory=lambda: float(os.environ.get("VIX_FEED_BASE_DELAY", "1.0"))
    )
    max_delay: float = field(
        default_factory=lambda: float(os.environ.get("VIX_FEED_MAX_DELAY", "60.0"))
    )


@dataclass
class TelegramConfig:
    """Configuration for Telegram notifications."""
    enabled: bool = field(default_factory=lambda: os.environ.get("TELEGRAM_ENABLED", "False").lower() == "true")
    bot_token: str = field(default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    chat_id: str = field(default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID", ""))
    notify_on_trade: bool = True
    notify_on_signal: bool = False
    notify_on_error: bool = True



