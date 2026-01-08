"""Stocktwits sentiment service for ES/MES trading.

This module fetches sentiment data from Stocktwits for ES_F (E-mini S&P 500 futures)
and SPY (S&P 500 ETF) to provide sentiment signals for MES trading decisions.

Key features:
- Fetches from Stocktwits public API (no auth required for basic access)
- 5-minute caching to respect rate limits
- Aggregates bullish/bearish sentiment into a [-1.0, 1.0] score
- Provides combined MES sentiment from ES_F and SPY
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from ..utils.logger import logger

# Configuration constants (can be overridden via environment variables)
SENTIMENT_REFRESH_INTERVAL_SECONDS = int(
    os.environ.get("STOCKTWITS_REFRESH_INTERVAL_SECONDS", "300")
)  # 5 minutes default
STOCKTWITS_REQUEST_TIMEOUT = float(
    os.environ.get("STOCKTWITS_REQUEST_TIMEOUT", "3.0")
)  # 3 seconds timeout
STOCKTWITS_API_BASE_URL = "https://api.stocktwits.com/api/2/streams/symbol"

# Sentiment thresholds for trading decisions
SENTIMENT_ENTRY_BLOCK_THRESHOLD = float(
    os.environ.get("SENTIMENT_ENTRY_BLOCK_THRESHOLD", "0.4")
)
SENTIMENT_PROTECT_POSITION_THRESHOLD = float(
    os.environ.get("SENTIMENT_PROTECT_POSITION_THRESHOLD", "0.6")
)
SENTIMENT_WEAK_THRESHOLD = float(
    os.environ.get("SENTIMENT_WEAK_THRESHOLD", "0.2")
)


@dataclass
class SentimentCache:
    """In-memory cache for sentiment data."""
    
    last_fetch_timestamp: float = 0.0
    last_scores: Dict[str, float] = field(default_factory=dict)
    refresh_interval_seconds: float = SENTIMENT_REFRESH_INTERVAL_SECONDS
    
    def is_stale(self) -> bool:
        """Check if the cache is stale and needs refresh."""
        if not self.last_scores:
            return True
        elapsed = time.time() - self.last_fetch_timestamp
        return elapsed >= self.refresh_interval_seconds
    
    def update(self, scores: Dict[str, float]) -> None:
        """Update the cache with new sentiment scores."""
        self.last_fetch_timestamp = time.time()
        self.last_scores = scores.copy()
    
    def get_cached_scores(self) -> Dict[str, float]:
        """Return cached scores (may be empty if never fetched)."""
        return self.last_scores.copy()


@dataclass
class SentimentResult:
    """Result of sentiment analysis for a symbol."""
    
    symbol: str
    score: float  # -1.0 (bearish) to 1.0 (bullish)
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    total_messages: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "score": self.score,
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "total_messages": self.total_messages,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
        }


# Global cache instance
_sentiment_cache = SentimentCache()


def calculate_sentiment_score_from_messages(
    messages: List[Dict[str, Any]]
) -> SentimentResult:
    """Calculate sentiment score from a list of Stocktwits messages.
    
    This is a pure function that can be unit tested without API calls.
    
    Args:
        messages: List of message dictionaries from Stocktwits API
        
    Returns:
        SentimentResult with calculated score and counts
    """
    bullish_count = 0
    bearish_count = 0
    neutral_count = 0
    
    for msg in messages:
        # Stocktwits sentiment is in entities.sentiment.basic
        entities = msg.get("entities", {})
        sentiment = entities.get("sentiment")
        
        if sentiment:
            basic = sentiment.get("basic")
            if basic == "Bullish":
                bullish_count += 1
            elif basic == "Bearish":
                bearish_count += 1
            else:
                neutral_count += 1
        else:
            # No sentiment label - count as neutral
            neutral_count += 1
    
    total_with_sentiment = bullish_count + bearish_count
    total_messages = bullish_count + bearish_count + neutral_count
    
    if total_with_sentiment == 0:
        # No sentiment data - return neutral
        score = 0.0
    else:
        # Score ranges from -1.0 (all bearish) to +1.0 (all bullish)
        score = (bullish_count - bearish_count) / total_with_sentiment
    
    return SentimentResult(
        symbol="",  # Will be set by caller
        score=score,
        bullish_count=bullish_count,
        bearish_count=bearish_count,
        neutral_count=neutral_count,
        total_messages=total_messages,
    )


def _fetch_stocktwits_symbol(symbol: str) -> SentimentResult:
    """Fetch sentiment for a single symbol from Stocktwits API.
    
    Args:
        symbol: The stock symbol (e.g., "ES_F", "SPY")
        
    Returns:
        SentimentResult with the sentiment score
    """
    url = f"{STOCKTWITS_API_BASE_URL}/{symbol}.json"
    
    # Use browser-like User-Agent to avoid 403 blocks
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=STOCKTWITS_REQUEST_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if data.get("response", {}).get("status") != 200:
            error_msg = data.get("errors", [{}])[0].get("message", "Unknown API error")
            logger.warning(f"Stocktwits API error for {symbol}: {error_msg}")
            return SentimentResult(symbol=symbol, score=0.0, error=error_msg)
        
        # Extract messages
        messages = data.get("messages", [])
        
        if not messages:
            logger.debug(f"No messages found for {symbol}")
            return SentimentResult(symbol=symbol, score=0.0)
        
        # Calculate sentiment
        result = calculate_sentiment_score_from_messages(messages)
        result.symbol = symbol
        
        logger.info(
            f"📊 Stocktwits {symbol}: score={result.score:.2f} "
            f"(bull={result.bullish_count}, bear={result.bearish_count}, "
            f"total={result.total_messages})"
        )
        
        return result
        
    except requests.Timeout:
        logger.warning(f"Stocktwits timeout for {symbol} (>{STOCKTWITS_REQUEST_TIMEOUT}s)")
        return SentimentResult(symbol=symbol, score=0.0, error="timeout")
        
    except requests.RequestException as e:
        logger.warning(f"Stocktwits request error for {symbol}: {e}")
        return SentimentResult(symbol=symbol, score=0.0, error=str(e))
        
    except (ValueError, KeyError) as e:
        logger.warning(f"Stocktwits JSON parsing error for {symbol}: {e}")
        return SentimentResult(symbol=symbol, score=0.0, error=str(e))


def get_stocktwits_sentiment(
    symbols: List[str],
    force_refresh: bool = False,
) -> Dict[str, float]:
    """Fetch sentiment scores for multiple symbols from Stocktwits.
    
    Uses caching to avoid excessive API calls. Cache is refreshed every
    5 minutes (configurable via STOCKTWITS_REFRESH_INTERVAL_SECONDS).
    
    Args:
        symbols: List of symbols to fetch (e.g., ["ES_F", "SPY"])
        force_refresh: If True, bypass cache and fetch fresh data
        
    Returns:
        Dictionary mapping symbol to sentiment score in [-1.0, 1.0]
    """
    global _sentiment_cache
    
    # Check cache
    if not force_refresh and not _sentiment_cache.is_stale():
        cached = _sentiment_cache.get_cached_scores()
        # Return cached scores if we have all requested symbols
        if all(s in cached for s in symbols):
            cache_age = time.time() - _sentiment_cache.last_fetch_timestamp
            logger.info(f"📦 Using cached sentiment (age: {cache_age:.0f}s): {cached}")
            return {s: cached.get(s, 0.0) for s in symbols}
    
    logger.info(f"🔄 Fetching fresh Stocktwits sentiment for: {symbols}")
    
    # Fetch fresh data
    scores: Dict[str, float] = {}
    
    for symbol in symbols:
        result = _fetch_stocktwits_symbol(symbol)
        scores[symbol] = result.score
        
        # Small delay between requests to be respectful of rate limits
        if len(symbols) > 1:
            time.sleep(0.5)
    
    # Update cache
    _sentiment_cache.update(scores)
    
    return scores


def get_mes_sentiment(force_refresh: bool = False) -> float:
    """Get combined sentiment score for MES trading.
    
    Fetches sentiment from ES_F (E-mini S&P futures) and SPY (ETF proxy)
    and combines them into a single score.
    
    Args:
        force_refresh: If True, bypass cache and fetch fresh data
        
    Returns:
        Combined sentiment score in [-1.0, 1.0]
    """
    symbols = ["ES_F", "SPY"]
    scores = get_stocktwits_sentiment(symbols, force_refresh=force_refresh)
    
    # Get individual scores, defaulting to 0.0 if missing
    es_score = scores.get("ES_F", 0.0)
    spy_score = scores.get("SPY", 0.0)
    
    # Simple average of both sentiment scores
    combined_score = (es_score + spy_score) / 2.0
    
    logger.info(
        f"📈 MES Sentiment: {combined_score:.2f} "
        f"(ES_F={es_score:.2f}, SPY={spy_score:.2f})"
    )
    
    return combined_score


def reset_sentiment_cache() -> None:
    """Reset the sentiment cache. Useful for testing."""
    global _sentiment_cache
    _sentiment_cache = SentimentCache()


def set_cache_refresh_interval(seconds: float) -> None:
    """Set the cache refresh interval. Useful for testing."""
    global _sentiment_cache
    _sentiment_cache.refresh_interval_seconds = seconds


@dataclass
class SentimentDecisionModifier:
    """Result of sentiment-based decision modification."""
    
    allow_trade: bool = True
    confidence_modifier: float = 1.0  # Multiplier for confidence
    reason: str = ""
    mes_sentiment: float = 0.0
    action_recommendation: str = "PROCEED"  # PROCEED, REDUCE_SIZE, WAIT, BLOCK
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow_trade": self.allow_trade,
            "confidence_modifier": self.confidence_modifier,
            "reason": self.reason,
            "mes_sentiment": self.mes_sentiment,
            "action_recommendation": self.action_recommendation,
        }


def evaluate_sentiment_for_entry(
    proposed_action: str,
    entry_block_threshold: float = SENTIMENT_ENTRY_BLOCK_THRESHOLD,
    weak_threshold: float = SENTIMENT_WEAK_THRESHOLD,
    force_refresh: bool = False,
) -> SentimentDecisionModifier:
    """Evaluate sentiment before opening a new position.
    
    Args:
        proposed_action: "BUY" or "SELL"
        entry_block_threshold: Sentiment threshold to block entry (default 0.4)
        weak_threshold: Sentiment threshold for reduced confidence (default 0.2)
        force_refresh: If True, bypass cache for fresh data
        
    Returns:
        SentimentDecisionModifier with trading recommendation
    """
    mes_sentiment = get_mes_sentiment(force_refresh=force_refresh)
    
    result = SentimentDecisionModifier(mes_sentiment=mes_sentiment)
    
    if proposed_action == "BUY":
        # Long entry - bearish sentiment is contradictory
        if mes_sentiment < -entry_block_threshold:
            result.allow_trade = False
            result.confidence_modifier = 0.0
            result.reason = f"Sentiment strongly bearish ({mes_sentiment:.2f}) - blocking long entry"
            result.action_recommendation = "BLOCK"
            logger.warning(f"🚫 {result.reason}")
            
        elif mes_sentiment < -weak_threshold:
            result.confidence_modifier = 0.8  # 20% reduction
            result.reason = f"Sentiment moderately bearish ({mes_sentiment:.2f}) - reducing confidence"
            result.action_recommendation = "REDUCE_SIZE"
            logger.info(f"⚠️ {result.reason}")
            
        elif mes_sentiment > weak_threshold:
            result.confidence_modifier = 1.1  # 10% boost
            result.reason = f"Sentiment supports long ({mes_sentiment:.2f})"
            result.action_recommendation = "PROCEED"
            logger.info(f"✅ {result.reason}")
        else:
            result.reason = f"Sentiment neutral ({mes_sentiment:.2f})"
            result.action_recommendation = "PROCEED"
            
    elif proposed_action == "SELL":
        # Short entry - bullish sentiment is contradictory
        if mes_sentiment > entry_block_threshold:
            result.allow_trade = False
            result.confidence_modifier = 0.0
            result.reason = f"Sentiment strongly bullish ({mes_sentiment:.2f}) - blocking short entry"
            result.action_recommendation = "BLOCK"
            logger.warning(f"🚫 {result.reason}")
            
        elif mes_sentiment > weak_threshold:
            result.confidence_modifier = 0.8  # 20% reduction
            result.reason = f"Sentiment moderately bullish ({mes_sentiment:.2f}) - reducing confidence"
            result.action_recommendation = "REDUCE_SIZE"
            logger.info(f"⚠️ {result.reason}")
            
        elif mes_sentiment < -weak_threshold:
            result.confidence_modifier = 1.1  # 10% boost
            result.reason = f"Sentiment supports short ({mes_sentiment:.2f})"
            result.action_recommendation = "PROCEED"
            logger.info(f"✅ {result.reason}")
        else:
            result.reason = f"Sentiment neutral ({mes_sentiment:.2f})"
            result.action_recommendation = "PROCEED"
    else:
        # HOLD - no sentiment check needed
        result.reason = "No entry signal - sentiment check not applicable"
    
    return result


def evaluate_sentiment_for_position(
    position_direction: str,  # "LONG" or "SHORT"
    protect_threshold: float = SENTIMENT_PROTECT_POSITION_THRESHOLD,
    force_refresh: bool = False,
) -> SentimentDecisionModifier:
    """Evaluate sentiment for an existing position.
    
    Args:
        position_direction: "LONG" or "SHORT"
        protect_threshold: Sentiment threshold to trigger protection (default 0.6)
        force_refresh: If True, bypass cache for fresh data
        
    Returns:
        SentimentDecisionModifier with position management recommendation
    """
    mes_sentiment = get_mes_sentiment(force_refresh=force_refresh)
    
    result = SentimentDecisionModifier(mes_sentiment=mes_sentiment)
    
    if position_direction == "LONG":
        # Long position - bearish sentiment is concerning
        if mes_sentiment < -protect_threshold:
            result.confidence_modifier = 0.5
            result.reason = f"Sentiment extremely bearish ({mes_sentiment:.2f}) - consider tightening stop or exiting"
            result.action_recommendation = "REDUCE_SIZE"
            logger.warning(f"⚠️ LONG POSITION AT RISK: {result.reason}")
            
        elif mes_sentiment < -SENTIMENT_ENTRY_BLOCK_THRESHOLD:
            result.confidence_modifier = 0.8
            result.reason = f"Sentiment bearish ({mes_sentiment:.2f}) - tighten stop"
            result.action_recommendation = "REDUCE_SIZE"
            logger.info(f"⚠️ {result.reason}")
        else:
            result.reason = f"Sentiment OK for long ({mes_sentiment:.2f})"
            result.action_recommendation = "PROCEED"
            
    elif position_direction == "SHORT":
        # Short position - bullish sentiment is concerning
        if mes_sentiment > protect_threshold:
            result.confidence_modifier = 0.5
            result.reason = f"Sentiment extremely bullish ({mes_sentiment:.2f}) - consider tightening stop or exiting"
            result.action_recommendation = "REDUCE_SIZE"
            logger.warning(f"⚠️ SHORT POSITION AT RISK: {result.reason}")
            
        elif mes_sentiment > SENTIMENT_ENTRY_BLOCK_THRESHOLD:
            result.confidence_modifier = 0.8
            result.reason = f"Sentiment bullish ({mes_sentiment:.2f}) - tighten stop"
            result.action_recommendation = "REDUCE_SIZE"
            logger.info(f"⚠️ {result.reason}")
        else:
            result.reason = f"Sentiment OK for short ({mes_sentiment:.2f})"
            result.action_recommendation = "PROCEED"
    
    return result


def get_sentiment_summary() -> Dict[str, Any]:
    """Get a summary of current sentiment state.
    
    Returns:
        Dictionary with current sentiment data and cache status
    """
    global _sentiment_cache
    
    cache_age = time.time() - _sentiment_cache.last_fetch_timestamp
    
    return {
        "cache_age_seconds": cache_age,
        "cache_stale": _sentiment_cache.is_stale(),
        "last_scores": _sentiment_cache.get_cached_scores(),
        "refresh_interval_seconds": _sentiment_cache.refresh_interval_seconds,
        "entry_block_threshold": SENTIMENT_ENTRY_BLOCK_THRESHOLD,
        "protect_position_threshold": SENTIMENT_PROTECT_POSITION_THRESHOLD,
        "weak_threshold": SENTIMENT_WEAK_THRESHOLD,
    }
