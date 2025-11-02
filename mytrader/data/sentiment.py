"""Sentiment data sources."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, AsyncIterator, Iterable

import pandas as pd
import tweepy
from textblob import TextBlob

from ..utils.logger import logger
from .base import DataCollector


class TwitterSentimentCollector(DataCollector):
    """Streams tweets and scores sentiment using TextBlob polarity with rate limiting."""

    def __init__(
        self, 
        bearer_token: str, 
        track_terms: Iterable[str], 
        lookback: int = 200,
        max_retries: int = 3,
        rate_limit_window: int = 900  # 15 minutes in seconds
    ) -> None:
        self.client = tweepy.Client(bearer_token=bearer_token)
        self.track_terms = list(track_terms)
        self.lookback = lookback
        self.max_retries = max_retries
        self.rate_limit_window = rate_limit_window
        self.request_count = 0
        self.window_start = datetime.utcnow()

    async def _check_rate_limit(self) -> None:
        """Implement rate limiting for Twitter API (450 requests per 15 min window)."""
        now = datetime.utcnow()
        elapsed = (now - self.window_start).total_seconds()
        
        if elapsed >= self.rate_limit_window:
            # Reset window
            self.window_start = now
            self.request_count = 0
        elif self.request_count >= 450:
            # Wait until window resets
            wait_time = self.rate_limit_window - elapsed
            logger.warning("Twitter rate limit reached. Waiting %.1f seconds...", wait_time)
            await asyncio.sleep(wait_time)
            self.window_start = datetime.utcnow()
            self.request_count = 0

    async def collect(self) -> pd.DataFrame:
        """Collect recent tweets with retry logic and rate limiting."""
        await self._check_rate_limit()
        
        query = " OR ".join(self.track_terms)
        
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.search_recent_tweets,
                    query=query,
                    max_results=min(self.lookback, 100),
                    tweet_fields=["created_at", "lang"],
                )
                self.request_count += 1
                
                tweets = response.data if response else None
                records: list[dict[str, Any]] = []
                
                if tweets:
                    for tweet in tweets:
                        if tweet.lang != "en":
                            continue
                        try:
                            score = TextBlob(tweet.text).sentiment.polarity
                            records.append({
                                "timestamp": tweet.created_at,
                                "text": tweet.text,
                                "sentiment": score,
                                "source": "twitter",
                            })
                        except Exception as e:
                            logger.debug("Error analyzing tweet sentiment: %s", e)
                            continue
                
                df = pd.DataFrame(records)
                if df.empty:
                    return pd.DataFrame(columns=["timestamp", "sentiment"]).set_index("timestamp")
                
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df.set_index("timestamp", inplace=True)
                return df[["sentiment"]]
                
            except tweepy.TooManyRequests as e:
                logger.warning("Twitter rate limit hit. Waiting...")
                await asyncio.sleep(self.rate_limit_window)
                self.window_start = datetime.utcnow()
                self.request_count = 0
            except tweepy.TweepyException as exc:
                if attempt < self.max_retries - 1:
                    delay = 2 ** attempt
                    logger.warning("Twitter API error (attempt %d/%d): %s. Retrying in %ds...", 
                                 attempt + 1, self.max_retries, exc, delay)
                    await asyncio.sleep(delay)
                else:
                    logger.error("Twitter collection failed after %d attempts: %s", self.max_retries, exc)
                    return pd.DataFrame(columns=["timestamp", "sentiment"]).set_index("timestamp")
        
        return pd.DataFrame(columns=["timestamp", "sentiment"]).set_index("timestamp")

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Stream sentiment data with error handling."""
        while True:
            try:
                df = await self.collect()
                if not df.empty:
                    yield {
                        "timestamp": df.index[-1].to_pydatetime(),
                        "sentiment": float(df.iloc[-1]["sentiment"]),
                        "source": "twitter",
                    }
            except Exception as e:
                logger.error("Twitter stream error: %s", e)
            await asyncio.sleep(60)
