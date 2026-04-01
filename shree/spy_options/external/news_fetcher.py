"""
RSS news sentiment for SPY/market using VADER.
Pulls from Yahoo Finance, MarketWatch, CNBC, Reuters.
TTL-cached; refreshes every `ttl_minutes` minutes.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from loguru import logger

_FEEDS = [
    "https://finance.yahoo.com/news/rssindex",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.reuters.com/reuters/businessNews",
]

_SPY_KEYWORDS = {
    "spy", "s&p", "s&p 500", "market", "stocks", "equities", "wall street",
    "fed", "federal reserve", "inflation", "cpi", "gdp", "jobs", "unemployment",
    "rate", "yield", "treasury", "macro", "economy",
}

_vader = SentimentIntensityAnalyzer()


@dataclass
class NewsSentimentState:
    score: float = 0.0          # -1.0 (bearish) to +1.0 (bullish)
    article_count: int = 0
    headline_sample: List[str] = field(default_factory=list)
    fetched_at: float = 0.0     # time.monotonic()

    def is_stale(self, ttl_s: float) -> bool:
        return (time.monotonic() - self.fetched_at) > ttl_s


def _is_relevant(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in _SPY_KEYWORDS)


async def _fetch_feed(url: str) -> List[str]:
    loop = asyncio.get_event_loop()
    try:
        parsed = await loop.run_in_executor(None, feedparser.parse, url)
        titles = []
        for entry in parsed.entries[:20]:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            combined = f"{title} {summary}"
            if _is_relevant(combined):
                titles.append(title)
        return titles
    except Exception as exc:
        logger.debug(f"[NewsFetcher] Feed {url} error: {exc}")
        return []


class NewsFetcher:
    def __init__(self, ttl_minutes: float = 10.0):
        self._ttl_s = ttl_minutes * 60.0
        self._state = NewsSentimentState()
        self._lock = asyncio.Lock()

    async def refresh_if_stale(self) -> None:
        if not self._state.is_stale(self._ttl_s):
            return
        async with self._lock:
            if not self._state.is_stale(self._ttl_s):
                return
            await self._fetch()

    async def _fetch(self) -> None:
        try:
            results = await asyncio.gather(*[_fetch_feed(u) for u in _FEEDS])
            all_headlines: List[str] = []
            for batch in results:
                all_headlines.extend(batch)

            if not all_headlines:
                self._state = NewsSentimentState(fetched_at=time.monotonic())
                return

            scores = [_vader.polarity_scores(h)["compound"] for h in all_headlines]
            avg_score = sum(scores) / len(scores)

            self._state = NewsSentimentState(
                score=avg_score,
                article_count=len(all_headlines),
                headline_sample=all_headlines[:3],
                fetched_at=time.monotonic(),
            )
            logger.debug(
                f"[NewsFetcher] {len(all_headlines)} headlines, score={avg_score:.3f}"
            )
        except Exception as exc:
            logger.warning(f"[NewsFetcher] Fetch failed: {exc}")
            self._state = NewsSentimentState(fetched_at=time.monotonic())

    @property
    def state(self) -> NewsSentimentState:
        return self._state
