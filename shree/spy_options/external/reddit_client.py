"""
Reddit sentiment from r/wallstreetbets and r/investing via asyncpraw.
Disabled by default — requires PRAW credentials (client_id, client_secret).
When disabled, returns neutral state silently.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional
from loguru import logger

try:
    import asyncpraw
    _PRAW_AVAILABLE = True
except ImportError:
    _PRAW_AVAILABLE = False

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()
_SUBREDDITS = ["wallstreetbets", "investing", "stocks"]
_SPY_TERMS = {"spy", "s&p", "market", "spx", "0dte", "options"}


@dataclass
class RedditSentimentState:
    score: float = 0.0      # -1.0 to +1.0
    post_count: int = 0
    fetched_at: float = 0.0
    available: bool = False

    def is_stale(self, ttl_s: float) -> bool:
        return (time.monotonic() - self.fetched_at) > ttl_s


class RedditClient:
    """
    Async Reddit sentiment client.

    To enable:
      1. Create a Reddit app at https://www.reddit.com/prefs/apps (script type)
      2. Set enabled=True and supply client_id, client_secret, user_agent
    """

    def __init__(
        self,
        enabled: bool = False,
        client_id: str = "",
        client_secret: str = "",
        user_agent: str = "ShreeBotSPY/1.0",
        ttl_minutes: float = 15.0,
    ):
        self._enabled = enabled and _PRAW_AVAILABLE and bool(client_id)
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent
        self._ttl_s = ttl_minutes * 60.0
        self._state = RedditSentimentState()

        if enabled and not _PRAW_AVAILABLE:
            logger.warning("[Reddit] asyncpraw not installed — Reddit disabled")
        elif enabled and not client_id:
            logger.warning("[Reddit] No client_id configured — Reddit disabled")

    async def refresh_if_stale(self) -> None:
        if not self._enabled:
            return
        if not self._state.is_stale(self._ttl_s):
            return
        await self._fetch()

    async def _fetch(self) -> None:
        try:
            reddit = asyncpraw.Reddit(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent=self._user_agent,
            )
            titles = []
            for sub_name in _SUBREDDITS:
                sub = await reddit.subreddit(sub_name)
                async for post in sub.hot(limit=25):
                    text = (post.title + " " + (post.selftext or "")).lower()
                    if any(t in text for t in _SPY_TERMS):
                        titles.append(post.title)
            await reddit.close()

            if not titles:
                self._state = RedditSentimentState(
                    fetched_at=time.monotonic(), available=True
                )
                return

            scores = [_vader.polarity_scores(t)["compound"] for t in titles]
            avg = sum(scores) / len(scores)
            self._state = RedditSentimentState(
                score=avg,
                post_count=len(titles),
                fetched_at=time.monotonic(),
                available=True,
            )
            logger.debug(f"[Reddit] {len(titles)} posts, score={avg:.3f}")
        except Exception as exc:
            logger.warning(f"[Reddit] Fetch failed: {exc}")
            self._state = RedditSentimentState(
                fetched_at=time.monotonic(), available=False
            )

    @property
    def state(self) -> RedditSentimentState:
        return self._state
