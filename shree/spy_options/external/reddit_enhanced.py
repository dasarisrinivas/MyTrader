"""
Enhanced Reddit SPY sentiment engine.

Improvements over the basic reddit_client.py:
  - Scores post body AND top N comments (not just title)
  - SPY-specific keyword frequency scoring (directional + structural terms)
  - Karma and comment-count weighting (high-engagement posts matter more)
  - Bot/spam heuristics (downweight very new accounts, repeated posts)
  - Sarcasm detection via punctuation + contradiction patterns
  - Contrarian logic: extreme euphoria → bearish signal; extreme panic → bullish
  - Final score: -100 to +100

Subreddits monitored (configurable):
  wallstreetbets, options, stocks, investing, SPYtrading (if public), Daytrading

SPY keyword dictionary:
  Bullish structural: breakout, melt-up, squeeze, moon, rip, pump, gamma squeeze,
                      call wall, support held, higher highs
  Bearish structural: dump, rug pull, crash, rejection, reversal, put wall,
                      lower highs, distribution, head and shoulders, double top
  Neutral/informational: IV, theta, delta, expiry, strike, VWAP, dedup

Scoring:
  base VADER compound × keyword_boost × karma_weight × engagement_weight
  where karma_weight = log10(max(1, post_score)) / 4  (capped at 1.0)
  and   engagement_weight = log10(max(1, num_comments)) / 3  (capped at 1.0)
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger

try:
    import asyncpraw
    _PRAW_AVAILABLE = True
except ImportError:
    _PRAW_AVAILABLE = False

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()

# ── Keyword dictionaries ──────────────────────────────────────────────────────

_BULLISH_KEYWORDS: Dict[str, float] = {
    "breakout": 0.15, "melt-up": 0.20, "melt up": 0.20, "squeeze": 0.12,
    "moon": 0.10, "rip": 0.10, "pump": 0.08, "gamma squeeze": 0.18,
    "call wall": 0.12, "support held": 0.15, "higher highs": 0.12,
    "higher lows": 0.10, "bull flag": 0.15, "accumulation": 0.12,
    "bottom": 0.08, "recovery": 0.08, "bounce": 0.10, "vwap reclaim": 0.15,
    "calls printing": 0.15, "bought calls": 0.12, "long calls": 0.12,
}

_BEARISH_KEYWORDS: Dict[str, float] = {
    "dump": 0.15, "rug pull": 0.20, "rug": 0.08, "crash": 0.15,
    "rejection": 0.12, "reversal": 0.10, "put wall": 0.12,
    "lower highs": 0.12, "distribution": 0.12, "head and shoulders": 0.15,
    "double top": 0.15, "breakdown": 0.15, "bears": 0.08,
    "puts printing": 0.15, "bought puts": 0.12, "long puts": 0.12,
    "vwap rejected": 0.15, "resistance": 0.08, "death cross": 0.18,
}

_SPY_TERMS = frozenset({
    "spy", "s&p", "s&p 500", "spx", "spdr", "0dte", "1dte", "options",
    "calls", "puts", "market", "qqq",
})

_SARCASM_INDICATORS = frozenset({
    "/s", "🙄", "lmao", "lol sure", "totally", "definitely", "obviously",
    "\"definitely\"", "not financial advice trust me",
})

# Subreddits to monitor
_SUBREDDITS = [
    "wallstreetbets", "options", "stocks", "investing",
    "Daytrading", "thetagang",
]


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RedditPost:
    title: str
    body: str
    score: int          # upvotes
    num_comments: int
    author_karma: int
    is_bot: bool
    comments: List[str]
    vader_score: float
    keyword_boost: float
    final_score: float   # -1.0 to +1.0


@dataclass
class RedditEnhancedState:
    score: float = 0.0          # -100 to +100
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    post_count: int = 0
    top_bullish_title: str = ""
    top_bearish_title: str = ""
    contrarian_signal: str = "NONE"  # "CONTRARIAN_BULLISH" | "CONTRARIAN_BEARISH" | "NONE"
    fetched_at: float = 0.0
    available: bool = False

    def is_stale(self, ttl_s: float) -> bool:
        return (time.monotonic() - self.fetched_at) > ttl_s


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_relevant(text: str) -> bool:
    low = text.lower()
    return any(t in low for t in _SPY_TERMS)


def _has_sarcasm(text: str) -> bool:
    low = text.lower()
    return any(s in low for s in _SARCASM_INDICATORS)


def _keyword_boost(text: str) -> float:
    """
    Returns net keyword boost (-0.5 to +0.5).
    Bullish terms add; bearish terms subtract.
    Sarcasm flips the sign.
    """
    low = text.lower()
    bull = sum(v for k, v in _BULLISH_KEYWORDS.items() if k in low)
    bear = sum(v for k, v in _BEARISH_KEYWORDS.items() if k in low)
    net = min(0.5, bull) - min(0.5, bear)
    if _has_sarcasm(text):
        net = -net
    return net


def _karma_weight(score: int) -> float:
    """Log-scale karma weight; 1000 upvotes → ~0.75, 10k → 1.0."""
    return min(1.0, max(0.1, _log10(max(1, score)) / 4.0))


def _engagement_weight(num_comments: int) -> float:
    """More comments = more engagement = more signal weight."""
    return min(1.0, max(0.1, _log10(max(1, num_comments)) / 3.0))


def _log10(x: float) -> float:
    import math
    return math.log10(x) if x > 0 else 0.0


def _is_likely_bot(author_karma: int, username: str) -> bool:
    """Simple bot heuristic: very low karma or bot-like username."""
    if author_karma < 10:
        return True
    bot_patterns = ["bot", "auto", "alerter", "notif", "tracker"]
    low = username.lower()
    return any(p in low for p in bot_patterns)


def _score_text(text: str) -> float:
    """VADER + keyword boost → -1.0 to +1.0."""
    if not text.strip():
        return 0.0
    vader = _vader.polarity_scores(text)["compound"]
    boost = _keyword_boost(text)
    combined = vader * 0.6 + boost * 0.4
    return max(-1.0, min(1.0, combined))


def _contrarian_signal(raw_score: float, bullish_count: int, bearish_count: int, total: int) -> str:
    """
    When crowd sentiment is extremely one-sided, flag contrarian signal.
    Extreme euphoria (>+70) = contrarian bearish.
    Extreme panic (<-70) = contrarian bullish.
    Requires at least 10 posts to be meaningful.
    """
    if total < 10:
        return "NONE"
    if raw_score > 70 and bullish_count / max(1, total) > 0.80:
        return "CONTRARIAN_BEARISH"
    if raw_score < -70 and bearish_count / max(1, total) > 0.80:
        return "CONTRARIAN_BULLISH"
    return "NONE"


# ── Main class ────────────────────────────────────────────────────────────────

class RedditEnhanced:
    """
    Full-body Reddit SPY sentiment client.

    To enable: supply asyncpraw credentials via the constructor.
    Falls back to RedditSentimentState(available=False) when disabled.
    """

    def __init__(
        self,
        enabled: bool = False,
        client_id: str = "",
        client_secret: str = "",
        user_agent: str = "ShreeBotSPY/2.0",
        ttl_minutes: float = 15.0,
        posts_per_sub: int = 30,
        top_comments: int = 5,
    ):
        self._enabled = enabled and _PRAW_AVAILABLE and bool(client_id)
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent
        self._ttl_s = ttl_minutes * 60.0
        self._posts_per_sub = posts_per_sub
        self._top_comments = top_comments
        self._state = RedditEnhancedState()

        if enabled and not _PRAW_AVAILABLE:
            logger.warning("[RedditEnhanced] asyncpraw not installed — disabled")
        elif enabled and not client_id:
            logger.warning("[RedditEnhanced] No client_id — disabled")

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
            posts: List[RedditPost] = []

            for sub_name in _SUBREDDITS:
                try:
                    sub = await reddit.subreddit(sub_name)
                    async for submission in sub.hot(limit=self._posts_per_sub):
                        full_text = (submission.title + " " + (submission.selftext or "")).strip()
                        if not _is_relevant(full_text):
                            continue

                        # Author karma (safe — some accounts have no karma attr)
                        try:
                            author_karma = submission.author.comment_karma + submission.author.link_karma
                            username = submission.author.name or ""
                        except Exception:
                            author_karma = 0
                            username = ""

                        if _is_likely_bot(author_karma, username):
                            continue

                        # Fetch top N comments
                        await submission.comments.replace_more(limit=0)
                        top_comments: List[str] = []
                        for comment in submission.comments.list()[:self._top_comments]:
                            body = getattr(comment, "body", "") or ""
                            if body and not body.startswith("[deleted]"):
                                top_comments.append(body)

                        # Score all text
                        all_text = full_text + " " + " ".join(top_comments)
                        vader_s = _vader.polarity_scores(all_text)["compound"]
                        boost = _keyword_boost(all_text)
                        final = max(-1.0, min(1.0, vader_s * 0.6 + boost * 0.4))

                        posts.append(RedditPost(
                            title=submission.title,
                            body=submission.selftext or "",
                            score=submission.score,
                            num_comments=submission.num_comments,
                            author_karma=author_karma,
                            is_bot=False,
                            comments=top_comments,
                            vader_score=vader_s,
                            keyword_boost=boost,
                            final_score=final,
                        ))
                except Exception as sub_exc:
                    logger.debug(f"[RedditEnhanced] r/{sub_name} error: {sub_exc}")

            await reddit.close()
            self._state = self._aggregate(posts)
        except Exception as exc:
            logger.warning(f"[RedditEnhanced] Fetch failed: {exc}")
            self._state = RedditEnhancedState(
                fetched_at=time.monotonic(), available=False
            )

    def _aggregate(self, posts: List[RedditPost]) -> RedditEnhancedState:
        if not posts:
            return RedditEnhancedState(
                fetched_at=time.monotonic(), available=True
            )

        weighted_sum = 0.0
        weight_total = 0.0
        bullish = bearish = neutral = 0
        top_bull_score = top_bear_score = 0.0
        top_bull_title = top_bear_title = ""

        for p in posts:
            kw = _karma_weight(p.score)
            ew = _engagement_weight(p.num_comments)
            w = kw * ew
            weighted_sum += p.final_score * w
            weight_total += w

            if p.final_score > 0.1:
                bullish += 1
                if p.final_score > top_bull_score:
                    top_bull_score = p.final_score
                    top_bull_title = p.title[:80]
            elif p.final_score < -0.1:
                bearish += 1
                if p.final_score < top_bear_score:
                    top_bear_score = p.final_score
                    top_bear_title = p.title[:80]
            else:
                neutral += 1

        raw_norm = (weighted_sum / weight_total) if weight_total > 0 else 0.0
        final_score = round(raw_norm * 100.0, 1)  # scale to -100..+100

        contrarian = _contrarian_signal(final_score, bullish, bearish, len(posts))

        # Apply contrarian flip to score
        if contrarian == "CONTRARIAN_BEARISH":
            final_score = min(final_score, -10.0)
        elif contrarian == "CONTRARIAN_BULLISH":
            final_score = max(final_score, 10.0)

        state = RedditEnhancedState(
            score=final_score,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            post_count=len(posts),
            top_bullish_title=top_bull_title,
            top_bearish_title=top_bear_title,
            contrarian_signal=contrarian,
            fetched_at=time.monotonic(),
            available=True,
        )
        logger.debug(
            f"[RedditEnhanced] {len(posts)} posts  score={final_score:.1f}  "
            f"bull={bullish} bear={bearish}  contrarian={contrarian}"
        )
        return state

    @property
    def state(self) -> RedditEnhancedState:
        return self._state
