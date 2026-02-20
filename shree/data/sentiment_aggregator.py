"""
Multi-Source Sentiment Aggregator for MES Trading.

Aggregates sentiment from multiple sources:
- Stocktwits (ES_F, SPY)
- Reddit (r/wallstreetbets, r/stocks)
- Twitter/X ($SPY, SPX, ES mentions)

Combined sentiment is used to:
1. Filter entry decisions (block contradictory trades)
2. Manage existing positions (tighten stops on sentiment reversal)

Configuration:
- SENTIMENT_REFRESH_INTERVAL_SECONDS: Minimum time between API refreshes (default: 300)
- SENTIMENT_STOCKTWITS_WEIGHT: Weight for Stocktwits score (default: 0.4)
- SENTIMENT_REDDIT_WEIGHT: Weight for Reddit score (default: 0.3)
- SENTIMENT_TWITTER_WEIGHT: Weight for Twitter score (default: 0.3)
- SENTIMENT_ENTRY_BLOCK_THRESHOLD: Block entry if sentiment contradicts (default: 0.4)
- SENTIMENT_STRONG_EXIT_THRESHOLD: Tighten stops on reversal (default: 0.6)

API Credentials (optional, set via environment variables):
- REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
- TWITTER_BEARER_TOKEN or TWITTER_API_KEY/SECRET

Author: Shree Bot
Created: January 2026
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import requests

from ..utils.logger import logger

# ============================================================
# CONFIGURATION CONSTANTS
# ============================================================

# Cache refresh interval (5 minutes)
SENTIMENT_REFRESH_INTERVAL_SECONDS = int(
    os.environ.get("SENTIMENT_REFRESH_INTERVAL_SECONDS", "300")
)

# API request timeout (don't block trading loop)
SENTIMENT_REQUEST_TIMEOUT = float(
    os.environ.get("SENTIMENT_REQUEST_TIMEOUT", "3.0")
)

# Source weights for combined score (must sum to 1.0)
# NOTE: Twitter disabled by default - weights adjusted to Stocktwits 55%, Reddit 45%
STOCKTWITS_WEIGHT = float(os.environ.get("SENTIMENT_STOCKTWITS_WEIGHT", "0.55"))
REDDIT_WEIGHT = float(os.environ.get("SENTIMENT_REDDIT_WEIGHT", "0.45"))
TWITTER_WEIGHT = float(os.environ.get("SENTIMENT_TWITTER_WEIGHT", "0.0"))

# Twitter enable flag - set to "true" to enable Twitter sentiment
TWITTER_ENABLED = os.environ.get("SENTIMENT_TWITTER_ENABLED", "false").lower() == "true"

# Entry/exit thresholds
ENTRY_BLOCK_THRESHOLD = float(os.environ.get("SENTIMENT_ENTRY_BLOCK_THRESHOLD", "0.4"))
STRONG_EXIT_THRESHOLD = float(os.environ.get("SENTIMENT_STRONG_EXIT_THRESHOLD", "0.6"))

# CONFIDENCE MODIFIER CAPS (Jan 2026 - Research Review)
# These caps prevent sentiment from over-leveraging positions or creating
# excessive risk. The max boost is intentionally conservative (10%) to avoid
# over-weighting sentiment as a primary signal.
CONFIDENCE_MODIFIER_MAX_BOOST = 1.1   # Max boost for supporting sentiment
CONFIDENCE_MODIFIER_MIN_REDUCTION = 0.5  # Min reduction (50% of original)
# Note: Blocked trades have modifier 0.0 (no trade allowed)

# API endpoints
STOCKTWITS_API_BASE = "https://api.stocktwits.com/api/2/streams/symbol"

# Reddit API (OAuth2)
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "ShreeBot/1.0")

# Twitter/X API
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN", "")

# Symbols to track
MES_PROXY_SYMBOLS = ["ES_F", "SPY"]
REDDIT_SEARCH_TERMS = ["SPY", "SPX", "ES", "S&P 500", "S&P500", "futures"]
TWITTER_SEARCH_TERMS = ["$SPY", "SPX", "ES futures", "S&P 500"]

# Subreddits to monitor
REDDIT_SUBREDDITS = ["wallstreetbets", "stocks", "options"]

# HTTP headers
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# ============================================================
# DATA STRUCTURES
# ============================================================

class SentimentSource(Enum):
    """Sentiment data sources."""
    STOCKTWITS = "stocktwits"
    REDDIT = "reddit"
    TWITTER = "twitter"
    VIX = "vix"  # Market-based fear gauge
    PUT_CALL = "put_call"  # Options sentiment
    COMBINED = "combined"


@dataclass
class SourceSentiment:
    """Sentiment result from a single source."""
    source: SentimentSource
    score: float  # -1.0 (bearish) to 1.0 (bullish)
    confidence: float = 1.0  # 0.0-1.0, how reliable is this score
    sample_count: int = 0  # Number of messages/posts analyzed
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def is_valid(self) -> bool:
        """Check if this result is usable."""
        return self.error is None and self.sample_count > 0


@dataclass
class CombinedSentiment:
    """Combined sentiment from all sources."""
    score: float  # -1.0 to 1.0
    stocktwits: SourceSentiment
    reddit: SourceSentiment
    twitter: SourceSentiment
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "score": round(self.score, 3),
            "stocktwits": {
                "score": round(self.stocktwits.score, 3),
                "samples": self.stocktwits.sample_count,
                "error": self.stocktwits.error,
            },
            "reddit": {
                "score": round(self.reddit.score, 3),
                "samples": self.reddit.sample_count,
                "error": self.reddit.error,
            },
            "twitter": {
                "score": round(self.twitter.score, 3),
                "samples": self.twitter.sample_count,
                "error": self.twitter.error,
            },
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SentimentDecision:
    """Decision based on sentiment analysis."""
    allow_trade: bool
    action_recommendation: str  # "PROCEED", "BLOCK", "REDUCE_SIZE", "TIGHTEN_STOP"
    confidence_modifier: float  # 0.0-1.5 multiplier for signal confidence
    combined_score: float
    reason: str
    source_breakdown: Dict[str, float] = field(default_factory=dict)


# ============================================================
# GLOBAL CACHE
# ============================================================

@dataclass
class SentimentCache:
    """Global cache for sentiment data."""
    last_fetch_time: Optional[datetime] = None
    last_result: Optional[CombinedSentiment] = None
    refresh_interval: timedelta = field(
        default_factory=lambda: timedelta(seconds=SENTIMENT_REFRESH_INTERVAL_SECONDS)
    )
    
    def is_stale(self) -> bool:
        """Check if cache needs refresh."""
        if self.last_fetch_time is None or self.last_result is None:
            return True
        elapsed = datetime.now(timezone.utc) - self.last_fetch_time
        return elapsed >= self.refresh_interval
    
    def update(self, result: CombinedSentiment) -> None:
        """Update cache with new result."""
        self.last_fetch_time = datetime.now(timezone.utc)
        self.last_result = result
    
    def get_cached(self) -> Optional[CombinedSentiment]:
        """Get cached result if available."""
        return self.last_result
    
    def get_age_seconds(self) -> float:
        """Get age of cached data in seconds."""
        if self.last_fetch_time is None:
            return float("inf")
        elapsed = datetime.now(timezone.utc) - self.last_fetch_time
        return elapsed.total_seconds()


# Global cache instance
_sentiment_cache = SentimentCache()


def set_cache_refresh_interval(seconds: int) -> None:
    """Configure the cache refresh interval."""
    global _sentiment_cache
    _sentiment_cache.refresh_interval = timedelta(seconds=seconds)
    logger.debug(f"Sentiment cache refresh interval set to {seconds}s")


# ============================================================
# NLP SENTIMENT ANALYSIS
# ============================================================

# Try to import VADER for sentiment analysis
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _vader_analyzer: Optional[SentimentIntensityAnalyzer] = None
    
    def _get_vader() -> Optional[SentimentIntensityAnalyzer]:
        """Lazy-load VADER analyzer."""
        global _vader_analyzer
        if _vader_analyzer is None:
            try:
                import nltk
                nltk.download('vader_lexicon', quiet=True)
                _vader_analyzer = SentimentIntensityAnalyzer()
            except Exception as e:
                logger.warning(f"Failed to initialize VADER: {e}")
                return None
        return _vader_analyzer
    
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    _vader_analyzer = None
    
    def _get_vader():
        return None

# Try TextBlob as fallback
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


def analyze_text_sentiment(text: str) -> float:
    """Analyze sentiment of text using available NLP tools.
    
    Returns:
        Sentiment score in [-1.0, 1.0]
    """
    if not text or not text.strip():
        return 0.0
    
    # Clean text
    text = _clean_text(text)
    
    # Try VADER first (best for social media)
    vader = _get_vader()
    if vader is not None:
        try:
            scores = vader.polarity_scores(text)
            # compound is already in [-1, 1]
            return scores['compound']
        except Exception:
            pass
    
    # Try TextBlob as fallback
    if TEXTBLOB_AVAILABLE:
        try:
            blob = TextBlob(text)
            # polarity is in [-1, 1]
            return blob.sentiment.polarity
        except Exception:
            pass
    
    # Last resort: keyword-based scoring
    return _keyword_sentiment(text)


def _clean_text(text: str) -> str:
    """Clean text for sentiment analysis."""
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove mentions (@user)
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (keep the word)
    text = re.sub(r'#(\w+)', r'\1', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()


def _keyword_sentiment(text: str) -> float:
    """Simple keyword-based sentiment scoring as fallback.
    
    FEB 20 2026: Added geopolitical, oil/energy, and macro-event keywords
    that directly impact MES via risk-off / risk-on flows.
    
    Returns:
        Score in [-1.0, 1.0]
    """
    text_lower = text.lower()
    
    # Bullish keywords
    bullish_words = [
        'bull', 'bullish', 'buy', 'long', 'calls', 'moon', 'rocket',
        'pump', 'green', 'rally', 'breakout', 'higher', 'up', 'gain',
        'profit', 'win', 'winner', 'strong', 'support', 'bounce',
        # Macro-bullish
        'rate cut', 'dovish', 'stimulus', 'soft landing', 'goldilocks',
        'risk on', 'risk-on', 'ceasefire', 'peace deal', 'de-escalation',
    ]
    
    # Bearish keywords
    bearish_words = [
        'bear', 'bearish', 'sell', 'short', 'puts', 'crash', 'dump',
        'red', 'drop', 'breakdown', 'lower', 'down', 'loss', 'lose',
        'loser', 'weak', 'resistance', 'fade', 'tank', 'plunge',
        # Geopolitical / risk-off
        'sanctions', 'tariff', 'tariffs', 'war', 'military', 'missile',
        'iran', 'invasion', 'geopolitical', 'escalation', 'retaliation',
        'nuclear', 'conflict', 'embargo', 'blockade',
        # Oil / energy shock (bearish for equities via cost-push inflation)
        'oil spike', 'crude spike', 'oil surge', 'energy crisis',
        'opec cut', 'supply disruption',
        # Macro-bearish
        'rate hike', 'hawkish', 'stagflation', 'recession', 'hot inflation',
        'sticky inflation', 'risk off', 'risk-off', 'flight to safety',
    ]
    
    bullish_count = sum(1 for word in bullish_words if word in text_lower)
    bearish_count = sum(1 for word in bearish_words if word in text_lower)
    
    total = bullish_count + bearish_count
    if total == 0:
        return 0.0
    
    return (bullish_count - bearish_count) / total


def compute_sentiment_from_messages(messages: List[str]) -> float:
    """Compute aggregate sentiment from a list of messages.
    
    This is a pure function for easy testing.
    
    Args:
        messages: List of text messages to analyze
        
    Returns:
        Aggregate sentiment score in [-1.0, 1.0]
    """
    if not messages:
        return 0.0
    
    scores = [analyze_text_sentiment(msg) for msg in messages]
    scores = [s for s in scores if s != 0.0]  # Filter neutral
    
    if not scores:
        return 0.0
    
    return sum(scores) / len(scores)


# ============================================================
# STOCKTWITS SENTIMENT
# ============================================================

def get_stocktwits_sentiment(symbols: List[str] = None) -> SourceSentiment:
    """Fetch sentiment from Stocktwits for given symbols.
    
    WEIGHTING METHOD (Jan 2026 - Research Review Documentation):
    This function uses VOLUME-WEIGHTED (pooled) sentiment scoring,
    meaning messages from ES_F and SPY are pooled together and scored
    as a single aggregate. This gives more weight to the symbol with
    more discussion activity (usually SPY during market hours).
    
    Alternative: Equal weighting (50/50 average) would treat both symbols
    equally regardless of message volume. The pooled method is preferred
    because it better reflects overall crowd sentiment.
    
    Args:
        symbols: List of symbols (default: ES_F, SPY)
        
    Returns:
        SourceSentiment with aggregated score
    """
    symbols = symbols or MES_PROXY_SYMBOLS
    
    total_bullish = 0
    total_bearish = 0
    total_messages = 0
    errors = []
    
    headers = {
        "User-Agent": BROWSER_USER_AGENT,
        "Accept": "application/json",
    }
    
    for symbol in symbols:
        try:
            url = f"{STOCKTWITS_API_BASE}/{symbol}.json"
            response = requests.get(
                url, 
                headers=headers, 
                timeout=SENTIMENT_REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            data = response.json()
            messages = data.get("messages", [])
            
            for msg in messages:
                # FIX: Handle None sentiment - entities.sentiment can be None, not just missing
                entities = msg.get("entities") or {}
                sentiment = entities.get("sentiment") or {}
                basic = sentiment.get("basic") if sentiment else None
                
                if basic == "Bullish":
                    total_bullish += 1
                elif basic == "Bearish":
                    total_bearish += 1
                
                total_messages += 1
            
            # Small delay between symbols
            if len(symbols) > 1:
                time.sleep(0.3)
                
        except requests.Timeout:
            errors.append(f"{symbol}: timeout")
            logger.warning(f"Stocktwits timeout for {symbol}")
        except requests.RequestException as e:
            errors.append(f"{symbol}: {str(e)[:50]}")
            logger.warning(f"Stocktwits error for {symbol}: {e}")
        except Exception as e:
            errors.append(f"{symbol}: {str(e)[:50]}")
            logger.warning(f"Stocktwits parsing error for {symbol}: {e}")
    
    # Calculate score
    total_with_sentiment = total_bullish + total_bearish
    if total_with_sentiment == 0:
        score = 0.0
    else:
        score = (total_bullish - total_bearish) / total_with_sentiment
    
    # Calculate confidence based on sample size
    confidence = min(1.0, total_messages / 50)  # Full confidence at 50+ messages
    
    error_msg = "; ".join(errors) if errors else None
    
    logger.info(
        f"📊 Stocktwits: score={score:.2f} "
        f"(bull={total_bullish}, bear={total_bearish}, total={total_messages}) "
        f"[ts={datetime.now(timezone.utc).strftime('%H:%M:%S')}]"
    )
    
    return SourceSentiment(
        source=SentimentSource.STOCKTWITS,
        score=score,
        confidence=confidence,
        sample_count=total_messages,
        error=error_msg,
    )


# ============================================================
# REDDIT SENTIMENT
# ============================================================

def get_reddit_sentiment() -> SourceSentiment:
    """Fetch sentiment from Reddit (r/wallstreetbets, r/stocks).
    
    Uses Reddit's public JSON API (no credentials needed).
    OAuth2 method kept as optional fallback for higher rate limits.
    
    Returns:
        SourceSentiment with aggregated score
    """
    # Always use public API first (simpler, no credentials needed)
    # If you have OAuth credentials and need higher rate limits,
    # set REDDIT_USE_OAUTH=true in environment
    use_oauth = os.environ.get("REDDIT_USE_OAUTH", "false").lower() == "true"
    
    if use_oauth and REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
        return _get_reddit_sentiment_oauth()
    else:
        return _get_reddit_sentiment_public()


def _get_reddit_sentiment_oauth() -> SourceSentiment:
    """Fetch Reddit sentiment using OAuth2 API.
    
    NOTE (Jan 2026 - Research Review):
    - Deduplication added to prevent double-counting posts, as identified
      in research review.
    """
    try:
        # Get OAuth token
        auth = requests.auth.HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
        headers = {"User-Agent": REDDIT_USER_AGENT}
        data = {"grant_type": "client_credentials"}
        
        token_response = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=auth,
            headers=headers,
            data=data,
            timeout=SENTIMENT_REQUEST_TIMEOUT,
        )
        token_response.raise_for_status()
        token = token_response.json().get("access_token")
        
        if not token:
            logger.warning("Reddit OAuth: No access token received")
            return _get_reddit_sentiment_public()
        
        # Search for posts
        headers["Authorization"] = f"Bearer {token}"
        
        all_texts = []
        seen_post_ids: set = set()  # RESEARCH FIX: Deduplicate posts
        search_query = " OR ".join(REDDIT_SEARCH_TERMS)
        
        for subreddit in REDDIT_SUBREDDITS:
            try:
                search_url = f"https://oauth.reddit.com/r/{subreddit}/search"
                params = {
                    "q": search_query,
                    "sort": "new",
                    "limit": 25,
                    "t": "day",
                    "restrict_sr": True,
                }
                
                response = requests.get(
                    search_url,
                    headers=headers,
                    params=params,
                    timeout=SENTIMENT_REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                
                posts = response.json().get("data", {}).get("children", [])
                for post in posts:
                    post_data = post.get("data", {})
                    
                    # RESEARCH FIX: Skip already-seen posts
                    post_id = post_data.get("id", "")
                    if post_id in seen_post_ids:
                        continue
                    seen_post_ids.add(post_id)
                    
                    title = post_data.get("title", "")
                    selftext = post_data.get("selftext", "")
                    if title:
                        all_texts.append(title)
                    if selftext and len(selftext) < 1000:
                        all_texts.append(selftext)
                        
            except Exception as e:
                logger.warning(f"Reddit search error for r/{subreddit}: {e}")
                continue
        
        if not all_texts:
            logger.info("📱 Reddit: No relevant posts found")
            return SourceSentiment(
                source=SentimentSource.REDDIT,
                score=0.0,
                confidence=0.0,
                sample_count=0,
            )
        
        # Analyze sentiment
        score = compute_sentiment_from_messages(all_texts)
        confidence = min(1.0, len(all_texts) / 30)
        
        logger.info(f"📱 Reddit (OAuth): score={score:.2f} (samples={len(all_texts)}, unique_posts={len(seen_post_ids)})")
        
        return SourceSentiment(
            source=SentimentSource.REDDIT,
            score=score,
            confidence=confidence,
            sample_count=len(all_texts),
        )
        
    except Exception as e:
        logger.warning(f"Reddit OAuth failed, falling back to public: {e}")
        return _get_reddit_sentiment_public()


def _get_reddit_sentiment_public() -> SourceSentiment:
    """Fetch Reddit sentiment using public JSON endpoint (no auth required).
    
    Uses Reddit's raw HTTP API with search.json endpoint.
    Rate-limited but works without any credentials.
    
    NOTE (Jan 2026 - Research Review):
    - Deduplication added to prevent double-counting posts found under
      multiple search terms, as identified in research review.
    
    Example: https://www.reddit.com/r/wallstreetbets/search.json?q=SPY&restrict_sr=1&sort=new
    """
    all_texts = []
    seen_post_ids: set = set()  # RESEARCH FIX: Deduplicate posts across search terms
    
    headers = {"User-Agent": REDDIT_USER_AGENT}
    
    # Search terms to look for
    search_terms = ["SPY", "SPX", "ES", "S&P"]
    
    for subreddit in REDDIT_SUBREDDITS[:2]:  # Limit to avoid rate limits
        for term in search_terms[:2]:  # Search top 2 terms per subreddit
            try:
                # Use search.json endpoint for targeted results
                url = f"https://www.reddit.com/r/{subreddit}/search.json"
                params = {
                    "q": term,
                    "restrict_sr": 1,  # Restrict to subreddit
                    "sort": "new",
                    "limit": 15,
                    "t": "day",  # Last 24 hours
                }
                
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=SENTIMENT_REQUEST_TIMEOUT,
                )
                
                if response.status_code == 429:
                    logger.warning(f"Reddit rate limit hit for r/{subreddit}")
                    time.sleep(2.0)
                    continue
                    
                response.raise_for_status()
                
                posts = response.json().get("data", {}).get("children", [])
                
                for post in posts:
                    post_data = post.get("data", {})
                    
                    # RESEARCH FIX: Skip already-seen posts to avoid double-counting
                    post_id = post_data.get("id", "")
                    if post_id in seen_post_ids:
                        continue
                    seen_post_ids.add(post_id)
                    
                    title = post_data.get("title", "")
                    selftext = post_data.get("selftext", "")
                    
                    if title:
                        all_texts.append(title)
                    if selftext and len(selftext) < 500:
                        all_texts.append(selftext)
                
                # Delay between requests to be respectful
                time.sleep(0.5)
                
            except requests.RequestException as e:
                logger.warning(f"Reddit search error for r/{subreddit} ({term}): {e}")
                continue
            except Exception as e:
                logger.warning(f"Reddit parsing error for r/{subreddit} ({term}): {e}")
                continue
    
    if not all_texts:
        logger.info("📱 Reddit (public): No relevant posts found")
        return SourceSentiment(
            source=SentimentSource.REDDIT,
            score=0.0,
            confidence=0.0,
            sample_count=0,
            error="No credentials and no relevant posts found",
        )
    
    score = compute_sentiment_from_messages(all_texts)
    confidence = min(1.0, len(all_texts) / 20)
    
    logger.info(f"📱 Reddit (public): score={score:.2f} (samples={len(all_texts)}, unique_posts={len(seen_post_ids)})")
    
    return SourceSentiment(
        source=SentimentSource.REDDIT,
        score=score,
        confidence=confidence,
        sample_count=len(all_texts),
    )


# ============================================================
# TWITTER / X SENTIMENT
# ============================================================

def get_twitter_sentiment() -> SourceSentiment:
    """Fetch sentiment from Twitter/X.
    
    Uses Twitter API v2 if bearer token available AND Twitter is enabled.
    Returns neutral (excluded from calculations) when disabled.
    
    Returns:
        SourceSentiment with aggregated score
    """
    # Check if Twitter is explicitly disabled
    if not TWITTER_ENABLED:
        # Return with 0 confidence so it's excluded from weighted calculation
        return SourceSentiment(
            source=SentimentSource.TWITTER,
            score=0.0,
            confidence=0.0,
            sample_count=0,
            error="Twitter disabled in configuration",
        )
    
    if not TWITTER_BEARER_TOKEN:
        logger.debug("Twitter: No bearer token configured, returning neutral")
        return SourceSentiment(
            source=SentimentSource.TWITTER,
            score=0.0,
            confidence=0.0,
            sample_count=0,
            error="No Twitter credentials configured",
        )
    
    try:
        headers = {
            "Authorization": f"Bearer {TWITTER_BEARER_TOKEN}",
            "User-Agent": BROWSER_USER_AGENT,
        }
        
        # Build search query
        query = " OR ".join(TWITTER_SEARCH_TERMS)
        query += " -is:retweet lang:en"  # Exclude retweets, English only
        
        url = "https://api.twitter.com/2/tweets/search/recent"
        params = {
            "query": query,
            "max_results": 50,
            "tweet.fields": "text,created_at",
        }
        
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=SENTIMENT_REQUEST_TIMEOUT,
        )
        
        if response.status_code == 401:
            logger.warning("Twitter: Invalid bearer token")
            return SourceSentiment(
                source=SentimentSource.TWITTER,
                score=0.0,
                confidence=0.0,
                sample_count=0,
                error="Invalid Twitter credentials",
            )
        
        if response.status_code == 429:
            logger.warning("Twitter: Rate limit exceeded")
            return SourceSentiment(
                source=SentimentSource.TWITTER,
                score=0.0,
                confidence=0.0,
                sample_count=0,
                error="Twitter rate limit exceeded",
            )
        
        response.raise_for_status()
        data = response.json()
        
        tweets = data.get("data", [])
        if not tweets:
            logger.info("🐦 Twitter: No recent tweets found")
            return SourceSentiment(
                source=SentimentSource.TWITTER,
                score=0.0,
                confidence=0.0,
                sample_count=0,
            )
        
        # Extract text and analyze
        texts = [tweet.get("text", "") for tweet in tweets]
        score = compute_sentiment_from_messages(texts)
        confidence = min(1.0, len(texts) / 30)
        
        logger.info(f"🐦 Twitter: score={score:.2f} (samples={len(texts)})")
        
        return SourceSentiment(
            source=SentimentSource.TWITTER,
            score=score,
            confidence=confidence,
            sample_count=len(texts),
        )
        
    except requests.Timeout:
        logger.warning("Twitter: Request timeout")
        return SourceSentiment(
            source=SentimentSource.TWITTER,
            score=0.0,
            confidence=0.0,
            sample_count=0,
            error="Request timeout",
        )
    except requests.RequestException as e:
        logger.warning(f"Twitter API error: {e}")
        return SourceSentiment(
            source=SentimentSource.TWITTER,
            score=0.0,
            confidence=0.0,
            sample_count=0,
            error=str(e)[:100],
        )
    except Exception as e:
        logger.warning(f"Twitter parsing error: {e}")
        return SourceSentiment(
            source=SentimentSource.TWITTER,
            score=0.0,
            confidence=0.0,
            sample_count=0,
            error=str(e)[:100],
        )


# ============================================================
# VIX-BASED MARKET SENTIMENT (Real market data, not social media)
# ============================================================

# Global VIX cache (fetched once per session or on demand)
_vix_cache: Dict[str, Any] = {"value": None, "timestamp": None}
_VIX_CACHE_TTL_SECONDS = 300  # 5 minutes

# VX Futures Feed integration (real-time IBKR data)
try:
    from .vx_futures_feed import get_vx_feed
    VX_FEED_AVAILABLE = True
except ImportError:
    VX_FEED_AVAILABLE = False


def get_vix_sentiment(vix_value: Optional[float] = None) -> SourceSentiment:
    """Get sentiment based on VIX/VX (CBOE Volatility Index / Futures).
    
    Priority:
    1. Use provided vix_value if given
    2. Try VX Futures Feed (real-time IBKR data) - more accurate
    3. Try cached value if fresh
    4. Fall back to Yahoo Finance (free, no API key)
    
    VIX is often called the "fear gauge" - it measures expected volatility:
    - VIX < 12: Extreme complacency (bullish, but watch for reversal)
    - VIX 12-15: Low fear (bullish)
    - VIX 15-20: Normal (neutral)
    - VIX 20-25: Elevated fear (cautious)
    - VIX 25-30: High fear (bearish, but contrarian opportunities)
    - VIX > 30: Extreme fear (bearish, potential capitulation)
    
    Args:
        vix_value: Current VIX value (if None, tries VX feed, cache, or Yahoo Finance)
        
    Returns:
        SourceSentiment with score based on VIX level
    """
    global _vix_cache
    source_label = "provided"
    
    try:
        # Priority 1: Use provided value
        if vix_value is not None:
            source_label = "provided"
        else:
            # Priority 2: Try VX Futures Feed (real-time IBKR data)
            if VX_FEED_AVAILABLE:
                vx_feed = get_vx_feed()
                if vx_feed and not vx_feed.is_stale():
                    vx_price = vx_feed.get_vx_price()
                    if vx_price is not None:
                        vix_value = vx_price
                        source_label = "vx_futures"
                        logger.debug(f"VIX sentiment using VX futures price: {vix_value:.2f}")
            
            # Priority 3: Check cache
            if vix_value is None:
                cache_age = 0
                if _vix_cache["timestamp"]:
                    cache_age = (datetime.now(timezone.utc) - _vix_cache["timestamp"]).total_seconds()
                
                if _vix_cache["value"] and cache_age < _VIX_CACHE_TTL_SECONDS:
                    vix_value = _vix_cache["value"]
                    source_label = "cache"
            
            # Priority 4: Fall back to Yahoo Finance
            if vix_value is None:
                vix_value = _fetch_vix_from_yahoo()
                if vix_value:
                    _vix_cache["value"] = vix_value
                    _vix_cache["timestamp"] = datetime.now(timezone.utc)
                    source_label = "yahoo"
        
        if vix_value is None:
            return SourceSentiment(
                source=SentimentSource.VIX,
                score=0.0,
                confidence=0.0,
                sample_count=0,
                error="VIX data unavailable",
            )
        
        # =================================================================
        # VIX/VX INTERPRETATION - IMPORTANT NUANCE
        # =================================================================
        # VIX measures EXPECTED VOLATILITY, not direction.
        # Higher VIX = market expects larger moves (up OR down).
        # 
        # Common relationship: ES down → VIX up (hedging demand)
        # But ~20% of the time they move together (rallies with hedging).
        #
        # For SENTIMENT scoring, we use VIX as a RISK REGIME indicator:
        # - Low VIX (<15): Complacent, good for trend-following
        # - Normal VIX (15-20): Standard conditions
        # - Elevated VIX (20-30): Uncertainty, need stronger confirmation
        # - Extreme VIX (>30): Crisis/panic, mean-reversion opportunities
        #
        # The score here influences entry filters, NOT direction.
        # A negative score means "be more cautious/require more confirmation"
        # NOT "the market is going down."
        # =================================================================
        
        if vix_value < 12:
            # Extreme complacency - low vol environment
            # Good for trend-following, but watch for vol expansion
            score = 0.3
            confidence = 0.7
            regime = "COMPLACENT"
        elif vix_value < 15:
            # Low volatility - favorable for normal trading
            score = 0.4
            confidence = 0.8
            regime = "LOW_VOL"
        elif vix_value < 20:
            # Normal range - standard conditions
            score = 0.0
            confidence = 0.6
            regime = "NORMAL"
        elif vix_value < 25:
            # Elevated uncertainty - require stronger signals
            score = -0.2
            confidence = 0.7
            regime = "ELEVATED"
        elif vix_value < 30:
            # High uncertainty - be very selective
            score = -0.4
            confidence = 0.8
            regime = "HIGH_UNCERTAINTY"
        else:
            # Extreme volatility - crisis/panic mode
            # Score is negative (cautious) but watch for mean-reversion
            # opportunities as panic often creates overshoots
            score = -0.5
            confidence = 0.6  # Lower confidence due to unpredictability
            regime = "CRISIS"
        
        logger.info(
            f"📊 VIX Regime: VIX={vix_value:.2f} -> {regime} "
            f"(score={score:+.2f}, conf={confidence:.2f}, source={source_label})"
        )
        
        return SourceSentiment(
            source=SentimentSource.VIX,
            score=score,
            confidence=confidence,
            sample_count=1,  # VIX is a single aggregate metric
        )
        
    except Exception as e:
        logger.warning(f"⚠️ VIX sentiment fetch failed: {e}")
        return SourceSentiment(
            source=SentimentSource.VIX,
            score=0.0,
            confidence=0.0,
            sample_count=0,
            error=str(e)[:100],
        )


def _fetch_vix_from_yahoo() -> Optional[float]:
    """Fetch current VIX value from Yahoo Finance (free, no API key)."""
    try:
        # Yahoo Finance quote page for VIX
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1m&range=1d"
        headers = {"User-Agent": BROWSER_USER_AGENT}
        
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        result = data.get("chart", {}).get("result", [])
        if result:
            meta = result[0].get("meta", {})
            vix_value = meta.get("regularMarketPrice")
            if vix_value:
                logger.debug(f"VIX fetched from Yahoo: {vix_value:.2f}")
                return float(vix_value)
        
        return None
        
    except Exception as e:
        logger.debug(f"Yahoo VIX fetch failed: {e}")
        return None


def set_vix_value(vix_value: float) -> None:
    """Manually set VIX value (e.g., from IBKR subscription).
    
    Call this from the trading manager when you have real-time VIX data.
    """
    global _vix_cache
    _vix_cache["value"] = vix_value
    _vix_cache["timestamp"] = datetime.now(timezone.utc)
    logger.debug(f"VIX cache updated: {vix_value:.2f}")


# ============================================================
# CNN FEAR & GREED INDEX (Market-based composite)
# ============================================================

_fear_greed_cache: Dict[str, Any] = {"value": None, "timestamp": None}
_FEAR_GREED_CACHE_TTL_SECONDS = 900  # 15 minutes (doesn't change frequently)


def get_fear_greed_sentiment() -> SourceSentiment:
    """Get sentiment from CNN Fear & Greed Index.
    
    This is a composite of 7 market indicators:
    1. Stock Price Momentum (S&P 500 vs 125-day MA)
    2. Stock Price Strength (52-week highs vs lows)
    3. Stock Price Breadth (McClellan Volume Summation)
    4. Put/Call Ratio
    5. Junk Bond Demand
    6. Market Volatility (VIX)
    7. Safe Haven Demand
    
    Scale: 0-100 (0=Extreme Fear, 50=Neutral, 100=Extreme Greed)
    
    Returns:
        SourceSentiment with score converted to [-1, 1]
    """
    global _fear_greed_cache
    
    try:
        # Check cache
        cache_age = 0
        if _fear_greed_cache["timestamp"]:
            cache_age = (datetime.now(timezone.utc) - _fear_greed_cache["timestamp"]).total_seconds()
        
        if _fear_greed_cache["value"] is not None and cache_age < _FEAR_GREED_CACHE_TTL_SECONDS:
            fg_value = _fear_greed_cache["value"]
        else:
            fg_value = _fetch_fear_greed_index()
            if fg_value is not None:
                _fear_greed_cache["value"] = fg_value
                _fear_greed_cache["timestamp"] = datetime.now(timezone.utc)
        
        if fg_value is None:
            return SourceSentiment(
                source=SentimentSource.COMBINED,  # Use COMBINED as placeholder
                score=0.0,
                confidence=0.0,
                sample_count=0,
                error="Fear & Greed data unavailable",
            )
        
        # Convert 0-100 to -1 to 1
        # 0 = Extreme Fear = -1.0
        # 50 = Neutral = 0.0
        # 100 = Extreme Greed = 1.0
        score = (fg_value - 50) / 50
        
        # Confidence based on how extreme the reading is
        distance_from_neutral = abs(fg_value - 50)
        confidence = min(1.0, 0.5 + (distance_from_neutral / 100))
        
        logger.info(
            f"📊 Fear & Greed Index: {fg_value:.0f} -> score={score:+.2f} "
            f"(confidence={confidence:.2f}) [ts={datetime.now(timezone.utc).strftime('%H:%M:%S')}]"
        )
        
        return SourceSentiment(
            source=SentimentSource.COMBINED,
            score=score,
            confidence=confidence,
            sample_count=7,  # Based on 7 indicators
        )
        
    except Exception as e:
        logger.warning(f"⚠️ Fear & Greed fetch failed: {e}")
        return SourceSentiment(
            source=SentimentSource.COMBINED,
            score=0.0,
            confidence=0.0,
            sample_count=0,
            error=str(e)[:100],
        )


def _fetch_fear_greed_index() -> Optional[float]:
    """Fetch Fear & Greed index from CNN (web scraping).
    
    Note: CNN may change their page structure - this is a best-effort fetch.
    Falls back to alternative API if scraping fails.
    """
    try:
        # Try Alternative.me API first (more reliable)
        url = "https://api.alternative.me/fng/?limit=1"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("data"):
                value = int(data["data"][0].get("value", 50))
                classification = data["data"][0].get("value_classification", "")
                logger.debug(f"Fear & Greed from Alternative.me: {value} ({classification})")
                return float(value)
    except Exception as e:
        logger.debug(f"Alternative.me F&G fetch failed: {e}")
    
    # Fallback: try to estimate from VIX
    try:
        vix_result = get_vix_sentiment()
        if vix_result.is_valid():
            # Convert VIX sentiment to F&G scale
            # VIX sentiment: -1 (fear) to +1 (greed)
            # F&G scale: 0 (fear) to 100 (greed)
            estimated_fg = (vix_result.score + 1) * 50
            logger.debug(f"Fear & Greed estimated from VIX: {estimated_fg:.0f}")
            return estimated_fg
    except Exception:
        pass
    
    return None


# ============================================================
# COMBINED SENTIMENT
# ============================================================

def _compute_weighted_sentiment(
    stocktwits: SourceSentiment,
    reddit: SourceSentiment,
    twitter: SourceSentiment,
    stocktwits_weight: float = STOCKTWITS_WEIGHT,
    reddit_weight: float = REDDIT_WEIGHT,
    twitter_weight: float = TWITTER_WEIGHT,
) -> float:
    """Compute weighted combined sentiment score.
    
    Weights are adjusted based on source validity and confidence.
    
    Returns:
        Combined score in [-1.0, 1.0]
    """
    # Collect valid sources with weights
    sources = []
    
    if stocktwits.is_valid():
        effective_weight = stocktwits_weight * stocktwits.confidence
        sources.append((stocktwits.score, effective_weight))
    
    if reddit.is_valid():
        effective_weight = reddit_weight * reddit.confidence
        sources.append((reddit.score, effective_weight))
    
    if twitter.is_valid():
        effective_weight = twitter_weight * twitter.confidence
        sources.append((twitter.score, effective_weight))
    
    if not sources:
        # No valid sources, try to use any score we have
        scores_with_data = []
        if stocktwits.sample_count > 0:
            scores_with_data.append(stocktwits.score)
        if reddit.sample_count > 0:
            scores_with_data.append(reddit.score)
        if twitter.sample_count > 0:
            scores_with_data.append(twitter.score)
        
        if scores_with_data:
            return sum(scores_with_data) / len(scores_with_data)
        return 0.0
    
    # Normalize weights
    total_weight = sum(w for _, w in sources)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(score * weight for score, weight in sources)
    combined = weighted_sum / total_weight
    
    # Clamp to [-1, 1]
    return max(-1.0, min(1.0, combined))


def get_combined_mes_sentiment(force_refresh: bool = False) -> CombinedSentiment:
    """Get combined sentiment from all sources for MES trading.
    
    Uses caching to avoid excessive API calls (5-minute refresh).
    
    Args:
        force_refresh: Bypass cache and fetch fresh data
        
    Returns:
        CombinedSentiment with scores from all sources
    """
    global _sentiment_cache
    
    # Check cache
    if not force_refresh and not _sentiment_cache.is_stale():
        cached = _sentiment_cache.get_cached()
        if cached is not None:
            age = _sentiment_cache.get_age_seconds()
            logger.info(f"📦 Using cached sentiment (age: {age:.0f}s): {cached.score:.2f}")
            return cached
    
    fetch_id = datetime.now(timezone.utc).strftime("%H%M%S")
    logger.info(f"🔄 Fetching fresh multi-source sentiment... [fetch_id={fetch_id}]")
    
    # Fetch from all sources (social + market-based)
    stocktwits = get_stocktwits_sentiment()
    reddit = get_reddit_sentiment()
    twitter = get_twitter_sentiment()
    
    # NEW: Also fetch market-based sentiment (VIX)
    vix_sentiment = get_vix_sentiment()
    
    # Compute combined score from social sources
    social_score = _compute_weighted_sentiment(stocktwits, reddit, twitter)
    
    # Blend with VIX sentiment if available (VIX is more reliable than social)
    # Weight: 60% social, 40% VIX (if VIX is valid)
    if vix_sentiment.is_valid():
        combined_score = (social_score * 0.6) + (vix_sentiment.score * 0.4)
        logger.info(f"   Blended with VIX: social={social_score:+.2f}, vix={vix_sentiment.score:+.2f} -> {combined_score:+.2f}")
    else:
        combined_score = social_score
    
    result = CombinedSentiment(
        score=combined_score,
        stocktwits=stocktwits,
        reddit=reddit,
        twitter=twitter,
    )
    
    # Update cache
    _sentiment_cache.update(result)
    
    # Log summary
    logger.info("=" * 50)
    logger.info("📊 MULTI-SOURCE SENTIMENT SUMMARY")
    logger.info("=" * 50)
    logger.info(f"   Stocktwits: {stocktwits.score:+.2f} ({stocktwits.sample_count} samples)")
    logger.info(f"   Reddit:     {reddit.score:+.2f} ({reddit.sample_count} samples)")
    logger.info(f"   Twitter:    {twitter.score:+.2f} ({twitter.sample_count} samples)")
    logger.info(f"   VIX:        {vix_sentiment.score:+.2f} (market-based)")
    logger.info(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"   COMBINED:   {combined_score:+.2f}")
    logger.info("=" * 50)
    
    return result


def get_mes_sentiment_score(force_refresh: bool = False) -> float:
    """Convenience function to get just the combined score.
    
    Args:
        force_refresh: Bypass cache
        
    Returns:
        Combined sentiment score in [-1.0, 1.0]
    """
    result = get_combined_mes_sentiment(force_refresh=force_refresh)
    return result.score


# ============================================================
# DECISION FUNCTIONS
# ============================================================

def evaluate_sentiment_for_entry(
    proposed_action: str,
    entry_block_threshold: float = ENTRY_BLOCK_THRESHOLD,
    weak_threshold: float = 0.2,
    force_refresh: bool = False,
    rsi_value: float = None,  # JAN 8 2026 FIX: Add RSI for contrarian logic
) -> SentimentDecision:
    """Evaluate if sentiment supports a proposed entry.
    
    JAN 8 2026 FIX: Added contrarian logic.
    When sentiment is extreme but RSI is extreme in the opposite direction,
    this is a CONTRARIAN signal - the crowd is wrong at extremes.
    
    Example: Bearish sentiment (-0.75) + oversold RSI (31) = contrarian BUY opportunity
    (crowd is max bearish but price is at technical support = bounce likely)
    
    Args:
        proposed_action: "LONG", "SHORT", "BUY", or "SELL"
        entry_block_threshold: Block if sentiment contradicts by this amount
        weak_threshold: Reduce confidence if sentiment weakly contradicts
        force_refresh: Bypass sentiment cache
        rsi_value: Optional RSI for contrarian logic (if None, no contrarian override)
        
    Returns:
        SentimentDecision with recommendation
    """
    combined = get_combined_mes_sentiment(force_refresh=force_refresh)
    score = combined.score
    
    # JAN 8 2026 FIX: Normalize action - handle BUY/SELL as well as LONG/SHORT
    action_upper = proposed_action.upper()
    if action_upper == "BUY":
        action_upper = "LONG"
    elif action_upper == "SELL":
        action_upper = "SHORT"
    
    # Source breakdown for logging
    source_breakdown = {
        "stocktwits": combined.stocktwits.score,
        "reddit": combined.reddit.score,
        "twitter": combined.twitter.score,
    }
    
    # ============================================================
    # JAN 8 2026 FIX: CONTRARIAN LOGIC
    # When sentiment is extreme but RSI is extreme OPPOSITE direction,
    # the crowd is likely wrong → contrarian opportunity
    # ============================================================
    contrarian_override = False
    contrarian_reason = None
    
    if rsi_value is not None:
        # Contrarian BUY: Bearish sentiment + oversold RSI → crowd is wrong, buy the dip
        if action_upper == "LONG" and score < -weak_threshold and rsi_value < 35:
            contrarian_override = True
            contrarian_reason = (
                f"CONTRARIAN BUY: Extreme bearish sentiment ({score:.2f}) + "
                f"oversold RSI ({rsi_value:.1f}) = crowd is max pessimistic at support"
            )
            logger.info(f"🔄 {contrarian_reason}")
            return SentimentDecision(
                allow_trade=True,
                action_recommendation="CONTRARIAN_BUY",
                confidence_modifier=CONFIDENCE_MODIFIER_MAX_BOOST,  # Boost confidence
                combined_score=score,
                reason=contrarian_reason,
                source_breakdown=source_breakdown,
            )
        
        # Contrarian SELL: Bullish sentiment + overbought RSI → crowd is wrong, sell the top
        elif action_upper == "SHORT" and score > weak_threshold and rsi_value > 65:
            contrarian_override = True
            contrarian_reason = (
                f"CONTRARIAN SELL: Extreme bullish sentiment ({score:.2f}) + "
                f"overbought RSI ({rsi_value:.1f}) = crowd is max optimistic at resistance"
            )
            logger.info(f"🔄 {contrarian_reason}")
            return SentimentDecision(
                allow_trade=True,
                action_recommendation="CONTRARIAN_SELL",
                confidence_modifier=CONFIDENCE_MODIFIER_MAX_BOOST,  # Boost confidence
                combined_score=score,
                reason=contrarian_reason,
                source_breakdown=source_breakdown,
            )
    
    # LONG entry evaluation
    # NOTE (Jan 2026 - Research Review): Thresholds are symmetric for LONG/SHORT
    if action_upper == "LONG":
        if score < -entry_block_threshold:
            return SentimentDecision(
                allow_trade=False,
                action_recommendation="BLOCK",
                confidence_modifier=0.0,
                combined_score=score,
                reason=f"Strong bearish sentiment ({score:.2f}) blocks LONG entry",
                source_breakdown=source_breakdown,
            )
        elif score < -weak_threshold:
            return SentimentDecision(
                allow_trade=True,
                action_recommendation="REDUCE_SIZE",
                confidence_modifier=0.7,  # 30% reduction
                combined_score=score,
                reason=f"Moderate bearish sentiment ({score:.2f}) - reduce position size",
                source_breakdown=source_breakdown,
            )
        elif score > weak_threshold:
            return SentimentDecision(
                allow_trade=True,
                action_recommendation="PROCEED",
                confidence_modifier=CONFIDENCE_MODIFIER_MAX_BOOST,  # Max 10% boost
                combined_score=score,
                reason=f"Bullish sentiment ({score:.2f}) supports LONG entry",
                source_breakdown=source_breakdown,
            )
        else:
            return SentimentDecision(
                allow_trade=True,
                action_recommendation="PROCEED",
                confidence_modifier=1.0,
                combined_score=score,
                reason=f"Neutral sentiment ({score:.2f})",
                source_breakdown=source_breakdown,
            )
    
    # SHORT entry evaluation
    elif action_upper == "SHORT":
        if score > entry_block_threshold:
            return SentimentDecision(
                allow_trade=False,
                action_recommendation="BLOCK",
                confidence_modifier=0.0,
                combined_score=score,
                reason=f"Strong bullish sentiment ({score:.2f}) blocks SHORT entry",
                source_breakdown=source_breakdown,
            )
        elif score > weak_threshold:
            return SentimentDecision(
                allow_trade=True,
                action_recommendation="REDUCE_SIZE",
                confidence_modifier=0.7,  # 30% reduction
                combined_score=score,
                reason=f"Moderate bullish sentiment ({score:.2f}) - reduce position size",
                source_breakdown=source_breakdown,
            )
        elif score < -weak_threshold:
            return SentimentDecision(
                allow_trade=True,
                action_recommendation="PROCEED",
                confidence_modifier=CONFIDENCE_MODIFIER_MAX_BOOST,  # Max 10% boost
                combined_score=score,
                reason=f"Bearish sentiment ({score:.2f}) supports SHORT entry",
                source_breakdown=source_breakdown,
            )
        else:
            return SentimentDecision(
                allow_trade=True,
                action_recommendation="PROCEED",
                confidence_modifier=1.0,
                combined_score=score,
                reason=f"Neutral sentiment ({score:.2f})",
                source_breakdown=source_breakdown,
            )
    
    # Default: allow trade
    return SentimentDecision(
        allow_trade=True,
        action_recommendation="PROCEED",
        confidence_modifier=1.0,
        combined_score=score,
        reason=f"Unknown action '{proposed_action}', proceeding with sentiment {score:.2f}",
        source_breakdown=source_breakdown,
    )


def evaluate_sentiment_for_position(
    current_position: str,
    exit_threshold: float = STRONG_EXIT_THRESHOLD,
    monitor_threshold: float = 0.3,  # RESEARCH FIX: Make symmetric and explicit
    force_refresh: bool = False,
) -> SentimentDecision:
    """Evaluate sentiment impact on existing position.
    
    NOTE (Jan 2026 - Research Review):
    - Thresholds are symmetric for LONG/SHORT positions
    - exit_threshold: extreme reversal → tighten stop
    - monitor_threshold: moderate reversal → increase monitoring
    
    Args:
        current_position: "LONG" or "SHORT"
        exit_threshold: Tighten stop if sentiment reverses by this amount
        monitor_threshold: Monitor closely if sentiment reverses by this amount
        force_refresh: Bypass sentiment cache
        
    Returns:
        SentimentDecision with position management recommendation
    """
    combined = get_combined_mes_sentiment(force_refresh=force_refresh)
    score = combined.score
    
    position_upper = current_position.upper()
    
    source_breakdown = {
        "stocktwits": combined.stocktwits.score,
        "reddit": combined.reddit.score,
        "twitter": combined.twitter.score,
    }
    
    # LONG position evaluation
    if position_upper == "LONG":
        if score < -exit_threshold:
            return SentimentDecision(
                allow_trade=True,  # Don't block existing position
                action_recommendation="TIGHTEN_STOP",
                confidence_modifier=CONFIDENCE_MODIFIER_MIN_REDUCTION,
                combined_score=score,
                reason=f"Strong bearish reversal ({score:.2f}) - tighten stop on LONG",
                source_breakdown=source_breakdown,
            )
        elif score < -monitor_threshold:
            return SentimentDecision(
                allow_trade=True,
                action_recommendation="MONITOR",
                confidence_modifier=0.8,
                combined_score=score,
                reason=f"Bearish sentiment ({score:.2f}) - monitor LONG closely",
                source_breakdown=source_breakdown,
            )
    
    # SHORT position evaluation
    elif position_upper == "SHORT":
        if score > exit_threshold:
            return SentimentDecision(
                allow_trade=True,
                action_recommendation="TIGHTEN_STOP",
                confidence_modifier=CONFIDENCE_MODIFIER_MIN_REDUCTION,
                combined_score=score,
                reason=f"Strong bullish reversal ({score:.2f}) - tighten stop on SHORT",
                source_breakdown=source_breakdown,
            )
        elif score > monitor_threshold:
            return SentimentDecision(
                allow_trade=True,
                action_recommendation="MONITOR",
                confidence_modifier=0.8,
                combined_score=score,
                reason=f"Bullish sentiment ({score:.2f}) - monitor SHORT closely",
                source_breakdown=source_breakdown,
            )
    
    # Position is fine
    return SentimentDecision(
        allow_trade=True,
        action_recommendation="HOLD",
        confidence_modifier=1.0,
        combined_score=score,
        reason=f"Sentiment ({score:.2f}) supports current {position_upper} position",
        source_breakdown=source_breakdown,
    )


# ============================================================
# SENTIMENT-DERIVED TREND (JAN 8 2026)
# ============================================================

@dataclass
class SentimentTrend:
    """Trend bias derived from sentiment data.
    
    Used to complement technical trend detection with social sentiment.
    When sentiment and technicals align, higher confidence trades.
    When they diverge, caution or contrarian opportunities.
    """
    trend: str  # "BULLISH", "BEARISH", "NEUTRAL"
    strength: float  # 0.0-1.0, how strong is the sentiment trend
    combined_score: float  # Raw sentiment score
    source_agreement: float  # 0.0-1.0, how much sources agree
    recommendation: str  # Human-readable advice
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def get_sentiment_trend(
    bullish_threshold: float = 0.20,
    bearish_threshold: float = -0.20,
    strong_threshold: float = 0.40,
    force_refresh: bool = False,
) -> SentimentTrend:
    """Derive trend bias from multi-source sentiment.
    
    This can be used as a leading indicator to complement technical trends.
    Sentiment often shifts BEFORE price confirms.
    
    Thresholds:
    - bullish_threshold: Score above this = BULLISH trend
    - bearish_threshold: Score below this = BEARISH trend
    - strong_threshold: Absolute score above this = STRONG signal
    
    Args:
        bullish_threshold: Score threshold for bullish trend
        bearish_threshold: Score threshold for bearish trend
        strong_threshold: Score magnitude for "strong" classification
        force_refresh: Bypass sentiment cache
        
    Returns:
        SentimentTrend with trend direction and confidence
    """
    combined = get_combined_mes_sentiment(force_refresh=force_refresh)
    score = combined.score
    
    # Calculate source agreement (how aligned are the sources?)
    sources = [
        combined.stocktwits.score if combined.stocktwits.is_valid() else 0,
        combined.reddit.score if combined.reddit.is_valid() else 0,
        combined.twitter.score if combined.twitter.is_valid() else 0,
    ]
    valid_sources = [s for s in sources if s != 0]
    
    if len(valid_sources) >= 2:
        # Check if sources agree on direction
        all_bullish = all(s > 0 for s in valid_sources)
        all_bearish = all(s < 0 for s in valid_sources)
        source_agreement = 1.0 if (all_bullish or all_bearish) else 0.5
    else:
        source_agreement = 0.3  # Low agreement if only one source
    
    # Determine trend direction
    if score >= bullish_threshold:
        trend = "BULLISH"
        strength = min(1.0, abs(score) / strong_threshold) if strong_threshold > 0 else 1.0
    elif score <= bearish_threshold:
        trend = "BEARISH"
        strength = min(1.0, abs(score) / strong_threshold) if strong_threshold > 0 else 1.0
    else:
        trend = "NEUTRAL"
        strength = 0.0
    
    # Generate recommendation
    if trend == "BULLISH" and strength > 0.8:
        recommendation = "Strong bullish sentiment - favor LONG entries"
    elif trend == "BULLISH":
        recommendation = "Mild bullish sentiment - slight LONG bias"
    elif trend == "BEARISH" and strength > 0.8:
        recommendation = "Strong bearish sentiment - favor SHORT entries"
    elif trend == "BEARISH":
        recommendation = "Mild bearish sentiment - slight SHORT bias"
    else:
        recommendation = "Neutral sentiment - rely on technicals"
    
    logger.info(f"📈 SENTIMENT TREND: {trend} (strength={strength:.2f}, agreement={source_agreement:.2f})")
    
    return SentimentTrend(
        trend=trend,
        strength=strength,
        combined_score=score,
        source_agreement=source_agreement,
        recommendation=recommendation,
    )


def combine_sentiment_with_technical_trend(
    technical_trend: str,
    sentiment_trend: Optional[SentimentTrend] = None,
    force_refresh: bool = False,
) -> Tuple[str, float, str]:
    """Combine technical and sentiment trends into a unified view.
    
    Returns a confidence-weighted trend that considers both inputs.
    
    Agreement cases:
    - Technical UPTREND + Sentiment BULLISH = HIGH confidence UPTREND
    - Technical DOWNTREND + Sentiment BEARISH = HIGH confidence DOWNTREND
    
    Divergence cases (potential opportunities or warnings):
    - Technical UPTREND + Sentiment BEARISH = Caution, potential reversal
    - Technical DOWNTREND + Sentiment BULLISH = Caution, potential bounce
    
    Args:
        technical_trend: From EMA/VWAP analysis ("UPTREND", "DOWNTREND", "NEUTRAL", "CHOP")
        sentiment_trend: Optional pre-fetched SentimentTrend (will fetch if None)
        force_refresh: Bypass sentiment cache
        
    Returns:
        Tuple of (combined_trend, confidence_boost, explanation)
        - combined_trend: "UPTREND", "DOWNTREND", or "NEUTRAL"
        - confidence_boost: -0.2 to +0.2 (add to signal confidence)
        - explanation: Human-readable description
    """
    if sentiment_trend is None:
        sentiment_trend = get_sentiment_trend(force_refresh=force_refresh)
    
    tech_upper = technical_trend.upper()
    sent_upper = sentiment_trend.trend.upper()
    
    # Map technical to simple direction
    tech_is_up = tech_upper in ("UPTREND", "BULLISH")
    tech_is_down = tech_upper in ("DOWNTREND", "BEARISH")
    tech_is_neutral = tech_upper in ("NEUTRAL", "CHOP", "RANGING")
    
    sent_is_up = sent_upper == "BULLISH"
    sent_is_down = sent_upper == "BEARISH"
    sent_is_neutral = sent_upper == "NEUTRAL"
    
    strength = sentiment_trend.strength
    agreement = sentiment_trend.source_agreement
    
    # === AGREEMENT CASES (boost confidence) ===
    if tech_is_up and sent_is_up:
        boost = 0.10 + (0.10 * strength * agreement)  # Up to +0.20
        return "UPTREND", boost, f"✅ ALIGNED: Technical & sentiment both bullish (+{boost:.2f})"
    
    if tech_is_down and sent_is_down:
        boost = 0.10 + (0.10 * strength * agreement)
        return "DOWNTREND", boost, f"✅ ALIGNED: Technical & sentiment both bearish (+{boost:.2f})"
    
    # === DIVERGENCE CASES (reduce confidence or warn) ===
    if tech_is_up and sent_is_down:
        penalty = -0.05 - (0.10 * strength * agreement)  # Up to -0.15
        return "UPTREND", penalty, f"⚠️ DIVERGENCE: Tech bullish but sentiment bearish ({penalty:+.2f})"
    
    if tech_is_down and sent_is_up:
        penalty = -0.05 - (0.10 * strength * agreement)
        return "DOWNTREND", penalty, f"⚠️ DIVERGENCE: Tech bearish but sentiment bullish ({penalty:+.2f})"
    
    # === NEUTRAL TECHNICAL (sentiment can provide direction) ===
    if tech_is_neutral:
        if sent_is_up and strength > 0.5:
            # Strong bullish sentiment in neutral market = slight upward bias
            boost = 0.05 * strength
            return "UPTREND", boost, f"📊 SENTIMENT LEAD: Neutral tech, bullish sentiment (+{boost:.2f})"
        if sent_is_down and strength > 0.5:
            boost = 0.05 * strength
            return "DOWNTREND", boost, f"📊 SENTIMENT LEAD: Neutral tech, bearish sentiment (+{boost:.2f})"
    
    # === DEFAULT: Use technical, no modification ===
    return technical_trend, 0.0, "Sentiment neutral - using technical trend"


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Data structures
    "SentimentSource",
    "SourceSentiment",
    "CombinedSentiment",
    "SentimentDecision",
    "SentimentTrend",  # JAN 8 2026
    # Individual source fetchers
    "get_stocktwits_sentiment",
    "get_reddit_sentiment",
    "get_twitter_sentiment",
    "get_vix_sentiment",  # JAN 9 2026 - Market-based sentiment
    "get_fear_greed_sentiment",  # JAN 9 2026 - CNN Fear & Greed
    "set_vix_value",  # JAN 9 2026 - Set VIX from IBKR
    # Combined sentiment
    "get_combined_mes_sentiment",
    "get_mes_sentiment_score",
    # Trend derivation (JAN 8 2026)
    "get_sentiment_trend",
    "combine_sentiment_with_technical_trend",
    # Decision functions
    "evaluate_sentiment_for_entry",
    "evaluate_sentiment_for_position",
    # NLP helpers
    "analyze_text_sentiment",
    "compute_sentiment_from_messages",
    # Cache control
    "set_cache_refresh_interval",
]
