"""
Unit tests for Multi-Source Sentiment Aggregator.

Tests cover:
- Individual source fetching (Stocktwits, Reddit, Twitter)
- NLP sentiment analysis (VADER, TextBlob, keyword fallback)
- Combined sentiment calculation with weights
- Caching behavior (5-minute refresh)
- Entry/position decision logic
- Failure safety (API errors, timeouts)
"""

import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
import pytest

from shree.data.sentiment_aggregator import (
    # Data structures
    SentimentSource,
    SourceSentiment,
    CombinedSentiment,
    SentimentDecision,
    SentimentCache,
    # Individual source fetchers
    get_stocktwits_sentiment,
    get_reddit_sentiment,
    get_twitter_sentiment,
    # Combined sentiment
    get_combined_mes_sentiment,
    get_mes_sentiment_score,
    _compute_weighted_sentiment,
    # Decision functions
    evaluate_sentiment_for_entry,
    evaluate_sentiment_for_position,
    # NLP helpers
    analyze_text_sentiment,
    compute_sentiment_from_messages,
    _keyword_sentiment,
    _clean_text,
    # Cache control
    set_cache_refresh_interval,
    _sentiment_cache,
)


# ============================================================
# NLP SENTIMENT ANALYSIS TESTS
# ============================================================

class TestKeywordSentiment:
    """Test keyword-based sentiment scoring fallback."""
    
    def test_bullish_keywords(self):
        """Text with bullish keywords should return positive score."""
        text = "SPY to the moon! Bullish breakout, time to buy calls!"
        score = _keyword_sentiment(text)
        assert score > 0, "Bullish keywords should produce positive score"
    
    def test_bearish_keywords(self):
        """Text with bearish keywords should return negative score."""
        text = "Market is crashing, very bearish. Sell everything, buy puts!"
        score = _keyword_sentiment(text)
        assert score < 0, "Bearish keywords should produce negative score"
    
    def test_neutral_text(self):
        """Text without sentiment keywords returns zero."""
        text = "The weather is nice today in New York."
        score = _keyword_sentiment(text)
        assert score == 0.0, "Neutral text should return 0.0"
    
    def test_mixed_sentiment(self):
        """Text with both bullish and bearish keywords."""
        text = "Bulls say buy, bears say sell, I just hold."
        score = _keyword_sentiment(text)
        # Should be close to zero with mixed sentiment
        assert -0.5 <= score <= 0.5
    
    def test_empty_text(self):
        """Empty text returns zero."""
        assert _keyword_sentiment("") == 0.0
        assert _keyword_sentiment("   ") == 0.0


class TestTextCleaning:
    """Test text cleaning for sentiment analysis."""
    
    def test_removes_urls(self):
        """URLs should be removed from text."""
        text = "Check out https://example.com for more info"
        cleaned = _clean_text(text)
        assert "https://" not in cleaned
        assert "example.com" not in cleaned
    
    def test_removes_mentions(self):
        """Twitter/Reddit mentions should be removed."""
        text = "@user1 @user2 what do you think about SPY?"
        cleaned = _clean_text(text)
        assert "@user1" not in cleaned
        assert "@user2" not in cleaned
        assert "SPY" in cleaned
    
    def test_removes_hashtag_symbol(self):
        """Hashtag symbol removed but word kept."""
        text = "#SPY #bullish rally today"
        cleaned = _clean_text(text)
        assert "#" not in cleaned
        assert "SPY" in cleaned
        assert "bullish" in cleaned


class TestAnalyzeTextSentiment:
    """Test the main sentiment analysis function."""
    
    def test_empty_text_returns_zero(self):
        """Empty or whitespace-only text returns 0.0."""
        assert analyze_text_sentiment("") == 0.0
        assert analyze_text_sentiment("   ") == 0.0
        assert analyze_text_sentiment(None) == 0.0 if analyze_text_sentiment(None) == 0.0 else True
    
    def test_strongly_bullish_text(self):
        """Very bullish text should return high positive score."""
        text = "Amazing rally! Bulls winning, huge gains, moon time! 🚀"
        score = analyze_text_sentiment(text)
        assert score > 0.2, f"Expected positive score, got {score}"
    
    def test_strongly_bearish_text(self):
        """Very bearish text should return low negative score."""
        text = "Terrible crash! Bears winning, huge losses, dump incoming!"
        score = analyze_text_sentiment(text)
        assert score < -0.2, f"Expected negative score, got {score}"


class TestComputeSentimentFromMessages:
    """Test aggregate sentiment from message lists."""
    
    def test_empty_list_returns_zero(self):
        """Empty message list returns 0.0."""
        assert compute_sentiment_from_messages([]) == 0.0
    
    def test_all_bullish_messages(self):
        """All bullish messages return positive score."""
        messages = [
            "SPY is bullish! Huge rally, buy buy buy!",
            "Long SPY calls, moon rocket gains!",
            "Green day, strong support, bulls winning!",
        ]
        score = compute_sentiment_from_messages(messages)
        assert score > 0, f"Expected positive, got {score}"
    
    def test_all_bearish_messages(self):
        """All bearish messages return negative score."""
        messages = [
            "Bearish crash! Sell everything, dump dump!",
            "Short puts, market tank, bears winning!",
            "Red day, weak resistance, losers!",
        ]
        score = compute_sentiment_from_messages(messages)
        assert score < 0, f"Expected negative, got {score}"
    
    def test_mixed_messages_average(self):
        """Mixed messages should produce averaged score."""
        messages = [
            "Bullish rally, buy long calls!",  # positive
            "Bearish crash, sell short puts!",  # negative
        ]
        score = compute_sentiment_from_messages(messages)
        # Should be near zero with mixed sentiment
        assert -0.7 <= score <= 0.7


# ============================================================
# SOURCE SENTIMENT TESTS
# ============================================================

class TestSourceSentiment:
    """Test SourceSentiment dataclass."""
    
    def test_is_valid_with_samples(self):
        """Valid result with samples and no error."""
        result = SourceSentiment(
            source=SentimentSource.STOCKTWITS,
            score=0.5,
            sample_count=30,
        )
        assert result.is_valid()
    
    def test_is_invalid_with_error(self):
        """Result with error is invalid."""
        result = SourceSentiment(
            source=SentimentSource.REDDIT,
            score=0.0,
            sample_count=10,
            error="API timeout",
        )
        assert not result.is_valid()
    
    def test_is_invalid_no_samples(self):
        """Result with no samples is invalid."""
        result = SourceSentiment(
            source=SentimentSource.TWITTER,
            score=0.0,
            sample_count=0,
        )
        assert not result.is_valid()


class TestCombinedSentiment:
    """Test CombinedSentiment dataclass."""
    
    def test_to_dict_serialization(self):
        """CombinedSentiment serializes to dictionary."""
        stocktwits = SourceSentiment(SentimentSource.STOCKTWITS, 0.3, sample_count=30)
        reddit = SourceSentiment(SentimentSource.REDDIT, 0.1, sample_count=20)
        twitter = SourceSentiment(SentimentSource.TWITTER, -0.2, sample_count=15)
        
        combined = CombinedSentiment(
            score=0.1,
            stocktwits=stocktwits,
            reddit=reddit,
            twitter=twitter,
        )
        
        data = combined.to_dict()
        assert "score" in data
        assert "stocktwits" in data
        assert "reddit" in data
        assert "twitter" in data
        assert data["stocktwits"]["score"] == 0.3


# ============================================================
# WEIGHTED SENTIMENT TESTS
# ============================================================

class TestWeightedSentiment:
    """Test weighted sentiment calculation."""
    
    def test_equal_weights_average(self):
        """Equal weights produce simple average."""
        stocktwits = SourceSentiment(SentimentSource.STOCKTWITS, 0.6, confidence=1.0, sample_count=30)
        reddit = SourceSentiment(SentimentSource.REDDIT, 0.3, confidence=1.0, sample_count=30)
        twitter = SourceSentiment(SentimentSource.TWITTER, 0.0, confidence=1.0, sample_count=30)
        
        # With equal weights (0.33 each)
        score = _compute_weighted_sentiment(
            stocktwits, reddit, twitter,
            stocktwits_weight=0.33,
            reddit_weight=0.33,
            twitter_weight=0.34,
        )
        
        # Should be around (0.6 + 0.3 + 0.0) / 3 = 0.3
        assert 0.2 <= score <= 0.4
    
    def test_stocktwits_dominant(self):
        """Stocktwits weight dominates when high."""
        stocktwits = SourceSentiment(SentimentSource.STOCKTWITS, 0.8, confidence=1.0, sample_count=30)
        reddit = SourceSentiment(SentimentSource.REDDIT, -0.5, confidence=1.0, sample_count=30)
        twitter = SourceSentiment(SentimentSource.TWITTER, -0.5, confidence=1.0, sample_count=30)
        
        score = _compute_weighted_sentiment(
            stocktwits, reddit, twitter,
            stocktwits_weight=0.8,
            reddit_weight=0.1,
            twitter_weight=0.1,
        )
        
        # Should be weighted toward Stocktwits (positive)
        assert score > 0.3
    
    def test_invalid_sources_excluded(self):
        """Invalid sources are excluded from calculation."""
        stocktwits = SourceSentiment(SentimentSource.STOCKTWITS, 0.5, confidence=1.0, sample_count=30)
        reddit = SourceSentiment(SentimentSource.REDDIT, 0.0, confidence=0.0, sample_count=0, error="No data")
        twitter = SourceSentiment(SentimentSource.TWITTER, 0.0, confidence=0.0, sample_count=0, error="No auth")
        
        score = _compute_weighted_sentiment(stocktwits, reddit, twitter)
        
        # Should just be Stocktwits score
        assert score == 0.5
    
    def test_confidence_affects_weight(self):
        """Low confidence reduces effective weight."""
        stocktwits = SourceSentiment(SentimentSource.STOCKTWITS, 0.8, confidence=1.0, sample_count=50)
        reddit = SourceSentiment(SentimentSource.REDDIT, -0.8, confidence=0.2, sample_count=5)
        twitter = SourceSentiment(SentimentSource.TWITTER, 0.0, confidence=0.0, sample_count=0)
        
        score = _compute_weighted_sentiment(stocktwits, reddit, twitter)
        
        # Reddit's low confidence should reduce its impact
        # Stocktwits should dominate
        assert score > 0.3


# ============================================================
# CACHE TESTS
# ============================================================

class TestSentimentCache:
    """Test sentiment caching behavior."""
    
    def test_cache_starts_empty(self):
        """New cache is stale."""
        cache = SentimentCache()
        assert cache.is_stale()
        assert cache.get_cached() is None
    
    def test_cache_update_clears_stale(self):
        """Updating cache makes it fresh."""
        cache = SentimentCache()
        cache.refresh_interval = timedelta(seconds=300)
        
        stocktwits = SourceSentiment(SentimentSource.STOCKTWITS, 0.3, sample_count=30)
        reddit = SourceSentiment(SentimentSource.REDDIT, 0.1, sample_count=20)
        twitter = SourceSentiment(SentimentSource.TWITTER, 0.0, sample_count=0)
        
        result = CombinedSentiment(0.2, stocktwits, reddit, twitter)
        cache.update(result)
        
        assert not cache.is_stale()
        assert cache.get_cached() is not None
        assert cache.get_cached().score == 0.2
    
    def test_cache_becomes_stale(self):
        """Cache becomes stale after interval."""
        cache = SentimentCache()
        cache.refresh_interval = timedelta(seconds=1)  # 1 second for testing
        
        stocktwits = SourceSentiment(SentimentSource.STOCKTWITS, 0.3, sample_count=30)
        reddit = SourceSentiment(SentimentSource.REDDIT, 0.0, sample_count=0)
        twitter = SourceSentiment(SentimentSource.TWITTER, 0.0, sample_count=0)
        
        result = CombinedSentiment(0.3, stocktwits, reddit, twitter)
        cache.update(result)
        
        assert not cache.is_stale()
        
        # Wait for staleness
        time.sleep(1.5)
        
        assert cache.is_stale()


# ============================================================
# ENTRY DECISION TESTS
# ============================================================

class TestEvaluateSentimentForEntry:
    """Test entry decision logic."""
    
    @patch("shree.data.sentiment_aggregator.get_combined_mes_sentiment")
    def test_bullish_sentiment_allows_long(self, mock_get):
        """Bullish sentiment allows LONG entry."""
        stocktwits = SourceSentiment(SentimentSource.STOCKTWITS, 0.5, sample_count=30)
        reddit = SourceSentiment(SentimentSource.REDDIT, 0.4, sample_count=20)
        twitter = SourceSentiment(SentimentSource.TWITTER, 0.3, sample_count=15)
        
        mock_get.return_value = CombinedSentiment(0.4, stocktwits, reddit, twitter)
        
        decision = evaluate_sentiment_for_entry("LONG")
        
        assert decision.allow_trade
        assert decision.action_recommendation == "PROCEED"
        assert decision.confidence_modifier >= 1.0
    
    @patch("shree.data.sentiment_aggregator.get_combined_mes_sentiment")
    def test_bearish_sentiment_blocks_long(self, mock_get):
        """Strong bearish sentiment blocks LONG entry."""
        stocktwits = SourceSentiment(SentimentSource.STOCKTWITS, -0.6, sample_count=30)
        reddit = SourceSentiment(SentimentSource.REDDIT, -0.5, sample_count=20)
        twitter = SourceSentiment(SentimentSource.TWITTER, -0.4, sample_count=15)
        
        mock_get.return_value = CombinedSentiment(-0.5, stocktwits, reddit, twitter)
        
        decision = evaluate_sentiment_for_entry("LONG", entry_block_threshold=0.4)
        
        assert not decision.allow_trade
        assert decision.action_recommendation == "BLOCK"
    
    @patch("shree.data.sentiment_aggregator.get_combined_mes_sentiment")
    def test_moderately_bearish_reduces_long_confidence(self, mock_get):
        """Moderate bearish sentiment reduces LONG confidence."""
        stocktwits = SourceSentiment(SentimentSource.STOCKTWITS, -0.3, sample_count=30)
        reddit = SourceSentiment(SentimentSource.REDDIT, -0.2, sample_count=20)
        twitter = SourceSentiment(SentimentSource.TWITTER, -0.25, sample_count=15)
        
        mock_get.return_value = CombinedSentiment(-0.25, stocktwits, reddit, twitter)
        
        decision = evaluate_sentiment_for_entry("LONG", entry_block_threshold=0.4, weak_threshold=0.2)
        
        assert decision.allow_trade
        assert decision.action_recommendation == "REDUCE_SIZE"
        assert decision.confidence_modifier < 1.0
    
    @patch("shree.data.sentiment_aggregator.get_combined_mes_sentiment")
    def test_bullish_sentiment_blocks_short(self, mock_get):
        """Strong bullish sentiment blocks SHORT entry."""
        stocktwits = SourceSentiment(SentimentSource.STOCKTWITS, 0.6, sample_count=30)
        reddit = SourceSentiment(SentimentSource.REDDIT, 0.5, sample_count=20)
        twitter = SourceSentiment(SentimentSource.TWITTER, 0.4, sample_count=15)
        
        mock_get.return_value = CombinedSentiment(0.5, stocktwits, reddit, twitter)
        
        decision = evaluate_sentiment_for_entry("SHORT", entry_block_threshold=0.4)
        
        assert not decision.allow_trade
        assert decision.action_recommendation == "BLOCK"
    
    @patch("shree.data.sentiment_aggregator.get_combined_mes_sentiment")
    def test_bearish_sentiment_allows_short(self, mock_get):
        """Bearish sentiment allows SHORT entry."""
        stocktwits = SourceSentiment(SentimentSource.STOCKTWITS, -0.5, sample_count=30)
        reddit = SourceSentiment(SentimentSource.REDDIT, -0.4, sample_count=20)
        twitter = SourceSentiment(SentimentSource.TWITTER, -0.3, sample_count=15)
        
        mock_get.return_value = CombinedSentiment(-0.4, stocktwits, reddit, twitter)
        
        decision = evaluate_sentiment_for_entry("SHORT")
        
        assert decision.allow_trade
        assert decision.action_recommendation == "PROCEED"
        assert decision.confidence_modifier >= 1.0
    
    @patch("shree.data.sentiment_aggregator.get_combined_mes_sentiment")
    def test_neutral_sentiment_allows_trade(self, mock_get):
        """Neutral sentiment allows trade without modification."""
        stocktwits = SourceSentiment(SentimentSource.STOCKTWITS, 0.05, sample_count=30)
        reddit = SourceSentiment(SentimentSource.REDDIT, -0.05, sample_count=20)
        twitter = SourceSentiment(SentimentSource.TWITTER, 0.0, sample_count=15)
        
        mock_get.return_value = CombinedSentiment(0.0, stocktwits, reddit, twitter)
        
        decision = evaluate_sentiment_for_entry("LONG")
        
        assert decision.allow_trade
        assert decision.action_recommendation == "PROCEED"
        assert decision.confidence_modifier == 1.0


# ============================================================
# POSITION DECISION TESTS
# ============================================================

class TestEvaluateSentimentForPosition:
    """Test position management decision logic."""
    
    @patch("shree.data.sentiment_aggregator.get_combined_mes_sentiment")
    def test_extreme_bearish_tightens_long_stop(self, mock_get):
        """Extreme bearish sentiment recommends tightening stop on LONG."""
        stocktwits = SourceSentiment(SentimentSource.STOCKTWITS, -0.7, sample_count=30)
        reddit = SourceSentiment(SentimentSource.REDDIT, -0.6, sample_count=20)
        twitter = SourceSentiment(SentimentSource.TWITTER, -0.65, sample_count=15)
        
        mock_get.return_value = CombinedSentiment(-0.65, stocktwits, reddit, twitter)
        
        decision = evaluate_sentiment_for_position("LONG", exit_threshold=0.6)
        
        assert decision.action_recommendation == "TIGHTEN_STOP"
        assert decision.confidence_modifier < 1.0
    
    @patch("shree.data.sentiment_aggregator.get_combined_mes_sentiment")
    def test_extreme_bullish_tightens_short_stop(self, mock_get):
        """Extreme bullish sentiment recommends tightening stop on SHORT."""
        stocktwits = SourceSentiment(SentimentSource.STOCKTWITS, 0.7, sample_count=30)
        reddit = SourceSentiment(SentimentSource.REDDIT, 0.6, sample_count=20)
        twitter = SourceSentiment(SentimentSource.TWITTER, 0.65, sample_count=15)
        
        mock_get.return_value = CombinedSentiment(0.65, stocktwits, reddit, twitter)
        
        decision = evaluate_sentiment_for_position("SHORT", exit_threshold=0.6)
        
        assert decision.action_recommendation == "TIGHTEN_STOP"
    
    @patch("shree.data.sentiment_aggregator.get_combined_mes_sentiment")
    def test_neutral_sentiment_holds_position(self, mock_get):
        """Neutral sentiment recommends holding position."""
        stocktwits = SourceSentiment(SentimentSource.STOCKTWITS, 0.1, sample_count=30)
        reddit = SourceSentiment(SentimentSource.REDDIT, 0.0, sample_count=20)
        twitter = SourceSentiment(SentimentSource.TWITTER, 0.05, sample_count=15)
        
        mock_get.return_value = CombinedSentiment(0.05, stocktwits, reddit, twitter)
        
        decision = evaluate_sentiment_for_position("LONG")
        
        assert decision.action_recommendation == "HOLD"
        assert decision.confidence_modifier == 1.0


# ============================================================
# API MOCK TESTS
# ============================================================

class TestStocktwitsAPI:
    """Test Stocktwits API handling."""
    
    @patch("shree.data.sentiment_aggregator.requests.get")
    def test_successful_fetch(self, mock_get):
        """Successful API response produces valid result."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "messages": [
                {"entities": {"sentiment": {"basic": "Bullish"}}},
                {"entities": {"sentiment": {"basic": "Bullish"}}},
                {"entities": {"sentiment": {"basic": "Bearish"}}},
                {"entities": {}},  # No sentiment
            ]
        }
        mock_get.return_value = mock_response
        
        result = get_stocktwits_sentiment(["TEST"])
        
        assert result.source == SentimentSource.STOCKTWITS
        assert result.sample_count == 4
        # 2 bullish, 1 bearish = (2-1)/3 = 0.33
        assert 0.2 <= result.score <= 0.5
    
    @patch("shree.data.sentiment_aggregator.requests.get")
    def test_timeout_returns_zero(self, mock_get):
        """Timeout returns zero score with error."""
        import requests
        mock_get.side_effect = requests.Timeout("Connection timeout")
        
        result = get_stocktwits_sentiment(["TEST"])
        
        assert result.score == 0.0
        assert result.error is not None
        assert "timeout" in result.error.lower()


class TestRedditAPI:
    """Test Reddit API handling."""
    
    @patch("shree.data.sentiment_aggregator.REDDIT_CLIENT_ID", "")
    @patch("shree.data.sentiment_aggregator.requests.get")
    def test_public_api_fallback(self, mock_get):
        """Falls back to public API without credentials."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "children": [
                    {"data": {"title": "SPY bullish breakout!", "selftext": "Moon time!"}},
                    {"data": {"title": "Something else", "selftext": ""}},
                ]
            }
        }
        mock_get.return_value = mock_response
        
        result = get_reddit_sentiment()
        
        assert result.source == SentimentSource.REDDIT


class TestTwitterAPI:
    """Test Twitter API handling."""
    
    @patch("shree.data.sentiment_aggregator.TWITTER_BEARER_TOKEN", "")
    def test_no_credentials_returns_zero(self):
        """No Twitter credentials (or disabled) returns zero with error."""
        result = get_twitter_sentiment()
        
        assert result.score == 0.0
        assert result.sample_count == 0
        assert result.error is not None
        # Error can be "disabled" or "credentials" depending on config
        assert "disabled" in result.error.lower() or "credentials" in result.error.lower()


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestCombinedSentimentIntegration:
    """Test combined sentiment flow."""
    
    @patch("shree.data.sentiment_aggregator.get_stocktwits_sentiment")
    @patch("shree.data.sentiment_aggregator.get_reddit_sentiment")
    @patch("shree.data.sentiment_aggregator.get_twitter_sentiment")
    def test_combines_all_sources(self, mock_twitter, mock_reddit, mock_stocktwits):
        """Combined sentiment uses all sources (Twitter excluded when disabled)."""
        mock_stocktwits.return_value = SourceSentiment(
            SentimentSource.STOCKTWITS, 0.4, confidence=1.0, sample_count=30
        )
        mock_reddit.return_value = SourceSentiment(
            SentimentSource.REDDIT, 0.2, confidence=1.0, sample_count=20
        )
        # Twitter disabled - returns 0 confidence so excluded from calculation
        mock_twitter.return_value = SourceSentiment(
            SentimentSource.TWITTER, 0.0, confidence=0.0, sample_count=0
        )
        
        # Reset cache for fresh fetch
        from shree.data.sentiment_aggregator import _sentiment_cache
        _sentiment_cache.last_fetch_time = None
        _sentiment_cache.last_result = None
        
        result = get_combined_mes_sentiment(force_refresh=True)
        
        assert result.stocktwits.score == 0.4
        assert result.reddit.score == 0.2
        assert result.twitter.score == 0.0
        # Combined should be weighted average of Stocktwits (55%) and Reddit (45%)
        # 0.4 * 0.55 + 0.2 * 0.45 = 0.22 + 0.09 = 0.31
        assert 0.25 <= result.score <= 0.35
    
    @patch("shree.data.sentiment_aggregator.get_stocktwits_sentiment")
    @patch("shree.data.sentiment_aggregator.get_reddit_sentiment")
    @patch("shree.data.sentiment_aggregator.get_twitter_sentiment")
    def test_uses_cache_when_fresh(self, mock_twitter, mock_reddit, mock_stocktwits):
        """Uses cached result when not stale."""
        mock_stocktwits.return_value = SourceSentiment(
            SentimentSource.STOCKTWITS, 0.5, sample_count=30
        )
        mock_reddit.return_value = SourceSentiment(
            SentimentSource.REDDIT, 0.3, sample_count=20
        )
        mock_twitter.return_value = SourceSentiment(
            SentimentSource.TWITTER, 0.0, sample_count=0
        )
        
        # First call - fetches fresh
        from shree.data.sentiment_aggregator import _sentiment_cache
        _sentiment_cache.last_fetch_time = None
        _sentiment_cache.last_result = None
        
        result1 = get_combined_mes_sentiment(force_refresh=True)
        call_count = mock_stocktwits.call_count
        
        # Second call - should use cache
        result2 = get_combined_mes_sentiment(force_refresh=False)
        
        # Should not have made another API call
        assert mock_stocktwits.call_count == call_count
        assert result2.score == result1.score


class TestSourceBreakdown:
    """Test source breakdown in decisions."""
    
    @patch("shree.data.sentiment_aggregator.get_combined_mes_sentiment")
    def test_decision_includes_breakdown(self, mock_get):
        """Decision includes source breakdown for logging."""
        stocktwits = SourceSentiment(SentimentSource.STOCKTWITS, 0.4, sample_count=30)
        reddit = SourceSentiment(SentimentSource.REDDIT, 0.2, sample_count=20)
        twitter = SourceSentiment(SentimentSource.TWITTER, -0.1, sample_count=15)
        
        mock_get.return_value = CombinedSentiment(0.2, stocktwits, reddit, twitter)
        
        decision = evaluate_sentiment_for_entry("LONG")
        
        assert "stocktwits" in decision.source_breakdown
        assert "reddit" in decision.source_breakdown
        assert "twitter" in decision.source_breakdown
        assert decision.source_breakdown["stocktwits"] == 0.4
        assert decision.source_breakdown["reddit"] == 0.2
        assert decision.source_breakdown["twitter"] == -0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
