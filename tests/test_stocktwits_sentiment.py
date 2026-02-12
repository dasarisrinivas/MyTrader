"""Tests for Stocktwits sentiment service."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from shree.data.stocktwits_sentiment import (
    SentimentCache,
    SentimentResult,
    calculate_sentiment_score_from_messages,
    evaluate_sentiment_for_entry,
    evaluate_sentiment_for_position,
    get_mes_sentiment,
    get_stocktwits_sentiment,
    reset_sentiment_cache,
    set_cache_refresh_interval,
)


class TestSentimentCalculation:
    """Test sentiment score calculation from message lists."""

    def test_all_bullish_returns_positive_one(self):
        """All bullish messages should return score = 1.0."""
        messages = [
            {"entities": {"sentiment": {"basic": "Bullish"}}},
            {"entities": {"sentiment": {"basic": "Bullish"}}},
            {"entities": {"sentiment": {"basic": "Bullish"}}},
        ]
        result = calculate_sentiment_score_from_messages(messages)
        assert result.score == 1.0
        assert result.bullish_count == 3
        assert result.bearish_count == 0

    def test_all_bearish_returns_negative_one(self):
        """All bearish messages should return score = -1.0."""
        messages = [
            {"entities": {"sentiment": {"basic": "Bearish"}}},
            {"entities": {"sentiment": {"basic": "Bearish"}}},
            {"entities": {"sentiment": {"basic": "Bearish"}}},
        ]
        result = calculate_sentiment_score_from_messages(messages)
        assert result.score == -1.0
        assert result.bullish_count == 0
        assert result.bearish_count == 3

    def test_equal_bullish_bearish_returns_zero(self):
        """Equal bullish and bearish should return score = 0.0."""
        messages = [
            {"entities": {"sentiment": {"basic": "Bullish"}}},
            {"entities": {"sentiment": {"basic": "Bearish"}}},
            {"entities": {"sentiment": {"basic": "Bullish"}}},
            {"entities": {"sentiment": {"basic": "Bearish"}}},
        ]
        result = calculate_sentiment_score_from_messages(messages)
        assert result.score == 0.0
        assert result.bullish_count == 2
        assert result.bearish_count == 2

    def test_mixed_sentiment_weighted_correctly(self):
        """3 bullish, 1 bearish should return 0.5."""
        messages = [
            {"entities": {"sentiment": {"basic": "Bullish"}}},
            {"entities": {"sentiment": {"basic": "Bullish"}}},
            {"entities": {"sentiment": {"basic": "Bullish"}}},
            {"entities": {"sentiment": {"basic": "Bearish"}}},
        ]
        result = calculate_sentiment_score_from_messages(messages)
        # (3 - 1) / 4 = 0.5
        assert result.score == 0.5
        assert result.bullish_count == 3
        assert result.bearish_count == 1

    def test_no_sentiment_labels_returns_zero(self):
        """Messages without sentiment labels should return 0.0."""
        messages = [
            {"body": "Just watching the market"},
            {"entities": {}},
            {"entities": {"other_field": "value"}},
        ]
        result = calculate_sentiment_score_from_messages(messages)
        assert result.score == 0.0
        assert result.bullish_count == 0
        assert result.bearish_count == 0
        assert result.neutral_count == 3

    def test_empty_messages_returns_zero(self):
        """Empty message list should return 0.0."""
        result = calculate_sentiment_score_from_messages([])
        assert result.score == 0.0
        assert result.total_messages == 0

    def test_neutral_messages_ignored_in_score(self):
        """Neutral messages should not affect the score calculation."""
        messages = [
            {"entities": {"sentiment": {"basic": "Bullish"}}},
            {"entities": {}},  # Neutral
            {"entities": {"sentiment": {"basic": "Bearish"}}},
            {"body": "No sentiment"},  # Neutral
        ]
        result = calculate_sentiment_score_from_messages(messages)
        # Only 1 bullish, 1 bearish with sentiment labels
        # (1 - 1) / 2 = 0.0
        assert result.score == 0.0
        assert result.bullish_count == 1
        assert result.bearish_count == 1
        assert result.neutral_count == 2


class TestSentimentCache:
    """Test sentiment caching behavior."""

    def test_cache_starts_empty(self):
        """New cache should be stale."""
        cache = SentimentCache()
        assert cache.is_stale()
        assert cache.get_cached_scores() == {}

    def test_cache_update_refreshes_timestamp(self):
        """Updating cache should refresh timestamp."""
        cache = SentimentCache()
        cache.update({"ES_F": 0.5, "SPY": 0.3})
        assert not cache.is_stale()
        assert cache.get_cached_scores() == {"ES_F": 0.5, "SPY": 0.3}

    def test_cache_becomes_stale_after_interval(self):
        """Cache should become stale after refresh interval."""
        cache = SentimentCache(refresh_interval_seconds=0.1)
        cache.update({"ES_F": 0.5})
        assert not cache.is_stale()
        
        # Wait for cache to expire
        time.sleep(0.15)
        assert cache.is_stale()

    def test_cache_returns_copy_of_scores(self):
        """Cache should return a copy, not the original dict."""
        cache = SentimentCache()
        cache.update({"ES_F": 0.5})
        scores = cache.get_cached_scores()
        scores["SPY"] = 0.3  # Modify the returned dict
        
        # Original cache should be unchanged
        assert cache.get_cached_scores() == {"ES_F": 0.5}


class TestEvaluateSentimentForEntry:
    """Test sentiment evaluation for new trade entries."""

    @patch('shree.data.stocktwits_sentiment.get_mes_sentiment')
    def test_bullish_sentiment_allows_long(self, mock_sentiment):
        """Bullish sentiment should allow long entry."""
        mock_sentiment.return_value = 0.5
        result = evaluate_sentiment_for_entry("BUY")
        
        assert result.allow_trade is True
        assert result.confidence_modifier >= 1.0  # Should boost confidence
        assert "supports long" in result.reason.lower()

    @patch('shree.data.stocktwits_sentiment.get_mes_sentiment')
    def test_bearish_sentiment_blocks_long(self, mock_sentiment):
        """Strongly bearish sentiment should block long entry."""
        mock_sentiment.return_value = -0.5  # Below -0.4 threshold
        result = evaluate_sentiment_for_entry("BUY", entry_block_threshold=0.4)
        
        assert result.allow_trade is False
        assert result.action_recommendation == "BLOCK"
        assert "blocking long" in result.reason.lower()

    @patch('shree.data.stocktwits_sentiment.get_mes_sentiment')
    def test_moderately_bearish_reduces_long_confidence(self, mock_sentiment):
        """Moderately bearish sentiment should reduce long confidence."""
        mock_sentiment.return_value = -0.3  # Between -0.4 and -0.2
        result = evaluate_sentiment_for_entry("BUY", entry_block_threshold=0.4, weak_threshold=0.2)
        
        assert result.allow_trade is True
        assert result.confidence_modifier < 1.0  # Should reduce confidence
        assert result.action_recommendation == "REDUCE_SIZE"

    @patch('shree.data.stocktwits_sentiment.get_mes_sentiment')
    def test_bullish_sentiment_blocks_short(self, mock_sentiment):
        """Strongly bullish sentiment should block short entry."""
        mock_sentiment.return_value = 0.5  # Above +0.4 threshold
        result = evaluate_sentiment_for_entry("SELL", entry_block_threshold=0.4)
        
        assert result.allow_trade is False
        assert result.action_recommendation == "BLOCK"
        assert "blocking short" in result.reason.lower()

    @patch('shree.data.stocktwits_sentiment.get_mes_sentiment')
    def test_bearish_sentiment_allows_short(self, mock_sentiment):
        """Bearish sentiment should allow short entry."""
        mock_sentiment.return_value = -0.5
        result = evaluate_sentiment_for_entry("SELL")
        
        assert result.allow_trade is True
        assert result.confidence_modifier >= 1.0  # Should boost confidence
        assert "supports short" in result.reason.lower()

    @patch('shree.data.stocktwits_sentiment.get_mes_sentiment')
    def test_neutral_sentiment_allows_trade(self, mock_sentiment):
        """Neutral sentiment should allow trade with no modifier."""
        mock_sentiment.return_value = 0.1  # Within neutral zone
        result = evaluate_sentiment_for_entry("BUY", weak_threshold=0.2)
        
        assert result.allow_trade is True
        assert result.confidence_modifier == 1.0
        assert "neutral" in result.reason.lower()


class TestEvaluateSentimentForPosition:
    """Test sentiment evaluation for existing positions."""

    @patch('shree.data.stocktwits_sentiment.get_mes_sentiment')
    def test_extremely_bearish_warns_for_long(self, mock_sentiment):
        """Extremely bearish sentiment should warn about long position."""
        mock_sentiment.return_value = -0.7  # Below -0.6 threshold
        result = evaluate_sentiment_for_position("LONG", protect_threshold=0.6)
        
        assert result.action_recommendation == "REDUCE_SIZE"
        assert result.confidence_modifier < 1.0
        assert "tightening stop" in result.reason.lower() or "exiting" in result.reason.lower()

    @patch('shree.data.stocktwits_sentiment.get_mes_sentiment')
    def test_extremely_bullish_warns_for_short(self, mock_sentiment):
        """Extremely bullish sentiment should warn about short position."""
        mock_sentiment.return_value = 0.7  # Above +0.6 threshold
        result = evaluate_sentiment_for_position("SHORT", protect_threshold=0.6)
        
        assert result.action_recommendation == "REDUCE_SIZE"
        assert result.confidence_modifier < 1.0

    @patch('shree.data.stocktwits_sentiment.get_mes_sentiment')
    def test_neutral_sentiment_ok_for_position(self, mock_sentiment):
        """Neutral sentiment should be OK for existing position."""
        mock_sentiment.return_value = 0.1
        result = evaluate_sentiment_for_position("LONG")
        
        assert result.action_recommendation == "PROCEED"
        assert result.confidence_modifier == 1.0


class TestMesSentiment:
    """Test combined MES sentiment calculation."""

    @patch('shree.data.stocktwits_sentiment._fetch_stocktwits_symbol')
    def test_combines_es_and_spy_sentiment(self, mock_fetch):
        """MES sentiment should average ES_F and SPY scores."""
        reset_sentiment_cache()
        
        # Mock different scores for each symbol
        def mock_fetch_impl(symbol):
            if symbol == "ES_F":
                return SentimentResult(symbol="ES_F", score=0.6)
            else:
                return SentimentResult(symbol="SPY", score=0.4)
        
        mock_fetch.side_effect = mock_fetch_impl
        
        result = get_mes_sentiment(force_refresh=True)
        # (0.6 + 0.4) / 2 = 0.5
        assert result == 0.5

    @patch('shree.data.stocktwits_sentiment._fetch_stocktwits_symbol')
    def test_handles_missing_symbol_as_zero(self, mock_fetch):
        """Missing symbol should be treated as 0.0."""
        reset_sentiment_cache()
        
        # Only ES_F returns data
        def mock_fetch_impl(symbol):
            if symbol == "ES_F":
                return SentimentResult(symbol="ES_F", score=0.8)
            else:
                return SentimentResult(symbol="SPY", score=0.0, error="timeout")
        
        mock_fetch.side_effect = mock_fetch_impl
        
        result = get_mes_sentiment(force_refresh=True)
        # (0.8 + 0.0) / 2 = 0.4
        assert result == 0.4


class TestCacheIntegration:
    """Test cache behavior with the main functions."""

    @patch('shree.data.stocktwits_sentiment._fetch_stocktwits_symbol')
    def test_uses_cache_when_fresh(self, mock_fetch):
        """Should use cached data when fresh."""
        reset_sentiment_cache()
        set_cache_refresh_interval(300)  # 5 minutes
        
        mock_fetch.return_value = SentimentResult(symbol="ES_F", score=0.5)
        
        # First call should hit the API
        get_stocktwits_sentiment(["ES_F"], force_refresh=True)
        assert mock_fetch.call_count == 1
        
        # Second call should use cache
        get_stocktwits_sentiment(["ES_F"], force_refresh=False)
        assert mock_fetch.call_count == 1  # No additional calls

    @patch('shree.data.stocktwits_sentiment._fetch_stocktwits_symbol')
    def test_refreshes_when_stale(self, mock_fetch):
        """Should refresh data when cache is stale."""
        reset_sentiment_cache()
        set_cache_refresh_interval(0.1)  # Very short for testing
        
        mock_fetch.return_value = SentimentResult(symbol="ES_F", score=0.5)
        
        # First call
        get_stocktwits_sentiment(["ES_F"], force_refresh=True)
        assert mock_fetch.call_count == 1
        
        # Wait for cache to expire
        time.sleep(0.15)
        
        # Should refresh
        get_stocktwits_sentiment(["ES_F"], force_refresh=False)
        assert mock_fetch.call_count == 2  # Additional call made


class TestSentimentResult:
    """Test SentimentResult dataclass."""

    def test_to_dict_serialization(self):
        """SentimentResult should serialize to dict correctly."""
        result = SentimentResult(
            symbol="ES_F",
            score=0.5,
            bullish_count=10,
            bearish_count=5,
            neutral_count=3,
            total_messages=18,
        )
        
        d = result.to_dict()
        assert d["symbol"] == "ES_F"
        assert d["score"] == 0.5
        assert d["bullish_count"] == 10
        assert d["bearish_count"] == 5
        assert d["neutral_count"] == 3
        assert d["total_messages"] == 18
        assert "timestamp" in d
        assert d["error"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
