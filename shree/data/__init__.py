"""Data collectors package."""

# Legacy single-source Stocktwits sentiment
from .stocktwits_sentiment import (
    get_mes_sentiment,
    get_stocktwits_sentiment,
    evaluate_sentiment_for_entry,
    evaluate_sentiment_for_position,
    get_sentiment_summary,
    reset_sentiment_cache,
    set_cache_refresh_interval,
    calculate_sentiment_score_from_messages,
    SentimentResult,
    SentimentDecisionModifier,
)

# Multi-source sentiment aggregator (Stocktwits + Reddit + Twitter)
from .sentiment_aggregator import (
    get_combined_mes_sentiment,
    get_mes_sentiment_score,
    get_stocktwits_sentiment as multi_get_stocktwits,
    get_reddit_sentiment,
    get_twitter_sentiment,
    evaluate_sentiment_for_entry as multi_evaluate_entry,
    evaluate_sentiment_for_position as multi_evaluate_position,
    analyze_text_sentiment,
    compute_sentiment_from_messages,
    set_cache_refresh_interval as multi_set_cache_interval,
    SentimentSource,
    SourceSentiment,
    CombinedSentiment,
    SentimentDecision,
)

__all__ = [
    # Legacy Stocktwits
    "get_mes_sentiment",
    "get_stocktwits_sentiment",
    "evaluate_sentiment_for_entry",
    "evaluate_sentiment_for_position",
    "get_sentiment_summary",
    "reset_sentiment_cache",
    "set_cache_refresh_interval",
    "calculate_sentiment_score_from_messages",
    "SentimentResult",
    "SentimentDecisionModifier",
    # Multi-source sentiment
    "get_combined_mes_sentiment",
    "get_mes_sentiment_score",
    "get_reddit_sentiment",
    "get_twitter_sentiment",
    "multi_evaluate_entry",
    "multi_evaluate_position",
    "analyze_text_sentiment",
    "compute_sentiment_from_messages",
    "multi_set_cache_interval",
    "SentimentSource",
    "SourceSentiment",
    "CombinedSentiment",
    "SentimentDecision",
]