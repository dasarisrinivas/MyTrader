"""Sentiment aggregator with AWS Comprehend integration."""
from __future__ import annotations

from typing import Dict, List, Optional

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from ..utils.logger import logger


class SentimentAggregator:
    """Aggregate sentiment from multiple sources using AWS Comprehend."""
    
    def __init__(
        self,
        region_name: str = "us-east-1",
        enable_comprehend: bool = True,
    ):
        """Initialize sentiment aggregator.
        
        Args:
            region_name: AWS region for Comprehend
            enable_comprehend: Enable/disable AWS Comprehend
        """
        self.enable_comprehend = enable_comprehend
        self.comprehend_client = None
        
        if enable_comprehend:
            if not BOTO3_AVAILABLE:
                logger.warning(
                    "boto3 not available, AWS Comprehend disabled. "
                    "Install with: pip install boto3"
                )
                self.enable_comprehend = False
            else:
                try:
                    self.comprehend_client = boto3.client(
                        service_name="comprehend",
                        region_name=region_name
                    )
                    logger.info(f"Initialized AWS Comprehend client in {region_name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize Comprehend client: {e}")
                    self.enable_comprehend = False
    
    def analyze_text_sentiment(self, text: str, language_code: str = "en") -> Dict:
        """Analyze sentiment of text using AWS Comprehend.
        
        Args:
            text: Text to analyze
            language_code: Language code (default: "en")
            
        Returns:
            Dictionary with sentiment scores
        """
        if not self.enable_comprehend or self.comprehend_client is None:
            return {
                "sentiment": "NEUTRAL",
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "mixed": 0.0,
                "normalized_score": 0.0,
            }
        
        try:
            # Truncate text if too long (AWS Comprehend limit)
            if len(text) > 5000:
                text = text[:5000]
            
            response = self.comprehend_client.detect_sentiment(
                Text=text,
                LanguageCode=language_code
            )
            
            sentiment = response.get("Sentiment", "NEUTRAL")
            scores = response.get("SentimentScore", {})
            
            positive = scores.get("Positive", 0.0)
            negative = scores.get("Negative", 0.0)
            neutral = scores.get("Neutral", 0.0)
            mixed = scores.get("Mixed", 0.0)
            
            # Normalize to -1.0 to +1.0 scale
            normalized_score = positive - negative
            
            return {
                "sentiment": sentiment,
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "mixed": mixed,
                "normalized_score": normalized_score,
            }
            
        except (BotoCoreError, ClientError) as e:
            logger.error(f"AWS Comprehend API error: {e}")
            return {
                "sentiment": "NEUTRAL",
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "mixed": 0.0,
                "normalized_score": 0.0,
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                "sentiment": "NEUTRAL",
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "mixed": 0.0,
                "normalized_score": 0.0,
            }
    
    def aggregate_sentiment(
        self,
        news_headlines: Optional[List[str]] = None,
        social_media_posts: Optional[List[str]] = None,
        existing_sentiment: Optional[float] = None,
    ) -> float:
        """Aggregate sentiment from multiple sources.
        
        Args:
            news_headlines: List of news headlines
            social_media_posts: List of social media posts
            existing_sentiment: Existing sentiment score from other sources
            
        Returns:
            Normalized sentiment score (-1.0 to +1.0)
        """
        sentiment_scores = []
        
        # Add existing sentiment if provided
        if existing_sentiment is not None:
            sentiment_scores.append(existing_sentiment)
        
        # Analyze news headlines
        if news_headlines:
            for headline in news_headlines[:10]:  # Limit to 10 headlines
                result = self.analyze_text_sentiment(headline)
                sentiment_scores.append(result["normalized_score"])
        
        # Analyze social media posts
        if social_media_posts:
            for post in social_media_posts[:20]:  # Limit to 20 posts
                result = self.analyze_text_sentiment(post)
                sentiment_scores.append(result["normalized_score"])
        
        # Calculate weighted average
        if not sentiment_scores:
            return 0.0
        
        # Weight recent sources more heavily
        weights = [1.0 / (i + 1) for i in range(len(sentiment_scores))]
        weighted_sum = sum(s * w for s, w in zip(sentiment_scores, weights))
        total_weight = sum(weights)
        
        aggregated = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Clamp to valid range
        aggregated = max(-1.0, min(1.0, aggregated))
        
        logger.info(
            f"Aggregated sentiment: {aggregated:.3f} from {len(sentiment_scores)} sources"
        )
        
        return aggregated
    
    def get_market_sentiment(
        self,
        symbol: str,
        news_api_key: Optional[str] = None,
    ) -> Dict[str, float]:
        """Get market sentiment for a specific symbol.
        
        This is a placeholder for integrating with news APIs.
        In production, you would fetch news from NewsAPI, Finnhub, etc.
        
        Args:
            symbol: Trading symbol
            news_api_key: API key for news service
            
        Returns:
            Dictionary with sentiment breakdown by source
        """
        # Placeholder implementation
        # In production, integrate with actual news APIs
        
        sentiment_by_source = {
            "news": 0.0,
            "social_media": 0.0,
            "overall": 0.0,
        }
        
        # Example: If you have news API integration
        # news_headlines = fetch_news_headlines(symbol, news_api_key)
        # news_sentiment = self.aggregate_sentiment(news_headlines=news_headlines)
        # sentiment_by_source["news"] = news_sentiment
        
        logger.debug(f"Market sentiment for {symbol}: {sentiment_by_source}")
        
        return sentiment_by_source
