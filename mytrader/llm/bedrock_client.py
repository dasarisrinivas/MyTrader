"""AWS Bedrock client for LLM integration."""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from ..utils.logger import logger
from .data_schema import TradeRecommendation, TradingContext


class BedrockClient:
    """Client for AWS Bedrock LLM API with rate limiting."""
    
    def __init__(
        self,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        region_name: str = "us-east-1",
        max_tokens: int = 2048,
        temperature: float = 0.3,
        max_requests_per_minute: int = 20,
    ):
        """Initialize Bedrock client.
        
        Args:
            model_id: Bedrock model identifier (Claude 3 or Titan)
            region_name: AWS region
            max_tokens: Maximum tokens in response
            temperature: Model temperature (0.0-1.0, lower = more deterministic)
            max_requests_per_minute: Rate limit for API requests
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for AWS Bedrock integration. "
                "Install with: pip install boto3"
            )
        
        self.model_id = model_id
        self.region_name = region_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_requests_per_minute = max_requests_per_minute
        
        # Rate limiting
        self._request_times = []
        self._min_interval = 60.0 / max_requests_per_minute  # seconds between requests
        
        try:
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=region_name
            )
            logger.info(f"Initialized Bedrock client with model: {model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise
    
    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting by sleeping if necessary."""
        now = datetime.now()
        
        # Remove old request times (older than 1 minute)
        cutoff = now - timedelta(minutes=1)
        self._request_times = [t for t in self._request_times if t > cutoff]
        
        # If at capacity, wait until oldest request expires
        if len(self._request_times) >= self.max_requests_per_minute:
            oldest = self._request_times[0]
            wait_until = oldest + timedelta(minutes=1)
            wait_seconds = (wait_until - now).total_seconds()
            
            if wait_seconds > 0:
                logger.debug(f"Rate limit reached, waiting {wait_seconds:.1f}s")
                time.sleep(wait_seconds)
                # Remove the oldest after waiting
                self._request_times.pop(0)
        
        # Also enforce minimum interval between requests
        if self._request_times:
            last_request = self._request_times[-1]
            elapsed = (now - last_request).total_seconds()
            if elapsed < self._min_interval:
                sleep_time = self._min_interval - elapsed
                logger.debug(f"Enforcing minimum interval, waiting {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        # Record this request
        self._request_times.append(datetime.now())
    
    def _build_prompt(self, context: TradingContext) -> str:
        """Build structured prompt for LLM."""
        prompt = f"""You are an expert trading advisor analyzing market conditions for {context.symbol}.

CURRENT MARKET DATA:
- Price: ${context.current_price:.2f}
- Timestamp: {context.timestamp}

TECHNICAL INDICATORS:
- RSI (14): {context.rsi:.2f} (Oversold <30, Overbought >70)
- MACD: {context.macd:.4f}, Signal: {context.macd_signal:.4f}, Histogram: {context.macd_hist:.4f}
- ATR (14): {context.atr:.2f} (Volatility measure)
- ADX: {context.adx or 0:.2f} (Trend strength, >25 = strong trend)
- Bollinger Band %: {context.bb_percent or 0.5:.2f} (0=lower band, 1=upper band)

SENTIMENT ANALYSIS:
- Overall Sentiment Score: {context.sentiment_score:.2f} (-1.0 = very bearish, +1.0 = very bullish)
- Sources: {context.sentiment_sources or {}}

CURRENT POSITION:
- Position: {context.current_position} contracts
- Unrealized P&L: ${context.unrealized_pnl:.2f}

RISK METRICS:
- Portfolio Heat: {context.portfolio_heat:.2%}
- Daily P&L: ${context.daily_pnl:.2f}
- Win Rate: {context.win_rate:.2%}

MARKET REGIME:
- Market Regime: {context.market_regime or 'Unknown'}
- Volatility Regime: {context.volatility_regime or 'Normal'}

TASK:
Analyze the above data and provide a trading recommendation. Consider:
1. Technical indicator alignment and strength
2. Sentiment corroboration
3. Risk management (position size, stops)
4. Market regime appropriateness

IMPORTANT: Also extract the overall market sentiment from your analysis.
- sentiment_score should be -1.0 (very bearish) to +1.0 (very bullish) to 0.0 (neutral)
- This should reflect YOUR interpretation of market conditions, not just echo the input sentiment
- Consider all factors: technicals, sentiment data, price action, volatility

Respond ONLY with valid JSON in this exact format:
{{
    "trade_decision": "BUY" or "SELL" or "HOLD",
    "confidence": 0.0 to 1.0,
    "sentiment_score": -1.0 to 1.0,
    "suggested_position_size": integer (1-4),
    "suggested_stop_loss": float or null,
    "suggested_take_profit": float or null,
    "reasoning": "Brief explanation of your decision",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risk_assessment": "Brief risk analysis"
}}

JSON Response:"""
        return prompt
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse LLM response and extract JSON."""
        try:
            # Try to find JSON in response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                return json.loads(json_text)
            else:
                logger.error("No JSON found in response")
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text}")
            return {}
    
    def _invoke_claude(self, prompt: str, retry_count: int = 3) -> str:
        """Invoke Claude model with retry logic.
        
        Args:
            prompt: The prompt to send
            retry_count: Number of retries for transient errors
            
        Returns:
            Response text from model
        """
        last_error = None
        
        for attempt in range(retry_count):
            try:
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
                
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=body
                )
                
                response_body = json.loads(response.get("body").read())
                return response_body.get("content", [{}])[0].get("text", "")
                
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                
                # Retry on throttling or service errors
                if error_code in ("ThrottlingException", "ServiceUnavailableException"):
                    last_error = e
                    wait_time = (2 ** attempt)  # Exponential backoff
                    logger.warning(f"AWS error {error_code}, retrying in {wait_time}s (attempt {attempt + 1}/{retry_count})")
                    time.sleep(wait_time)
                    continue
                else:
                    # Don't retry on other errors
                    raise
            except Exception as e:
                last_error = e
                logger.error(f"Error invoking Claude: {e}")
                raise
        
        # All retries failed
        if last_error:
            logger.error(f"All retry attempts failed: {last_error}")
            raise last_error
        
        return ""
    
    def _invoke_titan(self, prompt: str, retry_count: int = 3) -> str:
        """Invoke Titan model with retry logic.
        
        Args:
            prompt: The prompt to send
            retry_count: Number of retries for transient errors
            
        Returns:
            Response text from model
        """
        last_error = None
        
        for attempt in range(retry_count):
            try:
                body = json.dumps({
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": self.max_tokens,
                        "temperature": self.temperature,
                        "topP": 0.9,
                    }
                })
                
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=body
                )
                
                response_body = json.loads(response.get("body").read())
                return response_body.get("results", [{}])[0].get("outputText", "")
                
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                
                # Retry on throttling or service errors
                if error_code in ("ThrottlingException", "ServiceUnavailableException"):
                    last_error = e
                    wait_time = (2 ** attempt)  # Exponential backoff
                    logger.warning(f"AWS error {error_code}, retrying in {wait_time}s (attempt {attempt + 1}/{retry_count})")
                    time.sleep(wait_time)
                    continue
                else:
                    # Don't retry on other errors
                    raise
            except Exception as e:
                last_error = e
                logger.error(f"Error invoking Titan: {e}")
                raise
        
        # All retries failed
        if last_error:
            logger.error(f"All retry attempts failed: {last_error}")
            raise last_error
        
        return ""
    
    def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Generate text response from LLM (generic method).
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Override default max tokens
            temperature: Override default temperature
            
        Returns:
            Generated text response
        """
        # Save original values
        orig_max_tokens = self.max_tokens
        orig_temperature = self.temperature
        
        try:
            # Override if provided
            if max_tokens is not None:
                self.max_tokens = max_tokens
            if temperature is not None:
                self.temperature = temperature
            
            # Enforce rate limiting
            self._enforce_rate_limit()
            
            # Invoke model based on model ID
            if "claude" in self.model_id.lower():
                response = self._invoke_claude(prompt)
            elif "titan" in self.model_id.lower():
                response = self._invoke_titan(prompt)
            else:
                logger.error(f"Unsupported model: {self.model_id}")
                return ""
            
            return response
            
        finally:
            # Restore original values
            self.max_tokens = orig_max_tokens
            self.temperature = orig_temperature
    
    def get_trade_recommendation(
        self,
        context: TradingContext,
        timeout_seconds: float = 10.0
    ) -> Optional[TradeRecommendation]:
        """Get trade recommendation from LLM.
        
        Args:
            context: Trading context with market data and indicators
            timeout_seconds: Maximum time to wait for response
            
        Returns:
            TradeRecommendation or None if request fails
        """
        start_time = time.time()
        
        try:
            # Enforce rate limiting
            self._enforce_rate_limit()
            
            # Build prompt
            prompt = self._build_prompt(context)
            
            # Invoke model based on model ID
            if "claude" in self.model_id.lower():
                response_text = self._invoke_claude(prompt)
            elif "titan" in self.model_id.lower():
                response_text = self._invoke_titan(prompt)
            else:
                logger.error(f"Unsupported model: {self.model_id}")
                return None
            
            # Parse response
            parsed = self._parse_response(response_text)
            
            if not parsed:
                logger.error("Empty response from LLM")
                return None
            
            # Build recommendation
            processing_time = (time.time() - start_time) * 1000
            
            recommendation = TradeRecommendation(
                trade_decision=parsed.get("trade_decision", "HOLD"),
                confidence=float(parsed.get("confidence", 0.0)),
                sentiment_score=float(parsed.get("sentiment_score", 0.0)),
                suggested_position_size=int(parsed.get("suggested_position_size", 1)),
                suggested_stop_loss=parsed.get("suggested_stop_loss"),
                suggested_take_profit=parsed.get("suggested_take_profit"),
                reasoning=parsed.get("reasoning", ""),
                key_factors=parsed.get("key_factors", []),
                risk_assessment=parsed.get("risk_assessment", ""),
                model_name=self.model_id,
                processing_time_ms=processing_time,
                raw_response=parsed,
            )
            
            logger.info(
                f"LLM Recommendation: {recommendation.trade_decision} "
                f"(confidence: {recommendation.confidence:.2f}, sentiment: {recommendation.sentiment_score:+.2f}) - {recommendation.reasoning}"
            )
            
            return recommendation
            
        except (BotoCoreError, ClientError) as e:
            logger.error(f"AWS Bedrock API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting LLM recommendation: {e}")
            return None
