"""AWS Bedrock client for LLM integration."""
from __future__ import annotations

import json
import time
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
    """Client for AWS Bedrock LLM API."""
    
    def __init__(
        self,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        region_name: str = "us-east-1",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ):
        """Initialize Bedrock client.
        
        Args:
            model_id: Bedrock model identifier (Claude 3 or Titan)
            region_name: AWS region
            max_tokens: Maximum tokens in response
            temperature: Model temperature (0.0-1.0, lower = more deterministic)
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
        
        try:
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=region_name
            )
            logger.info(f"Initialized Bedrock client with model: {model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise
    
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

Respond ONLY with valid JSON in this exact format:
{{
    "trade_decision": "BUY" or "SELL" or "HOLD",
    "confidence": 0.0 to 1.0,
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
    
    def _invoke_claude(self, prompt: str) -> str:
        """Invoke Claude model."""
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
    
    def _invoke_titan(self, prompt: str) -> str:
        """Invoke Titan model."""
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
                f"(confidence: {recommendation.confidence:.2f}) - {recommendation.reasoning}"
            )
            
            return recommendation
            
        except (BotoCoreError, ClientError) as e:
            logger.error(f"AWS Bedrock API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting LLM recommendation: {e}")
            return None
