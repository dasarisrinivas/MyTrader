"""Hybrid Bedrock Client for event-driven LLM analysis.

This module provides:
- Event-driven Bedrock calls (not on every tick)
- LRU caching with context hash
- Retry with exponential backoff
- SQLite logging of all calls
- Bias modifier output (does NOT override risk rules)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from ..utils.logger import logger
from .sqlite_manager import get_sqlite_manager, BedrockSQLiteManager


# Cost estimates per 1K tokens (approximate, varies by model)
MODEL_COSTS = {
    "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 0.003, "output": 0.015},
    "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {"input": 0.003, "output": 0.015},
    "amazon.titan-text-express-v1": {"input": 0.0002, "output": 0.0006},
    "amazon.titan-text-lite-v1": {"input": 0.00015, "output": 0.0002},
}

# Default costs if model not in list
DEFAULT_COST = {"input": 0.003, "output": 0.015}


class HybridBedrockClient:
    """Event-driven Bedrock client with caching and logging.
    
    This client is designed for the hybrid architecture where:
    - Local engine handles fast, real-time execution
    - Bedrock is called only on specific events (not every tick)
    - Results are used as bias modifiers, not trade overrides
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        region_name: Optional[str] = None,
        max_tokens: int = 300,
        temperature: float = 0.3,
        max_retries: int = 3,
        cache_size: int = 100,
        cache_ttl_seconds: int = 300,
        db_path: str = "data/bedrock_hybrid.db",
        daily_quota: int = 1000,
        daily_cost_limit: float = 50.0,
    ):
        """Initialize Hybrid Bedrock client.
        
        Args:
            model_id: Bedrock model ID (or set AWS_BEDROCK_MODEL env var)
            region_name: AWS region (or set AWS_REGION env var)
            max_tokens: Maximum tokens in response
            temperature: Model temperature (0.0-1.0)
            max_retries: Maximum retry attempts
            cache_size: LRU cache size
            cache_ttl_seconds: Cache TTL in seconds
            db_path: Path to SQLite database
            daily_quota: Maximum API calls per day
            daily_cost_limit: Maximum cost in USD per day
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for Bedrock integration. "
                "Install with: pip install boto3"
            )
        
        # Configuration from environment or parameters
        self.model_id = model_id or os.environ.get(
            "AWS_BEDROCK_MODEL", 
            "anthropic.claude-3-sonnet-20240229-v1:0"
        )
        self.region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache_ttl_seconds = cache_ttl_seconds
        
        # Initialize Bedrock client
        try:
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.region_name
            )
            logger.info(f"Initialized HybridBedrockClient with model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise
        
        # Initialize SQLite manager for logging
        self.db_manager = get_sqlite_manager(
            db_path=db_path,
            daily_quota=daily_quota,
            daily_cost_limit=daily_cost_limit,
        )
        
        # In-memory cache for recent context hashes
        self._cache: Dict[str, Tuple[Dict, float]] = {}  # {hash: (result, timestamp)}
        self._cache_size = cache_size
        
        # Track API call count for rate limiting
        self._call_count = 0
        self._last_call_time: Optional[float] = None
        
        logger.info(f"HybridBedrockClient ready (region={self.region_name}, model={self.model_id})")
    
    def _compute_context_hash(self, context: str) -> str:
        """Compute hash of context for caching.
        
        Args:
            context: Context string
            
        Returns:
            SHA256 hash (first 16 chars)
        """
        return hashlib.sha256(context.encode()).hexdigest()[:16]
    
    def _check_cache(self, context_hash: str) -> Optional[Dict]:
        """Check if context hash is in cache and not expired.
        
        Args:
            context_hash: Hash of context
            
        Returns:
            Cached result or None
        """
        if context_hash in self._cache:
            result, timestamp = self._cache[context_hash]
            if time.time() - timestamp < self.cache_ttl_seconds:
                logger.debug(f"Cache hit for context hash: {context_hash}")
                return result
            else:
                # Expired, remove from cache
                del self._cache[context_hash]
        
        # Also check SQLite for persistent cache
        db_result = self.db_manager.get_call_by_context_hash(context_hash)
        if db_result:
            # Check if not too old
            ts = datetime.fromisoformat(db_result["ts"].replace("Z", "+00:00"))
            age_seconds = (datetime.now(timezone.utc) - ts).total_seconds()
            
            if age_seconds < self.cache_ttl_seconds:
                logger.debug(f"DB cache hit for context hash: {context_hash}")
                try:
                    return json.loads(db_result["response"]) if db_result["response"] else None
                except json.JSONDecodeError:
                    pass
        
        return None
    
    def _update_cache(self, context_hash: str, result: Dict) -> None:
        """Update cache with new result.
        
        Args:
            context_hash: Hash of context
            result: Result to cache
        """
        # Evict oldest entries if cache is full
        if len(self._cache) >= self._cache_size:
            oldest_hash = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_hash]
        
        self._cache[context_hash] = (result, time.time())
    
    def _estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        """Estimate cost for API call.
        
        Args:
            tokens_in: Input tokens
            tokens_out: Output tokens
            
        Returns:
            Estimated cost in USD
        """
        costs = MODEL_COSTS.get(self.model_id, DEFAULT_COST)
        return (tokens_in / 1000 * costs["input"]) + (tokens_out / 1000 * costs["output"])
    
    def _invoke_model_with_retry(
        self,
        prompt: str,
        trigger: str,
        context_hash: str,
    ) -> Tuple[Dict, int, int, float]:
        """Invoke Bedrock model with retry logic.
        
        Args:
            prompt: The prompt to send
            trigger: What triggered this call
            context_hash: Hash of context for logging
            
        Returns:
            Tuple of (parsed_response, tokens_in, tokens_out, latency_ms)
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            start_time = time.time()
            
            try:
                # Build request body based on model type
                if "claude" in self.model_id.lower():
                    body = json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ]
                    })
                elif "titan" in self.model_id.lower():
                    body = json.dumps({
                        "inputText": prompt,
                        "textGenerationConfig": {
                            "maxTokenCount": self.max_tokens,
                            "temperature": self.temperature,
                            "topP": 0.9,
                        }
                    })
                else:
                    raise ValueError(f"Unsupported model: {self.model_id}")
                
                # Make API call
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=body
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Parse response based on model type
                response_body = json.loads(response["body"].read())
                
                if "claude" in self.model_id.lower():
                    response_text = response_body.get("content", [{}])[0].get("text", "")
                    tokens_in = response_body.get("usage", {}).get("input_tokens", 0)
                    tokens_out = response_body.get("usage", {}).get("output_tokens", 0)
                else:
                    response_text = response_body.get("results", [{}])[0].get("outputText", "")
                    # Titan doesn't return token counts, estimate
                    tokens_in = len(prompt.split()) * 1.3  # rough estimate
                    tokens_out = len(response_text.split()) * 1.3
                
                # Parse JSON from response
                parsed = self._parse_json_response(response_text)
                
                self._call_count += 1
                self._last_call_time = time.time()
                
                return parsed, int(tokens_in), int(tokens_out), latency_ms
                
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                last_error = e
                
                if error_code in ("ThrottlingException", "ServiceUnavailableException"):
                    wait_time = (2 ** attempt) + (attempt * 0.5)  # Exponential backoff
                    logger.warning(
                        f"Bedrock {error_code}, retrying in {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # Log error and re-raise
                    self.db_manager.log_bedrock_call(
                        trigger=trigger,
                        prompt=prompt,
                        response=None,
                        model=self.model_id,
                        status="error",
                        error_message=str(e),
                        context_hash=context_hash,
                    )
                    raise
                    
            except Exception as e:
                last_error = e
                logger.error(f"Error invoking Bedrock model: {e}")
                
                # Log error
                self.db_manager.log_bedrock_call(
                    trigger=trigger,
                    prompt=prompt,
                    response=None,
                    model=self.model_id,
                    status="error",
                    error_message=str(e),
                    context_hash=context_hash,
                )
                raise
        
        # All retries exhausted
        error_msg = f"All {self.max_retries} retries failed: {last_error}"
        logger.error(error_msg)
        
        self.db_manager.log_bedrock_call(
            trigger=trigger,
            prompt=prompt,
            response=None,
            model=self.model_id,
            status="error",
            error_message=error_msg,
            context_hash=context_hash,
        )
        
        raise RuntimeError(error_msg)
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """Parse JSON from model response.
        
        Args:
            response_text: Raw response text
            
        Returns:
            Parsed JSON dict (or empty dict on failure)
        """
        try:
            # Find JSON in response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                return json.loads(json_text)
            else:
                logger.warning("No JSON found in Bedrock response")
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Bedrock response: {e}")
            return {}
    
    def bedrock_analyze(
        self,
        context: str,
        trigger: str = "manual",
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Analyze market context using Bedrock.
        
        This is the main entry point for event-driven analysis.
        
        Args:
            context: Structured context string from RAG context builder
            trigger: What triggered this call (for logging)
            max_tokens: Override default max tokens
            
        Returns:
            Dict with: bias, confidence, action, rationale
        """
        # Check quota first
        within_quota, quota_msg = self.db_manager.check_quota()
        if not within_quota:
            logger.warning(f"Bedrock quota exceeded: {quota_msg}")
            return {
                "bias": "NEUTRAL",
                "confidence": 0.0,
                "action": "HOLD",
                "rationale": f"Quota exceeded: {quota_msg}",
                "error": "quota_exceeded",
            }
        
        # Compute context hash for caching
        context_hash = self._compute_context_hash(context)
        
        # Check cache first
        cached_result = self._check_cache(context_hash)
        if cached_result:
            logger.info(f"Using cached Bedrock result for trigger: {trigger}")
            return cached_result
        
        # Build prompt
        prompt = self._build_analysis_prompt(context)
        
        # Make API call with retry
        try:
            tokens_used = max_tokens or self.max_tokens
            parsed, tokens_in, tokens_out, latency_ms = self._invoke_model_with_retry(
                prompt=prompt,
                trigger=trigger,
                context_hash=context_hash,
            )
            
            # Extract bias result
            bias = parsed.get("bias", "NEUTRAL")
            confidence = float(parsed.get("confidence", 0.0))
            action = parsed.get("action", "HOLD")
            rationale = parsed.get("rationale", "")
            
            # Estimate cost
            cost_estimate = self._estimate_cost(tokens_in, tokens_out)
            
            # Log successful call
            self.db_manager.log_bedrock_call(
                trigger=trigger,
                prompt=prompt,
                response=json.dumps(parsed),
                model=self.model_id,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_estimate=cost_estimate,
                latency_ms=latency_ms,
                status="success",
                context_hash=context_hash,
                bias_result=bias,
                confidence=confidence,
            )
            
            # Build result
            result = {
                "bias": bias,
                "confidence": confidence,
                "action": action,
                "rationale": rationale,
                "key_factors": parsed.get("key_factors", []),
                "risk_notes": parsed.get("risk_notes", ""),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_estimate": cost_estimate,
                "latency_ms": latency_ms,
            }
            
            # Update cache
            self._update_cache(context_hash, result)
            
            logger.info(
                f"Bedrock analysis complete: bias={bias}, confidence={confidence:.2f}, "
                f"action={action}, latency={latency_ms:.0f}ms, cost=${cost_estimate:.4f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Bedrock analysis failed: {e}")
            return {
                "bias": "NEUTRAL",
                "confidence": 0.0,
                "action": "HOLD",
                "rationale": f"Analysis failed: {str(e)}",
                "error": str(e),
            }
    
    def _build_analysis_prompt(self, context: str) -> str:
        """Build the analysis prompt for Bedrock.
        
        Args:
            context: Structured context from RAG context builder
            
        Returns:
            Full prompt string
        """
        prompt = f"""You are an expert futures trading analyst. Analyze the following market context and provide a directional bias assessment.

{context}

IMPORTANT CONSTRAINTS:
- Your output is a BIAS MODIFIER only
- It does NOT override stop-loss, max-loss, or risk rules
- Keep confidence low (0.3-0.5) unless signals are very clear
- Be conservative - "NEUTRAL" is acceptable

Respond ONLY with valid JSON in this exact format:
{{
    "bias": "BULLISH" or "BEARISH" or "NEUTRAL",
    "confidence": 0.0 to 1.0,
    "action": "BUY" or "SELL" or "HOLD",
    "rationale": "Brief 1-2 sentence explanation",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risk_notes": "Any risk concerns"
}}

JSON Response:"""
        return prompt
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status and statistics.
        
        Returns:
            Dict with status information
        """
        daily_stats = self.db_manager.get_daily_stats()
        within_quota, quota_msg = self.db_manager.check_quota()
        
        return {
            "model_id": self.model_id,
            "region": self.region_name,
            "daily_calls": daily_stats.get("call_count", 0),
            "daily_cost": daily_stats.get("total_cost", 0.0),
            "quota_status": quota_msg,
            "within_quota": within_quota,
            "cache_size": len(self._cache),
            "total_calls": self._call_count,
            "last_call_time": self._last_call_time,
        }
    
    def get_recent_calls(self, limit: int = 10) -> list:
        """Get recent Bedrock calls.
        
        Args:
            limit: Maximum number of calls to return
            
        Returns:
            List of recent calls
        """
        return self.db_manager.get_recent_bedrock_calls(limit=limit)


def init_bedrock_client(
    model_id: Optional[str] = None,
    region_name: Optional[str] = None,
    **kwargs
) -> HybridBedrockClient:
    """Initialize Bedrock client from environment or parameters.
    
    Reads configuration from:
    - AWS_REGION environment variable
    - AWS_BEDROCK_MODEL environment variable
    - Function parameters (override env vars)
    
    Args:
        model_id: Bedrock model ID
        region_name: AWS region
        **kwargs: Additional arguments for HybridBedrockClient
        
    Returns:
        Initialized HybridBedrockClient
    """
    return HybridBedrockClient(
        model_id=model_id,
        region_name=region_name,
        **kwargs
    )
