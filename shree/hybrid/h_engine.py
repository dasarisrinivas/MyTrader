"""H-Engine: Heuristic LLM + RAG Engine for Trade Confirmation.

Event-triggered engine that provides AI-enhanced trade analysis.
Called ONLY when D-engine produces a candidate signal, not on every tick.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import logger
from .d_engine import DEngineSignal


@dataclass
class RAGContext:
    """Context retrieved from RAG for a trade candidate."""
    
    similar_trades: List[Dict[str, Any]] = field(default_factory=list)
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    total_count: int = 0
    similarity_scores: List[float] = field(default_factory=list)
    
    def avg_similarity(self) -> float:
        """Get average similarity score."""
        if not self.similarity_scores:
            return 0.0
        return sum(self.similarity_scores) / len(self.similarity_scores)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "similar_trades_count": len(self.similar_trades),
            "win_rate": self.win_rate,
            "avg_pnl": self.avg_pnl,
            "total_count": self.total_count,
            "avg_similarity": self.avg_similarity(),
        }


@dataclass
class HEngineAdvisory:
    """Advisory output from the H-engine (LLM + RAG)."""
    
    # LLM recommendation
    recommendation: str  # "LONG", "SHORT", "HOLD"
    model_confidence: float  # 0.0-1.0
    explanation: str
    suggested_position_size_pct: float = 1.0
    
    # RAG context
    rag_context: Optional[RAGContext] = None
    rag_similarity_score: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    llm_model: str = ""
    prompt_hash: str = ""
    latency_ms: float = 0.0
    cached: bool = False
    
    # Original D-engine signal reference
    d_engine_action: str = ""
    d_engine_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation": self.recommendation,
            "model_confidence": self.model_confidence,
            "explanation": self.explanation,
            "suggested_position_size_pct": self.suggested_position_size_pct,
            "rag_context": self.rag_context.to_dict() if self.rag_context else None,
            "rag_similarity_score": self.rag_similarity_score,
            "timestamp": self.timestamp.isoformat(),
            "llm_model": self.llm_model,
            "prompt_hash": self.prompt_hash,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "d_engine_action": self.d_engine_action,
            "d_engine_score": self.d_engine_score,
        }


# Prompt template for trade confirmation (structured JSON response)
# NOTE: Double braces {{ }} are escaped for Python .format()
TRADE_CONFIRMATION_PROMPT = """You are a quantitative trading assistant for ES/MES futures. Analyze the trade candidate and provide a recommendation.

INPUT DATA:
ticker: {ticker}
timeframe: {timeframe}
timestamp: {timestamp}
close: {close}
atr: {atr}

TECHNICAL SCORES:
- RSI: {rsi} (value: {rsi_value})
- MACD: {macd_score} (histogram: {macd_hist})
- EMA trend: {ema_score} (diff %: {ema_diff_pct})
- Volume: {volume_score}
- Overall: {technical_score}

KEY LEVELS:
- Near PDH: {near_pdh}
- Near PDL: {near_pdl}
- Near Weekly High: {near_weekly_high}
- Near Weekly Low: {near_weekly_low}

PROPOSED TRADE:
- Action: {proposed_action}
- Entry: {entry_price}
- Stop Loss: {stop_loss}
- Take Profit: {take_profit}

SIMILAR HISTORICAL TRADES (from RAG):
{rag_summary}

TASK:
Based on the technical analysis and similar historical trades, evaluate this trade candidate.
Return ONLY valid JSON (no markdown, no code blocks):

{{"recommendation": "LONG" or "SHORT" or "HOLD", "model_confidence": <0.0-1.0>, "explanation": "<brief explanation max 50 words>", "suggested_position_size_pct": <0.0-1.0>}}

GUIDELINES:
- If the proposed action aligns with technicals AND historical success rate > 50%, confirm with higher confidence
- If near resistance (PDH/weekly high) for LONG or near support (PDL/weekly low) for SHORT, reduce confidence
- If historical trades in similar conditions had losses, recommend HOLD or reduce position size
- Be conservative - when uncertain, recommend HOLD
- Position size 1.0 = full position, 0.5 = half, 0.0 = no trade"""


class HeuristicEngine:
    """Event-driven LLM + RAG engine for trade confirmation.
    
    This engine:
    - Is called ONLY when D-engine produces a candidate
    - Retrieves similar historical trades from RAG
    - Calls LLM for confirmation/rejection
    - Caches results to avoid repeated calls
    - Respects rate limits and cost controls
    """
    
    def __init__(
        self,
        llm_client: Any = None,  # HybridBedrockClient
        rag_storage: Any = None,  # RAGStorage
        rag_engine: Any = None,   # RAGEngine for embeddings
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize H-engine.
        
        Args:
            llm_client: LLM client for Bedrock calls
            rag_storage: RAG storage for trade history
            rag_engine: RAG engine for embedding-based retrieval
            config: Configuration overrides
        """
        self.llm_client = llm_client
        self.rag_storage = rag_storage
        self.rag_engine = rag_engine
        
        # Default configuration
        self.config = {
            "ticker": "MES",
            "timeframe": "1m",
            "top_k": 5,
            "cache_ttl_seconds": 300,
            "min_interval_seconds": 60,
            "max_calls_per_hour": 10,
            **(config or {})
        }
        
        # Call tracking for rate limiting
        self._call_count = 0
        self._call_timestamps: List[float] = []
        self._last_call_time: Optional[float] = None
        
        # Cache: {context_hash: (advisory, timestamp)}
        self._cache: Dict[str, Tuple[HEngineAdvisory, float]] = {}
        
        logger.info(f"HeuristicEngine initialized with config: {self.config}")
    
    def should_call(self) -> Tuple[bool, str]:
        """Check if we should make an LLM call based on rate limits.
        
        Returns:
            Tuple of (can_call, reason)
        """
        now = time.time()
        
        # Check minimum interval
        if self._last_call_time:
            elapsed = now - self._last_call_time
            if elapsed < self.config["min_interval_seconds"]:
                return False, f"Interval: {self.config['min_interval_seconds'] - elapsed:.0f}s remaining"
        
        # Check hourly limit
        hour_ago = now - 3600
        recent_calls = [t for t in self._call_timestamps if t > hour_ago]
        if len(recent_calls) >= self.config["max_calls_per_hour"]:
            return False, f"Hourly limit reached: {len(recent_calls)}/{self.config['max_calls_per_hour']}"
        
        return True, "OK"
    
    def _generate_context_hash(self, d_signal: DEngineSignal) -> str:
        """Generate a hash for caching based on signal context."""
        context = {
            "action": d_signal.action,
            "price": round(d_signal.entry_price, 1),
            "score": round(d_signal.technical_score, 2),
            "candle_time": d_signal.candle_close_time.isoformat() if d_signal.candle_close_time else "",
        }
        return hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()[:16]
    
    def _check_cache(self, context_hash: str) -> Optional[HEngineAdvisory]:
        """Check if we have a cached result for this context."""
        if context_hash not in self._cache:
            return None
        
        advisory, cache_time = self._cache[context_hash]
        
        # Check TTL
        if time.time() - cache_time > self.config["cache_ttl_seconds"]:
            del self._cache[context_hash]
            return None
        
        advisory.cached = True
        return advisory
    
    def _retrieve_rag_context(self, d_signal: DEngineSignal) -> RAGContext:
        """Retrieve similar historical trades from RAG.
        
        Args:
            d_signal: D-engine signal with market context
            
        Returns:
            RAGContext with similar trades
        """
        rag_context = RAGContext()
        
        if not self.rag_storage:
            logger.debug("RAG storage not available")
            return rag_context
        
        try:
            # Build bucket query based on signal characteristics
            buckets = {
                "signal_type": d_signal.action,
                # Could add volatility bucket, time of day, etc.
            }
            
            # Get stats for this bucket
            stats = self.rag_storage.get_bucket_stats(buckets)
            if stats:
                rag_context.win_rate = stats.get("win_rate", 0.0)
                rag_context.avg_pnl = stats.get("avg_pnl", 0.0)
                rag_context.total_count = stats.get("count", 0)
            
            # Get similar trades
            similar = self.rag_storage.retrieve_similar_trades(
                buckets, 
                limit=self.config["top_k"]
            )
            if similar:
                rag_context.similar_trades = similar
                # Assume uniform similarity for bucket-based retrieval
                rag_context.similarity_scores = [0.7] * len(similar)
            
            logger.debug(f"RAG retrieved {len(rag_context.similar_trades)} similar trades")
            
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
        
        return rag_context
    
    def _build_prompt(self, d_signal: DEngineSignal, rag_context: RAGContext) -> str:
        """Build the LLM prompt from signal and RAG context."""
        # Format RAG summary
        if rag_context.similar_trades:
            rag_summary_parts = []
            for i, trade in enumerate(rag_context.similar_trades[:5], 1):
                outcome = "WIN" if trade.get("pnl", 0) > 0 else "LOSS"
                pnl = trade.get("pnl", 0)
                rag_summary_parts.append(f"{i}. {trade.get('signal_type', 'N/A')} → {outcome} (P&L: ${pnl:.2f})")
            rag_summary = "\n".join(rag_summary_parts)
            rag_summary += f"\nOverall: {rag_context.total_count} trades, {rag_context.win_rate:.1%} win rate, avg P&L: ${rag_context.avg_pnl:.2f}"
        else:
            rag_summary = "No similar historical trades found."
        
        # Format RAG contexts for JSON
        rag_contexts_json = json.dumps([
            {
                "signal": t.get("signal_type"),
                "pnl": t.get("pnl"),
                "outcome": "win" if t.get("pnl", 0) > 0 else "loss"
            }
            for t in rag_context.similar_trades[:3]
        ])
        
        prompt = TRADE_CONFIRMATION_PROMPT.format(
            ticker=self.config["ticker"],
            timeframe=self.config["timeframe"],
            timestamp=d_signal.candle_close_time.isoformat() if d_signal.candle_close_time else "",
            close=d_signal.entry_price,
            atr=d_signal.metadata.get("atr", 0),
            rsi=d_signal.indicator_scores.get("rsi", 0.5),
            rsi_value=d_signal.metadata.get("rsi", 50),
            macd_score=d_signal.indicator_scores.get("macd", 0.5),
            macd_hist=d_signal.metadata.get("macd_hist", 0),
            ema_score=d_signal.indicator_scores.get("ema", 0.5),
            ema_diff_pct=d_signal.ema_diff_pct,
            volume_score=d_signal.indicator_scores.get("volume", 0.5),
            technical_score=d_signal.technical_score,
            near_pdh=str(d_signal.near_pdh).lower(),
            near_pdl=str(d_signal.near_pdl).lower(),
            near_weekly_high=str(d_signal.near_weekly_high).lower(),
            near_weekly_low=str(d_signal.near_weekly_low).lower(),
            proposed_action=d_signal.action,
            entry_price=d_signal.entry_price,
            stop_loss=d_signal.stop_loss,
            take_profit=d_signal.take_profit,
            rag_summary=rag_summary,
        )
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response JSON."""
        try:
            # Try to extract JSON from response
            response = response.strip()
            
            # Handle potential markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return {
                "recommendation": "HOLD",
                "model_confidence": 0.0,
                "explanation": "Failed to parse LLM response",
                "suggested_position_size_pct": 0.0,
            }
    
    def evaluate(self, d_signal: DEngineSignal) -> HEngineAdvisory:
        """Evaluate a D-engine candidate signal using LLM + RAG.
        
        Args:
            d_signal: Candidate signal from D-engine
            
        Returns:
            HEngineAdvisory with recommendation and confidence
        """
        start_time = time.time()
        
        # Generate context hash for caching
        context_hash = self._generate_context_hash(d_signal)
        
        # Check cache first
        cached = self._check_cache(context_hash)
        if cached:
            logger.info(f"H-engine: Using cached result for {context_hash}")
            return cached
        
        # Check rate limits
        can_call, reason = self.should_call()
        if not can_call:
            logger.info(f"H-engine: Rate limited - {reason}")
            # Return conservative advisory
            return HEngineAdvisory(
                recommendation="HOLD",
                model_confidence=0.0,
                explanation=f"Rate limited: {reason}",
                d_engine_action=d_signal.action,
                d_engine_score=d_signal.technical_score,
            )
        
        # Retrieve RAG context
        rag_context = self._retrieve_rag_context(d_signal)
        
        # Build prompt
        prompt = self._build_prompt(d_signal, rag_context)
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:16]
        
        # Call LLM
        advisory = HEngineAdvisory(
            recommendation="HOLD",
            model_confidence=0.0,
            explanation="LLM call failed",
            rag_context=rag_context,
            rag_similarity_score=rag_context.avg_similarity(),
            prompt_hash=prompt_hash,
            d_engine_action=d_signal.action,
            d_engine_score=d_signal.technical_score,
        )
        
        if self.llm_client:
            try:
                logger.info(f"H-engine: Calling LLM for {d_signal.action} candidate")
                
                # Call the LLM
                response = self.llm_client.invoke(
                    prompt=prompt,
                    trigger="trade_confirmation",
                    context_hash=context_hash,
                )
                
                # Parse response
                if response:
                    parsed = self._parse_llm_response(response)
                    advisory.recommendation = parsed.get("recommendation", "HOLD")
                    advisory.model_confidence = float(parsed.get("model_confidence", 0.0))
                    advisory.explanation = parsed.get("explanation", "")
                    advisory.suggested_position_size_pct = float(parsed.get("suggested_position_size_pct", 1.0))
                    advisory.llm_model = getattr(self.llm_client, "model_id", "unknown")
                
                # Track call
                self._call_count += 1
                self._call_timestamps.append(time.time())
                self._last_call_time = time.time()
                
            except Exception as e:
                logger.error(f"H-engine LLM call failed: {e}")
                advisory.explanation = f"LLM error: {str(e)}"
        else:
            logger.warning("H-engine: No LLM client configured")
            advisory.explanation = "LLM client not configured"
        
        # Calculate latency
        advisory.latency_ms = (time.time() - start_time) * 1000
        
        # Cache result
        self._cache[context_hash] = (advisory, time.time())
        
        logger.info(
            f"H-engine: {advisory.recommendation} (conf={advisory.model_confidence:.2f}) "
            f"- {advisory.explanation[:50]}... [{advisory.latency_ms:.0f}ms]"
        )
        
        return advisory
    
    def get_call_stats(self) -> Dict[str, Any]:
        """Get call statistics for monitoring."""
        now = time.time()
        hour_ago = now - 3600
        recent_calls = [t for t in self._call_timestamps if t > hour_ago]
        
        return {
            "total_calls": self._call_count,
            "calls_last_hour": len(recent_calls),
            "max_calls_per_hour": self.config["max_calls_per_hour"],
            "cache_size": len(self._cache),
            "last_call_time": self._last_call_time,
        }
