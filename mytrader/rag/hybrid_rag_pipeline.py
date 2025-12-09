"""Hybrid RAG Pipeline - 3-Layer Decision System (Rules → RAG → LLM).

This is the core trading decision pipeline that combines:
1. LAYER 1: Rule Engine (deterministic filters - always on)
2. LAYER 2: RAG Retrieval (similar trades and docs - only on signal)
3. LAYER 3: LLM Decision (final judgment - only on signal)

The pipeline ensures safe, explainable, and context-aware trading decisions.
Uses CST (Central Standard Time) for all timestamps.
"""
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# Import CST utilities
try:
    from ..utils.timezone_utils import now_cst, today_cst, format_cst, CST
except ImportError:
    from zoneinfo import ZoneInfo
    CST = ZoneInfo("America/Chicago")
    def now_cst():
        return datetime.now(CST)
    def today_cst():
        return datetime.now(CST).strftime("%Y-%m-%d")


class TradeAction(Enum):
    """Possible trade actions from the pipeline."""
    BUY = "BUY"
    SELL = "SELL"
    SCALP_BUY = "SCALP_BUY"      # Lower confidence buy for low-vol
    SCALP_SELL = "SCALP_SELL"    # Lower confidence sell for low-vol
    HOLD = "HOLD"
    BLOCKED = "BLOCKED"


class FilterResult(Enum):
    """Result from a rule filter."""
    PASS = "PASS"
    BLOCK = "BLOCK"
    WARN = "WARN"


@dataclass
class RuleEngineResult:
    """Result from Layer 1: Rule Engine."""
    signal: TradeAction
    score: float  # 0-100 strength of signal
    
    # Filter details
    filters_passed: List[str] = field(default_factory=list)
    filters_blocked: List[str] = field(default_factory=list)
    filters_warned: List[str] = field(default_factory=list)
    
    # Indicator values at decision time
    indicators: Dict[str, float] = field(default_factory=dict)
    
    # Market context
    market_trend: str = ""
    volatility_regime: str = ""
    
    @property
    def should_proceed(self) -> bool:
        """Whether to proceed to Layer 2."""
        return self.signal in [TradeAction.BUY, TradeAction.SELL] and not self.filters_blocked


@dataclass
class RAGRetrievalResult:
    """Result from Layer 2: RAG Retrieval."""
    documents: List[Tuple[str, str, float]] = field(default_factory=list)  # (doc_id, content, score)
    similar_trades: List[Dict[str, Any]] = field(default_factory=list)
    
    # Aggregated insights
    historical_win_rate: float = 0.5
    similar_trade_count: int = 0
    avg_pnl_similar: float = 0.0
    
    # Context summary for LLM
    context_summary: str = ""
    
    @property
    def has_context(self) -> bool:
        """Whether meaningful context was retrieved."""
        return len(self.documents) > 0 or len(self.similar_trades) > 0


@dataclass
class LLMDecisionResult:
    """Result from Layer 3: LLM Decision."""
    action: TradeAction
    confidence: float  # 0-100
    reasoning: str
    
    # Risk parameters suggested by LLM
    suggested_stop_loss: float = 0.0
    suggested_take_profit: float = 0.0
    position_size_factor: float = 1.0  # 0.5 = half size, 1.0 = full, 1.5 = larger
    
    # Raw LLM response for logging
    raw_response: str = ""


@dataclass
class HybridPipelineResult:
    """Combined result from all three layers."""
    # Final decision
    final_action: TradeAction
    final_confidence: float
    final_reasoning: str
    
    # Layer results
    rule_engine: RuleEngineResult
    rag_retrieval: RAGRetrievalResult
    llm_decision: Optional[LLMDecisionResult] = None
    
    # Execution parameters
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size: float = 1.0
    
    # Timing
    timestamp: str = ""
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "final_action": self.final_action.value,
            "final_confidence": self.final_confidence,
            "final_reasoning": self.final_reasoning,
            "rule_engine": {
                "signal": self.rule_engine.signal.value,
                "score": self.rule_engine.score,
                "filters_passed": self.rule_engine.filters_passed,
                "filters_blocked": self.rule_engine.filters_blocked,
                "market_trend": self.rule_engine.market_trend,
                "volatility_regime": self.rule_engine.volatility_regime,
            },
            "rag_retrieval": {
                "documents_count": len(self.rag_retrieval.documents),
                "similar_trades_count": self.rag_retrieval.similar_trade_count,
                "historical_win_rate": self.rag_retrieval.historical_win_rate,
            },
            "llm_decision": {
                "action": self.llm_decision.action.value if self.llm_decision else None,
                "confidence": self.llm_decision.confidence if self.llm_decision else None,
                "reasoning": self.llm_decision.reasoning if self.llm_decision else None,
            },
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "timestamp": self.timestamp,
        }


class RuleEngine:
    """Layer 1: Deterministic Rule Engine.
    
    Applies hard filters and generates trading signals based on:
    - Trend alignment (EMA stack)
    - Key level proximity (PDH/PDL)
    - Volatility filters (ATR)
    - RSI/MACD signals
    - Time-based filters
    - Cooldown checks
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize rule engine with configuration.
        
        Args:
            config: Rule engine configuration
        """
        self.config = config
        
        # Filter thresholds
        self.atr_min = config.get("atr_min", 0.15)  # Lowered from 0.3 for low-vol markets
        self.atr_max = config.get("atr_max", 5.0)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.pdh_proximity_pct = config.get("pdh_proximity_pct", 0.3)
        self.cooldown_minutes = config.get("cooldown_minutes", 15)
        
        # Signal weights
        self.trend_weight = config.get("trend_weight", 30)
        self.momentum_weight = config.get("momentum_weight", 25)
        self.level_weight = config.get("level_weight", 25)
        self.volume_weight = config.get("volume_weight", 20)
        
        self.last_trade_time: Optional[datetime] = None
        
        logger.info("RuleEngine initialized")
    
    def evaluate(self, market_data: Dict[str, Any]) -> RuleEngineResult:
        """Evaluate market data against rules.
        
        Args:
            market_data: Current market data with indicators
            
        Returns:
            RuleEngineResult with signal and filters
        """
        result = RuleEngineResult(
            signal=TradeAction.HOLD,
            score=0.0,
            indicators={},
        )
        
        # Extract indicators
        price = market_data.get("close", market_data.get("price", 0))
        ema_9 = market_data.get("ema_9", price)
        ema_20 = market_data.get("ema_20", price)
        ema_50 = market_data.get("ema_50", price)
        rsi = market_data.get("rsi", 50)
        macd_hist = market_data.get("macd_hist", 0)
        atr = market_data.get("atr", 0)
        pdh = market_data.get("pdh", 0)
        pdl = market_data.get("pdl", 0)
        volume_ratio = market_data.get("volume_ratio", 1.0)
        
        result.indicators = {
            "price": price,
            "ema_9": ema_9,
            "ema_20": ema_20,
            "rsi": rsi,
            "macd_hist": macd_hist,
            "atr": atr,
            "pdh": pdh,
            "pdl": pdl,
        }
        
        # Determine trend - ENHANCED for micro-trends
        open_price = market_data.get("open", price)
        pct_change = (price - open_price) / open_price * 100 if open_price > 0 else 0
        ema_diff_pct = (ema_9 - ema_20) / ema_20 * 100 if ema_20 > 0 else 0
        
        # Strong trend: aligned EMAs
        if price > ema_9 > ema_20:
            result.market_trend = "UPTREND"
        elif price < ema_9 < ema_20:
            result.market_trend = "DOWNTREND"
        # Micro trend: 0.1%+ move with price above/below short EMA
        elif pct_change >= 0.1 and price > ema_9:
            result.market_trend = "MICRO_UP"
        elif pct_change <= -0.1 and price < ema_9:
            result.market_trend = "MICRO_DOWN"
        # Weak trend: price above/below EMA but small move
        elif price > ema_9 and ema_diff_pct > 0.02:
            result.market_trend = "WEAK_UP"
        elif price < ema_9 and ema_diff_pct < -0.02:
            result.market_trend = "WEAK_DOWN"
        else:
            result.market_trend = "CHOP"
        
        # Determine volatility regime
        avg_atr = market_data.get("atr_20_avg", atr)
        atr_ratio = atr / avg_atr if avg_atr > 0 else 1
        if atr_ratio > 1.3:
            result.volatility_regime = "HIGH"
        elif atr_ratio < 0.7:
            result.volatility_regime = "LOW"
        else:
            result.volatility_regime = "MEDIUM"
        
        # ===== HARD FILTERS (blockers) =====
        
        # ATR filter - RELAXED for low-vol days
        atr_min_threshold = self.atr_min if result.volatility_regime != "LOW" else 0.05
        if atr < atr_min_threshold:
            result.filters_warned.append(f"VERY_LOW_ATR ({atr:.2f})")  # Warn but don't block
        elif atr > self.atr_max:
            result.filters_blocked.append(f"ATR_TOO_HIGH ({atr:.2f} > {self.atr_max})")
        else:
            result.filters_passed.append("ATR_OK")
        
        # Cooldown filter
        if self.last_trade_time:
            elapsed = (now_cst() - self.last_trade_time).total_seconds() / 60
            if elapsed < self.cooldown_minutes:
                result.filters_blocked.append(f"COOLDOWN ({elapsed:.1f} < {self.cooldown_minutes} min)")
        else:
            result.filters_passed.append("COOLDOWN_OK")
        
        # Time filter (avoid first/last 15 min of session) - CST hours
        current_time = now_cst()
        hour = current_time.hour
        minute = current_time.minute
        
        # Market hours check (CST - ES futures trade Sun 5PM to Fri 4PM CST)
        if 8 <= hour <= 15:  # Core trading hours 8 AM - 3 PM CST
            result.filters_passed.append("MARKET_HOURS_OK")
        else:
            result.filters_warned.append("OUTSIDE_MARKET_HOURS")
        
        # If any hard filters blocked, return early
        if result.filters_blocked:
            result.signal = TradeAction.BLOCKED
            return result
        
        # ===== SIGNAL GENERATION =====
        
        buy_score = 0
        sell_score = 0
        
        # Determine if scalp mode (low vol)
        is_scalp_mode = result.volatility_regime == "LOW"
        scalp_threshold = 20  # Lower threshold for scalps
        
        # Trend component - ENHANCED with micro-trends
        if result.market_trend == "UPTREND":
            buy_score += self.trend_weight
        elif result.market_trend == "DOWNTREND":
            sell_score += self.trend_weight
        elif result.market_trend == "MICRO_UP":
            buy_score += self.trend_weight * 0.7  # 70% credit for micro
        elif result.market_trend == "MICRO_DOWN":
            sell_score += self.trend_weight * 0.7
        elif result.market_trend == "WEAK_UP":
            buy_score += self.trend_weight * 0.4  # 40% credit for weak
        elif result.market_trend == "WEAK_DOWN":
            sell_score += self.trend_weight * 0.4
        
        # Momentum component (RSI + MACD)
        if rsi < self.rsi_oversold:
            buy_score += self.momentum_weight * 0.6
            result.filters_passed.append("RSI_OVERSOLD")
        elif rsi > self.rsi_overbought:
            sell_score += self.momentum_weight * 0.6
            result.filters_passed.append("RSI_OVERBOUGHT")
        
        if macd_hist > 0:
            buy_score += self.momentum_weight * 0.4
        elif macd_hist < 0:
            sell_score += self.momentum_weight * 0.4
        
        # Level proximity component
        if pdh > 0 and pdl > 0:
            pdh_dist_pct = abs(price - pdh) / price * 100
            pdl_dist_pct = abs(price - pdl) / price * 100
            
            if pdl_dist_pct < self.pdh_proximity_pct:
                # Near PDL - potential bounce buy
                buy_score += self.level_weight * 0.7
                result.filters_passed.append("NEAR_PDL")
            elif pdh_dist_pct < self.pdh_proximity_pct:
                # Near PDH - potential rejection sell
                sell_score += self.level_weight * 0.7
                result.filters_warned.append("NEAR_PDH")
        
        # Volume component
        if volume_ratio > 1.5:
            # High volume increases conviction
            buy_score *= 1.1
            sell_score *= 1.1
            result.filters_passed.append("HIGH_VOLUME")
        
        # Determine final signal - use scalp threshold in low-vol
        normal_threshold = self.config.get("signal_threshold", 40)
        signal_threshold = scalp_threshold if is_scalp_mode else normal_threshold
        
        if buy_score > sell_score and buy_score >= signal_threshold:
            result.signal = TradeAction.BUY if not is_scalp_mode else TradeAction.SCALP_BUY
            result.score = min(buy_score, 100)
        elif sell_score > buy_score and sell_score >= signal_threshold:
            result.signal = TradeAction.SELL if not is_scalp_mode else TradeAction.SCALP_SELL
            result.score = min(sell_score, 100)
        else:
            result.signal = TradeAction.HOLD
            result.score = max(buy_score, sell_score)
        
        return result
    
    def record_trade(self) -> None:
        """Record that a trade was made (for cooldown tracking)."""
        self.last_trade_time = now_cst()


class RAGRetriever:
    """Layer 2: RAG Document Retrieval.
    
    Retrieves relevant context:
    - Similar historical trades
    - Strategy documentation
    - Recent market summaries
    - Mistake notes from similar setups
    """
    
    def __init__(
        self,
        embedding_builder: Optional[Any] = None,
        storage_manager: Optional[Any] = None,
    ):
        """Initialize RAG retriever.
        
        Args:
            embedding_builder: FAISS embedding builder
            storage_manager: RAG storage manager
        """
        self.embedding_builder = embedding_builder
        self.storage_manager = storage_manager
        
        logger.info("RAGRetriever initialized")
    
    def retrieve(
        self,
        rule_result: RuleEngineResult,
        market_data: Dict[str, Any],
        top_k: int = 5,
    ) -> RAGRetrievalResult:
        """Retrieve relevant context for a trading signal.
        
        Args:
            rule_result: Result from rule engine
            market_data: Current market data
            top_k: Number of documents to retrieve
            
        Returns:
            RAGRetrievalResult with context
        """
        result = RAGRetrievalResult()
        
        # Build query from rule result and market context
        query = self._build_query(rule_result, market_data)
        
        # Search for similar documents
        if self.embedding_builder:
            try:
                market_context = {
                    "trend": rule_result.market_trend,
                    "volatility_regime": rule_result.volatility_regime,
                    "near_pdh": "NEAR_PDH" in rule_result.filters_warned,
                    "near_pdl": "NEAR_PDL" in rule_result.filters_passed,
                }
                
                docs = self.embedding_builder.search_with_context(
                    query=query,
                    market_context=market_context,
                    top_k=top_k,
                )
                
                result.documents = [(d[0], d[1], d[2]) for d in docs]
                
            except Exception as e:
                logger.warning(f"RAG document search failed: {e}")
        
        # Get similar historical trades
        if self.storage_manager:
            try:
                similar_trades = self.storage_manager.get_similar_trades(
                    action=rule_result.signal.value if rule_result.signal in [TradeAction.BUY, TradeAction.SELL] else "BUY",
                    market_trend=rule_result.market_trend,
                    volatility_regime=rule_result.volatility_regime,
                    price_near_pdh="NEAR_PDH" in rule_result.filters_warned,
                    price_near_pdl="NEAR_PDL" in rule_result.filters_passed,
                    limit=5,
                )
                
                result.similar_trades = [t.to_dict() for t in similar_trades]
                result.similar_trade_count = len(similar_trades)
                
                # Calculate win rate for similar trades
                if similar_trades:
                    wins = sum(1 for t in similar_trades if t.result == "WIN")
                    result.historical_win_rate = wins / len(similar_trades)
                    result.avg_pnl_similar = sum(t.pnl for t in similar_trades) / len(similar_trades)
                
            except Exception as e:
                logger.warning(f"Similar trade retrieval failed: {e}")
        
        # Generate context summary for LLM
        result.context_summary = self._generate_context_summary(result, rule_result)
        
        return result
    
    def _build_query(
        self,
        rule_result: RuleEngineResult,
        market_data: Dict[str, Any],
    ) -> str:
        """Build a search query from rule result and market data.
        
        Args:
            rule_result: Rule engine result
            market_data: Market data
            
        Returns:
            Query string
        """
        parts = []
        
        # Action type
        parts.append(f"{rule_result.signal.value} signal")
        
        # Market context
        parts.append(f"in {rule_result.market_trend} market")
        parts.append(f"with {rule_result.volatility_regime} volatility")
        
        # Key indicators
        indicators = rule_result.indicators
        parts.append(f"RSI at {indicators.get('rsi', 50):.1f}")
        
        if indicators.get("macd_hist", 0) > 0:
            parts.append("bullish MACD")
        else:
            parts.append("bearish MACD")
        
        # Level proximity
        if "NEAR_PDL" in rule_result.filters_passed:
            parts.append("price near previous day low")
        if "NEAR_PDH" in rule_result.filters_warned:
            parts.append("price near previous day high")
        
        return ", ".join(parts)
    
    def _generate_context_summary(
        self,
        rag_result: RAGRetrievalResult,
        rule_result: RuleEngineResult,
    ) -> str:
        """Generate a context summary for the LLM.
        
        Args:
            rag_result: RAG retrieval result
            rule_result: Rule engine result
            
        Returns:
            Context summary text
        """
        lines = []
        
        lines.append("=== RAG Context Summary ===")
        
        # Similar trades summary
        if rag_result.similar_trades:
            lines.append(f"\nFound {rag_result.similar_trade_count} similar historical trades:")
            lines.append(f"- Historical win rate: {rag_result.historical_win_rate:.0%}")
            lines.append(f"- Average P&L: ${rag_result.avg_pnl_similar:.2f}")
        else:
            lines.append("\nNo similar historical trades found.")
        
        # Document context
        if rag_result.documents:
            lines.append(f"\nRelevant documents ({len(rag_result.documents)}):")
            for doc_id, content, score in rag_result.documents[:3]:
                # Truncate content for summary
                preview = content[:200] + "..." if len(content) > 200 else content
                lines.append(f"- [{score:.2f}] {preview}")
        
        return "\n".join(lines)


class LLMDecisionMaker:
    """Layer 3: LLM Final Decision.
    
    Uses Claude/Bedrock to make final trading decision with:
    - Rule engine signal
    - RAG context
    - Market data
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        config: Dict[str, Any] = None,
    ):
        """Initialize LLM decision maker.
        
        Args:
            llm_client: Bedrock client for LLM calls
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.config = config or {}
        
        self.min_confidence_threshold = self.config.get("min_confidence", 60)
        
        logger.info("LLMDecisionMaker initialized")
    
    def decide(
        self,
        rule_result: RuleEngineResult,
        rag_result: RAGRetrievalResult,
        market_data: Dict[str, Any],
    ) -> LLMDecisionResult:
        """Make final trading decision using LLM.
        
        Args:
            rule_result: Result from rule engine
            rag_result: Result from RAG retrieval
            market_data: Current market data
            
        Returns:
            LLMDecisionResult with action and reasoning
        """
        # Build prompt
        prompt = self._build_prompt(rule_result, rag_result, market_data)
        
        # Call LLM
        if self.llm_client:
            try:
                response = self._call_llm(prompt)
                return self._parse_response(response)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
        
        # Fallback to rule-based decision if LLM unavailable
        return self._fallback_decision(rule_result, rag_result)
    
    def _build_prompt(
        self,
        rule_result: RuleEngineResult,
        rag_result: RAGRetrievalResult,
        market_data: Dict[str, Any],
    ) -> str:
        """Build the LLM prompt.
        
        Args:
            rule_result: Rule engine result
            rag_result: RAG result
            market_data: Market data
            
        Returns:
            Prompt string
        """
        return f"""You are a professional SPY futures trader making a trading decision.

=== RULE ENGINE SIGNAL ===
Signal: {rule_result.signal.value}
Score: {rule_result.score:.1f}/100
Market Trend: {rule_result.market_trend}
Volatility: {rule_result.volatility_regime}
Filters Passed: {', '.join(rule_result.filters_passed) or 'None'}
Filters Warned: {', '.join(rule_result.filters_warned) or 'None'}

=== CURRENT INDICATORS ===
Price: {rule_result.indicators.get('price', 0):.2f}
RSI: {rule_result.indicators.get('rsi', 50):.1f}
MACD Histogram: {rule_result.indicators.get('macd_hist', 0):.4f}
ATR: {rule_result.indicators.get('atr', 0):.2f}
PDH: {rule_result.indicators.get('pdh', 0):.2f}
PDL: {rule_result.indicators.get('pdl', 0):.2f}

{rag_result.context_summary}

=== YOUR TASK ===
Based on the above information, decide whether to:
1. CONFIRM the rule engine signal and take the trade
2. REJECT the signal and stay out
3. MODIFY the signal (change action or adjust confidence)

Respond in this EXACT format:
ACTION: [BUY|SELL|HOLD]
CONFIDENCE: [0-100]
STOP_LOSS_POINTS: [number of points for stop loss]
TAKE_PROFIT_POINTS: [number of points for take profit]
POSITION_SIZE: [0.5|1.0|1.5 - relative position size]
REASONING: [2-3 sentences explaining your decision]
"""
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            LLM response text
        """
        import json
        
        response = self.llm_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            })
        )
        
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
    
    def _parse_response(self, response: str) -> LLMDecisionResult:
        """Parse LLM response into structured result.
        
        Args:
            response: Raw LLM response
            
        Returns:
            LLMDecisionResult
        """
        result = LLMDecisionResult(
            action=TradeAction.HOLD,
            confidence=0,
            reasoning="",
            raw_response=response,
        )
        
        lines = response.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("ACTION:"):
                action_str = line.replace("ACTION:", "").strip().upper()
                if action_str == "BUY":
                    result.action = TradeAction.BUY
                elif action_str == "SELL":
                    result.action = TradeAction.SELL
                else:
                    result.action = TradeAction.HOLD
            
            elif line.startswith("CONFIDENCE:"):
                try:
                    result.confidence = float(line.replace("CONFIDENCE:", "").strip())
                except:
                    result.confidence = 50
            
            elif line.startswith("STOP_LOSS_POINTS:"):
                try:
                    result.suggested_stop_loss = float(line.replace("STOP_LOSS_POINTS:", "").strip())
                except:
                    pass
            
            elif line.startswith("TAKE_PROFIT_POINTS:"):
                try:
                    result.suggested_take_profit = float(line.replace("TAKE_PROFIT_POINTS:", "").strip())
                except:
                    pass
            
            elif line.startswith("POSITION_SIZE:"):
                try:
                    result.position_size_factor = float(line.replace("POSITION_SIZE:", "").strip())
                except:
                    result.position_size_factor = 1.0
            
            elif line.startswith("REASONING:"):
                result.reasoning = line.replace("REASONING:", "").strip()
        
        return result
    
    def _fallback_decision(
        self,
        rule_result: RuleEngineResult,
        rag_result: RAGRetrievalResult,
    ) -> LLMDecisionResult:
        """Fallback decision when LLM is unavailable.
        
        Args:
            rule_result: Rule engine result
            rag_result: RAG result
            
        Returns:
            LLMDecisionResult based on rules only
        """
        # Use rule engine signal with RAG adjustment
        confidence = rule_result.score
        
        # Adjust based on historical win rate
        if rag_result.similar_trade_count > 0:
            if rag_result.historical_win_rate > 0.6:
                confidence *= 1.1
            elif rag_result.historical_win_rate < 0.4:
                confidence *= 0.8
        
        return LLMDecisionResult(
            action=rule_result.signal if rule_result.signal in [TradeAction.BUY, TradeAction.SELL] else TradeAction.HOLD,
            confidence=min(confidence, 100),
            reasoning=f"Rule-based decision: {rule_result.signal.value} with score {rule_result.score:.1f}",
        )


class HybridRAGPipeline:
    """Main pipeline orchestrating all three layers.
    
    Usage:
        pipeline = HybridRAGPipeline(config)
        result = pipeline.process(market_data)
        
        if result.final_action in [TradeAction.BUY, TradeAction.SELL]:
            execute_trade(result)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        llm_client: Optional[Any] = None,
        embedding_builder: Optional[Any] = None,
        storage_manager: Optional[Any] = None,
    ):
        """Initialize the hybrid pipeline.
        
        Args:
            config: Pipeline configuration
            llm_client: Bedrock client for LLM
            embedding_builder: FAISS embedding builder
            storage_manager: RAG storage manager
        """
        self.config = config
        
        # Initialize layers
        self.rule_engine = RuleEngine(config.get("rule_engine", {}))
        self.rag_retriever = RAGRetriever(embedding_builder, storage_manager)
        self.llm_decision = LLMDecisionMaker(llm_client, config.get("llm", {}))
        
        # Pipeline settings
        self.skip_llm_on_low_score = config.get("skip_llm_on_low_score", True)
        self.min_score_for_llm = config.get("min_score_for_llm", 30)
        self.min_confidence_for_trade = config.get("min_confidence_for_trade", 60)
        
        logger.info("HybridRAGPipeline initialized")
    
    def process(self, market_data: Dict[str, Any]) -> HybridPipelineResult:
        """Process market data through all three layers.
        
        Args:
            market_data: Current market data with indicators
            
        Returns:
            HybridPipelineResult with final decision
        """
        import time
        start_time = time.time()
        
        # Layer 1: Rule Engine (always runs)
        rule_result = self.rule_engine.evaluate(market_data)
        logger.debug(f"Layer 1 - Rule Engine: {rule_result.signal.value} ({rule_result.score:.1f})")
        
        # Early exit if blocked or no signal
        if not rule_result.should_proceed:
            return HybridPipelineResult(
                final_action=rule_result.signal,
                final_confidence=0,
                final_reasoning=f"Blocked by filters: {', '.join(rule_result.filters_blocked)}",
                rule_engine=rule_result,
                rag_retrieval=RAGRetrievalResult(),
                timestamp=now_cst().isoformat(),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Layer 2: RAG Retrieval (only on signal)
        rag_result = self.rag_retriever.retrieve(rule_result, market_data)
        logger.debug(f"Layer 2 - RAG: {rag_result.similar_trade_count} similar trades, {len(rag_result.documents)} docs")
        
        # Layer 3: LLM Decision (only on signal with sufficient score)
        llm_result = None
        if not self.skip_llm_on_low_score or rule_result.score >= self.min_score_for_llm:
            llm_result = self.llm_decision.decide(rule_result, rag_result, market_data)
            logger.debug(f"Layer 3 - LLM: {llm_result.action.value} ({llm_result.confidence:.1f}%)")
        
        # Determine final action
        if llm_result:
            final_action = llm_result.action
            final_confidence = llm_result.confidence
            final_reasoning = llm_result.reasoning
            
            # Apply confidence threshold
            if final_confidence < self.min_confidence_for_trade:
                final_action = TradeAction.HOLD
                final_reasoning = f"Confidence too low ({final_confidence:.0f}% < {self.min_confidence_for_trade}%)"
        else:
            # Use rule engine result directly
            final_action = rule_result.signal
            final_confidence = rule_result.score
            final_reasoning = f"Rule-based: {rule_result.signal.value}"
        
        # Calculate stop loss and take profit
        atr = rule_result.indicators.get("atr", 1.0)
        price = rule_result.indicators.get("price", 0)
        
        if llm_result and llm_result.suggested_stop_loss > 0:
            stop_loss = llm_result.suggested_stop_loss
            take_profit = llm_result.suggested_take_profit
        else:
            # Default: 1.5 ATR stop, 2 ATR target
            stop_loss = atr * 1.5
            take_profit = atr * 2.0
        
        result = HybridPipelineResult(
            final_action=final_action,
            final_confidence=final_confidence,
            final_reasoning=final_reasoning,
            rule_engine=rule_result,
            rag_retrieval=rag_result,
            llm_decision=llm_result,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=llm_result.position_size_factor if llm_result else 1.0,
            timestamp=now_cst().isoformat(),
            processing_time_ms=(time.time() - start_time) * 1000,
        )
        
        logger.info(
            f"Pipeline result: {result.final_action.value} "
            f"(conf={result.final_confidence:.0f}%, time={result.processing_time_ms:.0f}ms)"
        )
        
        return result
    
    def record_trade(self) -> None:
        """Record that a trade was executed (for cooldown)."""
        self.rule_engine.record_trade()


def create_hybrid_pipeline(
    config: Dict[str, Any],
    llm_client: Optional[Any] = None,
) -> HybridRAGPipeline:
    """Factory function to create a HybridRAGPipeline.
    
    Args:
        config: Pipeline configuration
        llm_client: Optional Bedrock client
        
    Returns:
        HybridRAGPipeline instance
    """
    # Try to import and initialize RAG components
    embedding_builder = None
    storage_manager = None
    
    try:
        from mytrader.rag.embedding_builder import create_embedding_builder
        embedding_builder = create_embedding_builder()
    except ImportError:
        logger.warning("Embedding builder not available")
    
    try:
        from mytrader.rag.rag_storage_manager import get_rag_storage
        storage_manager = get_rag_storage()
    except ImportError:
        logger.warning("RAG storage manager not available")
    
    return HybridRAGPipeline(
        config=config,
        llm_client=llm_client,
        embedding_builder=embedding_builder,
        storage_manager=storage_manager,
    )
