"""Hybrid RAG Pipeline - 3-Layer Decision System (Rules → RAG → LLM).

This is the core trading decision pipeline that combines:
1. LAYER 1: Rule Engine (deterministic filters - always on)
2. LAYER 2: RAG Retrieval (similar trades and docs - only on signal)
3. LAYER 3: LLM Decision (final judgment - only on signal)

The pipeline ensures safe, explainable, and context-aware trading decisions.
Uses CST (Central Standard Time) for all timestamps.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from collections import deque

from loguru import logger

from mytrader.utils.hold_reason import HoldReason
from mytrader.utils.structured_logging import log_structured_event

from .retrieval_strategies import (
    recency_weight_from_timestamp,
    hybrid_trade_score,
)

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
    daily_bias: str = "NEUTRAL"  # BULLISH, BEARISH, or NEUTRAL based on PDH/PDL/EMA50

    @property
    def is_actionable_signal(self) -> bool:
        """Whether the rule engine produced a tradable action."""
        return self.signal in [
            TradeAction.BUY,
            TradeAction.SELL,
            TradeAction.SCALP_BUY,
            TradeAction.SCALP_SELL,
        ]
    
    @property
    def should_proceed(self) -> bool:
        """Whether to proceed to Layer 2."""
        return self.is_actionable_signal and not self.filters_blocked


@dataclass
class RAGRetrievalResult:
    """Result from Layer 2: RAG Retrieval."""
    documents: List[Tuple[str, str, float]] = field(default_factory=list)  # (doc_id, content, score)
    similar_trades: List[Dict[str, Any]] = field(default_factory=list)
    trade_priority_scores: List[float] = field(default_factory=list)
    
    # Aggregated insights
    historical_win_rate: float = 0.5
    similar_trade_count: int = 0
    avg_pnl_similar: float = 0.0
    weighted_win_rate: float = 0.5
    
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
    hold_reason: Optional[HoldReason] = None
    
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
                "weighted_win_rate": self.rag_retrieval.weighted_win_rate,
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
            "hold_reason": self.hold_reason.to_dict() if self.hold_reason else None,
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
        close_price = market_data.get("close", price)
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
            "close": close_price,
            "ema_9": ema_9,
            "ema_20": ema_20,
            "rsi": rsi,
            "macd_hist": macd_hist,
            "atr": atr,
            "pdh": pdh,
            "pdl": pdl,
        }
        
        # Determine trend - ENHANCED for micro-trends
        # Use EMA distance and price position instead of candle open (which is too short-term)
        open_price = market_data.get("open", price)
        candle_pct_change = (price - open_price) / open_price * 100 if open_price > 0 else 0
        ema_diff_pct = (ema_9 - ema_20) / ema_20 * 100 if ema_20 > 0 else 0
        
        # Price distance from EMA_20 (more stable reference)
        price_vs_ema20_pct = (price - ema_20) / ema_20 * 100 if ema_20 > 0 else 0
        price_vs_ema9_pct = (price - ema_9) / ema_9 * 100 if ema_9 > 0 else 0
        
        # Strong trend: aligned EMAs
        if price > ema_9 > ema_20:
            result.market_trend = "UPTREND"
        elif price < ema_9 < ema_20:
            result.market_trend = "DOWNTREND"
        # Micro trend: price 0.05%+ above/below EMA20 with EMA9 curling in same direction
        elif price_vs_ema20_pct >= 0.05 and ema_diff_pct > 0:
            result.market_trend = "MICRO_UP"
        elif price_vs_ema20_pct <= -0.05 and ema_diff_pct < 0:
            result.market_trend = "MICRO_DOWN"
        # Weak trend: price above/below EMA9 with positive/negative EMA slope
        elif price > ema_9 and ema_diff_pct > 0.01:
            result.market_trend = "WEAK_UP"
        elif price < ema_9 and ema_diff_pct < -0.01:
            result.market_trend = "WEAK_DOWN"
        # Range-bound: price oscillating around EMAs
        elif abs(price_vs_ema9_pct) < 0.03 and abs(ema_diff_pct) < 0.02:
            result.market_trend = "RANGE"
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
        
        # ===== DAILY BIAS - Broader market context =====
        # Compare current price to previous day's range and EMA50 for daily bias
        daily_bias = "NEUTRAL"
        pdh_pct = (price - pdh) / pdh * 100 if pdh > 0 else 0
        pdl_pct = (price - pdl) / pdl * 100 if pdl > 0 else 0
        ema50_pct = (price - ema_50) / ema_50 * 100 if ema_50 > 0 else 0
        
        # Strong DOWN day: price below PDL (previous day low) or significantly below EMA50
        if pdl_pct < -0.3 or ema50_pct < -0.5:  
            daily_bias = "BEARISH"
            result.filters_passed.append(f"DAILY_BIAS:BEARISH(pdl={pdl_pct:.2f}%,ema50={ema50_pct:.2f}%)")
        # Strong UP day: price above PDH (previous day high) or significantly above EMA50
        elif pdh_pct > 0.3 or ema50_pct > 0.5:
            daily_bias = "BULLISH"
            result.filters_passed.append(f"DAILY_BIAS:BULLISH(pdh={pdh_pct:.2f}%,ema50={ema50_pct:.2f}%)")
        else:
            result.filters_passed.append(f"DAILY_BIAS:NEUTRAL(pdl={pdl_pct:.2f}%,pdh={pdh_pct:.2f}%)")
        
        result.daily_bias = daily_bias  # Store for later use
        
        # ===== SIGNAL GENERATION =====
        
        buy_score = 0
        sell_score = 0
        score_details = []  # For debugging
        
        # Determine if scalp mode (low vol or range-bound)
        is_scalp_mode = result.volatility_regime == "LOW" or result.market_trend in ["RANGE", "CHOP"]
        scalp_threshold = 15  # LOWERED from 20 to trigger more in quiet markets
        
        # Trend component - ENHANCED with micro-trends
        if result.market_trend == "UPTREND":
            buy_score += self.trend_weight
            score_details.append(f"UPTREND:+{self.trend_weight}")
        elif result.market_trend == "DOWNTREND":
            sell_score += self.trend_weight
            score_details.append(f"DOWNTREND:+{self.trend_weight}")
        elif result.market_trend == "MICRO_UP":
            pts = self.trend_weight * 0.7
            buy_score += pts
            score_details.append(f"MICRO_UP:+{pts:.1f}")
        elif result.market_trend == "MICRO_DOWN":
            pts = self.trend_weight * 0.7
            sell_score += pts
            score_details.append(f"MICRO_DOWN:+{pts:.1f}")
        elif result.market_trend == "WEAK_UP":
            pts = self.trend_weight * 0.4
            buy_score += pts
            score_details.append(f"WEAK_UP:+{pts:.1f}")
        elif result.market_trend == "WEAK_DOWN":
            pts = self.trend_weight * 0.4
            sell_score += pts
            score_details.append(f"WEAK_DOWN:+{pts:.1f}")
        elif result.market_trend in ["RANGE", "CHOP"]:
            # Range/Chop: Use mean reversion - RELAXED thresholds for more signals
            if rsi < 48:  # RELAXED from 45 - slight oversold
                pts = self.trend_weight * 0.4  # INCREASED from 0.3
                buy_score += pts
                score_details.append(f"RANGE_RSI<48:+{pts:.1f}")
            elif rsi > 52:  # RELAXED from 55 - slight overbought
                pts = self.trend_weight * 0.4
                sell_score += pts
                score_details.append(f"RANGE_RSI>52:+{pts:.1f}")
            else:
                score_details.append(f"RANGE_NEUTRAL(RSI={rsi:.1f})")
            result.filters_passed.append("RANGE_REVERSION")
        else:
            score_details.append(f"NO_TREND({result.market_trend})")
        
        # Momentum component (RSI + MACD) - MORE GRANULAR
        # RSI: Award points more progressively
        if rsi < self.rsi_oversold:  # < 40
            pts = self.momentum_weight * 0.6
            buy_score += pts
            result.filters_passed.append("RSI_OVERSOLD")
            score_details.append(f"RSI_OVERSOLD({rsi:.1f}):+{pts:.1f}")
        elif rsi < 45:  # 40-45: somewhat oversold
            pts = self.momentum_weight * 0.3
            buy_score += pts
            score_details.append(f"RSI_LOW({rsi:.1f}):+{pts:.1f}")
        elif rsi > self.rsi_overbought:  # > 60
            pts = self.momentum_weight * 0.6
            sell_score += pts
            result.filters_passed.append("RSI_OVERBOUGHT")
            score_details.append(f"RSI_OVERBOUGHT({rsi:.1f}):+{pts:.1f}")
        elif rsi > 55:  # 55-60: somewhat overbought
            pts = self.momentum_weight * 0.3
            sell_score += pts
            score_details.append(f"RSI_HIGH({rsi:.1f}):+{pts:.1f}")
        else:
            score_details.append(f"RSI_NEUTRAL({rsi:.1f})")
        
        # MACD: Award points more granularly based on magnitude
        if macd_hist > 0.5:  # Strong positive
            pts = self.momentum_weight * 0.5
            buy_score += pts
            score_details.append(f"MACD_STRONG_POS({macd_hist:.2f}):+{pts:.1f}")
        elif macd_hist > 0:  # Weak positive
            pts = self.momentum_weight * 0.25
            buy_score += pts
            score_details.append(f"MACD_POS({macd_hist:.2f}):+{pts:.1f}")
        elif macd_hist < -0.5:  # Strong negative
            pts = self.momentum_weight * 0.5
            sell_score += pts
            score_details.append(f"MACD_STRONG_NEG({macd_hist:.2f}):+{pts:.1f}")
        elif macd_hist < 0:  # Weak negative
            pts = self.momentum_weight * 0.25
            sell_score += pts
            score_details.append(f"MACD_NEG({macd_hist:.2f}):+{pts:.1f}")
        else:
            score_details.append(f"MACD_ZERO({macd_hist:.2f})")
        
        # Level proximity component - RELAXED proximity threshold
        # When near key levels with supporting RSI, this is a HIGH-PROBABILITY mean reversion setup
        level_proximity_pct = 0.3  # RELAXED from 0.15% to 0.3%
        if pdh > 0 and pdl > 0:
            pdh_dist_pct = abs(price - pdh) / price * 100
            pdl_dist_pct = abs(price - pdl) / price * 100
            
            if pdl_dist_pct < level_proximity_pct:
                # Near PDL - potential bounce buy
                # BOOST: Near PDL with oversold RSI = strong mean reversion buy
                if rsi < 50:  # RSI supporting buy at PDL
                    pts = self.level_weight * 1.2  # BOOSTED for confluence
                    score_details.append(f"NEAR_PDL+RSI_SUPPORT({pdl_dist_pct:.2f}%):+{pts:.1f}")
                else:
                    pts = self.level_weight * 0.7
                    score_details.append(f"NEAR_PDL({pdl_dist_pct:.2f}%):+{pts:.1f}")
                buy_score += pts
                result.filters_passed.append("NEAR_PDL")
            elif pdh_dist_pct < level_proximity_pct:
                # Near PDH - potential rejection sell
                # BOOST: Near PDH with overbought RSI = strong mean reversion sell
                if rsi > 50:  # RSI supporting sell at PDH
                    pts = self.level_weight * 1.2  # BOOSTED for confluence
                    score_details.append(f"NEAR_PDH+RSI_SUPPORT({pdh_dist_pct:.2f}%):+{pts:.1f}")
                else:
                    pts = self.level_weight * 0.7
                    score_details.append(f"NEAR_PDH({pdh_dist_pct:.2f}%):+{pts:.1f}")
                sell_score += pts
                result.filters_warned.append("NEAR_PDH")
            else:
                score_details.append(f"NO_LEVEL(PDH:{pdh_dist_pct:.2f}%,PDL:{pdl_dist_pct:.2f}%)")
        else:
            score_details.append("NO_PDH_PDL")
        
        # Volume component
        if volume_ratio > 1.5:
            # High volume increases conviction
            buy_score *= 1.1
            sell_score *= 1.1
            result.filters_passed.append("HIGH_VOLUME")
            score_details.append(f"HIGH_VOL(x1.1)")
        
        # ===== DAILY BIAS ADJUSTMENT =====
        # Penalize signals that go against the daily bias, boost signals that align
        if daily_bias == "BEARISH":
            # On bearish days, boost sell signals and penalize buy signals
            sell_score *= 1.3  # 30% boost to sell
            buy_score *= 0.6   # 40% penalty to buy (don't buy falling knives)
            score_details.append(f"DAILY_BEARISH(SELL*1.3,BUY*0.6)")
        elif daily_bias == "BULLISH":
            # On bullish days, boost buy signals and penalize sell signals
            buy_score *= 1.3   # 30% boost to buy
            sell_score *= 0.6  # 40% penalty to sell (don't short strength)
            score_details.append(f"DAILY_BULLISH(BUY*1.3,SELL*0.6)")
        else:
            score_details.append("DAILY_NEUTRAL")
        
        # Determine final signal - use scalp threshold in low-vol/range
        normal_threshold = self.config.get("signal_threshold", 40)
        signal_threshold = scalp_threshold if is_scalp_mode else normal_threshold
        
        # Log score details for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"SCORE_DEBUG: buy={buy_score:.1f}, sell={sell_score:.1f}, "
                   f"threshold={signal_threshold}, scalp_mode={is_scalp_mode}, daily_bias={daily_bias}")
        logger.info(f"SCORE_BREAKDOWN: {' | '.join(score_details)}")
        
        if buy_score > sell_score and buy_score >= signal_threshold:
            result.signal = TradeAction.BUY if not is_scalp_mode else TradeAction.SCALP_BUY
            result.score = min(buy_score, 100)
        elif sell_score > buy_score and sell_score >= signal_threshold:
            result.signal = TradeAction.SELL if not is_scalp_mode else TradeAction.SCALP_SELL
            result.score = min(sell_score, 100)
        else:
            result.signal = TradeAction.HOLD
            result.score = max(buy_score, sell_score)
        
        # Store breakdown for external access
        result.indicators["score_breakdown"] = score_details
        
        return result
    
    def record_trade(self) -> None:
        """Record that a trade was executed (for cooldown tracking)."""
        self.rule_engine.record_trade()


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
                
                priority_scores = []
                weighted_wins = 0.0
                weight_sum = 0.0
                enriched_trades: List[Dict[str, Any]] = []
                
                for idx, trade in enumerate(similar_trades):
                    trade_dict = trade.to_dict()
                    similarity_rank = 1.0 - (idx / max(1, len(similar_trades)))
                    recency_weight = recency_weight_from_timestamp(trade_dict.get("timestamp"))
                    priority = hybrid_trade_score(similarity_rank, recency_weight, trade_dict.get("pnl"))
                    
                    trade_dict["similarity_rank"] = round(similarity_rank, 3)
                    trade_dict["recency_weight"] = round(recency_weight, 3)
                    trade_dict["priority_score"] = round(priority, 3)
                    
                    priority_scores.append(priority)
                    weight_sum += priority
                    if trade_dict.get("result") == "WIN":
                        weighted_wins += priority
                    
                    enriched_trades.append(trade_dict)
                
                result.similar_trades = enriched_trades
                result.similar_trade_count = len(similar_trades)
                result.trade_priority_scores = priority_scores
                
                if similar_trades:
                    wins = sum(1 for t in similar_trades if t.result == "WIN")
                    result.historical_win_rate = wins / len(similar_trades)
                    result.avg_pnl_similar = sum(t.pnl for t in similar_trades) / len(similar_trades)
                    result.weighted_win_rate = weighted_wins / weight_sum if weight_sum else result.historical_win_rate
                
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
        band = self.config.get("uncertainty_band")
        if isinstance(band, (list, tuple)) and len(band) == 2:
            self.uncertainty_band = (float(band[0]), float(band[1]))
        else:
            self.uncertainty_band = (0.35, 0.65)
        self.call_cooldown_seconds = self.config.get("call_cooldown_seconds", 60)
        self.response_cache_ttl = self.config.get("response_cache_ttl_seconds", 900)
        self._response_cache: Dict[str, Tuple[float, LLMDecisionResult]] = {}
        self._last_call_candle: Optional[str] = None
        self._last_call_time: Optional[float] = None
        self._cache_store_path = Path(self.config.get("cache_store_path", "data/hybrid_llm_cache.json"))
        self._recent_call_times: deque[float] = deque()
        self._restore_cache_from_disk()
        
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
        cache_key = self._make_cache_key(prompt, market_data)
        cached = self._get_cached_response(cache_key)
        should_call, suppression_reason = self._should_invoke_llm(rule_result, rag_result, market_data)
        
        if cached and not should_call:
            logger.debug(f"LLM skipped ({suppression_reason}); using cached response")
            return cached
        
        if self.llm_client and should_call:
            try:
                response = self._call_llm(prompt, len(prompt))
                decision = self._parse_response(response)
                self._store_cached_response(cache_key, decision)
                self._last_call_candle = market_data.get("candle_timestamp")
                self._last_call_time = time.time()
                return decision
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
        
        if cached:
            logger.debug("LLM unavailable; using cached decision")
            return cached
        
        # Fallback to rule-based decision if LLM unavailable or skipped
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
        recent_bars = market_data.get("recent_bars") or []
        if recent_bars:
            window = recent_bars[-50:]
            closes = ", ".join(f"{bar['ts']}: {bar['close']:.2f}" for bar in window)
            recent_section = f"\n=== LAST {len(window)} CLOSES ===\n{closes}\n"
        else:
            recent_section = ""
        
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

{recent_section}

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
    
    def _call_llm(self, prompt: str, prompt_chars: int) -> str:
        """Call the LLM API.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            LLM response text
        """
        start_time = time.time()
        log_structured_event(
            agent="hybrid_pipeline",
            event_type="bedrock.call_start",
            message="Invoking Bedrock LLM",
            payload={"prompt_chars": prompt_chars},
        )
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
        latency_ms = (time.time() - start_time) * 1000
        now = time.time()
        self._recent_call_times.append(now)
        while self._recent_call_times and now - self._recent_call_times[0] > 60:
            self._recent_call_times.popleft()
        log_structured_event(
            agent="hybrid_pipeline",
            event_type="bedrock.call_complete",
            message="Bedrock call complete",
            payload={
                "latency_ms": latency_ms,
                "prompt_chars": prompt_chars,
                "calls_last_minute": len(self._recent_call_times),
            },
        )
        return result["content"][0]["text"]

    def _should_invoke_llm(
        self,
        rule_result: RuleEngineResult,
        rag_result: RAGRetrievalResult,
        market_data: Dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        normalized_score = max(0.0, min(1.0, rule_result.score / 100.0))
        band_low, band_high = self.uncertainty_band
        in_band = band_low <= normalized_score <= band_high
        rag_bias_sell = rag_result.weighted_win_rate < 0.45
        rag_bias_buy = rag_result.weighted_win_rate > 0.55
        signal = rule_result.signal
        conflict = (
            signal in (TradeAction.BUY, TradeAction.SCALP_BUY) and rag_bias_sell
        ) or (
            signal in (TradeAction.SELL, TradeAction.SCALP_SELL) and rag_bias_buy
        )
        candle_ts = market_data.get("candle_timestamp")
        if self._last_call_candle == candle_ts and not conflict:
            return False, "already_called_this_candle"
        if self._last_call_time and not conflict:
            if time.time() - self._last_call_time < self.call_cooldown_seconds:
                return False, "cooldown_active"
        if not in_band and not conflict:
            return False, "outside_uncertainty_band"
        return True, None

    def _make_cache_key(self, prompt: str, market_data: Dict[str, Any]) -> str:
        symbol = market_data.get("symbol", "UNKNOWN")
        timeframe = market_data.get("timeframe", "1m")
        candle_ts = market_data.get("candle_timestamp")
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        return f"{symbol}|{timeframe}|{candle_ts}|{prompt_hash}"

    def _get_cached_response(self, cache_key: str) -> Optional[LLMDecisionResult]:
        entry = self._response_cache.get(cache_key)
        if not entry:
            return None
        stored_at, result = entry
        if time.time() - stored_at > self.response_cache_ttl:
            self._response_cache.pop(cache_key, None)
            return None
        return self._clone_llm_result(result)

    def _store_cached_response(self, cache_key: str, decision: LLMDecisionResult) -> None:
        cloned = self._clone_llm_result(decision)
        self._response_cache[cache_key] = (time.time(), cloned)
        self._persist_cache_entry(cache_key, cloned)

    def _clone_llm_result(self, decision: LLMDecisionResult) -> LLMDecisionResult:
        return LLMDecisionResult(
            action=decision.action,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            suggested_stop_loss=decision.suggested_stop_loss,
            suggested_take_profit=decision.suggested_take_profit,
            position_size_factor=decision.position_size_factor,
            raw_response=decision.raw_response,
        )

    def _restore_cache_from_disk(self) -> None:
        if not self._cache_store_path:
            return
        try:
            if not self._cache_store_path.exists():
                return
            payload = json.loads(self._cache_store_path.read_text())
        except Exception:
            return
        for key, value in payload.items():
            ts = value.get("timestamp")
            if not ts:
                continue
            if time.time() - ts > self.response_cache_ttl:
                continue
            action_value = value.get("action", TradeAction.HOLD.value)
            try:
                action = TradeAction(action_value)
            except ValueError:
                action = TradeAction.HOLD
            result = LLMDecisionResult(
                action=action,
                confidence=value.get("confidence", 0),
                reasoning=value.get("reasoning", ""),
                suggested_stop_loss=value.get("stop_loss", 0.0),
                suggested_take_profit=value.get("take_profit", 0.0),
                position_size_factor=value.get("position_size", 1.0),
            )
            self._response_cache[key] = (ts, result)

    def _persist_cache_entry(self, cache_key: str, decision: LLMDecisionResult) -> None:
        if not self._cache_store_path:
            return
        try:
            existing = {}
            if self._cache_store_path.exists():
                existing = json.loads(self._cache_store_path.read_text())
            existing[cache_key] = {
                "timestamp": time.time(),
                "action": decision.action.value,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "stop_loss": decision.suggested_stop_loss,
                "take_profit": decision.suggested_take_profit,
                "position_size": decision.position_size_factor,
            }
            while len(existing) > 50:
                oldest_key = min(existing.items(), key=lambda item: item[1].get("timestamp", 0))[0]
                existing.pop(oldest_key, None)
            self._cache_store_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_store_path.write_text(json.dumps(existing))
        except Exception as exc:
            logger.debug(f"Skipping cache persistence: {exc}")
    
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
        self.min_confidence_for_trade = config.get("min_confidence_for_trade", 40)
        level_cfg = config.get("level_confirmation_settings", {})
        self.level_confirmation_settings = {
            "enabled": bool(
                level_cfg.get(
                    "level_confirmation_enabled",
                    level_cfg.get("enabled", config.get("level_confirmation_enabled", True)),
                )
            ),
            "proximity_pct": float(
                level_cfg.get(
                    "level_confirm_proximity_pct",
                    level_cfg.get("proximity_pct", config.get("level_confirm_proximity_pct", 0.15)),
                )
                or 0.15
            ),
            "buffer_atr_mult": float(
                level_cfg.get(
                    "level_confirm_buffer_atr_mult",
                    level_cfg.get("buffer_atr_mult", config.get("level_confirm_buffer_atr_mult", 0.10)),
                )
                or 0.10
            ),
            "min_buffer_points": float(
                level_cfg.get(
                    "level_confirm_min_buffer_points",
                    level_cfg.get(
                        "min_buffer_points",
                        level_cfg.get("pdh_buffer", config.get("level_confirm_min_buffer_points", 0.50)),
                    ),
                )
                or 0.50
            ),
            "max_wait_candles": int(
                level_cfg.get(
                    "level_confirm_max_wait_candles",
                    level_cfg.get("max_wait_candles", config.get("level_confirm_max_wait_candles", 3)),
                )
                or 3
            ),
            "timeout_mode": str(
                level_cfg.get(
                    "level_confirm_timeout_mode",
                    level_cfg.get("timeout_mode", config.get("level_confirm_timeout_mode", "SOFT_PENALTY")),
                )
                or "SOFT_PENALTY"
            ),
            "timeout_penalty": float(
                level_cfg.get(
                    "level_confirm_timeout_penalty",
                    level_cfg.get("timeout_penalty", config.get("level_confirm_timeout_penalty", 0.12)),
                )
                or 0.12
            ),
        }
        self._level_confirm_wait = {"BUY": 0, "SELL": 0}
        
        logger.info("HybridRAGPipeline initialized")

    def _reset_level_confirm_wait(self, direction: Optional[str] = None) -> None:
        """Reset wait counters for level confirmation."""
        if direction:
            if direction in self._level_confirm_wait:
                self._level_confirm_wait[direction] = 0
            return
        for key in self._level_confirm_wait:
            self._level_confirm_wait[key] = 0

    def _action_direction(self, action: TradeAction) -> Optional[str]:
        """Map trade action to BUY/SELL direction for gating."""
        if action in (TradeAction.BUY, TradeAction.SCALP_BUY):
            return "BUY"
        if action in (TradeAction.SELL, TradeAction.SCALP_SELL):
            return "SELL"
        return None

    def _apply_level_confirmation(
        self,
        final_action: TradeAction,
        final_confidence: float,
        final_reasoning: str,
        indicators: Dict[str, Any],
    ) -> Tuple[TradeAction, float, str, Optional[HoldReason], Optional[str]]:
        """AUTO confirmation gate near PDH/PDL with timeout + soft penalty."""
        cfg = getattr(self, "level_confirmation_settings", {})
        if not cfg.get("enabled", True):
            self._reset_level_confirm_wait()
            return final_action, final_confidence, final_reasoning, None, None

        direction = self._action_direction(final_action)
        if not direction:
            self._reset_level_confirm_wait()
            return final_action, final_confidence, final_reasoning, None, None

        pdh = indicators.get("pdh")
        pdl = indicators.get("pdl")
        atr = indicators.get("atr", 0.0) or 0.0
        close_price = indicators.get("close")
        price = indicators.get("price", close_price)
        price_ref = close_price if close_price is not None else price
        price_source = "close" if close_price is not None else "price"
        if price_ref is None or price_ref == 0:
            self._reset_level_confirm_wait(direction)
            return final_action, final_confidence, final_reasoning, None, None
        if close_price is None:
            logger.debug(
                "LEVEL_CONFIRMATION using fallback price source={} (price={}, close_missing=True)",
                price_source,
                price_ref,
            )

        buffer_points = max(
            float(cfg.get("min_buffer_points", 0.50) or 0.50),
            atr * float(cfg.get("buffer_atr_mult", 0.10) or 0.10),
        )
        proximity_pct = float(cfg.get("proximity_pct", 0.15) or 0.15)
        max_wait_candles = max(1, int(cfg.get("max_wait_candles", 3) or 3))
        timeout_mode = str(cfg.get("timeout_mode", "SOFT_PENALTY") or "SOFT_PENALTY").upper()
        timeout_penalty = float(cfg.get("timeout_penalty", 0.12) or 0.12)
        penalty_value = timeout_penalty
        if final_confidence > 1 and timeout_penalty <= 1:
            # Treat sub-1.0 penalties as percentage points when confidence is 0-100.
            penalty_value = timeout_penalty * 100

        def _valid_level(level: Optional[float]) -> bool:
            return level is not None and level > 0

        def _is_near(level: Optional[float]) -> bool:
            return _valid_level(level) and price_ref > 0 and abs(price_ref - float(level)) / price_ref * 100 <= proximity_pct

        near_pdh = _is_near(pdh)
        near_pdl = _is_near(pdl)

        target_type = None
        target_level = None
        condition_met = True
        if direction == "BUY":
            if near_pdl:
                target_type = "PDL_RECLAIM"
                target_level = float(pdl or 0) + buffer_points
                condition_met = price_ref > target_level
            elif near_pdh:
                target_type = "PDH_BREAK"
                target_level = float(pdh or 0) + buffer_points
                condition_met = price_ref > target_level
        elif direction == "SELL":
            if near_pdh:
                target_type = "PDH_REJECT"
                target_level = float(pdh or 0) - buffer_points
                condition_met = price_ref < target_level
            elif near_pdl:
                target_type = "PDL_BREAK"
                target_level = float(pdl or 0) - buffer_points
                condition_met = price_ref < target_level

        if not target_type or target_level is None:
            # Not near a key level; no gating.
            self._reset_level_confirm_wait(direction)
            return final_action, final_confidence, final_reasoning, None, None

        if condition_met:
            self._reset_level_confirm_wait(direction)
            return final_action, final_confidence, final_reasoning, None, None

        wait_count = self._level_confirm_wait.get(direction, 0) + 1
        self._level_confirm_wait[direction] = wait_count

        if wait_count >= max_wait_candles:
            # Timeout: allow trade with optional soft penalty instead of freezing.
            self._level_confirm_wait[direction] = 0
            if timeout_mode == "SOFT_PENALTY" and penalty_value > 0:
                penalized_conf = max(0.0, final_confidence - penalty_value)
                reason = (
                    f"Level confirmation timeout ({direction}) on {target_type} "
                    f"after {wait_count} candles: conf {final_confidence:.1f}% -> {penalized_conf:.1f}%"
                )
                logger.info(
                    "⏱️ LEVEL_CONFIRMATION timeout target={} wait={}/{} price_src={} price={} buffer={} conf={}->{}",
                    target_type,
                    wait_count,
                    max_wait_candles,
                    price_source,
                    price_ref,
                    buffer_points,
                    f"{final_confidence:.1f}%",
                    f"{penalized_conf:.1f}%",
                )
                final_reasoning = f"{final_reasoning}; {reason}" if final_reasoning else reason
                return final_action, penalized_conf, final_reasoning, None, reason

            reason = (
                f"Level confirmation timeout ({direction}) on {target_type} after {wait_count} candles"
            )
            logger.info(
                "⏱️ LEVEL_CONFIRMATION timeout target={} wait={}/{} price_src={} price={} buffer={}",
                target_type,
                wait_count,
                max_wait_candles,
                price_source,
                price_ref,
                buffer_points,
            )
            final_reasoning = f"{final_reasoning}; {reason}" if final_reasoning else reason
            return final_action, final_confidence, final_reasoning, None, reason

        comparison = "above" if direction == "BUY" else "below"
        confirmation_reason = (
            f"waiting for {price_source} {comparison} {target_level:.2f} "
            f"(buffer={buffer_points:.4f}, target={target_type}, wait={wait_count}/{max_wait_candles})"
        )
        final_reasoning = f"{final_reasoning}; {confirmation_reason}" if final_reasoning else confirmation_reason
        hold_reason = HoldReason(
            gate="hybrid_pipeline.rule_engine",
            reason_code="LEVEL_CONFIRMATION",
            reason_detail=confirmation_reason,
            context={
                "price": price,
                "close": close_price,
                "pdh": pdh,
                "pdl": pdl,
                "buffer_points": buffer_points,
                "near_pdh": near_pdh,
                "near_pdl": near_pdl,
                "wait_count": wait_count,
                "max_wait_candles": max_wait_candles,
                "target_type": target_type,
                "target_level": target_level,
                "price_source": price_source,
            },
        )
        logger.info(
            "🚫 HOLD [LEVEL_CONFIRMATION] target={} price_src={} price={} pdh={} pdl={} buffer={} near_pdh={} near_pdl={} wait={}/{}",
            target_type,
            price_source,
            price_ref,
            pdh,
            pdl,
            buffer_points,
            near_pdh,
            near_pdl,
            wait_count,
            max_wait_candles,
        )
        return TradeAction.HOLD, 0.0, final_reasoning, hold_reason, confirmation_reason
    
    def _evaluate_regime_filter(
        self,
        rag_result: RAGRetrievalResult,
        market_data: Dict[str, Any],
    ) -> Tuple[bool, float, str, Dict[str, Any]]:
        """Evaluate the RAG regime filter and return gating decisions.
        
        Returns:
            (hard_block, penalty, reason, context)
        """
        cfg = self.config.get("rag_regime_filter", {})
        enabled = cfg.get("enabled", True)
        if not enabled:
            return False, 0.0, "", {}

        strictness = str(cfg.get("strictness", cfg.get("mode", "relaxed")) or "relaxed").lower()
        min_sample_for_hard_block = int(cfg.get("min_sample_for_hard_block", 30) or 0)
        soft_penalty = float(cfg.get("soft_penalty_when_below", 0.10) or 0.0)
        hard_block_when_below = bool(cfg.get("hard_block_when_below", False))
        min_similar_trades_cfg = int(cfg.get("min_similar_trades", self.config.get("min_similar_trades", 2)) or 0)

        base_min_win_rate = float(cfg.get("min_win_rate", self.config.get("min_weighted_win_rate", 0.15)) or 0.15)
        soft_floor = min(
            base_min_win_rate,
            float(
                cfg.get(
                    "min_weighted_win_rate_soft_floor",
                    self.config.get("min_weighted_win_rate_soft_floor", base_min_win_rate),
                )
                or base_min_win_rate
            ),
        )
        full_threshold_trades = max(
            int(
                cfg.get(
                    "min_similar_trades_for_full_threshold",
                    self.config.get("min_similar_trades_for_full_threshold", 0),
                )
                or 0
            ),
            0,
        )
        use_relaxed_threshold = (
            strictness != "strict"
            and rag_result.similar_trade_count < full_threshold_trades
            and soft_floor < base_min_win_rate
        )
        effective_min_win_rate = soft_floor if use_relaxed_threshold else base_min_win_rate

        # Futures-friendly relaxation using market regime context
        trend = str(market_data.get("trend") or market_data.get("market_trend") or "").upper()
        volatility = str(market_data.get("volatility") or market_data.get("volatility_regime") or "").upper()
        regime = str(market_data.get("regime") or market_data.get("market_regime") or "").upper()
        combined_regime = regime or (f"{trend}_{volatility}" if trend and volatility else trend or volatility)
        normalized_regime_tokens = {
            token
            for token in {combined_regime, regime, trend, volatility}
            if token
        }
        if cfg.get("expand_generic_regimes", True):
            # Add base tokens to catch generic allowed values like TRENDING/RANGING
            for token in list(normalized_regime_tokens):
                if "TREND" in token:
                    normalized_regime_tokens.add("TRENDING")
                if "RANGE" in token:
                    normalized_regime_tokens.add("RANGING")
                if "VOL" in token:
                    normalized_regime_tokens.add("VOLATILE")

        allowed_regimes_cfg = cfg.get(
            "allowed_regimes",
            [
                "TRENDING_UP",
                "TRENDING_DOWN",
                "RANGING_HIGH_VOL",
                "RANGING_LOW_VOL",
                "BREAKOUT_PENDING",
                "REVERSAL_SETUP",
                "TRENDING",
                "RANGING",
                "VOLATILE",
                "QUIET",
                "UNCERTAIN",
            ],
        )
        allowed_regimes = {r.upper() for r in allowed_regimes_cfg}
        evening_relaxed = bool(cfg.get("evening_relaxed", False))
        current_hour = datetime.now().hour
        asset_class = str(market_data.get("asset_class") or market_data.get("instrument_type") or "").upper()
        futures_mode = bool(cfg.get("futures_mode", True) or "FUT" in asset_class)

        relax_reason = ""
        if futures_mode and evening_relaxed and 17 <= current_hour <= 23:
            relax_reason = "Evening session - regime requirements relaxed"
        elif futures_mode and normalized_regime_tokens.intersection(allowed_regimes):
            relax_reason = f"Futures regime allowed ({combined_regime or trend or volatility})"

        if relax_reason:
            # Apply futures-mode relaxation adjustments (overrides initial strictness)
            strictness = "low"
            hard_block_when_below = False
            min_similar_trades_cfg = max(0, min_similar_trades_cfg - 1)
            base_min_win_rate = max(0.05, base_min_win_rate * 0.8)
            soft_penalty = soft_penalty * 0.5
            effective_min_win_rate = min(effective_min_win_rate, base_min_win_rate)
            use_relaxed_threshold = True

        if not relax_reason:
            if strictness == "medium":
                hard_block_when_below = False
                base_min_win_rate = max(0.05, base_min_win_rate * 0.9)
                effective_min_win_rate = min(effective_min_win_rate, base_min_win_rate)
                min_similar_trades_cfg = max(0, min_similar_trades_cfg - 1)
            elif strictness == "low":
                hard_block_when_below = False
                base_min_win_rate = max(0.05, base_min_win_rate * 0.85)
                effective_min_win_rate = min(effective_min_win_rate, base_min_win_rate)
                min_similar_trades_cfg = max(0, min_similar_trades_cfg - 1)

        wins = sum(1 for t in rag_result.similar_trades if str(t.get("result", "")).upper() == "WIN")
        n = int(rag_result.similar_trade_count or 0)
        smoothed_win_rate = (wins + 1) / (n + 2)

        below_threshold = smoothed_win_rate < effective_min_win_rate or n < min_similar_trades_cfg
        hard_block = (
            below_threshold
            and hard_block_when_below
            and n >= min_sample_for_hard_block
            and smoothed_win_rate < effective_min_win_rate
        )

        reason = (
            f"Regime check: similar_trades={n} (min={min_similar_trades_cfg}), "
            f"smoothed_win_rate={smoothed_win_rate:.2f} (min={effective_min_win_rate:.2f}, "
            f"mode={'relaxed' if use_relaxed_threshold else 'strict'})"
        )
        if relax_reason:
            reason = f"{relax_reason}; {reason}"
        if soft_penalty > 0 and below_threshold:
            penalty_pct = soft_penalty * 100 if soft_penalty <= 1 else soft_penalty
            reason = f"{reason}; applying regime penalty {penalty_pct:.1f}pts"

        context = {
            "similar_trades": n,
            "wins": wins,
            "smoothed_win_rate": smoothed_win_rate,
            "weighted_win_rate": rag_result.weighted_win_rate,
            "effective_min_win_rate": effective_min_win_rate,
            "min_similar_trades": min_similar_trades_cfg,
            "full_threshold_trades": full_threshold_trades,
            "threshold_mode": "relaxed" if use_relaxed_threshold else "strict",
            "mode": strictness,
            "penalty": soft_penalty,
            "hard_block": hard_block,
            "min_sample_for_hard_block": min_sample_for_hard_block,
            "regime": combined_regime,
            "trend": trend,
            "volatility": volatility,
            "relax_reason": relax_reason,
            "allowed_regimes": sorted(allowed_regimes),
        }
        return hard_block, soft_penalty if below_threshold else 0.0, reason, context

    def process(self, market_data: Dict[str, Any]) -> HybridPipelineResult:
        """Process market data through all three layers."""
        start_time = time.time()
        hold_reason: Optional[HoldReason] = None
        regime_penalty = 0.0
        regime_reason = ""
        regime_context: Dict[str, Any] = {}

        # Layer 1: Rule Engine (always runs)
        rule_result = self.rule_engine.evaluate(market_data)
        logger.debug(f"Layer 1 - Rule Engine: {rule_result.signal.value} ({rule_result.score:.1f})")

        # Early exit if blocked or no actionable signal
        if not rule_result.should_proceed:
            reason_code = "FILTER_BLOCK" if rule_result.filters_blocked else "NO_SIGNAL"
            reason_detail = (
                f"blocked filters: {', '.join(rule_result.filters_blocked)}"
                if rule_result.filters_blocked
                else f"no actionable signal ({rule_result.signal.value})"
            )
            hold_reason = HoldReason(
                gate="hybrid_pipeline.rule_engine",
                reason_code=reason_code,
                reason_detail=reason_detail,
                context={
                    "filters_blocked": rule_result.filters_blocked,
                    "signal": rule_result.signal.value,
                    "score": rule_result.score,
                    "filters_passed": rule_result.filters_passed,
                    "filters_warned": rule_result.filters_warned,
                },
            )
            logger.info(
                "🚫 HOLD [{}] gate={} detail={} filters={}",
                hold_reason.reason_code,
                hold_reason.gate,
                hold_reason.reason_detail,
                rule_result.filters_blocked,
            )
            reasoning = (
                f"Blocked by filters: {', '.join(rule_result.filters_blocked)}"
                if rule_result.filters_blocked
                else f"No actionable signal ({rule_result.signal.value}, score={rule_result.score:.1f})"
            )
            return HybridPipelineResult(
                final_action=rule_result.signal,
                final_confidence=0,
                final_reasoning=reasoning,
                rule_engine=rule_result,
                rag_retrieval=RAGRetrievalResult(),
                hold_reason=hold_reason,
                timestamp=now_cst().isoformat(),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Layer 2: RAG Retrieval (only on signal)
        rag_result = self.rag_retriever.retrieve(rule_result, market_data)
        logger.debug(f"Layer 2 - RAG: {rag_result.similar_trade_count} similar trades, {len(rag_result.documents)} docs")

        # Regime filter using RAG context before invoking the LLM
        hard_block, regime_penalty, regime_reason, regime_context = self._evaluate_regime_filter(
            rag_result, market_data
        )
        if hard_block:
            hold_reason = HoldReason(
                gate="hybrid_pipeline.rag",
                reason_code="RAG_REGIME_FILTER",
                reason_detail=regime_reason,
                context=regime_context,
            )
            logger.info(
                "🚫 HOLD [{}] gate={} detail={}",
                hold_reason.reason_code,
                hold_reason.gate,
                hold_reason.reason_detail,
            )
            return HybridPipelineResult(
                final_action=TradeAction.HOLD,
                final_confidence=0,
                final_reasoning=regime_reason,
                rule_engine=rule_result,
                rag_retrieval=rag_result,
                hold_reason=hold_reason,
                timestamp=now_cst().isoformat(),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Layer 3: LLM Decision (only on signal with sufficient score)
        llm_result = None
        if not self.skip_llm_on_low_score or rule_result.score >= self.min_score_for_llm:
            llm_result = self.llm_decision.decide(rule_result, rag_result, market_data)
            logger.debug(f"Layer 3 - LLM: {llm_result.action.value} ({llm_result.confidence:.1f}%)")

        # Determine final action
        hold_reason = None
        if llm_result:
            final_action = llm_result.action
            final_confidence = llm_result.confidence
            final_reasoning = llm_result.reasoning

            # Apply confidence threshold
            if final_confidence < self.min_confidence_for_trade:
                final_action = TradeAction.HOLD
                final_reasoning = f"Confidence too low ({final_confidence:.0f}% < {self.min_confidence_for_trade}%)"
                hold_reason = HoldReason(
                    gate="hybrid_pipeline.llm",
                    reason_code="CONFIDENCE_TOO_LOW",
                    reason_detail=final_reasoning,
                    context={
                        "confidence": final_confidence,
                        "threshold": self.min_confidence_for_trade,
                        "llm_action": llm_result.action.value,
                    },
                )
        else:
            # Use rule engine result directly
            final_action = rule_result.signal
            final_confidence = rule_result.score
            final_reasoning = f"Rule-based: {rule_result.signal.value}"

        # Apply regime penalty (soft gating) after LLM/rule decision
        if regime_penalty > 0 and final_confidence is not None:
            penalty_value = regime_penalty * 100 if regime_penalty <= 1 and final_confidence > 1 else regime_penalty
            penalized_conf = max(0.0, final_confidence - penalty_value)
            regime_reason_suffix = regime_reason or "Regime penalty applied"
            logger.info(
                "📉 Regime penalty applied ({}) conf={:.1f}->{:.1f}",
                regime_reason_suffix,
                final_confidence,
                penalized_conf,
            )
            final_reasoning = f"{final_reasoning}; {regime_reason_suffix}" if final_reasoning else regime_reason_suffix
            final_confidence = penalized_conf

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

        if final_action in (TradeAction.BUY, TradeAction.SELL, TradeAction.SCALP_BUY, TradeAction.SCALP_SELL):
            (
                final_action,
                final_confidence,
                final_reasoning,
                hold_reason_update,
                _,
            ) = self._apply_level_confirmation(
                final_action=final_action,
                final_confidence=final_confidence,
                final_reasoning=final_reasoning,
                indicators=rule_result.indicators,
            )
            if hold_reason_update:
                hold_reason = hold_reason_update

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
            hold_reason=hold_reason if final_action == TradeAction.HOLD else None,
        )

        if result.hold_reason:
            logger.info(
                "🚫 HOLD [{}] gate={} detail={}",
                result.hold_reason.reason_code,
                result.hold_reason.gate,
                result.hold_reason.reason_detail,
            )
        logger.info(
            f"Pipeline result: {result.final_action.value} "
            f"(conf={result.final_confidence:.0f}%, time={result.processing_time_ms:.0f}ms)"
        )

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
                
                priority_scores = []
                weighted_wins = 0.0
                weight_sum = 0.0
                enriched_trades: List[Dict[str, Any]] = []
                
                for idx, trade in enumerate(similar_trades):
                    trade_dict = trade.to_dict()
                    similarity_rank = 1.0 - (idx / max(1, len(similar_trades)))
                    recency_weight = recency_weight_from_timestamp(trade_dict.get("timestamp"))
                    priority = hybrid_trade_score(similarity_rank, recency_weight, trade_dict.get("pnl"))
                    
                    trade_dict["similarity_rank"] = round(similarity_rank, 3)
                    trade_dict["recency_weight"] = round(recency_weight, 3)
                    trade_dict["priority_score"] = round(priority, 3)
                    
                    priority_scores.append(priority)
                    weight_sum += priority
                    if trade_dict.get("result") == "WIN":
                        weighted_wins += priority
                    
                    enriched_trades.append(trade_dict)
                
                result.similar_trades = enriched_trades
                result.similar_trade_count = len(similar_trades)
                result.trade_priority_scores = priority_scores
                
                if similar_trades:
                    wins = sum(1 for t in similar_trades if t.result == "WIN")
                    result.historical_win_rate = wins / len(similar_trades)
                    result.avg_pnl_similar = sum(t.pnl for t in similar_trades) / len(similar_trades)
                    result.weighted_win_rate = weighted_wins / weight_sum if weight_sum else result.historical_win_rate
                
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
        band = self.config.get("uncertainty_band")
        if isinstance(band, (list, tuple)) and len(band) == 2:
            self.uncertainty_band = (float(band[0]), float(band[1]))
        else:
            self.uncertainty_band = (0.35, 0.65)
        self.call_cooldown_seconds = self.config.get("call_cooldown_seconds", 60)
        self.response_cache_ttl = self.config.get("response_cache_ttl_seconds", 900)
        self._response_cache: Dict[str, Tuple[float, LLMDecisionResult]] = {}
        self._last_call_candle: Optional[str] = None
        self._last_call_time: Optional[float] = None
        self._cache_store_path = Path(self.config.get("cache_store_path", "data/hybrid_llm_cache.json"))
        self._recent_call_times: deque[float] = deque()
        self._restore_cache_from_disk()
        
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
        cache_key = self._make_cache_key(prompt, market_data)
        cached = self._get_cached_response(cache_key)
        should_call, suppression_reason = self._should_invoke_llm(rule_result, rag_result, market_data)
        
        if cached and not should_call:
            logger.debug(f"LLM skipped ({suppression_reason}); using cached response")
            return cached
        
        if self.llm_client and should_call:
            try:
                response = self._call_llm(prompt, len(prompt))
                decision = self._parse_response(response)
                self._store_cached_response(cache_key, decision)
                self._last_call_candle = market_data.get("candle_timestamp")
                self._last_call_time = time.time()
                return decision
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
        
        if cached:
            logger.debug("LLM unavailable; using cached decision")
            return cached
        
        # Fallback to rule-based decision if LLM unavailable or skipped
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
        recent_bars = market_data.get("recent_bars") or []
        if recent_bars:
            window = recent_bars[-50:]
            closes = ", ".join(f"{bar['ts']}: {bar['close']:.2f}" for bar in window)
            recent_section = f"\n=== LAST {len(window)} CLOSES ===\n{closes}\n"
        else:
            recent_section = ""
        
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

{recent_section}

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
    
    def _call_llm(self, prompt: str, prompt_chars: int) -> str:
        """Call the LLM API.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            LLM response text
        """
        start_time = time.time()
        log_structured_event(
            agent="hybrid_pipeline",
            event_type="bedrock.call_start",
            message="Invoking Bedrock LLM",
            payload={"prompt_chars": prompt_chars},
        )
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
        latency_ms = (time.time() - start_time) * 1000
        now = time.time()
        self._recent_call_times.append(now)
        while self._recent_call_times and now - self._recent_call_times[0] > 60:
            self._recent_call_times.popleft()
        log_structured_event(
            agent="hybrid_pipeline",
            event_type="bedrock.call_complete",
            message="Bedrock call complete",
            payload={
                "latency_ms": latency_ms,
                "prompt_chars": prompt_chars,
                "calls_last_minute": len(self._recent_call_times),
            },
        )
        return result["content"][0]["text"]

    def _should_invoke_llm(
        self,
        rule_result: RuleEngineResult,
        rag_result: RAGRetrievalResult,
        market_data: Dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        normalized_score = max(0.0, min(1.0, rule_result.score / 100.0))
        band_low, band_high = self.uncertainty_band
        in_band = band_low <= normalized_score <= band_high
        rag_bias_sell = rag_result.weighted_win_rate < 0.45
        rag_bias_buy = rag_result.weighted_win_rate > 0.55
        signal = rule_result.signal
        conflict = (
            signal in (TradeAction.BUY, TradeAction.SCALP_BUY) and rag_bias_sell
        ) or (
            signal in (TradeAction.SELL, TradeAction.SCALP_SELL) and rag_bias_buy
        )
        candle_ts = market_data.get("candle_timestamp")
        if self._last_call_candle == candle_ts and not conflict:
            return False, "already_called_this_candle"
        if self._last_call_time and not conflict:
            if time.time() - self._last_call_time < self.call_cooldown_seconds:
                return False, "cooldown_active"
        if not in_band and not conflict:
            return False, "outside_uncertainty_band"
        return True, None

    def _make_cache_key(self, prompt: str, market_data: Dict[str, Any]) -> str:
        symbol = market_data.get("symbol", "UNKNOWN")
        timeframe = market_data.get("timeframe", "1m")
        candle_ts = market_data.get("candle_timestamp")
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        return f"{symbol}|{timeframe}|{candle_ts}|{prompt_hash}"

    def _get_cached_response(self, cache_key: str) -> Optional[LLMDecisionResult]:
        entry = self._response_cache.get(cache_key)
        if not entry:
            return None
        stored_at, result = entry
        if time.time() - stored_at > self.response_cache_ttl:
            self._response_cache.pop(cache_key, None)
            return None
        return self._clone_llm_result(result)

    def _store_cached_response(self, cache_key: str, decision: LLMDecisionResult) -> None:
        cloned = self._clone_llm_result(decision)
        self._response_cache[cache_key] = (time.time(), cloned)
        self._persist_cache_entry(cache_key, cloned)

    def _clone_llm_result(self, decision: LLMDecisionResult) -> LLMDecisionResult:
        return LLMDecisionResult(
            action=decision.action,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            suggested_stop_loss=decision.suggested_stop_loss,
            suggested_take_profit=decision.suggested_take_profit,
            position_size_factor=decision.position_size_factor,
            raw_response=decision.raw_response,
        )

    def _restore_cache_from_disk(self) -> None:
        if not self._cache_store_path:
            return
        try:
            if not self._cache_store_path.exists():
                return
            payload = json.loads(self._cache_store_path.read_text())
        except Exception:
            return
        for key, value in payload.items():
            ts = value.get("timestamp")
            if not ts:
                continue
            if time.time() - ts > self.response_cache_ttl:
                continue
            action_value = value.get("action", TradeAction.HOLD.value)
            try:
                action = TradeAction(action_value)
            except ValueError:
                action = TradeAction.HOLD
            result = LLMDecisionResult(
                action=action,
                confidence=value.get("confidence", 0),
                reasoning=value.get("reasoning", ""),
                suggested_stop_loss=value.get("stop_loss", 0.0),
                suggested_take_profit=value.get("take_profit", 0.0),
                position_size_factor=value.get("position_size", 1.0),
            )
            self._response_cache[key] = (ts, result)

    def _persist_cache_entry(self, cache_key: str, decision: LLMDecisionResult) -> None:
        if not self._cache_store_path:
            return
        try:
            existing = {}
            if self._cache_store_path.exists():
                existing = json.loads(self._cache_store_path.read_text())
            existing[cache_key] = {
                "timestamp": time.time(),
                "action": decision.action.value,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "stop_loss": decision.suggested_stop_loss,
                "take_profit": decision.suggested_take_profit,
                "position_size": decision.position_size_factor,
            }
            while len(existing) > 50:
                oldest_key = min(existing.items(), key=lambda item: item[1].get("timestamp", 0))[0]
                existing.pop(oldest_key, None)
            self._cache_store_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_store_path.write_text(json.dumps(existing))
        except Exception as exc:
            logger.debug(f"Skipping cache persistence: {exc}")
    
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
        self.min_confidence_for_trade = config.get("min_confidence_for_trade", 40)
        level_cfg = config.get("level_confirmation_settings", {})
        self.level_confirmation_settings = {
            "enabled": bool(
                level_cfg.get(
                    "level_confirmation_enabled",
                    level_cfg.get("enabled", config.get("level_confirmation_enabled", True)),
                )
            ),
            "proximity_pct": float(
                level_cfg.get(
                    "level_confirm_proximity_pct",
                    level_cfg.get("proximity_pct", config.get("level_confirm_proximity_pct", 0.15)),
                )
                or 0.15
            ),
            "buffer_atr_mult": float(
                level_cfg.get(
                    "level_confirm_buffer_atr_mult",
                    level_cfg.get("buffer_atr_mult", config.get("level_confirm_buffer_atr_mult", 0.10)),
                )
                or 0.10
            ),
            "min_buffer_points": float(
                level_cfg.get(
                    "level_confirm_min_buffer_points",
                    level_cfg.get(
                        "min_buffer_points",
                        level_cfg.get("pdh_buffer", config.get("level_confirm_min_buffer_points", 0.50)),
                    ),
                )
                or 0.50
            ),
            "max_wait_candles": int(
                level_cfg.get(
                    "level_confirm_max_wait_candles",
                    level_cfg.get("max_wait_candles", config.get("level_confirm_max_wait_candles", 3)),
                )
                or 3
            ),
            "timeout_mode": str(
                level_cfg.get(
                    "level_confirm_timeout_mode",
                    level_cfg.get("timeout_mode", config.get("level_confirm_timeout_mode", "SOFT_PENALTY")),
                )
                or "SOFT_PENALTY"
            ),
            "timeout_penalty": float(
                level_cfg.get(
                    "level_confirm_timeout_penalty",
                    level_cfg.get("timeout_penalty", config.get("level_confirm_timeout_penalty", 0.12)),
                )
                or 0.12
            ),
        }
        self._level_confirm_wait = {"BUY": 0, "SELL": 0}
        
        logger.info("HybridRAGPipeline initialized")

    def _reset_level_confirm_wait(self, direction: Optional[str] = None) -> None:
        """Reset wait counters for level confirmation."""
        if direction:
            if direction in self._level_confirm_wait:
                self._level_confirm_wait[direction] = 0
            return
        for key in self._level_confirm_wait:
            self._level_confirm_wait[key] = 0

    def _action_direction(self, action: TradeAction) -> Optional[str]:
        """Map trade action to BUY/SELL direction for gating."""
        if action in (TradeAction.BUY, TradeAction.SCALP_BUY):
            return "BUY"
        if action in (TradeAction.SELL, TradeAction.SCALP_SELL):
            return "SELL"
        return None

    def _apply_level_confirmation(
        self,
        final_action: TradeAction,
        final_confidence: float,
        final_reasoning: str,
        indicators: Dict[str, Any],
    ) -> Tuple[TradeAction, float, str, Optional[HoldReason], Optional[str]]:
        """AUTO confirmation gate near PDH/PDL with timeout + soft penalty."""
        cfg = getattr(self, "level_confirmation_settings", {})
        if not cfg.get("enabled", True):
            self._reset_level_confirm_wait()
            return final_action, final_confidence, final_reasoning, None, None

        direction = self._action_direction(final_action)
        if not direction:
            self._reset_level_confirm_wait()
            return final_action, final_confidence, final_reasoning, None, None

        pdh = indicators.get("pdh")
        pdl = indicators.get("pdl")
        atr = indicators.get("atr", 0.0) or 0.0
        close_price = indicators.get("close")
        price = indicators.get("price", close_price)
        price_ref = close_price if close_price is not None else price
        price_source = "close" if close_price is not None else "price"
        if price_ref is None or price_ref == 0:
            self._reset_level_confirm_wait(direction)
            return final_action, final_confidence, final_reasoning, None, None
        if close_price is None:
            logger.debug(
                "LEVEL_CONFIRMATION using fallback price source={} (price={}, close_missing=True)",
                price_source,
                price_ref,
            )

        buffer_points = max(
            float(cfg.get("min_buffer_points", 0.50) or 0.50),
            atr * float(cfg.get("buffer_atr_mult", 0.10) or 0.10),
        )
        proximity_pct = float(cfg.get("proximity_pct", 0.15) or 0.15)
        max_wait_candles = max(1, int(cfg.get("max_wait_candles", 3) or 3))
        timeout_mode = str(cfg.get("timeout_mode", "SOFT_PENALTY") or "SOFT_PENALTY").upper()
        timeout_penalty = float(cfg.get("timeout_penalty", 0.12) or 0.12)
        penalty_value = timeout_penalty
        if final_confidence > 1 and timeout_penalty <= 1:
            # Treat sub-1.0 penalties as percentage points when confidence is 0-100.
            penalty_value = timeout_penalty * 100

        def _valid_level(level: Optional[float]) -> bool:
            return level is not None and level > 0

        def _is_near(level: Optional[float]) -> bool:
            return _valid_level(level) and price_ref > 0 and abs(price_ref - float(level)) / price_ref * 100 <= proximity_pct

        near_pdh = _is_near(pdh)
        near_pdl = _is_near(pdl)

        target_type = None
        target_level = None
        condition_met = True
        if direction == "BUY":
            if near_pdl:
                target_type = "PDL_RECLAIM"
                target_level = float(pdl or 0) + buffer_points
                condition_met = price_ref > target_level
            elif near_pdh:
                target_type = "PDH_BREAK"
                target_level = float(pdh or 0) + buffer_points
                condition_met = price_ref > target_level
        elif direction == "SELL":
            if near_pdh:
                target_type = "PDH_REJECT"
                target_level = float(pdh or 0) - buffer_points
                condition_met = price_ref < target_level
            elif near_pdl:
                target_type = "PDL_BREAK"
                target_level = float(pdl or 0) - buffer_points
                condition_met = price_ref < target_level

        if not target_type or target_level is None:
            # Not near a key level; no gating.
            self._reset_level_confirm_wait(direction)
            return final_action, final_confidence, final_reasoning, None, None

        if condition_met:
            self._reset_level_confirm_wait(direction)
            return final_action, final_confidence, final_reasoning, None, None

        wait_count = self._level_confirm_wait.get(direction, 0) + 1
        self._level_confirm_wait[direction] = wait_count

        if wait_count >= max_wait_candles:
            # Timeout: allow trade with optional soft penalty instead of freezing.
            self._level_confirm_wait[direction] = 0
            if timeout_mode == "SOFT_PENALTY" and penalty_value > 0:
                penalized_conf = max(0.0, final_confidence - penalty_value)
                reason = (
                    f"Level confirmation timeout ({direction}) on {target_type} "
                    f"after {wait_count} candles: conf {final_confidence:.1f}% -> {penalized_conf:.1f}%"
                )
                logger.info(
                    "⏱️ LEVEL_CONFIRMATION timeout target={} wait={}/{} price_src={} price={} buffer={} conf={}->{}",
                    target_type,
                    wait_count,
                    max_wait_candles,
                    price_source,
                    price_ref,
                    buffer_points,
                    f"{final_confidence:.1f}%",
                    f"{penalized_conf:.1f}%",
                )
                final_reasoning = f"{final_reasoning}; {reason}" if final_reasoning else reason
                return final_action, penalized_conf, final_reasoning, None, reason

            reason = (
                f"Level confirmation timeout ({direction}) on {target_type} after {wait_count} candles"
            )
            logger.info(
                "⏱️ LEVEL_CONFIRMATION timeout target={} wait={}/{} price_src={} price={} buffer={}",
                target_type,
                wait_count,
                max_wait_candles,
                price_source,
                price_ref,
                buffer_points,
            )
            final_reasoning = f"{final_reasoning}; {reason}" if final_reasoning else reason
            return final_action, final_confidence, final_reasoning, None, reason

        comparison = "above" if direction == "BUY" else "below"
        confirmation_reason = (
            f"waiting for {price_source} {comparison} {target_level:.2f} "
            f"(buffer={buffer_points:.4f}, target={target_type}, wait={wait_count}/{max_wait_candles})"
        )
        final_reasoning = f"{final_reasoning}; {confirmation_reason}" if final_reasoning else confirmation_reason
        hold_reason = HoldReason(
            gate="hybrid_pipeline.rule_engine",
            reason_code="LEVEL_CONFIRMATION",
            reason_detail=confirmation_reason,
            context={
                "price": price,
                "close": close_price,
                "pdh": pdh,
                "pdl": pdl,
                "buffer_points": buffer_points,
                "near_pdh": near_pdh,
                "near_pdl": near_pdl,
                "wait_count": wait_count,
                "max_wait_candles": max_wait_candles,
                "target_type": target_type,
                "target_level": target_level,
                "price_source": price_source,
            },
        )
        logger.info(
            "🚫 HOLD [LEVEL_CONFIRMATION] target={} price_src={} price={} pdh={} pdl={} buffer={} near_pdh={} near_pdl={} wait={}/{}",
            target_type,
            price_source,
            price_ref,
            pdh,
            pdl,
            buffer_points,
            near_pdh,
            near_pdl,
            wait_count,
            max_wait_candles,
        )
        return TradeAction.HOLD, 0.0, final_reasoning, hold_reason, confirmation_reason
    
    def process(self, market_data: Dict[str, Any]) -> HybridPipelineResult:
        """Process market data through all three layers.
        
        Args:
            market_data: Current market data with indicators
            
        Returns:
            HybridPipelineResult with final decision
        """
        start_time = time.time()
        hold_reason: Optional[HoldReason] = None
        
        # Layer 1: Rule Engine (always runs)
        rule_result = self.rule_engine.evaluate(market_data)
        logger.debug(f"Layer 1 - Rule Engine: {rule_result.signal.value} ({rule_result.score:.1f})")
        
        # Early exit if blocked or no actionable signal
        if not rule_result.should_proceed:
            reason_code = "FILTER_BLOCK" if rule_result.filters_blocked else "NO_SIGNAL"
            reason_detail = (
                f"blocked filters: {', '.join(rule_result.filters_blocked)}"
                if rule_result.filters_blocked
                else f"no actionable signal ({rule_result.signal.value})"
            )
            hold_reason = HoldReason(
                gate="hybrid_pipeline.rule_engine",
                reason_code=reason_code,
                reason_detail=reason_detail,
                context={
                    "filters_blocked": rule_result.filters_blocked,
                    "signal": rule_result.signal.value,
                    "score": rule_result.score,
                    "filters_passed": rule_result.filters_passed,
                    "filters_warned": rule_result.filters_warned,
                },
            )
            logger.info(
                "🚫 HOLD [{}] gate={} detail={} filters={}",
                hold_reason.reason_code,
                hold_reason.gate,
                hold_reason.reason_detail,
                rule_result.filters_blocked,
            )
            reasoning = (
                f"Blocked by filters: {', '.join(rule_result.filters_blocked)}"
                if rule_result.filters_blocked
                else f"No actionable signal ({rule_result.signal.value}, score={rule_result.score:.1f})"
            )
            return HybridPipelineResult(
                final_action=rule_result.signal,
                final_confidence=0,
                final_reasoning=reasoning,
                rule_engine=rule_result,
                rag_retrieval=RAGRetrievalResult(),
                hold_reason=hold_reason,
                timestamp=now_cst().isoformat(),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Layer 2: RAG Retrieval (only on signal)
        rag_result = self.rag_retriever.retrieve(rule_result, market_data)
        logger.debug(f"Layer 2 - RAG: {rag_result.similar_trade_count} similar trades, {len(rag_result.documents)} docs")

        # Regime filter using RAG context before invoking the LLM
        min_similar_trades = self.config.get("min_similar_trades", 2)
        min_weighted_win_rate = self.config.get("min_weighted_win_rate", 0.45)
        soft_floor = min(
            min_weighted_win_rate,
            self.config.get("min_weighted_win_rate_soft_floor", min_weighted_win_rate),
        )
        full_threshold_trades = max(
            min_similar_trades,
            self.config.get("min_similar_trades_for_full_threshold", 0),
        )
        use_relaxed_threshold = (
            rag_result.similar_trade_count < full_threshold_trades
            and soft_floor < min_weighted_win_rate
        )
        effective_min_win_rate = soft_floor if use_relaxed_threshold else min_weighted_win_rate
        threshold_mode = "relaxed" if use_relaxed_threshold else "strict"
        if (
            rag_result.similar_trade_count < min_similar_trades
            or rag_result.weighted_win_rate < effective_min_win_rate
        ):
            reasoning = (
                "Regime filter triggered: "
                f"similar_trades={rag_result.similar_trade_count} (min={min_similar_trades}), "
                f"weighted_win_rate={rag_result.weighted_win_rate:.2f} "
                f"(min={effective_min_win_rate:.2f}, mode={threshold_mode})"
            )
            hold_reason = HoldReason(
                gate="hybrid_pipeline.rag",
                reason_code="RAG_REGIME_FILTER",
                reason_detail=reasoning,
                context={
                    "similar_trades": rag_result.similar_trade_count,
                    "weighted_win_rate": rag_result.weighted_win_rate,
                    "effective_min_weighted_win_rate": effective_min_win_rate,
                    "threshold_mode": threshold_mode,
                    "full_threshold_trades": full_threshold_trades,
                },
            )
            logger.info(
                "🚫 HOLD [{}] gate={} detail={}",
                hold_reason.reason_code,
                hold_reason.gate,
                hold_reason.reason_detail,
            )
            return HybridPipelineResult(
                final_action=TradeAction.HOLD,
                final_confidence=0,
                final_reasoning=reasoning,
                rule_engine=rule_result,
                rag_retrieval=rag_result,
                hold_reason=hold_reason,
                timestamp=now_cst().isoformat(),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Layer 3: LLM Decision (only on signal with sufficient score)
        llm_result = None
        if not self.skip_llm_on_low_score or rule_result.score >= self.min_score_for_llm:
            llm_result = self.llm_decision.decide(rule_result, rag_result, market_data)
            logger.debug(f"Layer 3 - LLM: {llm_result.action.value} ({llm_result.confidence:.1f}%)")
        
        # Determine final action
        hold_reason = None
        if llm_result:
            final_action = llm_result.action
            final_confidence = llm_result.confidence
            final_reasoning = llm_result.reasoning
            
            # Apply confidence threshold
            if final_confidence < self.min_confidence_for_trade:
                final_action = TradeAction.HOLD
                final_reasoning = f"Confidence too low ({final_confidence:.0f}% < {self.min_confidence_for_trade}%)"
                hold_reason = HoldReason(
                    gate="hybrid_pipeline.llm",
                    reason_code="CONFIDENCE_TOO_LOW",
                    reason_detail=final_reasoning,
                    context={
                        "confidence": final_confidence,
                        "threshold": self.min_confidence_for_trade,
                        "llm_action": llm_result.action.value,
                    },
                )
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

        if final_action in (TradeAction.BUY, TradeAction.SELL, TradeAction.SCALP_BUY, TradeAction.SCALP_SELL):
            (
                final_action,
                final_confidence,
                final_reasoning,
                hold_reason_update,
                _,
            ) = self._apply_level_confirmation(
                final_action=final_action,
                final_confidence=final_confidence,
            final_reasoning=final_reasoning,
            indicators=rule_result.indicators,
        )
        if hold_reason_update:
            hold_reason = hold_reason_update

    def _evaluate_regime_filter(
        self,
        rag_result: RAGRetrievalResult,
    ) -> Tuple[bool, float, str, Dict[str, Any]]:
        """Evaluate the RAG regime filter and return gating decisions.
        
        Returns:
            (hard_block, penalty, reason, context)
        """
        cfg = self.config.get("rag_regime_filter", {})
        enabled = cfg.get("enabled", True)
        if not enabled:
            return False, 0.0, "", {}

        min_sample_for_hard_block = int(cfg.get("min_sample_for_hard_block", 30) or 0)
        soft_penalty = float(cfg.get("soft_penalty_when_below", 0.10) or 0.0)
        hard_block_when_below = bool(cfg.get("hard_block_when_below", False))
        min_similar_trades_cfg = int(cfg.get("min_similar_trades", self.config.get("min_similar_trades", 2)) or 0)

        base_min_win_rate = float(cfg.get("min_win_rate", self.config.get("min_weighted_win_rate", 0.45)) or 0.45)
        soft_floor = min(
            base_min_win_rate,
            float(
                cfg.get(
                    "min_weighted_win_rate_soft_floor",
                    self.config.get("min_weighted_win_rate_soft_floor", base_min_win_rate),
                )
                or base_min_win_rate
            ),
        )
        full_threshold_trades = max(
            int(
                cfg.get(
                    "min_similar_trades_for_full_threshold",
                    self.config.get("min_similar_trades_for_full_threshold", 0),
                )
                or 0
            ),
            0,
        )
        use_relaxed_threshold = rag_result.similar_trade_count < full_threshold_trades and soft_floor < base_min_win_rate
        effective_min_win_rate = soft_floor if use_relaxed_threshold else base_min_win_rate

        wins = sum(1 for t in rag_result.similar_trades if str(t.get("result", "")).upper() == "WIN")
        n = int(rag_result.similar_trade_count or 0)
        smoothed_win_rate = (wins + 1) / (n + 2)

        below_threshold = smoothed_win_rate < effective_min_win_rate or n < min_similar_trades_cfg
        hard_block = (
            below_threshold
            and hard_block_when_below
            and n >= min_sample_for_hard_block
            and smoothed_win_rate < effective_min_win_rate
        )

        reason = (
            f"Regime check: similar_trades={n} (min={min_similar_trades_cfg}), "
            f"smoothed_win_rate={smoothed_win_rate:.2f} (min={effective_min_win_rate:.2f}, "
            f"mode={'relaxed' if use_relaxed_threshold else 'strict'})"
        )
        if soft_penalty > 0 and below_threshold:
            penalty_pct = soft_penalty * 100 if soft_penalty <= 1 else soft_penalty
            reason = f"{reason}; applying regime penalty {penalty_pct:.1f}pts"

        context = {
            "similar_trades": n,
            "wins": wins,
            "smoothed_win_rate": smoothed_win_rate,
            "weighted_win_rate": rag_result.weighted_win_rate,
            "effective_min_win_rate": effective_min_win_rate,
            "min_similar_trades": min_similar_trades_cfg,
            "full_threshold_trades": full_threshold_trades,
            "threshold_mode": "relaxed" if use_relaxed_threshold else "strict",
            "penalty": soft_penalty,
            "hard_block": hard_block,
            "min_sample_for_hard_block": min_sample_for_hard_block,
        }
        return hard_block, soft_penalty if below_threshold else 0.0, reason, context
        
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
            hold_reason=hold_reason if final_action == TradeAction.HOLD else None,
        )
        
        if result.hold_reason:
            logger.info(
                "🚫 HOLD [{}] gate={} detail={}",
                result.hold_reason.reason_code,
                result.hold_reason.gate,
                result.hold_reason.reason_detail,
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
    embedding_builder: Optional[Any] = None,
    storage_manager: Optional[Any] = None,
) -> HybridRAGPipeline:
    """Factory function to create a HybridRAGPipeline.
    
    Args:
        config: Pipeline configuration
        llm_client: Optional Bedrock client
        embedding_builder: Optional shared embedding builder instance
        storage_manager: Optional shared RAG storage manager
        
    Returns:
        HybridRAGPipeline instance
    """
    # Try to import and initialize RAG components if not supplied
    if embedding_builder is None:
        try:
            from mytrader.rag.embedding_builder import create_embedding_builder
            embedding_builder = create_embedding_builder()
        except ImportError:
            logger.warning("Embedding builder not available")
    if storage_manager is None:
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
