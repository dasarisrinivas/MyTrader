"""Hybrid RAG Pipeline - 3-Layer Decision System (Rules → RAG → LLM).

This is the core trading decision pipeline that combines:
1. LAYER 1: Rule Engine (deterministic filters - always on)
2. LAYER 2: RAG Retrieval (similar trades and docs - only on signal)
3. LAYER 3: LLM Decision (final judgment - only on signal)

The pipeline ensures safe, explainable, and context-aware trading decisions.
Uses CST (Central Standard Time) for all timestamps.

Session-aware trading (24h ES/MES):
- RTH (8:30-15:00 CT): Standard parameters
- Evening (17:00-23:00 CT): Higher thresholds, wider stops
- Overnight (23:00-03:00 CT): Strictest thresholds
- Pre-Market (03:00-08:30 CT): Elevated thresholds
- Maintenance (16:00-17:00 CT): BLOCKED
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
from mytrader.utils.session_manager import get_session_manager, TradingSession

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
    def format_cst(dt):
        return dt.strftime("%Y-%m-%d %H:%M:%S")


class TradeAction(Enum):
    """Possible trade actions from the pipeline."""
    BUY = "BUY"
    SELL = "SELL"
    SCALP_BUY = "SCALP_BUY"      # Lower confidence buy for low-vol
    SCALP_SELL = "SCALP_SELL"    # Lower confidence sell for low-vol
    HOLD = "HOLD"
    BLOCKED = "BLOCK"


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
    
    # Daily Trend Confirmation Gate (Feb 2026)
    # Requires 2 of 3: VWAP slope positive on 5m, EMA21 > EMA50 on 5m/15m, ADX > 20
    daily_trend_confirmed: bool = True  # Default True to not break existing behavior
    daily_trend_conditions_met: int = 0  # Count of conditions met (0-3)
    daily_trend_details: Dict[str, Any] = field(default_factory=dict)  # Details for debugging

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
        self.atr_max = config.get("atr_max", 20.0)  # Increased from 5.0 to allow normal volatility
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.pdh_proximity_pct = config.get("pdh_proximity_pct", 0.3)
        self.cooldown_minutes = config.get("cooldown_minutes", 15)
        self.chop_ema_spread_min_pct = float(config.get("chop_ema_spread_min_pct", 0.0005) or 0.0)
        self.oversold_extension_rsi_min = float(config.get("oversold_extension_rsi_min", 40.0))
        
        # Signal weights
        self.trend_weight = config.get("trend_weight", 30)
        self.momentum_weight = config.get("momentum_weight", 25)
        self.level_weight = config.get("level_weight", 25)
        self.volume_weight = config.get("volume_weight", 20)
        
        self.last_trade_time: Optional[datetime] = None
        
        logger.info("RuleEngine initialized")

    def _get_time_based_adjustments(self, current_time: datetime) -> Dict[str, float]:
        """Get adjustments based on time of day (CST)."""
        hour = current_time.hour
        
        # Evening session (6 PM - 11 PM CST) - futures are active
        if 18 <= hour <= 23:
            return {
                "min_confidence_adjustment": -0.05,  # Lower threshold
                "atr_min_adjustment": -0.05,        # Allow lower volatility
                "volume_requirement": 0.8,          # Relax volume requirements
                "range_requirement": 0.7,           # Allow tighter ranges
            }
        # Pre-market (3 AM - 8 AM CST)
        if 3 <= hour <= 8:
            return {
                "min_confidence_adjustment": 0.05,
                "atr_min_adjustment": 0.0,
                "volume_requirement": 1.2,
                "range_requirement": 1.1,
            }
        return {}
    
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
        
        # MANDATORY FIX DATA EXTRACTION
        # Robust key extraction handling various case formats (common in backtesting vs live)
        
        # ADX
        adx = market_data.get("adx", market_data.get("ADX", market_data.get("ADX_14", 0)))
        
        # Bollinger Bands
        bb_upper = market_data.get("bb_upper", market_data.get("BB_upper", market_data.get("BB_UPPER", 0)))
        bb_lower = market_data.get("bb_lower", market_data.get("BB_lower", market_data.get("BB_LOWER", 0)))
        
        # RSI
        rsi = market_data.get("rsi", market_data.get("RSI", market_data.get("RSI_14", 50)))
        
        # MACD
        macd_hist = market_data.get("macd_hist", market_data.get("MACD_hist", market_data.get("MACD_HIST", 0)))
        
        # ATR
        atr = market_data.get("atr", market_data.get("ATR", market_data.get("ATR_14", 0)))
        
        # EMAs
        ema_9 = market_data.get("ema_9", market_data.get("EMA_9", price))
        ema_20 = market_data.get("ema_20", market_data.get("EMA_20", price))
        ema_50 = market_data.get("ema_50", market_data.get("EMA_50", price))
        
        # High/Low pointers
        pdh = market_data.get("pdh", market_data.get("PDH", 0))
        pdl = market_data.get("pdl", market_data.get("PDL", 0)) 
        high = market_data.get("high", market_data.get("High", price))
        low = market_data.get("low", market_data.get("Low", price))
        
        # VWAP
        vwap = market_data.get("vwap", market_data.get("VWAP", 0))

        result.indicators = {
            "price": price,
            "close": close_price,
            "high": high,
            "low": low,
            "ema_9": ema_9,
            "ema_20": ema_20,
            "ema_50": ema_50,
            "rsi": rsi,
            "macd_hist": macd_hist,
            "atr": atr,
            "pdh": pdh,
            "pdl": pdl,
            "adx": adx,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
        }
        
        # ===== REGIME VALIDATION (MANDATORY FIX #1) =====
        is_trending = adx > 20  # Relaxed from 25 to match Harness
        
        ema_spread = abs(ema_9 - ema_50) / ema_50 if ema_50 > 0 else 0
        ema_spread_threshold = self.chop_ema_spread_min_pct
        if ema_spread_threshold > 0 and ema_spread < ema_spread_threshold:
            result.market_trend = "CHOP_RANGE"
            result.filters_blocked.append(
                "FORCE_HOLD: CHOP_RANGE (EMA spread "
                f"{ema_spread:.4f} < {ema_spread_threshold * 100:.2f}%)"
            )

        # ===== ENHANCED TREND DETECTION (Jan 2026) =====
        # Uses multi-bar confirmation, EMA_50 as anchor, and sentiment integration
        # to avoid whipsaw from reactive 1-minute candle trend flipping
        
        # Get external higher-timeframe trend if available (from signal processor's 5m builder)
        htf_trend = market_data.get("htf_trend", market_data.get("5m_trend", market_data.get("trend_5m", "")))
        sentiment_bias = market_data.get("sentiment_bias", "NEUTRAL")  # BULLISH, BEARISH, NEUTRAL
        sentiment_score = market_data.get("sentiment_score", 0.0)  # -1.0 to 1.0
        
        # Calculate EMA relationships
        ema_diff_pct = (ema_9 - ema_20) / ema_20 * 100 if ema_20 > 0 else 0
        ema_20_vs_50_pct = (ema_20 - ema_50) / ema_50 * 100 if ema_50 > 0 else 0
        price_vs_ema20_pct = (price - ema_20) / ema_20 * 100 if ema_20 > 0 else 0
        price_vs_ema50_pct = (price - ema_50) / ema_50 * 100 if ema_50 > 0 else 0
        
        # Use MACD histogram for momentum confirmation (multi-bar smoothed)
        macd_bullish = macd_hist > 0
        macd_bearish = macd_hist < 0
        
        # RSI context for trend validation
        rsi_bullish = rsi > 50
        rsi_bearish = rsi < 50
        rsi_extreme_bullish = rsi > 60
        rsi_extreme_bearish = rsi < 40
        
        # --- Multi-factor trend scoring ---
        # Score: positive = bullish, negative = bearish
        trend_score = 0.0
        trend_factors = []
        
        # Factor 1: EMA alignment (most weight - 40%)
        if price > ema_9 > ema_20 > ema_50:
            trend_score += 40
            trend_factors.append("EMA_STACK_UP")
        elif price < ema_9 < ema_20 < ema_50:
            trend_score -= 40
            trend_factors.append("EMA_STACK_DOWN")
        elif price > ema_20 and ema_diff_pct > 0:
            trend_score += 20
            trend_factors.append("EMA_BIAS_UP")
        elif price < ema_20 and ema_diff_pct < 0:
            trend_score -= 20
            trend_factors.append("EMA_BIAS_DOWN")
        
        # Factor 2: EMA_50 anchor position (20%)
        if price_vs_ema50_pct > 0.1:
            trend_score += 20
            trend_factors.append("ABOVE_EMA50")
        elif price_vs_ema50_pct < -0.1:
            trend_score -= 20
            trend_factors.append("BELOW_EMA50")
        
        # Factor 3: MACD momentum (15%)
        if macd_bullish:
            trend_score += 15
            trend_factors.append("MACD_POS")
        elif macd_bearish:
            trend_score -= 15
            trend_factors.append("MACD_NEG")
        
        # Factor 4: RSI position (10%)
        if rsi_extreme_bullish:
            trend_score += 10
            trend_factors.append("RSI_STRONG")
        elif rsi_extreme_bearish:
            trend_score -= 10
            trend_factors.append("RSI_WEAK")
        elif rsi_bullish:
            trend_score += 5
        elif rsi_bearish:
            trend_score -= 5
        
        # Factor 5: Higher timeframe confirmation (10%)
        if htf_trend in ("UPTREND", "MICRO_UP", "WEAK_UP"):
            trend_score += 10
            trend_factors.append("HTF_UP")
        elif htf_trend in ("DOWNTREND", "MICRO_DOWN", "WEAK_DOWN"):
            trend_score -= 10
            trend_factors.append("HTF_DOWN")
        
        # Factor 6: Sentiment bias (5%)
        if sentiment_bias == "BULLISH" or sentiment_score > 0.3:
            trend_score += 5
            trend_factors.append("SENTIMENT_BULL")
        elif sentiment_bias == "BEARISH" or sentiment_score < -0.3:
            trend_score -= 5
            trend_factors.append("SENTIMENT_BEAR")
        
        # --- Determine final trend based on score ---
        # Strong trend: >= 60 or <= -60 (at least 3 confirming factors)
        # Micro trend: 30-60 range
        # Weak trend: 10-30 range  
        # Chop/Range: -10 to 10
        
        # MANDATORY FIX #1 Applied: No trend unless ADX > 25
        if result.market_trend == "CHOP_RANGE":
             pass # Already set by chop filter
        elif trend_score >= 60 and is_trending:
            result.market_trend = "UPTREND"
        elif trend_score <= -60 and is_trending:
            result.market_trend = "DOWNTREND"
        elif trend_score >= 30 and is_trending:
            result.market_trend = "MICRO_UP"
        elif trend_score <= -30 and is_trending:
            result.market_trend = "MICRO_DOWN"
        elif trend_score >= 10 and is_trending:
            result.market_trend = "WEAK_UP"
        elif trend_score <= -10 and is_trending:
            result.market_trend = "WEAK_DOWN"
        elif abs(ema_diff_pct) < 0.02 and abs(price_vs_ema20_pct) < 0.05:
            result.market_trend = "RANGE"
        else:
            result.market_trend = "CHOP"
        
        # Store trend analysis details for debugging/Telegram
        result.indicators["trend_score"] = trend_score
        result.indicators["trend_factors"] = trend_factors
        result.indicators["htf_trend"] = htf_trend
        result.indicators["sentiment_bias"] = sentiment_bias
        
        logger.debug(
            f"Trend Detection: score={trend_score:.1f} factors={trend_factors} "
            f"-> {result.market_trend} (HTF={htf_trend}, Sentiment={sentiment_bias})"
        )
        
        # Determine volatility regime (ATR-based + VX futures overlay)
        avg_atr = market_data.get("atr_20_avg", atr)
        atr_ratio = atr / avg_atr if avg_atr > 0 else 1
        
        # Base regime from ATR
        if atr_ratio > 1.3:
            result.volatility_regime = "HIGH"
        elif atr_ratio < 0.7:
            result.volatility_regime = "LOW"
        else:
            result.volatility_regime = "MEDIUM"
        
        # VX Futures overlay - if VX is elevated, treat volatility as HIGH
        # This catches market-wide fear even when MES ATR is normal
        vx_price = market_data.get("vx_price")
        if vx_price and vx_price >= 25:
            if result.volatility_regime != "HIGH":
                logger.info(f"📈 VX override: VX={vx_price:.1f}>=25, upgrading {result.volatility_regime} -> HIGH")
                result.volatility_regime = "HIGH"
                result.filters_warned.append(f"VX_ELEVATED ({vx_price:.1f})")
        elif vx_price and vx_price >= 20:
            if result.volatility_regime == "LOW":
                logger.info(f"📈 VX caution: VX={vx_price:.1f}>=20, upgrading LOW -> MEDIUM")
                result.volatility_regime = "MEDIUM"
                result.filters_warned.append(f"VX_CAUTIOUS ({vx_price:.1f})")
        
        result.indicators["vx_price"] = vx_price if vx_price else 0.0
        result.indicators["atr_ratio"] = atr_ratio
        
        # ===== HARD FILTERS (blockers) =====
        
        # SESSION GATE: Check if trading allowed for current session
        # Use simulated time if provided in market_data (for backtesting), else use current time
        if "time" in market_data:
             current_time = market_data["time"]
        elif "timestamp" in market_data:
             current_time = market_data["timestamp"]
        else:
             current_time = now_cst()

        session_mgr = get_session_manager(self.config)
        current_session = session_mgr.get_current_session(current_time)
        session_config = session_mgr.get_session_config(current_session)
        trading_allowed, session_reason = session_mgr.is_trading_allowed(current_time)
        
        # ===== MANDATORY FIX #3: Overnight Session Separation =====
        # Trend-following logic is mathematically incompatible with overnight microstructure.
        # However, checking 'market_trend' here blocks Mean Reversion strategies which occur precisely
        # during strong deviations (which register as trends).
        # We rely on Model 1 (Trend Pullback) explicitly checking for RTH.
        
        if current_session != TradingSession.RTH and current_session not in [TradingSession.MAINTENANCE, TradingSession.WEEKEND]:
             # Block if not at extremes (Enforcing Mean Reversion behavior)
             is_at_extreme = (price >= bb_upper) or (price <= bb_lower)
             # Allow if we are close to extreme (within 1 tick or so) or if Model 2 logic handles it.
             # Actually, simpler to just allow the Entry Models to decide.
             pass
             
             # if not is_at_extreme and bb_upper > 0:
             #    result.filters_blocked.append(f"BLOCKED: WAITING_FOR_EXTREME (Price {price:.2f} not outside BB {bb_lower:.2f}-{bb_upper:.2f})")
             #    result.signal = TradeAction.HOLD

        if not trading_allowed:
            result.filters_blocked.append(f"SESSION_BLOCK ({session_reason})")
            logger.warning("🚫 SESSION BLOCK: session=%s reason=%s", current_session.value, session_reason)
        else:
            result.filters_passed.append(f"SESSION_OK ({current_session.value})")
        
        # Store session info for downstream use
        result.indicators["current_session"] = current_session.value
        result.indicators["session_min_confidence"] = session_config.min_confidence
        
        # ATR filter - RELAXED for low-vol days
        time_adjustments = self._get_time_based_adjustments(current_time)
        atr_min_threshold = self.atr_min if result.volatility_regime != "LOW" else 0.05
        atr_min_threshold = max(0.0, atr_min_threshold + time_adjustments.get("atr_min_adjustment", 0.0))
        if atr < atr_min_threshold:
            result.filters_warned.append(f"VERY_LOW_ATR ({atr:.2f})")  # Warn but don't block
        elif atr > self.atr_max:
            result.filters_blocked.append(f"ATR_TOO_HIGH ({atr:.2f} > {self.atr_max})")
        else:
            result.filters_passed.append("ATR_OK")
        
        # Cooldown filter - SESSION-AWARE
        effective_cooldown = max(self.cooldown_minutes, session_config.cooldown_after_loss_minutes // 2)
        if self.last_trade_time:
            elapsed = (now_cst() - self.last_trade_time).total_seconds() / 60
            if elapsed < effective_cooldown:
                result.filters_blocked.append(f"COOLDOWN ({elapsed:.1f} < {effective_cooldown} min)")
        else:
            result.filters_passed.append("COOLDOWN_OK")
        
        # Time filter - Note session for logging
        hour = current_time.hour
        minute = current_time.minute
        
        # Market hours check (CST - ES futures trade Sun 5PM to Fri 4PM CST)
        if current_session == TradingSession.RTH:
            result.filters_passed.append("RTH_SESSION")
        elif current_session in [TradingSession.MAINTENANCE, TradingSession.WEEKEND]:
            result.filters_blocked.append(f"MARKET_CLOSED ({current_session.value})")
        else:
            result.filters_passed.append(f"ETH_SESSION ({current_session.value})")
        
        # JAN 2026 FIX: Warn on CHOP/RANGE but do not block (let Entry Models decide)
        if result.market_trend in ["CHOP", "RANGE"]:
            # Check if we have a specific entry model firing despite chop (e.g. Mean Reversion)
            # We don't block here anymore.
            # result.filters_blocked.append(f"CHOP_MARKET_BLOCK ({result.market_trend})") # DEPRECATED
            logger.debug("⚠️ CHOP MARKET WARNING: trend=%s - requiring specific setup", result.market_trend)
        
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
        
        # ===== DAILY TREND CONFIRMATION GATE (Feb 2026) =====
        # Validates higher-timeframe trend alignment before allowing BUY_CONTINUATION entries.
        # Requires at least 2 of 3 conditions to be true:
        # 1. VWAP slope positive on 5m timeframe
        # 2. EMA21 > EMA50 on 5m or 15m
        # 3. ADX > 20
        # If gate fails, BUY_CONTINUATION entries are blocked for the session.
        
        daily_trend_conditions = 0
        daily_trend_details = {}
        
        # Condition 1: VWAP slope positive on 5m timeframe
        # Get 5m VWAP slope from market_data (computed by IndicatorBuilder)
        vwap_slope_5m = market_data.get("vwap_slope_5m", market_data.get("5m_vwap_slope", 0.0))
        # If not available, try to estimate from VWAP vs prev VWAP
        if vwap_slope_5m == 0.0:
            prev_vwap = market_data.get("prev_vwap", market_data.get("vwap_prev", vwap))
            if prev_vwap > 0 and vwap > 0:
                vwap_slope_5m = (vwap - prev_vwap) / prev_vwap
        
        vwap_slope_positive = vwap_slope_5m > 0.0001  # Small positive threshold
        if vwap_slope_positive:
            daily_trend_conditions += 1
        daily_trend_details["vwap_slope_5m"] = vwap_slope_5m
        daily_trend_details["vwap_slope_positive"] = vwap_slope_positive
        
        # Condition 2: EMA21 > EMA50 on 5m or 15m
        # Check 5m EMAs
        ema21_5m = market_data.get("5m_EMA_21", market_data.get("ema21_5m", 0.0))
        ema50_5m = market_data.get("5m_EMA_50", market_data.get("ema50_5m", 0.0))
        # Check 15m EMAs
        ema21_15m = market_data.get("15m_EMA_21", market_data.get("ema21_15m", 0.0))
        ema50_15m = market_data.get("15m_EMA_50", market_data.get("ema50_15m", 0.0))
        
        # Use 5m if available, else fall back to 15m, else use 1m EMAs
        ema_bullish_5m = ema21_5m > ema50_5m if (ema21_5m > 0 and ema50_5m > 0) else False
        ema_bullish_15m = ema21_15m > ema50_15m if (ema21_15m > 0 and ema50_15m > 0) else False
        
        # If neither 5m nor 15m EMAs available, use 1m as fallback
        if not (ema21_5m > 0 or ema21_15m > 0):
            ema_bullish_5m = ema_20 > ema_50 if ema_50 > 0 else False
        
        ema_alignment_bullish = ema_bullish_5m or ema_bullish_15m
        if ema_alignment_bullish:
            daily_trend_conditions += 1
        daily_trend_details["ema21_5m"] = ema21_5m
        daily_trend_details["ema50_5m"] = ema50_5m
        daily_trend_details["ema21_15m"] = ema21_15m
        daily_trend_details["ema50_15m"] = ema50_15m
        daily_trend_details["ema_bullish_5m"] = ema_bullish_5m
        daily_trend_details["ema_bullish_15m"] = ema_bullish_15m
        daily_trend_details["ema_alignment_bullish"] = ema_alignment_bullish
        
        # Condition 3: ADX > 20 (already computed above)
        adx_trending = adx > 20
        if adx_trending:
            daily_trend_conditions += 1
        daily_trend_details["adx"] = adx
        daily_trend_details["adx_trending"] = adx_trending
        
        # Gate passes if at least 2 of 3 conditions are true
        daily_trend_confirmed = daily_trend_conditions >= 2
        
        result.daily_trend_confirmed = daily_trend_confirmed
        result.daily_trend_conditions_met = daily_trend_conditions
        result.daily_trend_details = daily_trend_details
        
        # Log the gate status
        if daily_trend_confirmed:
            result.filters_passed.append(
                f"DAILY_TREND_CONFIRMED({daily_trend_conditions}/3: "
                f"VWAP={'+' if vwap_slope_positive else '-'}, "
                f"EMA={'+' if ema_alignment_bullish else '-'}, "
                f"ADX={'+' if adx_trending else '-'})"
            )
        else:
            result.filters_warned.append(
                f"DAILY_TREND_UNCONFIRMED({daily_trend_conditions}/3: "
                f"VWAP={'+' if vwap_slope_positive else '-'}, "
                f"EMA={'+' if ema_alignment_bullish else '-'}, "
                f"ADX={'+' if adx_trending else '-'})"
            )
            logger.debug(
                f"⚠️ DAILY_TREND_GATE: {daily_trend_conditions}/3 conditions met - "
                f"BUY_CONTINUATION may be blocked"
            )
        
        # ===== MANDATORY GUARDRAILS (Jan 16 2026) =====
        # Strict capital preservation rules
        trades_today = market_data.get("trades_today", 0)
        daily_losses = market_data.get("daily_losses", 0) 
        
        if trades_today >= 3:
            result.filters_blocked.append(f"MAX_TRADES_HIT ({trades_today} >= 3)")
            result.signal = TradeAction.HOLD
            return result
            
        if daily_losses >= 2:
            result.filters_blocked.append(f"LOSS_STREAK_HALT ({daily_losses} >= 2)")
            result.signal = TradeAction.HOLD
            return result
        
        # ===== SIGNAL GENERATION =====
        
        buy_score = 0
        sell_score = 0
        score_details = []  # For debugging
        
        # ===== POSITIVE EXPECTANCY ENTRY MODELS (Jan 16 2026) =====
        # Explicit, statistically justified entry logic for "Low-Frequency, Positive-Expectancy"
        
        vwap = market_data.get("vwap", price)
        # Ensure we have OHLC for candle analysis if available, else fallback to close
        candle_high = market_data.get("high", price)
        candle_low = market_data.get("low", price)
        
        # MODEL #1: RTH Trend Pullback Continuation
        # Purpose: Trade mean reversion within trend, not breakouts.
        # Expectation: 1-2 trades/session, 40-55% WR, Asymmetric R:R
        if current_session == TradingSession.RTH:
             # Logic: Moderate trend (ADX>20) - Relaxed from 25
             if adx > 20:
                 # LONG: Uptrend (EMA9>EMA50), Price touches or dips near EMA9
                 # Check if Low touched EMA9 zone (within 0.05%)
                 if (ema_9 > ema_50 and 
                     candle_low <= ema_9 * 1.0005 and
                     price >= ema_9 * 0.9995 and # Close near or above
                     40 <= rsi <= 65 and
                     price < bb_upper): 
                     
                     buy_score += 60 
                     score_details.append("ENTRY_MODEL_1_LONG:RTH_TREND_PULLBACK")
                     result.filters_passed.append("SETUP_RTH_PULLBACK_LONG")

                 # SHORT: Downtrend (EMA9<EMA50), High touches or pops near EMA9
                 elif (ema_9 < ema_50 and
                       candle_high >= ema_9 * 0.9995 and
                       price <= ema_9 * 1.0005 and
                       35 <= rsi <= 60 and
                       price > bb_lower):
                       
                       sell_score += 60
                       score_details.append("ENTRY_MODEL_1_SHORT:RTH_TREND_PULLBACK")
                       result.filters_passed.append("SETUP_RTH_PULLBACK_SHORT")

        # MODEL #2: Extreme Mean Reversion (Any Session, but careful in RTH)
        # Purpose: Exploit liquidity/volatility extremes
        # Exit: TP VWAP/EMA20, SL 0.15-0.2%
        
        # Check Reversion conditions regardless of session (Engine logic handles blockers)
        dist_to_vwap = abs(price - vwap) / vwap if vwap > 0 else 0
        
        # Debug Model 2 conditions
        if (price <= bb_lower * 1.01 or price >= bb_upper * 0.99) and (rsi < 40 or rsi > 60):
             print(f"DEBUG M2 Check: P={price:.2f} BBL={bb_lower:.2f} BBU={bb_upper:.2f} RSI={rsi:.1f} DistV={dist_to_vwap:.5f} SESS={current_session}")

        # LONG: Extreme Low - Relaxed BB check by 0.25% tolerance (was 0.1%)
        if (price <= bb_lower * 1.0025 and
             rsi < 35 and
             dist_to_vwap > 0.003):
             
             buy_score += 50 # slightly less than Trend Model to prefer trend if conflicting
             # Boost for Overnight
             if current_session != TradingSession.RTH:
                 buy_score += 15 
             
             score_details.append(f"ENTRY_MODEL_2_LONG:EXTREME_REVERSION(dist_vwap={dist_to_vwap:.4f})")
             result.filters_passed.append("SETUP_REVERSION_LONG")

        # SHORT: Extreme High - Relaxed BB check by 0.25% tolerance (was 0.1%)
        elif (price >= bb_upper * 0.9975 and
              rsi > 65 and
              dist_to_vwap > 0.003):

             sell_score += 50
             if current_session != TradingSession.RTH:
                 sell_score += 15

             score_details.append(f"ENTRY_MODEL_2_SHORT:EXTREME_REVERSION(dist_vwap={dist_to_vwap:.4f})")
             result.filters_passed.append("SETUP_REVERSION_SHORT")

        # Add trend factors to score_details for Telegram visibility
        trend_factors = result.indicators.get("trend_factors", [])
        trend_score_val = result.indicators.get("trend_score", 0)
        if trend_factors:
            score_details.append(f"TREND_SCORE:{trend_score_val:+.0f}({'+'.join(trend_factors[:3])})")
        
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
            # JAN 2026 FIX: Check for momentum breakout risk even in chop/range
            breakout_bullish = macd_hist > 0.5
            breakout_bearish = macd_hist < -0.5

            if rsi < 48:  # RELAXED from 45 - slight oversold
                pts = self.trend_weight * 0.4  # INCREASED from 0.3
                if breakout_bearish:
                    pts *= 0.2 # Penalty for buying a bearish breakdown
                    score_details.append(f"RANGE_RSI<48(BREAKDOWN_RISK):+{pts:.1f}")
                else:
                    score_details.append(f"RANGE_RSI<48:+{pts:.1f}")
                buy_score += pts
            elif rsi > 52:  # RELAXED from 55 - slight overbought
                pts = self.trend_weight * 0.4
                if breakout_bullish:
                    pts *= 0.2 # Penalty for selling a bullish breakout
                    score_details.append(f"RANGE_RSI>52(BREAKOUT_RISK):+{pts:.1f}")
                else:
                    score_details.append(f"RANGE_RSI>52:+{pts:.1f}")
                sell_score += pts
            else:
                score_details.append(f"RANGE_NEUTRAL(RSI={rsi:.1f})")
            result.filters_passed.append("RANGE_REVERSION")
        else:
            score_details.append(f"NO_TREND({result.market_trend})")
        
        # Momentum component (RSI + MACD) - MORE GRANULAR
        # JAN 2026 FIX: RSI scoring now respects acceptance vs exhaustion
        # High RSI in uptrend = CONTINUATION (don't short)
        # High RSI in downtrend = potential reversal (sell signal)
        
        # First determine if we're in an EMA stack up condition (acceptance)
        ema_stack_up = result.market_trend in ["UPTREND", "MICRO_UP", "WEAK_UP"]
        ema_stack_down = result.market_trend in ["DOWNTREND", "MICRO_DOWN", "WEAK_DOWN"]
        macd_positive = macd_hist > 0.2  # Strong positive MACD
        
        # ACCEPTANCE condition: bullish continuation setup
        is_bullish_acceptance = ema_stack_up and macd_positive
        
        # RSI: Award points based on trend context, not just absolute levels
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
            # JAN 2026 FIX: High RSI in bullish acceptance = CONTINUATION, not reversal!
            # Only add to sell_score if NOT in bullish acceptance
            if is_bullish_acceptance:
                # Bullish acceptance with high RSI = momentum is strong, don't short
                pts = self.momentum_weight * 0.2  # Small boost to buy for strong momentum
                buy_score += pts
                result.filters_warned.append("RSI_HIGH_IN_ACCEPTANCE")
                score_details.append(f"RSI_HIGH_ACCEPTANCE({rsi:.1f}):BUY+{pts:.1f}")
            else:
                # Not in acceptance - high RSI can trigger mean reversion sell
                pts = self.momentum_weight * 0.6
                sell_score += pts
                result.filters_passed.append("RSI_OVERBOUGHT")
                score_details.append(f"RSI_OVERBOUGHT({rsi:.1f}):+{pts:.1f}")
        elif rsi > 55:  # 55-60: ideal continuation zone for bulls
            if is_bullish_acceptance:
                # In bullish acceptance, RSI 55-60 is the sweet spot for continuation
                pts = self.momentum_weight * 0.3
                buy_score += pts
                score_details.append(f"RSI_BULLISH_ZONE({rsi:.1f}):BUY+{pts:.1f}")
            else:
                # Not in acceptance - somewhat overbought for mean reversion
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
                
                # JAN 2026 FIX: Check for momentum breakout risk
                # If MACD is strong positive, fading PDH is dangerous (Breakout Risk)
                breakout_risk = False
                if macd_hist > 0.5: # Strong momentum
                     breakout_risk = True
                     
                if rsi > 50:  # RSI supporting sell at PDH
                    pts = self.level_weight * 1.2  # BOOSTED for confluence
                    if breakout_risk:
                         pts *= 0.2 # Penalty for fading breakout momentum
                         score_details.append(f"NEAR_PDH+RSI_SUPPORT(BREAKOUT_RISK):+{pts:.1f}")
                    else:
                         score_details.append(f"NEAR_PDH+RSI_SUPPORT({pdh_dist_pct:.2f}%):+{pts:.1f}")
                else:
                    pts = self.level_weight * 0.7
                    if breakout_risk:
                         pts *= 0.2
                         score_details.append(f"NEAR_PDH(BREAKOUT_RISK):+{pts:.1f}")
                    else:
                         score_details.append(f"NEAR_PDH({pdh_dist_pct:.2f}%):+{pts:.1f}")
                sell_score += pts
                result.filters_warned.append("NEAR_PDH")
            else:
                score_details.append(f"NO_LEVEL(PDH:{pdh_dist_pct:.2f}%,PDL:{pdl_dist_pct:.2f}%)")
        else:
            score_details.append("NO_PDH_PDL")
        
        # Volume component
        volume_boost_threshold = 1.5 * float(time_adjustments.get("volume_requirement", 1.0))
        if volume_ratio > volume_boost_threshold:
            # High volume increases conviction
            buy_score *= 1.1
            sell_score *= 1.1
            result.filters_passed.append("HIGH_VOLUME")
            score_details.append(f"HIGH_VOL(x1.1 thr={volume_boost_threshold:.2f})")
        
        # ===== DAILY BIAS ADJUSTMENT =====
        # Penalize signals that go against the daily bias, boost signals that align
        if daily_bias == "BEARISH":
            # On bearish days, boost sell signals and penalize buy signals
            sell_score *= 1.3  # 30% boost to sell
            # JAN 2026 FIX: Don't penalize mean reversion buys if deeply oversold
            if rsi < 35:
                buy_score *= 0.95 # minimal penalty for contrarian play
                score_details.append(f"BEARISH_BUT_OVERSOLD(BUY*0.95)")
            else:
                buy_score *= 0.6   # 40% penalty to buy (don't buy falling knives)
                score_details.append(f"DAILY_BEARISH(SELL*1.3,BUY*0.6)")
        elif daily_bias == "BULLISH":
            # On bullish days, boost buy signals and penalize sell signals
            buy_score *= 1.3   # 30% boost to buy
            # JAN 2026 FIX: In bullish acceptance, NEVER boost sell signals
            # High RSI in bullish acceptance = continuation, not reversal
            if is_bullish_acceptance:
                sell_score *= 0.4  # Strong penalty - don't short acceptance
                score_details.append(f"BULLISH_ACCEPTANCE(BUY*1.3,SELL*0.4)")
            elif rsi > 65:
                # Not in acceptance but overbought - small penalty for contrarian
                sell_score *= 0.95
                score_details.append(f"BULLISH_BUT_OVERBOUGHT(SELL*0.95)")
            else:
                sell_score *= 0.6  # 40% penalty to sell (don't short strength)
                score_details.append(f"DAILY_BULLISH(BUY*1.3,SELL*0.6)")
        else:
            score_details.append("DAILY_NEUTRAL")
        
        # ===== JAN 2026 FIX: MEAN-REVERSION OVERRIDE =====
        # Only apply mean reversion when NOT in acceptance phase
        # Fading strength in acceptance = losing trade
        mean_reversion_override = False
        if pdh > 0 and pdl > 0:
            pdh_dist_pct = abs(price - pdh) / price * 100
            pdl_dist_pct = abs(price - pdl) / price * 100
            
            # Near PDL (support) with oversold RSI → FORCE BUY
            # This is safe - buying support with oversold RSI
            if pdl_dist_pct < 0.3 and rsi < 35:
                if sell_score > buy_score:
                    logger.warning(
                        f"🔄 MEAN-REVERSION OVERRIDE: Near PDL ({pdl_dist_pct:.2f}%) "
                        f"+ oversold RSI ({rsi:.1f}) → Flipping SELL→BUY"
                    )
                    # Swap scores to favor BUY
                    old_buy, old_sell = buy_score, sell_score
                    buy_score = max(old_sell, old_buy * 1.3)
                    sell_score = old_buy * 0.5
                    mean_reversion_override = True
                    score_details.append(f"MEAN_REV_OVERRIDE(PDL+RSI{rsi:.0f})")
            
            # Near PDH (resistance) with overbought RSI → ONLY force SELL if NOT in acceptance
            # JAN 2026 FIX: Never flip to SELL during bullish acceptance - could break out
            elif pdh_dist_pct < 0.3 and rsi > 65:
                if is_bullish_acceptance:
                    # In acceptance, high RSI near PDH = potential breakout, NOT reversal
                    logger.info(
                        f"📈 ACCEPTANCE PROTECTION: Near PDH but in bullish acceptance - "
                        f"NOT forcing mean reversion (could breakout)"
                    )
                    score_details.append(f"PDH_BUT_ACCEPTANCE(NO_FLIP)")
                elif buy_score > sell_score:
                    logger.warning(
                        f"🔄 MEAN-REVERSION OVERRIDE: Near PDH ({pdh_dist_pct:.2f}%) "
                        f"+ overbought RSI ({rsi:.1f}) → Flipping BUY→SELL"
                    )
                    # Swap scores to favor SELL
                    old_buy, old_sell = buy_score, sell_score
                    sell_score = max(old_buy, old_sell * 1.3)
                    buy_score = old_sell * 0.5
                    mean_reversion_override = True
                    score_details.append(f"MEAN_REV_OVERRIDE(PDH+RSI{rsi:.0f})")
        
        # Determine final signal - use scalp threshold in low-vol/range
        # JAN 8 2026 FIX: Raised default threshold from 40 to 55 to filter marginal trades
        # Analysis showed 47% confidence trades had high failure rate
        normal_threshold = self.config.get("signal_threshold", 55)
        threshold_adjustment = 1 + time_adjustments.get("min_confidence_adjustment", 0.0)
        normal_threshold = max(1, int(normal_threshold * threshold_adjustment))
        signal_threshold = scalp_threshold if is_scalp_mode else normal_threshold
        
        # Log score details for debugging
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
            
        # ===== MANDATORY FIX #2: Don't Sell the Hole =====
        if result.signal in [TradeAction.SELL, TradeAction.SCALP_SELL]:
            dist_to_ema20 = abs(price - ema_20) / ema_20 if ema_20 > 0 else 0
            if rsi < self.oversold_extension_rsi_min:
                 result.filters_blocked.append(
                     "BLOCKED: OVERSOLD_EXTENSION (RSI "
                     f"{rsi:.1f} < {self.oversold_extension_rsi_min:.1f})"
                 )
                 result.signal = TradeAction.HOLD
            elif price < bb_lower and bb_lower > 0:
                 result.filters_blocked.append(f"BLOCKED: OVERSOLD_EXTENSION (Price < BB Low)")
                 result.signal = TradeAction.HOLD
            elif dist_to_ema20 > 0.003:
                 result.filters_blocked.append(f"BLOCKED: OVERSOLD_EXTENSION (EMA Extension {dist_to_ema20:.4f} > 0.3%)")
                 result.signal = TradeAction.HOLD
        
        # Store breakdown for external access
        result.indicators["score_breakdown"] = score_details
        
        return result
    
    def record_trade(self) -> None:
        """Record that a trade was executed (for cooldown tracking)."""
        try:
            self.last_trade_time = now_cst()
        except Exception:
            # Defensive: avoid crashing on timestamp failures
            self.last_trade_time = datetime.now(timezone.utc)


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
                # Determine action for retrieval (map SCALP_X to X, default to BUY)
                action_str = rule_result.signal.value
                if "SELL" in action_str:
                    retrieval_action = "SELL"
                elif "BUY" in action_str:
                    retrieval_action = "BUY"
                else:
                    retrieval_action = "BUY"

                similar_trades = self.storage_manager.get_similar_trades(
                    action=retrieval_action,
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
        
        # Normalize confidence to 0-1 scale (detect 0-100 scale and convert)
        raw_min_conf = self.config.get("min_confidence", 0.6)
        self.min_confidence_threshold = raw_min_conf / 100.0 if raw_min_conf > 1.0 else raw_min_conf
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
                features_summary = {
                    "trend": rule_result.market_trend,
                    "rsi": rule_result.indicators.get("rsi", market_data.get("rsi", 50)),
                    "volatility": rule_result.volatility_regime,
                    "score": rule_result.score,
                }
                decision = self._generate_llm_signal(
                    prompt=prompt,
                    features_summary=features_summary,
                    rag_context=rag_result,
                )
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
    
    def _generate_llm_signal(
        self,
        prompt: str,
        features_summary: Dict[str, Any],
        rag_context: Any,
    ) -> LLMDecisionResult:
        """Call LLM and enforce directional bias when it responds HOLD.
        
        UPDATED Jan 2026: Raised minimum confidence from 25% to 45% for HOLD->action overrides.
        Also require RSI to be more extreme (>65 or <35) to prevent low-conviction trades.
        """
        try:
            llm_response = self._call_llm(prompt, len(prompt))
            decision = self._parse_response(llm_response)
            
            if decision.action == TradeAction.HOLD:
                trend = features_summary.get("trend", "UNKNOWN")
                rsi = features_summary.get("rsi", 50)
                
                # Jan 2026 fix: Block HOLD overrides in CHOP/RANGE markets
                if trend in ["CHOP", "RANGE", "UNKNOWN"]:
                    logger.info(f"🚫 Keeping HOLD in {trend} market - no override")
                    return decision
                
                # Jan 2026 fix: Require more extreme RSI (was 60/40, now 65/35)
                if trend == "DOWNTREND" and rsi > 65:
                    logger.info("🔄 Converting DOWNTREND HOLD to SELL signal")
                    decision.action = TradeAction.SELL
                    decision.confidence = max(decision.confidence, 45.0)  # RAISED from 25% to 45%
                    decision.reasoning = decision.reasoning or ""
                    decision.reasoning += f" | Downtrend with RSI {rsi}, converted from HOLD"
                elif trend == "UPTREND" and rsi < 35:
                    logger.info("🔄 Converting UPTREND HOLD to BUY signal")
                    decision.action = TradeAction.BUY
                    decision.confidence = max(decision.confidence, 45.0)  # RAISED from 25% to 45%
                    decision.reasoning = decision.reasoning or ""
                    decision.reasoning += f" | Uptrend with RSI {rsi}, converted from HOLD"
            
            return decision
        
        except Exception as e:
            logger.error(f"❌ LLM signal generation failed: {e}")
            return self._generate_fallback_technical_signal(features_summary)
    
    def _generate_fallback_technical_signal(self, features: Dict[str, Any]) -> LLMDecisionResult:
        """Generate a simple technical signal as fallback."""
        trend = features.get("trend", "UNKNOWN")
        rsi = features.get("rsi", 50)
        
        if trend == "DOWNTREND" and rsi > 65:
            return LLMDecisionResult(
                action=TradeAction.SELL,
                confidence=30.0,
                reasoning="Technical fallback: Downtrend + overbought RSI",
            )
        if trend == "UPTREND" and rsi < 35:
            return LLMDecisionResult(
                action=TradeAction.BUY,
                confidence=30.0,
                reasoning="Technical fallback: Uptrend + oversold RSI",
            )
        return LLMDecisionResult(
            action=TradeAction.HOLD,
            confidence=10.0,
            reasoning="Technical fallback: No clear setup",
        )

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
            action=(
                rule_result.signal
                if rule_result.signal
                in [
                    TradeAction.BUY,
                    TradeAction.SELL,
                    TradeAction.SCALP_BUY,
                    TradeAction.SCALP_SELL,
                ]
                else TradeAction.HOLD
            ),
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
        raw_min_llm = config.get("min_score_for_llm", 0.3)
        self.min_score_for_llm = raw_min_llm / 100.0 if raw_min_llm > 1.0 else raw_min_llm
        raw_min_trade = config.get("min_confidence_for_trade", 0.4)
        self.min_confidence_for_trade = raw_min_trade / 100.0 if raw_min_trade > 1.0 else raw_min_trade
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
        # Circuit breaker for consecutive blocks
        self._consecutive_blocks = 0
        self._max_consecutive_blocks = int(config.get("max_consecutive_blocks", 5))
        no_signal_cfg = config.get("no_signal_gate", {}) or {}
        self._no_signal_allow_weak = bool(no_signal_cfg.get("allow_weak_signals", False))
        self._no_signal_allow_chop = bool(no_signal_cfg.get("allow_chop_bias", False))
        
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
    
    def _is_exit_context(self, rule_result: RuleEngineResult, market_data: Dict[str, Any]) -> bool:
        """Return True when the signal would close an existing position."""
        qty = int(market_data.get("current_position_qty") or 0)
        if qty == 0:
            return False
        direction = self._action_direction(rule_result.signal)
        if qty > 0 and direction == "SELL":
            return True
        if qty < 0 and direction == "BUY":
            return True
        return False

    def _maybe_relax_no_signal(self, rule_result: RuleEngineResult) -> Optional[TradeAction]:
        """Allow weak directional signals in trending, active markets instead of hard HOLD."""
        if rule_result.filters_blocked or rule_result.signal != TradeAction.HOLD:
            return None

        if not self._no_signal_allow_weak:
            return None

        trend = rule_result.market_trend
        volatility = rule_result.volatility_regime

        if trend in ["UPTREND", "MICRO_UP", "WEAK_UP"] and volatility in ["MEDIUM", "HIGH"]:
            return TradeAction.SCALP_BUY
        if trend in ["DOWNTREND", "MICRO_DOWN", "WEAK_DOWN"] and volatility in ["MEDIUM", "HIGH"]:
            return TradeAction.SCALP_SELL

        if self._no_signal_allow_chop and trend in ["CHOP", "RANGE", "CHOP_RANGE"] and volatility in ["MEDIUM", "HIGH"]:
            if rule_result.daily_bias == "BULLISH":
                return TradeAction.SCALP_BUY
            if rule_result.daily_bias == "BEARISH":
                return TradeAction.SCALP_SELL

        return None

    def _get_dynamic_win_rate_threshold(self, similar_trades: List[Dict[str, Any]]) -> float:
        """Dynamically relax regime threshold for rough periods or off-peak trading."""
        base_threshold = 0.15
        now = datetime.now()

        if len(similar_trades) >= 5:
            recent_window = similar_trades[-10:] if len(similar_trades) >= 10 else similar_trades
            recent_losses = sum(1 for trade in recent_window if (trade or {}).get("pnl", 0) < 0)
            if recent_losses >= 8:
                logger.warning("🆘 Emergency mode: multiple recent losses, lowering regime threshold")
                return 0.05

        if now.weekday() >= 5:  # Weekend relaxation
            return base_threshold * 0.8
        if 18 <= now.hour <= 23:  # Evening session relaxation
            return base_threshold * 0.9

        return base_threshold

    def _generate_forced_exit_signal(
        self,
        current_position: int,
        market_data: Dict[str, Any],
    ) -> HybridPipelineResult:
        """Generate emergency exit signal that bypasses all filters."""
        current_price = market_data.get("close", market_data.get("price", 0))

        if current_position < 0:
            action = TradeAction.BUY
            reasoning = f"Emergency exit: Closing SHORT position of {current_position}"
        elif current_position > 0:
            action = TradeAction.SELL
            reasoning = f"Emergency exit: Closing LONG position of {current_position}"
        else:
            return self._create_hold_result(None, HoldReason.no_signal())

        rule_result = RuleEngineResult(
            signal=action,
            score=50.0,
            filters_passed=["EMERGENCY_EXIT"],
            market_trend=str(market_data.get("trend", "UNKNOWN")),
            volatility_regime=str(market_data.get("volatility", "MEDIUM")),
        )

        llm_result = LLMDecisionResult(
            action=action,
            confidence=60.0,
            reasoning=reasoning,
            suggested_stop_loss=0.0,
            suggested_take_profit=0.0,
        )

        logger.warning(f"🚨 Emergency position exit signal: {action}")

        return HybridPipelineResult(
            final_action=action,
            final_confidence=60.0,
            final_reasoning=reasoning,
            rule_engine=rule_result,
            rag_retrieval=RAGRetrievalResult(),
            llm_decision=llm_result,
            entry_price=current_price,
            stop_loss=0.0,
            take_profit=0.0,
            position_size=abs(current_position),
            timestamp=format_cst(now_cst()),
            processing_time_ms=0.0,
        )

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

        def _valid_level(level: Optional[float]) -> bool:
            return level is not None and level > 0

        def _is_near(level: Optional[float]) -> bool:
            return _valid_level(level) and price_ref > 0 and abs(price_ref - float(level)) / price_ref * 100 <= proximity_pct

        # Debug current level context to understand gating behavior
        self.debug_level_confirmation(price_ref, pdh, pdl)

        # EMERGENCY OVERRIDE: if price is already essentially at PDH/PDL, skip reclaim wait
        # Trigger when at least one level is available; override handler checks each independently.
        if any(_valid_level(level) for level in (pdl, pdh)):
            override_hit, override_reason = self._emergency_level_override(price_ref, pdl, pdh)
            if override_hit:
                self._reset_level_confirm_wait(direction)
                final_reasoning = f"{final_reasoning}; {override_reason}" if final_reasoning else override_reason
                return final_action, final_confidence, final_reasoning, None, override_reason

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

        # CRITICAL FIX: make PDL reclaim less sticky when price is effectively above
        if target_type == "PDL_RECLAIM" and _valid_level(pdl):
            if abs(price_ref - float(pdl)) <= 1.0 and price_ref >= float(pdl) - 0.5:
                logger.info(f"✅ PDL reclaim achieved: price={price_ref}, pdl={pdl}")
                self._reset_level_confirm_wait(direction)
                success_reason = "PDL_RECLAIM_SUCCESS"
                final_reasoning = f"{final_reasoning}; {success_reason}" if final_reasoning else success_reason
                return final_action, final_confidence, final_reasoning, None, success_reason
            # Allow more time before timing out when hugging PDL
            max_wait_candles = max(max_wait_candles, 10)

        if not target_type or target_level is None:
            # Not near a key level; no gating.
            self._reset_level_confirm_wait(direction)
            return final_action, final_confidence, final_reasoning, None, None

        if condition_met:
            self._reset_level_confirm_wait(direction)
            return final_action, final_confidence, final_reasoning, None, None

        pre_wait = self._level_confirm_wait.get(direction, 0)
        price_gap = (price_ref - float(pdl)) if _valid_level(pdl) else float("nan")
        should_pass = (price_ref >= float(pdl)) if _valid_level(pdl) else False
        logger.info(
            "🔍 Level Confirmation State:\n"
            f"   Current Price: {price_ref}\n"
            f"   PDL: {pdl}\n"
            f"   Price - PDL: {price_gap:+0.2f}\n"
            f"   Wait Count: {pre_wait}/{max_wait_candles}\n"
            f"   Should Pass: {should_pass}"
        )

        wait_count = pre_wait + 1
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
        rule_result: Optional[RuleEngineResult] = None,
    ) -> Tuple[bool, float, str, Dict[str, Any]]:
        """Evaluate the RAG regime filter and return gating decisions.
        
        Returns:
            (hard_block, penalty, reason, context)
        """
        # BYPASS: allow exits regardless of regime stats
        position_qty = int(market_data.get("current_position_qty", 0) or 0)
        if rule_result and position_qty != 0:
            action = rule_result.signal
            if (position_qty > 0 and action == TradeAction.SELL) or (
                position_qty < 0 and action == TradeAction.BUY
            ):
                return False, 0.0, "Position exit bypass - regime filter skipped", {
                    "position_qty": position_qty,
                    "action": action.value,
                }

        cfg = self.config.get("rag_regime_filter", {})
        enabled = cfg.get("enabled", True)
        if not enabled:
            return False, 0.0, "", {}

        strictness = str(cfg.get("strictness", cfg.get("mode", "relaxed")) or "relaxed").lower()
        min_sample_for_hard_block = int(cfg.get("min_sample_for_hard_block", 30) or 0)
        soft_penalty = float(cfg.get("soft_penalty_when_below", 0.10) or 0.0)
        hard_block_when_below = bool(cfg.get("hard_block_when_below", False))
        min_similar_trades_cfg = int(cfg.get("min_similar_trades", self.config.get("min_similar_trades", 2)) or 0)

        config_min_win_rate = float(cfg.get("min_win_rate", self.config.get("min_weighted_win_rate", 0.15)) or 0.15)
        dynamic_min = self._get_dynamic_win_rate_threshold(rag_result.similar_trades)
        base_min_win_rate = min(config_min_win_rate, dynamic_min)
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

    def process(
        self,
        market_data: Dict[str, Any],
        current_position: Optional[Any] = None,
        features: Optional[Dict[str, Any]] = None,
    ) -> HybridPipelineResult:
        """Process market data through all three layers (position-aware)."""
        start_time = time.time()
        hold_reason: Optional[HoldReason] = None
        regime_penalty = 0.0
        regime_reason = ""
        regime_context: Dict[str, Any] = {}

        if current_position is not None:
            market_data.setdefault(
                "current_position_qty",
                getattr(current_position, "quantity", 0) if current_position else 0,
            )
            market_data.setdefault(
                "current_position_avg_cost",
                getattr(current_position, "avg_cost", market_data.get("price")) if current_position else market_data.get("price"),
            )

        # Layer 1: Rule Engine (always runs)
        rule_result = self.rule_engine.evaluate(market_data)
        logger.debug(f"Layer 1 - Rule Engine: {rule_result.signal.value} ({rule_result.score:.1f})")

        # Circuit breaker for consecutive rule blocks
        if rule_result.filters_blocked:
            self._consecutive_blocks += 1
            if self._consecutive_blocks >= self._max_consecutive_blocks:
                logger.warning("🔧 Circuit breaker: {} consecutive blocks, relaxing rules", self._consecutive_blocks)
                rule_result.filters_warned.extend(["CIRCUIT_BREAKER_ACTIVATED"])
                rule_result.filters_blocked = []
                self._consecutive_blocks = 0
        else:
            self._consecutive_blocks = 0

        # Early exit if blocked or no actionable signal
        if not rule_result.should_proceed:
            relaxed_action = self._maybe_relax_no_signal(rule_result)
            if relaxed_action:
                logger.info(
                    f"🔓 Relaxing NO_SIGNAL gate for trending market ({rule_result.signal.value} -> {relaxed_action.value})"
                )
                rule_result.signal = relaxed_action
                rule_result.score = max(rule_result.score, 10)
                rule_result.filters_warned.append("NO_SIGNAL_RELAXED")
            else:
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
            rag_result, market_data, rule_result
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
            # Use config multipliers (Jan 2026 audit fix)
            # Get from one_minute config, with sensible defaults
            one_min_cfg = self.config.get("one_minute", {})
            stop_mult = float(one_min_cfg.get("stop_atr_multiplier", 3.0))  # Was 1.5
            tp_mult = float(one_min_cfg.get("take_profit_multiple", 2.0))
            
            # Calculate ATR-based stops
            stop_loss = atr * stop_mult
            take_profit = atr * stop_mult * tp_mult  # R:R based on stop
            
            # Enforce minimum stop distance (from risk_gate config)
            risk_gate_cfg = self.config.get("risk_gate", {})
            min_stop_points = float(risk_gate_cfg.get("min_stop_points", 4.0))
            if stop_loss < min_stop_points:
                logger.debug(
                    f"📏 Stop {stop_loss:.2f} below min {min_stop_points}, adjusting to {min_stop_points}"
                )
                stop_loss = min_stop_points
                take_profit = min_stop_points * tp_mult

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

    def debug_level_confirmation(self, current_price: float, pdh: Optional[float], pdl: Optional[float]) -> None:
        """Debug the level confirmation logic for visibility when gating trades."""
        try:
            pdh_val = float(pdh) if pdh is not None else None
            pdl_val = float(pdl) if pdl is not None else None
            logger.info("🔍 Level Confirmation Debug:")
            logger.info(f"   Current Price: {current_price}")
            logger.info(f"   PDH: {pdh_val} (diff: {((current_price - pdh_val) if pdh_val else float('nan')):+0.2f})")
            logger.info(f"   PDL: {pdl_val} (diff: {((current_price - pdl_val) if pdl_val else float('nan')):+0.2f})")
            logger.info(f"   Above PDL: {pdl_val is not None and current_price > pdl_val}")
            logger.info(f"   Below PDH: {pdh_val is not None and current_price < pdh_val}")
            if pdh_val is not None and pdl_val is not None:
                logger.info(f"   Range Size: {pdh_val - pdl_val:0.2f}")
        except Exception:
            # Avoid breaking pipeline due to debug logging issues
            logger.debug("Level confirmation debug logging failed", exc_info=True)

    def _emergency_pdl_override(self, current_price: float, pdl: float) -> Tuple[bool, str]:
        """Emergency override for PDL reclaim situations when price is already above PDL."""
        try:
            proximity = self._emergency_proximity_threshold(pdl)
            if current_price is not None and pdl is not None and abs(current_price - pdl) <= proximity:
                logger.warning("🚨 Emergency PDL override: price={} near pdl={}", current_price, pdl)
                return True, "EMERGENCY_NEAR_PDL"
        except Exception:
            pass
        return False, "No emergency override needed"

    def _emergency_level_override(
        self,
        current_price: float,
        pdl: Optional[float],
        pdh: Optional[float],
    ) -> Tuple[bool, str]:
        """Emergency override for obvious level confirmations."""
        try:
            for level, label in ((pdl, "PDL"), (pdh, "PDH")):
                if current_price is None or level is None:
                    continue
                proximity = self._emergency_proximity_threshold(level)
                if abs(current_price - level) <= proximity:
                    logger.warning(
                        "🚨 EMERGENCY LEVEL OVERRIDE: price={} very close to {}={}",
                        current_price,
                        label.lower(),
                        level,
                    )
                    return True, f"EMERGENCY_NEAR_{label}"
        except Exception:
            pass
        return False, "No emergency override"

    def _emergency_proximity_threshold(self, level: float) -> float:
        """Scale emergency proximity by instrument price; keep a sensible floor."""
        try:
            return max(0.25, abs(float(level)) * 0.0005)  # ~0.05% of price with 0.25pt floor
        except Exception:
            return 0.25

    def record_trade(self) -> None:
        """Record that a trade was made (for cooldown tracking)."""
        self.last_trade_time = now_cst()


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
