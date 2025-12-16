"""Trade Logger - Automatically saves full trade metadata to RAG storage.

This module captures and logs all trade details for:
- RAG retrieval of similar trades
- Performance analysis
- Mistake detection and learning
- Uses CST (Central Standard Time) for all timestamps
"""
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TYPE_CHECKING, List

from loguru import logger

from mytrader.rag.rag_storage_manager import (
    RAGStorageManager,
    TradeRecord,
    get_rag_storage,
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

if TYPE_CHECKING:
    from mytrader.llm.trade_logger import TradeLogger as MetricsLogger


def _to_utc_datetime(timestamp: str) -> datetime:
    """Convert ISO timestamps to timezone-aware UTC datetimes."""
    ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


class TradeLogger:
    """Logs trades with full context for RAG and analysis.
    
    Captures:
    - Entry/exit prices
    - All indicator values at entry
    - LLM reasoning and confidence
    - Market context (trend, volatility, levels)
    - Result and P&L
    """
    
    def __init__(self, storage: Optional[RAGStorageManager] = None):
        """Initialize trade logger.
        
        Args:
            storage: RAG storage manager instance
        """
        self.storage = storage or get_rag_storage()
        self.active_trades: Dict[str, TradeRecord] = {}
        self._metrics_logger: Optional["MetricsLogger"] = None
        self._metrics_logger_error_logged = False
        
        logger.info("TradeLogger initialized")
    
    def log_entry(
        self,
        action: str,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        take_profit: float,
        market_data: Dict[str, Any],
        pipeline_result: Optional[Any] = None,
    ) -> str:
        """Log a trade entry.
        
        Args:
            action: BUY or SELL
            entry_price: Entry price
            quantity: Position size
            stop_loss: Stop loss level
            take_profit: Take profit level
            market_data: Market data at entry
            pipeline_result: HybridPipelineResult from decision
            
        Returns:
            Trade ID for tracking
        """
        trade_id = str(uuid.uuid4())[:8]
        timestamp = now_cst().isoformat()
        
        # Determine time of day (CST hours)
        hour = now_cst().hour
        if hour < 10:  # Before 10 AM CST
            time_of_day = "OPEN"
        elif hour < 14:  # 10 AM - 2 PM CST
            time_of_day = "MIDDAY"
        else:  # After 2 PM CST
            time_of_day = "CLOSE"
        
        # Get day of week
        day_of_week = now_cst().strftime("%A").upper()
        
        # Extract indicator values
        rsi = market_data.get("rsi", 50)
        macd_hist = market_data.get("macd_hist", 0)
        ema_9 = market_data.get("ema_9", entry_price)
        ema_20 = market_data.get("ema_20", entry_price)
        atr = market_data.get("atr", 0)
        
        # Key levels
        pdh = market_data.get("pdh", 0)
        pdl = market_data.get("pdl", 0)
        weekly_high = market_data.get("weekly_high", 0)
        weekly_low = market_data.get("weekly_low", 0)
        pivot = market_data.get("pivot", 0)
        
        # Calculate level proximity
        safe_entry = entry_price if entry_price is not None else market_data.get("close", 0.0) or 0.0
        safe_pdh = pdh or 0.0
        safe_pdl = pdl or 0.0
        price_vs_pdh_pct = ((safe_entry - safe_pdh) / safe_pdh * 100) if safe_pdh > 0 else 0
        price_vs_pdl_pct = ((safe_entry - safe_pdl) / safe_pdl * 100) if safe_pdl > 0 else 0
        
        # Extract pipeline results if available
        llm_action = ""
        llm_confidence = 0
        llm_reasoning = ""
        rule_engine_signal = ""
        rule_engine_score = 0
        filters_passed = []
        filters_blocked = []
        market_trend = ""
        volatility_regime = ""
        rag_docs_used = []
        rag_similarity_scores = []
        
        if pipeline_result:
            # From LLM decision
            if pipeline_result.llm_decision:
                llm_action = pipeline_result.llm_decision.action.value
                llm_confidence = pipeline_result.llm_decision.confidence
                llm_reasoning = pipeline_result.llm_decision.reasoning
            
            # From rule engine
            rule_engine_signal = pipeline_result.rule_engine.signal.value
            rule_engine_score = pipeline_result.rule_engine.score
            filters_passed = pipeline_result.rule_engine.filters_passed
            filters_blocked = pipeline_result.rule_engine.filters_blocked
            market_trend = pipeline_result.rule_engine.market_trend
            volatility_regime = pipeline_result.rule_engine.volatility_regime
            
            # From RAG retrieval
            if pipeline_result.rag_retrieval.documents:
                rag_docs_used = [d[0] for d in pipeline_result.rag_retrieval.documents]
                rag_similarity_scores = [d[2] for d in pipeline_result.rag_retrieval.documents]
        
        # Create trade record
        trade = TradeRecord(
            trade_id=trade_id,
            timestamp=timestamp,
            action=action,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            
            # Indicators
            rsi=rsi,
            macd_hist=macd_hist,
            ema_9=ema_9,
            ema_20=ema_20,
            atr=atr,
            
            # Levels
            pdh=pdh,
            pdl=pdl,
            weekly_high=weekly_high,
            weekly_low=weekly_low,
            pivot=pivot,
            price_vs_pdh_pct=price_vs_pdh_pct,
            price_vs_pdl_pct=price_vs_pdl_pct,
            
            # LLM
            llm_action=llm_action,
            llm_confidence=llm_confidence,
            llm_reasoning=llm_reasoning,
            
            # RAG
            rag_docs_used=rag_docs_used,
            rag_similarity_scores=rag_similarity_scores,
            
            # Rule engine
            rule_engine_signal=rule_engine_signal,
            rule_engine_score=rule_engine_score,
            filters_passed=filters_passed,
            filters_blocked=filters_blocked,
            
            # Market context
            market_trend=market_trend,
            volatility_regime=volatility_regime,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
        )
        
        # Store as active trade
        self.active_trades[trade_id] = trade
        
        logger.info(
            f"Trade entry logged: {trade_id} - {action} {quantity} @ {entry_price:.2f} "
            f"(SL: {stop_loss:.2f}, TP: {take_profit:.2f})"
        )
        
        return trade_id
    
    def log_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
    ) -> Optional[TradeRecord]:
        """Log a trade exit and save to storage.
        
        Args:
            trade_id: Trade ID from log_entry
            exit_price: Exit price
            exit_reason: Reason for exit (TP_HIT, SL_HIT, MANUAL, TRAILING, etc.)
            
        Returns:
            Completed TradeRecord or None if trade not found
        """
        if trade_id not in self.active_trades:
            logger.warning(f"Trade {trade_id} not found in active trades")
            return None
        
        trade = self.active_trades[trade_id]
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        
        # Calculate P&L
        if trade.action == "BUY":
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity
        else:  # SELL
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity
        
        # Calculate P&L percentage
        trade.pnl_pct = (trade.pnl / trade.entry_price) * 100 if trade.entry_price > 0 else 0
        
        # Determine result
        if trade.pnl > 0:
            trade.result = "WIN"
        elif trade.pnl < 0:
            trade.result = "LOSS"
        else:
            trade.result = "BREAKEVEN"
        
        # Calculate duration
        try:
            entry_time = _to_utc_datetime(trade.timestamp)
            duration = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60
            trade.duration_minutes = duration
        except Exception:
            trade.duration_minutes = 0
        
        # Save to storage
        filepath = self.storage.save_trade(trade)
        
        # Remove from active trades
        del self.active_trades[trade_id]
        
        logger.info(
            f"Trade exit logged: {trade_id} - {trade.result} "
            f"P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%) "
            f"Duration: {trade.duration_minutes:.1f} min"
        )
        
        return trade
    
    def log_complete_trade(
        self,
        action: str,
        entry_price: float,
        exit_price: float,
        quantity: int,
        stop_loss: float,
        take_profit: float,
        exit_reason: str,
        duration_minutes: float,
        market_data: Dict[str, Any],
        pipeline_result: Optional[Any] = None,
    ) -> TradeRecord:
        """Log a complete trade (entry + exit) in one call.
        
        Useful for backtesting or when trade is already complete.
        
        Args:
            action: BUY or SELL
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position size
            stop_loss: Stop loss level
            take_profit: Take profit level
            exit_reason: Reason for exit
            duration_minutes: Trade duration
            market_data: Market data at entry
            pipeline_result: HybridPipelineResult from decision
            
        Returns:
            Completed TradeRecord
        """
        trade_id = self.log_entry(
            action=action,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            market_data=market_data,
            pipeline_result=pipeline_result,
        )
        
        # Override duration
        if trade_id in self.active_trades:
            self.active_trades[trade_id].duration_minutes = duration_minutes
        
        return self.log_exit(trade_id, exit_price, exit_reason)
    
    def get_active_trades(self) -> Dict[str, TradeRecord]:
        """Get all currently active (open) trades.
        
        Returns:
            Dictionary of trade_id -> TradeRecord
        """
        return self.active_trades.copy()
    
    def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        """Get a specific active trade.
        
        Args:
            trade_id: Trade ID
            
        Returns:
            TradeRecord or None if not found
        """
        return self.active_trades.get(trade_id)

    # ------------------------------------------------------------------
    # Structural metric persistence (backed by llm.trade_logger)
    # ------------------------------------------------------------------
    def _get_metrics_logger(self) -> Optional["MetricsLogger"]:
        """Lazy-load the SQLite-backed metrics logger."""
        if self._metrics_logger is not None:
            return self._metrics_logger
        if self._metrics_logger_error_logged:
            return None
        try:
            from mytrader.llm.trade_logger import TradeLogger as MetricsLogger  # Local import
            self._metrics_logger = MetricsLogger()
            logger.info("Metrics logger ready for structural persistence")
        except Exception as exc:
            logger.warning(f"Unable to initialize metrics logger: {exc}")
            self._metrics_logger_error_logged = True
            self._metrics_logger = None
        return self._metrics_logger

    def record_market_metrics(self, metrics: Dict[str, Any]) -> None:
        """Persist structural market metrics for downstream decision making."""
        metrics_logger = self._get_metrics_logger()
        if not metrics_logger:
            raise RuntimeError("Metrics logger unavailable")
        metrics_logger.record_market_metrics(metrics)

    def get_recent_market_metrics(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch stored structural metrics for historical context."""
        metrics_logger = self._get_metrics_logger()
        if not metrics_logger:
            raise RuntimeError("Metrics logger unavailable")
        return metrics_logger.get_recent_market_metrics(limit=limit)


# Singleton instance
_trade_logger: Optional[TradeLogger] = None


def get_trade_logger() -> TradeLogger:
    """Get the singleton trade logger instance.
    
    Returns:
        TradeLogger instance
    """
    global _trade_logger
    if _trade_logger is None:
        _trade_logger = TradeLogger()
    return _trade_logger
