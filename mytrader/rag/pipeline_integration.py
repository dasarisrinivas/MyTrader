"""Hybrid RAG Pipeline Integration for Live Trading Manager.

This module provides integration between the 3-layer hybrid RAG pipeline
and the live trading manager. It can be used as an alternative decision
engine that replaces or augments the existing signal generation.

Usage:
    from mytrader.rag.pipeline_integration import HybridPipelineIntegration
    
    # In LiveTradingManager
    self.hybrid_pipeline = HybridPipelineIntegration(settings)
    
    # In _process_trading_cycle
    result = await self.hybrid_pipeline.process(features, current_price)
    if result.should_trade:
        await self._place_order(result.signal, current_price, features)
"""
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from mytrader.rag.hybrid_rag_pipeline import (
    HybridRAGPipeline,
    TradeAction,
    HybridPipelineResult,
    create_hybrid_pipeline,
)
from mytrader.rag.rag_storage_manager import get_rag_storage
from mytrader.rag.embedding_builder import create_embedding_builder
from mytrader.rag.trade_logger import TradeLogger, get_trade_logger
from mytrader.rag.mistake_analyzer import MistakeAnalyzer, get_mistake_analyzer
from mytrader.rag.rag_daily_updater import RAGDailyUpdater, create_daily_updater


class HybridSignal:
    """Signal object compatible with existing trading manager."""
    
    def __init__(
        self,
        action: str,
        confidence: float,
        metadata: Dict[str, Any] = None,
    ):
        self.action = action
        self.confidence = confidence
        self.metadata = metadata or {}


class HybridPipelineIntegration:
    """Integrates the 3-layer hybrid RAG pipeline with live trading.
    
    This class:
    1. Wraps the HybridRAGPipeline
    2. Converts between existing feature format and pipeline format
    3. Logs trades automatically
    4. Analyzes mistakes on losing trades
    """
    
    def __init__(
        self,
        settings: Any,
        llm_client: Optional[Any] = None,
        enabled: bool = True,
    ):
        """Initialize the hybrid pipeline integration.
        
        Args:
            settings: Application settings
            llm_client: Bedrock client for LLM calls
            enabled: Whether to use hybrid pipeline
        """
        self.settings = settings
        self.enabled = enabled
        
        # Get hybrid config from settings if available
        hybrid_config = {}
        if hasattr(settings, 'hybrid'):
            hybrid_config = {
                "rule_engine": {
                    "atr_min": getattr(settings.hybrid, 'atr_min', 0.15),  # Lowered for low-vol markets
                    "atr_max": getattr(settings.hybrid, 'atr_max', 5.0),
                    "rsi_oversold": getattr(settings.hybrid, 'rsi_oversold', 30),
                    "rsi_overbought": getattr(settings.hybrid, 'rsi_overbought', 70),
                    "cooldown_minutes": getattr(settings.hybrid, 'cooldown_minutes', 15),
                    "signal_threshold": getattr(settings.hybrid, 'signal_threshold', 40),
                },
                "llm": {
                    "min_confidence": getattr(settings.hybrid, 'min_confidence', 60),
                },
                "min_confidence_for_trade": getattr(settings.hybrid, 'min_confidence_for_trade', 60),
            }
        
        # Initialize components
        self.storage = get_rag_storage()
        self.trade_logger = get_trade_logger()
        self.mistake_analyzer = get_mistake_analyzer()
        
        # Try to initialize embedding builder
        try:
            self.embedding_builder = create_embedding_builder()
        except Exception as e:
            logger.warning(f"Could not initialize embedding builder: {e}")
            self.embedding_builder = None
        
        # Initialize pipeline
        self.pipeline = create_hybrid_pipeline(
            config=hybrid_config,
            llm_client=llm_client,
        )
        
        # Initialize daily updater
        self.daily_updater = create_daily_updater(storage=self.storage)
        
        # Track current trade for logging
        self._current_trade_id: Optional[str] = None
        
        logger.info("HybridPipelineIntegration initialized")
    
    def convert_features_to_market_data(
        self,
        features,
        current_price: float,
    ) -> Dict[str, Any]:
        """Convert pandas features DataFrame to market data dict.
        
        Args:
            features: Features DataFrame with indicators
            current_price: Current price
            
        Returns:
            Market data dictionary for pipeline
        """
        if features.empty:
            return {"close": current_price, "price": current_price}
        
        row = features.iloc[-1]
        
        # Extract indicators with safe defaults
        market_data = {
            "price": current_price,
            "close": float(row.get("close", current_price)),
            "open": float(row.get("open", current_price)),
            "high": float(row.get("high", current_price)),
            "low": float(row.get("low", current_price)),
            
            # EMAs
            "ema_9": float(row.get("EMA_9", row.get("ema_9", current_price))),
            "ema_20": float(row.get("EMA_20", row.get("ema_20", current_price))),
            "ema_50": float(row.get("SMA_50", row.get("ema_50", current_price))),
            
            # Momentum
            "rsi": float(row.get("RSI_14", row.get("rsi", 50))),
            "macd_hist": float(row.get("MACD", row.get("macd_hist", 0))),
            
            # Volatility
            "atr": float(row.get("ATR_14", row.get("atr", 0))),
            "atr_20_avg": float(row.get("ATR_20_avg", row.get("atr", 1))),
            
            # Levels (may need to be set elsewhere)
            "pdh": float(row.get("PDH", row.get("pdh", 0))),
            "pdl": float(row.get("PDL", row.get("pdl", 0))),
            "weekly_high": float(row.get("weekly_high", 0)),
            "weekly_low": float(row.get("weekly_low", 0)),
            "pivot": float(row.get("pivot", 0)),
            
            # Volume
            "volume_ratio": float(row.get("volume_ratio", 1.0)),
            
            # Additional
            "volatility": float(row.get("volatility_5m", row.get("volatility", 0))),
        }
        
        return market_data
    
    def process_sync(
        self,
        features,
        current_price: float,
    ) -> Tuple[HybridSignal, HybridPipelineResult]:
        """Process features through the hybrid pipeline (synchronous).
        
        Args:
            features: Features DataFrame
            current_price: Current price
            
        Returns:
            Tuple of (HybridSignal, HybridPipelineResult)
        """
        if not self.enabled:
            return HybridSignal("HOLD", 0.0), None
        
        # Convert features to market data
        market_data = self.convert_features_to_market_data(features, current_price)
        
        # Process through pipeline
        result = self.pipeline.process(market_data)
        
        # Convert to signal format
        signal = HybridSignal(
            action=result.final_action.value if result.final_action != TradeAction.BLOCKED else "HOLD",
            confidence=result.final_confidence / 100,  # Convert to 0-1 range
            metadata={
                "hybrid_reasoning": result.final_reasoning,
                "rule_engine_score": result.rule_engine.score,
                "filters_passed": result.rule_engine.filters_passed,
                "filters_blocked": result.rule_engine.filters_blocked,
                "market_trend": result.rule_engine.market_trend,
                "volatility_regime": result.rule_engine.volatility_regime,
                "rag_docs_count": len(result.rag_retrieval.documents),
                "rag_similar_trades": result.rag_retrieval.similar_trade_count,
                "llm_confidence": result.llm_decision.confidence if result.llm_decision else None,
                "stop_loss_points": result.stop_loss,
                "take_profit_points": result.take_profit,
                "position_size_factor": result.position_size,
            }
        )
        
        logger.info(
            f"Hybrid Pipeline: {signal.action} "
            f"(conf={signal.confidence:.2f}, "
            f"trend={result.rule_engine.market_trend}, "
            f"vol={result.rule_engine.volatility_regime})"
        )
        
        return signal, result
    
    async def process(
        self,
        features,
        current_price: float,
    ) -> Tuple[HybridSignal, HybridPipelineResult]:
        """Process features through the hybrid pipeline (async wrapper).
        
        Args:
            features: Features DataFrame
            current_price: Current price
            
        Returns:
            Tuple of (HybridSignal, HybridPipelineResult)
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.process_sync,
            features,
            current_price,
        )
    
    def log_trade_entry(
        self,
        action: str,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        take_profit: float,
        market_data: Dict[str, Any],
        pipeline_result: Optional[HybridPipelineResult] = None,
    ) -> str:
        """Log a trade entry for RAG.
        
        Args:
            action: BUY or SELL
            entry_price: Entry price
            quantity: Position size
            stop_loss: Stop loss level
            take_profit: Take profit level
            market_data: Market data at entry
            pipeline_result: Pipeline result for context
            
        Returns:
            Trade ID
        """
        trade_id = self.trade_logger.log_entry(
            action=action,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            market_data=market_data,
            pipeline_result=pipeline_result,
        )
        
        self._current_trade_id = trade_id
        return trade_id
    
    def log_trade_exit(
        self,
        exit_price: float,
        exit_reason: str,
    ) -> Optional[Any]:
        """Log a trade exit and analyze if it was a loss.
        
        Args:
            exit_price: Exit price
            exit_reason: Reason for exit
            
        Returns:
            Trade record or None
        """
        if not self._current_trade_id:
            logger.warning("No current trade ID for exit logging")
            return None
        
        trade = self.trade_logger.log_exit(
            trade_id=self._current_trade_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
        )
        
        # Analyze if it was a loss
        if trade and trade.result == "LOSS":
            logger.info(f"Analyzing losing trade {trade.trade_id}")
            self.mistake_analyzer.save_mistake_note(trade)
        
        self._current_trade_id = None
        return trade
    
    def record_trade_for_cooldown(self) -> None:
        """Record that a trade was executed (for pipeline cooldown)."""
        self.pipeline.record_trade()
    
    async def run_daily_update(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the daily RAG update process.
        
        Args:
            market_data: End of day market data
            
        Returns:
            Update results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.daily_updater.run_daily_update,
            market_data,
            False,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline and RAG statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "enabled": self.enabled,
            "trade_logger_active_trades": len(self.trade_logger.get_active_trades()),
        }
        
        # Add storage stats
        if self.storage:
            trade_stats = self.storage.get_trade_stats(days=30)
            stats["rag_trade_stats"] = trade_stats
        
        # Add embedding stats
        if self.embedding_builder:
            stats["embedding_stats"] = self.embedding_builder.get_stats()
        
        # Add mistake summary
        if self.mistake_analyzer:
            stats["mistake_summary"] = self.mistake_analyzer.get_mistake_summary(days=7, min_trades=3)
        
        return stats


def create_hybrid_integration(
    settings: Any,
    llm_client: Optional[Any] = None,
) -> HybridPipelineIntegration:
    """Factory function to create HybridPipelineIntegration.
    
    Args:
        settings: Application settings
        llm_client: Optional Bedrock client
        
    Returns:
        HybridPipelineIntegration instance
    """
    # Check if hybrid is enabled in settings
    enabled = True
    if hasattr(settings, 'hybrid'):
        enabled = getattr(settings.hybrid, 'enabled', True)
    
    return HybridPipelineIntegration(
        settings=settings,
        llm_client=llm_client,
        enabled=enabled,
    )
