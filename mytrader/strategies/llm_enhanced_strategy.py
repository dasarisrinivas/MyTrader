"""LLM-enhanced trading strategy that combines traditional signals with AI intelligence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..utils.logger import logger
from .base import BaseStrategy, Signal
from .rsi_macd_sentiment import RsiMacdSentimentStrategy
from ..llm.bedrock_client import BedrockClient
from ..llm.data_schema import TradingContext
from ..llm.trade_advisor import TradeAdvisor


@dataclass
class LLMEnhancedStrategy(BaseStrategy):
    """Strategy that enhances traditional signals with LLM intelligence.
    
    This strategy wraps an existing strategy and enhances its signals
    with AWS Bedrock LLM recommendations.
    """
    
    name: str = "llm_enhanced"
    base_strategy: Optional[BaseStrategy] = None
    enable_llm: bool = True
    min_llm_confidence: float = 0.7
    llm_override_mode: bool = False
    
    def __post_init__(self):
        """Initialize with default base strategy if not provided."""
        if self.base_strategy is None:
            self.base_strategy = RsiMacdSentimentStrategy()
            logger.info("LLMEnhancedStrategy using default RsiMacdSentimentStrategy")
        
        # Initialize trade advisor
        try:
            bedrock_client = BedrockClient() if self.enable_llm else None
            self.trade_advisor = TradeAdvisor(
                bedrock_client=bedrock_client,
                min_confidence_threshold=self.min_llm_confidence,
                enable_llm=self.enable_llm,
                llm_override_mode=self.llm_override_mode,
            )
            logger.info(
                f"LLMEnhancedStrategy initialized (LLM {'enabled' if self.enable_llm else 'disabled'})"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize LLM components: {e}")
            self.enable_llm = False
            self.trade_advisor = TradeAdvisor(enable_llm=False)
    
    def _build_trading_context(self, features: pd.DataFrame) -> TradingContext:
        """Build trading context from features DataFrame."""
        latest = features.iloc[-1]
        
        return TradingContext(
            symbol=latest.get("symbol", "ES"),
            current_price=float(latest.get("close", 0.0)),
            timestamp=latest.name if hasattr(latest, "name") else pd.Timestamp.now(),
            rsi=float(latest.get("RSI_14", 50.0)),
            macd=float(latest.get("MACD_12_26_9", 0.0)),
            macd_signal=float(latest.get("MACDsignal_12_26_9", 0.0)),
            macd_hist=float(latest.get("MACDhist_12_26_9", 0.0)),
            atr=float(latest.get("ATR_14", 0.0)),
            adx=float(latest.get("ADX_14", 0.0)) if "ADX_14" in latest else None,
            bb_percent=float(latest.get("BB_percent", 0.5)) if "BB_percent" in latest else None,
            sentiment_score=float(latest.get("sentiment_score", 0.0)),
            sentiment_sources=None,  # Could be enhanced with source breakdown
            current_position=int(latest.get("position", 0)) if "position" in latest else 0,
            unrealized_pnl=float(latest.get("unrealized_pnl", 0.0)) if "unrealized_pnl" in latest else 0.0,
            portfolio_heat=float(latest.get("portfolio_heat", 0.0)) if "portfolio_heat" in latest else 0.0,
            daily_pnl=float(latest.get("daily_pnl", 0.0)) if "daily_pnl" in latest else 0.0,
            win_rate=float(latest.get("win_rate", 0.0)) if "win_rate" in latest else 0.0,
            market_regime=latest.get("market_regime"),
            volatility_regime=latest.get("volatility_regime"),
        )
    
    def generate(self, features: pd.DataFrame) -> Signal:
        """Generate trading signal enhanced with LLM intelligence.
        
        Args:
            features: DataFrame with technical indicators and market data
            
        Returns:
            Enhanced trading signal
        """
        if len(features) < 2:
            return Signal(action="HOLD", confidence=0.0, metadata={})
        
        # Get traditional signal from base strategy
        traditional_signal = self.base_strategy.generate(features)
        
        # If LLM disabled, return traditional signal
        if not self.enable_llm:
            return traditional_signal
        
        # Build trading context for LLM
        try:
            context = self._build_trading_context(features)
            
            # Get LLM-enhanced signal
            enhanced_signal, llm_rec = self.trade_advisor.enhance_signal(
                traditional_signal,
                context
            )
            
            # Add LLM metadata to signal
            if llm_rec:
                enhanced_signal.metadata.update({
                    "llm_recommendation": llm_rec.to_dict(),
                    "strategy": "llm_enhanced"
                })
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error in LLM enhancement: {e}", exc_info=True)
            # Fallback to traditional signal on error
            traditional_signal.metadata["llm_error"] = str(e)
            return traditional_signal
    
    def update_config(
        self,
        min_llm_confidence: Optional[float] = None,
        llm_override_mode: Optional[bool] = None,
        enable_llm: Optional[bool] = None,
    ):
        """Update strategy configuration at runtime.
        
        Args:
            min_llm_confidence: New confidence threshold
            llm_override_mode: New override mode setting
            enable_llm: Enable/disable LLM
        """
        if min_llm_confidence is not None:
            self.min_llm_confidence = min_llm_confidence
        
        if llm_override_mode is not None:
            self.llm_override_mode = llm_override_mode
        
        if enable_llm is not None:
            self.enable_llm = enable_llm
        
        # Update trade advisor
        self.trade_advisor.update_config(
            min_confidence_threshold=min_llm_confidence,
            llm_override_mode=llm_override_mode,
            enable_llm=enable_llm,
        )
