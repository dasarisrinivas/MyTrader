"""Trade advisor that combines traditional signals with LLM intelligence."""
from __future__ import annotations

from typing import Optional

from ..strategies.base import Signal
from ..utils.logger import logger
from .bedrock_client import BedrockClient
from .data_schema import TradingContext, TradeRecommendation


class TradeAdvisor:
    """Advisor that enhances trading decisions with LLM intelligence."""
    
    def __init__(
        self,
        bedrock_client: Optional[BedrockClient] = None,
        min_confidence_threshold: float = 0.7,
        enable_llm: bool = True,
        llm_override_mode: bool = False,
    ):
        """Initialize trade advisor.
        
        Args:
            bedrock_client: AWS Bedrock client (if None, creates default)
            min_confidence_threshold: Minimum confidence to execute trades
            enable_llm: Enable/disable LLM integration
            llm_override_mode: If True, LLM can override traditional signals
        """
        self.bedrock_client = bedrock_client
        self.min_confidence_threshold = min_confidence_threshold
        self.enable_llm = enable_llm
        self.llm_override_mode = llm_override_mode
        
        if self.enable_llm and self.bedrock_client is None:
            try:
                self.bedrock_client = BedrockClient()
                logger.info("TradeAdvisor initialized with default Bedrock client")
            except Exception as e:
                logger.warning(f"Failed to initialize Bedrock client: {e}")
                self.enable_llm = False
                logger.info("TradeAdvisor running without LLM enhancement")
    
    def enhance_signal(
        self,
        traditional_signal: Signal,
        context: TradingContext,
    ) -> tuple[Signal, Optional[TradeRecommendation]]:
        """Enhance traditional trading signal with LLM intelligence.
        
        Args:
            traditional_signal: Signal from traditional strategy
            context: Current market context
            
        Returns:
            Tuple of (enhanced_signal, llm_recommendation)
        """
        # If LLM disabled, return traditional signal
        if not self.enable_llm or self.bedrock_client is None:
            return traditional_signal, None
        
        try:
            # Get LLM recommendation
            llm_rec = self.bedrock_client.get_trade_recommendation(context)
            
            if llm_rec is None:
                logger.warning("LLM returned None, using traditional signal")
                return traditional_signal, None
            
            # Check confidence threshold
            if llm_rec.confidence < self.min_confidence_threshold:
                logger.info(
                    f"LLM confidence {llm_rec.confidence:.2f} below threshold "
                    f"{self.min_confidence_threshold:.2f}, downgrading to HOLD"
                )
                enhanced_signal = Signal(
                    action="HOLD",
                    confidence=llm_rec.confidence,
                    metadata={
                        **traditional_signal.metadata,
                        "llm_decision": llm_rec.trade_decision,
                        "llm_confidence": llm_rec.confidence,
                        "llm_reasoning": llm_rec.reasoning,
                        "reason": "LLM confidence below threshold"
                    }
                )
                return enhanced_signal, llm_rec
            
            # LLM override mode: LLM decision takes precedence
            if self.llm_override_mode:
                enhanced_signal = Signal(
                    action=llm_rec.trade_decision,
                    confidence=llm_rec.confidence,
                    metadata={
                        **traditional_signal.metadata,
                        "traditional_action": traditional_signal.action,
                        "traditional_confidence": traditional_signal.confidence,
                        "llm_decision": llm_rec.trade_decision,
                        "llm_confidence": llm_rec.confidence,
                        "llm_reasoning": llm_rec.reasoning,
                        "mode": "llm_override"
                    }
                )
                logger.info(
                    f"LLM override: {traditional_signal.action} -> {llm_rec.trade_decision} "
                    f"(confidence: {llm_rec.confidence:.2f})"
                )
                return enhanced_signal, llm_rec
            
            # Consensus mode: Both signals must agree
            if traditional_signal.action == llm_rec.trade_decision:
                # Signals agree - boost confidence
                enhanced_confidence = min(
                    1.0,
                    (traditional_signal.confidence + llm_rec.confidence) / 2 * 1.1
                )
                enhanced_signal = Signal(
                    action=traditional_signal.action,
                    confidence=enhanced_confidence,
                    metadata={
                        **traditional_signal.metadata,
                        "llm_decision": llm_rec.trade_decision,
                        "llm_confidence": llm_rec.confidence,
                        "llm_reasoning": llm_rec.reasoning,
                        "consensus": True,
                        "mode": "consensus"
                    }
                )
                logger.info(
                    f"Signals agree: {traditional_signal.action} "
                    f"(confidence boosted to {enhanced_confidence:.2f})"
                )
                return enhanced_signal, llm_rec
            
            elif llm_rec.trade_decision == "HOLD":
                # LLM suggests caution - downgrade to HOLD
                enhanced_signal = Signal(
                    action="HOLD",
                    confidence=llm_rec.confidence,
                    metadata={
                        **traditional_signal.metadata,
                        "traditional_action": traditional_signal.action,
                        "llm_decision": llm_rec.trade_decision,
                        "llm_confidence": llm_rec.confidence,
                        "llm_reasoning": llm_rec.reasoning,
                        "reason": "LLM suggests caution",
                        "mode": "consensus"
                    }
                )
                logger.info(f"LLM suggests caution: {traditional_signal.action} -> HOLD")
                return enhanced_signal, llm_rec
            
            else:
                # Conflicting signals - be conservative
                logger.warning(
                    f"Signal conflict: Traditional={traditional_signal.action}, "
                    f"LLM={llm_rec.trade_decision}. Defaulting to HOLD."
                )
                enhanced_signal = Signal(
                    action="HOLD",
                    confidence=0.5,
                    metadata={
                        **traditional_signal.metadata,
                        "traditional_action": traditional_signal.action,
                        "traditional_confidence": traditional_signal.confidence,
                        "llm_decision": llm_rec.trade_decision,
                        "llm_confidence": llm_rec.confidence,
                        "llm_reasoning": llm_rec.reasoning,
                        "reason": "Signal conflict - conservative hold",
                        "mode": "consensus"
                    }
                )
                return enhanced_signal, llm_rec
            
        except Exception as e:
            logger.error(f"Error in LLM enhancement: {e}", exc_info=True)
            return traditional_signal, None
    
    def update_config(
        self,
        min_confidence_threshold: Optional[float] = None,
        llm_override_mode: Optional[bool] = None,
        enable_llm: Optional[bool] = None,
    ):
        """Update advisor configuration at runtime.
        
        Args:
            min_confidence_threshold: New confidence threshold
            llm_override_mode: New override mode setting
            enable_llm: Enable/disable LLM
        """
        if min_confidence_threshold is not None:
            self.min_confidence_threshold = min_confidence_threshold
            logger.info(f"Updated confidence threshold to {min_confidence_threshold}")
        
        if llm_override_mode is not None:
            self.llm_override_mode = llm_override_mode
            logger.info(f"Updated LLM override mode to {llm_override_mode}")
        
        if enable_llm is not None:
            self.enable_llm = enable_llm
            logger.info(f"LLM enhancement {'enabled' if enable_llm else 'disabled'}")
