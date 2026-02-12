"""RAG-Enhanced Trade Advisor that retrieves trading knowledge before making decisions."""
from __future__ import annotations

import time
from typing import Optional

from ..strategies.base import Signal
from ..utils.logger import logger
from .bedrock_client import BedrockClient
from .data_schema import TradingContext, TradeRecommendation
from .rag_engine import RAGEngine


class RAGEnhancedTradeAdvisor:
    """Trade advisor enhanced with RAG for knowledge-grounded decisions.
    
    This advisor retrieves relevant trading knowledge from a knowledge base
    before invoking the LLM for trade recommendations. The retrieved context
    helps the LLM make more informed, factual decisions based on established
    trading principles.
    """
    
    def __init__(
        self,
        bedrock_client: Optional[BedrockClient] = None,
        rag_engine: Optional[RAGEngine] = None,
        min_confidence_threshold: float = 0.7,
        enable_llm: bool = True,
        enable_rag: bool = True,
        llm_override_mode: bool = False,
        rag_top_k: int = 3,
        rag_score_threshold: float = 0.5,
        call_interval_seconds: int = 60,
    ):
        """Initialize RAG-enhanced trade advisor.
        
        Args:
            bedrock_client: AWS Bedrock client (if None, creates default)
            rag_engine: RAG engine instance (optional)
            min_confidence_threshold: Minimum confidence to execute trades
            enable_llm: Enable/disable LLM integration
            enable_rag: Enable/disable RAG retrieval
            llm_override_mode: If True, LLM can override traditional signals
            rag_top_k: Number of documents to retrieve
            rag_score_threshold: Minimum similarity score for retrieval
            call_interval_seconds: Minimum seconds between LLM calls (rate limiting)
        """
        self.bedrock_client = bedrock_client
        self.rag_engine = rag_engine
        self.min_confidence_threshold = min_confidence_threshold
        self.enable_llm = enable_llm
        self.enable_rag = enable_rag and rag_engine is not None
        self.llm_override_mode = llm_override_mode
        self.rag_top_k = rag_top_k
        self.rag_score_threshold = rag_score_threshold
        self.call_interval_seconds = call_interval_seconds
        
        # Rate limiting
        self._last_llm_call_time = 0.0
        self._last_recommendation: Optional[TradeRecommendation] = None
        
        # Initialize Bedrock client if not provided
        if self.enable_llm and self.bedrock_client is None:
            try:
                self.bedrock_client = BedrockClient()
                logger.info("RAGEnhancedTradeAdvisor initialized with default Bedrock client")
            except Exception as e:
                logger.warning(f"Failed to initialize Bedrock client: {e}")
                self.enable_llm = False
                logger.info("RAGEnhancedTradeAdvisor running without LLM enhancement")
        
        if self.enable_rag:
            logger.info("RAG enhancement ENABLED - retrieving knowledge before decisions")
        else:
            logger.info("RAG enhancement DISABLED - using direct LLM calls")
        
        logger.info(f"LLM call rate limit: minimum {call_interval_seconds}s between calls")
    
    def _build_context_query(self, context: TradingContext) -> str:
        """Build query for RAG retrieval based on trading context.
        
        Args:
            context: Current trading context
            
        Returns:
            Query string for knowledge base retrieval
        """
        # Build a descriptive query based on market conditions
        conditions = []
        
        # RSI conditions
        if context.rsi < 30:
            conditions.append("RSI oversold below 30")
        elif context.rsi > 70:
            conditions.append("RSI overbought above 70")
        
        # MACD conditions
        if context.macd > context.macd_signal:
            conditions.append("MACD bullish crossover")
        elif context.macd < context.macd_signal:
            conditions.append("MACD bearish crossover")
        
        # Sentiment
        if context.sentiment_score > 0.3:
            conditions.append("positive market sentiment")
        elif context.sentiment_score < -0.3:
            conditions.append("negative market sentiment")
        
        # Volatility
        if context.atr > 0:
            conditions.append(f"ATR volatility {context.atr:.2f}")
        
        # Market regime
        if context.market_regime:
            conditions.append(f"{context.market_regime} market")
        
        # Build query
        if conditions:
            query = f"Trading strategy for {context.symbol} with " + ", ".join(conditions[:3])
        else:
            query = f"Trading strategy for {context.symbol} with current market conditions"
        
        return query
    
    def _build_augmented_prompt(
        self,
        context: TradingContext,
        retrieved_knowledge: str
    ) -> str:
        """Build augmented prompt with retrieved knowledge.
        
        Args:
            context: Trading context
            retrieved_knowledge: Retrieved trading knowledge
            
        Returns:
            Augmented prompt string
        """
        prompt = f"""You are an expert trading advisor analyzing market conditions for {context.symbol}.

RELEVANT TRADING KNOWLEDGE:
{retrieved_knowledge}

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
Based on the trading knowledge provided above and the current market conditions, provide a trading recommendation. 
Apply the principles and strategies from the knowledge base to this specific situation.

Consider:
1. How do the retrieved trading principles apply to these indicators?
2. What do the RSI, MACD, and sentiment suggest based on the knowledge?
3. How should risk be managed given the ATR and current position?
4. Is the market regime appropriate for trading?

IMPORTANT: Extract overall market sentiment from your analysis.
- sentiment_score should be -1.0 (very bearish) to +1.0 (very bullish) to 0.0 (neutral)
- This should reflect YOUR interpretation considering all factors

Respond ONLY with valid JSON in this exact format:
{{
    "trade_decision": "BUY" or "SELL" or "HOLD",
    "confidence": 0.0 to 1.0,
    "sentiment_score": -1.0 to 1.0,
    "suggested_position_size": integer (1-4),
    "suggested_stop_loss": float or null,
    "suggested_take_profit": float or null,
    "reasoning": "Brief explanation referencing the trading knowledge",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risk_assessment": "Brief risk analysis based on the knowledge"
}}

JSON Response:"""
        
        return prompt
    
    def enhance_signal(
        self,
        traditional_signal: Signal,
        context: TradingContext,
    ) -> tuple[Signal, Optional[TradeRecommendation]]:
        """Enhance trading signal with RAG-augmented LLM intelligence.
        
        Args:
            traditional_signal: Signal from traditional strategy
            context: Current market context
            
        Returns:
            Tuple of (enhanced_signal, llm_recommendation)
        """
        # If LLM disabled, return traditional signal
        if not self.enable_llm or self.bedrock_client is None:
            return traditional_signal, None
        
        # Rate limiting: Check if enough time has passed since last LLM call
        current_time = time.time()
        time_since_last_call = current_time - self._last_llm_call_time
        
        if time_since_last_call < self.call_interval_seconds:
            wait_seconds = self.call_interval_seconds - time_since_last_call
            logger.info(
                f"⏳ Rate limit: {wait_seconds:.0f}s until next LLM call - using cached recommendation"
            )
            
            # Use cached recommendation if available
            if self._last_recommendation is not None:
                # Apply the cached recommendation logic
                return self._apply_recommendation(traditional_signal, self._last_recommendation, [])
            else:
                # No cached recommendation, return traditional signal
                logger.info("No cached recommendation available, using traditional signal")
                return traditional_signal, None
        
        try:
            # Step 1: Retrieve relevant trading knowledge (if RAG enabled)
            retrieved_docs = []
            retrieved_knowledge = ""
            
            if self.enable_rag and self.rag_engine:
                try:
                    # Build query based on market conditions
                    query = self._build_context_query(context)
                    logger.info(f"RAG Query: {query}")
                    
                    # Retrieve relevant documents
                    results = self.rag_engine.retrieve_context(
                        query=query,
                        top_k=self.rag_top_k,
                        score_threshold=self.rag_score_threshold
                    )
                    
                    if results:
                        retrieved_docs = [doc for doc, score in results]
                        # Format knowledge for prompt
                        knowledge_parts = []
                        for i, (doc, score) in enumerate(results, 1):
                            knowledge_parts.append(f"[Knowledge {i}] (relevance: {score:.2f})\n{doc}")
                        retrieved_knowledge = "\n\n".join(knowledge_parts)
                        
                        logger.info(f"Retrieved {len(results)} relevant documents for decision")
                    else:
                        logger.warning("No relevant knowledge retrieved, using standard prompt")
                        retrieved_knowledge = "No specific trading knowledge retrieved. Use general principles."
                    
                except Exception as rag_error:
                    logger.error(f"RAG retrieval error: {rag_error}")
                    retrieved_knowledge = "Error retrieving knowledge. Use general principles."
            
            # Step 2: Get LLM recommendation with or without RAG context
            if self.enable_rag and retrieved_knowledge:
                # Build augmented prompt with retrieved knowledge
                prompt = self._build_augmented_prompt(context, retrieved_knowledge)
                
                # Get LLM response with custom prompt
                response_text = self.bedrock_client.generate_text(prompt)
                
                # Parse response
                import json
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                
                if json_start >= 0 and json_end > json_start:
                    parsed = json.loads(response_text[json_start:json_end])
                    
                    # Build recommendation
                    llm_rec = TradeRecommendation(
                        trade_decision=parsed.get("trade_decision", "HOLD"),
                        confidence=float(parsed.get("confidence", 0.0)),
                        sentiment_score=float(parsed.get("sentiment_score", 0.0)),
                        suggested_position_size=int(parsed.get("suggested_position_size", 1)),
                        suggested_stop_loss=parsed.get("suggested_stop_loss"),
                        suggested_take_profit=parsed.get("suggested_take_profit"),
                        reasoning=parsed.get("reasoning", ""),
                        key_factors=parsed.get("key_factors", []),
                        risk_assessment=parsed.get("risk_assessment", ""),
                        model_name=self.bedrock_client.model_id,
                        raw_response=parsed,
                    )
                else:
                    logger.error("Failed to parse JSON from RAG-enhanced LLM response")
                    llm_rec = None
            else:
                # Use standard BedrockClient method
                llm_rec = self.bedrock_client.get_trade_recommendation(context)
            
            if llm_rec is None:
                logger.warning("LLM returned None, using traditional signal")
                return traditional_signal, None
            
            # Add RAG metadata to recommendation
            if retrieved_docs:
                if llm_rec.raw_response is None:
                    llm_rec.raw_response = {}
                llm_rec.raw_response["rag_documents"] = retrieved_docs
                llm_rec.raw_response["rag_enabled"] = True
            
            # Update rate limiting
            self._last_llm_call_time = time.time()
            self._last_recommendation = llm_rec
            
            # Step 3: Apply recommendation with consensus logic
            return self._apply_recommendation(traditional_signal, llm_rec, retrieved_docs)
            
        except Exception as e:
            logger.error(f"Error in RAG-enhanced LLM signal enhancement: {e}", exc_info=True)
            return traditional_signal, None
    
    def _apply_recommendation(
        self,
        traditional_signal: Signal,
        llm_rec: TradeRecommendation,
        retrieved_docs: list
    ) -> tuple[Signal, TradeRecommendation]:
        """Apply LLM recommendation with consensus logic.
        
        Args:
            traditional_signal: Signal from traditional strategy
            llm_rec: LLM recommendation
            retrieved_docs: List of retrieved RAG documents
            
        Returns:
            Tuple of (enhanced_signal, llm_recommendation)
        """
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
                    "llm_sentiment": llm_rec.sentiment_score,
                    "llm_reasoning": llm_rec.reasoning,
                    "rag_enabled": self.enable_rag,
                    "rag_docs_retrieved": len(retrieved_docs),
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
                    "llm_sentiment": llm_rec.sentiment_score,
                    "llm_reasoning": llm_rec.reasoning,
                    "rag_enabled": self.enable_rag,
                    "rag_docs_retrieved": len(retrieved_docs),
                    "mode": "llm_override"
                }
            )
            logger.info(
                f"LLM OVERRIDE: {llm_rec.trade_decision} (LLM conf: {llm_rec.confidence:.2f}, "
                f"Traditional: {traditional_signal.action}, RAG docs: {len(retrieved_docs)})"
            )
            return enhanced_signal, llm_rec
        
        # Consensus mode: LLM and traditional must agree
        if traditional_signal.action == llm_rec.trade_decision:
            # Agreement: boost confidence
            combined_confidence = (traditional_signal.confidence + llm_rec.confidence) / 2
            enhanced_signal = Signal(
                action=traditional_signal.action,
                confidence=combined_confidence,
                metadata={
                    **traditional_signal.metadata,
                    "llm_decision": llm_rec.trade_decision,
                    "llm_confidence": llm_rec.confidence,
                    "llm_sentiment": llm_rec.sentiment_score,
                    "llm_reasoning": llm_rec.reasoning,
                    "rag_enabled": self.enable_rag,
                    "rag_docs_retrieved": len(retrieved_docs),
                    "mode": "consensus_agreement"
                }
            )
            logger.info(
                f"CONSENSUS: {traditional_signal.action} (combined conf: {combined_confidence:.2f}, "
                f"RAG docs: {len(retrieved_docs)})"
            )
        else:
            # Disagreement: downgrade to HOLD
            enhanced_signal = Signal(
                action="HOLD",
                confidence=min(traditional_signal.confidence, llm_rec.confidence),
                metadata={
                    **traditional_signal.metadata,
                    "traditional_action": traditional_signal.action,
                    "traditional_confidence": traditional_signal.confidence,
                    "llm_decision": llm_rec.trade_decision,
                    "llm_confidence": llm_rec.confidence,
                    "llm_sentiment": llm_rec.sentiment_score,
                    "llm_reasoning": llm_rec.reasoning,
                    "rag_enabled": self.enable_rag,
                    "rag_docs_retrieved": len(retrieved_docs),
                    "mode": "consensus_disagree",
                    "reason": "Traditional and LLM signals disagree"
                }
            )
            logger.info(
                f"DISAGREEMENT: HOLD (Traditional: {traditional_signal.action}, "
                f"LLM: {llm_rec.trade_decision}, RAG docs: {len(retrieved_docs)})"
            )
        
        return enhanced_signal, llm_rec
    
    def update_config(
        self,
        min_confidence_threshold: Optional[float] = None,
        llm_override_mode: Optional[bool] = None,
        enable_llm: Optional[bool] = None,
        enable_rag: Optional[bool] = None,
        rag_top_k: Optional[int] = None,
        rag_score_threshold: Optional[float] = None,
    ):
        """Update advisor configuration at runtime.
        
        Args:
            min_confidence_threshold: New confidence threshold
            llm_override_mode: New override mode setting
            enable_llm: Enable/disable LLM
            enable_rag: Enable/disable RAG
            rag_top_k: Number of documents to retrieve
            rag_score_threshold: Minimum similarity score
        """
        if min_confidence_threshold is not None:
            self.min_confidence_threshold = min_confidence_threshold
        
        if llm_override_mode is not None:
            self.llm_override_mode = llm_override_mode
        
        if enable_llm is not None:
            self.enable_llm = enable_llm
        
        if enable_rag is not None:
            self.enable_rag = enable_rag and self.rag_engine is not None
        
        if rag_top_k is not None:
            self.rag_top_k = rag_top_k
        
        if rag_score_threshold is not None:
            self.rag_score_threshold = rag_score_threshold
        
        logger.info(f"RAGEnhancedTradeAdvisor config updated: RAG={self.enable_rag}, LLM={self.enable_llm}")
