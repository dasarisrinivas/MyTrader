"""
RAG-Enhanced Signal Validator
Uses RAG to retrieve relevant trading rules and validate signals before execution.
"""

from typing import Dict, Optional, Tuple
from loguru import logger
import pandas as pd


class RAGSignalValidator:
    """
    Validates trading signals using RAG-retrieved trading rules and best practices.
    """
    
    def __init__(self, rag_engine=None, min_rag_score: float = 0.6):
        """
        Initialize RAG validator.
        
        Args:
            rag_engine: RAGEngine instance (optional, will check if available)
            min_rag_score: Minimum RAG relevance score to consider rules
        """
        self.rag_engine = rag_engine
        self.min_rag_score = min_rag_score
        self.enabled = rag_engine is not None
        
        if self.enabled:
            logger.info("✅ RAG Signal Validator initialized")
        else:
            logger.warning("⚠️  RAG Signal Validator disabled (no RAG engine)")
    
    def validate_signal(
        self,
        action: str,
        confidence: float,
        risk_params: Dict,
        market_context: Dict,
        df: pd.DataFrame
    ) -> Tuple[str, float, Optional[str]]:
        """
        Validate signal using RAG-retrieved trading rules.
        
        Args:
            action: Proposed action ("BUY", "SELL", "HOLD")
            confidence: Signal confidence (0-1)
            risk_params: Risk parameters (stop_loss, take_profit, atr)
            market_context: Market conditions (bias, volatility, etc.)
            df: Price data DataFrame
            
        Returns:
            (validated_action, adjusted_confidence, validation_reason)
        """
        if not self.enabled or action == "HOLD":
            return action, confidence, None
        
        # Build query for RAG
        query = self._build_validation_query(
            action, confidence, risk_params, market_context, df
        )
        
        # Retrieve relevant trading rules
        try:
            retrieved_docs = self.rag_engine.retrieve_context(
                query=query,
                top_k=2,
                score_threshold=self.min_rag_score
            )
            
            if not retrieved_docs:
                logger.debug("No relevant RAG rules found, allowing signal")
                return action, confidence, None
            
            # Analyze retrieved rules
            validation_result = self._analyze_rules(
                action, confidence, risk_params, market_context, retrieved_docs
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"RAG validation error: {e}")
            return action, confidence, None
    
    def _build_validation_query(
        self,
        action: str,
        confidence: float,
        risk_params: Dict,
        market_context: Dict,
        df: pd.DataFrame
    ) -> str:
        """Build query to retrieve relevant trading rules."""
        
        # Extract key metrics
        current_price = df['close'].iloc[-1]
        volatility = market_context.get('volatility', 'medium')
        bias = market_context.get('bias', 'neutral')
        atr = risk_params.get('atr', 0.0)
        
        # Calculate risk:reward
        if action == "BUY":
            stop_loss = risk_params.get('stop_loss_long', current_price - atr * 2)
            take_profit = risk_params.get('take_profit_long', current_price + atr * 3)
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
        else:  # SELL
            stop_loss = risk_params.get('stop_loss_short', current_price + atr * 2)
            take_profit = risk_params.get('take_profit_short', current_price - atr * 3)
            risk = abs(stop_loss - current_price)
            reward = abs(current_price - take_profit)
        
        risk_reward = reward / risk if risk > 0 else 0
        
        query = f"""
Trading Signal Validation:
- Action: {action}
- Confidence: {confidence:.2f}
- Market Bias: {bias}
- Volatility: {volatility}
- Risk:Reward Ratio: {risk_reward:.2f}
- ATR: {atr:.2f}

What are the key rules for {action} signals in {volatility} volatility {bias} markets?
When should we avoid taking trades? What are the risk management best practices?
"""
        return query.strip()
    
    def _analyze_rules(
        self,
        action: str,
        confidence: float,
        risk_params: Dict,
        market_context: Dict,
        retrieved_docs: list
    ) -> Tuple[str, float, Optional[str]]:
        """
        Analyze retrieved rules and adjust signal accordingly.
        
        Returns:
            (action, adjusted_confidence, reason)
        """
        rules_text = "\n\n".join([doc for doc, score in retrieved_docs])
        rules_lower = rules_text.lower()
        
        # Extract key warnings/rules
        warnings = []
        confidence_penalty = 0.0
        
        # Check for risk warnings
        volatility = market_context.get('volatility', 'medium')
        
        # High volatility warnings
        if volatility == "high":
            if any(word in rules_lower for word in ["avoid high volatility", "reduce size in volatile", "wait for calm"]):
                warnings.append("High volatility - RAG suggests caution")
                confidence_penalty += 0.15
        
        # Trend direction warnings
        bias = market_context.get('bias', 'neutral')
        if action == "BUY" and bias == "bearish":
            if any(word in rules_lower for word in ["don't fight the trend", "trade with trend", "avoid counter-trend"]):
                warnings.append("Counter-trend trade - RAG suggests avoiding")
                confidence_penalty += 0.20
        elif action == "SELL" and bias == "bullish":
            if any(word in rules_lower for word in ["don't fight the trend", "trade with trend", "avoid counter-trend"]):
                warnings.append("Counter-trend trade - RAG suggests avoiding")
                confidence_penalty += 0.20
        
        # Risk:reward check
        atr = risk_params.get('atr', 0.0)
        current_price = market_context.get('price', 0.0)
        
        if action == "BUY":
            stop_loss = risk_params.get('stop_loss_long', current_price - atr * 2)
            take_profit = risk_params.get('take_profit_long', current_price + atr * 3)
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
        else:
            stop_loss = risk_params.get('stop_loss_short', current_price + atr * 2)
            take_profit = risk_params.get('take_profit_short', current_price - atr * 3)
            risk = abs(stop_loss - current_price)
            reward = abs(current_price - take_profit)
        
        risk_reward = reward / risk if risk > 0 else 0
        
        if risk_reward < 1.5:
            if any(word in rules_lower for word in ["minimum risk reward", "2:1 ratio", "risk:reward"]):
                warnings.append(f"Poor risk:reward ({risk_reward:.2f}) - RAG suggests 2:1 minimum")
                confidence_penalty += 0.10
        
        # Apply confidence adjustment
        adjusted_confidence = max(0.0, confidence - confidence_penalty)
        
        # Determine final action
        if adjusted_confidence < 0.5:
            final_action = "HOLD"
            reason = f"RAG validation REJECTED signal: {', '.join(warnings)}"
            logger.warning(f"🚫 {reason}")
        elif warnings:
            final_action = action
            reason = f"RAG warnings: {', '.join(warnings)}"
            logger.info(f"⚠️  {reason}")
        else:
            final_action = action
            reason = "RAG validation passed"
            logger.info(f"✅ {reason}")
        
        return final_action, adjusted_confidence, reason
    
    def get_trading_advice(self, query: str) -> Optional[str]:
        """
        Get trading advice for a specific query using RAG.
        
        Args:
            query: Trading question or scenario
            
        Returns:
            Advice text or None
        """
        if not self.enabled:
            return None
        
        try:
            result = self.rag_engine.generate_with_rag(
                query=query,
                top_k=3,
                score_threshold=self.min_rag_score,
                max_tokens=300
            )
            
            return result.get('response', '')
            
        except Exception as e:
            logger.error(f"RAG advice error: {e}")
            return None
