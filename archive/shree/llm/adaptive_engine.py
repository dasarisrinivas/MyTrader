"""Adaptive learning engine for semi-autonomous trading."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import logger
from .bedrock_client import BedrockClient
from .performance_analyzer import DailyMetrics, PerformanceAnalyzer, TradePattern
from .prompt_templates import PromptTemplates


@dataclass
class StrategyAdjustment:
    """Represents a suggested strategy parameter adjustment."""
    
    parameter: str
    old_value: Any
    new_value: Any
    reasoning: str
    confidence: float
    risk_level: str
    timestamp: datetime
    approved: bool = False
    applied: bool = False
    performance_before: Optional[dict] = None
    performance_after: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "parameter": self.parameter,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "timestamp": self.timestamp.isoformat(),
            "approved": self.approved,
            "applied": self.applied,
            "performance_before": self.performance_before,
            "performance_after": self.performance_after,
        }


@dataclass
class SafetyConstraints:
    """Safety constraints for parameter adjustments."""
    
    # RSI thresholds
    rsi_buy_min: float = 20.0
    rsi_buy_max: float = 45.0
    rsi_sell_min: float = 55.0
    rsi_sell_max: float = 80.0
    
    # Sentiment thresholds
    sentiment_buy_min: float = -1.0
    sentiment_buy_max: float = 0.0
    sentiment_sell_min: float = 0.0
    sentiment_sell_max: float = 1.0
    sentiment_weight_min: float = 0.1
    sentiment_weight_max: float = 0.9
    
    # Risk management
    stop_loss_min: float = 10.0
    stop_loss_max: float = 50.0
    take_profit_min: float = 15.0
    take_profit_max: float = 100.0
    
    # Position sizing (never change these programmatically)
    max_position_size_limit: int = 5
    max_daily_loss_limit: float = 3000.0
    
    # LLM thresholds
    min_confidence_min: float = 0.5
    min_confidence_max: float = 0.9
    
    # Change limits (max % change per adjustment)
    max_parameter_change_pct: float = 0.20  # 20% max change
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "rsi_buy_min": self.rsi_buy_min,
            "rsi_buy_max": self.rsi_buy_max,
            "rsi_sell_min": self.rsi_sell_min,
            "rsi_sell_max": self.rsi_sell_max,
            "sentiment_buy_min": self.sentiment_buy_min,
            "sentiment_buy_max": self.sentiment_buy_max,
            "sentiment_sell_min": self.sentiment_sell_min,
            "sentiment_sell_max": self.sentiment_sell_max,
            "sentiment_weight_min": self.sentiment_weight_min,
            "sentiment_weight_max": self.sentiment_weight_max,
            "stop_loss_min": self.stop_loss_min,
            "stop_loss_max": self.stop_loss_max,
            "take_profit_min": self.take_profit_min,
            "take_profit_max": self.take_profit_max,
            "min_confidence_min": self.min_confidence_min,
            "min_confidence_max": self.min_confidence_max,
        }


class AdaptiveLearningEngine:
    """Engine for autonomous strategy learning and adaptation."""
    
    def __init__(
        self,
        bedrock_client: Optional[BedrockClient] = None,
        analyzer: Optional[PerformanceAnalyzer] = None,
        safety_constraints: Optional[SafetyConstraints] = None,
        auto_approve_threshold: float = 0.8,
        require_human_approval: bool = True,
    ):
        """Initialize adaptive learning engine.
        
        Args:
            bedrock_client: AWS Bedrock client for LLM calls
            analyzer: Performance analyzer instance
            safety_constraints: Safety constraints for adjustments
            auto_approve_threshold: Confidence threshold for auto-approval
            require_human_approval: Whether human approval is required
        """
        self.bedrock_client = bedrock_client or BedrockClient()
        self.analyzer = analyzer or PerformanceAnalyzer()
        self.safety_constraints = safety_constraints or SafetyConstraints()
        self.auto_approve_threshold = auto_approve_threshold
        self.require_human_approval = require_human_approval
        
        self.prompt_templates = PromptTemplates()
        
        logger.info("AdaptiveLearningEngine initialized")
    
    def analyze_daily_performance(
        self,
        date: Optional[str] = None
    ) -> Tuple[Optional[DailyMetrics], List[TradePattern], str]:
        """Analyze daily performance and generate insights.
        
        Args:
            date: Date to analyze (defaults to today)
            
        Returns:
            Tuple of (metrics, patterns, summary_text)
        """
        # Calculate metrics
        metrics = self.analyzer.calculate_daily_metrics(date)
        
        if metrics is None:
            logger.warning(f"No trading data found for {date or 'today'}")
            return None, [], "No trading data available for analysis."
        
        # Get historical context
        historical_metrics = self.analyzer.get_historical_metrics(days=7)
        
        # Identify patterns
        patterns = self.analyzer.identify_patterns(metrics, historical_metrics)
        
        # Generate human-readable summary
        summary_text = self.analyzer.generate_performance_summary(metrics, patterns)
        
        logger.info(f"Daily analysis complete: {metrics.total_trades} trades, ${metrics.net_pnl:.2f} P&L")
        
        return metrics, patterns, summary_text
    
    def generate_llm_summary(
        self,
        metrics: DailyMetrics,
        patterns: List[TradePattern],
        recent_trades: Optional[List[dict]] = None
    ) -> str:
        """Generate LLM-powered natural language summary.
        
        Args:
            metrics: Daily metrics
            patterns: Identified patterns
            recent_trades: Recent trade details
            
        Returns:
            LLM-generated summary
        """
        if recent_trades is None:
            # Get recent trades from analyzer
            from .trade_logger import TradeLogger
            logger_instance = TradeLogger()
            recent_trades = logger_instance.get_recent_trades(limit=10)
        
        # Prepare data for prompt
        metrics_dict = metrics.to_dict()
        patterns_dict = [
            {
                "pattern_type": p.pattern_type,
                "description": p.description,
                "severity": p.severity,
                "affected_trades": p.affected_trades,
                "impact_pnl": p.impact_pnl,
                "recommendation": p.recommendation
            }
            for p in patterns
        ]
        
        # Generate prompt
        prompt = self.prompt_templates.daily_summary_prompt(
            metrics_dict,
            patterns_dict,
            recent_trades
        )
        
        # Get LLM response
        try:
            request = self.prompt_templates.format_llm_request(prompt, temperature=0.4)
            response = self.bedrock_client.invoke_model(request)
            
            summary = response.get("content", [{}])[0].get("text", "No summary generated.")
            logger.info("Generated LLM daily summary")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate LLM summary: {e}")
            return "LLM summary generation failed. Using automated analysis."
    
    def generate_self_assessment(
        self,
        daily_metrics: DailyMetrics,
        patterns: List[TradePattern],
        current_config: dict
    ) -> str:
        """Generate LLM self-assessment of trading behavior.
        
        Args:
            daily_metrics: Today's metrics
            patterns: Identified patterns
            current_config: Current strategy configuration
            
        Returns:
            LLM self-assessment text
        """
        # Get historical context
        historical_metrics = self.analyzer.get_historical_metrics(days=14)
        historical_dict = [m.to_dict() for m in historical_metrics]
        
        # Prepare patterns
        patterns_dict = [
            {
                "pattern_type": p.pattern_type,
                "description": p.description,
                "severity": p.severity,
                "recommendation": p.recommendation
            }
            for p in patterns
        ]
        
        # Generate prompt
        prompt = self.prompt_templates.self_assessment_prompt(
            daily_metrics.to_dict(),
            historical_dict,
            patterns_dict,
            current_config
        )
        
        # Get LLM response
        try:
            request = self.prompt_templates.format_llm_request(prompt, temperature=0.3)
            response = self.bedrock_client.invoke_model(request)
            
            assessment = response.get("content", [{}])[0].get("text", "")
            logger.info("Generated LLM self-assessment")
            return assessment
            
        except Exception as e:
            logger.error(f"Failed to generate self-assessment: {e}")
            return ""
    
    def suggest_parameter_adjustments(
        self,
        self_assessment: str,
        daily_metrics: DailyMetrics,
        patterns: List[TradePattern],
        current_config: dict
    ) -> Tuple[List[StrategyAdjustment], str]:
        """Get LLM suggestions for parameter adjustments.
        
        Args:
            self_assessment: LLM's self-assessment
            daily_metrics: Today's metrics
            patterns: Identified patterns
            current_config: Current configuration
            
        Returns:
            Tuple of (adjustments list, reasoning)
        """
        # Prepare patterns
        patterns_dict = [
            {
                "pattern_type": p.pattern_type,
                "description": p.description,
                "severity": p.severity,
                "recommendation": p.recommendation
            }
            for p in patterns
        ]
        
        # Generate prompt
        prompt = self.prompt_templates.strategy_adjustment_prompt(
            self_assessment,
            daily_metrics.to_dict(),
            patterns_dict,
            current_config,
            self.safety_constraints.to_dict()
        )
        
        # Get LLM response
        try:
            request = self.prompt_templates.format_llm_request(
                prompt,
                temperature=0.2  # Lower temperature for parameter suggestions
            )
            response = self.bedrock_client.invoke_model(request)
            
            response_text = response.get("content", [{}])[0].get("text", "")
            
            # Parse JSON from response
            # LLM might wrap JSON in markdown code blocks
            json_text = response_text
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].split("```")[0].strip()
            
            suggestion = json.loads(json_text)
            
            # Validate and create adjustments
            adjustments = self._create_adjustments(
                suggestion.get("suggested_changes", {}),
                current_config,
                suggestion.get("reasoning", ""),
                suggestion.get("confidence", 0.5),
                suggestion.get("risk_level", "medium")
            )
            
            logger.info(f"Generated {len(adjustments)} parameter adjustment suggestions")
            return adjustments, suggestion.get("reasoning", "")
            
        except Exception as e:
            logger.error(f"Failed to generate parameter suggestions: {e}", exc_info=True)
            return [], ""
    
    def _create_adjustments(
        self,
        suggested_changes: Dict[str, Any],
        current_config: dict,
        reasoning: str,
        confidence: float,
        risk_level: str
    ) -> List[StrategyAdjustment]:
        """Create validated adjustment objects.
        
        Args:
            suggested_changes: Dictionary of parameter changes
            current_config: Current configuration
            reasoning: Reasoning for changes
            confidence: LLM confidence
            risk_level: Risk level assessment
            
        Returns:
            List of validated adjustments
        """
        adjustments = []
        
        for param, new_value in suggested_changes.items():
            # Get current value
            old_value = current_config.get(param)
            
            if old_value is None:
                logger.warning(f"Parameter {param} not found in current config")
                continue
            
            # Validate against safety constraints
            if not self._validate_parameter_change(param, old_value, new_value):
                logger.warning(f"Parameter change rejected: {param} = {new_value} violates safety constraints")
                continue
            
            # Check change magnitude
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                if old_value != 0:
                    pct_change = abs((new_value - old_value) / old_value)
                    if pct_change > self.safety_constraints.max_parameter_change_pct:
                        logger.warning(
                            f"Parameter change too large: {param} {old_value} → {new_value} "
                            f"({pct_change:.1%} > {self.safety_constraints.max_parameter_change_pct:.1%})"
                        )
                        # Scale down the change
                        direction = 1 if new_value > old_value else -1
                        new_value = old_value * (1 + direction * self.safety_constraints.max_parameter_change_pct)
                        logger.info(f"Scaled to {param} = {new_value}")
            
            adjustment = StrategyAdjustment(
                parameter=param,
                old_value=old_value,
                new_value=new_value,
                reasoning=reasoning,
                confidence=confidence,
                risk_level=risk_level,
                timestamp=datetime.now()
            )
            
            # Auto-approve if confidence is high enough and not requiring human approval
            if not self.require_human_approval and confidence >= self.auto_approve_threshold:
                adjustment.approved = True
                logger.info(f"Auto-approved adjustment: {param} (confidence: {confidence:.2f})")
            
            adjustments.append(adjustment)
        
        return adjustments
    
    def _validate_parameter_change(
        self,
        parameter: str,
        old_value: Any,
        new_value: Any
    ) -> bool:
        """Validate parameter change against safety constraints.
        
        Args:
            parameter: Parameter name
            old_value: Current value
            new_value: Proposed new value
            
        Returns:
            True if change is valid
        """
        c = self.safety_constraints
        
        validation_rules = {
            "rsi_buy": (c.rsi_buy_min, c.rsi_buy_max),
            "rsi_sell": (c.rsi_sell_min, c.rsi_sell_max),
            "sentiment_buy": (c.sentiment_buy_min, c.sentiment_buy_max),
            "sentiment_sell": (c.sentiment_sell_min, c.sentiment_sell_max),
            "sentiment_weight": (c.sentiment_weight_min, c.sentiment_weight_max),
            "stop_loss_ticks": (c.stop_loss_min, c.stop_loss_max),
            "take_profit_ticks": (c.take_profit_min, c.take_profit_max),
            "min_confidence_threshold": (c.min_confidence_min, c.min_confidence_max),
        }
        
        # Never allow changes to critical safety parameters
        forbidden_params = ["max_position_size", "max_daily_loss", "max_daily_trades"]
        if parameter in forbidden_params:
            logger.error(f"Attempted to change forbidden parameter: {parameter}")
            return False
        
        if parameter in validation_rules:
            min_val, max_val = validation_rules[parameter]
            if not (min_val <= new_value <= max_val):
                logger.warning(
                    f"Parameter {parameter} = {new_value} outside bounds [{min_val}, {max_val}]"
                )
                return False
        
        return True
    
    def run_daily_cycle(
        self,
        current_config: dict,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run complete daily analysis and learning cycle.
        
        Args:
            current_config: Current strategy configuration
            date: Date to analyze (defaults to today)
            
        Returns:
            Dictionary with analysis results and suggestions
        """
        logger.info(f"Starting daily learning cycle for {date or 'today'}")
        
        # Step 1: Analyze performance
        metrics, patterns, summary = self.analyze_daily_performance(date)
        
        if metrics is None:
            return {
                "success": False,
                "error": "No trading data available"
            }
        
        # Step 2: Generate LLM summary
        llm_summary = self.generate_llm_summary(metrics, patterns)
        
        # Step 3: Generate self-assessment
        self_assessment = self.generate_self_assessment(metrics, patterns, current_config)
        
        # Step 4: Get parameter adjustment suggestions
        adjustments, reasoning = self.suggest_parameter_adjustments(
            self_assessment,
            metrics,
            patterns,
            current_config
        )
        
        result = {
            "success": True,
            "date": metrics.date,
            "metrics": metrics.to_dict(),
            "patterns": [p.__dict__ for p in patterns],
            "automated_summary": summary,
            "llm_summary": llm_summary,
            "self_assessment": self_assessment,
            "suggested_adjustments": [adj.to_dict() for adj in adjustments],
            "adjustment_reasoning": reasoning,
            "requires_approval": self.require_human_approval,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(
            f"Daily cycle complete: {len(adjustments)} adjustments suggested, "
            f"{sum(1 for a in adjustments if a.approved)} auto-approved"
        )
        
        return result
