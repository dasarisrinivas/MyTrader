"""SPY Futures LLM Insight Generator.

Generates AI-powered insights and recommendations for SPY Futures trading
using AWS Bedrock Claude.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..llm.bedrock_client import BedrockClient
from ..utils.logger import logger
from .spy_futures_analyzer import SPYFuturesPerformance, SPYFuturesTrade


@dataclass
class SPYInsight:
    """Individual trading insight for SPY Futures."""
    type: str  # pattern, timing, risk, signal
    category: str
    description: str
    severity: str  # info, warning, critical
    confidence: float
    reasoning: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SPYSuggestion:
    """Trading adjustment suggestion for SPY Futures."""
    parameter: str
    current_value: any
    suggested_value: any
    reasoning: str
    expected_impact: str
    confidence: float
    priority: str  # low, medium, high
    requires_approval: bool
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SPYDailyReport:
    """Complete daily SPY Futures trading report for dashboard."""
    date: str
    symbol: str
    
    # Performance metrics
    performance: Dict
    
    # LLM analysis
    observations: List[str]
    insights: List[SPYInsight]
    suggestions: Dict[str, any]
    warnings: List[str]
    
    # Patterns identified
    profitable_patterns: List[str]
    losing_patterns: List[str]
    
    # Market context
    market_conditions: str
    volatility_assessment: str
    
    # Recommendations
    trade_frequency_recommendation: Optional[str] = None
    position_sizing_recommendation: Optional[str] = None
    timing_recommendations: List[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON/dashboard."""
        result = {
            "date": self.date,
            "symbol": self.symbol,
            "performance": self.performance,
            "observations": self.observations,
            "insights": [i.to_dict() for i in self.insights],
            "suggestions": self.suggestions,
            "warnings": self.warnings,
            "profitable_patterns": self.profitable_patterns,
            "losing_patterns": self.losing_patterns,
            "market_conditions": self.market_conditions,
            "volatility_assessment": self.volatility_assessment
        }
        
        if self.trade_frequency_recommendation:
            result["trade_frequency_recommendation"] = self.trade_frequency_recommendation
        if self.position_sizing_recommendation:
            result["position_sizing_recommendation"] = self.position_sizing_recommendation
        if self.timing_recommendations:
            result["timing_recommendations"] = self.timing_recommendations
        
        return result


class SPYFuturesInsightGenerator:
    """Generate AI insights for SPY Futures trading using LLM."""
    
    def __init__(self, bedrock_client: Optional[BedrockClient] = None):
        """Initialize insight generator.
        
        Args:
            bedrock_client: Optional Bedrock client instance
        """
        self.bedrock_client = bedrock_client or BedrockClient()
        logger.info("SPYFuturesInsightGenerator initialized")
    
    def generate_analysis_prompt(
        self,
        performance: SPYFuturesPerformance,
        trades: List[SPYFuturesTrade]
    ) -> str:
        """Generate comprehensive analysis prompt for LLM.
        
        Args:
            performance: Performance summary
            trades: List of recent trades
            
        Returns:
            Formatted prompt string
        """
        # Recent sample trades for context
        recent_trades = sorted(trades, key=lambda t: t.timestamp, reverse=True)[:10]
        
        prompt = f"""You are an expert SPY Futures trading analyst. Analyze the following paper trading performance for {performance.symbol} and provide structured insights.

PERFORMANCE SUMMARY - {performance.date}
Symbol: {performance.symbol} (SPY E-mini Futures)
Period: Last {performance.total_trades} trades

CORE METRICS:
- Total Trades: {performance.total_trades} ({performance.closed_trades} closed, {performance.open_positions} open)
- Win Rate: {performance.win_rate:.1%}
- Total P&L: ${performance.total_pnl:,.2f}
- Profit Factor: {performance.profit_factor:.2f}
- Max Drawdown: ${performance.max_drawdown:,.2f}

P&L BREAKDOWN:
- Gross Profit: ${performance.gross_profit:,.2f} ({performance.winning_trades} wins)
- Gross Loss: ${performance.gross_loss:,.2f} ({performance.losing_trades} losses)
- Average Win: ${performance.average_win:,.2f}
- Average Loss: ${performance.average_loss:,.2f}
- Largest Win: ${performance.largest_win:,.2f}
- Largest Loss: ${performance.largest_loss:,.2f}

TRADING BEHAVIOR:
- Average Holding Time: {performance.average_holding_time_minutes:.1f} minutes

SIGNAL PERFORMANCE:
"""
        
        for signal, count in performance.trades_by_signal.items():
            pnl = performance.pnl_by_signal.get(signal, 0)
            win_rate = performance.win_rate_by_signal.get(signal, 0)
            prompt += f"- {signal}: {count} trades, ${pnl:,.2f} P&L, {win_rate:.1%} WR\n"
        
        prompt += "\nHOURLY PERFORMANCE (EST):\n"
        for hour in sorted(performance.trades_by_hour.keys()):
            count = performance.trades_by_hour[hour]
            pnl = performance.pnl_by_hour.get(hour, 0)
            prompt += f"- {hour:02d}:00: {count} trades, ${pnl:,.2f} P&L\n"
        
        if performance.llm_enhanced_trades > 0:
            prompt += f"\nLLM ENHANCEMENT:\n"
            prompt += f"- LLM-Enhanced Trades: {performance.llm_enhanced_trades}\n"
            prompt += f"- LLM-Enhanced P&L: ${performance.llm_enhanced_pnl:,.2f}\n"
            if performance.llm_average_confidence:
                prompt += f"- Average Confidence: {performance.llm_average_confidence:.2f}\n"
        
        prompt += f"\nRECENT TRADES (last 10):\n"
        for i, trade in enumerate(recent_trades, 1):
            status = "CLOSED" if trade.pnl is not None else "OPEN"
            pnl_str = f"${trade.pnl:,.2f}" if trade.pnl is not None else "N/A"
            holding = f"{trade.holding_time_minutes}m" if trade.holding_time_minutes else "N/A"
            
            prompt += f"{i}. {trade.timestamp[:19]} | {trade.action} {trade.quantity} @ ${trade.price} "
            prompt += f"| {trade.signal_type} | {status} | P&L: {pnl_str} | Hold: {holding}\n"
            
            if trade.llm_confidence:
                prompt += f"   LLM: {trade.llm_confidence:.2f} confidence - {trade.llm_reasoning}\n"
        
        prompt += """

ANALYSIS REQUEST:

Please provide a structured analysis in JSON format with the following sections:

1. **observations**: Array of key observations about trading patterns (strings)

2. **insights**: Array of detailed insights, each with:
   - type: "pattern" | "timing" | "risk" | "signal"
   - category: specific category name
   - description: detailed description
   - severity: "info" | "warning" | "critical"
   - confidence: 0.0 to 1.0
   - reasoning: why this matters

3. **suggestions**: Object with specific parameter recommendations:
   - trade_frequency_limit: suggested trades per hour/day
   - sentiment_confidence_threshold: minimum confidence (0-1)
   - preferred_trading_hours: array of best hours (EST)
   - position_sizing_adjustment: "increase" | "decrease" | "maintain"
   - stop_loss_adjustment: suggested adjustment in ticks
   - take_profit_adjustment: suggested adjustment in ticks

4. **warnings**: Array of critical issues requiring attention (strings)

5. **profitable_patterns**: Array of patterns associated with winning trades

6. **losing_patterns**: Array of patterns associated with losing trades

7. **market_conditions**: Overall market assessment

8. **volatility_assessment**: Volatility analysis and implications

IMPORTANT GUIDELINES:
- Focus ONLY on SPY Futures (ES/MES) - do not suggest other instruments
- Be specific and actionable in recommendations
- Provide confidence scores for all insights
- Identify both what's working and what needs improvement
- Consider market hours, holding times, and signal performance
- Highlight if LLM enhancement is helping or hurting
- Flag any risky patterns requiring immediate attention

Return ONLY valid JSON, no additional text or markdown formatting."""

        return prompt
    
    def parse_llm_response(self, response_text: str) -> Dict:
        """Parse LLM JSON response.
        
        Args:
            response_text: Raw LLM response text
            
        Returns:
            Parsed dictionary
        """
        try:
            # Clean response (remove markdown code blocks if present)
            text = response_text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            # Parse JSON
            data = json.loads(text)
            return data
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            
            # Return minimal structure
            return {
                "observations": ["LLM response parsing failed"],
                "insights": [],
                "suggestions": {},
                "warnings": ["Could not parse LLM analysis"],
                "profitable_patterns": [],
                "losing_patterns": [],
                "market_conditions": "Unknown",
                "volatility_assessment": "Unknown"
            }
    
    def generate_insights(
        self,
        performance: SPYFuturesPerformance,
        trades: List[SPYFuturesTrade]
    ) -> SPYDailyReport:
        """Generate complete daily insights report.
        
        Args:
            performance: Performance summary
            trades: List of trades
            
        Returns:
            Complete daily report with insights
        """
        logger.info(f"Generating SPY Futures insights for {performance.date}")
        
        # Generate prompt
        prompt = self.generate_analysis_prompt(performance, trades)
        
        # Call LLM
        try:
            response = self.bedrock_client.generate_text(
                prompt=prompt,
                max_tokens=3000,
                temperature=0.3
            )
            
            # Parse response
            analysis = self.parse_llm_response(response)
            
            # Convert insights to objects
            insights = []
            for insight_data in analysis.get("insights", []):
                insights.append(SPYInsight(
                    type=insight_data.get("type", "pattern"),
                    category=insight_data.get("category", "general"),
                    description=insight_data.get("description", ""),
                    severity=insight_data.get("severity", "info"),
                    confidence=insight_data.get("confidence", 0.5),
                    reasoning=insight_data.get("reasoning", "")
                ))
            
            # Build report
            report = SPYDailyReport(
                date=performance.date,
                symbol=performance.symbol,
                performance={
                    "total_trades": performance.total_trades,
                    "win_rate": round(performance.win_rate * 100, 1),  # Convert to percentage
                    "profit_loss": round(performance.total_pnl, 2),
                    "max_drawdown": round(performance.max_drawdown, 2),
                    "profit_factor": round(performance.profit_factor, 2),
                    "average_win": round(performance.average_win, 2),
                    "average_loss": round(performance.average_loss, 2),
                    "holding_time_avg": round(performance.average_holding_time_minutes, 1)
                },
                observations=analysis.get("observations", []),
                insights=insights,
                suggestions=analysis.get("suggestions", {}),
                warnings=analysis.get("warnings", []),
                profitable_patterns=analysis.get("profitable_patterns", []),
                losing_patterns=analysis.get("losing_patterns", []),
                market_conditions=analysis.get("market_conditions", "Normal"),
                volatility_assessment=analysis.get("volatility_assessment", "Moderate"),
                trade_frequency_recommendation=analysis.get("suggestions", {}).get("trade_frequency_limit"),
                position_sizing_recommendation=analysis.get("suggestions", {}).get("position_sizing_adjustment"),
                timing_recommendations=analysis.get("suggestions", {}).get("preferred_trading_hours")
            )
            
            logger.info(f"Generated {len(insights)} insights, {len(report.warnings)} warnings")
            return report
        
        except Exception as e:
            logger.error(f"Error generating insights: {e}", exc_info=True)
            
            # Return basic report with error
            return SPYDailyReport(
                date=performance.date,
                symbol=performance.symbol,
                performance={
                    "total_trades": performance.total_trades,
                    "win_rate": round(performance.win_rate * 100, 1),
                    "profit_loss": round(performance.total_pnl, 2),
                    "max_drawdown": round(performance.max_drawdown, 2)
                },
                observations=[],
                insights=[],
                suggestions={},
                warnings=[f"LLM analysis failed: {str(e)}"],
                profitable_patterns=[],
                losing_patterns=[],
                market_conditions="Unknown",
                volatility_assessment="Unknown"
            )
    
    def save_report(
        self,
        report: SPYDailyReport,
        output_dir: Optional[Path] = None
    ) -> Path:
        """Save report to JSON file.
        
        Args:
            report: Daily report
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "reports" / "spy_futures_daily"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        filename = f"spy_futures_report_{report.date}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved SPY Futures report: {filepath}")
        return filepath
