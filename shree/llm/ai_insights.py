"""AI insight generator for live trading performance."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.logger import logger
from .bedrock_client import BedrockClient
from .trade_analyzer import TradeRecord, TradeSummary


@dataclass
class TradingInsight:
    """AI-generated trading insight."""
    
    insight_type: str  # "observation", "suggestion", "warning", "trend"
    category: str  # "signal_quality", "timing", "risk_management", "market_conditions"
    description: str
    severity: str  # "low", "medium", "high"
    confidence: float
    reasoning: str
    data_evidence: Optional[Dict]


@dataclass
class TradingRecommendation:
    """Structured trading recommendation."""
    
    parameter: str
    current_value: Optional[any]
    suggested_value: any
    reasoning: str
    expected_impact: str
    confidence: float
    risk_level: str  # "low", "medium", "high"


@dataclass
class AIInsightReport:
    """Complete AI insight report."""
    
    timestamp: str
    period_start: str
    period_end: str
    summary_text: str
    observations: List[TradingInsight]
    recommendations: List[TradingRecommendation]
    market_trends: List[str]
    behavioral_patterns: List[str]
    warnings: List[str]
    raw_llm_response: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "summary_text": self.summary_text,
            "observations": [
                {
                    "type": o.insight_type,
                    "category": o.category,
                    "description": o.description,
                    "severity": o.severity,
                    "confidence": o.confidence,
                    "reasoning": o.reasoning,
                    "data_evidence": o.data_evidence
                }
                for o in self.observations
            ],
            "recommendations": [
                {
                    "parameter": r.parameter,
                    "current_value": r.current_value,
                    "suggested_value": r.suggested_value,
                    "reasoning": r.reasoning,
                    "expected_impact": r.expected_impact,
                    "confidence": r.confidence,
                    "risk_level": r.risk_level
                }
                for r in self.recommendations
            ],
            "market_trends": self.market_trends,
            "behavioral_patterns": self.behavioral_patterns,
            "warnings": self.warnings,
            "raw_llm_response": self.raw_llm_response
        }


class AIInsightGenerator:
    """Generates AI-powered insights from live trading performance."""
    
    def __init__(
        self,
        bedrock_client: Optional[BedrockClient] = None,
        reports_dir: Optional[Path] = None
    ):
        """Initialize AI insight generator.
        
        Args:
            bedrock_client: Bedrock client for LLM calls
            reports_dir: Directory for insight reports
        """
        self.bedrock_client = bedrock_client or BedrockClient()
        
        if reports_dir is None:
            project_root = Path(__file__).parent.parent.parent
            reports_dir = project_root / "reports" / "ai_insights"
        
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("AIInsightGenerator initialized")
    
    def generate_analysis_prompt(
        self,
        summary: TradeSummary,
        recent_trades: List[TradeRecord]
    ) -> str:
        """Generate comprehensive analysis prompt for LLM.
        
        Args:
            summary: Trade summary
            recent_trades: Recent trade records
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are an expert algorithmic trading analyst reviewing live paper trading performance.

TRADING PERFORMANCE SUMMARY
Period: {summary.period_start} to {summary.period_end}
========================================

Overall Performance:
- Total Trades: {summary.total_trades} ({summary.closed_trades} closed, {summary.open_trades} open)
- Win Rate: {summary.win_rate:.1%} ({summary.winning_trades}W / {summary.losing_trades}L / {summary.breakeven_trades}BE)
- Total P&L: ${summary.total_pnl:,.2f}
- Profit Factor: {summary.profit_factor:.2f}
- Sharpe Ratio: {summary.sharpe_ratio:.2f}
- Max Drawdown: ${summary.max_drawdown:,.2f}

Trade Quality:
- Average Win: ${summary.avg_win:,.2f}
- Average Loss: ${summary.avg_loss:,.2f}
- Largest Win: ${summary.largest_win:,.2f}
- Largest Loss: ${summary.largest_loss:,.2f}
- Average Holding Time: {summary.avg_holding_time:.1f} minutes

Signal Performance Analysis:
"""
        
        for signal_type, count in summary.trades_by_signal.items():
            wr = summary.win_rate_by_signal.get(signal_type, 0)
            pnl = summary.pnl_by_signal.get(signal_type, 0)
            prompt += f"""
- {signal_type}: {count} trades, {wr:.1%} win rate, ${pnl:,.2f} P&L"""
        
        if summary.trades_by_condition:
            prompt += "\n\nMarket Condition Performance:"
            for condition, count in summary.trades_by_condition.items():
                wr = summary.win_rate_by_condition.get(condition, 0)
                prompt += f"""
- {condition}: {count} trades, {wr:.1%} win rate"""
        
        prompt += f"""

LLM Enhancement Performance:
- LLM Enhanced Trades: {summary.llm_enhanced_trades}
- LLM Accuracy: {summary.llm_accuracy:.1%}
- Average LLM Confidence: {summary.avg_llm_confidence:.1%}

Timing Analysis (Trades by Hour):
"""
        
        for hour in sorted(summary.trades_by_hour.keys()):
            count = summary.trades_by_hour[hour]
            wr = summary.win_rate_by_hour.get(hour, 0)
            prompt += f"\n- {hour:02d}:00: {count} trades, {wr:.1%} win rate"
        
        # Add sample of recent trades
        prompt += "\n\nRecent Trade Examples (Last 5):\n"
        for i, trade in enumerate(recent_trades[-5:], 1):
            prompt += f"""
Trade {i}:
  {trade.action} {trade.quantity} @ ${trade.entry_price:.2f}"""
            
            if trade.exit_price:
                prompt += f" → ${trade.exit_price:.2f}"
            
            prompt += f"""
  Result: {trade.outcome}, P&L: ${trade.realized_pnl:.2f}, Duration: {trade.holding_time_minutes:.1f}min
  Signal: {trade.signal_type}, Sentiment: {trade.sentiment_score if trade.sentiment_score else 'N/A'}"""
            
            if trade.llm_reasoning:
                prompt += f"\n  LLM Reasoning: {trade.llm_reasoning[:100]}..."
        
        prompt += """

ANALYSIS TASKS:

1. OBSERVATIONS - Identify key patterns, issues, or anomalies:
   - Trading frequency patterns (are we overtrading? undertrading?)
   - Signal effectiveness (which signals work best/worst?)
   - Timing issues (poor performance at specific hours?)
   - Market condition adaptation (how does the bot perform in different conditions?)
   - LLM performance quality

2. BEHAVIORAL PATTERNS - Describe the bot's trading behavior:
   - Strategy drift or consistency
   - Entry/exit timing patterns
   - Risk management effectiveness
   - Sentiment interpretation quality

3. MARKET TRENDS - Summarize recent market/sentiment trends affecting performance:
   - Overall market direction
   - Volatility patterns
   - Volume conditions
   - Sentiment quality

4. RECOMMENDATIONS - Suggest specific, actionable improvements:
   - Trading logic adjustments
   - Sentiment interpretation changes
   - Execution timing improvements
   - Risk management enhancements
   - Parameter tuning suggestions

5. WARNINGS - Flag any concerning patterns that need immediate attention

Provide your analysis in the following JSON format:

{
  "summary": "Brief 2-3 sentence overview of performance",
  "observations": [
    {
      "type": "observation|suggestion|warning|trend",
      "category": "signal_quality|timing|risk_management|market_conditions",
      "description": "Clear description",
      "severity": "low|medium|high",
      "confidence": 0.0-1.0,
      "reasoning": "Why this matters",
      "evidence": {"relevant": "data points"}
    }
  ],
  "behavioral_patterns": [
    "Pattern 1: description",
    "Pattern 2: description"
  ],
  "market_trends": [
    "Trend 1: description",
    "Trend 2: description"
  ],
  "recommendations": [
    {
      "parameter": "parameter name or logic area",
      "current_value": "current state",
      "suggested_value": "suggested change",
      "reasoning": "why this will help",
      "expected_impact": "what to expect",
      "confidence": 0.0-1.0,
      "risk_level": "low|medium|high"
    }
  ],
  "warnings": [
    "Warning 1: description",
    "Warning 2: description"
  ]
}

Focus on actionable insights that can improve trading consistency and reduce drawdown.
Be specific with numbers and timeframes when possible.
"""
        
        return prompt
    
    def parse_llm_response(
        self,
        response_text: str,
        summary: TradeSummary
    ) -> AIInsightReport:
        """Parse LLM response into structured insights.
        
        Args:
            response_text: Raw LLM response
            summary: Trade summary for context
            
        Returns:
            Structured insight report
        """
        try:
            # Extract JSON from response
            json_text = response_text
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(json_text)
            
            # Parse observations
            observations = []
            for obs_data in data.get("observations", []):
                observations.append(TradingInsight(
                    insight_type=obs_data.get("type", "observation"),
                    category=obs_data.get("category", "general"),
                    description=obs_data.get("description", ""),
                    severity=obs_data.get("severity", "medium"),
                    confidence=float(obs_data.get("confidence", 0.5)),
                    reasoning=obs_data.get("reasoning", ""),
                    data_evidence=obs_data.get("evidence")
                ))
            
            # Parse recommendations
            recommendations = []
            for rec_data in data.get("recommendations", []):
                recommendations.append(TradingRecommendation(
                    parameter=rec_data.get("parameter", ""),
                    current_value=rec_data.get("current_value"),
                    suggested_value=rec_data.get("suggested_value"),
                    reasoning=rec_data.get("reasoning", ""),
                    expected_impact=rec_data.get("expected_impact", ""),
                    confidence=float(rec_data.get("confidence", 0.5)),
                    risk_level=rec_data.get("risk_level", "medium")
                ))
            
            report = AIInsightReport(
                timestamp=datetime.now().isoformat(),
                period_start=summary.period_start,
                period_end=summary.period_end,
                summary_text=data.get("summary", "No summary provided"),
                observations=observations,
                recommendations=recommendations,
                market_trends=data.get("market_trends", []),
                behavioral_patterns=data.get("behavioral_patterns", []),
                warnings=data.get("warnings", []),
                raw_llm_response=response_text
            )
            
            logger.info(
                f"Parsed AI insights: {len(observations)} observations, "
                f"{len(recommendations)} recommendations"
            )
            
            return report
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            
            # Return minimal report
            return AIInsightReport(
                timestamp=datetime.now().isoformat(),
                period_start=summary.period_start,
                period_end=summary.period_end,
                summary_text="Failed to parse LLM response",
                observations=[],
                recommendations=[],
                market_trends=[],
                behavioral_patterns=[],
                warnings=["Failed to parse AI analysis"],
                raw_llm_response=response_text
            )
    
    def generate_insights(
        self,
        summary: TradeSummary,
        recent_trades: List[TradeRecord]
    ) -> AIInsightReport:
        """Generate AI-powered insights from trading performance.
        
        Args:
            summary: Trade summary
            recent_trades: Recent trade records
            
        Returns:
            AI insight report
        """
        logger.info("Generating AI insights from trading performance...")
        
        # Generate prompt
        prompt = self.generate_analysis_prompt(summary, recent_trades)
        
        # Get LLM response
        try:
            request = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,  # Longer for detailed analysis
                "temperature": 0.4,  # Balanced creativity and consistency
                "system": """You are an expert algorithmic trading analyst with deep expertise in:
- Technical analysis and signal quality
- Risk management and drawdown control
- Market microstructure and execution timing
- Sentiment analysis effectiveness
- Machine learning model behavior
- Trading psychology and behavioral patterns

Provide detailed, data-driven analysis with specific, actionable recommendations.
Focus on improving consistency and reducing risk while maintaining profitability.""",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = self.bedrock_client.invoke_model(request)
            response_text = response.get("content", [{}])[0].get("text", "")
            
            # Parse response
            report = self.parse_llm_response(response_text, summary)
            
            logger.info("AI insights generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate AI insights: {e}", exc_info=True)
            
            # Return error report
            return AIInsightReport(
                timestamp=datetime.now().isoformat(),
                period_start=summary.period_start,
                period_end=summary.period_end,
                summary_text="Failed to generate AI insights",
                observations=[],
                recommendations=[],
                market_trends=[],
                behavioral_patterns=[],
                warnings=[f"AI insight generation failed: {str(e)}"],
                raw_llm_response=""
            )
    
    def save_report(
        self,
        report: AIInsightReport,
        format: str = "json"
    ) -> Path:
        """Save insight report to file.
        
        Args:
            report: Insight report
            format: Output format ("json" or "markdown")
            
        Returns:
            Path to saved report
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        if format == "json":
            filename = f"ai_insights_{date_str}.json"
            filepath = self.reports_dir / filename
            
            with open(filepath, "w") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
        
        elif format == "markdown":
            filename = f"daily_review_{date_str}.md"
            filepath = self.reports_dir / filename
            
            md_content = self._generate_markdown_report(report)
            
            with open(filepath, "w") as f:
                f.write(md_content)
        
        logger.info(f"AI insight report saved: {filepath}")
        return filepath
    
    def _generate_markdown_report(self, report: AIInsightReport) -> str:
        """Generate markdown-formatted report.
        
        Args:
            report: Insight report
            
        Returns:
            Markdown content
        """
        md = f"""# Daily Trading Review - {report.period_end}

**Period:** {report.period_start} to {report.period_end}  
**Generated:** {report.timestamp}

---

## Executive Summary

{report.summary_text}

---

## Key Observations

"""
        
        for obs in report.observations:
            severity_icon = {"low": "ℹ️", "medium": "⚠️", "high": "🚨"}.get(obs.severity, "•")
            md += f"""
### {severity_icon} {obs.description}

**Category:** {obs.category}  
**Severity:** {obs.severity.upper()}  
**Confidence:** {obs.confidence:.0%}

{obs.reasoning}

"""
            if obs.data_evidence:
                md += "**Evidence:**\n"
                for key, value in obs.data_evidence.items():
                    md += f"- {key}: {value}\n"
                md += "\n"
        
        if report.behavioral_patterns:
            md += "---\n\n## Behavioral Patterns\n\n"
            for pattern in report.behavioral_patterns:
                md += f"- {pattern}\n"
            md += "\n"
        
        if report.market_trends:
            md += "---\n\n## Market Trends\n\n"
            for trend in report.market_trends:
                md += f"- {trend}\n"
            md += "\n"
        
        if report.recommendations:
            md += "---\n\n## Recommendations\n\n"
            for i, rec in enumerate(report.recommendations, 1):
                risk_icon = {"low": "✅", "medium": "⚠️", "high": "🔴"}.get(rec.risk_level, "•")
                md += f"""
### {i}. {risk_icon} {rec.parameter}

**Current:** {rec.current_value}  
**Suggested:** {rec.suggested_value}  
**Confidence:** {rec.confidence:.0%} | **Risk:** {rec.risk_level.upper()}

**Reasoning:**  
{rec.reasoning}

**Expected Impact:**  
{rec.expected_impact}

"""
        
        if report.warnings:
            md += "---\n\n## ⚠️ Warnings\n\n"
            for warning in report.warnings:
                md += f"- 🚨 {warning}\n"
            md += "\n"
        
        md += """---

## Next Steps

1. Review all recommendations carefully
2. Validate suggestions against current market conditions
3. Implement low-risk changes first
4. Monitor impact over next trading session
5. Document any manual adjustments made

---

*This report is generated by AI analysis and should be reviewed by a human trader before implementation.*
"""
        
        return md
