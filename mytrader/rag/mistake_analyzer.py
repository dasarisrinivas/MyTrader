"""Mistake Analyzer - Generates analysis and notes for losing trades.

This module analyzes losing trades to:
- Identify common mistake patterns
- Generate actionable insights
- Create searchable mistake notes for RAG
- Help improve future trading decisions
- Uses CST (Central Standard Time) for all timestamps
"""
import json
from collections import Counter
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

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


class MistakePattern:
    """Known mistake patterns to detect."""
    
    COUNTER_TREND = "counter_trend"
    NEAR_RESISTANCE_BUY = "near_resistance_buy"
    NEAR_SUPPORT_SELL = "near_support_sell"
    OVERBOUGHT_BUY = "overbought_buy"
    OVERSOLD_SELL = "oversold_sell"
    LOW_CONFIDENCE = "low_confidence"
    HIGH_VOLATILITY = "high_volatility"
    OPENING_TRADE = "opening_volatility"
    QUICK_EXIT = "quick_exit"
    OVERRIDDEN_FILTERS = "overridden_filters"
    CHOP_MARKET = "chop_market"
    FRIDAY_CLOSE = "friday_close"


class MistakeAnalyzer:
    """Analyzes losing trades and generates mistake notes.
    
    Features:
    - Pattern detection for common mistakes
    - Severity scoring
    - Markdown note generation
    - Aggregated insights
    """
    
    # Pattern definitions with detection logic and severity
    PATTERNS = {
        MistakePattern.COUNTER_TREND: {
            "name": "Counter-Trend Entry",
            "severity": "HIGH",
            "description": "Entered against the prevailing trend",
            "advice": "Wait for trend alignment or use tighter stops for counter-trend trades",
        },
        MistakePattern.NEAR_RESISTANCE_BUY: {
            "name": "Bought Near Resistance",
            "severity": "HIGH",
            "description": "Entered long too close to PDH or resistance level",
            "advice": "Wait for breakout confirmation or look for pullback entries",
        },
        MistakePattern.NEAR_SUPPORT_SELL: {
            "name": "Sold Near Support",
            "severity": "HIGH",
            "description": "Entered short too close to PDL or support level",
            "advice": "Wait for breakdown confirmation or look for retest entries",
        },
        MistakePattern.OVERBOUGHT_BUY: {
            "name": "Overbought Buy",
            "severity": "MEDIUM",
            "description": "Bought when RSI indicated overbought conditions",
            "advice": "Wait for RSI pullback or use it as a warning sign",
        },
        MistakePattern.OVERSOLD_SELL: {
            "name": "Oversold Sell",
            "severity": "MEDIUM",
            "description": "Sold when RSI indicated oversold conditions",
            "advice": "Wait for RSI bounce or use it as a warning sign",
        },
        MistakePattern.LOW_CONFIDENCE: {
            "name": "Low Confidence Trade",
            "severity": "MEDIUM",
            "description": "Entered with LLM confidence below threshold",
            "advice": "Only take trades with confidence above 60%",
        },
        MistakePattern.HIGH_VOLATILITY: {
            "name": "High Volatility Entry",
            "severity": "MEDIUM",
            "description": "ATR was unusually high, increasing risk",
            "advice": "Reduce position size or widen stops in high volatility",
        },
        MistakePattern.OPENING_TRADE: {
            "name": "Opening Volatility",
            "severity": "LOW",
            "description": "Traded during the first 15-30 minutes of session",
            "advice": "Wait for initial volatility to settle",
        },
        MistakePattern.QUICK_EXIT: {
            "name": "Premature Exit",
            "severity": "LOW",
            "description": "Trade was closed very quickly (possible stop too tight)",
            "advice": "Consider wider stops or avoid entries in chop",
        },
        MistakePattern.OVERRIDDEN_FILTERS: {
            "name": "Overridden Filters",
            "severity": "HIGH",
            "description": "Trade was taken despite filter warnings",
            "advice": "Respect the filter system - it exists for a reason",
        },
        MistakePattern.CHOP_MARKET: {
            "name": "Chop Market Trade",
            "severity": "MEDIUM",
            "description": "Market was in chop/sideways mode",
            "advice": "Reduce trading in choppy conditions or use mean-reversion",
        },
        MistakePattern.FRIDAY_CLOSE: {
            "name": "Friday Close Position",
            "severity": "LOW",
            "description": "Held position into Friday close",
            "advice": "Consider closing positions before weekend",
        },
    }
    
    def __init__(self, storage: Optional[RAGStorageManager] = None):
        """Initialize mistake analyzer.
        
        Args:
            storage: RAG storage manager instance
        """
        self.storage = storage or get_rag_storage()
        
        logger.info("MistakeAnalyzer initialized")
    
    def analyze_trade(self, trade: TradeRecord) -> Dict[str, Any]:
        """Analyze a single losing trade for mistakes.
        
        Args:
            trade: The losing trade to analyze
            
        Returns:
            Analysis dictionary with patterns, severity, and advice
        """
        if trade.result != "LOSS":
            return {"patterns": [], "severity": "NONE", "is_loss": False}
        
        detected_patterns = []
        
        # Check for counter-trend
        if trade.action == "BUY" and trade.market_trend == "DOWNTREND":
            detected_patterns.append(MistakePattern.COUNTER_TREND)
        elif trade.action == "SELL" and trade.market_trend == "UPTREND":
            detected_patterns.append(MistakePattern.COUNTER_TREND)
        
        # Check level proximity
        if trade.action == "BUY" and abs(trade.price_vs_pdh_pct) < 0.5:
            detected_patterns.append(MistakePattern.NEAR_RESISTANCE_BUY)
        elif trade.action == "SELL" and abs(trade.price_vs_pdl_pct) < 0.5:
            detected_patterns.append(MistakePattern.NEAR_SUPPORT_SELL)
        
        # Check RSI
        if trade.action == "BUY" and trade.rsi > 70:
            detected_patterns.append(MistakePattern.OVERBOUGHT_BUY)
        elif trade.action == "SELL" and trade.rsi < 30:
            detected_patterns.append(MistakePattern.OVERSOLD_SELL)
        
        # Check confidence
        if trade.llm_confidence < 60 and trade.llm_confidence > 0:
            detected_patterns.append(MistakePattern.LOW_CONFIDENCE)
        
        # Check volatility
        if trade.volatility_regime == "HIGH":
            detected_patterns.append(MistakePattern.HIGH_VOLATILITY)
        
        # Check time of day
        if trade.time_of_day == "OPEN":
            detected_patterns.append(MistakePattern.OPENING_TRADE)
        
        # Check duration
        if trade.duration_minutes < 3:
            detected_patterns.append(MistakePattern.QUICK_EXIT)
        
        # Check overridden filters
        if trade.filters_blocked:
            detected_patterns.append(MistakePattern.OVERRIDDEN_FILTERS)
        
        # Check market condition
        if trade.market_trend == "CHOP":
            detected_patterns.append(MistakePattern.CHOP_MARKET)
        
        # Check Friday close
        if trade.day_of_week == "FRIDAY" and trade.time_of_day == "CLOSE":
            detected_patterns.append(MistakePattern.FRIDAY_CLOSE)
        
        # Calculate overall severity
        severities = [self.PATTERNS[p]["severity"] for p in detected_patterns]
        if "HIGH" in severities:
            overall_severity = "HIGH"
        elif "MEDIUM" in severities:
            overall_severity = "MEDIUM"
        elif severities:
            overall_severity = "LOW"
        else:
            overall_severity = "UNKNOWN"
        
        return {
            "patterns": detected_patterns,
            "pattern_details": [self.PATTERNS[p] for p in detected_patterns],
            "severity": overall_severity,
            "is_loss": True,
            "pnl": trade.pnl,
        }
    
    def generate_mistake_note(self, trade: TradeRecord) -> str:
        """Generate a detailed mistake note for a losing trade.
        
        Args:
            trade: The losing trade
            
        Returns:
            Markdown-formatted mistake note
        """
        analysis = self.analyze_trade(trade)
        
        if not analysis["is_loss"]:
            return ""
        
        # Build the note
        lines = []
        
        try:
            ts = datetime.fromisoformat(trade.timestamp.replace("Z", "+00:00"))
            date_str = ts.strftime("%Y-%m-%d %H:%M")
        except:
            date_str = trade.timestamp
        
        lines.append(f"# Mistake Analysis – {date_str}")
        lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append(f"- **Trade ID:** {trade.trade_id}")
        lines.append(f"- **Action:** {trade.action}")
        lines.append(f"- **P&L:** ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
        lines.append(f"- **Severity:** {analysis['severity']}")
        lines.append("")
        
        # Detected patterns
        lines.append("## Detected Patterns")
        if analysis["patterns"]:
            for pattern in analysis["patterns"]:
                info = self.PATTERNS[pattern]
                emoji = "🔴" if info["severity"] == "HIGH" else "🟡" if info["severity"] == "MEDIUM" else "🟢"
                lines.append(f"### {emoji} {info['name']}")
                lines.append(f"**Description:** {info['description']}")
                lines.append(f"**Advice:** {info['advice']}")
                lines.append("")
        else:
            lines.append("No clear patterns detected. Possible market noise or black swan event.")
            lines.append("")
        
        # Trade details
        lines.append("## Trade Details")
        lines.append(f"- **Entry Price:** {trade.entry_price:.2f}")
        lines.append(f"- **Exit Price:** {trade.exit_price:.2f}")
        lines.append(f"- **Stop Loss:** {trade.stop_loss:.2f}")
        lines.append(f"- **Take Profit:** {trade.take_profit:.2f}")
        lines.append(f"- **Duration:** {trade.duration_minutes:.1f} minutes")
        lines.append(f"- **Exit Reason:** {trade.exit_reason}")
        lines.append("")
        
        # Market context
        lines.append("## Market Context")
        lines.append(f"- **Trend:** {trade.market_trend}")
        lines.append(f"- **Volatility:** {trade.volatility_regime}")
        lines.append(f"- **Time of Day:** {trade.time_of_day}")
        lines.append(f"- **Day:** {trade.day_of_week}")
        lines.append(f"- **PDH:** {trade.pdh:.2f}")
        lines.append(f"- **PDL:** {trade.pdl:.2f}")
        lines.append(f"- **Price vs PDH:** {trade.price_vs_pdh_pct:.2f}%")
        lines.append(f"- **Price vs PDL:** {trade.price_vs_pdl_pct:.2f}%")
        lines.append("")
        
        # Indicators
        lines.append("## Indicators at Entry")
        lines.append(f"- **RSI:** {trade.rsi:.1f}")
        lines.append(f"- **MACD Histogram:** {trade.macd_hist:.4f}")
        lines.append(f"- **ATR:** {trade.atr:.2f}")
        lines.append(f"- **EMA 9:** {trade.ema_9:.2f}")
        lines.append(f"- **EMA 20:** {trade.ema_20:.2f}")
        lines.append("")
        
        # LLM decision
        lines.append("## LLM Decision")
        lines.append(f"- **Action:** {trade.llm_action}")
        lines.append(f"- **Confidence:** {trade.llm_confidence:.0f}%")
        lines.append(f"- **Reasoning:** {trade.llm_reasoning}")
        lines.append("")
        
        # Filters
        lines.append("## Filter Status")
        lines.append(f"- **Passed:** {', '.join(trade.filters_passed) or 'None'}")
        lines.append(f"- **Blocked:** {', '.join(trade.filters_blocked) or 'None'}")
        lines.append("")
        
        # Action items
        lines.append("## Action Items")
        for pattern in analysis["patterns"]:
            info = self.PATTERNS[pattern]
            lines.append(f"- [ ] {info['advice']}")
        lines.append("")
        
        lines.append("---")
        lines.append("*Auto-generated by MistakeAnalyzer*")
        
        return "\n".join(lines)
    
    def save_mistake_note(self, trade: TradeRecord) -> Optional[str]:
        """Analyze a trade and save mistake note if it's a loss.
        
        Args:
            trade: Trade to analyze
            
        Returns:
            Path to saved note or None if not a loss
        """
        if trade.result != "LOSS":
            return None
        
        note_content = self.generate_mistake_note(trade)
        analysis = self.analyze_trade(trade)
        
        # Create a combined analysis string for storage
        analysis_text = "\n".join([
            f"Pattern: {self.PATTERNS[p]['name']} ({self.PATTERNS[p]['severity']})"
            for p in analysis["patterns"]
        ])
        
        filepath = self.storage.save_mistake_note(trade, analysis_text)
        
        logger.info(f"Saved mistake note for trade {trade.trade_id}: {filepath}")
        return filepath
    
    def get_mistake_summary(
        self,
        days: int = 30,
        min_trades: int = 5,
    ) -> Dict[str, Any]:
        """Get aggregated mistake analysis for recent trades.
        
        Args:
            days: Number of days to analyze
            min_trades: Minimum trades required for meaningful analysis
            
        Returns:
            Summary dictionary with patterns and recommendations
        """
        start_date = now_cst() - timedelta(days=days)
        trades = self.storage.load_trades(
            start_date=start_date,
            result_filter="LOSS",
            limit=500,
        )
        
        if len(trades) < min_trades:
            return {
                "status": "insufficient_data",
                "trades_analyzed": len(trades),
                "min_required": min_trades,
            }
        
        # Analyze all trades
        pattern_counts = Counter()
        severity_counts = Counter()
        total_loss = 0
        
        for trade in trades:
            analysis = self.analyze_trade(trade)
            for pattern in analysis["patterns"]:
                pattern_counts[pattern] += 1
            severity_counts[analysis["severity"]] += 1
            total_loss += abs(trade.pnl)
        
        # Get top patterns
        top_patterns = pattern_counts.most_common(5)
        
        # Generate recommendations
        recommendations = []
        for pattern, count in top_patterns:
            pct = count / len(trades) * 100
            if pct > 20:  # Pattern appears in >20% of losses
                info = self.PATTERNS[pattern]
                recommendations.append({
                    "pattern": info["name"],
                    "frequency": f"{pct:.0f}%",
                    "severity": info["severity"],
                    "advice": info["advice"],
                })
        
        return {
            "status": "complete",
            "period_days": days,
            "trades_analyzed": len(trades),
            "total_loss": total_loss,
            "avg_loss_per_trade": total_loss / len(trades),
            "severity_breakdown": dict(severity_counts),
            "top_patterns": [
                {"pattern": self.PATTERNS[p]["name"], "count": c, "pct": c/len(trades)*100}
                for p, c in top_patterns
            ],
            "recommendations": recommendations,
        }
    
    def generate_weekly_review(self) -> str:
        """Generate a weekly mistake review document.
        
        Returns:
            Markdown-formatted weekly review
        """
        summary = self.get_mistake_summary(days=7)
        
        lines = []
        lines.append(f"# Weekly Mistake Review")
        lines.append(f"*Generated: {now_cst().strftime('%Y-%m-%d %H:%M CST')}*")
        lines.append("")
        
        if summary["status"] == "insufficient_data":
            lines.append(f"Insufficient data: Only {summary['trades_analyzed']} losing trades "
                        f"(need {summary['min_required']} for analysis)")
            return "\n".join(lines)
        
        lines.append("## Overview")
        lines.append(f"- **Losing Trades:** {summary['trades_analyzed']}")
        lines.append(f"- **Total Loss:** ${summary['total_loss']:.2f}")
        lines.append(f"- **Avg Loss/Trade:** ${summary['avg_loss_per_trade']:.2f}")
        lines.append("")
        
        lines.append("## Top Mistake Patterns")
        for pattern in summary["top_patterns"][:5]:
            lines.append(f"- **{pattern['pattern']}:** {pattern['count']} trades ({pattern['pct']:.0f}%)")
        lines.append("")
        
        lines.append("## Severity Breakdown")
        for severity, count in summary["severity_breakdown"].items():
            lines.append(f"- {severity}: {count}")
        lines.append("")
        
        lines.append("## Recommendations")
        for i, rec in enumerate(summary["recommendations"], 1):
            lines.append(f"### {i}. {rec['pattern']} ({rec['frequency']} of losses)")
            lines.append(f"**Action:** {rec['advice']}")
            lines.append("")
        
        return "\n".join(lines)


# Singleton instance
_mistake_analyzer: Optional[MistakeAnalyzer] = None


def get_mistake_analyzer() -> MistakeAnalyzer:
    """Get the singleton mistake analyzer instance.
    
    Returns:
        MistakeAnalyzer instance
    """
    global _mistake_analyzer
    if _mistake_analyzer is None:
        _mistake_analyzer = MistakeAnalyzer()
    return _mistake_analyzer
