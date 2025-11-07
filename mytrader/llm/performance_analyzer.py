"""Daily performance analyzer for autonomous learning."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.logger import logger


@dataclass
class DailyMetrics:
    """Daily performance metrics."""
    
    date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_holding_time_minutes: float
    total_commission: float
    net_pnl: float
    
    # Signal analysis
    rsi_signal_count: int
    macd_signal_count: int
    sentiment_signal_count: int
    momentum_signal_count: int
    
    # LLM performance
    llm_accuracy: float
    llm_avg_confidence: float
    llm_enhanced_trades: int
    
    # Risk metrics
    largest_position_size: int
    avg_position_size: float
    total_risk_exposure: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "date": self.date,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "max_win": self.max_win,
            "max_loss": self.max_loss,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "avg_holding_time_minutes": self.avg_holding_time_minutes,
            "total_commission": self.total_commission,
            "net_pnl": self.net_pnl,
            "rsi_signal_count": self.rsi_signal_count,
            "macd_signal_count": self.macd_signal_count,
            "sentiment_signal_count": self.sentiment_signal_count,
            "momentum_signal_count": self.momentum_signal_count,
            "llm_accuracy": self.llm_accuracy,
            "llm_avg_confidence": self.llm_avg_confidence,
            "llm_enhanced_trades": self.llm_enhanced_trades,
            "largest_position_size": self.largest_position_size,
            "avg_position_size": self.avg_position_size,
            "total_risk_exposure": self.total_risk_exposure,
        }


@dataclass
class TradePattern:
    """Identified trade pattern or behavioral trend."""
    
    pattern_type: str  # e.g., "overtrading", "missed_entries", "false_positives"
    description: str
    severity: str  # "low", "medium", "high"
    affected_trades: int
    impact_pnl: float
    recommendation: str


class PerformanceAnalyzer:
    """Analyzes daily trading performance and identifies patterns."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize performance analyzer.
        
        Args:
            db_path: Path to trade logger database
        """
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "data" / "llm_trades.db"
        
        self.db_path = Path(db_path)
        
        if not self.db_path.exists():
            logger.warning(f"Trade database not found: {self.db_path}")
    
    def calculate_daily_metrics(
        self,
        date: Optional[str] = None,
        commission_per_contract: float = 2.4
    ) -> Optional[DailyMetrics]:
        """Calculate comprehensive daily metrics.
        
        Args:
            date: Date to analyze (YYYY-MM-DD), defaults to today
            commission_per_contract: Commission cost per contract
            
        Returns:
            DailyMetrics object or None if no data
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        if not self.db_path.exists():
            logger.error(f"Database not found: {self.db_path}")
            return None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get all closed trades for the day
            cursor = conn.execute(
                """
                SELECT 
                    t.*,
                    l.trade_decision, l.confidence, l.reasoning, l.key_factors
                FROM trade_outcomes t
                LEFT JOIN llm_recommendations l ON t.id = l.trade_outcome_id
                WHERE DATE(t.timestamp) = ?
                    AND t.outcome IN ('WIN', 'LOSS', 'BREAKEVEN')
                ORDER BY t.timestamp
                """,
                (date,)
            )
            
            trades = [dict(row) for row in cursor.fetchall()]
        
        if not trades:
            logger.info(f"No closed trades found for {date}")
            return None
        
        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t["outcome"] == "WIN")
        losing_trades = sum(1 for t in trades if t["outcome"] == "LOSS")
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        pnls = [t["realized_pnl"] for t in trades]
        total_pnl = sum(pnls)
        
        wins = [t["realized_pnl"] for t in trades if t["outcome"] == "WIN"]
        losses = [t["realized_pnl"] for t in trades if t["outcome"] == "LOSS"]
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        max_win = max(wins) if wins else 0.0
        max_loss = min(losses) if losses else 0.0
        
        # Profit factor
        total_wins = sum(wins) if wins else 0.0
        total_losses = abs(sum(losses)) if losses else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        # Sharpe ratio (simplified daily version)
        if len(pnls) > 1:
            sharpe_ratio = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = running_max - cumulative_pnl
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # Holding times
        holding_times = [t["trade_duration_minutes"] for t in trades if t["trade_duration_minutes"]]
        avg_holding_time = np.mean(holding_times) if holding_times else 0.0
        
        # Commission
        total_commission = sum(t["quantity"] * commission_per_contract * 2 for t in trades)  # *2 for entry+exit
        net_pnl = total_pnl - total_commission
        
        # Signal analysis (parse entry_context)
        rsi_signals = 0
        macd_signals = 0
        sentiment_signals = 0
        momentum_signals = 0
        
        for trade in trades:
            if trade.get("entry_context"):
                try:
                    context = json.loads(trade["entry_context"])
                    indicators = context.get("technical_indicators", {})
                    
                    # Check which signals triggered
                    if "rsi" in indicators:
                        rsi = indicators["rsi"]
                        if rsi < 35 or rsi > 65:
                            rsi_signals += 1
                    
                    if "macd" in indicators:
                        macd_hist = indicators.get("macd_histogram", 0)
                        if abs(macd_hist) > 0.5:
                            macd_signals += 1
                    
                    if context.get("sentiment_score") is not None:
                        sentiment = context["sentiment_score"]
                        if abs(sentiment) > 0.5:
                            sentiment_signals += 1
                    
                    if context.get("momentum_score") is not None:
                        momentum = context.get("momentum_score", 0)
                        if abs(momentum) > 0.5:
                            momentum_signals += 1
                            
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Could not parse entry_context: {e}")
        
        # LLM performance
        llm_trades = [t for t in trades if t.get("trade_decision") is not None]
        llm_enhanced_trades = len(llm_trades)
        
        if llm_enhanced_trades > 0:
            # Calculate LLM accuracy (trades where LLM decision matched outcome)
            correct_predictions = 0
            confidences = []
            
            for trade in llm_trades:
                decision = trade["trade_decision"]
                outcome = trade["outcome"]
                confidence = trade.get("confidence", 0.0)
                confidences.append(confidence)
                
                # Check if LLM was correct
                if decision in ["BUY", "SELL"] and outcome == "WIN":
                    correct_predictions += 1
                elif decision == "HOLD" and outcome in ["BREAKEVEN", "LOSS"]:
                    correct_predictions += 1
            
            llm_accuracy = correct_predictions / llm_enhanced_trades
            llm_avg_confidence = np.mean(confidences)
        else:
            llm_accuracy = 0.0
            llm_avg_confidence = 0.0
        
        # Position sizing analysis
        quantities = [t["quantity"] for t in trades]
        largest_position = max(quantities) if quantities else 0
        avg_position = np.mean(quantities) if quantities else 0.0
        total_risk_exposure = sum(quantities)
        
        return DailyMetrics(
            date=date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_win=max_win,
            max_loss=max_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_holding_time_minutes=avg_holding_time,
            total_commission=total_commission,
            net_pnl=net_pnl,
            rsi_signal_count=rsi_signals,
            macd_signal_count=macd_signals,
            sentiment_signal_count=sentiment_signals,
            momentum_signal_count=momentum_signals,
            llm_accuracy=llm_accuracy,
            llm_avg_confidence=llm_avg_confidence,
            llm_enhanced_trades=llm_enhanced_trades,
            largest_position_size=largest_position,
            avg_position_size=avg_position,
            total_risk_exposure=total_risk_exposure,
        )
    
    def identify_patterns(
        self,
        metrics: DailyMetrics,
        historical_metrics: Optional[List[DailyMetrics]] = None
    ) -> List[TradePattern]:
        """Identify trading patterns and behavioral issues.
        
        Args:
            metrics: Today's metrics
            historical_metrics: Previous days' metrics for comparison
            
        Returns:
            List of identified patterns
        """
        patterns = []
        
        # Check for overtrading
        if metrics.total_trades > 15:
            patterns.append(TradePattern(
                pattern_type="overtrading",
                description=f"Executed {metrics.total_trades} trades, which may indicate overtrading",
                severity="medium" if metrics.total_trades < 25 else "high",
                affected_trades=metrics.total_trades,
                impact_pnl=metrics.total_commission,
                recommendation="Consider tightening entry criteria or increasing confidence threshold"
            ))
        
        # Check for poor win rate
        if metrics.win_rate < 0.45 and metrics.total_trades >= 5:
            patterns.append(TradePattern(
                pattern_type="low_win_rate",
                description=f"Win rate of {metrics.win_rate:.1%} is below acceptable threshold",
                severity="high" if metrics.win_rate < 0.35 else "medium",
                affected_trades=metrics.losing_trades,
                impact_pnl=sum([metrics.avg_loss * metrics.losing_trades]),
                recommendation="Review entry signals and consider more conservative thresholds"
            ))
        
        # Check for poor profit factor
        if metrics.profit_factor < 1.2 and metrics.total_trades >= 5:
            patterns.append(TradePattern(
                pattern_type="poor_profit_factor",
                description=f"Profit factor of {metrics.profit_factor:.2f} indicates avg losses too high",
                severity="high" if metrics.profit_factor < 1.0 else "medium",
                affected_trades=metrics.total_trades,
                impact_pnl=metrics.net_pnl,
                recommendation="Widen stop losses or tighten take profit targets to improve risk/reward"
            ))
        
        # Check for excessive drawdown
        if metrics.max_drawdown > 1000:
            patterns.append(TradePattern(
                pattern_type="high_drawdown",
                description=f"Max drawdown of ${metrics.max_drawdown:.2f} exceeds comfort level",
                severity="high" if metrics.max_drawdown > 2000 else "medium",
                affected_trades=metrics.total_trades,
                impact_pnl=-metrics.max_drawdown,
                recommendation="Reduce position sizes or implement tighter stop losses"
            ))
        
        # Check for LLM performance
        if metrics.llm_enhanced_trades >= 5:
            if metrics.llm_accuracy < 0.5:
                patterns.append(TradePattern(
                    pattern_type="poor_llm_accuracy",
                    description=f"LLM accuracy of {metrics.llm_accuracy:.1%} suggests model needs review",
                    severity="medium",
                    affected_trades=metrics.llm_enhanced_trades,
                    impact_pnl=metrics.net_pnl,
                    recommendation="Consider retraining LLM or adjusting confidence thresholds"
                ))
            
            if metrics.llm_avg_confidence < 0.65:
                patterns.append(TradePattern(
                    pattern_type="low_llm_confidence",
                    description=f"Average LLM confidence of {metrics.llm_avg_confidence:.1%} is low",
                    severity="low",
                    affected_trades=metrics.llm_enhanced_trades,
                    impact_pnl=0.0,
                    recommendation="LLM is uncertain; consider gathering more training data"
                ))
        
        # Signal analysis
        total_signals = (metrics.rsi_signal_count + metrics.macd_signal_count + 
                        metrics.sentiment_signal_count + metrics.momentum_signal_count)
        
        if total_signals > 0:
            sentiment_ratio = metrics.sentiment_signal_count / total_signals
            if sentiment_ratio > 0.6 and metrics.win_rate < 0.5:
                patterns.append(TradePattern(
                    pattern_type="sentiment_overreliance",
                    description=f"Sentiment signals ({sentiment_ratio:.1%}) underperforming",
                    severity="medium",
                    affected_trades=metrics.sentiment_signal_count,
                    impact_pnl=metrics.net_pnl * sentiment_ratio,
                    recommendation="Reduce sentiment weight in strategy configuration"
                ))
        
        # Compare with historical average if available
        if historical_metrics and len(historical_metrics) >= 5:
            avg_win_rate = np.mean([m.win_rate for m in historical_metrics])
            avg_pnl = np.mean([m.net_pnl for m in historical_metrics])
            
            # Performance degradation
            if metrics.win_rate < avg_win_rate - 0.15:
                patterns.append(TradePattern(
                    pattern_type="performance_degradation",
                    description=f"Win rate {metrics.win_rate:.1%} significantly below historical avg {avg_win_rate:.1%}",
                    severity="high",
                    affected_trades=metrics.total_trades,
                    impact_pnl=metrics.net_pnl - avg_pnl,
                    recommendation="Market conditions may have changed; consider recalibration"
                ))
        
        return patterns
    
    def get_historical_metrics(
        self,
        days: int = 30,
        commission_per_contract: float = 2.4
    ) -> List[DailyMetrics]:
        """Get historical daily metrics.
        
        Args:
            days: Number of days to retrieve
            commission_per_contract: Commission per contract
            
        Returns:
            List of daily metrics
        """
        metrics_list = []
        
        end_date = datetime.now()
        for i in range(days):
            date = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
            metrics = self.calculate_daily_metrics(date, commission_per_contract)
            if metrics:
                metrics_list.append(metrics)
        
        return metrics_list
    
    def generate_performance_summary(
        self,
        metrics: DailyMetrics,
        patterns: List[TradePattern]
    ) -> str:
        """Generate human-readable performance summary.
        
        Args:
            metrics: Daily metrics
            patterns: Identified patterns
            
        Returns:
            Performance summary text
        """
        summary_lines = [
            f"Daily Performance Summary - {metrics.date}",
            "=" * 50,
            "",
            "Trading Metrics:",
            f"  Total Trades: {metrics.total_trades}",
            f"  Win Rate: {metrics.win_rate:.1%} ({metrics.winning_trades}W / {metrics.losing_trades}L)",
            f"  Net P&L: ${metrics.net_pnl:,.2f} (Gross: ${metrics.total_pnl:,.2f}, Commission: ${metrics.total_commission:,.2f})",
            f"  Profit Factor: {metrics.profit_factor:.2f}",
            f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}",
            f"  Max Drawdown: ${metrics.max_drawdown:,.2f}",
            "",
            "Trade Analysis:",
            f"  Avg Win: ${metrics.avg_win:,.2f} | Max Win: ${metrics.max_win:,.2f}",
            f"  Avg Loss: ${metrics.avg_loss:,.2f} | Max Loss: ${metrics.max_loss:,.2f}",
            f"  Avg Holding Time: {metrics.avg_holding_time_minutes:.1f} minutes",
            "",
            "Signal Performance:",
            f"  RSI Signals: {metrics.rsi_signal_count}",
            f"  MACD Signals: {metrics.macd_signal_count}",
            f"  Sentiment Signals: {metrics.sentiment_signal_count}",
            f"  Momentum Signals: {metrics.momentum_signal_count}",
            "",
            "LLM Performance:",
            f"  Enhanced Trades: {metrics.llm_enhanced_trades}",
            f"  LLM Accuracy: {metrics.llm_accuracy:.1%}",
            f"  Avg Confidence: {metrics.llm_avg_confidence:.1%}",
            "",
        ]
        
        if patterns:
            summary_lines.extend([
                "Identified Patterns:",
                "-" * 50,
            ])
            for pattern in patterns:
                summary_lines.extend([
                    f"  [{pattern.severity.upper()}] {pattern.pattern_type}",
                    f"    {pattern.description}",
                    f"    Impact: {pattern.affected_trades} trades, ${pattern.impact_pnl:,.2f}",
                    f"    Recommendation: {pattern.recommendation}",
                    "",
                ])
        else:
            summary_lines.append("No significant patterns identified.")
        
        return "\n".join(summary_lines)
