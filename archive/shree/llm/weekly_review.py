"""Weekly performance review and optimization module."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.logger import logger
from .adaptive_engine import AdaptiveLearningEngine, SafetyConstraints
from .bedrock_client import BedrockClient
from .config_manager import ConfigurationManager
from .performance_analyzer import DailyMetrics, PerformanceAnalyzer
from .prompt_templates import PromptTemplates


@dataclass
class WeeklyPerformance:
    """Weekly aggregated performance metrics."""
    
    start_date: str
    end_date: str
    total_trades: int
    total_wins: int
    total_losses: int
    win_rate: float
    total_pnl: float
    avg_daily_pnl: float
    best_day_pnl: float
    worst_day_pnl: float
    total_commission: float
    net_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    avg_holding_time: float
    days_traded: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "avg_daily_pnl": self.avg_daily_pnl,
            "best_day_pnl": self.best_day_pnl,
            "worst_day_pnl": self.worst_day_pnl,
            "total_commission": self.total_commission,
            "net_pnl": self.net_pnl,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
            "avg_holding_time": self.avg_holding_time,
            "days_traded": self.days_traded,
        }


class WeeklyReviewEngine:
    """Engine for weekly performance review and strategic optimization."""
    
    def __init__(
        self,
        analyzer: Optional[PerformanceAnalyzer] = None,
        bedrock_client: Optional[BedrockClient] = None,
        config_manager: Optional[ConfigurationManager] = None,
        reports_dir: Optional[Path] = None
    ):
        """Initialize weekly review engine.
        
        Args:
            analyzer: Performance analyzer instance
            bedrock_client: Bedrock client for LLM calls
            config_manager: Configuration manager
            reports_dir: Directory for weekly reports
        """
        self.analyzer = analyzer or PerformanceAnalyzer()
        self.bedrock_client = bedrock_client or BedrockClient()
        self.config_manager = config_manager or ConfigurationManager()
        
        if reports_dir is None:
            project_root = Path(__file__).parent.parent.parent
            reports_dir = project_root / "reports" / "weekly_reviews"
        
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.prompt_templates = PromptTemplates()
        
        logger.info("WeeklyReviewEngine initialized")
    
    def calculate_weekly_performance(
        self,
        end_date: Optional[str] = None,
        days: int = 7
    ) -> Tuple[Optional[WeeklyPerformance], List[DailyMetrics]]:
        """Calculate weekly aggregated performance.
        
        Args:
            end_date: End date (defaults to today)
            days: Number of days to include (default 7)
            
        Returns:
            Tuple of (weekly performance, daily metrics list)
        """
        if end_date is None:
            end_date_obj = datetime.now()
        else:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        
        start_date_obj = end_date_obj - timedelta(days=days - 1)
        
        # Collect daily metrics
        daily_metrics_list = []
        for i in range(days):
            date = (start_date_obj + timedelta(days=i)).strftime("%Y-%m-%d")
            metrics = self.analyzer.calculate_daily_metrics(date)
            if metrics:
                daily_metrics_list.append(metrics)
        
        if not daily_metrics_list:
            logger.warning("No trading data found for weekly review")
            return None, []
        
        # Aggregate metrics
        total_trades = sum(m.total_trades for m in daily_metrics_list)
        total_wins = sum(m.winning_trades for m in daily_metrics_list)
        total_losses = sum(m.losing_trades for m in daily_metrics_list)
        win_rate = total_wins / total_trades if total_trades > 0 else 0.0
        
        total_pnl = sum(m.total_pnl for m in daily_metrics_list)
        total_commission = sum(m.total_commission for m in daily_metrics_list)
        net_pnl = total_pnl - total_commission
        
        daily_pnls = [m.net_pnl for m in daily_metrics_list]
        avg_daily_pnl = np.mean(daily_pnls) if daily_pnls else 0.0
        best_day_pnl = max(daily_pnls) if daily_pnls else 0.0
        worst_day_pnl = min(daily_pnls) if daily_pnls else 0.0
        
        # Sharpe ratio (weekly)
        if len(daily_pnls) > 1:
            sharpe_ratio = np.mean(daily_pnls) / np.std(daily_pnls) if np.std(daily_pnls) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown (cumulative across week)
        cumulative_pnl = np.cumsum(daily_pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = running_max - cumulative_pnl
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # Profit factor
        wins_pnl = sum(m.avg_win * m.winning_trades for m in daily_metrics_list if m.winning_trades > 0)
        losses_pnl = abs(sum(m.avg_loss * m.losing_trades for m in daily_metrics_list if m.losing_trades > 0))
        profit_factor = wins_pnl / losses_pnl if losses_pnl > 0 else 0.0
        
        # Average holding time
        holding_times = [m.avg_holding_time_minutes for m in daily_metrics_list if m.total_trades > 0]
        avg_holding_time = np.mean(holding_times) if holding_times else 0.0
        
        weekly_perf = WeeklyPerformance(
            start_date=start_date_obj.strftime("%Y-%m-%d"),
            end_date=end_date_obj.strftime("%Y-%m-%d"),
            total_trades=total_trades,
            total_wins=total_wins,
            total_losses=total_losses,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_daily_pnl=avg_daily_pnl,
            best_day_pnl=best_day_pnl,
            worst_day_pnl=worst_day_pnl,
            total_commission=total_commission,
            net_pnl=net_pnl,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            avg_holding_time=avg_holding_time,
            days_traded=len(daily_metrics_list)
        )
        
        logger.info(
            f"Weekly performance: {total_trades} trades, ${net_pnl:.2f} P&L, "
            f"{win_rate:.1%} WR over {len(daily_metrics_list)} days"
        )
        
        return weekly_perf, daily_metrics_list
    
    def generate_weekly_review(
        self,
        weekly_perf: WeeklyPerformance,
        daily_metrics: List[DailyMetrics],
        parameter_changes: List[dict]
    ) -> str:
        """Generate LLM-powered weekly review.
        
        Args:
            weekly_perf: Weekly performance summary
            daily_metrics: Daily metrics for each day
            parameter_changes: History of parameter changes
            
        Returns:
            LLM-generated weekly review
        """
        # Prepare data for prompt
        weekly_dict = weekly_perf.to_dict()
        daily_dicts = [m.to_dict() for m in daily_metrics]
        
        # Generate prompt
        prompt = self.prompt_templates.weekly_review_prompt(
            daily_dicts,
            parameter_changes,
            weekly_dict
        )
        
        # Get LLM response
        try:
            request = self.prompt_templates.format_llm_request(prompt, temperature=0.4)
            response = self.bedrock_client.invoke_model(request)
            
            review = response.get("content", [{}])[0].get("text", "")
            logger.info("Generated weekly LLM review")
            return review
            
        except Exception as e:
            logger.error(f"Failed to generate weekly review: {e}")
            return "Weekly review generation failed."
    
    def evaluate_parameter_changes(
        self,
        daily_metrics_before: List[DailyMetrics],
        daily_metrics_after: List[DailyMetrics]
    ) -> dict:
        """Evaluate impact of parameter changes.
        
        Args:
            daily_metrics_before: Metrics before parameter change
            daily_metrics_after: Metrics after parameter change
            
        Returns:
            Evaluation dictionary
        """
        if not daily_metrics_before or not daily_metrics_after:
            return {
                "evaluation": "insufficient_data",
                "recommendation": "needs_more_data"
            }
        
        # Calculate average metrics before and after
        avg_before = {
            "win_rate": np.mean([m.win_rate for m in daily_metrics_before]),
            "net_pnl": np.mean([m.net_pnl for m in daily_metrics_before]),
            "profit_factor": np.mean([m.profit_factor for m in daily_metrics_before]),
            "sharpe_ratio": np.mean([m.sharpe_ratio for m in daily_metrics_before]),
            "trades_per_day": np.mean([m.total_trades for m in daily_metrics_before]),
        }
        
        avg_after = {
            "win_rate": np.mean([m.win_rate for m in daily_metrics_after]),
            "net_pnl": np.mean([m.net_pnl for m in daily_metrics_after]),
            "profit_factor": np.mean([m.profit_factor for m in daily_metrics_after]),
            "sharpe_ratio": np.mean([m.sharpe_ratio for m in daily_metrics_after]),
            "trades_per_day": np.mean([m.total_trades for m in daily_metrics_after]),
        }
        
        # Calculate improvements
        improvements = {
            "win_rate_change": avg_after["win_rate"] - avg_before["win_rate"],
            "pnl_change": avg_after["net_pnl"] - avg_before["net_pnl"],
            "profit_factor_change": avg_after["profit_factor"] - avg_before["profit_factor"],
            "sharpe_change": avg_after["sharpe_ratio"] - avg_before["sharpe_ratio"],
            "trades_change": avg_after["trades_per_day"] - avg_before["trades_per_day"],
        }
        
        # Determine if changes were beneficial
        beneficial_count = sum([
            improvements["win_rate_change"] > 0.02,  # 2% improvement
            improvements["pnl_change"] > 50,  # $50 improvement
            improvements["profit_factor_change"] > 0.1,
            improvements["sharpe_change"] > 0.1,
        ])
        
        if beneficial_count >= 3:
            evaluation = "highly_beneficial"
            recommendation = "keep_changes"
        elif beneficial_count >= 2:
            evaluation = "moderately_beneficial"
            recommendation = "keep_changes"
        elif improvements["pnl_change"] < -100:  # Worse by $100+
            evaluation = "detrimental"
            recommendation = "rollback"
        else:
            evaluation = "neutral"
            recommendation = "monitor_further"
        
        return {
            "evaluation": evaluation,
            "recommendation": recommendation,
            "metrics_before": avg_before,
            "metrics_after": avg_after,
            "improvements": improvements,
            "beneficial_count": beneficial_count
        }
    
    def run_weekly_review(
        self,
        end_date: Optional[str] = None,
        days: int = 7,
        save_report: bool = True
    ) -> Dict[str, any]:
        """Run complete weekly review cycle.
        
        Args:
            end_date: End date for review
            days: Number of days to review
            save_report: Whether to save report to file
            
        Returns:
            Dictionary with review results
        """
        logger.info(f"Starting weekly review for last {days} days")
        
        # Calculate weekly performance
        weekly_perf, daily_metrics = self.calculate_weekly_performance(end_date, days)
        
        if weekly_perf is None:
            return {
                "success": False,
                "error": "No trading data available for weekly review"
            }
        
        # Get parameter change history
        parameter_changes = self.config_manager.get_update_history(limit=20)
        
        # Filter changes to this week
        start_date = datetime.strptime(weekly_perf.start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(weekly_perf.end_date, "%Y-%m-%d")
        
        weekly_changes = [
            change for change in parameter_changes
            if "timestamp" in change and
            start_date <= datetime.fromisoformat(change["timestamp"]) <= end_date_obj
        ]
        
        # Generate LLM review
        llm_review = self.generate_weekly_review(
            weekly_perf,
            daily_metrics,
            weekly_changes
        )
        
        # Evaluate parameter changes if any were made
        change_evaluation = None
        if weekly_changes:
            # Get metrics before first change
            first_change_date = datetime.fromisoformat(weekly_changes[0]["timestamp"])
            metrics_before = self.analyzer.get_historical_metrics(days=3)  # 3 days before
            metrics_after = [m for m in daily_metrics if datetime.strptime(m.date, "%Y-%m-%d") >= first_change_date]
            
            if metrics_before and metrics_after:
                change_evaluation = self.evaluate_parameter_changes(metrics_before, metrics_after)
        
        # Compile report
        report = {
            "success": True,
            "period": {
                "start_date": weekly_perf.start_date,
                "end_date": weekly_perf.end_date,
                "days": days
            },
            "weekly_performance": weekly_perf.to_dict(),
            "daily_breakdown": [m.to_dict() for m in daily_metrics],
            "parameter_changes": weekly_changes,
            "change_evaluation": change_evaluation,
            "llm_review": llm_review,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save report if requested
        if save_report:
            report_path = self._save_report(report, weekly_perf.end_date)
            report["report_path"] = str(report_path)
            logger.info(f"Weekly report saved to: {report_path}")
        
        logger.info("Weekly review complete")
        
        return report
    
    def _save_report(self, report: dict, end_date: str) -> Path:
        """Save weekly report to file.
        
        Args:
            report: Report dictionary
            end_date: End date for filename
            
        Returns:
            Path to saved report
        """
        filename = f"weekly_review_{end_date}.json"
        report_path = self.reports_dir / filename
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        return report_path
    
    def suggest_weekly_adjustments(
        self,
        weekly_perf: WeeklyPerformance,
        daily_metrics: List[DailyMetrics],
        current_config: dict
    ) -> Tuple[List, str]:
        """Suggest parameter adjustments based on weekly review.
        
        Args:
            weekly_perf: Weekly performance
            daily_metrics: Daily metrics
            current_config: Current configuration
            
        Returns:
            Tuple of (adjustment suggestions, reasoning)
        """
        # Use adaptive engine to generate suggestions
        from .adaptive_engine import AdaptiveLearningEngine
        
        engine = AdaptiveLearningEngine(
            bedrock_client=self.bedrock_client,
            analyzer=self.analyzer
        )
        
        # Use most recent day's metrics for analysis
        if daily_metrics:
            latest_metrics = daily_metrics[-1]
            patterns = self.analyzer.identify_patterns(latest_metrics, daily_metrics[:-1])
            
            # Generate self-assessment
            self_assessment = engine.generate_self_assessment(
                latest_metrics,
                patterns,
                current_config
            )
            
            # Get suggestions
            adjustments, reasoning = engine.suggest_parameter_adjustments(
                self_assessment,
                latest_metrics,
                patterns,
                current_config
            )
            
            return adjustments, reasoning
        
        return [], "Insufficient data for weekly adjustments"
