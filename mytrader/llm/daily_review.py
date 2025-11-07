"""Daily paper trading review system."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from ..utils.logger import logger
from .ai_insights import AIInsightGenerator, AIInsightReport
from .trade_analyzer import LiveTradeAnalyzer, TradeSummary


class DailyReviewOrchestrator:
    """Orchestrates daily paper trading performance review."""
    
    def __init__(
        self,
        analyzer: Optional[LiveTradeAnalyzer] = None,
        insight_generator: Optional[AIInsightGenerator] = None,
        reports_dir: Optional[Path] = None
    ):
        """Initialize daily review orchestrator.
        
        Args:
            analyzer: Live trade analyzer
            insight_generator: AI insight generator
            reports_dir: Directory for reports
        """
        self.analyzer = analyzer or LiveTradeAnalyzer()
        self.insight_generator = insight_generator or AIInsightGenerator()
        
        if reports_dir is None:
            project_root = Path(__file__).parent.parent.parent
            reports_dir = project_root / "reports" / "daily_reviews"
        
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("DailyReviewOrchestrator initialized")
    
    def run_daily_review(
        self,
        days: int = 3,
        use_database: bool = True,
        save_json: bool = True,
        save_markdown: bool = True
    ) -> Dict:
        """Run complete daily paper trading review.
        
        Args:
            days: Number of days to analyze
            use_database: Use trade database vs CSV
            save_json: Save JSON report
            save_markdown: Save markdown report
            
        Returns:
            Complete review results
        """
        logger.info("=" * 70)
        logger.info("STARTING DAILY PAPER TRADING REVIEW")
        logger.info("=" * 70)
        
        # Step 1: Analyze recent performance
        logger.info(f"\n[1/4] Analyzing last {days} days of paper trading...")
        summary, trades = self.analyzer.analyze_recent_performance(
            days=days,
            use_database=use_database
        )
        
        if summary.total_trades == 0:
            logger.warning("No trades found to analyze")
            return {
                "success": False,
                "error": "No trading data available",
                "summary": summary.to_dict()
            }
        
        # Log basic summary
        logger.info(f"\n📊 Performance Summary:")
        logger.info(f"  Trades: {summary.total_trades} ({summary.closed_trades} closed)")
        logger.info(f"  Win Rate: {summary.win_rate:.1%}")
        logger.info(f"  Total P&L: ${summary.total_pnl:,.2f}")
        logger.info(f"  Profit Factor: {summary.profit_factor:.2f}")
        logger.info(f"  Max Drawdown: ${summary.max_drawdown:,.2f}")
        
        # Step 2: Generate AI insights
        logger.info(f"\n[2/4] Generating AI-powered insights...")
        insights = self.insight_generator.generate_insights(summary, trades)
        
        # Log key insights
        logger.info(f"\n🧠 AI Analysis:")
        logger.info(f"  {insights.summary_text}")
        logger.info(f"  Observations: {len(insights.observations)}")
        logger.info(f"  Recommendations: {len(insights.recommendations)}")
        logger.info(f"  Warnings: {len(insights.warnings)}")
        
        if insights.observations:
            logger.info(f"\n  Top Observations:")
            for obs in insights.observations[:3]:
                logger.info(f"    [{obs.severity.upper()}] {obs.description}")
        
        if insights.recommendations:
            logger.info(f"\n  Top Recommendations:")
            for rec in insights.recommendations[:3]:
                logger.info(f"    • {rec.parameter}: {rec.current_value} → {rec.suggested_value}")
        
        if insights.warnings:
            logger.info(f"\n  ⚠️  Warnings:")
            for warning in insights.warnings:
                logger.info(f"    • {warning}")
        
        # Step 3: Save reports
        logger.info(f"\n[3/4] Saving reports...")
        
        saved_files = []
        
        if save_json:
            json_path = self.insight_generator.save_report(insights, format="json")
            saved_files.append(str(json_path))
            logger.info(f"  ✓ JSON report: {json_path}")
        
        if save_markdown:
            md_path = self.insight_generator.save_report(insights, format="markdown")
            saved_files.append(str(md_path))
            logger.info(f"  ✓ Markdown report: {md_path}")
        
        # Step 4: Create combined summary
        logger.info(f"\n[4/4] Creating combined summary...")
        
        combined_report = {
            "timestamp": datetime.now().isoformat(),
            "period": {
                "start": summary.period_start,
                "end": summary.period_end,
                "days": days
            },
            "performance_summary": summary.to_dict(),
            "ai_insights": insights.to_dict(),
            "report_files": saved_files
        }
        
        # Save combined report
        date_str = datetime.now().strftime("%Y-%m-%d")
        combined_path = self.reports_dir / f"daily_review_combined_{date_str}.json"
        
        with open(combined_path, "w") as f:
            json.dump(combined_report, f, indent=2, default=str)
        
        logger.info(f"  ✓ Combined report: {combined_path}")
        
        # Generate summary for human review
        logger.info(f"\n" + "=" * 70)
        logger.info("DAILY REVIEW COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\n📋 Review Summary:")
        logger.info(f"  Period: {summary.period_start} to {summary.period_end}")
        logger.info(f"  Trades Analyzed: {summary.total_trades}")
        logger.info(f"  Win Rate: {summary.win_rate:.1%}")
        logger.info(f"  Net P&L: ${summary.total_pnl:,.2f}")
        logger.info(f"  AI Insights: {len(insights.observations)} observations, {len(insights.recommendations)} recommendations")
        logger.info(f"\n📁 Reports Generated:")
        for filepath in saved_files:
            logger.info(f"  • {filepath}")
        logger.info(f"\n✅ Review complete. Please check reports for detailed analysis.")
        logger.info("=" * 70)
        
        return {
            "success": True,
            "summary": summary.to_dict(),
            "insights": insights.to_dict(),
            "report_files": saved_files,
            "combined_report_path": str(combined_path)
        }
    
    def get_best_performing_signal(self, summary: TradeSummary) -> Optional[str]:
        """Identify best performing signal type.
        
        Args:
            summary: Trade summary
            
        Returns:
            Best signal type name or None
        """
        if not summary.pnl_by_signal:
            return None
        
        return max(summary.pnl_by_signal.items(), key=lambda x: x[1])[0]
    
    def get_worst_performing_signal(self, summary: TradeSummary) -> Optional[str]:
        """Identify worst performing signal type.
        
        Args:
            summary: Trade summary
            
        Returns:
            Worst signal type name or None
        """
        if not summary.pnl_by_signal:
            return None
        
        return min(summary.pnl_by_signal.items(), key=lambda x: x[1])[0]
    
    def get_best_trading_hour(self, summary: TradeSummary) -> Optional[int]:
        """Identify best trading hour.
        
        Args:
            summary: Trade summary
            
        Returns:
            Best hour (0-23) or None
        """
        if not summary.win_rate_by_hour:
            return None
        
        return max(summary.win_rate_by_hour.items(), key=lambda x: x[1])[0]
    
    def generate_quick_summary(
        self,
        summary: TradeSummary,
        insights: AIInsightReport
    ) -> str:
        """Generate quick text summary for logging.
        
        Args:
            summary: Trade summary
            insights: AI insights
            
        Returns:
            Quick summary text
        """
        best_signal = self.get_best_performing_signal(summary)
        worst_signal = self.get_worst_performing_signal(summary)
        best_hour = self.get_best_trading_hour(summary)
        
        lines = [
            f"Paper Trading Review: {summary.period_start} to {summary.period_end}",
            f"Trades: {summary.total_trades} | Win Rate: {summary.win_rate:.1%} | P&L: ${summary.total_pnl:,.2f}",
        ]
        
        if best_signal:
            lines.append(f"Best Signal: {best_signal} (${summary.pnl_by_signal.get(best_signal, 0):,.2f})")
        
        if worst_signal:
            lines.append(f"Worst Signal: {worst_signal} (${summary.pnl_by_signal.get(worst_signal, 0):,.2f})")
        
        if best_hour is not None:
            lines.append(f"Best Hour: {best_hour:02d}:00 ({summary.win_rate_by_hour.get(best_hour, 0):.1%} WR)")
        
        if insights.recommendations:
            lines.append(f"AI Recommendations: {len(insights.recommendations)}")
        
        if insights.warnings:
            lines.append(f"⚠️  Warnings: {len(insights.warnings)}")
        
        return " | ".join(lines)


def run_daily_paper_trading_review(
    days: int = 3,
    use_database: bool = True
) -> Dict:
    """Convenience function for scheduled execution.
    
    Args:
        days: Number of days to analyze
        use_database: Use trade database vs CSV
        
    Returns:
        Review results
    """
    orchestrator = DailyReviewOrchestrator()
    return orchestrator.run_daily_review(days=days, use_database=use_database)
