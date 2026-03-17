"""Autonomous trading orchestrator - main coordination system."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.logger import logger
from .adaptive_engine import AdaptiveLearningEngine, SafetyConstraints, StrategyAdjustment
from .bedrock_client import BedrockClient
from .config_manager import ConfigurationManager
from .performance_analyzer import PerformanceAnalyzer
from .weekly_review import WeeklyReviewEngine


class AutonomousTradingOrchestrator:
    """Main orchestrator for semi-autonomous trading system."""
    
    def __init__(
        self,
        require_human_approval: bool = True,
        auto_approve_threshold: float = 0.85,
        enable_auto_rollback: bool = True,
        rollback_threshold_pnl: float = -500.0,  # Auto-rollback if P&L drops by $500+
        safety_constraints: Optional[SafetyConstraints] = None,
        reports_dir: Optional[Path] = None
    ):
        """Initialize autonomous trading orchestrator.
        
        Args:
            require_human_approval: Require human approval for parameter changes
            auto_approve_threshold: Confidence threshold for auto-approval
            enable_auto_rollback: Enable automatic rollback on poor performance
            rollback_threshold_pnl: P&L threshold for auto-rollback
            safety_constraints: Safety constraints for adjustments
            reports_dir: Directory for reports
        """
        self.require_human_approval = require_human_approval
        self.auto_approve_threshold = auto_approve_threshold
        self.enable_auto_rollback = enable_auto_rollback
        self.rollback_threshold_pnl = rollback_threshold_pnl
        
        # Initialize components
        self.bedrock_client = BedrockClient()
        self.analyzer = PerformanceAnalyzer()
        self.config_manager = ConfigurationManager()
        self.adaptive_engine = AdaptiveLearningEngine(
            bedrock_client=self.bedrock_client,
            analyzer=self.analyzer,
            safety_constraints=safety_constraints,
            auto_approve_threshold=auto_approve_threshold,
            require_human_approval=require_human_approval
        )
        self.weekly_engine = WeeklyReviewEngine(
            analyzer=self.analyzer,
            bedrock_client=self.bedrock_client,
            config_manager=self.config_manager,
            reports_dir=reports_dir
        )
        
        if reports_dir is None:
            project_root = Path(__file__).parent.parent.parent
            reports_dir = project_root / "reports" / "autonomous"
        
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"AutonomousTradingOrchestrator initialized "
            f"(human_approval={require_human_approval}, auto_rollback={enable_auto_rollback})"
        )
    
    def run_daily_analysis_and_learning(
        self,
        date: Optional[str] = None,
        apply_changes: bool = False,
        dry_run: bool = False
    ) -> Dict:
        """Run daily analysis, learning, and optional parameter adjustment.
        
        Args:
            date: Date to analyze (defaults to today)
            apply_changes: Whether to apply approved adjustments
            dry_run: Simulate changes without applying
            
        Returns:
            Dictionary with complete analysis results
        """
        logger.info("=" * 60)
        logger.info("STARTING DAILY AUTONOMOUS ANALYSIS CYCLE")
        logger.info("=" * 60)
        
        # Get current configuration
        current_config = self.config_manager.get_current_strategy_params()
        
        # Run daily learning cycle
        analysis_result = self.adaptive_engine.run_daily_cycle(current_config, date)
        
        if not analysis_result["success"]:
            logger.error(f"Daily analysis failed: {analysis_result.get('error')}")
            return analysis_result
        
        # Save daily report
        report_path = self._save_daily_report(analysis_result)
        analysis_result["report_path"] = str(report_path)
        
        logger.info("\n" + "=" * 60)
        logger.info("DAILY PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(analysis_result["automated_summary"])
        
        logger.info("\n" + "=" * 60)
        logger.info("LLM PERFORMANCE ANALYSIS")
        logger.info("=" * 60)
        logger.info(analysis_result["llm_summary"])
        
        logger.info("\n" + "=" * 60)
        logger.info("SELF-ASSESSMENT")
        logger.info("=" * 60)
        logger.info(analysis_result["self_assessment"])
        
        # Handle adjustments
        adjustments = [
            StrategyAdjustment(**adj) for adj in analysis_result["suggested_adjustments"]
        ]
        
        if adjustments:
            logger.info("\n" + "=" * 60)
            logger.info("SUGGESTED PARAMETER ADJUSTMENTS")
            logger.info("=" * 60)
            for adj in adjustments:
                status = "✓ AUTO-APPROVED" if adj.approved else "⚠ REQUIRES APPROVAL"
                logger.info(
                    f"{status} | {adj.parameter}: {adj.old_value} → {adj.new_value} "
                    f"(confidence: {adj.confidence:.1%}, risk: {adj.risk_level})"
                )
            logger.info(f"\nReasoning: {analysis_result['adjustment_reasoning']}")
            
            # Apply changes if requested
            if apply_changes:
                if self.require_human_approval:
                    logger.warning(
                        "Human approval required. Use approve_and_apply_adjustments() "
                        "to manually approve and apply changes."
                    )
                    analysis_result["action_required"] = "human_approval_needed"
                else:
                    # Auto-apply approved adjustments
                    application_result = self.config_manager.apply_adjustments(
                        adjustments,
                        performance_before=analysis_result["metrics"],
                        dry_run=dry_run
                    )
                    analysis_result["application_result"] = application_result
                    
                    if application_result["success"]:
                        logger.info(
                            f"✓ Applied {application_result['applied_count']} parameter changes"
                        )
                    else:
                        logger.error(f"✗ Failed to apply changes: {application_result['message']}")
        else:
            logger.info("\n" + "=" * 60)
            logger.info("No parameter adjustments suggested - performance is stable")
            logger.info("=" * 60)
        
        logger.info("\n" + "=" * 60)
        logger.info("DAILY CYCLE COMPLETE")
        logger.info("=" * 60)
        
        return analysis_result
    
    def approve_and_apply_adjustments(
        self,
        adjustments: List[StrategyAdjustment],
        approval_indices: Optional[List[int]] = None,
        dry_run: bool = False
    ) -> Dict:
        """Manually approve and apply parameter adjustments.
        
        Args:
            adjustments: List of suggested adjustments
            approval_indices: Indices of adjustments to approve (None = all)
            dry_run: Simulate without applying
            
        Returns:
            Application result dictionary
        """
        if approval_indices is None:
            # Approve all
            for adj in adjustments:
                adj.approved = True
        else:
            # Approve selected
            for idx in approval_indices:
                if 0 <= idx < len(adjustments):
                    adjustments[idx].approved = True
        
        # Get current metrics before applying
        metrics = self.analyzer.calculate_daily_metrics()
        performance_before = metrics.to_dict() if metrics else None
        
        # Apply adjustments
        result = self.config_manager.apply_adjustments(
            adjustments,
            performance_before=performance_before,
            dry_run=dry_run
        )
        
        if result["success"]:
            logger.info(f"Successfully applied {result['applied_count']} adjustments")
        else:
            logger.error(f"Failed to apply adjustments: {result['message']}")
        
        return result
    
    def check_and_rollback_if_needed(
        self,
        days_since_change: int = 1
    ) -> Dict:
        """Check recent performance and rollback if degraded significantly.
        
        Args:
            days_since_change: Number of days since last parameter change
            
        Returns:
            Dictionary with rollback decision and results
        """
        if not self.enable_auto_rollback:
            return {
                "rollback_checked": True,
                "rollback_needed": False,
                "reason": "Auto-rollback disabled"
            }
        
        logger.info("Checking if rollback is needed...")
        
        # Get recent performance
        recent_metrics = self.analyzer.get_historical_metrics(days=days_since_change)
        
        if not recent_metrics:
            return {
                "rollback_checked": True,
                "rollback_needed": False,
                "reason": "No recent trading data"
            }
        
        # Calculate total P&L since change
        total_pnl = sum(m.net_pnl for m in recent_metrics)
        
        logger.info(f"Performance since last change: ${total_pnl:.2f} over {len(recent_metrics)} days")
        
        # Check if rollback is needed
        if total_pnl < self.rollback_threshold_pnl:
            logger.warning(
                f"⚠ Performance degraded significantly (${total_pnl:.2f} < ${self.rollback_threshold_pnl:.2f})"
            )
            logger.warning("Initiating automatic rollback...")
            
            # Perform rollback
            rollback_success = self.config_manager.rollback_last_update()
            
            if rollback_success:
                logger.info("✓ Successfully rolled back to previous configuration")
                return {
                    "rollback_checked": True,
                    "rollback_needed": True,
                    "rollback_executed": True,
                    "reason": f"Performance degraded by ${total_pnl:.2f}",
                    "threshold": self.rollback_threshold_pnl
                }
            else:
                logger.error("✗ Rollback failed")
                return {
                    "rollback_checked": True,
                    "rollback_needed": True,
                    "rollback_executed": False,
                    "reason": "Rollback execution failed"
                }
        else:
            logger.info(f"✓ Performance acceptable (${total_pnl:.2f}), no rollback needed")
            return {
                "rollback_checked": True,
                "rollback_needed": False,
                "total_pnl": total_pnl,
                "threshold": self.rollback_threshold_pnl
            }
    
    def run_weekly_review_and_optimization(
        self,
        end_date: Optional[str] = None,
        apply_weekly_suggestions: bool = False
    ) -> Dict:
        """Run weekly performance review and suggest strategic adjustments.
        
        Args:
            end_date: End date for review
            apply_weekly_suggestions: Whether to apply weekly suggestions
            
        Returns:
            Weekly review results
        """
        logger.info("=" * 60)
        logger.info("STARTING WEEKLY REVIEW AND OPTIMIZATION")
        logger.info("=" * 60)
        
        # Run weekly review
        review_result = self.weekly_engine.run_weekly_review(
            end_date=end_date,
            days=7,
            save_report=True
        )
        
        if not review_result["success"]:
            logger.error(f"Weekly review failed: {review_result.get('error')}")
            return review_result
        
        # Display results
        weekly_perf = review_result["weekly_performance"]
        logger.info("\n" + "=" * 60)
        logger.info("WEEKLY PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Period: {weekly_perf['start_date']} to {weekly_perf['end_date']}")
        logger.info(f"Total Trades: {weekly_perf['total_trades']}")
        logger.info(f"Win Rate: {weekly_perf['win_rate']:.1%}")
        logger.info(f"Net P&L: ${weekly_perf['net_pnl']:,.2f}")
        logger.info(f"Sharpe Ratio: {weekly_perf['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: ${weekly_perf['max_drawdown']:,.2f}")
        
        logger.info("\n" + "=" * 60)
        logger.info("LLM WEEKLY REVIEW")
        logger.info("=" * 60)
        logger.info(review_result["llm_review"])
        
        # Evaluate parameter changes if any were made
        if review_result.get("change_evaluation"):
            eval_result = review_result["change_evaluation"]
            logger.info("\n" + "=" * 60)
            logger.info("PARAMETER CHANGE EVALUATION")
            logger.info("=" * 60)
            logger.info(f"Evaluation: {eval_result['evaluation']}")
            logger.info(f"Recommendation: {eval_result['recommendation']}")
            
            if eval_result.get("improvements"):
                logger.info("\nPerformance Changes:")
                for metric, change in eval_result["improvements"].items():
                    logger.info(f"  {metric}: {change:+.2f}")
            
            # Handle rollback recommendation
            if eval_result["recommendation"] == "rollback":
                logger.warning("⚠ Weekly review recommends rolling back recent changes")
                if self.enable_auto_rollback:
                    rollback_result = self.config_manager.rollback_last_update()
                    review_result["auto_rollback"] = rollback_result
        
        # Get weekly adjustment suggestions
        current_config = self.config_manager.get_current_strategy_params()
        from .adaptive_engine import StrategyAdjustment
        
        # Convert daily metrics back to objects
        from .performance_analyzer import DailyMetrics
        daily_metrics = [DailyMetrics(**m) for m in review_result["daily_breakdown"]]
        
        from .weekly_review import WeeklyPerformance
        weekly_perf_obj = WeeklyPerformance(**weekly_perf)
        
        weekly_adjustments, reasoning = self.weekly_engine.suggest_weekly_adjustments(
            weekly_perf_obj,
            daily_metrics,
            current_config
        )
        
        if weekly_adjustments:
            logger.info("\n" + "=" * 60)
            logger.info("WEEKLY ADJUSTMENT SUGGESTIONS")
            logger.info("=" * 60)
            for adj in weekly_adjustments:
                logger.info(
                    f"  {adj.parameter}: {adj.old_value} → {adj.new_value} "
                    f"(confidence: {adj.confidence:.1%})"
                )
            logger.info(f"\nReasoning: {reasoning}")
            
            review_result["weekly_adjustments"] = [adj.to_dict() for adj in weekly_adjustments]
            
            if apply_weekly_suggestions and not self.require_human_approval:
                application_result = self.config_manager.apply_adjustments(
                    weekly_adjustments,
                    performance_before=weekly_perf
                )
                review_result["application_result"] = application_result
        
        logger.info("\n" + "=" * 60)
        logger.info("WEEKLY REVIEW COMPLETE")
        logger.info("=" * 60)
        
        return review_result
    
    def _save_daily_report(self, analysis_result: dict) -> Path:
        """Save daily analysis report.
        
        Args:
            analysis_result: Analysis result dictionary
            
        Returns:
            Path to saved report
        """
        date = analysis_result.get("date", datetime.now().strftime("%Y-%m-%d"))
        filename = f"daily_analysis_{date}.json"
        report_path = self.reports_dir / filename
        
        with open(report_path, "w") as f:
            json.dump(analysis_result, f, indent=2, default=str)
        
        logger.info(f"Daily report saved: {report_path}")
        return report_path
    
    def get_system_status(self) -> Dict:
        """Get current system status and recent performance.
        
        Returns:
            System status dictionary
        """
        # Get recent metrics
        recent_metrics = self.analyzer.get_historical_metrics(days=7)
        
        if recent_metrics:
            avg_daily_pnl = sum(m.net_pnl for m in recent_metrics) / len(recent_metrics)
            avg_win_rate = sum(m.win_rate for m in recent_metrics) / len(recent_metrics)
            total_trades = sum(m.total_trades for m in recent_metrics)
        else:
            avg_daily_pnl = 0.0
            avg_win_rate = 0.0
            total_trades = 0
        
        # Get recent parameter changes
        recent_changes = self.config_manager.get_update_history(limit=5)
        
        # Get current config
        current_params = self.config_manager.get_current_strategy_params()
        
        return {
            "status": "operational",
            "recent_performance": {
                "days": len(recent_metrics),
                "avg_daily_pnl": avg_daily_pnl,
                "avg_win_rate": avg_win_rate,
                "total_trades": total_trades
            },
            "recent_parameter_changes": len(recent_changes),
            "current_parameters": current_params,
            "configuration": {
                "human_approval_required": self.require_human_approval,
                "auto_rollback_enabled": self.enable_auto_rollback,
                "rollback_threshold": self.rollback_threshold_pnl
            },
            "timestamp": datetime.now().isoformat()
        }


# Convenience function for scheduled daily runs
def run_daily_autonomous_cycle(
    apply_changes: bool = False,
    dry_run: bool = True
) -> Dict:
    """Run daily autonomous cycle (for scheduled execution).
    
    Args:
        apply_changes: Whether to apply parameter changes
        dry_run: Simulate changes without applying
        
    Returns:
        Analysis results
    """
    orchestrator = AutonomousTradingOrchestrator(
        require_human_approval=True,  # Safe default
        enable_auto_rollback=True
    )
    
    return orchestrator.run_daily_analysis_and_learning(
        apply_changes=apply_changes,
        dry_run=dry_run
    )


# Convenience function for weekly reviews
def run_weekly_autonomous_review() -> Dict:
    """Run weekly review (for scheduled execution).
    
    Returns:
        Weekly review results
    """
    orchestrator = AutonomousTradingOrchestrator(
        require_human_approval=True,
        enable_auto_rollback=True
    )
    
    return orchestrator.run_weekly_review_and_optimization(
        apply_weekly_suggestions=False
    )
