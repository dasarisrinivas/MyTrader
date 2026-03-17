"""SPY Futures Daily Review Orchestrator.

Coordinates daily SPY Futures performance analysis, LLM insights generation,
and dashboard integration.
"""
from __future__ import annotations

import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from ..utils.logger import logger
from .spy_futures_analyzer import SPYFuturesAnalyzer
from .spy_futures_insights import SPYFuturesInsightGenerator


class SPYFuturesDailyOrchestrator:
    """Orchestrates daily SPY Futures performance review and dashboard push."""
    
    def __init__(
        self,
        analyzer: Optional[SPYFuturesAnalyzer] = None,
        insight_generator: Optional[SPYFuturesInsightGenerator] = None,
        dashboard_url: str = "http://localhost:8000"
    ):
        """Initialize orchestrator.
        
        Args:
            analyzer: SPY Futures analyzer
            insight_generator: Insight generator
            dashboard_url: Dashboard API base URL
        """
        self.analyzer = analyzer or SPYFuturesAnalyzer()
        self.insight_generator = insight_generator or SPYFuturesInsightGenerator()
        self.dashboard_url = dashboard_url.rstrip('/')
        
        logger.info("SPYFuturesDailyOrchestrator initialized")
        logger.info(f"  Dashboard URL: {self.dashboard_url}")
    
    def run_daily_review(
        self,
        days: int = 1,
        use_database: bool = True,
        symbol_filter: str = "ES",
        push_to_dashboard: bool = True,
        save_local: bool = True
    ) -> Dict:
        """Run complete SPY Futures daily review.
        
        Args:
            days: Number of days to analyze
            use_database: Use database vs CSV
            symbol_filter: SPY symbol filter (ES, MES, SPY)
            push_to_dashboard: Push results to dashboard API
            save_local: Save results locally
            
        Returns:
            Complete review results
        """
        logger.info("=" * 70)
        logger.info("STARTING SPY FUTURES DAILY REVIEW")
        logger.info("=" * 70)
        logger.info(f"Symbol: {symbol_filter}")
        logger.info(f"Period: Last {days} day(s)")
        logger.info(f"Dashboard Push: {'Enabled' if push_to_dashboard else 'Disabled'}")
        
        # Step 1: Analyze performance
        logger.info(f"\n[1/4] Analyzing SPY Futures performance...")
        performance, trades = self.analyzer.analyze_daily_performance(
            days=days,
            use_database=use_database,
            symbol_filter=symbol_filter
        )
        
        if performance.total_trades == 0:
            logger.warning("No SPY Futures trades found")
            return {
                "success": False,
                "error": "No SPY Futures trading data available",
                "performance": performance.to_dict()
            }
        
        # Log performance summary
        logger.info(f"\n📊 SPY Futures Performance:")
        logger.info(f"  Symbol: {performance.symbol}")
        logger.info(f"  Trades: {performance.total_trades} ({performance.closed_trades} closed)")
        logger.info(f"  Win Rate: {performance.win_rate:.1%}")
        logger.info(f"  Total P&L: ${performance.total_pnl:,.2f}")
        logger.info(f"  Profit Factor: {performance.profit_factor:.2f}")
        logger.info(f"  Max Drawdown: ${performance.max_drawdown:,.2f}")
        logger.info(f"  Avg Holding Time: {performance.average_holding_time_minutes:.1f} minutes")
        
        # Step 2: Generate LLM insights
        logger.info(f"\n[2/4] Generating AI-powered insights...")
        report = self.insight_generator.generate_insights(performance, trades)
        
        # Log insights summary
        logger.info(f"\n🧠 AI Analysis:")
        logger.info(f"  Observations: {len(report.observations)}")
        logger.info(f"  Insights: {len(report.insights)}")
        logger.info(f"  Suggestions: {len(report.suggestions)}")
        logger.info(f"  Warnings: {len(report.warnings)}")
        
        if report.observations:
            logger.info(f"\n  Key Observations:")
            for obs in report.observations[:3]:
                logger.info(f"    • {obs}")
        
        if report.warnings:
            logger.info(f"\n  ⚠️  Warnings:")
            for warning in report.warnings:
                logger.info(f"    • {warning}")
        
        # Step 3: Save locally
        saved_files = []
        if save_local:
            logger.info(f"\n[3/4] Saving reports locally...")
            
            # Save report
            report_path = self.insight_generator.save_report(report)
            saved_files.append(str(report_path))
            logger.info(f"  ✓ Saved report: {report_path}")
            
            # Save for dashboard review
            project_root = Path(__file__).parent.parent.parent
            review_dir = project_root / "reports" / "dashboard_ready"
            review_dir.mkdir(parents=True, exist_ok=True)
            
            review_file = review_dir / f"spy_futures_{report.date}.json"
            with open(review_file, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            saved_files.append(str(review_file))
            logger.info(f"  ✓ Dashboard-ready: {review_file}")
        
        # Step 4: Push to dashboard
        dashboard_push_success = False
        if push_to_dashboard:
            logger.info(f"\n[4/4] Pushing to dashboard...")
            try:
                dashboard_push_success = self._push_to_dashboard(report)
                if dashboard_push_success:
                    logger.info(f"  ✓ Successfully pushed to dashboard")
                else:
                    logger.warning(f"  ⚠️  Dashboard push failed (check logs)")
            except Exception as e:
                logger.error(f"  ❌ Dashboard push error: {e}")
        
        # Generate summary
        logger.info(f"\n" + "=" * 70)
        logger.info("SPY FUTURES DAILY REVIEW COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\n📋 Review Summary:")
        logger.info(f"  Date: {report.date}")
        logger.info(f"  Symbol: {report.symbol}")
        logger.info(f"  Trades: {performance.total_trades}")
        logger.info(f"  Win Rate: {performance.win_rate:.1%}")
        logger.info(f"  Net P&L: ${performance.total_pnl:,.2f}")
        logger.info(f"  AI Insights: {len(report.insights)} insights")
        
        if report.suggestions:
            logger.info(f"\n🎯 Top Suggestions:")
            for key, value in list(report.suggestions.items())[:3]:
                logger.info(f"    • {key}: {value}")
        
        if saved_files:
            logger.info(f"\n📁 Saved Reports:")
            for filepath in saved_files:
                logger.info(f"  • {filepath}")
        
        if push_to_dashboard:
            status = "✅ Pushed" if dashboard_push_success else "❌ Failed"
            logger.info(f"\n📊 Dashboard: {status}")
            if dashboard_push_success:
                logger.info(f"  View at: {self.dashboard_url}")
        
        logger.info("=" * 70)
        
        return {
            "success": True,
            "date": report.date,
            "symbol": report.symbol,
            "performance": performance.to_dict(),
            "report": report.to_dict(),
            "saved_files": saved_files,
            "dashboard_pushed": dashboard_push_success
        }
    
    def _push_to_dashboard(self, report) -> bool:
        """Push report to dashboard API.
        
        Args:
            report: Daily report
            
        Returns:
            True if successful, False otherwise
        """
        try:
            endpoint = f"{self.dashboard_url}/api/trading-summary"
            
            # Prepare payload
            payload = {
                "date": report.date,
                "performance": report.performance,
                "observations": report.observations,
                "suggestions": report.suggestions,
                "warnings": report.warnings,
                "insights": [i.to_dict() for i in report.insights],
                "profitable_patterns": report.profitable_patterns,
                "losing_patterns": report.losing_patterns
            }
            
            # Make request
            response = requests.post(
                endpoint,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Dashboard push successful")
                return True
            else:
                logger.warning(f"Dashboard returned status {response.status_code}: {response.text}")
                return False
        
        except requests.exceptions.ConnectionError:
            logger.warning("Could not connect to dashboard API (is it running?)")
            return False
        
        except Exception as e:
            logger.error(f"Dashboard push failed: {e}", exc_info=True)
            return False
    
    def get_market_hours_status(self) -> Dict:
        """Get SPY Futures market hours status.
        
        Returns:
            Market status information
        """
        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()  # 0 = Monday, 6 = Sunday
        
        # SPY Futures trade nearly 24/5
        # Regular hours: 9:30 AM - 4:00 PM ET (14:30 - 21:00 UTC)
        # Extended hours: 6:00 PM Sun - 5:00 PM Fri ET
        
        # Simplified check
        is_weekend = weekday >= 5  # Saturday or Sunday
        is_regular_hours = 9 <= hour < 16  # 9 AM - 4 PM local
        
        if is_weekend:
            status = "closed"
            message = "Weekend - Market Closed"
        elif is_regular_hours:
            status = "open"
            message = "Regular Trading Hours"
        else:
            status = "extended"
            message = "Extended Hours Trading"
        
        return {
            "status": status,
            "message": message,
            "timestamp": now.isoformat(),
            "weekday": weekday,
            "hour": hour
        }


def run_spy_futures_daily_review(
    days: int = 1,
    use_database: bool = True,
    symbol: str = "ES",
    push_to_dashboard: bool = True
) -> Dict:
    """Convenience function for scheduled execution.
    
    Args:
        days: Number of days to analyze
        use_database: Use database vs CSV
        symbol: SPY symbol (ES, MES, SPY)
        push_to_dashboard: Push to dashboard
        
    Returns:
        Review results
    """
    orchestrator = SPYFuturesDailyOrchestrator()
    return orchestrator.run_daily_review(
        days=days,
        use_database=use_database,
        symbol_filter=symbol,
        push_to_dashboard=push_to_dashboard
    )
