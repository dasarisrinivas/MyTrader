#!/usr/bin/env python3
"""CLI script for SPY Futures daily performance review.

Analyzes SPY Futures (ES/MES) paper trading performance, generates AI insights,
and pushes results to the React dashboard.

Usage:
    # Basic review (ES, last 1 day, push to dashboard)
    python run_spy_futures_review.py
    
    # Review last 3 days
    python run_spy_futures_review.py --days 3
    
    # Analyze MES instead of ES
    python run_spy_futures_review.py --symbol MES
    
    # Skip dashboard push
    python run_spy_futures_review.py --no-dashboard
    
    # Use CSV logs
    python run_spy_futures_review.py --csv

Scheduled Execution (Linux/macOS):
    # Run daily at 6:00 PM
    0 18 * * * cd /path/to/MyTrader && python run_spy_futures_review.py
    
Scheduled Execution (Windows):
    # Use Task Scheduler - see docs/WINDOWS_TASK_SCHEDULER_SETUP.md
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mytrader.llm.spy_futures_orchestrator import SPYFuturesDailyOrchestrator
from mytrader.utils.logger import logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SPY Futures daily performance review with AI insights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic review (ES, last day)
  %(prog)s
  
  # Review last 3 days
  %(prog)s --days 3
  
  # Analyze MES (Micro E-mini)
  %(prog)s --symbol MES
  
  # Skip dashboard push
  %(prog)s --no-dashboard
  
  # Use CSV logs
  %(prog)s --csv
  
  # Full custom
  %(prog)s --days 7 --symbol ES --csv --no-dashboard

Dashboard: http://localhost:8000
Reports: reports/spy_futures_daily/
        """
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of days to analyze (default: 1)"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default="ES",
        choices=["ES", "MES", "SPY"],
        help="SPY Futures symbol to analyze (default: ES)"
    )
    
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Use CSV logs instead of database"
    )
    
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Skip pushing results to dashboard"
    )
    
    parser.add_argument(
        "--dashboard-url",
        type=str,
        default="http://localhost:8000",
        help="Dashboard API base URL (default: http://localhost:8000)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving reports locally"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    """Run SPY Futures daily review."""
    args = parse_args()
    
    # Display header
    print("\n" + "=" * 70)
    print("SPY FUTURES DAILY PERFORMANCE REVIEW")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbol: {args.symbol}")
    print(f"Analysis Period: Last {args.days} day(s)")
    print(f"Data Source: {'CSV Logs' if args.csv else 'Trade Database'}")
    print(f"Dashboard Push: {'Disabled' if args.no_dashboard else 'Enabled'}")
    print(f"Dashboard URL: {args.dashboard_url}")
    print("=" * 70 + "\n")
    
    try:
        # Initialize orchestrator
        orchestrator = SPYFuturesDailyOrchestrator(
            dashboard_url=args.dashboard_url
        )
        
        # Run review
        result = orchestrator.run_daily_review(
            days=args.days,
            use_database=not args.csv,
            symbol_filter=args.symbol,
            push_to_dashboard=not args.no_dashboard,
            save_local=not args.no_save
        )
        
        # Check result
        if not result.get("success"):
            error = result.get("error", "Unknown error")
            logger.error(f"SPY Futures review failed: {error}")
            print(f"\n❌ Review failed: {error}")
            return 1
        
        # Extract data
        performance = result.get("performance", {})
        report = result.get("report", {})
        
        # Display summary
        print("\n" + "=" * 70)
        print("REVIEW COMPLETE - DASHBOARD READY")
        print("=" * 70)
        
        # Performance highlights
        print(f"\n📊 {args.symbol} Performance Highlights:")
        print(f"  • Trades: {performance.get('total_trades', 0)}")
        print(f"  • Win Rate: {performance.get('win_rate', 0):.1%}")
        print(f"  • Net P&L: ${performance.get('total_pnl', 0):,.2f}")
        print(f"  • Profit Factor: {performance.get('profit_factor', 0):.2f}")
        print(f"  • Max Drawdown: ${performance.get('max_drawdown', 0):,.2f}")
        print(f"  • Avg Hold Time: {performance.get('average_holding_time_minutes', 0):.1f} min")
        
        # AI insights
        observations = report.get("observations", [])
        suggestions = report.get("suggestions", {})
        warnings = report.get("warnings", [])
        
        if observations:
            print(f"\n🔍 Key Observations:")
            for obs in observations[:3]:
                print(f"  • {obs}")
            if len(observations) > 3:
                print(f"  ... and {len(observations) - 3} more")
        
        if suggestions:
            print(f"\n🎯 AI Suggestions:")
            for key, value in list(suggestions.items())[:5]:
                print(f"  • {key}: {value}")
        
        if warnings:
            print(f"\n⚠️  Warnings ({len(warnings)}):")
            for warning in warnings[:3]:
                print(f"  • {warning}")
        
        # Saved files
        saved_files = result.get("saved_files", [])
        if saved_files:
            print(f"\n📁 Saved Reports:")
            for filepath in saved_files:
                print(f"  • {filepath}")
        
        # Dashboard status
        dashboard_pushed = result.get("dashboard_pushed", False)
        if not args.no_dashboard:
            if dashboard_pushed:
                print(f"\n✅ Dashboard Updated Successfully")
                print(f"   View at: {args.dashboard_url}")
                print(f"   API endpoint: {args.dashboard_url}/api/spy-futures/latest-summary")
            else:
                print(f"\n⚠️  Dashboard Push Failed")
                print(f"   Is the dashboard running? Start with:")
                print(f"   cd dashboard/backend && python dashboard_api.py")
        
        print("\n" + "=" * 70)
        print("Next Steps:")
        print("1. Open dashboard at http://localhost:8000")
        print("2. Review AI insights and recommendations")
        print("3. Evaluate suggested parameter adjustments")
        print("4. Monitor pattern analysis for trade improvements")
        print("=" * 70 + "\n")
        
        logger.info(f"SPY Futures review completed successfully for {args.symbol}")
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Review interrupted by user")
        logger.warning("SPY Futures review interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"SPY Futures review failed: {e}", exc_info=True)
        print(f"\n❌ Review failed: {e}")
        print("Check logs for details: logs/trading.log")
        return 1


if __name__ == "__main__":
    sys.exit(main())
