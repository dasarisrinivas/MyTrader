#!/usr/bin/env python3
"""CLI script for running daily paper trading review.

This script analyzes recent paper trading performance, generates AI-powered
insights, and produces comprehensive reports. Designed for scheduled execution
after market close.

Usage:
    # Analyze last 3 days with default settings
    python run_daily_review.py
    
    # Analyze last 7 days
    python run_daily_review.py --days 7
    
    # Use CSV logs instead of database
    python run_daily_review.py --csv
    
    # Skip markdown report generation
    python run_daily_review.py --no-markdown
    
Scheduled Execution (Linux/macOS):
    # Add to crontab for 6:00 PM daily execution
    0 18 * * * cd /path/to/MyTrader && python run_daily_review.py
    
Scheduled Execution (Windows):
    # Use Task Scheduler to run daily at 6:00 PM
    # Action: python.exe
    # Arguments: run_daily_review.py
    # Start in: C:\\path\\to\\MyTrader
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mytrader.llm.daily_review import DailyReviewOrchestrator
from mytrader.utils.logger import logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run daily paper trading review with AI insights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic daily review (last 3 days)
  %(prog)s
  
  # Review last week
  %(prog)s --days 7
  
  # Use CSV logs instead of database
  %(prog)s --csv
  
  # Skip markdown report
  %(prog)s --no-markdown
  
  # JSON only with 5 days
  %(prog)s --days 5 --no-markdown
  
Reports are saved to: reports/daily_reviews/
        """
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Number of days to analyze (default: 3)"
    )
    
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Use CSV logs instead of database"
    )
    
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip JSON report generation"
    )
    
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip Markdown report generation"
    )
    
    parser.add_argument(
        "--reports-dir",
        type=Path,
        help="Custom directory for reports"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    """Run daily paper trading review."""
    args = parse_args()
    
    # Display header
    print("\n" + "=" * 70)
    print("PAPER TRADING DAILY REVIEW")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analysis Period: Last {args.days} days")
    print(f"Data Source: {'CSV Logs' if args.csv else 'Trade Database'}")
    print("=" * 70 + "\n")
    
    try:
        # Initialize orchestrator
        orchestrator = DailyReviewOrchestrator(reports_dir=args.reports_dir)
        
        # Run daily review
        result = orchestrator.run_daily_review(
            days=args.days,
            use_database=not args.csv,
            save_json=not args.no_json,
            save_markdown=not args.no_markdown
        )
        
        # Check result
        if not result.get("success"):
            error = result.get("error", "Unknown error")
            logger.error(f"Daily review failed: {error}")
            print(f"\n❌ Daily review failed: {error}")
            return 1
        
        # Display summary
        summary = result.get("summary", {})
        insights = result.get("insights", {})
        
        print("\n" + "=" * 70)
        print("REVIEW COMPLETE - ACTION ITEMS")
        print("=" * 70)
        
        # Performance highlights
        print("\n📊 Performance Highlights:")
        print(f"  • Trades: {summary.get('total_trades', 0)}")
        print(f"  • Win Rate: {summary.get('win_rate', 0):.1%}")
        print(f"  • Net P&L: ${summary.get('total_pnl', 0):,.2f}")
        print(f"  • Profit Factor: {summary.get('profit_factor', 0):.2f}")
        
        # AI recommendations
        recommendations = insights.get("recommendations", [])
        if recommendations:
            print(f"\n🎯 Top AI Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"  {i}. {rec.get('parameter', 'unknown')}: "
                      f"{rec.get('current_value', 'N/A')} → {rec.get('suggested_value', 'N/A')}")
                print(f"     Reason: {rec.get('reasoning', 'No reason provided')[:80]}...")
        else:
            print("\n✅ No parameter adjustments recommended at this time")
        
        # Warnings
        warnings = insights.get("warnings", [])
        if warnings:
            print(f"\n⚠️  Warnings ({len(warnings)}):")
            for warning in warnings[:3]:
                print(f"  • {warning}")
            if len(warnings) > 3:
                print(f"  ... and {len(warnings) - 3} more (see detailed report)")
        
        # Generated reports
        report_files = result.get("report_files", [])
        if report_files:
            print(f"\n📁 Generated Reports:")
            for filepath in report_files:
                print(f"  • {filepath}")
        
        print("\n" + "=" * 70)
        print("Next Steps:")
        print("1. Review detailed reports in reports/daily_reviews/")
        print("2. Evaluate AI recommendations before implementation")
        print("3. Address any warnings flagged by the system")
        print("4. Update strategy parameters manually if needed")
        print("=" * 70 + "\n")
        
        logger.info("Daily review completed successfully")
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Review interrupted by user")
        logger.warning("Daily review interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Daily review failed with error: {e}", exc_info=True)
        print(f"\n❌ Daily review failed: {e}")
        print("Check logs for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
