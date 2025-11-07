"""Run autonomous trading analysis and learning cycle."""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mytrader.llm.autonomous_orchestrator import AutonomousTradingOrchestrator
from mytrader.utils.logger import logger


def main():
    """Main entry point for autonomous trading system."""
    parser = argparse.ArgumentParser(
        description="Semi-Autonomous LLM Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run daily analysis (view only, no changes)
  python run_autonomous_trading.py daily
  
  # Run daily analysis and apply changes (with human approval)
  python run_autonomous_trading.py daily --apply
  
  # Run daily analysis with auto-approval (high confidence only)
  python run_autonomous_trading.py daily --apply --no-approval --threshold 0.85
  
  # Run weekly review
  python run_autonomous_trading.py weekly
  
  # Check system status
  python run_autonomous_trading.py status
  
  # Check if rollback is needed
  python run_autonomous_trading.py check-rollback
  
  # Manual rollback
  python run_autonomous_trading.py rollback
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Daily analysis command
    daily_parser = subparsers.add_parser("daily", help="Run daily analysis and learning cycle")
    daily_parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply approved parameter changes"
    )
    daily_parser.add_argument(
        "--no-approval",
        action="store_true",
        help="Disable human approval requirement (auto-approve high confidence changes)"
    )
    daily_parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Auto-approval confidence threshold (default: 0.85)"
    )
    daily_parser.add_argument(
        "--date",
        type=str,
        help="Specific date to analyze (YYYY-MM-DD), defaults to today"
    )
    daily_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate changes without applying (preview mode)"
    )
    
    # Weekly review command
    weekly_parser = subparsers.add_parser("weekly", help="Run weekly performance review")
    weekly_parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply weekly adjustment suggestions"
    )
    weekly_parser.add_argument(
        "--end-date",
        type=str,
        help="End date for review period (YYYY-MM-DD), defaults to today"
    )
    
    # Status command
    subparsers.add_parser("status", help="Get system status and recent performance")
    
    # Rollback check command
    rollback_check_parser = subparsers.add_parser(
        "check-rollback",
        help="Check if performance degraded and rollback if needed"
    )
    rollback_check_parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of days since last change to evaluate (default: 1)"
    )
    
    # Manual rollback command
    subparsers.add_parser("rollback", help="Manually rollback to previous configuration")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Initialize orchestrator
        orchestrator = AutonomousTradingOrchestrator(
            require_human_approval=not getattr(args, "no_approval", False),
            auto_approve_threshold=getattr(args, "threshold", 0.85),
            enable_auto_rollback=True
        )
        
        if args.command == "daily":
            logger.info("Running daily autonomous analysis...")
            result = orchestrator.run_daily_analysis_and_learning(
                date=getattr(args, "date", None),
                apply_changes=args.apply,
                dry_run=getattr(args, "dry_run", False)
            )
            
            if result["success"]:
                logger.info("✓ Daily analysis completed successfully")
                if result.get("report_path"):
                    logger.info(f"  Report saved: {result['report_path']}")
                return 0
            else:
                logger.error(f"✗ Daily analysis failed: {result.get('error')}")
                return 1
        
        elif args.command == "weekly":
            logger.info("Running weekly performance review...")
            result = orchestrator.run_weekly_review_and_optimization(
                end_date=getattr(args, "end_date", None),
                apply_weekly_suggestions=args.apply
            )
            
            if result["success"]:
                logger.info("✓ Weekly review completed successfully")
                if result.get("report_path"):
                    logger.info(f"  Report saved: {result['report_path']}")
                return 0
            else:
                logger.error(f"✗ Weekly review failed: {result.get('error')}")
                return 1
        
        elif args.command == "status":
            logger.info("Fetching system status...")
            status = orchestrator.get_system_status()
            
            print("\n" + "=" * 60)
            print("AUTONOMOUS TRADING SYSTEM STATUS")
            print("=" * 60)
            print(f"Status: {status['status'].upper()}")
            print(f"\nRecent Performance ({status['recent_performance']['days']} days):")
            print(f"  Avg Daily P&L: ${status['recent_performance']['avg_daily_pnl']:,.2f}")
            print(f"  Avg Win Rate: {status['recent_performance']['avg_win_rate']:.1%}")
            print(f"  Total Trades: {status['recent_performance']['total_trades']}")
            print(f"\nConfiguration:")
            print(f"  Human Approval: {'Required' if status['configuration']['human_approval_required'] else 'Auto'}")
            print(f"  Auto-Rollback: {'Enabled' if status['configuration']['auto_rollback_enabled'] else 'Disabled'}")
            print(f"  Rollback Threshold: ${status['configuration']['rollback_threshold']:.2f}")
            print(f"\nRecent Parameter Changes: {status['recent_parameter_changes']}")
            print("=" * 60)
            
            return 0
        
        elif args.command == "check-rollback":
            logger.info("Checking if rollback is needed...")
            result = orchestrator.check_and_rollback_if_needed(
                days_since_change=args.days
            )
            
            if result["rollback_needed"]:
                if result.get("rollback_executed"):
                    logger.warning("⚠ Rollback executed due to poor performance")
                    return 2  # Special exit code for rollback
                else:
                    logger.error("✗ Rollback needed but failed to execute")
                    return 1
            else:
                logger.info("✓ Performance acceptable, no rollback needed")
                return 0
        
        elif args.command == "rollback":
            logger.info("Manually rolling back to previous configuration...")
            from mytrader.llm.config_manager import ConfigurationManager
            
            config_mgr = ConfigurationManager()
            success = config_mgr.rollback_last_update()
            
            if success:
                logger.info("✓ Successfully rolled back configuration")
                return 0
            else:
                logger.error("✗ Rollback failed")
                return 1
        
        else:
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
