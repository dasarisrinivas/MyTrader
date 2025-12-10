"""Entry point for RAG-enhanced MyTrader bot."""
import argparse
import asyncio
import os
import signal
from mytrader.config import Settings
from mytrader.execution.live_trading_manager import LiveTradingManager
from mytrader.utils.logger import configure_logging, logger
from mytrader.utils.settings_loader import load_settings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MyTrader RAG-Enhanced Trading Bot")
    parser.add_argument(
        "--simulation", "-s",
        action="store_true",
        help="Run in simulation mode (no real orders placed)"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=None,
        help="Override cooldown period in minutes (default: from config)"
    )
    return parser.parse_args()


async def main():
    # Parse command line arguments
    args = parse_args()
    
    # Configure logging with file output
    configure_logging(log_file="logs/live_trading.log", level="INFO", serialize=False)
    
    mode_str = "SIMULATION" if args.simulation else "LIVE"
    logger.info(f"🚀 Starting MyTrader RAG-Enhanced Bot ({mode_str} MODE)")
    
    if args.simulation:
        logger.warning("=" * 60)
        logger.warning("🔶 SIMULATION MODE - NO REAL ORDERS WILL BE PLACED")
        logger.warning("=" * 60)
    
    # Load settings
    settings = load_settings(args.config)
    
    # Initialize manager with simulation mode
    manager = LiveTradingManager(settings, simulation_mode=args.simulation)
    
    # Override cooldown if specified
    if args.cooldown is not None:
        manager._cooldown_seconds = args.cooldown * 60
        logger.info(f"⏱️ Cooldown overridden to {args.cooldown} minutes")
    
    # Handle shutdown signals
    def handle_signal(sig, frame):
        logger.info("🛑 Shutdown signal received")
        asyncio.create_task(manager.stop())
        
    # Register signal handlers (only works in main thread)
    try:
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    except ValueError:
        pass # Ignore if not in main thread
    
    # Start trading
    await manager.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
