"""Entry point for RAG-enhanced MyTrader bot."""
import asyncio
import os
import signal
from mytrader.config import Settings
from mytrader.execution.live_trading_manager import LiveTradingManager
from mytrader.utils.logger import configure_logging, logger
from mytrader.utils.settings_loader import load_settings

async def main():
    # Configure logging with file output
    configure_logging(log_file="logs/live_trading.log", level="INFO", serialize=False)
    
    logger.info("🚀 Starting MyTrader RAG-Enhanced Bot")
    
    # Load settings
    settings = load_settings("config.yaml")
    
    # Initialize manager
    manager = LiveTradingManager(settings)
    
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
