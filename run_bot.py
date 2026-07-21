"""Entry point for the MES SIGNAL-ONLY bot.

JUL 21 2026: IB account converted to Cash — CME futures can no longer be
traded.  The entire execution stack (orders, positions, fills, brackets,
reconciliation) was deleted.  This process analyzes the market and emits
BUY/SELL/HOLD signals with confidence, levels and reasoning.  It never places
orders.  IB is used as a market-data feed only.
"""
import argparse
import asyncio
import os
import signal

from shree.signal_bot import MesSignalBot
from shree.utils.logger import configure_logging, logger
from shree.utils.settings_loader import load_settings


def parse_args():
    parser = argparse.ArgumentParser(description="Shree MES Signal-Only Bot")
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    # Accepted for launcher compatibility (start_bot.sh) — the bot is
    # signal-only, so every run is already order-free.
    parser.add_argument("--simulation", "-s", action="store_true",
                        help=argparse.SUPPRESS)
    return parser.parse_args()


async def main():
    args = parse_args()
    configure_logging(log_file="logs/live_trading.log", level="INFO", serialize=False)
    configure_logging(log_file="logs/bot.log", level="INFO", serialize=False)

    try:
        os.makedirs("logs", exist_ok=True)
        with open("logs/bot.pid", "w") as _pf:
            _pf.write(str(os.getpid()))
    except OSError as _exc:
        logger.warning(f"Could not write logs/bot.pid: {_exc}")

    logger.info("🚀 Starting MES SIGNAL-ONLY bot — no orders will ever be placed")
    settings = load_settings(args.config)
    bot = MesSignalBot(settings)

    loop = asyncio.get_running_loop()

    def handle_signal(sig, frame):
        logger.info("🛑 Shutdown signal received")
        loop.create_task(bot.stop())

    try:
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    except ValueError:
        pass  # not in main thread

    await bot.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except BaseException as e:
        import sys
        import traceback
        logger.critical(f"Fatal error: {e}")
        logger.critical(traceback.format_exc())
        print(f"FATAL: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
