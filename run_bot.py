"""Entry point for RAG-enhanced Shree bot."""
import argparse
import asyncio
import os
import signal
from shree.config import Settings
from shree.execution.live_trading_manager import LiveTradingManager
from shree.utils.logger import configure_logging, logger
from shree.utils.settings_loader import load_settings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Shree RAG-Enhanced Trading Bot")
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
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Reset last-trade cooldown and release any stale order locks on startup",
    )
    return parser.parse_args()


async def main():
    # Parse command line arguments
    args = parse_args()
    
    # Configure logging with file output.
    # Keep live_trading.log (detailed runtime log) and also tee into bot.log (audit log used by tooling).
    configure_logging(log_file="logs/live_trading.log", level="INFO", serialize=False)
    configure_logging(log_file="logs/bot.log", level="INFO", serialize=False)
    
    mode_str = "SIMULATION" if args.simulation else "LIVE"
    logger.info(f"🚀 Starting Shree RAG-Enhanced Bot ({mode_str} MODE)")
    
    if args.simulation:
        logger.warning("=" * 60)
        logger.warning("🔶 SIMULATION MODE - NO REAL ORDERS WILL BE PLACED")
        logger.warning("=" * 60)
    
    # Load settings
    settings = load_settings(args.config)

    # APR 22 2026 Fix #6: Paper-vs-live startup guardrail.
    # Prevent the "I thought I was on paper" disaster: if DEPLOY_ENV=paper, the
    # resolved IB port MUST be 4002 (paper gateway). If DEPLOY_ENV=prod, MUST be 4001.
    # Env var IBKR_PORT is what the runtime actually uses; fall back to config.
    _deploy_env = (os.environ.get("DEPLOY_ENV") or "").strip().lower()
    _env_port = os.environ.get("IBKR_PORT")
    try:
        _cfg_port = getattr(settings.data, "ibkr_port", None)
    except Exception:
        _cfg_port = None
    try:
        _effective_port = int(_env_port) if _env_port else (int(_cfg_port) if _cfg_port else None)
    except (TypeError, ValueError):
        _effective_port = None

    _expected = {"paper": 4002, "prod": 4001, "live": 4001}.get(_deploy_env)
    if _deploy_env and _expected and _effective_port and _effective_port != _expected:
        msg = (
            f"❌ MODE MISMATCH: DEPLOY_ENV={_deploy_env!r} expects IB port {_expected}, "
            f"but resolved port is {_effective_port} "
            f"(env IBKR_PORT={_env_port!r}, config {_cfg_port!r}). "
            f"Refusing to start to prevent live-trading the wrong account."
        )
        logger.critical(msg)
        raise SystemExit(msg)
    if not _deploy_env:
        logger.warning(
            "⚠️  DEPLOY_ENV not set — skipping paper/live port guardrail. "
            "Set DEPLOY_ENV=paper or DEPLOY_ENV=prod before starting."
        )
    else:
        logger.info(
            f"✅ Deploy guardrail OK: DEPLOY_ENV={_deploy_env} → IB port {_effective_port}"
        )

    # Initialize manager with simulation mode
    manager = LiveTradingManager(
        settings,
        simulation_mode=args.simulation,
        reset_state_on_start=args.reset_state,
    )

    # Manual override: reset state if requested
    if args.reset_state:
        logger.warning("♻️  Manual override: resetting last trade time and releasing order lock")
        manager.reset_state()

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
    except BaseException as e:
        import sys, traceback
        logger.critical(f"Fatal error: {e}")
        logger.critical(traceback.format_exc())
        # Ensure the error makes it to disk even if loguru hasn't flushed
        print(f"FATAL: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
