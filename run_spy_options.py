#!/usr/bin/env python3
"""Standalone entry point for the SPY Options signal bot.

Signal-only — no orders are placed. Signals are sent via Telegram.
Requires IB Client Portal Gateway running and authenticated (default port 5000).

Usage::

    python run_spy_options.py                   # uses config.yaml
    python run_spy_options.py --config config.yaml
    python run_spy_options.py --log-level DEBUG # verbose IB request logging

Prerequisites:
    1. IB Client Portal Gateway downloaded and running:
       https://www.interactivebrokers.com/en/trading/ib-api.php
    2. Browser login completed (or SSO configured)
    3. config.yaml with spy_options.enabled = true
    4. Telegram bot token + chat_id in config.yaml telegram section

Log files:
    logs/spy_options.log — signal log (configurable via spy_options.log_file)
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from shree.config.spy_options import SpyOptionsConfig
from shree.spy_options.manager import SpyOptionsManager
from shree.utils.logger import configure_logging, logger
from shree.utils.settings_loader import load_settings


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ShreeBot — SPY Options Signal Bot")
    p.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return p.parse_args()



def main() -> None:
    args = _parse_args()

    settings = load_settings(args.config)
    cfg: SpyOptionsConfig = settings.spy_options

    if not cfg.enabled:
        print(
            f"[run_spy_options] SPY Options bot is DISABLED in {args.config}.\n"
            "Set 'spy_options:\n  enabled: true' to activate.",
            file=sys.stderr,
        )
        sys.exit(1)

    configure_logging(log_file=cfg.log_file, level=args.log_level)

    logger.info("=== ShreeBot SPY Options Signal Bot starting ===")
    logger.info("IB Gateway: {}:{}", cfg.ib.host, cfg.ib.port)

    telegram_cfg = getattr(settings, "telegram", None)
    manager = SpyOptionsManager(cfg, telegram_cfg=telegram_cfg)

    def _handle_signal(signum, frame):  # noqa: ANN001
        logger.info("Signal {} received — stopping SPY Options bot", signum)
        manager.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        manager.start()
    except Exception as exc:
        logger.opt(exception=True).error("SPY Options manager terminated: {}", exc)
        sys.exit(1)

    logger.info("=== ShreeBot SPY Options Signal Bot stopped ===")


if __name__ == "__main__":
    main()
