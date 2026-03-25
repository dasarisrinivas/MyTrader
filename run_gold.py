#!/usr/bin/env python3
"""Standalone entry point for the Gold futures intraday strategy.

Usage::

    python run_gold.py                          # uses config.yaml gold section
    python run_gold.py --config config.gold.yaml
    python run_gold.py --simulation             # dry-run: no orders sent
    python run_gold.py --reset-state            # clear daily P&L / cooldowns

The Gold strategy runs with its own IB client ID and persists state
separately from the MES bot.  Both bots can run simultaneously.

Log files:
    logs/gold_trading.log   — detailed trading log
    logs/gold_bot.log       — audit / summary log (human-readable)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

# Ensure the package root is importable when run directly
sys.path.insert(0, str(Path(__file__).parent))

from shree.config.gold import GoldStrategyConfig
from shree.execution.gold.manager import GoldTradingManager
from shree.execution.gold.state import GoldStateManager
from shree.utils.logger import configure_logging, logger
from shree.utils.settings_loader import load_settings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShreeBot — Gold Futures Intraday Strategy")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Dry-run mode: compute signals but do not send orders to IB",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Clear persisted daily state before starting",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()

    # ── Load settings ─────────────────────────────────────────────────────────
    settings = load_settings(args.config)
    gold_cfg: GoldStrategyConfig = settings.gold

    if not gold_cfg.enabled:
        print(
            "[run_gold] Gold strategy is DISABLED in config "
            f"({args.config}: gold.enabled = false).\n"
            "Set 'gold:\\n  enabled: true' to activate.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Override from CLI ─────────────────────────────────────────────────────
    if args.simulation:
        gold_cfg.simulation = True

    # ── Configure logging ─────────────────────────────────────────────────────
    configure_logging(log_file=gold_cfg.log_file, level=args.log_level)
    # Audit log: derive from trading log path (paper_gold_trading → paper_gold_bot)
    _audit_log = gold_cfg.log_file.replace("_trading.log", "_bot.log")
    if _audit_log == gold_cfg.log_file:
        _audit_log = "logs/gold_bot.log"  # fallback
    configure_logging(log_file=_audit_log, level=args.log_level)

    logger.info("=== ShreeBot Gold Strategy starting ===")
    logger.info(
        "Config: symbol={} exchange={} port={} clientId={} simulation={}",
        gold_cfg.symbol,
        gold_cfg.exchange,
        gold_cfg.ibkr_port,
        gold_cfg.ibkr_client_id,
        gold_cfg.simulation,
    )

    # ── Optional state reset ──────────────────────────────────────────────────
    if args.reset_state:
        from pathlib import Path as _Path
        sm = GoldStateManager(_Path(gold_cfg.state_file))
        from shree.execution.gold.state import GoldDayState
        sm.save(GoldDayState())
        logger.info("Gold state reset — counters cleared")

    # ── Create and run the manager ────────────────────────────────────────────
    telegram_cfg = getattr(settings, "telegram", None)
    manager = GoldTradingManager(gold_cfg, telegram_cfg=telegram_cfg)

    loop = asyncio.get_event_loop()

    def _handle_signal(signum, frame):  # noqa: ANN001
        logger.info("Signal {} received — stopping Gold manager", signum)
        manager.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        await manager.start()
    except Exception as exc:
        logger.opt(exception=True).error("Gold manager terminated with error: {}", exc)
        sys.exit(1)

    logger.info("=== ShreeBot Gold Strategy stopped ===")


if __name__ == "__main__":
    asyncio.run(main())
