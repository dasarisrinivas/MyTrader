"""Centralized logging utilities with CST timezone support."""
from __future__ import annotations

import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from loguru import logger

_CONFIGURED = False

# Central Time Zone
CST = ZoneInfo("America/Chicago")


def configure_logging(log_file: str | None = None, level: str = "INFO", serialize: bool = False) -> None:
    """Configure logging with CST timestamps.
    
    Args:
        log_file: Optional file path for log output
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        serialize: Whether to serialize logs as JSON
    """
    global _CONFIGURED
    # Allow multiple file sinks (e.g., tee into both live_trading.log and bot.log).
    # Only clear default handlers once per process.
    if not _CONFIGURED:
        logger.remove()
    
    # Custom format function that formats time in CST
    def cst_format(record):
        """Format record with CST time."""
        cst_time = record["time"].astimezone(CST)
        record["extra"]["cst_time"] = cst_time.strftime("%Y-%m-%d %H:%M:%S")
        return record
    
    # Format string using the custom cst_time
    log_format = (
        "<green>{extra[cst_time]}</green> CST | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>\n"
    )
    
    # Console output with CST (add once)
    if not _CONFIGURED:
        logger.add(
            sys.stderr,
            format=log_format,
            level=level,
            serialize=serialize,
            filter=lambda record: cst_format(record) or True,
        )
    
    # File output with CST
    if log_file:
        logger.add(
            log_file,
            format=log_format,
            level=level,
            rotation="10 MB",
            retention="10 days",
            serialize=serialize,
            filter=lambda record: cst_format(record) or True,
        )

    _CONFIGURED = True


__all__ = ["configure_logging", "logger", "CST"]
