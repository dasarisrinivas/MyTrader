"""Logging setup for SPY options bot — mirrors the MES bot's loguru pattern."""
from __future__ import annotations

import sys
from zoneinfo import ZoneInfo

from loguru import logger

ET = ZoneInfo("America/New_York")
_CONFIGURED = False


def configure_logging(log_file: str | None = None, level: str = "INFO") -> None:
    """Configure loguru with ET timestamps and rotating file output."""
    global _CONFIGURED

    if not _CONFIGURED:
        logger.remove()

    def _et_format(record):
        et_time = record["time"].astimezone(ET)
        record["extra"]["et_time"] = et_time.strftime("%Y-%m-%d %H:%M:%S")
        return record

    fmt = (
        "<green>{extra[et_time]}</green> ET | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>\n"
    )

    if not _CONFIGURED:
        logger.add(
            sys.stderr,
            format=fmt,
            level=level,
            filter=lambda r: _et_format(r) or True,
        )

    if log_file:
        logger.add(
            log_file,
            format=fmt,
            level=level,
            rotation="10 MB",
            retention="10 days",
            filter=lambda r: _et_format(r) or True,
        )

    _CONFIGURED = True


__all__ = ["configure_logging", "logger", "ET"]
