"""Centralized logging utilities."""
from __future__ import annotations

from loguru import logger


def configure_logging(log_file: str | None = None, level: str = "INFO") -> None:
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=level)
    if log_file:
        logger.add(log_file, level=level, rotation="10 MB", retention="10 days")


__all__ = ["configure_logging", "logger"]
