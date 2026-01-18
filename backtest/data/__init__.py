"""
Backtest Data Module
====================

Handles historical data acquisition, normalization, and continuous futures construction.
"""

from .ib_downloader import IBHistoricalDownloader
from .normalize import DataNormalizer, NormalizationConfig
from .roll import ContinuousFuturesBuilder, RollConfig

__all__ = [
    "IBHistoricalDownloader",
    "DataNormalizer",
    "NormalizationConfig",
    "ContinuousFuturesBuilder",
    "RollConfig",
]
