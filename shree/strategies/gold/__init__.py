"""Gold intraday strategy package.

Public surface::

    from shree.strategies.gold import GoldIntradayStrategy
    from shree.strategies.gold.regime import GoldRegime, GoldRegimeDetector
    from shree.strategies.gold.signals import GoldSignalGenerator, GoldSignalType
"""
from .strategy import GoldIntradayStrategy
from .regime import GoldRegime, GoldRegimeDetector
from .signals import GoldSignalGenerator, GoldSignalType

__all__ = [
    "GoldIntradayStrategy",
    "GoldRegime",
    "GoldRegimeDetector",
    "GoldSignalGenerator",
    "GoldSignalType",
]
