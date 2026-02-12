"""Entry modules package — split from the monolithic entry_modules.py.

Provides backward-compatible re-exports so existing imports like::

    from shree.strategies.entry_modules import IntegratedEntryManager

continue to work unchanged via the re-export shim in ``entry_modules.py``.

Submodules
----------
session_time        SessionWindow, SessionTimeManager
signals             PullbackAnalysis, EntrySignal (dataclasses)
buy_continuation    BuyContinuationModule
short_continuation  ShortContinuationModule
sell_exhaustion     SellExhaustionModule
evening_buy         EveningPatternAnalysis, EveningContinuationModule
evening_sell        EveningSellPatternAnalysis, EveningSellContinuationModule
integrated_manager  IntegratedEntryManager, create_entry_manager
"""

from .session_time import SessionWindow, SessionTimeManager
from .signals import PullbackAnalysis, EntrySignal
from .buy_continuation import BuyContinuationModule
from .short_continuation import ShortContinuationModule
from .sell_exhaustion import SellExhaustionModule
from .evening_buy import EveningPatternAnalysis, EveningContinuationModule
from .evening_sell import EveningSellPatternAnalysis, EveningSellContinuationModule
from .integrated_manager import IntegratedEntryManager, create_entry_manager

__all__ = [
    "SessionWindow",
    "SessionTimeManager",
    "PullbackAnalysis",
    "EntrySignal",
    "BuyContinuationModule",
    "ShortContinuationModule",
    "SellExhaustionModule",
    "EveningPatternAnalysis",
    "EveningContinuationModule",
    "EveningSellPatternAnalysis",
    "EveningSellContinuationModule",
    "IntegratedEntryManager",
    "create_entry_manager",
]
