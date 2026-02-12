"""Backward-compatibility shim — DO NOT ADD NEW CODE HERE.

All entry module classes have been split into ``shree.strategies.entry.*``.
This file exists solely so that existing imports continue to work::

    from shree.strategies.entry_modules import IntegratedEntryManager  # still works

New code should import from the canonical location::

    from shree.strategies.entry import IntegratedEntryManager

.. deprecated:: 2025-07
   This shim will be removed once all consumers migrate to ``entry.*``.
"""

from shree.strategies.entry import (  # noqa: F401 — re-exports for backward compat
    SessionWindow,
    SessionTimeManager,
    PullbackAnalysis,
    EntrySignal,
    BuyContinuationModule,
    ShortContinuationModule,
    SellExhaustionModule,
    EveningPatternAnalysis,
    EveningContinuationModule,
    EveningSellPatternAnalysis,
    EveningSellContinuationModule,
    IntegratedEntryManager,
    create_entry_manager,
)

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
