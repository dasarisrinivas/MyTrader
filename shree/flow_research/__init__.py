"""SPY options real-flow RESEARCH layer.

Observation only. This package NEVER imports the executor, manager, governor,
signal engine, or trading_manager, and NEVER writes to any production table.
It reads real OPRA option prints, classifies them, computes measurements
(not signals), and stores snapshots in its own database (data/flow_research.db)
for offline validation.

Hard wall: nothing here can place, size, gate, or influence a live order.
See docs/FLOW_RESEARCH_DESIGN.md for the full design and validation plan.
"""
from __future__ import annotations

__all__ = [
    "Print",
    "Snapshot",
    "classify_aggressor",
    "mark_sweeps",
    "mark_blocks",
    "is_clean_print",
    "compute_features",
]

from .models import Print, Snapshot
from .classify import classify_aggressor, mark_sweeps, mark_blocks, is_clean_print
from .features import compute_features
