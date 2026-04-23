"""Backtest & replay harness for rules_v2.

Two entry points:

  * ``replay.replay(...)`` — feed a DataFrame of 5m bars and a DataFrame of
    candidate legacy signals through ``RulesV2Engine``; emit a trade-log
    DataFrame (allowed / blocked / reason / leg_id / augmented_strike).

  * ``apr21_synthetic`` — deterministic Apr-21 tape mimicking the real-day
    summary (range morning, shallow-trend-down afternoon with VWAP-reverting
    pullbacks). Drives the regression test.
"""
from __future__ import annotations

from .apr21_synthetic import apr21_bars, apr21_legacy_signal_candidates
from .replay import ReplayResult, replay

__all__ = [
    "ReplayResult",
    "apr21_bars",
    "apr21_legacy_signal_candidates",
    "replay",
]
