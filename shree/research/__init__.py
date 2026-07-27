"""Canonical research evaluation infrastructure.

The replay engine here is THE evaluation path for strategy research. The legacy
SPY-barrier grade (spy_signals.pnl_pct) systematically overestimated tradeability
by masking the asymmetric losses of options (a 0DTE OTM call going wrong loses
~90%, not the ~20% the barrier implied). No strategy should be judged on the
barrier again — grade on real historical option P&L via `replay_engine`.

Research versions (tag every result):
  v1  SPY barrier (legacy, deprecated for tradeability claims)
  v2  Real option replay (bid/ask fills + commission + production exit)   <-- current
  v3  + execution latency / partial fills (future)
  v4  + multi-contract comparison as default (future)
"""
from .replay_engine import (
    RESEARCH_ENGINE_VERSION,
    ExecutionModel,
    ContractChoice,
    ReplayResult,
    replay_signal,
    scorecard,
)

__all__ = [
    "RESEARCH_ENGINE_VERSION", "ExecutionModel", "ContractChoice",
    "ReplayResult", "replay_signal", "scorecard",
]
