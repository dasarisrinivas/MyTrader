"""
Backtest Framework for Shree
===============================

This module provides a comprehensive 2-year historical backtesting capability
that reuses the exact same strategy/risk/execution decision logic used in live trading.

Key Components:
- run.py: CLI entrypoint for reproducible backtest runs
- engine.py: Event-driven bar iterator calling existing bot logic
- broker_sim.py: Simulated broker for fills, brackets, slippage, commissions
- analysis.py: Performance metrics, diagnostics, and learnings
- report.py: HTML/MD report generation

Data Components:
- data/ib_downloader.py: IB historical downloader with pacing + chunking
- data/normalize.py: Data normalization and schema unification
- data/roll.py: Continuous futures builder + roll calendar

Usage:
    python -m backtest.run --symbol MES --start 2024-01-09 --end 2026-01-09 --bar 1m --bar2 5m
"""

from .engine import BacktestEngine
from .broker_sim import BrokerSimulator
from .analysis import BacktestAnalyzer
from .report import ReportGenerator

__all__ = [
    "BacktestEngine",
    "BrokerSimulator",
    "BacktestAnalyzer",
    "ReportGenerator",
]

__version__ = "1.0.0"
