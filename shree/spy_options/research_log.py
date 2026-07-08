"""Trade research log — one complete row per CLOSED trade for offline analysis.

Captures the full feature vector available at entry (every SpySignal field plus
a snapshot of the ExternalContext — flow, breadth, tape, depth, cross-asset,
IV, gamma env, vol structure, VWAP bands, RSI, EDR, ORB, pivots, …) joined to
the REALIZED option P&L at exit (entry/exit premium, $ P&L, R multiple, hold,
exit reason).

Motivation (Jul 8 2026): the 60-day historical backtest can only see
price-derived features — flow/breadth/options-flow/IV/tape/dark-pool are
live-only and have no history, so a production feature-importance analysis is
impossible from backtest data alone. This log accumulates the real dataset —
real option P&L with the full live feature set — so that after ~100-200 trades
the same permutation-importance / SHAP analysis can be run on the true
production distribution.

Append-only JSONL. Writes are wrapped so a logging failure NEVER affects
trading. Load with: pandas.read_json(path, lines=True).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

from ..utils.logger import logger


class TradeResearchLog:
    """Append-only JSONL writer, one row per closed trade."""

    def __init__(self, path: str = "logs/trade_research.jsonl") -> None:
        self._path = path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        except Exception:
            pass

    def log_trade(self, row: Dict[str, Any]) -> None:
        """Write one trade row. Never raises — logging must not break trading."""
        try:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("TradeResearchLog write failed: {}", exc)
