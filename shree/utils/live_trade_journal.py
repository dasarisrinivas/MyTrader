"""
live_trade_journal.py — Append-only CSV journal for live MES trades.

APR 10 2026: Created to fix the critical logging blind spot identified
in the portfolio audit. Without this, live trade metrics (Sharpe, DD,
regime breakdown, win rate by signal) cannot be computed.

File: data/live_trades.csv
Schema mirrors backtest CSV + live-only fields (dd_tier, confidence, regime).
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from shree.utils.logger import logger
from shree.utils.timezone_utils import now_cst

_DEFAULT_PATH = Path("data/live_trades.csv")

_COLUMNS = [
    "symbol",
    "direction",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "quantity",
    "realized_pnl",
    "pnl_points",
    "commission",
    "exit_reason",
    "signal_type",
    "signal_confidence",
    "atr",
    "adx",
    "regime",
    "session_type",
    "stop_loss",
    "take_profit",
    "dd_tier",
    "entry_metadata",
]


def log_live_trade(
    trade_context: Dict[str, Any],
    exit_price: float,
    realized_pnl: float,
    exit_reason: str,
    exit_time: Optional[datetime] = None,
    commission: float = 4.80,
    dd_tier: int = 0,
    path: Path = _DEFAULT_PATH,
) -> None:
    """Append one trade row to the live trades CSV.

    Args:
        trade_context: The _open_trade_context dict from order_coordinator.
        exit_price: Fill price of the closing order.
        realized_pnl: Net realized P&L in dollars.
        exit_reason: "SL", "TP", "FLATTEN", "MANUAL", etc.
        exit_time: When the exit occurred (defaults to now CST).
        commission: Round-trip commission in dollars (default $4.80).
        dd_tier: Current drawdown tier at entry (0=normal, 1/2/3).
        path: Override path (for tests).
    """
    if not trade_context:
        logger.debug("live_trade_journal: no trade_context — skipping")
        return

    if exit_time is None:
        exit_time = now_cst()

    metadata = trade_context.get("metadata", {})
    entry_price = trade_context.get("entry_price", 0.0)
    qty = trade_context.get("quantity", 1) or 1
    point_value = 5.0  # MES = $5/point

    if entry_price and exit_price:
        if trade_context.get("is_long"):
            pnl_points = round(exit_price - entry_price, 2)
        else:
            pnl_points = round(entry_price - exit_price, 2)
    else:
        pnl_points = round(realized_pnl / (qty * point_value), 2) if point_value else 0.0

    row = {
        "symbol": "MES",
        "direction": "LONG" if trade_context.get("is_long") else "SHORT",
        "entry_time": trade_context.get("entry_time", ""),
        "entry_price": entry_price,
        "exit_time": exit_time.isoformat() if hasattr(exit_time, "isoformat") else str(exit_time),
        "exit_price": exit_price,
        "quantity": qty,
        "realized_pnl": round(realized_pnl, 2),
        "pnl_points": pnl_points,
        "commission": commission,
        "exit_reason": exit_reason,
        "signal_type": trade_context.get("signal_type", metadata.get("entry_type", "")),
        "signal_confidence": trade_context.get("signal_confidence", metadata.get("signal_confidence", "")),
        "atr": metadata.get("atr", metadata.get("atr_value", "")),
        "adx": metadata.get("adx_value", ""),
        "regime": trade_context.get("regime", metadata.get("market_state", "")),
        "session_type": metadata.get("session_type", ""),
        "stop_loss": trade_context.get("stop_loss", ""),
        "take_profit": trade_context.get("take_profit", ""),
        "dd_tier": dd_tier,
        "entry_metadata": str(metadata),
    }

    write_header = not path.exists() or path.stat().st_size == 0

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        logger.info(
            f"live_trade_journal: logged {row['direction']} "
            f"entry={row['entry_price']} exit={row['exit_price']} "
            f"pnl=${row['realized_pnl']} reason={row['exit_reason']}"
        )
    except Exception as exc:
        logger.warning(f"live_trade_journal: failed to write: {exc}")
