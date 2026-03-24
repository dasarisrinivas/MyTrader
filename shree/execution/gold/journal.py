"""Gold trade journal — append-only JSONL trade log + session metrics.

Each trade is written as a single JSON line to a daily file under
``journal_dir/{YYYY-MM-DD}.jsonl``.  At session end the manager calls
``print_session_summary()`` which logs human-readable metrics.

No database is used here — the journal is intentionally simple so it
survives crashes and is easy to audit.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from ...utils.logger import logger
from ...utils.timezone_utils import now_cst


@dataclass
class GoldTradeRecord:
    """One completed round-trip trade."""

    trade_id: str
    symbol: str
    action: str                          # "BUY" or "SELL"
    signal_type: str
    contracts: int

    entry_price: float
    stop_loss: float
    take_profit: float
    exit_price: float

    realized_pnl: float
    commission: float
    net_pnl: float

    entry_time: str                      # ISO-8601 UTC
    exit_time: str                       # ISO-8601 UTC
    hold_bars: int

    exit_reason: str                     # e.g. "PROFIT_TARGET", "STOP_LOSS", "TIME_STOP", "FLATTEN"
    regime: str
    atr_at_entry: float
    adx_at_entry: float

    win: bool = field(init=False)

    def __post_init__(self) -> None:
        self.win = self.net_pnl > 0


class GoldJournal:
    """Append-only trade journal with session-level metrics.

    Usage::

        journal = GoldJournal(Path("data/gold_journal"))
        journal.record(trade)
        journal.print_session_summary()
    """

    def __init__(self, journal_dir: Path = Path("data/gold_journal")) -> None:
        self._dir = journal_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._today_records: List[GoldTradeRecord] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, trade: GoldTradeRecord) -> None:
        """Append a trade to the daily JSONL file and in-memory list."""
        self._today_records.append(trade)

        today_str = now_cst().date().isoformat()
        fpath = self._dir / f"{today_str}.jsonl"

        try:
            with open(fpath, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(trade)) + "\n")
        except Exception as exc:
            logger.warning("GoldJournal: failed to write trade record: {}", exc)

        logger.info(
            "GoldJournal: {} {} {}×{} — exit={} pnl={:.2f} ({}) reason={}",
            trade.action,
            trade.symbol,
            trade.contracts,
            trade.signal_type,
            trade.exit_price,
            trade.net_pnl,
            "WIN" if trade.win else "LOSS",
            trade.exit_reason,
        )

    def print_session_summary(self) -> None:
        """Log session-level performance metrics to the standard logger."""
        records = self._today_records
        if not records:
            logger.info("GoldJournal: no trades recorded today")
            return

        wins = [r for r in records if r.win]
        losses = [r for r in records if not r.win]
        total_pnl = sum(r.net_pnl for r in records)
        gross_win = sum(r.net_pnl for r in wins)
        gross_loss = abs(sum(r.net_pnl for r in losses))
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
        win_rate = len(wins) / len(records) * 100

        avg_win = (gross_win / len(wins)) if wins else 0.0
        avg_loss = (gross_loss / len(losses)) if losses else 0.0

        exit_reasons: dict = {}
        for r in records:
            exit_reasons[r.exit_reason] = exit_reasons.get(r.exit_reason, 0) + 1

        logger.info(
            "=== GOLD SESSION SUMMARY ===\n"
            "  Trades:        %d  (W=%d L=%d  WR=%.1f%%)\n"
            "  Net P&L:       $%.2f\n"
            "  Profit Factor: %.2f\n"
            "  Avg Win:       $%.2f   Avg Loss: $%.2f\n"
            "  Exit reasons:  %s",
            len(records),
            len(wins),
            len(losses),
            win_rate,
            total_pnl,
            profit_factor,
            avg_win,
            avg_loss,
            json.dumps(exit_reasons),
        )

    def today_records(self) -> List[GoldTradeRecord]:
        """Return in-memory records for the current session."""
        return list(self._today_records)

    def reset_day(self) -> None:
        """Clear in-memory records (call at session rollover)."""
        self._today_records.clear()
