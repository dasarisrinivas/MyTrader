"""PDT (Pattern Day Trader) compliance tracker.

Tracks round-trip options trades in a rolling 5-trading-day window.
Persists state to JSON so it survives bot restarts.

A "round trip" = one open + one close of the same options position.
FINRA rule: max 3 round trips per 5 trading days for accounts < $25k.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

from logger import logger

# Approximate set of US market holidays (extend as needed)
_HOLIDAYS_2025_2026: set[date] = {
    date(2025, 1, 1),   date(2025, 1, 20),  date(2025, 2, 17),
    date(2025, 4, 18),  date(2025, 5, 26),  date(2025, 6, 19),
    date(2025, 7, 4),   date(2025, 9, 1),   date(2025, 11, 27),
    date(2025, 12, 25),
    date(2026, 1, 1),   date(2026, 1, 19),  date(2026, 2, 16),
    date(2026, 4, 3),   date(2026, 5, 25),  date(2026, 6, 19),
    date(2026, 7, 3),   date(2026, 9, 7),   date(2026, 11, 26),
    date(2026, 12, 25),
}


def _is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in _HOLIDAYS_2025_2026


def _last_n_trading_days(n: int, ref: date | None = None) -> list[date]:
    """Return list of the last n trading days up to and including ref."""
    ref = ref or date.today()
    days: list[date] = []
    current = ref
    while len(days) < n:
        if _is_trading_day(current):
            days.append(current)
        current -= timedelta(days=1)
    return days


class PDTTracker:
    """Persistent round-trip trade counter for PDT compliance."""

    def __init__(self, filepath: str = "spy_options_bot/pdt_log.json", max_trades: int = 3) -> None:
        self.filepath = Path(filepath)
        self.max_trades = max_trades
        self._trades: list[dict] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self.filepath.exists():
            try:
                data = json.loads(self.filepath.read_text())
                self._trades = data.get("trades", [])
                logger.info(f"PDT log loaded: {len(self._trades)} historical records from {self.filepath}")
            except Exception as exc:
                logger.warning(f"PDT log unreadable ({exc}), starting fresh")
                self._trades = []
        else:
            self._trades = []
            logger.info(f"PDT log not found at {self.filepath}, starting fresh")

    def _save(self) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.filepath.write_text(json.dumps({"trades": self._trades}, indent=2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_round_trip(
        self,
        strategy_type: str = "single",
        symbol: str = "SPY",
        description: str = "",
        trade_date: date | None = None,
    ) -> None:
        """Record a completed round-trip trade.

        Args:
            strategy_type: "single" (put or call) = 1 round-trip;
                           "strangle" = 2 round-trips (both legs in one call).
                           When closing strangle legs individually, pass "single"
                           per leg — each leg is a separate FINRA round-trip.
            symbol:        Underlying symbol.
            description:   Human-readable description for the log.
            trade_date:    Date of the trade (defaults to today).
        """
        slots = 2 if strategy_type == "strangle" else 1
        d = trade_date or date.today()
        for i in range(slots):
            leg_desc = description + (f" (leg {i + 1}/{slots})" if slots > 1 else "")
            entry = {
                "date": d.isoformat(),
                "symbol": symbol,
                "strategy_type": strategy_type,
                "description": leg_desc,
                "recorded_at": datetime.utcnow().isoformat() + "Z",
            }
            self._trades.append(entry)
        self._save()
        count = self.get_weekly_count()
        logger.info(
            f"PDT round-trip recorded ({slots} slot(s)): {description} | "
            f"weekly count: {count}/{self.max_trades}"
        )
        if count == self.max_trades - 1:
            logger.warning(f"PDT WARNING: {count}/{self.max_trades} trades used — 1 remaining")
        elif count >= self.max_trades:
            logger.warning(f"PDT LIMIT REACHED: {count}/{self.max_trades} — no new entries allowed")

    def get_weekly_count(self, ref: date | None = None) -> int:
        """Return number of round-trips in the last 5 trading days."""
        window = {d.isoformat() for d in _last_n_trading_days(5, ref)}
        return sum(1 for t in self._trades if t["date"] in window)

    def can_trade(self, ref: date | None = None) -> bool:
        """Return True if a new trade is allowed under PDT rules."""
        count = self.get_weekly_count(ref)
        allowed = count < self.max_trades
        if not allowed:
            logger.warning(
                f"PDT BLOCK: {count}/{self.max_trades} round-trips used in rolling 5-day window"
            )
        return allowed

    def slots_remaining(self, ref: date | None = None) -> int:
        """Return how many round-trip slots are still available this week."""
        return max(0, self.max_trades - self.get_weekly_count(ref))

    def status(self) -> dict:
        count = self.get_weekly_count()
        return {
            "weekly_count": count,
            "max_trades": self.max_trades,
            "remaining": max(0, self.max_trades - count),
            "can_trade": count < self.max_trades,
        }
