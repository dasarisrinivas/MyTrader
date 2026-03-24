"""Gold strategy state — persists across bot restarts.

Mirrors the approach of ``shree/utils/bot_state.py`` for MES but is
completely isolated: different file, different schema, no shared code path.

Persisted fields:
    consecutive_loss_count  int    — resets at day rollover (5 PM CT)
    cooldown_until          str    — ISO-8601 UTC; null when not active
    realized_pnl_today      float  — resets at day rollover
    trades_today            int    — resets at day rollover
    last_trade_date         str    — CT date "YYYY-MM-DD"
    written_at              str    — UTC ISO-8601 timestamp

File: configurable (default data/gold_state.json)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple

from ...utils.logger import logger
from ...utils.timezone_utils import now_cst


# How long a cooldown may be "in the past" before we silently discard it
_MAX_COOLDOWN_AGE_HOURS = 4


@dataclass
class GoldDayState:
    """Complete per-day state loaded from disk."""

    consecutive_loss_count: int = 0
    cooldown_until: Optional[datetime] = None   # timezone-aware UTC
    realized_pnl_today: float = 0.0
    trades_today: int = 0


class GoldStateManager:
    """Load and save Gold strategy runtime state atomically.

    Usage::

        sm = GoldStateManager(Path("data/gold_state.json"))
        state = sm.load()           # On startup
        state.realized_pnl_today -= 10.0
        sm.save(state)              # After each trade outcome
    """

    def __init__(self, path: Path = Path("data/gold_state.json")) -> None:
        self._path = path

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> GoldDayState:
        """Read state from disk; returns a fresh state on any error."""
        if not self._path.exists():
            logger.info("gold_state: no state file at {} — starting fresh", self._path)
            return GoldDayState()

        try:
            with open(self._path, "r") as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.warning("gold_state: could not read {}: {} — fresh start", self._path, exc)
            return GoldDayState()

        now_utc = datetime.now(timezone.utc)
        ct_today = now_cst().date().isoformat()

        # ── Day rollover ──────────────────────────────────────────────────────
        loss_count = int(data.get("consecutive_loss_count", 0))
        pnl = float(data.get("realized_pnl_today", 0.0))
        trades = int(data.get("trades_today", 0))
        last_date = data.get("last_trade_date", "")

        if last_date and last_date != ct_today:
            logger.info(
                "gold_state: day rollover %s → %s — resetting counters",
                last_date,
                ct_today,
            )
            loss_count = 0
            pnl = 0.0
            trades = 0

        # ── Cooldown ──────────────────────────────────────────────────────────
        cooldown_until: Optional[datetime] = None
        raw = data.get("cooldown_until")
        if raw:
            try:
                parsed = datetime.fromisoformat(raw)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                age_h = (now_utc - parsed).total_seconds() / 3600
                if parsed > now_utc:
                    remaining_min = (parsed - now_utc).total_seconds() / 60
                    logger.warning(
                        "gold_state: restoring active cooldown — %.0f min remaining",
                        remaining_min,
                    )
                    cooldown_until = parsed
                elif age_h <= _MAX_COOLDOWN_AGE_HOURS:
                    logger.info("gold_state: cooldown expired %.1f h ago — not restoring", age_h)
                else:
                    logger.info("gold_state: stale cooldown (%.1f h old) — discarded", age_h)
            except Exception as exc:
                logger.warning("gold_state: could not parse cooldown_until={!r}: {}", raw, exc)

        written_at = data.get("written_at", "unknown")
        logger.info(
            "gold_state: loaded — loss_count={} pnl={:.2f} trades={} cooldown={} written_at={}",
            loss_count,
            pnl,
            trades,
            "active" if cooldown_until else "none",
            written_at,
        )
        return GoldDayState(
            consecutive_loss_count=loss_count,
            cooldown_until=cooldown_until,
            realized_pnl_today=pnl,
            trades_today=trades,
        )

    def save(self, state: GoldDayState) -> None:
        """Atomically write state to disk (write-then-rename)."""
        now_utc = datetime.now(timezone.utc)
        ct_today = now_cst().date().isoformat()

        payload = {
            "consecutive_loss_count": state.consecutive_loss_count,
            "cooldown_until": (
                state.cooldown_until.isoformat() if state.cooldown_until else None
            ),
            "realized_pnl_today": round(state.realized_pnl_today, 2),
            "trades_today": state.trades_today,
            "last_trade_date": ct_today,
            "written_at": now_utc.isoformat(),
        }

        tmp = self._path.with_suffix(".json.tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self._path)
            logger.debug(
                "gold_state: saved — loss_count=%d pnl=%.2f trades=%d",
                state.consecutive_loss_count,
                state.realized_pnl_today,
                state.trades_today,
            )
        except Exception as exc:
            logger.warning("gold_state: failed to save: {}", exc)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
