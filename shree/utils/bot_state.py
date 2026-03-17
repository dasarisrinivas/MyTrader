"""
bot_state.py — Persist critical in-memory trading state across bot restarts.

MAR 13 2026: Created to fix Fix #14 (consecutive-loss cooldown) silently
resetting on restart. The bot restarts multiple times per day (IB reconnects,
manual restarts) which reset _consecutive_loss_count to 0, preventing the
3-loss cooldown from ever accumulating.

MAR 16 2026 (Fix #6): Added realized_pnl_today to survive restarts.
Previously, restarting the bot mid-day reset the daily P&L counter to 0,
making the $250 daily loss cap ineffective after a restart.

State persisted:
  - _consecutive_loss_count  (int)  — resets to 0 at day rollover (5 PM CT)
  - _extra_cooldown_until    (str, ISO format UTC)  — cleared if > 4h in past
  - realized_pnl_today       (float) — resets to 0.0 at day rollover

File: data/bot_state.json
Format:
  {
    "consecutive_loss_count": 2,
    "extra_cooldown_until": "2026-03-13T15:30:00+00:00",  # or null
    "realized_pnl_today": -55.0,
    "last_trade_date": "2026-03-13",   # CT date — used for day rollover
    "written_at": "2026-03-13T14:22:11+00:00"
  }
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple

from shree.utils.logger import logger
from shree.utils.timezone_utils import now_cst

# ── Path ────────────────────────────────────────────────────────────────────
_DEFAULT_STATE_PATH = Path("data/bot_state.json")

# Cooldowns older than this are considered stale and discarded on load
_MAX_COOLDOWN_AGE_HOURS = 4

# ── Public API ───────────────────────────────────────────────────────────────

def load_bot_state(
    path: Path = _DEFAULT_STATE_PATH,
) -> Tuple[int, Optional[datetime], float]:
    """Load persisted bot state from disk.

    Returns:
        (consecutive_loss_count, extra_cooldown_until, realized_pnl_today)

    Handles:
      - Missing file → returns (0, None, 0.0)
      - Corrupt JSON → returns (0, None, 0.0) with warning
      - Day rollover → resets consecutive_loss_count to 0, realized_pnl_today to 0.0
      - Stale cooldown (> 4h old) → discards it
    """
    if not path.exists():
        logger.info(f"bot_state: no state file at {path} — starting fresh")
        return 0, None, 0.0

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning(f"bot_state: failed to read {path}: {exc} — starting fresh")
        return 0, None, 0.0

    now_utc = datetime.now(timezone.utc)
    ct_today = now_cst().date().isoformat()

    # ── Day rollover: reset loss counter and daily P&L if last trade was on a different CT date ──
    loss_count = int(data.get("consecutive_loss_count", 0))
    realized_pnl_today = float(data.get("realized_pnl_today", 0.0))
    last_trade_date = data.get("last_trade_date", "")
    if last_trade_date and last_trade_date != ct_today:
        logger.info(
            f"bot_state: day rollover detected "
            f"(last_trade_date={last_trade_date}, today={ct_today}) "
            f"→ resetting consecutive_loss_count {loss_count} → 0, "
            f"realized_pnl_today {realized_pnl_today:.2f} → 0.00"
        )
        loss_count = 0
        realized_pnl_today = 0.0

    # ── Cooldown: parse and validate ──────────────────────────────────────────
    cooldown_until: Optional[datetime] = None
    raw_cooldown = data.get("extra_cooldown_until")
    if raw_cooldown:
        try:
            parsed = datetime.fromisoformat(raw_cooldown)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            age_hours = (now_utc - parsed).total_seconds() / 3600
            if parsed > now_utc:
                # Still in the future — restore it
                remaining_min = (parsed - now_utc).total_seconds() / 60
                logger.warning(
                    f"bot_state: restoring active cooldown — "
                    f"{remaining_min:.0f}m remaining (until {parsed.strftime('%H:%M:%S')} UTC)"
                )
                cooldown_until = parsed
            elif age_hours <= _MAX_COOLDOWN_AGE_HOURS:
                # Expired recently — log it but don't restore
                logger.info(
                    f"bot_state: cooldown expired {age_hours:.1f}h ago — not restoring"
                )
            else:
                logger.info(f"bot_state: stale cooldown ({age_hours:.1f}h old) — discarded")
        except Exception as exc:
            logger.warning(f"bot_state: could not parse extra_cooldown_until={raw_cooldown!r}: {exc}")

    written_at = data.get("written_at", "unknown")
    logger.info(
        f"bot_state: loaded — consecutive_loss_count={loss_count}, "
        f"realized_pnl_today={realized_pnl_today:.2f}, "
        f"cooldown={'active' if cooldown_until else 'none'}, "
        f"written_at={written_at}"
    )
    return loss_count, cooldown_until, realized_pnl_today


def save_bot_state(
    consecutive_loss_count: int,
    extra_cooldown_until: Optional[datetime],
    realized_pnl_today: float = 0.0,
    path: Path = _DEFAULT_STATE_PATH,
) -> None:
    """Persist bot state to disk atomically (write-then-rename).

    Args:
        consecutive_loss_count: Current consecutive loss streak (0 = none)
        extra_cooldown_until:   UTC datetime when cooldown expires, or None
        realized_pnl_today:     Cumulative realized P&L for today ($)
        path:                   Override default path (mainly for tests)
    """
    now_utc = datetime.now(timezone.utc)
    ct_today = now_cst().date().isoformat()

    data = {
        "consecutive_loss_count": consecutive_loss_count,
        "extra_cooldown_until": (
            extra_cooldown_until.isoformat() if extra_cooldown_until else None
        ),
        "realized_pnl_today": round(realized_pnl_today, 2),
        "last_trade_date": ct_today,
        "written_at": now_utc.isoformat(),
    }

    # Atomic write: temp file then rename to avoid corrupt reads mid-write
    tmp_path = path.with_suffix(".json.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
        logger.info(
            f"bot_state: saved — loss_count={consecutive_loss_count}, "
            f"realized_pnl_today={realized_pnl_today:.2f}, "
            f"cooldown={extra_cooldown_until}"
        )
    except Exception as exc:
        logger.warning(f"bot_state: failed to save state to {path}: {exc}")
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
