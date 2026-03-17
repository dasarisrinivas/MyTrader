"""Guard tracker — persists VIX spike and large move cooldown dates.

When a VIX spike or large SPY move is detected, entries are paused for
N calendar days. This state survives bot restarts via a JSON file.

Usage:
    guards = GuardTracker()
    if guards.is_blocked():
        reason = guards.block_reason()
        # skip entry

    # After detecting a VIX spike:
    guards.set_vix_skip(days=config.VIX_SPIKE_SKIP_DAYS)

    # After detecting a large SPY move:
    guards.set_move_skip(days=config.LARGE_MOVE_SKIP_DAYS)
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from logger import logger

GUARD_FILE = Path("spy_options_bot/guard_state.json")


class GuardTracker:
    def __init__(self, filepath: Path = GUARD_FILE) -> None:
        self._path = Path(filepath)
        self._state = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_blocked(self) -> bool:
        """True if either guard is still active today."""
        today = date.today()
        return self._vix_skip_until() > today or self._move_skip_until() > today

    def block_reason(self) -> str:
        today = date.today()
        parts = []
        if self._vix_skip_until() > today:
            parts.append(f"VIX spike cooldown until {self._vix_skip_until()}")
        if self._move_skip_until() > today:
            parts.append(f"large move cooldown until {self._move_skip_until()}")
        return " | ".join(parts)

    def set_vix_skip(self, days: int) -> None:
        until = date.today() + timedelta(days=days)
        self._state["vix_skip_until"] = until.isoformat()
        self._save()
        logger.warning(f"VIX spike guard: pausing entries until {until}")

    def set_move_skip(self, days: int) -> None:
        until = date.today() + timedelta(days=days)
        self._state["move_skip_until"] = until.isoformat()
        self._save()
        logger.warning(f"Large move guard: pausing entries until {until}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _vix_skip_until(self) -> date:
        raw = self._state.get("vix_skip_until", "2000-01-01")
        return date.fromisoformat(raw)

    def _move_skip_until(self) -> date:
        raw = self._state.get("move_skip_until", "2000-01-01")
        return date.fromisoformat(raw)

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                pass
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._state, indent=2))
