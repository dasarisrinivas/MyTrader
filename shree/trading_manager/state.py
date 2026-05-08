"""Persistent state for the Trading Manager.

Holds: today's realized PnL, today's trade count, consecutive W/L streak,
current posture (NORMAL / DEFENSIVE / SIT_OUT / KILLED), last decision id.

State is rebuilt on startup from disk + the orders.db, so the manager is
restart-safe.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import List, Optional


POSTURE_NORMAL    = "NORMAL"
POSTURE_DEFENSIVE = "DEFENSIVE"   # cut size, raise gates (soft pause / 1 trigger)
POSTURE_SIT_OUT   = "SIT_OUT"     # reject all new entries (hard pause / 2 triggers)
POSTURE_KILLED    = "KILLED"      # bots forcibly paused (daily loss breach)
POSTURE_LOCKED    = "LOCKED"      # multi-day strategy failure (3+ health triggers);
                                  # manual unlock required
POSTURE_PROBATION = "PROBATION"   # post-unlock: 1 small probe trade allowed; must
                                  # win to clear, else back to LOCKED


@dataclass
class ManagerState:
    session_date: str = ""                # YYYY-MM-DD (CT)
    realized_pnl_today: float = 0.0
    trades_today: int = 0
    consec_wins: int = 0
    consec_losses: int = 0
    posture: str = POSTURE_NORMAL
    last_signal_ts: str = ""
    last_decision_id: int = 0
    last_n_outcomes: List[str] = field(default_factory=list)  # WIN/LOSS strings
    notes: str = ""

    # Layer 1.5 health monitoring (multi-day forest view)
    health_status: str = "HEALTHY"            # HEALTHY/DEGRADED/SUSPECT/LOCKED/PROBATION
    health_triggers: List[str] = field(default_factory=list)
    health_summary: str = ""                  # human-readable last metrics snapshot
    health_last_check: str = ""               # ISO ts of last evaluation
    lock_reason: str = ""                     # why we're LOCKED
    lock_since: str = ""                      # when LOCKED first set
    probation_started: str = ""               # when PROBATION began
    probation_trade_count: int = 0            # trades attempted while in PROBATION

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ManagerState":
        data = json.loads(s)
        # tolerate added fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})


def load_state(path: str) -> ManagerState:
    if not os.path.exists(path):
        return ManagerState()
    try:
        with open(path, "r") as f:
            return ManagerState.from_json(f.read())
    except Exception:
        return ManagerState()


def save_state(state: ManagerState, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(state.to_json())
    os.replace(tmp, path)


def roll_session_if_needed(state: ManagerState, today: str) -> bool:
    """Reset daily counters at session boundary. Returns True if rolled.

    MAY 8 2026: streak counters now reset on session roll. Previously they
    persisted across sessions, which combined with `consec_losses>=2 → REJECT`
    produced a permanent lock-up: no entries → no outcomes → streak never
    breaks. A new trading day deserves a clean slate. Long-term memory of
    losing patterns is still preserved in learning.db (bucket stats).
    `last_n_outcomes` keeps the rolling-history view for journals.
    """
    if state.session_date != today:
        state.session_date = today
        state.realized_pnl_today = 0.0
        state.trades_today = 0
        # NEW: reset streak counters at session boundary so the bot can resume
        state.consec_wins = 0
        state.consec_losses = 0
        if state.posture == POSTURE_KILLED:
            # New day: bots can resume unless something else flips us back
            state.posture = POSTURE_NORMAL
        # Drop DEFENSIVE/SIT_OUT too if they were carried over — the next
        # signal evaluation will re-impose them if conditions persist.
        if state.posture in (POSTURE_DEFENSIVE, POSTURE_SIT_OUT):
            state.posture = POSTURE_NORMAL
        # IMPORTANT: LOCKED and PROBATION survive session rolls.
        # They are multi-day flags driven by the health module, not session
        # streaks. The session-roll-clears-everything pattern would defeat
        # the entire point of "bad logic detection across days".
        return True
    return False
