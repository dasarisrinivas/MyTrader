"""
Strategy state management for adaptive backtests.

This module centralizes the mutable parameters that Agents 2-4 adjust
while the backtest progresses (confidence thresholds, risk scalars, etc.).
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass
class StrategyState:
    """Persisted knobs that downstream agents can tune nightly."""

    decision_threshold: float = 0.58
    min_confidence: float = 0.55
    exploration_rate: float = 0.1
    risk_multiplier: float = 1.0
    max_trades_per_day: int = 6
    target_daily_trades: int = 2
    last_adjusted_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if not data["last_adjusted_at"]:
            data["last_adjusted_at"] = datetime.now(timezone.utc).isoformat()
        return data


DEFAULT_STRATEGY_STATE: Dict[str, Any] = StrategyState().to_dict()


class StrategyStateManager:
    """Load, persist, and update adaptive strategy parameters."""

    def __init__(self, state_path: Path):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state: Dict[str, Any] | None = None

    def load_state(self) -> Dict[str, Any]:
        """Return the current strategy state (creating defaults when missing)."""
        if self._state is None:
            if self.state_path.exists():
                with self.state_path.open("r", encoding="utf-8") as handle:
                    stored = json.load(handle)
                self._state = {**deepcopy(DEFAULT_STRATEGY_STATE), **stored}
            else:
                self._state = deepcopy(DEFAULT_STRATEGY_STATE)
                self.save_state()
        return deepcopy(self._state)

    def save_state(self) -> None:
        """Persist current state to disk."""
        if self._state is None:
            self._state = deepcopy(DEFAULT_STRATEGY_STATE)
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(self._state, handle, indent=2)

    def update_state(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Apply updates and return the resulting state snapshot."""
        state = self.load_state()
        state.update(updates)
        state["last_adjusted_at"] = datetime.now(timezone.utc).isoformat()
        self._state = state
        self.save_state()
        return deepcopy(self._state)

    def append_adjustment_record(self, adjustment: Dict[str, Any]) -> None:
        """Store one-line audit of a nightly adjustment."""
        history_file = self.state_path.parent / "strategy_adjustments.ndjson"
        record = {
            **adjustment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with history_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
