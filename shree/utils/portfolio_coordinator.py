"""
portfolio_coordinator.py — Cross-bot position awareness via shared state file.

APR 10 2026: Created to prevent MES and SPY options bots from taking
opposing directional risk simultaneously. On a $5k account, MES long +
SPY put-spread loss on a down-day = -$260 (-5.2%), and both bots being
long-biased into a gap event can reach -8% to -12%.

Protocol:
  1. Each bot calls `publish_position()` whenever it opens or closes.
  2. Before entering, a bot calls `check_portfolio_conflict()` to see if
     a sibling bot already has directional exposure that conflicts.

State file: data/portfolio_state.json
Format:
  {
    "mes": {"direction": "LONG", "entry_price": 6521.25, "updated_at": "..."},
    "spy_options": {"direction": "SHORT", "strategy": "put_spread", ...},
    "gold": null
  }

Conflict rules:
  - MES LONG  + SPY call_spread (bearish)  = CONFLICT
  - MES SHORT + SPY put_spread  (bullish)  = CONFLICT
  - MES any   + SPY strangle               = ALLOWED (delta-neutral)
  - Gold      + anything                   = ALLOWED (uncorrelated)

Design:
  - File-based, no shared memory or IPC needed
  - Atomic write (tmp + rename) to prevent corrupt reads
  - Stale position detection (>6h without update → ignore)
  - Each bot is responsible for its own entry — this module only advises
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from shree.utils.logger import logger

_DEFAULT_PATH = Path("data/portfolio_state.json")
_STALE_HOURS = 6  # Positions older than this are considered stale


def publish_position(
    bot_name: str,
    direction: Optional[str],
    entry_price: float = 0.0,
    strategy: str = "",
    path: Path = _DEFAULT_PATH,
) -> None:
    """Publish this bot's current position to the shared state file.

    Args:
        bot_name: "mes", "spy_options", or "gold"
        direction: "LONG", "SHORT", or None (flat)
        entry_price: Entry price (for reference)
        strategy: e.g. "put_spread", "call_spread", "strangle"
        path: Override path (for tests)
    """
    state = _read_state(path)

    if direction is None:
        state[bot_name] = None
    else:
        state[bot_name] = {
            "direction": direction.upper(),
            "entry_price": entry_price,
            "strategy": strategy,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    _write_state(state, path)
    logger.info(
        f"portfolio_coordinator: {bot_name} → "
        f"{direction or 'FLAT'}"
        f"{f' ({strategy})' if strategy else ''}"
    )


def check_portfolio_conflict(
    requesting_bot: str,
    proposed_direction: str,
    proposed_strategy: str = "",
    path: Path = _DEFAULT_PATH,
) -> Optional[str]:
    """Check if a proposed entry conflicts with sibling bot positions.

    Args:
        requesting_bot: "mes", "spy_options", or "gold"
        proposed_direction: "LONG" or "SHORT"
        proposed_strategy: e.g. "put_spread", "call_spread", "strangle"
        path: Override path

    Returns:
        None if no conflict, or a string describing the conflict reason.
    """
    state = _read_state(path)
    proposed_dir = proposed_direction.upper()
    now = datetime.now(timezone.utc)

    for bot_name, pos in state.items():
        if bot_name == requesting_bot:
            continue
        if pos is None:
            continue

        # Check staleness
        updated_at = pos.get("updated_at")
        if updated_at:
            try:
                ts = datetime.fromisoformat(updated_at)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_hours = (now - ts).total_seconds() / 3600
                if age_hours > _STALE_HOURS:
                    continue  # Stale — ignore
            except Exception:
                pass

        sibling_dir = pos.get("direction", "").upper()
        sibling_strategy = pos.get("strategy", "").lower()

        # Gold is uncorrelated to S&P — no conflict with MES or SPY
        if bot_name == "gold" or requesting_bot == "gold":
            continue

        # Strangle is delta-neutral — no directional conflict
        if sibling_strategy == "strangle" or proposed_strategy.lower() == "strangle":
            continue

        # MES direction vs SPY options strategy
        # put_spread = bullish (profits when SPY stays above strike)
        # call_spread = bearish (profits when SPY stays below strike)
        conflict = _check_sp500_conflict(
            requesting_bot, proposed_dir, proposed_strategy,
            bot_name, sibling_dir, sibling_strategy,
        )
        if conflict:
            return conflict

    return None


def _check_sp500_conflict(
    req_bot: str, req_dir: str, req_strat: str,
    sib_bot: str, sib_dir: str, sib_strat: str,
) -> Optional[str]:
    """Check MES vs SPY directional conflict.

    Conflict table:
      MES LONG  + SPY call_spread → CONFLICT (opposing bias)
      MES SHORT + SPY put_spread  → CONFLICT (opposing bias)
      MES LONG  + SPY put_spread  → OK (aligned bullish)
      MES SHORT + SPY call_spread → OK (aligned bearish)
    """
    # Determine effective bullish/bearish bias of each side
    def _bias(direction: str, strategy: str) -> str:
        strat = strategy.lower()
        if strat in ("put_spread", "sell_put_spread", "bull_put_spread"):
            return "BULLISH"
        if strat in ("call_spread", "sell_call_spread", "bear_call_spread"):
            return "BEARISH"
        # Raw futures direction
        if direction == "LONG":
            return "BULLISH"
        if direction == "SHORT":
            return "BEARISH"
        return "NEUTRAL"

    req_bias = _bias(req_dir, req_strat)
    sib_bias = _bias(sib_dir, sib_strat)

    if req_bias == "NEUTRAL" or sib_bias == "NEUTRAL":
        return None

    if req_bias != sib_bias:
        return (
            f"PORTFOLIO_CONFLICT: {req_bot} wants {req_dir}"
            f"{f' ({req_strat})' if req_strat else ''} [{req_bias}] "
            f"but {sib_bot} is {sib_dir}"
            f"{f' ({sib_strat})' if sib_strat else ''} [{sib_bias}]"
        )

    return None


def get_portfolio_state(path: Path = _DEFAULT_PATH) -> Dict[str, Any]:
    """Return the current portfolio state for display/logging."""
    return _read_state(path)


def _read_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_state(state: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning(f"portfolio_coordinator: write failed: {exc}")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
