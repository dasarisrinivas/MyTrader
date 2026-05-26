"""Append-only manager_decisions.jsonl writer.

The signal_processor hook reads from this file to enforce vetoes.
Format is one JSON object per line, schema:

  {
    "decision_id": int,
    "ts": ISO timestamp,
    "signal_ts": signal's ts,
    "signal_type": "EMA21_PB_LONG" | ...,
    "action": "BUY"|"SELL",
    "decision": "APPROVE"|"REJECT"|"MODIFY",
    "confidence": int,
    "position_size": "small"|"normal"|"aggressive",
    "reasoning": str,
    "risk_notes": str,
    "override": bool,
    "checks": [{"name": str, "passed": bool, "note": str}],
    "state_snapshot": {
      "realized_pnl_today": float,
      "trades_today": int,
      "consec_wins": int,
      "consec_losses": int,
      "posture": str,
    }
  }
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from .rules import Decision
from .signal_watcher import Signal
from .spy_signal_watcher import SpySignal
from .state import ManagerState

try:
    # Shadow-only expectancy computation (Phase 0 of the expectancy-gate
    # proposal). Import is guarded so a problem here can never stop a live
    # decision from being logged.
    from .expectancy_priors import shadow_expectancy as _shadow_expectancy
except Exception:  # pragma: no cover - defensive
    _shadow_expectancy = None


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_decision(
    path: str,
    decision_id: int,
    sig: Signal,
    decision: Decision,
    state: ManagerState,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rec = {
        "decision_id": decision_id,
        "ts": _now_iso(),
        "signal_ts": sig.ts,
        "signal_type": sig.signal_type,
        "action": sig.action,
        "strategy": sig.strategy,
        "close": sig.close,
        "stop_loss": sig.stop_loss,
        "take_profit": sig.take_profit,
        "rr": round(sig.rr, 2),
        "adx": sig.adx,
        "decision": decision.decision,
        "confidence": decision.confidence,
        "position_size": decision.position_size,
        "reasoning": decision.reasoning,
        "risk_notes": decision.risk_notes,
        "override": decision.override,
        "checks": [
            {"name": n, "passed": p, "note": note} for n, p, note in decision.checks
        ],
        "state_snapshot": {
            "realized_pnl_today": round(state.realized_pnl_today, 2),
            "trades_today": state.trades_today,
            "consec_wins": state.consec_wins,
            "consec_losses": state.consec_losses,
            "posture": state.posture,
            "health_status": getattr(state, "health_status", ""),
            "health_triggers": list(getattr(state, "health_triggers", []) or []),
        },
    }
    # Phase 0 SHADOW: log what an expectancy-based cold-start floor WOULD decide
    # for this signal, alongside the live (unchanged) decision. Purely additive;
    # never affects gating. Wrapped so it can't break the logging path.
    if _shadow_expectancy is not None:
        try:
            shadow = _shadow_expectancy(sig.signal_type, sig.rr)
            if shadow is not None:
                rec["shadow_expectancy"] = shadow
        except Exception:
            pass
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def append_spy_decision(
    path: str,
    decision_id: int,
    sig: SpySignal,
    decision: Decision,
    state: ManagerState,
) -> None:
    """Persist a SPY-options decision to the same manager_decisions.jsonl
    file the MES decisions go to. The schema overlaps where it can and adds
    options-specific fields. The 'kind' field disambiguates.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rec = {
        "decision_id": decision_id,
        "kind": "spy_signal",
        "ts": _now_iso(),
        "signal_ts": sig.ts,
        "signal_id": sig.signal_id,
        "signal_type": sig.signal_type,
        "right": sig.right,
        "strike": sig.strike,
        "expiry": sig.expiry,
        "dte": sig.dte,
        "confidence_in": round(sig.confidence, 4),
        "spy_price": sig.spy_price,
        "regime": sig.regime,
        "spread_pct": sig.spread_pct,
        "decision": decision.decision,
        "confidence": decision.confidence,
        "position_size": decision.position_size,
        "reasoning": decision.reasoning,
        "risk_notes": decision.risk_notes,
        "override": decision.override,
        "checks": [
            {"name": n, "passed": p, "note": note} for n, p, note in decision.checks
        ],
        "state_snapshot": {
            "realized_pnl_today": round(state.realized_pnl_today, 2),
            "trades_today": state.trades_today,
            "consec_wins": state.consec_wins,
            "consec_losses": state.consec_losses,
            "posture": state.posture,
            "health_status": getattr(state, "health_status", ""),
            "health_triggers": list(getattr(state, "health_triggers", []) or []),
        },
    }
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def append_posture_transition(
    path: str,
    prior: str,
    new: str,
    reason: str,
    state: ManagerState,
    triggers: Optional[list] = None,
) -> None:
    """Additive structured record for a posture transition, written alongside
    the existing log.warning at each transition site. One JSON object per line
    in posture_transitions.jsonl. Pure telemetry — wrapped so a logging failure
    can never propagate into the trading loop.
    """
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        rec = {
            "ts": _now_iso(),
            "prior_posture": prior,
            "new_posture": new,
            "reason": reason,
            "triggers": list(triggers) if triggers else list(getattr(state, "health_triggers", []) or []),
            "health_status": getattr(state, "health_status", ""),
            "reset_condition": {
                "realized_pnl_today": round(getattr(state, "realized_pnl_today", 0.0), 2),
                "trades_today": getattr(state, "trades_today", 0),
                "consec_losses": getattr(state, "consec_losses", 0),
            },
        }
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def latest_decision_by_signal_id(path: str, signal_id: str) -> Optional[dict]:
    """Linear scan from the tail backwards for the most recent decision
    with a matching `signal_id`. Used by the SPY options bot's veto hook.
    """
    if not os.path.exists(path) or not signal_id:
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = min(size, 64 * 1024)
            f.seek(size - chunk)
            tail = f.read().decode("utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    for line in reversed(tail):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("signal_id") == signal_id:
            return rec
    return None


def latest_decision_for_signal_ts(path: str, signal_ts: str) -> Optional[dict]:
    """Linear scan from the tail backwards for the most recent decision
    matching this signal_ts. Used by the signal_processor veto hook.

    Reads only the last ~200 lines to keep this O(1) regardless of file size.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = min(size, 64 * 1024)
            f.seek(size - chunk)
            tail = f.read().decode("utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    for line in reversed(tail):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("signal_ts") == signal_ts:
            return rec
    return None


def latest_decision(path: str) -> Optional[dict]:
    """Latest decision, period. Used as a fallback when ts matching fails."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = min(size, 16 * 1024)
            f.seek(size - chunk)
            tail = f.read().decode("utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    for line in reversed(tail):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None
