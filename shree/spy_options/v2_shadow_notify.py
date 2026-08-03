"""V1 vs V2 shadow-comparison Telegram notifications — OBSERVABILITY ONLY.

Sits strictly DOWNSTREAM of both decisions:

    CALL_SWEEP signal
        ├─> V1 decision (production logic)
        ├─> V2 shadow gate decision (v2_shadow_gate.py)
        └─> Telegram notification      <- this module

The executor NEVER imports this. A notification failure can never affect
trading: every function swallows all exceptions and returns None.

Three message types:
  1. Standard  — every CALL_SWEEP signal, both decisions side by side
  2. Disagreement (V1 trade / V2 skip) — "what did V2 save us from"
  3. V2 opportunity (V1 skip / V2 trade) — detects V2 being too restrictive

Every message carries signal_id so later replay can join:
    telegram_message -> signal_id -> spy_signals -> replay P&L

V2 IS SHADOW ONLY. Messages state NO ORDER SENT explicitly so there is no
ambiguity during live trading.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
_LEDGER = "logs/v1_v2_comparison.jsonl"

# Per-session counters (reset daily by reset_day()).
_state: Dict[str, Any] = {"day": None, "v1": 0, "v2_b": 0, "v2_c": 0}


def reset_day(day: Optional[str] = None) -> None:
    try:
        _state.update({"day": day or datetime.now(ET).date().isoformat(),
                       "v1": 0, "v2_b": 0, "v2_c": 0})
    except Exception:
        pass


def _roll(day: str) -> None:
    if _state.get("day") != day:
        reset_day(day)


def _et(ts=None) -> datetime:
    return (ts or datetime.now(timezone.utc)).astimezone(ET)


def build(sig, decision: Dict[str, Any], signal_id: Optional[int],
          v1_trade: bool, v1_reason: str, max_trades: int) -> Optional[Dict]:
    """Build message(s) + ledger record. Returns None on any failure."""
    try:
        now = _et()
        day = now.date().isoformat()
        _roll(day)

        take_b = bool(decision.get("take_B"))
        take_c = bool(decision.get("take_C"))
        score = decision.get("score")
        regime = decision.get("regime") or getattr(sig, "regime", "")

        if v1_trade:
            _state["v1"] += 1
        if take_b:
            _state["v2_b"] += 1
        if take_c:
            _state["v2_c"] += 1

        strike = getattr(sig, "strike", None)
        right = getattr(sig, "right", "")
        tier = getattr(sig, "confidence_tier", "")
        spy = getattr(sig, "spy_price", None)
        expiry = getattr(sig, "expiry_date", "")

        why_b = ("composite > 0 and regime==RANGE_BOUND" if take_b else
                 ("composite <= 0" if (score is not None and score <= 0)
                  else f"regime {regime} != RANGE_BOUND"))

        head = (
            f"🟢 <b>CALL_SWEEP SIGNAL</b>\n\n"
            f"Time: {now.strftime('%H:%M')} ET\n"
            f"SPY: {spy if spy is not None else '—'}\n"
            f"Expiry: {expiry}\n"
            f"Strike: {strike:.0f}{right}\n" if strike is not None else
            f"🟢 <b>CALL_SWEEP SIGNAL</b>\n\nTime: {now.strftime('%H:%M')} ET\n"
        )
        msg = (
            f"{head}"
            f"Tier: {tier}\n"
            f"signal_id: {signal_id if signal_id is not None else '—'}\n\n"
            f"<b>V1 DECISION</b> (production)\n"
            f"{'✅ TRADE CALL' if v1_trade else '❌ SKIP'}\n"
            f"Reason: {v1_reason}\n\n"
            f"<b>V2 SHADOW</b> — <i>SHADOW ONLY · NO ORDER SENT</i>\n"
            f"composite = {score if score is not None else 'n/a'}  regime = {regime}\n"
            f"Variant B: {'✅ WOULD TRADE' if take_b else '❌ SKIP'} ({why_b})\n"
            f"Variant C: {'✅ WOULD TRADE' if take_c else '❌ SKIP'} "
            f"(composite {'>' if take_c else '<='} 0)\n\n"
            f"<b>Shadow tracking</b>\n"
            f"V1 trades today: {_state['v1']}/{max_trades}\n"
            f"V2-B hypothetical today: {_state['v2_b']}\n"
            f"V2-C hypothetical today: {_state['v2_c']}"
        )

        alerts = []
        if v1_trade and not take_b:
            alerts.append(
                f"⚠️ <b>V1/V2 DISAGREEMENT</b>\n\n"
                f"V1: BUY CALL\nV2-B: SKIP\n\n"
                f"Signal: CALL_SWEEP {strike:.0f}{right}\n"
                f"Time: {now.strftime('%H:%M')} ET\n"
                f"signal_id: {signal_id}\n"
                f"Reason: {why_b}\n\n"
                f"<i>Tracking this trade — entry/exit/P&amp;L resolved at replay.</i>\n"
                f"<i>SHADOW ONLY · NO ORDER SENT by V2.</i>"
            )
        if (not v1_trade) and (take_b or take_c):
            alerts.append(
                f"🔵 <b>V2 OPPORTUNITY (SHADOW)</b>\n\n"
                f"V1: NO TRADE ({v1_reason})\n"
                f"V2: WOULD TRADE\n"
                f"Variant: {'B' if take_b else 'C'}\n\n"
                f"Signal: CALL_SWEEP {strike:.0f}{right}\n"
                f"Time: {now.strftime('%H:%M')} ET\n"
                f"signal_id: {signal_id}\n"
                f"Reason: composite &gt; frozen threshold\n\n"
                f"<i>Tracking only · NO ORDER SENT.</i>"
            )

        rec = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "ts_et": now.isoformat(), "session": day,
            "signal_id": signal_id, "signal_type": "CALL_SWEEP",
            "strike": strike, "right": right, "expiry_date": expiry,
            "tier": tier, "spy_price": spy, "regime": regime,
            "composite": score,
            "v1_trade": v1_trade, "v1_reason": v1_reason,
            "v2_take_B": take_b, "v2_take_C": take_c,
            "disagreement_v1_yes_v2_no": bool(v1_trade and not take_b),
            "v2_only_opportunity": bool((not v1_trade) and (take_b or take_c)),
            "v1_count_today": _state["v1"], "v2b_count_today": _state["v2_b"],
            "v2c_count_today": _state["v2_c"],
            "gate_version": decision.get("gate_version"),
        }
        try:
            os.makedirs("logs", exist_ok=True)
            with open(_LEDGER, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass
        return {"message": msg, "alerts": alerts, "record": rec}
    except Exception:
        return None
