"""Execution-quality telemetry — LOG ONLY, never alters trading behavior.

Audit 2026-07-31: the SPY options bot has executed ZERO orders in the audited
window, so slippage, fill rate, and every latency figure are UNKNOWN. That gap
must be closed BEFORE the first live fill, not after — otherwise the first real
trades produce no evaluable execution record.

Writes one JSON line per execution event to logs/execution_quality.jsonl:

    SUBMIT      bracket placed        -> intended limit, live NBBO at submit
    ACK         broker acknowledged   -> latency submit -> ack
    FILL        entry filled          -> avg fill px, slippage vs limit and mid,
                                         latency submit -> fill
    EXIT_FILL   TP/SL/manual exit     -> exit px, exit reason, hold minutes
    CANCEL      order cancelled       -> reason, latency
    REJECT      order rejected        -> reason (IB message)
    FLATTEN     EOD 0DTE flatten      -> latency

Every function swallows all exceptions: telemetry must never affect an order.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_PATH = "logs/execution_quality.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None      # drop NaN
    except (TypeError, ValueError):
        return None


def _i(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def record(event: str, **fields: Any) -> None:
    """Append one telemetry event. Never raises."""
    try:
        rec: Dict[str, Any] = {"ts_utc": _now(), "event": event}
        for k, v in fields.items():
            if isinstance(v, datetime):
                v = v.isoformat()
            rec[k] = v
        os.makedirs("logs", exist_ok=True)
        with open(_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


def _ms(a: Optional[datetime], b: Optional[datetime]) -> Optional[int]:
    try:
        if a is None or b is None:
            return None
        return int((b - a).total_seconds() * 1000)
    except Exception:
        return None


def submit(key: str, contract, qty: int, entry_limit: float,
           bid: Optional[float], ask: Optional[float],
           order_id: Optional[int], tp_price: Optional[float] = None,
           sl_stop: Optional[float] = None) -> None:
    """Bracket submitted. Captures the NBBO at submit so slippage is measurable."""
    mid = None
    if bid and ask and bid > 0 and ask > 0:
        mid = (bid + ask) / 2.0
    record(
        "SUBMIT", key=key, order_id=_i(order_id), qty=_i(qty),
        symbol=getattr(contract, "symbol", None),
        strike=_f(getattr(contract, "strike", None)),
        right=getattr(contract, "right", None),
        expiry=getattr(contract, "lastTradeDateOrContractMonth", None),
        conid=_i(getattr(contract, "conId", None)),
        entry_limit=_f(entry_limit), quote_bid=_f(bid), quote_ask=_f(ask),
        quote_mid=_f(mid),
        spread=_f((ask - bid) if (bid and ask) else None),
        tp_price=_f(tp_price), sl_stop=_f(sl_stop),
    )


def ack(key: str, order_id: Optional[int], status: Optional[str],
        submitted_at: Optional[datetime]) -> None:
    record("ACK", key=key, order_id=_i(order_id), status=status,
           latency_submit_to_ack_ms=_ms(submitted_at, datetime.now(timezone.utc)))


def fill(key: str, order_id: Optional[int], qty_filled: Optional[int],
         qty_ordered: Optional[int], avg_fill_price: Optional[float],
         entry_limit: Optional[float], quote_mid_at_submit: Optional[float],
         submitted_at: Optional[datetime], reprices: Optional[int] = None) -> None:
    """Entry fill. Slippage is signed: POSITIVE = paid MORE than reference."""
    afp, lim = _f(avg_fill_price), _f(entry_limit)
    mid = _f(quote_mid_at_submit)
    record(
        "FILL", key=key, order_id=_i(order_id),
        qty_filled=_i(qty_filled), qty_ordered=_i(qty_ordered),
        partial=(None if qty_filled is None or qty_ordered is None
                 else bool(qty_filled < qty_ordered)),
        avg_fill_price=afp, entry_limit=lim, quote_mid_at_submit=mid,
        slippage_vs_limit=(None if (afp is None or lim is None) else round(afp - lim, 4)),
        slippage_vs_mid=(None if (afp is None or mid is None) else round(afp - mid, 4)),
        slippage_vs_mid_usd=(None if (afp is None or mid is None)
                             else round((afp - mid) * 100.0 * (qty_filled or 1), 2)),
        latency_submit_to_fill_ms=_ms(submitted_at, datetime.now(timezone.utc)),
        entry_reprices=_i(reprices),
    )


def exit_fill(key: str, reason: str, exit_price: Optional[float],
              entry_price: Optional[float], qty: Optional[int],
              opened_at: Optional[datetime]) -> None:
    ex, en = _f(exit_price), _f(entry_price)
    record(
        "EXIT_FILL", key=key, exit_reason=reason,
        exit_price=ex, entry_price=en, qty=_i(qty),
        gross_usd=(None if (ex is None or en is None)
                   else round((ex - en) * 100.0 * (qty or 1), 2)),
        hold_minutes=(None if opened_at is None
                      else round((datetime.now(timezone.utc) -
                                  opened_at.replace(tzinfo=timezone.utc)
                                  ).total_seconds() / 60.0, 2)),
    )


def cancel(key: str, order_id: Optional[int], reason: str,
           submitted_at: Optional[datetime] = None) -> None:
    record("CANCEL", key=key, order_id=_i(order_id), reason=reason,
           latency_submit_to_cancel_ms=_ms(submitted_at, datetime.now(timezone.utc)))


def reject(key: str, order_id: Optional[int], reason: str,
           submitted_at: Optional[datetime] = None) -> None:
    record("REJECT", key=key, order_id=_i(order_id), reason=reason,
           latency_submit_to_reject_ms=_ms(submitted_at, datetime.now(timezone.utc)))


def flatten(key: str, reason: str = "0DTE EOD flatten") -> None:
    record("FLATTEN", key=key, reason=reason)
