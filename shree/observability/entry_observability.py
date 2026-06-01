"""Production observability — entry instrumentation (MES Phase 1).

PURPOSE
  Emit a single structured record per trade entry for production readiness.
  This is INSTRUMENTATION ONLY — it returns nothing used in any decision and is
  fully exception-isolated, so it cannot alter strategy behavior. Phase 1
  remains byte-identical whether or not this is called.

FIELDS (per spec)
  trade entry reason · regime label at entry · ADX at entry · setup type ·
  P&L attribution tag (segment key used for post-hoc attribution).

The record is written via the standard logger AND appended to an observability
store (JSONL) for offline attribution analysis.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    from loguru import logger
except Exception:  # pragma: no cover - logging must never break execution
    import logging
    logger = logging.getLogger("entry_obs")

OBS_STORE = os.environ.get("MES_OBS_STORE", "logs/entry_observability.jsonl")


def _regime_from_adx(adx: float) -> str:
    if adx >= 22:
        return "TRENDING"
    if adx <= 15:
        return "CHOPPY"
    return "MIXED"


def log_entry(
    *,
    timestamp: Any,
    setup: str,
    adx: float,
    session: Optional[str] = None,
    entry_reason: str = "",
    confidence: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Record one trade-entry observability event. NEVER raises."""
    try:
        adx = float(adx) if adx is not None else 0.0
        regime = _regime_from_adx(adx)
        # P&L attribution tag = the segment key used in the durability audit
        attribution_tag = f"{setup}/{regime}/{session or 'NA'}"
        rec = {
            "ts": str(timestamp),
            "event": "entry",
            "setup": setup,
            "adx": round(adx, 2),
            "regime": regime,
            "session": session,
            "entry_reason": (entry_reason or "")[:120],
            "confidence": confidence,
            "attribution_tag": attribution_tag,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            rec["extra"] = extra
        logger.info(
            "📈 ENTRY_OBS setup={} regime={} adx={:.1f} session={} tag={}",
            setup, regime, adx, session, attribution_tag,
        )
        try:
            os.makedirs(os.path.dirname(OBS_STORE) or ".", exist_ok=True)
            with open(OBS_STORE, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception:
            pass  # store failure must never affect the run
    except Exception:
        # Instrumentation must never break execution.
        pass
