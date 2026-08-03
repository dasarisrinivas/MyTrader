"""V2 directional-composite gate — SHADOW ONLY. Logs decisions, never trades.

Purpose: forward-test the Variant-2 gate from the 2026-08-03 directional-gating
study WITHOUT putting unvalidated logic in the live order path. This module is
pure observation: it scores each dispatched signal, records take/skip, and writes
one JSONL line. It NEVER gates, sizes, prices, or blocks anything.

FROZEN PARAMETERS — fitted on TRAIN ONLY (82 signals / 11 sessions, sessions
< 2026-08-03 study cut 2026-07-23). Hardcoded deliberately so the gate can never
be silently refitted. Changing any number below invalidates the pre-registered
stability study and requires a new pre-registration.

    gate = composite(external_composite⁻, equity_pc⁻) > 0  AND  regime == RANGE_BOUND

Variants logged in parallel (all pre-registered 2026-08-03):
    A  baseline        — every CALL_SWEEP signal
    B  composite>0 AND regime==RANGE_BOUND      (Variant 2 as built)
    C  composite>0, all regimes                 (regime term dropped)

Standing evidence at freeze time (7 OOS sessions — NOT significant):
    session-level 95% CI: A [-706,+2118]  B [-383,+2181]  C [-193,+2159]
    all include zero; B concentration 57% (fails criterion 5); LOSO -49%.
The study exists to determine whether these hold up. Nothing here authorizes
trading.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ── FROZEN (do not edit without a new pre-registration) ──────────────────────
GATE_VERSION = "v2-shadow-1.0-frozen-20260803"
TRAIN_WINDOW = "82 signals / 11 sessions, < 2026-07-23"
_FEATURES = (
    # (field, sign, train_mean, train_std)
    ("external_composite", -1.0, -0.086618, 0.183650),
    ("equity_pc",          -1.0,  0.578659, 0.253492),
)
_BULLISH_REGIMES = {"RANGE_BOUND"}
_PATH = "logs/v2_shadow_gate.jsonl"


def composite_score(sig) -> Optional[float]:
    """Mean signed z-score over the frozen features. None if a field is absent."""
    try:
        parts = []
        for field, sign, mu, sd in _FEATURES:
            v = getattr(sig, field, None)
            if v is None or sd <= 0:
                return None
            parts.append(sign * ((float(v) - mu) / sd))
        return sum(parts) / len(parts)
    except Exception:
        return None


def evaluate(sig) -> Optional[Dict[str, Any]]:
    """Return the shadow decision for one signal. Never raises."""
    try:
        score = composite_score(sig)
        regime = getattr(sig, "regime", None) or ""
        if score is None:
            return {"gate_version": GATE_VERSION, "score": None,
                    "take_A": True, "take_B": False, "take_C": False,
                    "reason": "missing feature"}
        return {
            "gate_version": GATE_VERSION,
            "score": round(score, 6),
            "regime": regime,
            "take_A": True,                                        # baseline
            "take_B": bool(score > 0 and regime in _BULLISH_REGIMES),
            "take_C": bool(score > 0),
            "reason": "",
        }
    except Exception:
        return None


def record(sig) -> None:
    """Log the shadow decision for a dispatched signal. LOG ONLY — never gates.

    Swallows every exception: this must never affect signal dispatch or orders.
    """
    try:
        d = evaluate(sig)
        if d is None:
            return
        rec = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "signal_type": getattr(getattr(sig, "signal_type", None), "value",
                                   str(getattr(sig, "signal_type", ""))),
            "strike": getattr(sig, "strike", None),
            "right": getattr(sig, "right", None),
            "expiry_date": getattr(sig, "expiry_date", None),
            "dte": getattr(sig, "dte", None),
            "confidence": getattr(sig, "confidence", None),
            "confidence_tier": getattr(sig, "confidence_tier", None),
            "spy_price": getattr(sig, "spy_price", None),
            # frozen-gate inputs, recorded so the decision is auditable later
            "external_composite": getattr(sig, "external_composite", None),
            "equity_pc": getattr(sig, "equity_pc", None),
            **d,
        }
        os.makedirs("logs", exist_ok=True)
        with open(_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass
