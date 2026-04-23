"""Delta-based strike selection — replaces fixed ATM picking.

Root cause #5 on Apr 21: 708P selected when SPY traded at 703 gave a put
with |delta| ≈ 0.75 — every 30¢ VWAP bounce translated to a 20–25% move in
premium, producing directionally-correct trades that scratched at −8.6%.

This module picks a strike by *target delta* for each signal type, using
the ``OptionQuote`` list already populated by ``ChainBuilder``. It never
touches IB Gateway — pure data-shape-in, strike-out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from .config import StrikeSelectorConfig


@dataclass
class StrikePick:
    strike: float
    expiry: str
    expiry_date: str
    right: str                   # "C" or "P"
    delta: float
    mid: float
    reason: str


def _window_for(signal_type: str, cfg: StrikeSelectorConfig) -> Tuple[float, float]:
    mapping = {
        "TREND_CONTINUATION": cfg.trend_continuation,
        "ORB_BREAKOUT": cfg.orb_breakout,
        "PC_RATIO_EXTREME": cfg.pc_ratio_extreme,
        "CALL_SWEEP": cfg.call_sweep,
        "PUT_SWEEP": cfg.put_sweep,
    }
    return mapping.get(signal_type, (0.30, 0.45))


def select_strike(
    quotes: Iterable,
    signal_type: str,
    right: str,
    cfg: StrikeSelectorConfig,
    vix: Optional[float] = None,
    ivr: Optional[float] = None,
    prefer_0dte: bool = False,
) -> Optional[StrikePick]:
    """Return the best strike by closest absolute-delta match within window.

    ``quotes`` is expected to be an iterable of objects with attributes
    ``strike``, ``expiry``, ``expiry_date``, ``right``, ``delta``, ``bid``,
    ``ask``. The underlying ``OptionQuote`` type in chain_builder.py matches.
    """
    lo, hi = _window_for(signal_type, cfg)

    candidates: List = []
    for q in quotes:
        if q.right != right:
            continue
        ad = abs(q.delta or 0.0)
        if ad < cfg.min_delta_floor:
            # Allow deep OTM only under elevated VIX + IVR
            vix_ok = vix is not None and vix >= cfg.min_delta_floor_vix_override
            ivr_ok = ivr is not None and ivr >= cfg.min_delta_floor_ivr_override
            if not (vix_ok and ivr_ok):
                continue
        if not (lo <= ad <= hi):
            continue
        candidates.append(q)

    if not candidates:
        return None

    # Prefer 0DTE in low-IV regimes; otherwise keep whatever the chain offers
    def _dte_for(q) -> int:
        return getattr(q, "dte", 0)

    if prefer_0dte and ivr is not None and ivr < cfg.prefer_0dte_ivr_max:
        zero_dte = [q for q in candidates if _dte_for(q) == 0]
        if zero_dte:
            candidates = zero_dte

    # Pick the candidate closest to the midpoint of the target window
    target = (lo + hi) / 2.0
    best = min(candidates, key=lambda q: abs(abs(q.delta or 0.0) - target))

    mid = ((best.bid or 0.0) + (best.ask or 0.0)) / 2.0
    return StrikePick(
        strike=best.strike,
        expiry=best.expiry,
        expiry_date=getattr(best, "expiry_date", ""),
        right=best.right,
        delta=best.delta or 0.0,
        mid=mid,
        reason=(
            f"delta={abs(best.delta or 0.0):.2f} in window "
            f"[{lo:.2f},{hi:.2f}] for {signal_type}"
        ),
    )
