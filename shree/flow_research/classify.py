"""Print classification: aggressor side, sweeps, blocks, condition filtering.

The hardest and most abuse-prone part of any flow product. Every function here
is deliberately conservative and tags its own confidence so the validation layer
can down-weight or drop weak classifications. See docs/FLOW_RESEARCH_DESIGN.md
section 7 (risks) — a broken classifier is caught by the shuffle test.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Set

from .models import (
    Print,
    BUY,
    SELL,
    MID,
    UNKNOWN,
    SRC_QUOTE,
    SRC_TICK,
    SRC_NONE,
)


# OPRA-style condition codes that make a print NON-directional or unreliable.
# These are the codes that flow-product marketing quietly counts as "buys".
# Codes are vendor-dependent; this default set is overridable per feed.
DEFAULT_EXCLUDED_CONDITIONS: Set[str] = {
    "SPREAD",       # leg of a multi-leg spread — not a directional call/put buy
    "COMBO",        # combination order leg
    "MULTILEG",
    "AUCTION",      # opening/closing auction print — not real-time intent
    "LATE",         # late / out-of-sequence report
    "OUT_OF_SEQ",
    "CANCEL",
    "CORRECTION",
}

# ISO / sweep-related condition codes (used for sweep inference, not exclusion).
ISO_CONDITIONS: Set[str] = {"ISO", "INTERMARKET_SWEEP", "SWEEP"}


def is_clean_print(p: Print, excluded: Set[str] | None = None) -> bool:
    """True if the print is a plain, directional, real-time execution.

    Spread legs, auctions, late/corrected prints are excluded from all NET
    measures (they may still be stored and counted diagnostically).
    """
    ex = excluded if excluded is not None else DEFAULT_EXCLUDED_CONDITIONS
    for c in (p.condition_codes or []):
        if c.upper() in ex:
            return False
    return True


def classify_aggressor(p: Print, eps: float = 1e-9) -> Print:
    """Lee-Ready style aggressor classification against synchronized NBBO.

    Rules (in order):
      trade >= ask - eps           -> BUY  (lifted the offer)
      trade <= bid + eps           -> SELL (hit the bid)
      trade  > midpoint            -> BUY  (quote rule, weaker but QUOTE-based)
      trade  < midpoint            -> SELL
      trade == midpoint            -> MID  (ambiguous, excluded from net)
      no usable quote              -> UNKNOWN / SRC_NONE

    Mutates and returns `p` (sets aggressor + aggressor_src).
    """
    bid, ask = p.bid, p.ask
    if bid is None or ask is None or ask <= 0 or bid <= 0 or ask < bid:
        p.aggressor = UNKNOWN
        p.aggressor_src = SRC_NONE
        return p

    px = float(p.trade_px)
    if px >= ask - eps:
        p.aggressor, p.aggressor_src = BUY, SRC_QUOTE
        return p
    if px <= bid + eps:
        p.aggressor, p.aggressor_src = SELL, SRC_QUOTE
        return p

    mid = (bid + ask) / 2.0
    if px > mid + eps:
        p.aggressor, p.aggressor_src = BUY, SRC_QUOTE
    elif px < mid - eps:
        p.aggressor, p.aggressor_src = SELL, SRC_QUOTE
    else:
        p.aggressor, p.aggressor_src = MID, SRC_QUOTE
    return p


def classify_all(prints: Iterable[Print]) -> List[Print]:
    """Classify a batch of prints (aggressor only). Returns the list."""
    out = [classify_aggressor(p) for p in prints]
    return out


def mark_blocks(prints: Sequence[Print], min_size_by_root: dict | None = None) -> None:
    """Flag institutional-size single prints. Root-specific thresholds.

    Defaults are intentionally high: SPY option day-flow is dominated by tiny
    retail lots, so a "block" must be large to mean anything.
    """
    thresholds = min_size_by_root or {"SPY": 250, "SPX": 50}
    for p in prints:
        floor = thresholds.get(p.root.upper(), 250)
        p.is_block = int(p.size) >= floor


def mark_sweeps(
    prints: Sequence[Print],
    window_ms: int = 500,
    min_venues: int = 2,
) -> None:
    """Flag sweeps: same contract + same aggressor side executing across
    multiple exchanges within a tight time window (taker urgency).

    Requires exchange codes and quote-classified aggressor. Prints must be
    time-sortable by ts_utc (ISO strings sort lexically when zero-padded).
    """
    # group by (contract, side); a sweep = >= min_venues distinct exchanges
    # within window_ms of each other.
    def _key(p: Print):
        return (p.root, p.expiry, p.strike, p.right, p.aggressor)

    # bucket indices by key
    buckets: dict = {}
    for i, p in enumerate(prints):
        if p.aggressor not in (BUY, SELL):
            continue
        buckets.setdefault(_key(p), []).append(i)

    for _key_val, idxs in buckets.items():
        idxs.sort(key=lambda i: prints[i].ts_utc)
        # sliding window over prints of this contract+side
        lo = 0
        for hi in range(len(idxs)):
            # shrink window from the left until within window_ms
            while lo < hi and _ms_gap(prints[idxs[lo]].ts_utc,
                                      prints[idxs[hi]].ts_utc) > window_ms:
                lo += 1
            venues = {prints[idxs[k]].exchange for k in range(lo, hi + 1)
                      if prints[idxs[k]].exchange}
            if len(venues) >= min_venues:
                for k in range(lo, hi + 1):
                    prints[idxs[k]].is_sweep = True


def _ms_gap(ts_a: str, ts_b: str) -> float:
    """Millisecond gap between two ISO timestamps. Best-effort; large on parse
    failure so unrelated prints are never grouped."""
    from datetime import datetime

    def _p(s: str):
        try:
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s)
        except (ValueError, AttributeError):
            return None

    a, b = _p(ts_a), _p(ts_b)
    if a is None or b is None:
        return float("inf")
    return abs((b - a).total_seconds()) * 1000.0
