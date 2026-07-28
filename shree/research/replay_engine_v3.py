"""Replay Engine v3.0 — MULTI-LEG measurement instrument.

Sanctioned unfreeze: multi-leg replay ONLY. This module is ADDITIVE — it imports
v2 (`replay_engine.py`) and does not modify it, so every single-leg scorecard
stays bit-for-bit reproducible.

v3.0 scope: vertical spreads (debit + credit, calls + puts) and straddles /
strangles. Calendars, iron condors, and other structures are DEFERRED.

PRICING (no optimistic fills):
    entry  long leg = ASK   short leg = BID
    exit   long leg = BID   short leg = ASK
    fees   $0.65 / contract / leg / side  ->  $2.60 round trip for 2 legs

LEG EXTRACTION RULE (hard): the `strike` COLUMN of a signal row is an ATM/VWAP
reference and is NOT a leg. Verified defect: LONG_STRADDLE rows carry
strike=680.0 while the real legs are 685C + 685P. Legs are parsed ONLY from
`suggested_trade`. Unparseable text fails closed (returns None) — never guessed.

KNOWN LIMITATION (permanent, must appear in every v3 report):
    American-style SPY options — EARLY ASSIGNMENT of short legs is NOT MODELED.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .replay_engine import QuoteCache, _prevailing, RESEARCH_ENGINE_VERSION

RESEARCH_ENGINE_V3_VERSION = "v3.0-multi-leg"
COMMISSION_PER_LEG_SIDE = 0.65
EARLY_ASSIGNMENT_MODELED = False
V3_LIMITATION = ("American-style SPY options: early assignment of short legs is "
                 "NOT modeled.")

_LEG_RE = re.compile(r"\b(Buy|Sell)\s+(\d+(?:\.\d+)?)\s*([CP])\b", re.IGNORECASE)


@dataclass(frozen=True)
class Leg:
    side: str        # "buy" | "sell"
    right: str       # "C" | "P"
    strike: float
    expiry: str      # YYYYMMDD
    qty: int = 1
    symbol: str = "SPY"

    @property
    def sign(self) -> int:
        return 1 if self.side == "buy" else -1


@dataclass
class Structure:
    kind: str                # vertical_debit|vertical_credit|straddle|strangle
    legs: List[Leg]
    width: Optional[float] = None     # strike distance for verticals
    source: str = "parsed"            # parsed | synthetic


@dataclass
class MultiLegResult:
    ok: bool
    reason: str = ""
    kind: str = ""
    legs: List[Dict] = field(default_factory=list)
    entry_ts: Optional[str] = None
    exit_ts: Optional[str] = None
    entry_net: Optional[float] = None   # +debit paid / -credit received (per share)
    exit_net: Optional[float] = None    # liquidation value (per share)
    gross_dollar: Optional[float] = None
    fees: Optional[float] = None
    net_dollar: Optional[float] = None
    max_loss: Optional[float] = None
    max_profit: Optional[float] = None   # None == unbounded
    mfe: Optional[float] = None
    mae: Optional[float] = None
    invariant_ok: Optional[bool] = None


# ── parsing ──────────────────────────────────────────────────────────────────

def parse_structure(suggested_trade: str, expiry_date: str) -> Optional[Structure]:
    """Parse legs from `suggested_trade`. Fails closed on anything unexpected.

    Handles:
      "Buy 693C / Sell 698C exp APR26"          -> vertical
      "Buy 660P / Sell 655P exp MAY26"          -> vertical
      "Long Straddle: Buy 685C + Buy 685P ..."  -> straddle
      "... Buy 690C + Buy 685P ..."             -> strangle
    `expiry_date` (YYYYMMDD) comes from the signal row — the text carries only a
    month code, which does not identify a contract.
    """
    if not suggested_trade or not expiry_date:
        return None
    found = _LEG_RE.findall(suggested_trade)
    if len(found) != 2:
        return None                      # v3.0 supports exactly 2 legs
    legs = [Leg(side=s.lower(), right=r.upper(), strike=float(k),
                expiry=str(expiry_date)) for s, k, r in found]
    a, b = legs
    if a.side == "buy" and b.side == "buy":
        if a.right != b.right:
            kind = "straddle" if a.strike == b.strike else "strangle"
            return Structure(kind=kind, legs=legs)
        return None                      # two same-right buys: unsupported
    if a.side != b.side:
        if a.right != b.right:
            return None                  # diagonal-by-right: unsupported in v3.0
        return Structure(kind="vertical", legs=legs,
                         width=abs(a.strike - b.strike))
    return None


def synthetic_vertical(right: str, long_strike: float, short_strike: float,
                       expiry: str) -> Structure:
    """Construct a vertical directly (used for credit structures, which have no
    historical signals). kind is resolved to debit/credit at pricing time."""
    legs = [Leg("buy", right.upper(), float(long_strike), expiry),
            Leg("sell", right.upper(), float(short_strike), expiry)]
    return Structure(kind="vertical", legs=legs,
                     width=abs(long_strike - short_strike), source="synthetic")


# ── pricing ──────────────────────────────────────────────────────────────────

def _key(dt: datetime) -> str:
    return dt.replace(tzinfo=None).isoformat(timespec="milliseconds")


def _leg_quotes(cache: QuoteCache, leg: Leg, session_ymd: str):
    return cache.get(leg.expiry, leg.strike, leg.right, session_ymd)


def _price_at(cache: QuoteCache, legs: List[Leg], when: datetime,
              session_ymd: str, mode: str) -> Tuple[Optional[float], List[Dict]]:
    """mode='entry' -> pay ask on longs, receive bid on shorts (net debit +)
       mode='exit'  -> receive bid on longs, pay ask on shorts (liquidation)
    Returns (net_per_share, per_leg_detail) or (None, []) if any leg unquoted —
    both legs must be synchronized at the same timestamp (V2)."""
    net = 0.0
    detail = []
    for leg in legs:
        try:
            q, k = _leg_quotes(cache, leg, session_ymd)
        except Exception:
            return None, []
        if not q:
            return None, []
        bid, ask = _prevailing(q, k, _key(when))
        if not bid or not ask:
            return None, []
        if mode == "entry":
            px = ask if leg.side == "buy" else bid
        else:
            px = bid if leg.side == "buy" else ask
        net += leg.sign * px * leg.qty
        detail.append({"side": leg.side, "right": leg.right, "strike": leg.strike,
                       "expiry": leg.expiry, "qty": leg.qty,
                       "bid": bid, "ask": ask, "used": px})
    return net, detail


def replay_multileg(cache: QuoteCache, structure: Structure,
                    entry_dt: datetime, exit_dt: datetime,
                    commission_per_leg_side: float = COMMISSION_PER_LEG_SIDE
                    ) -> MultiLegResult:
    if entry_dt.date() != exit_dt.date():
        return MultiLegResult(False, "cross-session (v3.0 intraday only)")
    session = entry_dt.strftime("%Y%m%d")

    entry_net, entry_detail = _price_at(cache, structure.legs, entry_dt,
                                        session, "entry")
    if entry_net is None:
        return MultiLegResult(False, "no synchronized NBBO at entry")
    exit_net, _ = _price_at(cache, structure.legs, exit_dt, session, "exit")
    if exit_net is None:
        return MultiLegResult(False, "no synchronized NBBO at exit")

    fees = commission_per_leg_side * len(structure.legs) * 2
    gross = (exit_net - entry_net) * 100.0
    net = gross - fees

    # kind resolution + risk bounds
    kind = structure.kind
    max_loss = max_profit = None
    if kind == "vertical" and structure.width is not None:
        if entry_net > 0:
            kind = "vertical_debit"
            max_loss = entry_net * 100.0 + fees
            max_profit = (structure.width - entry_net) * 100.0 - fees
        else:
            kind = "vertical_credit"
            credit = -entry_net
            max_profit = credit * 100.0 - fees
            max_loss = (structure.width - credit) * 100.0 + fees
    elif kind in ("straddle", "strangle"):
        max_loss = entry_net * 100.0 + fees      # long premium
        max_profit = None                        # unbounded

    # MFE / MAE over the liquidation path between entry and exit
    mfe = mae = None
    try:
        q0, k0 = _leg_quotes(cache, structure.legs[0], session)
        lo, hi = _key(entry_dt), _key(exit_dt)
        stamps = [t for t in k0 if lo <= t <= hi]
        step = max(1, len(stamps) // 120)        # cap work; ~120 samples
        for ts in stamps[::step]:
            when = datetime.fromisoformat(ts)
            v, _ = _price_at(cache, structure.legs, when, session, "exit")
            if v is None:
                continue
            pnl = (v - entry_net) * 100.0 - fees
            mfe = pnl if mfe is None else max(mfe, pnl)
            mae = pnl if mae is None else min(mae, pnl)
    except Exception:
        pass
    if mfe is None:
        mfe, mae = net, net

    inv_ok = True
    if max_loss is not None and net < -(max_loss + 0.01):
        inv_ok = False
    if max_profit is not None and net > (max_profit + 0.01):
        inv_ok = False

    return MultiLegResult(
        ok=True, kind=kind, legs=entry_detail,
        entry_ts=_key(entry_dt), exit_ts=_key(exit_dt),
        entry_net=round(entry_net, 4), exit_net=round(exit_net, 4),
        gross_dollar=round(gross, 2), fees=round(fees, 2),
        net_dollar=round(net, 2),
        max_loss=round(max_loss, 2) if max_loss is not None else None,
        max_profit=round(max_profit, 2) if max_profit is not None else None,
        mfe=round(mfe, 2), mae=round(mae, 2), invariant_ok=inv_ok)


# ── provenance (V7) ──────────────────────────────────────────────────────────

def provenance(dataset_ids: Optional[List[int]] = None) -> Dict:
    here = Path(__file__).parent

    def h(p):
        return hashlib.sha256((here / p).read_bytes()).hexdigest()[:16]

    ds = ""
    if dataset_ids:
        ds = hashlib.sha256(",".join(str(i) for i in sorted(dataset_ids))
                            .encode()).hexdigest()[:16]
    return {
        "engine_v2_version": RESEARCH_ENGINE_VERSION,
        "engine_v2_hash": h("replay_engine.py"),
        "engine_v3_version": RESEARCH_ENGINE_V3_VERSION,
        "engine_v3_hash": h("replay_engine_v3.py"),
        "constitution_hash": h("promotion_constitution.py"),
        "config_hash": hashlib.sha256(
            f"{COMMISSION_PER_LEG_SIDE}".encode()).hexdigest()[:16],
        "dataset_hash": ds,
        "early_assignment_modeled": EARLY_ASSIGNMENT_MODELED,
        "limitation": V3_LIMITATION,
    }
