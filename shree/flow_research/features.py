"""Flow feature computation — MEASUREMENTS ONLY, never signals.

Given a window of classified prints, produce a Snapshot of numbers. No
thresholds, no BUY/SELL decision, no score that a trader could act on. The
validation layer decides later whether any of these numbers predict anything.

Separation rule (design doc section 2): NET measures use only prints that are
  - aggressor in (BUY, SELL)          (direction is known)
  - aggressor_src == QUOTE            (classification is trustworthy)
  - clean condition codes             (not a spread leg / auction / late print)
Everything else is diagnostic only and excluded from the net measures.
"""
from __future__ import annotations

from typing import List, Sequence

from .models import Print, Snapshot, BUY, SELL, SRC_QUOTE
from .classify import is_clean_print


ATM_BAND = 0.005  # ±0.5% of spot counts as "at the money"


def _usable_for_net(p: Print) -> bool:
    return (
        p.aggressor in (BUY, SELL)
        and p.aggressor_src == SRC_QUOTE
        and is_clean_print(p)
    )


def compute_features(
    prints: Sequence[Print],
    *,
    snapshot_kind: str,
    session_date: str,
    ts_et: str,
    window_s: int,
    signal_id: int | None = None,
) -> Snapshot:
    """Compute all Tier-A and Tier-B measures over the given print window."""
    snap = Snapshot(
        snapshot_kind=snapshot_kind,
        session_date=session_date,
        ts_et=ts_et,
        window_s=window_s,
        signal_id=signal_id,
        n_prints=len(prints),
    )

    used: List[Print] = [p for p in prints if _usable_for_net(p)]
    snap.n_prints_used = len(used)
    if not used:
        return snap

    # ── Tier A ────────────────────────────────────────────────────────────
    net_call = 0.0
    net_put = 0.0
    total_prem = 0.0        # gross |premium| of used prints
    dw = 0.0                # delta-weighted signed exposure
    sweep_prem = 0.0
    block_prem = 0.0
    open_prem = 0.0

    for p in used:
        prem = p.premium
        s = p.signed_sign            # +1 buy / -1 sell
        total_prem += prem
        if p.is_call:
            net_call += s * prem
        else:
            net_put += s * prem
        if p.delta is not None:
            # directional exposure actually transacted; put delta is negative,
            # so buying puts correctly contributes bearish (negative) flow.
            dw += s * float(p.delta) * int(p.size) * 100.0
        if p.is_sweep:
            sweep_prem += prem
        if p.is_block:
            block_prem += prem
        if p.oc_estimate == "OPEN":
            open_prem += prem

    snap.net_call_prem = net_call
    snap.net_put_prem = net_put
    snap.pc_prem_imbalance = (
        (net_call - net_put) / total_prem if total_prem > 0 else 0.0
    )
    snap.dw_flow = dw
    snap.sweep_intensity = sweep_prem / total_prem if total_prem > 0 else 0.0
    snap.block_prem = block_prem
    snap.oc_open_ratio = open_prem / total_prem if total_prem > 0 else 0.0

    # ── Tier B ────────────────────────────────────────────────────────────
    snap.expiry_concentration = _herfindahl(
        _premium_by(used, key=lambda p: p.expiry)
    )
    snap.strike_repetition = _max_share(
        _premium_by(used, key=lambda p: (p.expiry, p.strike, p.right))
    )
    snap.atm_vs_wing = _atm_share(used)
    snap.iv_weighted_side = _iv_weighted_side(used)

    return snap


# ── helpers ──────────────────────────────────────────────────────────────

def _premium_by(prints: Sequence[Print], key) -> dict:
    out: dict = {}
    for p in prints:
        out[key(p)] = out.get(key(p), 0.0) + p.premium
    return out


def _herfindahl(prem_map: dict) -> float:
    """HHI of premium concentration across buckets. 1.0 = all in one bucket."""
    total = sum(prem_map.values())
    if total <= 0:
        return 0.0
    return sum((v / total) ** 2 for v in prem_map.values())


def _max_share(prem_map: dict) -> float:
    total = sum(prem_map.values())
    if total <= 0:
        return 0.0
    return max(prem_map.values()) / total


def _atm_share(prints: Sequence[Print]) -> float:
    """Premium share within ±ATM_BAND of spot. Needs underlying_px on prints."""
    total = 0.0
    atm = 0.0
    for p in prints:
        prem = p.premium
        total += prem
        u = p.underlying_px
        if u and u > 0 and abs(p.strike - u) / u <= ATM_BAND:
            atm += prem
    return atm / total if total > 0 else 0.0


def _iv_weighted_side(prints: Sequence[Print]) -> float:
    """IV-and-premium weighted net direction in [-1, 1].

    Distinguishes directional bets (low IV) from vol/hedge trades (high IV):
    signed direction weighted by iv*premium. Near 0 = balanced or non-IV data.
    """
    num = 0.0
    den = 0.0
    for p in prints:
        if p.iv is None:
            continue
        w = float(p.iv) * p.premium
        num += p.signed_sign * w
        den += w
    return num / den if den > 0 else 0.0
