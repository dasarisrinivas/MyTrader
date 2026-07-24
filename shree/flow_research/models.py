"""Dataclasses for the flow research layer.

`Print` mirrors one row of `spy_flow_prints` (raw OPRA tape).
`Snapshot` mirrors one row of `shadow_flow` (computed measurements).

Both are plain data. No behavior that touches trading.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# Aggressor classification values.
BUY = "BUY"
SELL = "SELL"
MID = "MID"
UNKNOWN = "UNKNOWN"

# aggressor_src provenance (honesty tag — validation weights/filters on this).
SRC_QUOTE = "QUOTE"   # classified against synchronized NBBO (trustworthy)
SRC_TICK = "TICK"     # tick-rule fallback vs prior print (weak)
SRC_ISO = "ISO_COND"  # inferred from ISO/sweep condition code only (weak)
SRC_NONE = "NONE"     # no basis — do NOT use in net measures


@dataclass
class Print:
    """One executed option print. Raw tape; immutable in spirit."""
    ts_utc: str
    ts_et: str
    session_date: str
    root: str            # 'SPY' | 'SPX'
    expiry: str          # YYYY-MM-DD
    strike: float
    right: str           # 'C' | 'P'
    trade_px: float
    size: int
    exchange: Optional[str] = None
    condition_codes: List[str] = field(default_factory=list)
    underlying_px: Optional[float] = None
    dte: Optional[int] = None
    # NBBO at print (for classification)
    bid: Optional[float] = None
    ask: Optional[float] = None
    # derived classification (filled by classify.py)
    aggressor: str = UNKNOWN
    aggressor_src: str = SRC_NONE
    is_sweep: bool = False
    is_block: bool = False
    oc_estimate: str = "UNKNOWN"   # 'OPEN' | 'CLOSE' | 'UNKNOWN'
    # greeks at print
    delta: Optional[float] = None
    gamma: Optional[float] = None
    iv: Optional[float] = None
    greeks_src: Optional[str] = None
    # provenance
    data_source: Optional[str] = None

    @property
    def premium(self) -> float:
        """Dollar premium of the print (price * size * 100)."""
        return float(self.trade_px) * int(self.size) * 100.0

    @property
    def is_call(self) -> bool:
        return self.right.upper() == "C"

    @property
    def signed_sign(self) -> int:
        """+1 if buyer-initiated, -1 if seller-initiated, 0 otherwise."""
        if self.aggressor == BUY:
            return 1
        if self.aggressor == SELL:
            return -1
        return 0

    def to_row(self) -> Dict:
        d = asdict(self)
        d.pop("condition_codes", None)
        d["condition_codes"] = ",".join(self.condition_codes or [])
        d["is_sweep"] = 1 if self.is_sweep else 0
        d["is_block"] = 1 if self.is_block else 0
        d["premium"] = self.premium
        return d


@dataclass
class Snapshot:
    """One computed flow snapshot joined to a context (signal/rejected/interval)."""
    snapshot_kind: str        # 'SIGNAL' | 'REJECTED' | 'INTERVAL'
    session_date: str
    ts_et: str
    window_s: int
    signal_id: Optional[int] = None
    # Tier A — orthogonal measures
    net_call_prem: float = 0.0
    net_put_prem: float = 0.0
    pc_prem_imbalance: float = 0.0
    dw_flow: float = 0.0
    sweep_intensity: float = 0.0
    block_prem: float = 0.0
    oc_open_ratio: float = 0.0
    # Tier B — context
    expiry_concentration: float = 0.0
    strike_repetition: float = 0.0
    atm_vs_wing: float = 0.0
    iv_weighted_side: float = 0.0
    # transparency
    n_prints: int = 0
    n_prints_used: int = 0

    def to_row(self) -> Dict:
        return asdict(self)
