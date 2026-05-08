"""Tail logs/spy_signals.jsonl and yield new SPY options signal rows.

The SPY options bot writes one JSON object per line to spy_signals.jsonl
*just before* it dispatches a signal to Telegram. The Trading Manager tails
that file, evaluates each row through the options-specific framework, and
writes a verdict to logs/manager_decisions.jsonl. The SPY options bot then
checks the verdict and aborts dispatch on REJECT.

Schema written by the SPY bot (see shree/spy_options/manager.py
_dispatch_signals → _emit_for_manager_review):

  {
    "ts": ISO timestamp (UTC),
    "kind": "spy_signal",
    "signal_id": "<dedup_key>:<expiry>:<ts>",   # unique join key
    "signal_type": "PC_RATIO_EXTREME" | "ORB_BREAKOUT" | ...,
    "strike": 690.0,
    "right": "C" | "P" | "BOTH",
    "expiry": "MAY26",
    "expiry_date": "20260515",
    "dte": 14,
    "confidence": 0.84,
    "confidence_tier": "MEDIUM" | "HIGH" | "EXTREME",
    "spy_price": 717.78,
    "vix": 18.2,
    "iv_rank": 27,
    "regime": "RANGE_BOUND" | "TREND_UP" | "TREND_DOWN" | ...,
    "delta": 0.805, "gamma": 0.001, "theta": -0.05, "vega": 0.10,
    "impl_vol": 0.18,
    "bid": 33.65, "ask": 33.90,
    "spread_pct": 0.74,
    "volume": 37,
    "open_interest": 2901,
    "sentiment_label": "BULLISH" | "BEARISH" | "NEUTRAL",
    "sentiment_score": -29,
    "reasoning": ["..."],
    "suggested_trade": "BUY CALL ..."
  }
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Generator, List, Optional


@dataclass
class SpySignal:
    raw: dict
    ts: str
    signal_id: str
    signal_type: str          # e.g. "PC_RATIO_EXTREME", "ORB_BREAKOUT"
    strike: float
    right: str                # "C" | "P" | "BOTH"
    expiry: str               # "MAY26"
    expiry_date: str          # "20260515"
    dte: int
    confidence: float         # 0.0–1.0
    confidence_tier: str      # MEDIUM / HIGH / EXTREME
    spy_price: float
    vix: float
    iv_rank: float
    regime: str
    delta: float
    gamma: float
    theta: float
    vega: float
    impl_vol: float
    bid: float
    ask: float
    spread_pct: float
    volume: int
    open_interest: int
    sentiment_label: str
    sentiment_score: float
    reasoning: List[str]
    suggested_trade: str

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        if self.ask > 0:
            return self.ask
        return 0.0

    @property
    def estimated_risk_per_contract_usd(self) -> float:
        """Heuristic: a 50% premium loss is the typical stop-out for a long
        option swing trade. SPY contracts are 100 shares.

        For straddles (right == 'BOTH'), assume both legs are at risk.
        """
        m = self.mid
        if m <= 0:
            return 0.0
        legs = 2 if self.right == "BOTH" else 1
        return m * 0.50 * 100.0 * legs

    @property
    def is_call(self) -> bool:
        return self.right == "C"

    @property
    def is_put(self) -> bool:
        return self.right == "P"


def _parse_spy_signal(d: dict) -> Optional[SpySignal]:
    if d.get("kind") != "spy_signal":
        return None
    if not d.get("signal_id"):
        return None
    try:
        return SpySignal(
            raw=d,
            ts=str(d.get("ts", "")),
            signal_id=str(d["signal_id"]),
            signal_type=str(d.get("signal_type", "")),
            strike=float(d.get("strike") or 0.0),
            right=str(d.get("right", "")),
            expiry=str(d.get("expiry", "")),
            expiry_date=str(d.get("expiry_date", "") or ""),
            dte=int(d.get("dte") or 0),
            confidence=float(d.get("confidence") or 0.0),
            confidence_tier=str(d.get("confidence_tier", "MEDIUM")),
            spy_price=float(d.get("spy_price") or 0.0),
            vix=float(d.get("vix") or 0.0),
            iv_rank=float(d.get("iv_rank") or 0.0),
            regime=str(d.get("regime", "")),
            delta=float(d.get("delta") or 0.0),
            gamma=float(d.get("gamma") or 0.0),
            theta=float(d.get("theta") or 0.0),
            vega=float(d.get("vega") or 0.0),
            impl_vol=float(d.get("impl_vol") or 0.0),
            bid=float(d.get("bid") or 0.0),
            ask=float(d.get("ask") or 0.0),
            spread_pct=float(d.get("spread_pct") or 0.0),
            volume=int(d.get("volume") or 0),
            open_interest=int(d.get("open_interest") or 0),
            sentiment_label=str(d.get("sentiment_label", "NEUTRAL")),
            sentiment_score=float(d.get("sentiment_score") or 0.0),
            reasoning=list(d.get("reasoning") or []),
            suggested_trade=str(d.get("suggested_trade", "")),
        )
    except (TypeError, ValueError):
        return None


class SpySignalTailer:
    """Polling tailer for spy_signals.jsonl, parallel to SignalTailer.

    Same semantics: opens at EOF on first read, follows append-only writes,
    handles file rotation by re-opening on inode change.
    """

    def __init__(self, path: str, start_at_end: bool = True):
        self.path = path
        self._fh = None
        self._inode: Optional[int] = None
        self._start_at_end = start_at_end

    def _open(self) -> bool:
        if not os.path.exists(self.path):
            return False
        st = os.stat(self.path)
        new_inode = st.st_ino
        if self._fh is not None and self._inode == new_inode:
            return True
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
        self._fh = open(self.path, "r")
        self._inode = new_inode
        if self._start_at_end:
            self._fh.seek(0, os.SEEK_END)
            self._start_at_end = False
        return True

    def poll(self) -> Generator[SpySignal, None, None]:
        if not self._open():
            return
        while True:
            line = self._fh.readline()
            if not line:
                try:
                    st = os.stat(self.path)
                    if st.st_ino != self._inode:
                        self._fh.close()
                        self._fh = None
                        if self._open():
                            continue
                except FileNotFoundError:
                    pass
                break
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            sig = _parse_spy_signal(d)
            if sig is not None:
                yield sig

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None
