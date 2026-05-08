"""Tail logs/decisions.jsonl and yield new SIGNAL rows in real time.

The bot writes one JSON object per line to decisions.jsonl on every cycle.
We care only about rows where outcome == "SIGNAL" (the actionable ones).

This implementation polls (no inotify dep) — small overhead since we already
poll on a 1s loop.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Generator, Optional


@dataclass
class Signal:
    raw: dict
    ts: str
    action: str           # BUY / SELL
    close: float
    stop_loss: float
    take_profit: float
    reason: str
    strategy: str
    adx: float
    rsi: float
    atr: float
    session_date: str

    @property
    def risk_points(self) -> float:
        return abs(self.close - self.stop_loss)

    @property
    def reward_points(self) -> float:
        return abs(self.take_profit - self.close)

    @property
    def rr(self) -> float:
        if self.risk_points <= 0:
            return 0.0
        return self.reward_points / self.risk_points

    @property
    def signal_type(self) -> str:
        # Reason field looks like "EMA21_PB_LONG | ADX=23 | ..." — first token is the type
        return self.reason.split(" ")[0] if self.reason else ""


def _parse_signal(d: dict) -> Optional[Signal]:
    if d.get("outcome") != "SIGNAL":
        return None
    if d.get("action") not in ("BUY", "SELL"):
        return None
    try:
        return Signal(
            raw=d,
            ts=str(d.get("ts", "")),
            action=str(d["action"]),
            close=float(d.get("close") or 0.0),
            stop_loss=float(d.get("stop_loss") or 0.0),
            take_profit=float(d.get("take_profit") or 0.0),
            reason=str(d.get("reason", "")),
            strategy=str(d.get("strategy", "")),
            adx=float(d.get("adx") or 0.0),
            rsi=float(d.get("rsi") or 0.0),
            atr=float(d.get("atr") or 0.0),
            session_date=str(d.get("session_date", "")),
        )
    except (TypeError, ValueError):
        return None


class SignalTailer:
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
        # (Re)open
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
        self._fh = open(self.path, "r")
        self._inode = new_inode
        if self._start_at_end:
            self._fh.seek(0, os.SEEK_END)
            self._start_at_end = False  # only on first open
        return True

    def poll(self) -> Generator[Signal, None, None]:
        """Yield any new Signal rows since last poll."""
        if not self._open():
            return
        while True:
            line = self._fh.readline()
            if not line:
                # Check for log rotation
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
            sig = _parse_signal(d)
            if sig is not None:
                yield sig

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None
