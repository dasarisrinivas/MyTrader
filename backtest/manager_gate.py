"""Execution-faithful Trade Manager gate for the backtest.

Calls the REAL live gate `shree.trading_manager.rules.evaluate` (plus the real
`_maybe_update_posture` and `streaks_from_recent` helpers) so research and live use
the same approval logic. The ONLY adaptation vs live is that ManagerState is evolved
from the backtest's own fills using the BAR timestamp, because the live manager's
state-refresh hard-codes `datetime.now()` + reads `orders.db` (which is why it was
never backtestable). `rules.evaluate` itself is pure, so the gate decision is faithful.

Enable in the backtest with env BT_WITH_MANAGER=1 (optional BT_MIN_RR, BT_ACCT_EQUITY).
Default OFF — no effect on existing backtests or any live code path.
"""
from __future__ import annotations
import dataclasses
import logging
from collections import Counter

_log = logging.getLogger("bt_manager_gate")
_log.addHandler(logging.NullHandler())


def _bucket(reason: str) -> str:
    r = (reason or "").lower()
    if "r:r" in r or "rr" in r and "required" in r:
        return "R:R<floor"
    if "risk" in r and "cap" in r:
        return "risk>cap"
    if "framework" in r or "regime" in r or "q1" in r or "q2" in r:
        return "regime/framework-fit"
    if "consec" in r or "pause" in r or "halt" in r or "sit_out" in r or "locked" in r:
        return "streak/posture"
    if "trade count" in r or "max_trades" in r:
        return "max-trades/day"
    return "other"


class BacktestManagerGate:
    """One per backtest run. evaluate() before opening; record_outcome() on close."""

    def __init__(self, min_rr: float | None = None, account_equity: float | None = None):
        from shree.trading_manager.config import ManagerConfig
        from shree.trading_manager.state import ManagerState
        kw = {}
        if min_rr is not None:
            kw["min_rr_ratio"] = float(min_rr)
            kw["soft_pause_min_rr"] = max(float(min_rr), 1.3)
        if account_equity is not None:
            kw["account_equity"] = float(account_equity)
        self.cfg = dataclasses.replace(ManagerConfig(), **kw) if kw else ManagerConfig()
        self.state = ManagerState()
        self.state.posture = "NORMAL"
        self._approved: list[tuple[str, float]] = []   # (iso_ts, pnl) of CLOSED approved trades
        self._cur_day = None
        self.stats = Counter()                          # decisions + reject buckets

    def _roll(self, day: str) -> None:
        if day != self._cur_day:
            self._cur_day = day
            self.state.realized_pnl_today = 0.0
            self.state.trades_today = 0

    def _refresh_state(self, day: str) -> list:
        from shree.trading_manager.executions_reader import TradeOutcome, streaks_from_recent
        rec = [TradeOutcome(order_id=str(i), timestamp=t, net_pnl=p, gross_pnl=p, commission=0.0)
               for i, (t, p) in enumerate(reversed(self._approved[-20:]))]
        if rec:
            w, l = streaks_from_recent(rec, since_iso=day + "T00:00:00")
        else:
            w, l = 0, 0
        self.state.consec_wins, self.state.consec_losses = w, l
        self.state.last_n_outcomes = ["WIN" if p > 0 else "LOSS"
                                      for _, p in reversed(self._approved[-5:])]
        return rec

    def evaluate(self, action, ts, close, stop_loss, take_profit, adx, rsi, atr, signal_type):
        """Return (approved: bool, decision). Mirrors the live manager veto."""
        from shree.trading_manager.signal_watcher import Signal
        from shree.trading_manager import rules, manager as M
        day = ts.strftime("%Y-%m-%d")
        self._roll(day)
        rec = self._refresh_state(day)
        M._maybe_update_posture(self.state, self.cfg, _log)
        sig = Signal(raw={}, ts=ts.isoformat(), action=action, close=float(close),
                     stop_loss=float(stop_loss), take_profit=float(take_profit),
                     reason=signal_type or "", strategy="es_fifteen_min",
                     adx=float(adx), rsi=float(rsi), atr=float(atr), session_date=day)
        d = rules.evaluate(sig, self.state, rec, self.cfg)
        approved = d.decision in ("APPROVE", "MODIFY")
        self.stats[d.decision] += 1
        if approved:
            self.state.trades_today += 1            # count opened trade for Q3 overtrading
        else:
            self.stats["reject:" + _bucket(d.reasoning)] += 1
        return approved, d

    def record_outcome(self, pnl: float, ts) -> None:
        """Call when an approved trade CLOSES, to evolve streak/posture/PnL state."""
        self._roll(ts.strftime("%Y-%m-%d"))
        self._approved.append((ts.isoformat(), float(pnl)))
        self.state.realized_pnl_today += float(pnl)

    def summary(self) -> dict:
        return dict(self.stats)
