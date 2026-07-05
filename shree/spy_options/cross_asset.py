"""Cross-asset confirmation for SPY options — real-time QQQ + IWM from IB.

A SPY move that the rest of the tape refuses to confirm is a fade, not a
follow.  Two decades of intraday SPY behaviour in three rules:

  1. QQQ leads.  Tech is the beta engine — a SPY breakout with QQQ lagging
     (or diverging) fails far more often than one QQQ leads or matches.
  2. IWM confirms risk appetite.  Small caps joining = broad risk-on;
     small caps bleeding while SPY grinds up = rally on narrow legs.
  3. Non-confirmation at session extremes is the strongest tell: SPY prints
     a new session high while QQQ does NOT → bearish divergence; SPY prints
     a new session low while QQQ holds → bullish divergence.

This module replaces the 10-minute-delayed yfinance sector proxy with live
5-min bars from the SAME IB gateway connection (one extra historical request
per symbol per poll — well inside pacing limits at a 60s cadence).

Outputs (merged into ExternalContext by the manager each poll):
  * qqq_rs / iwm_rs            — intraday relative strength vs SPY (pct pts)
  * qqq_trend                  — UP / DOWN / FLAT (EMA9 vs EMA21 + VWAP side)
  * cross_asset_divergence     — BEARISH_NONCONFIRM / BULLISH_NONCONFIRM / NONE
  * cross_asset_bias           — RISK_ON / RISK_OFF / MIXED / NEUTRAL

Consumption:
  * DynamicConfidence block 21 — alignment boosts / divergence penalties
  * Executor hard veto — calls blocked on BEARISH_NONCONFIRM, puts on
    BULLISH_NONCONFIRM (config: execution.require_cross_asset_confirm)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# State dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CrossAssetState:
    """Point-in-time cross-asset confirmation snapshot."""

    available: bool = False

    # Intraday change vs session open (%)
    spy_change_pct: float = 0.0
    qqq_change_pct: float = 0.0
    iwm_change_pct: float = 0.0

    # Relative strength vs SPY (percentage points; positive = leading SPY)
    qqq_rs: float = 0.0
    iwm_rs: float = 0.0

    # QQQ trend structure (the lead horse)
    qqq_trend: str = "FLAT"            # UP / DOWN / FLAT
    qqq_above_vwap: bool = False

    # Session-extreme non-confirmation
    cross_asset_divergence: str = "NONE"
    # BEARISH_NONCONFIRM — SPY new session high, QQQ did not confirm
    # BULLISH_NONCONFIRM — SPY new session low,  QQQ did not confirm

    # Composite risk appetite
    cross_asset_bias: str = "NEUTRAL"  # RISK_ON / RISK_OFF / MIXED / NEUTRAL


# ─────────────────────────────────────────────────────────────────────────────
# Pure computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ema_last(closes: List[float], period: int) -> float:
    if not closes:
        return 0.0
    if len(closes) < period:
        return closes[-1]
    k = 2.0 / (period + 1)
    e = sum(closes[:period]) / period
    for v in closes[period:]:
        e = v * k + e * (1 - k)
    return e


def _vwap(bars: List[Dict]) -> float:
    pv = vol = 0.0
    for b in bars:
        typ = (b["high"] + b["low"] + b["close"]) / 3.0
        v = max(float(b.get("volume", 1)), 1.0)
        pv += typ * v
        vol += v
    return pv / vol if vol > 0 else 0.0


def _trend(bars: List[Dict]) -> str:
    """UP / DOWN / FLAT from EMA9 vs EMA21 + VWAP side of last close."""
    closes = [b["close"] for b in bars]
    if len(closes) < 21:
        return "FLAT"
    e9, e21 = _ema_last(closes, 9), _ema_last(closes, 21)
    vwap = _vwap(bars)
    last = closes[-1]
    if e9 > e21 and last > vwap:
        return "UP"
    if e9 < e21 and last < vwap:
        return "DOWN"
    return "FLAT"


def compute_cross_asset_state(
    spy_bars: List[Dict],
    qqq_bars: List[Dict],
    iwm_bars: List[Dict],
    divergence_lookback: int = 3,
    divergence_tolerance_pct: float = 0.05,
) -> CrossAssetState:
    """Pure computation — testable without IB.

    Args:
        spy_bars / qqq_bars / iwm_bars: today's 5-min OHLCV bars (newest last).
        divergence_lookback: SPY session extreme must be within the last N bars
            for a non-confirmation to count as ACTIVE (stale divergences from
            hours ago are noise).
        divergence_tolerance_pct: QQQ counts as confirming if it came within
            this % of its own session extreme.
    """
    st = CrossAssetState()
    if len(spy_bars) < 21 or len(qqq_bars) < 21:
        return st
    st.available = True

    spy_open = spy_bars[0]["open"]
    qqq_open = qqq_bars[0]["open"]
    spy_last = spy_bars[-1]["close"]
    qqq_last = qqq_bars[-1]["close"]

    st.spy_change_pct = (spy_last - spy_open) / spy_open * 100.0
    st.qqq_change_pct = (qqq_last - qqq_open) / qqq_open * 100.0
    st.qqq_rs = round(st.qqq_change_pct - st.spy_change_pct, 3)

    if len(iwm_bars) >= 2:
        iwm_open = iwm_bars[0]["open"]
        iwm_last = iwm_bars[-1]["close"]
        st.iwm_change_pct = (iwm_last - iwm_open) / iwm_open * 100.0
        st.iwm_rs = round(st.iwm_change_pct - st.spy_change_pct, 3)

    st.qqq_trend = _trend(qqq_bars)
    st.qqq_above_vwap = qqq_last > _vwap(qqq_bars)

    # ── Session-extreme non-confirmation ──────────────────────────────────
    spy_highs = [b["high"] for b in spy_bars]
    spy_lows  = [b["low"]  for b in spy_bars]
    qqq_highs = [b["high"] for b in qqq_bars]
    qqq_lows  = [b["low"]  for b in qqq_bars]

    n = divergence_lookback
    tol = divergence_tolerance_pct / 100.0

    spy_session_high = max(spy_highs)
    spy_session_low  = min(spy_lows)
    qqq_session_high = max(qqq_highs)
    qqq_session_low  = min(qqq_lows)

    # SPY made its session high within the last n bars…
    if max(spy_highs[-n:]) >= spy_session_high:
        # …did QQQ come within tolerance of its own high in the same window?
        qqq_recent_high = max(qqq_highs[-n:])
        if qqq_recent_high < qqq_session_high * (1.0 - tol):
            st.cross_asset_divergence = "BEARISH_NONCONFIRM"

    # SPY made its session low within the last n bars…
    if min(spy_lows[-n:]) <= spy_session_low:
        qqq_recent_low = min(qqq_lows[-n:])
        if qqq_recent_low > qqq_session_low * (1.0 + tol):
            # bullish overrides bearish only if both somehow fire (fresh low wins)
            st.cross_asset_divergence = "BULLISH_NONCONFIRM"

    # ── Composite risk-appetite bias ──────────────────────────────────────
    qqq_up = st.qqq_rs > 0.10 or (st.qqq_trend == "UP" and st.qqq_rs > -0.05)
    qqq_dn = st.qqq_rs < -0.10 or (st.qqq_trend == "DOWN" and st.qqq_rs < 0.05)
    iwm_up = st.iwm_rs > 0.10
    iwm_dn = st.iwm_rs < -0.10

    if qqq_up and (iwm_up or abs(st.iwm_rs) <= 0.10):
        st.cross_asset_bias = "RISK_ON"
    elif qqq_dn and (iwm_dn or abs(st.iwm_rs) <= 0.10):
        st.cross_asset_bias = "RISK_OFF"
    elif (qqq_up and iwm_dn) or (qqq_dn and iwm_up):
        st.cross_asset_bias = "MIXED"

    return st


# ─────────────────────────────────────────────────────────────────────────────
# Feed (thin IB wrapper around the pure computation)
# ─────────────────────────────────────────────────────────────────────────────

class CrossAssetFeed:
    """Fetches live QQQ/IWM 5-min bars from the shared ib_insync connection.

    One reqHistoricalData per symbol per poll (60s cadence) — well inside IB
    pacing limits.  Contracts are qualified once at start().
    """

    def __init__(self, ib) -> None:
        self._ib = ib
        self._contracts: Dict[str, Any] = {}

    async def start(self) -> None:
        from ib_insync import Stock
        for sym in ("QQQ", "IWM"):
            try:
                qualified = await self._ib.qualifyContractsAsync(
                    Stock(sym, "SMART", "USD")
                )
                if qualified:
                    self._contracts[sym] = qualified[0]
                    logger.info("CrossAsset: {} qualified (conId={})", sym, qualified[0].conId)
            except Exception as exc:
                logger.warning("CrossAsset: could not qualify {}: {}", sym, exc)

    async def _bars_5m(self, sym: str) -> List[Dict]:
        contract = self._contracts.get(sym)
        if contract is None:
            return []
        try:
            bars = await self._ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr="1 D",
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=True,
                keepUpToDate=False,
                timeout=15,
            )
            return [
                {
                    "open": float(b.open), "high": float(b.high),
                    "low": float(b.low), "close": float(b.close),
                    "volume": int(b.volume),
                }
                for b in (bars or [])
            ]
        except Exception as exc:
            logger.warning("CrossAsset: {} bars fetch failed: {}", sym, exc)
            return []

    async def snapshot(self, spy_bars: List[Dict]) -> CrossAssetState:
        """Fetch fresh QQQ/IWM bars and compute the confirmation state."""
        if not self._contracts:
            return CrossAssetState()
        qqq = await self._bars_5m("QQQ")
        iwm = await self._bars_5m("IWM")
        return compute_cross_asset_state(spy_bars, qqq, iwm)
