"""
Macro signals via yfinance — two refresh tiers:

Daily (once per day):
  - Base levels for 10Y yield, DXY, oil, gold
  - Long-term trend direction

Intraday (every 15 minutes during RTH):
  - Current levels for all tickers via fast_info or 1-min bars
  - Intraday trend = last price vs today's open
  - SPY breadth proxy: SPY vs VWAP from 5-min bars (from IB; here estimated)
  - VIX intraday level
  - Advance/Decline proxy: SPY 1-min breadth heuristic

Composite macro score (-1.0 = strong headwind, +1.0 = strong tailwind):
  DXY↑ + TNX↑ + VIX↑  = bearish for SPY (headwind)
  DXY↓ + TNX↓ + VIX↓  = bullish for SPY (tailwind)
  Strong breadth       = bullish boost

Macro alignment with signal:
  Used by DynamicConfidence to adjust per-signal confidence.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, Optional

import yfinance as yf
from loguru import logger

_TICKERS = {
    "tnx":  "^TNX",
    "dxy":  "DX-Y.NYB",
    "oil":  "CL=F",
    "gold": "GC=F",
    "vix":  "^VIX",
    "spy":  "SPY",
}

_DAILY_TTL_S   = 86_400   # once per day
_INTRADAY_TTL_S = 900      # 15 minutes


def _trend(series) -> str:
    if len(series) < 2:
        return "FLAT"
    last = float(series.iloc[-1])
    avg  = float(series.iloc[:-1].mean())
    pct  = (last - avg) / avg if avg != 0 else 0.0
    if pct > 0.005:
        return "RISING"
    if pct < -0.005:
        return "FALLING"
    return "FLAT"


def _intraday_trend(open_: float, last: float) -> str:
    if open_ <= 0:
        return "FLAT"
    pct = (last - open_) / open_
    if pct > 0.0015:
        return "RISING"
    if pct < -0.0015:
        return "FALLING"
    return "FLAT"


@dataclass
class MacroState:
    # Levels
    tnx:  Optional[float] = None
    dxy:  Optional[float] = None
    oil:  Optional[float] = None
    gold: Optional[float] = None
    vix:  Optional[float] = None

    # Daily trends
    tnx_trend:  str = "FLAT"
    dxy_trend:  str = "FLAT"
    oil_trend:  str = "FLAT"
    gold_trend: str = "FLAT"
    vix_trend:  str = "FLAT"

    # Intraday trends (vs open)
    tnx_intraday:  str = "FLAT"
    dxy_intraday:  str = "FLAT"
    vix_intraday:  str = "FLAT"
    spy_intraday:  str = "FLAT"

    # Breadth
    spy_vs_open_pct: float = 0.0       # SPY % change from day open
    vix_change_pct:  float = 0.0       # VIX % change from day open

    # Freshness
    fetched_date:     Optional[date] = None
    intraday_fetched_at: float = 0.0
    available: bool = False

    def is_daily_stale(self) -> bool:
        return self.fetched_date is None or self.fetched_date < date.today()

    def is_intraday_stale(self) -> bool:
        return (time.monotonic() - self.intraday_fetched_at) > _INTRADAY_TTL_S

    @property
    def spy_headwind(self) -> float:
        """
        Composite headwind score: -1.0 (strong headwind) to +1.0 (tailwind).
        Uses intraday trends when available, falls back to daily.
        """
        tnx_t = self.tnx_intraday if self.tnx_intraday != "FLAT" else self.tnx_trend
        dxy_t = self.dxy_intraday if self.dxy_intraday != "FLAT" else self.dxy_trend
        vix_t = self.vix_intraday if self.vix_intraday != "FLAT" else self.vix_trend

        score = 0.0
        # TNX (yields): rising = headwind for equities
        if tnx_t == "RISING":   score -= 0.35
        elif tnx_t == "FALLING": score += 0.25
        # DXY: rising dollar = headwind
        if dxy_t == "RISING":   score -= 0.25
        elif dxy_t == "FALLING": score += 0.20
        # VIX: rising fear = headwind
        if vix_t == "RISING":   score -= 0.25
        elif vix_t == "FALLING": score += 0.20
        # Oil: rising oil = mild stagflation headwind
        if self.oil_trend == "RISING":  score -= 0.08
        # SPY breadth
        if self.spy_vs_open_pct > 0.3:  score += 0.10
        elif self.spy_vs_open_pct < -0.3: score -= 0.10

        return max(-1.0, min(1.0, score))

    @property
    def macro_label(self) -> str:
        hw = self.spy_headwind
        if hw <= -0.5:  return "STRONG_HEADWIND"
        if hw <= -0.2:  return "HEADWIND"
        if hw >= 0.5:   return "STRONG_TAILWIND"
        if hw >= 0.2:   return "TAILWIND"
        return "NEUTRAL"


class MacroSignals:
    def __init__(self):
        self._state = MacroState()
        self._lock = asyncio.Lock()

    async def refresh_if_stale(self) -> None:
        """Refresh daily data if needed AND intraday data if needed."""
        tasks = []
        if self._state.is_daily_stale():
            tasks.append(self._fetch_daily())
        if self._state.is_intraday_stale():
            tasks.append(self._fetch_intraday())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ── Daily fetch ─────────────────────────────────────────────────────────

    async def _fetch_daily(self) -> None:
        async with self._lock:
            if not self._state.is_daily_stale():
                return
            loop = asyncio.get_event_loop()
            try:
                updated = await loop.run_in_executor(None, self._daily_sync)
                # Merge into current state (preserve intraday fields)
                s = self._state
                s.tnx         = updated.tnx
                s.dxy         = updated.dxy
                s.oil         = updated.oil
                s.gold        = updated.gold
                s.vix         = updated.vix
                s.tnx_trend   = updated.tnx_trend
                s.dxy_trend   = updated.dxy_trend
                s.oil_trend   = updated.oil_trend
                s.gold_trend  = updated.gold_trend
                s.vix_trend   = updated.vix_trend
                s.fetched_date= updated.fetched_date
                s.available   = True
            except Exception as exc:
                logger.warning(f"[MacroSignals] Daily fetch failed: {exc}")
                self._state.fetched_date = date.today()

    def _daily_sync(self) -> MacroState:
        result: Dict[str, Optional[float]] = {}
        trends: Dict[str, str] = {}
        for key, sym in _TICKERS.items():
            if key in ("spy",):
                continue
            try:
                hist = yf.Ticker(sym).history(period="10d")
                if hist.empty:
                    result[key] = None; trends[key] = "FLAT"
                    continue
                closes = hist["Close"].dropna()
                result[key] = float(closes.iloc[-1])
                trends[key] = _trend(closes.iloc[-6:])
            except Exception as exc:
                logger.debug(f"[MacroSignals] {sym} daily error: {exc}")
                result[key] = None; trends[key] = "FLAT"

        s = MacroState(
            tnx=result.get("tnx"), tnx_trend=trends.get("tnx", "FLAT"),
            dxy=result.get("dxy"), dxy_trend=trends.get("dxy", "FLAT"),
            oil=result.get("oil"), oil_trend=trends.get("oil", "FLAT"),
            gold=result.get("gold"), gold_trend=trends.get("gold", "FLAT"),
            vix=result.get("vix"), vix_trend=trends.get("vix", "FLAT"),
            fetched_date=date.today(), available=True,
        )
        logger.info(
            "[MacroSignals] TNX={:.2f}({}) DXY={:.2f}({}) VIX={:.1f}({}) headwind={}",
            s.tnx or 0, s.tnx_trend, s.dxy or 0, s.dxy_trend,
            s.vix or 0, s.vix_trend, s.macro_label,
        )
        return s

    # ── Intraday fetch (15-min) ──────────────────────────────────────────────

    async def _fetch_intraday(self) -> None:
        loop = asyncio.get_event_loop()
        try:
            updates = await loop.run_in_executor(None, self._intraday_sync)
            s = self._state
            s.tnx_intraday      = updates.get("tnx_intraday", "FLAT")
            s.dxy_intraday      = updates.get("dxy_intraday", "FLAT")
            s.vix_intraday      = updates.get("vix_intraday", "FLAT")
            s.spy_intraday      = updates.get("spy_intraday", "FLAT")
            s.spy_vs_open_pct   = updates.get("spy_vs_open_pct", 0.0)
            s.vix_change_pct    = updates.get("vix_change_pct", 0.0)
            # Update current levels too
            if updates.get("vix_last"):
                s.vix = updates["vix_last"]
            s.intraday_fetched_at = time.monotonic()
        except Exception as exc:
            logger.debug(f"[MacroSignals] Intraday fetch error: {exc}")
            self._state.intraday_fetched_at = time.monotonic()

    def _intraday_sync(self) -> dict:
        out: dict = {}
        intraday_map = {
            "tnx_intraday": "^TNX",
            "dxy_intraday": "DX-Y.NYB",
            "vix_intraday": "^VIX",
            "spy_intraday": "SPY",
        }
        for key, sym in intraday_map.items():
            try:
                # Use 1-day with 5-min interval for intraday
                hist = yf.Ticker(sym).history(period="1d", interval="5m")
                if hist.empty or len(hist) < 2:
                    out[key] = "FLAT"
                    continue
                open_price = float(hist["Open"].iloc[0])
                last_price = float(hist["Close"].iloc[-1])
                out[key] = _intraday_trend(open_price, last_price)

                # Store specific values
                if sym == "^VIX":
                    out["vix_last"] = last_price
                    out["vix_change_pct"] = round(
                        (last_price - open_price) / open_price * 100, 2
                    ) if open_price > 0 else 0.0
                if sym == "SPY":
                    out["spy_vs_open_pct"] = round(
                        (last_price - open_price) / open_price * 100, 2
                    ) if open_price > 0 else 0.0
            except Exception as exc:
                logger.debug(f"[MacroSignals] {sym} intraday error: {exc}")
                out[key] = "FLAT"
        return out

    @property
    def state(self) -> MacroState:
        return self._state
