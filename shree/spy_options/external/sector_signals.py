"""Sector leadership, overnight context, relative strength, ES premium, USDJPY proxy.

All data via yfinance (free, no API key).

Modules:
  - Sector leadership: XLK, XLF, SMH, IWM, QQQ, XLE, XLI  vs day-open
  - Relative strength: QQQ and IWM vs SPY on intraday % change
  - Overnight/Globex context: prior session close, pre-market H/L, gap %, range
  - ES premium proxy: ES=F vs SPY (fair value approximation)
  - USD/JPY proxy: JPY=X  (risk-on/off indicator)

TTL: 10 minutes for intraday data, 24 hours for prior-close / USDJPY.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_SECTOR_LEADERS = {
    "XLK": "Tech",
    "XLF": "Financials",
    "SMH": "Semis",
    "IWM": "SmallCap",
    "QQQ": "Nasdaq",
    "XLE": "Energy",
    "XLI": "Industrials",
}

_INTRADAY_TTL_S = 600   # 10 minutes
_DAILY_TTL_S = 86_400   # 24 hours


@dataclass
class SectorState:
    # Sector leadership
    sector_bull_count: int = 0       # sectors above their day-open
    sector_bear_count: int = 0       # sectors below their day-open
    sector_label: str = "NEUTRAL"    # BULL_SWEEP / BEAR_SWEEP / MIXED / NEUTRAL

    # Relative strength vs SPY
    qqq_vs_spy_pct: float = 0.0      # QQQ intraday % – SPY intraday %
    iwm_vs_spy_pct: float = 0.0      # IWM intraday % – SPY intraday %

    # Per-ticker intraday change vs open (% rounded to 2dp)
    ticker_vs_open: dict = field(default_factory=dict)   # {"XLK": +0.42, ...}

    # Overnight / Globex context
    prior_spy_close: Optional[float] = None
    gap_pct: float = 0.0             # today_open vs prior_close in %
    spy_day_open: Optional[float] = None
    spy_overnight_high: Optional[float] = None
    spy_overnight_low: Optional[float] = None
    overnight_range_pct: float = 0.0
    above_overnight_high: bool = False
    below_overnight_low: bool = False

    # ES premium proxy
    es_premium: float = 0.0          # ES=F price – SPY price (approximate)

    # USD/JPY proxy (risk-on = JPY weaker = higher USDJPY)
    usdjpy: Optional[float] = None
    usdjpy_trend: str = "NEUTRAL"    # RISK_ON / RISK_OFF / NEUTRAL

    fetched_at_intraday: float = 0.0
    fetched_at_daily: float = 0.0
    available: bool = False


class SectorSignals:
    """Sector leadership and intraday context aggregator."""

    def __init__(self) -> None:
        self._state = SectorState()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> SectorState:
        return self._state

    def _intraday_stale(self) -> bool:
        return (time.monotonic() - self._state.fetched_at_intraday) > _INTRADAY_TTL_S

    def _daily_stale(self) -> bool:
        return (time.monotonic() - self._state.fetched_at_daily) > _DAILY_TTL_S

    async def refresh_if_stale(self) -> None:
        if not self._intraday_stale() and not self._daily_stale():
            return
        async with self._lock:
            if not self._intraday_stale() and not self._daily_stale():
                return
            loop = asyncio.get_event_loop()
            now = time.monotonic()
            try:
                if self._intraday_stale():
                    await loop.run_in_executor(None, self._fetch_intraday)
                    self._state.fetched_at_intraday = time.monotonic()
                if self._daily_stale():
                    await loop.run_in_executor(None, self._fetch_daily)
                    self._state.fetched_at_daily = time.monotonic()
                self._state.available = True
            except Exception as exc:
                logger.warning("SectorSignals fetch failed: %s", exc)

    # ------------------------------------------------------------------
    # Intraday fetch — sector ETFs + QQQ + IWM + SPY + ES=F
    # ------------------------------------------------------------------

    def _fetch_intraday(self) -> None:
        import yfinance as yf

        tickers = list(_SECTOR_LEADERS.keys()) + ["SPY", "ES=F"]
        try:
            data = yf.download(
                tickers=" ".join(tickers),
                period="1d",
                interval="5m",
                progress=False,
                auto_adjust=True,
                threads=True,
            )
        except Exception as exc:
            logger.warning("SectorSignals intraday download failed: %s", exc)
            return

        if data.empty:
            return

        def _get(field: str, ticker: str):
            try:
                if isinstance(data.columns, type(data.columns)) and hasattr(data.columns, "get_level_values"):
                    series = data[field][ticker].dropna()
                else:
                    series = data[field].dropna()
                return series
            except Exception:
                return None

        # SPY open + current for baseline
        spy_close_s = _get("Close", "SPY")
        spy_open_s = _get("Open", "SPY")
        if spy_close_s is None or spy_close_s.empty:
            return

        spy_last = float(spy_close_s.iloc[-1])
        spy_open = float(spy_open_s.iloc[0]) if spy_open_s is not None and not spy_open_s.empty else spy_last
        spy_intraday_pct = (spy_last / spy_open - 1) * 100 if spy_open > 0 else 0.0

        self._state.spy_day_open = spy_open

        # Current SPY price for overnight comparison
        spy_now = spy_last

        # Sector loop
        bull = 0
        bear = 0
        ticker_vs_open: dict = {}
        for ticker in _SECTOR_LEADERS:
            close_s = _get("Close", ticker)
            open_s = _get("Open", ticker)
            if close_s is None or close_s.empty:
                continue
            t_last = float(close_s.iloc[-1])
            t_open = float(open_s.iloc[0]) if open_s is not None and not open_s.empty else t_last
            pct = (t_last / t_open - 1) * 100 if t_open > 0 else 0.0
            ticker_vs_open[ticker] = round(pct, 2)
            if pct > 0:
                bull += 1
            elif pct < 0:
                bear += 1

        self._state.sector_bull_count = bull
        self._state.sector_bear_count = bear
        self._state.ticker_vs_open = ticker_vs_open
        self._state.sector_label = _sector_label(bull, bear)

        # Relative strength vs SPY
        for ticker, key in [("QQQ", "qqq_vs_spy_pct"), ("IWM", "iwm_vs_spy_pct")]:
            pct = ticker_vs_open.get(ticker)
            if pct is not None:
                setattr(self._state, key, round(pct - spy_intraday_pct, 3))

        # ES premium proxy
        try:
            es_close_s = _get("Close", "ES=F")
            if es_close_s is not None and not es_close_s.empty:
                es_last = float(es_close_s.iloc[-1])
                # ES is priced ~10× SPY; rough premium = (ES/10) − SPY
                self._state.es_premium = round(es_last / 10 - spy_last, 2)
        except Exception:
            pass

        # Overnight H/L: compare current price vs today's range
        try:
            spy_high_s = _get("High", "SPY")
            spy_low_s = _get("Low", "SPY")
            if spy_high_s is not None and spy_low_s is not None:
                day_high = float(spy_high_s.max())
                day_low = float(spy_low_s.min())
                self._state.spy_overnight_high = day_high
                self._state.spy_overnight_low = day_low
                self._state.above_overnight_high = spy_now >= day_high * 0.9995
                self._state.below_overnight_low = spy_now <= day_low * 1.0005
                rng = day_high - day_low
                self._state.overnight_range_pct = round(rng / day_low * 100, 3) if day_low > 0 else 0.0
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Daily fetch — prior close + USDJPY
    # ------------------------------------------------------------------

    def _fetch_daily(self) -> None:
        import yfinance as yf

        # Prior SPY close
        try:
            spy = yf.download("SPY", period="5d", interval="1d", progress=False, auto_adjust=True)
            if not spy.empty and len(spy) >= 2:
                prior_close = float(spy["Close"].iloc[-2])
                self._state.prior_spy_close = prior_close
                today_open = self._state.spy_day_open
                if today_open and prior_close > 0:
                    self._state.gap_pct = round((today_open / prior_close - 1) * 100, 3)
        except Exception as exc:
            logger.debug("SectorSignals: prior close fetch failed: %s", exc)

        # USDJPY proxy via JPY=X (USD per JPY → invert for USDJPY)
        try:
            jpy = yf.download("JPY=X", period="5d", interval="1d", progress=False, auto_adjust=True)
            if not jpy.empty:
                jpy_rate = float(jpy["Close"].iloc[-1])  # USD per 1 JPY
                usdjpy = 1.0 / jpy_rate if jpy_rate > 0 else None
                self._state.usdjpy = round(usdjpy, 3) if usdjpy else None

                # Compare to prior day
                if len(jpy) >= 2:
                    prior_jpy = float(jpy["Close"].iloc[-2])
                    prior_usdjpy = 1.0 / prior_jpy if prior_jpy > 0 else None
                    if usdjpy and prior_usdjpy:
                        chg = usdjpy - prior_usdjpy
                        if chg > 0.30:
                            self._state.usdjpy_trend = "RISK_ON"    # JPY weaker = risk-on
                        elif chg < -0.30:
                            self._state.usdjpy_trend = "RISK_OFF"   # JPY stronger = risk-off
                        else:
                            self._state.usdjpy_trend = "NEUTRAL"
        except Exception as exc:
            logger.debug("SectorSignals: USDJPY fetch failed: %s", exc)


def _sector_label(bull: int, bear: int) -> str:
    total = bull + bear
    if total == 0:
        return "NEUTRAL"
    ratio = bull / total
    if ratio >= 0.80:
        return "BULL_SWEEP"
    if ratio >= 0.60:
        return "BULL_LEANING"
    if ratio <= 0.20:
        return "BEAR_SWEEP"
    if ratio <= 0.40:
        return "BEAR_LEANING"
    return "MIXED"
