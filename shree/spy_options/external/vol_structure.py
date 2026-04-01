"""Volatility term structure signals via free yfinance data.

Sources:
  ^VIX   — 30-day implied vol
  ^VXV   — 93-day implied vol  (contango = VIX < VXV)
  ^VVIX  — vol-of-vol (fear of fear indicator)

Derived signals:
  vix_vxv_ratio     VIX / VXV  (<1.0 = contango, >1.0 = backwardation)
  vol_structure     STEEP_CONTANGO / CONTANGO / FLAT / BACKWARDATION / STEEP_BACKWARDATION
  vvix              latest VVIX value
  vvix_elevated     True when VVIX > 115 (tail risk concern)

TTL: 15 minutes (these move slowly intraday).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_TTL_S = 900  # 15 minutes


@dataclass
class VolStructureState:
    vix: Optional[float] = None
    vxv: Optional[float] = None
    vvix: Optional[float] = None
    vix_vxv_ratio: Optional[float] = None
    vol_structure: str = "FLAT"         # STEEP_CONTANGO / CONTANGO / FLAT / BACKWARDATION / STEEP_BACKWARDATION
    vvix_elevated: bool = False         # VVIX > 115
    fetched_at: float = 0.0
    available: bool = False


class VolStructure:
    """VIX term structure and vol-of-vol monitor."""

    def __init__(self) -> None:
        self._state = VolStructureState()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> VolStructureState:
        return self._state

    def is_stale(self) -> bool:
        return (time.monotonic() - self._state.fetched_at) > _TTL_S

    async def refresh_if_stale(self) -> None:
        if not self.is_stale():
            return
        async with self._lock:
            if not self.is_stale():
                return
            loop = asyncio.get_event_loop()
            try:
                state = await loop.run_in_executor(None, self._fetch_sync)
                state.fetched_at = time.monotonic()
                self._state = state
            except Exception as exc:
                logger.warning("VolStructure fetch failed: %s", exc)

    def _fetch_sync(self) -> VolStructureState:
        import yfinance as yf

        state = VolStructureState()
        tickers = ["^VIX", "^VXV", "^VVIX"]
        try:
            data = yf.download(
                " ".join(tickers),
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=False,
            )
        except Exception as exc:
            logger.warning("VolStructure: yfinance download failed: %s", exc)
            return state

        if data.empty:
            return state

        def _last(ticker: str) -> Optional[float]:
            try:
                if hasattr(data.columns, "get_level_values"):
                    s = data["Close"][ticker].dropna()
                else:
                    s = data["Close"].dropna()
                return float(s.iloc[-1]) if not s.empty else None
            except Exception:
                return None

        state.vix = _last("^VIX")
        state.vxv = _last("^VXV")
        state.vvix = _last("^VVIX")

        # VIX/VXV ratio
        if state.vix is not None and state.vxv is not None and state.vxv > 0:
            ratio = state.vix / state.vxv
            state.vix_vxv_ratio = round(ratio, 4)
            state.vol_structure = _structure_label(ratio)
            state.available = True

        # VVIX elevated flag
        if state.vvix is not None:
            state.vvix_elevated = state.vvix > 115.0

        return state


def _structure_label(ratio: float) -> str:
    """Classify term structure from VIX/VXV ratio."""
    if ratio < 0.85:
        return "STEEP_CONTANGO"
    if ratio < 0.95:
        return "CONTANGO"
    if ratio > 1.10:
        return "STEEP_BACKWARDATION"
    if ratio > 1.00:
        return "BACKWARDATION"
    return "FLAT"
