"""TREND_CONTINUATION signal generator.

The missing primitive — identified as root cause #4. The legacy engine has
ORB (regime-start) and PC_RATIO (contrarian) but nothing that enters *with*
an already-confirmed trend on a pullback.

Setup:
  1. Regime is TREND_UP or TREND_DOWN (per RegimeV2Detector)
  2. Price pulls back to touch VWAP ± pullback_band_pct
  3. The pullback bar is rejected (wick ≥ wick_ratio OR engulfing bar)
  4. Entry triggers on break of the rejection bar's extreme

The generator is deliberately simple — one continuation candidate per
evaluation cycle in each direction. The engine layer applies throttling,
entry-gate, and strike-selection downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from . import structure as _s
from .config import ContinuationConfig
from .regime import RegimeV2Context, TREND_DOWN, TREND_UP

ET = ZoneInfo("America/New_York")


@dataclass
class ContinuationCandidate:
    """A detected continuation setup — not a SpySignal yet.

    The engine converts this into a proper SpySignal (or enriches an existing
    flow signal) before dispatch.
    """

    direction: str                 # "C" or "P"
    trigger_price: float           # break of rejection bar extreme
    stop_price: float              # structural stop (other side of rejection bar)
    confidence: float              # 0.0–1.0
    reasons: List[str]
    rejection_bar_ts: datetime
    vwap: float
    spy_price: float


def _parse_et(hhmm: str) -> time:
    h, m = hhmm.split(":")
    return time(int(h), int(m))


def _now_et(now: Optional[datetime] = None) -> datetime:
    if now is None:
        return datetime.now(ET)
    if now.tzinfo is None:
        from datetime import timezone

        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(ET)


class ContinuationDetector:
    """Stateless detector that inspects the latest bars and regime context."""

    def __init__(self, cfg: Optional[ContinuationConfig] = None) -> None:
        self._cfg = cfg or ContinuationConfig()

    def in_window(self, now: Optional[datetime] = None) -> bool:
        t = _now_et(now).time()
        return _parse_et(self._cfg.min_time_et) <= t < _parse_et(self._cfg.max_time_et)

    def detect(
        self,
        bars: List[Dict],
        regime: RegimeV2Context,
        spy_price: float,
        now: Optional[datetime] = None,
    ) -> Optional[ContinuationCandidate]:
        cfg = self._cfg

        if regime.regime not in (TREND_UP, TREND_DOWN):
            return None
        if not self.in_window(now):
            return None
        if len(bars) < 3:
            return None

        vwap = regime.vwap

        # Compute the pullback anchor. On strong trend days price may spend
        # hours away from session VWAP; in that case a short EMA is the
        # relevant pullback line. "nearest" picks whichever of VWAP / EMA9
        # / EMA21 sits between price and the trend — the line the pullback
        # reaches first.
        closes = [b["close"] for b in bars]
        ema9_series = _s.ema(closes, 9)
        ema21_series = _s.ema(closes, 21)
        ema9 = ema9_series[-1] if ema9_series else vwap
        ema21 = ema21_series[-1] if ema21_series else vwap

        anchor_mode = cfg.pullback_anchor
        if anchor_mode == "vwap":
            anchor, anchor_name = vwap, "VWAP"
        elif anchor_mode == "ema9":
            anchor, anchor_name = ema9, "EMA9"
        elif anchor_mode == "ema21":
            anchor, anchor_name = ema21, "EMA21"
        else:  # "nearest"
            # For TREND_UP, the pullback comes DOWN — pick the HIGHEST anchor
            # that is still below current price.  For TREND_DOWN, pullback
            # comes UP — pick the LOWEST anchor still above current price.
            candidates = {"VWAP": vwap, "EMA9": ema9, "EMA21": ema21}
            if regime.regime == TREND_UP:
                above = {n: v for n, v in candidates.items() if v <= spy_price}
                if above:
                    anchor_name = max(above, key=lambda n: above[n])
                    anchor = above[anchor_name]
                else:
                    anchor, anchor_name = vwap, "VWAP"
            else:
                below = {n: v for n, v in candidates.items() if v >= spy_price}
                if below:
                    anchor_name = min(below, key=lambda n: below[n])
                    anchor = below[anchor_name]
                else:
                    anchor, anchor_name = ema9, "EMA9"

        # Invalidation: last close beyond the anchor in the trend-wrong direction.
        last_close = bars[-1]["close"]
        inv_gap = cfg.invalidation_vwap_pct * spy_price
        if regime.regime == TREND_UP and last_close < anchor - inv_gap:
            return None
        if regime.regime == TREND_DOWN and last_close > anchor + inv_gap:
            return None

        # Look for a rejection candle at the anchor in the prior 3 bars.
        band = cfg.pullback_band_pct * spy_price
        direction = "C" if regime.regime == TREND_UP else "P"

        prev_bar: Optional[Dict] = None
        for i in range(max(0, len(bars) - 3), len(bars)):
            bar = bars[i]
            low, high = bar["low"], bar["high"]

            touched_anchor = (low <= anchor + band) and (high >= anchor - band)
            if not touched_anchor:
                prev_bar = bar
                continue

            if direction == "C":
                is_rej = _s.is_bullish_rejection(bar, cfg.rejection_wick_ratio)
                is_eng = (
                    cfg.allow_engulfing
                    and prev_bar is not None
                    and _s.is_bullish_engulfing(prev_bar, bar)
                )
                if is_rej or is_eng:
                    trigger = bar["high"] + 0.01
                    stop = bar["low"] - 0.01
                    return ContinuationCandidate(
                        direction="C",
                        trigger_price=trigger,
                        stop_price=stop,
                        confidence=cfg.base_confidence,
                        reasons=[
                            f"TREND_UP pullback to {anchor_name}={anchor:.2f}, "
                            + ("rejection" if is_rej else "engulfing")
                        ],
                        rejection_bar_ts=bar.get("date", datetime.utcnow()),
                        vwap=vwap,
                        spy_price=spy_price,
                    )
            else:
                is_rej = _s.is_bearish_rejection(bar, cfg.rejection_wick_ratio)
                is_eng = (
                    cfg.allow_engulfing
                    and prev_bar is not None
                    and _s.is_bearish_engulfing(prev_bar, bar)
                )
                if is_rej or is_eng:
                    trigger = bar["low"] - 0.01
                    stop = bar["high"] + 0.01
                    return ContinuationCandidate(
                        direction="P",
                        trigger_price=trigger,
                        stop_price=stop,
                        confidence=cfg.base_confidence,
                        reasons=[
                            f"TREND_DOWN pullback to {anchor_name}={anchor:.2f}, "
                            + ("rejection" if is_rej else "engulfing")
                        ],
                        rejection_bar_ts=bar.get("date", datetime.utcnow()),
                        vwap=vwap,
                        spy_price=spy_price,
                    )
            prev_bar = bar

        return None
