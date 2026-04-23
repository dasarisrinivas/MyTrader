"""ORB_BREAKOUT time-window gate + confirmation checks.

Root cause #1 on Apr 21: six `ORB_BREAKOUT` signals fired between 15:08 and
18:35 CST — four to six *hours* after the opening range closed. By then the
"breakout" is not a breakout at all; the strategy is chasing.

This module enforces two gates:

  (a) Hard time gate — outside the configured window, ORB is rejected
      outright regardless of confidence or flow score.

  (b) Confirmation gate — inside the window, require:
        • two consecutive 5m closes beyond the ORB extreme
        • breakout bar volume ≥ 1.5× 20-bar average
        • at least min_vwap_distance_pct from VWAP (no VWAP-huggers)
        • RSI not in exhaustion zone opposite to trade direction
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from . import structure as _s
from .config import OrbGateConfig

ET = ZoneInfo("America/New_York")


@dataclass
class OrbGateResult:
    allowed: bool
    reason: str


def _parse_et(hhmm: str) -> time:
    h, m = hhmm.split(":")
    return time(int(h), int(m))


def _now_et(now: Optional[datetime] = None) -> datetime:
    """Return the current time in America/New_York.

    ``now`` defaults to ``datetime.now(ET)``. If a naive datetime is supplied,
    it is interpreted as UTC and converted to ET.
    """
    if now is None:
        return datetime.now(ET)
    if now.tzinfo is None:
        from datetime import timezone

        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(ET)


class OrbGate:
    """Stateless ORB-breakout gate.

    Call ``check()`` with the bars and breakout metadata. Returns a boolean +
    human-readable reason — never mutates the signal.
    """

    def __init__(self, cfg: Optional[OrbGateConfig] = None) -> None:
        self._cfg = cfg or OrbGateConfig()

    def in_window(self, now: Optional[datetime] = None) -> bool:
        start = _parse_et(self._cfg.window_start_et)
        end = _parse_et(self._cfg.window_end_et)
        t = _now_et(now).time()
        return start <= t < end

    def check(
        self,
        bars: List[Dict],
        direction: str,                 # "C" (bullish) or "P" (bearish)
        orb_high: Optional[float],
        orb_low: Optional[float],
        rsi_value: Optional[float],
        vwap_value: Optional[float],
        spy_price: float,
        now: Optional[datetime] = None,
    ) -> OrbGateResult:
        """Return (allowed, reason). Direction: C=breakout above ORB high,
        P=breakdown below ORB low."""
        cfg = self._cfg

        # ── Hard time gate ───────────────────────────────────────────────
        if not self.in_window(now):
            t_et = _now_et(now).strftime("%H:%M")
            return OrbGateResult(
                allowed=False,
                reason=(
                    f"ORB time-gate: {t_et} ET outside "
                    f"[{cfg.window_start_et}, {cfg.window_end_et}) — "
                    f"breakout is stale; regime has moved on"
                ),
            )

        if direction not in ("C", "P"):
            return OrbGateResult(allowed=False, reason=f"ORB direction invalid: {direction}")
        if orb_high is None or orb_low is None:
            return OrbGateResult(allowed=False, reason="ORB levels not yet established")

        # ── Confirmation: 2 consecutive 5m closes beyond ORB ──────────────
        if len(bars) < cfg.confirmation_bars:
            return OrbGateResult(
                allowed=False, reason=f"need {cfg.confirmation_bars} confirmation bars"
            )
        last = bars[-cfg.confirmation_bars:]
        if direction == "C":
            holds = all(b["close"] > orb_high for b in last)
        else:
            holds = all(b["close"] < orb_low for b in last)
        if not holds:
            return OrbGateResult(
                allowed=False,
                reason=f"ORB {direction}: {cfg.confirmation_bars} closes beyond ORB not held",
            )

        # ── Volume on breakout bar ───────────────────────────────────────
        volumes = [b["volume"] for b in bars]
        avg20 = sum(volumes[-21:-1]) / max(1, len(volumes[-21:-1]))
        breakout_vol = bars[-1]["volume"]
        if avg20 > 0 and breakout_vol < cfg.volume_multiplier * avg20:
            return OrbGateResult(
                allowed=False,
                reason=(
                    f"ORB volume weak: {breakout_vol} < "
                    f"{cfg.volume_multiplier:.1f}× avg20 ({avg20:.0f})"
                ),
            )

        # ── VWAP distance ────────────────────────────────────────────────
        if vwap_value is not None and spy_price > 0:
            distance_pct = abs(spy_price - vwap_value) / spy_price
            if distance_pct < cfg.min_vwap_distance_pct:
                return OrbGateResult(
                    allowed=False,
                    reason=(
                        f"ORB hugging VWAP: distance={distance_pct:.4f} "
                        f"< min={cfg.min_vwap_distance_pct:.4f} — fakeout-prone"
                    ),
                )
            # Direction consistency: bullish ORB must be above VWAP
            if direction == "C" and spy_price < vwap_value:
                return OrbGateResult(
                    allowed=False, reason="ORB bullish but price below VWAP"
                )
            if direction == "P" and spy_price > vwap_value:
                return OrbGateResult(
                    allowed=False, reason="ORB bearish but price above VWAP"
                )

        # ── RSI exhaustion ───────────────────────────────────────────────
        if rsi_value is not None:
            if direction == "C" and rsi_value > cfg.rsi_exhaustion_upper:
                return OrbGateResult(
                    allowed=False, reason=f"ORB bullish but RSI={rsi_value:.0f} >{cfg.rsi_exhaustion_upper}"
                )
            if direction == "P" and rsi_value < cfg.rsi_exhaustion_lower:
                return OrbGateResult(
                    allowed=False, reason=f"ORB bearish but RSI={rsi_value:.0f} <{cfg.rsi_exhaustion_lower}"
                )

        return OrbGateResult(allowed=True, reason="ORB confirmed (time, hold, volume, vwap, rsi)")
