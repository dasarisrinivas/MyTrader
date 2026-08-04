"""Entry-quality gate: exhaustion, parabolic, RSI, VWAP-extension filter.

Blocks entries where structural context says the move is already too stretched
to give new money a fair shot. Every rejection returns a reason string.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from . import structure as _s
from .config import EntryGateConfig
from .regime import RegimeV2Context


@dataclass
class EntryGateResult:
    allowed: bool
    reason: str


class EntryGate:
    def __init__(self, cfg: Optional[EntryGateConfig] = None) -> None:
        self._cfg = cfg or EntryGateConfig()

    def check(
        self,
        direction: str,              # "C" or "P"
        bars: List[Dict],
        spy_price: float,
        confidence: float,
        regime: RegimeV2Context,
        signal_type: str = "",
    ) -> EntryGateResult:
        cfg = self._cfg

        # ── Confidence floor — DISABLED AUG 4 2026 (defect D1) ───────────
        # Duplicated the engine-level confidence gate on the identical metric
        # (`sig.confidence`). With the engine gate passive since 2026-08-03,
        # this was the ONLY active confidence filter and it silently kept the
        # confidence experiment from running end-to-end: 91 of 125 entry_gate
        # rejections on 2026-08-04 came from here. See EntryGateConfig.
        if getattr(cfg, "confidence_floor_enabled", True) and \
                confidence < cfg.min_confidence:
            return EntryGateResult(
                allowed=False,
                reason=f"confidence {confidence:.2f} < min {cfg.min_confidence}",
            )

        # ── VWAP extension ───────────────────────────────────────────────
        # TREND_CONTINUATION signals are designed to enter AFTER the trend has
        # stretched away from VWAP — blocking them on "too extended" defeats
        # the whole purpose. The detector's own invalidation + pullback logic
        # already enforces entry quality, so we skip the VWAP-extension check
        # for continuations.
        if (
            signal_type != "TREND_CONTINUATION"
            and regime.vwap > 0
            and spy_price > 0
        ):
            extension = (spy_price - regime.vwap) / regime.vwap
            if direction == "C" and extension > cfg.max_vwap_extension_pct:
                return EntryGateResult(
                    allowed=False,
                    reason=(
                        f"exhaustion: +{extension:.4f} extended above VWAP "
                        f"(>{cfg.max_vwap_extension_pct:.4f})"
                    ),
                )
            if direction == "P" and -extension > cfg.max_vwap_extension_pct:
                return EntryGateResult(
                    allowed=False,
                    reason=(
                        f"exhaustion: {extension:.4f} extended below VWAP "
                        f"(>{cfg.max_vwap_extension_pct:.4f})"
                    ),
                )

        # ── Parabolic bar count ──────────────────────────────────────────
        atr_val = _s.atr(bars, 14) or 0.0
        if atr_val > 0 and len(bars) >= cfg.parabolic_bar_count:
            in_direction: List[Dict] = []
            for b in reversed(bars[-cfg.parabolic_bar_count:]):
                if direction == "C" and b["close"] > b["open"]:
                    in_direction.append(b)
                elif direction == "P" and b["close"] < b["open"]:
                    in_direction.append(b)
                else:
                    break
            if len(in_direction) >= cfg.parabolic_bar_count:
                big = sum(
                    1
                    for b in in_direction
                    if (b["high"] - b["low"]) >= 1.2 * atr_val
                )
                if big >= cfg.parabolic_bar_count:
                    return EntryGateResult(
                        allowed=False,
                        reason=(
                            f"parabolic: {cfg.parabolic_bar_count} consecutive "
                            f">1.2×ATR bars in direction"
                        ),
                    )

        # ── RSI exhaustion ───────────────────────────────────────────────
        # Same reasoning as VWAP extension: a TREND_CONTINUATION signal by
        # definition enters with the trend, which is expected to produce
        # an RSI reading at the trend-side extreme (RSI>70 in TREND_UP,
        # RSI<30 in TREND_DOWN). That extreme CONFIRMS the trend — it is
        # not exhaustion. For fresh-entry signals (ORB, PC_RATIO, sweeps),
        # the RSI check still applies.
        if signal_type != "TREND_CONTINUATION":
            closes = [b["close"] for b in bars]
            r = _s.rsi(closes, period=14)
            if r is not None:
                if direction == "C" and r > cfg.rsi_upper:
                    return EntryGateResult(
                        allowed=False,
                        reason=f"RSI={r:.0f} >{cfg.rsi_upper} (exhausted long)",
                    )
                if direction == "P" and r < cfg.rsi_lower:
                    return EntryGateResult(
                        allowed=False,
                        reason=f"RSI={r:.0f} <{cfg.rsi_lower} (exhausted short)",
                    )

        return EntryGateResult(allowed=True, reason="entry gate passed")
