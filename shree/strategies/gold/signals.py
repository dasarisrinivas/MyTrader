"""Gold intraday signal generator.

Implements two signal families:

1. VWAP / EMA Pullback
   - Pullback into VWAP / EMA21 zone after a confirmed trending regime
   - Entry on break of the confirmation bar's high (long) or low (short)

2. Opening Range Breakout (ORB)
   - Wait for the opening range to form (first N minutes)
   - Enter on close above OR-high (long) or below OR-low (short)

All signals include computed SL/TP levels so the risk manager can verify
them without re-deriving indicator values.

This module is *pure signal generation* — it never places orders.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, time
from enum import Enum
from typing import Dict, Optional, Tuple

import pandas as pd

from ...config.gold import (
    GoldEntryConfig,
    GoldExitConfig,
    GoldIndicatorConfig,
    GoldSessionBucket,
    GoldSessionBucketConfig,
    GoldSessionConfig,
)
from ...utils.logger import logger
from ...utils.news_calendar import is_in_lockout
from ...utils.timezone_utils import now_cst
from .regime import GoldRegime


class GoldSignalType(str, Enum):
    """Named signal types produced by this generator."""

    VWAP_PB_LONG = "VWAP_PB_LONG"       # VWAP pullback — long
    VWAP_PB_SHORT = "VWAP_PB_SHORT"     # VWAP pullback — short
    EMA_PB_LONG = "EMA_PB_LONG"         # EMA21 pullback — long
    EMA_PB_SHORT = "EMA_PB_SHORT"       # EMA21 pullback — short
    ORB_LONG = "ORB_LONG"               # Opening range breakout — long
    ORB_SHORT = "ORB_SHORT"             # Opening range breakdown — short
    NONE = "NONE"                        # No signal


@dataclass
class GoldSignal:
    """Fully specified signal including computed SL/TP.

    ``action`` mirrors the BaseStrategy convention: "BUY", "SELL", or "HOLD".
    """

    action: str                          # "BUY", "SELL", or "HOLD"
    signal_type: GoldSignalType = GoldSignalType.NONE
    confidence: float = 0.0
    entry_ref_price: float = 0.0         # Close of the triggering bar (reference)
    stop_loss: float = 0.0
    take_profit: float = 0.0
    atr: float = 0.0
    regime: GoldRegime = GoldRegime.NO_TRADE
    metadata: Dict = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.action in ("BUY", "SELL")


def _snap_to_tick(price: float, tick_size: float, direction: int) -> float:
    """Round price to the nearest tick, biased toward safety.

    direction = +1 → round up (used for targets / ask-side entries)
    direction = -1 → round down (used for stops on longs)
    """
    if tick_size <= 0:
        return price
    ticks = price / tick_size
    if direction >= 0:
        return math.ceil(ticks - 1e-9) * tick_size
    return math.floor(ticks + 1e-9) * tick_size


class GoldSignalGenerator:
    """Produce gold intraday signals from a completed OHLCV + indicator DataFrame.

    State held per session:
    - Opening range high/low (reset when session date changes)
    - Confirmation bar tracking for pullback setups
    - Post-loss cooldown bar count

    The generator is *not* responsible for position tracking or order
    placement — that belongs to GoldTradingManager.
    """

    def __init__(
        self,
        session: GoldSessionConfig,
        indicators: GoldIndicatorConfig,
        entry: GoldEntryConfig,
        exit_cfg: GoldExitConfig,
        tick_size: float = 0.10,   # MGC/GC default
    ) -> None:
        self._session = session
        self._ind = indicators
        self._entry = entry
        self._exit = exit_cfg
        self._tick_size = tick_size

        # ── Session-level state ───────────────────────────────────────────────
        self._session_date: Optional[date] = None
        self._or_high: Optional[float] = None
        self._or_low: Optional[float] = None
        self._or_formed: bool = False
        self._or_bar_count: int = 0

        # Post-loss cooldown (bars remaining)
        self._post_loss_cooldown_bars: int = 0

        # Direction-aware cooldown state
        self._loss_signal_family: Optional[str] = None   # e.g. "VWAP_PB", "EMA_PB", "ORB"
        self._loss_direction: Optional[str] = None        # "BUY" or "SELL"
        self._same_family_cooldown_bars: int = 0
        self._opposite_cooldown_bars: int = 0

    @staticmethod
    def _hold_signal(
        regime: GoldRegime,
        reason: Optional[str] = None,
        **metadata,
    ) -> GoldSignal:
        payload = dict(metadata)
        if reason:
            payload["block_reason"] = reason
        return GoldSignal(action="HOLD", regime=regime, metadata=payload)

    @staticmethod
    def _merge_block_reason(signal: GoldSignal, reason: str, **metadata) -> GoldSignal:
        payload = dict(signal.metadata)
        payload.update(metadata)
        payload["block_reason"] = reason
        signal.metadata = payload
        return signal

    # ── Public API ────────────────────────────────────────────────────────────

    @staticmethod
    def _signal_family(signal_type: GoldSignalType) -> str:
        """Extract the family prefix from a signal type (e.g. VWAP_PB, EMA_PB, ORB)."""
        name = signal_type.value  # e.g. "VWAP_PB_LONG", "ORB_SHORT"
        if name.startswith("VWAP_PB"):
            return "VWAP_PB"
        if name.startswith("EMA_PB"):
            return "EMA_PB"
        if name.startswith("ORB"):
            return "ORB"
        return "OTHER"

    def notify_loss(
        self,
        signal_type: Optional[GoldSignalType] = None,
        direction: Optional[str] = None,
    ) -> None:
        """Call after a stop-loss fill so the generator enforces a bar cooldown.

        When ``signal_type`` and ``direction`` are provided, the generator uses
        direction-aware cooldowns: longer for the same signal family/direction,
        shorter for opposite-direction setups.  When omitted, falls back to the
        uniform ``post_loss_cooldown_bars``.
        """
        # Always set the uniform fallback cooldown
        self._post_loss_cooldown_bars = self._entry.post_loss_cooldown_bars

        # Direction-aware cooldown (only if config enables it)
        same_family_bars = self._entry.post_loss_same_family_cooldown_bars
        if same_family_bars > 0 and signal_type is not None and direction is not None:
            self._loss_signal_family = self._signal_family(signal_type)
            self._loss_direction = direction
            self._same_family_cooldown_bars = same_family_bars
            self._opposite_cooldown_bars = self._entry.post_loss_opposite_cooldown_bars
            logger.debug(
                "GoldSignalGenerator: direction-aware cooldown — family={} dir={} same={}bars opp={}bars",
                self._loss_signal_family,
                self._loss_direction,
                self._same_family_cooldown_bars,
                self._opposite_cooldown_bars,
            )
        else:
            self._loss_signal_family = None
            self._loss_direction = None
            self._same_family_cooldown_bars = 0
            self._opposite_cooldown_bars = 0
            logger.debug(
                "GoldSignalGenerator: post-loss cooldown set to {} bars (uniform)",
                self._post_loss_cooldown_bars,
            )

    def generate(
        self,
        features: pd.DataFrame,
        regime: GoldRegime,
        bar_timestamp: Optional[pd.Timestamp] = None,
    ) -> GoldSignal:
        """Evaluate the latest completed bar and return a GoldSignal.

        Args:
            features:       DataFrame of *completed* bars with columns
                            [open, high, low, close, volume, ema9, ema21,
                             atr, adx, vwap].  Last row = most recent bar.
            regime:         Current market regime from GoldRegimeDetector.
            bar_timestamp:  Timestamp of the last bar (used for OR tracking).
                            If None, inferred from the DataFrame index.

        Returns:
            GoldSignal with action "BUY", "SELL", or "HOLD".
        """
        if len(features) == 0:
            return self._hold_signal(GoldRegime.NO_TRADE, "empty_features")

        # ── Resolve bar timestamp ─────────────────────────────────────────────
        if bar_timestamp is None and isinstance(features.index, pd.DatetimeIndex):
            bar_timestamp = features.index[-1]

        # ── Classify session bucket (Phase 5) ────────────────────────────────
        bucket = self._classify_session_bucket(bar_timestamp)
        bucket_cfg = self._get_bucket_config(bucket)

        # UNKNOWN bucket (13:30–18:00 ET gap) — no trading
        if bucket == GoldSessionBucket.UNKNOWN:
            return self._hold_signal(
                regime if regime else GoldRegime.NO_TRADE,
                "session_bucket_unknown",
                session_bucket=bucket.value,
            )

        # ── Update session state (OR tracking) ───────────────────────────────
        if bar_timestamp is not None:
            self._update_session_state(features, bar_timestamp)

        # ── News lockout gate ─────────────────────────────────────────────────
        if bar_timestamp is not None and self._session.news_lockout_windows_et:
            if is_in_lockout(bar_timestamp, self._session.news_lockout_windows_et):
                logger.debug("GoldSignalGenerator: news lockout active — HOLD")
                return self._hold_signal(regime if regime else GoldRegime.NO_TRADE, "news_lockout")

        # ── Post-loss cooldown (direction-aware) ─────────────────────────────
        # Snapshot counters *before* decrement to decide whether to block this bar
        in_uniform_cooldown = self._post_loss_cooldown_bars > 0
        in_family_cooldown = self._same_family_cooldown_bars > 0
        in_opposite_cooldown = self._opposite_cooldown_bars > 0

        # Decrement all active counters (happens every bar regardless)
        if in_uniform_cooldown:
            self._post_loss_cooldown_bars -= 1
        if in_family_cooldown:
            self._same_family_cooldown_bars -= 1
        if in_opposite_cooldown:
            self._opposite_cooldown_bars -= 1

        # During the *opposite* cooldown window (shortest), block everything
        if in_opposite_cooldown and in_family_cooldown:
            logger.debug(
                "GoldSignalGenerator: in post-loss cooldown (opposite dir, {} bars left)",
                self._opposite_cooldown_bars + 1,
            )
            return self._hold_signal(regime if regime else GoldRegime.NO_TRADE, "post_loss_cooldown")

        # During the uniform fallback cooldown, block if no direction-aware config
        if in_uniform_cooldown and not in_family_cooldown:
            logger.debug(
                "GoldSignalGenerator: in post-loss cooldown ({} bars left)",
                self._post_loss_cooldown_bars + 1,
            )
            return self._hold_signal(regime if regime else GoldRegime.NO_TRADE, "post_loss_cooldown")

        # ── Regime guard ─────────────────────────────────────────────────────
        if not regime in (GoldRegime.TRENDING_BULL, GoldRegime.TRENDING_BEAR):
            return self._hold_signal(regime, "regime_not_tradeable")

        row = features.iloc[-1]
        if pd.isna(row[["close", "ema9", "ema21", "atr", "vwap"]]).any():
            return self._hold_signal(regime, "missing_required_indicator")

        close = float(row["close"])
        ema9 = float(row["ema9"])
        ema21 = float(row["ema21"])
        atr = float(row["atr"])
        vwap = float(row["vwap"])

        # ── Minimum bar volume guard (session-bucket-adjusted) ─────────────────
        if "volume" in features.columns:
            volume = int(row.get("volume", 0))
            adjusted_min_volume = int(self._entry.min_bar_volume * bucket_cfg.volume_min_mult)
            if volume < adjusted_min_volume:
                return self._hold_signal(
                    regime,
                    "bar_volume_below_minimum",
                    volume=volume,
                    min_bar_volume=adjusted_min_volume,
                    session_bucket=bucket.value,
                )

        # ── ADX session-bucket gate ────────────────────────────────────────────
        if "adx" in features.columns:
            adx_val = float(row.get("adx", 0.0))
            adjusted_adx_min = self._entry.adx_trend_min * bucket_cfg.adx_min_mult
            if adx_val < adjusted_adx_min:
                return self._hold_signal(
                    regime,
                    "adx_below_bucket_minimum",
                    adx=adx_val,
                    adx_min=adjusted_adx_min,
                    session_bucket=bucket.value,
                )

        # ── ATR minimum ratio session-bucket gate ─────────────────────────────
        if close > 0 and atr > 0:
            atr_ratio = atr / close
            adjusted_atr_min = self._entry.atr_min_ratio * bucket_cfg.atr_min_ratio_mult
            if atr_ratio < adjusted_atr_min:
                return self._hold_signal(
                    regime,
                    "atr_ratio_below_bucket_minimum",
                    atr_ratio=atr_ratio,
                    atr_min_ratio=adjusted_atr_min,
                    session_bucket=bucket.value,
                )

        # ── Try signals in priority order (session-bucket-aware) ──────────────
        orb_signal = None
        # 1. ORB (higher conviction when range is clean)
        if self._entry.orb_enabled and self._or_formed and bucket_cfg.orb_enabled:
            orb_signal = self._check_orb(
                features, close, atr, vwap, regime,
                bar_ts=bar_timestamp, bucket_cfg=bucket_cfg,
            )
            if orb_signal.is_actionable:
                # Apply per-bucket confidence offset
                orb_signal.confidence = max(0.0, orb_signal.confidence + bucket_cfg.confidence_offset)
                orb_signal.metadata["session_bucket"] = bucket.value
                if self._is_same_family_cooldown_blocked(orb_signal):
                    pass  # Fall through to pullback; don't return yet
                else:
                    logger.debug(
                        "GoldSignalGenerator: {} in {} bucket (conf={:.2f})",
                        orb_signal.signal_type.value, bucket.value, orb_signal.confidence,
                    )
                    return orb_signal
        elif self._entry.orb_enabled and self._or_formed and not bucket_cfg.orb_enabled:
            orb_signal = self._hold_signal(regime, "orb_disabled_in_bucket", session_bucket=bucket.value)

        # 2. Pullback setups
        if bucket_cfg.pullback_enabled:
            pb_signal = self._check_pullback(
                features, close, ema9, ema21, atr, vwap, regime,
                bar_ts=bar_timestamp, bucket_cfg=bucket_cfg,
            )
            if pb_signal.is_actionable:
                # Apply per-bucket confidence offset
                pb_signal.confidence = max(0.0, pb_signal.confidence + bucket_cfg.confidence_offset)
                pb_signal.metadata["session_bucket"] = bucket.value
                if self._is_same_family_cooldown_blocked(pb_signal):
                    pass  # Blocked — fall through to HOLD
                else:
                    logger.debug(
                        "GoldSignalGenerator: {} in {} bucket (conf={:.2f})",
                        pb_signal.signal_type.value, bucket.value, pb_signal.confidence,
                    )
                    return pb_signal
        else:
            pb_signal = self._hold_signal(regime, "pullback_disabled_in_bucket", session_bucket=bucket.value)

        block_reason = pb_signal.metadata.get("block_reason") or (orb_signal.metadata.get("block_reason") if orb_signal else None)
        extra = {"session_bucket": bucket.value}
        if orb_signal and orb_signal.metadata.get("block_reason"):
            extra["orb_block_reason"] = orb_signal.metadata.get("block_reason")
        if pb_signal.metadata.get("block_reason"):
            extra["pullback_block_reason"] = pb_signal.metadata.get("block_reason")
        return self._hold_signal(regime, block_reason or "no_setup", **extra)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_same_family_cooldown_blocked(self, signal: GoldSignal) -> bool:
        """Return True if the signal matches the losing family/direction and is still in cooldown."""
        if self._same_family_cooldown_bars <= 0:
            return False
        if self._loss_signal_family is None or self._loss_direction is None:
            return False

        candidate_family = self._signal_family(signal.signal_type)
        candidate_direction = signal.action  # "BUY" or "SELL"

        is_same_family = candidate_family == self._loss_signal_family
        is_same_direction = candidate_direction == self._loss_direction

        if is_same_family and is_same_direction:
            logger.debug(
                "GoldSignalGenerator: BLOCKED by same-family cooldown — {} {} "
                "(lost on {} {}, {} bars remaining)",
                candidate_family,
                candidate_direction,
                self._loss_signal_family,
                self._loss_direction,
                self._same_family_cooldown_bars,
            )
            return True

        return False

    def _update_session_state(
        self, features: pd.DataFrame, bar_timestamp: pd.Timestamp
    ) -> None:
        """Reset OR state on new session date; accumulate OR bars."""
        try:
            ts_local = bar_timestamp.tz_convert("America/New_York")
        except Exception:
            ts_local = bar_timestamp

        today = ts_local.date()

        if today != self._session_date:
            # New session — reset everything
            self._session_date = today
            self._or_high = None
            self._or_low = None
            self._or_formed = False
            self._or_bar_count = 0
            logger.debug("GoldSignalGenerator: new session {} — OR reset", today)

        if not self._or_formed:
            try:
                session_open = time(
                    int(self._session.session_open_et.split(":")[0]),
                    int(self._session.session_open_et.split(":")[1]),
                )
            except Exception:
                session_open = time(8, 20)

            bar_time = ts_local.time()
            if bar_time < session_open:
                return

            # Count bars since session open
            self._or_bar_count += 1
            row = features.iloc[-1]
            bar_high = float(row.get("high", row["close"]))
            bar_low = float(row.get("low", row["close"]))

            if self._or_high is None or bar_high > self._or_high:
                self._or_high = bar_high
            if self._or_low is None or bar_low < self._or_low:
                self._or_low = bar_low

            if self._or_bar_count >= self._session.opening_range_minutes:
                # Validate OR range is meaningful
                if self._or_high is not None and self._or_low is not None:
                    or_range = self._or_high - self._or_low
                    mid = (self._or_high + self._or_low) / 2
                    if mid > 0 and or_range / mid >= self._entry.orb_min_range_ratio:
                        self._or_formed = True
                        logger.info(
                            "GoldSignalGenerator: OR formed — high=%.2f low=%.2f range=%.2f",
                            self._or_high,
                            self._or_low,
                            or_range,
                        )

    def _is_extended_hours(self, bar_ts: Optional[pd.Timestamp]) -> bool:
        """Return True when the bar falls in the extended/overnight session."""
        if not self._session.extended_hours_enabled or bar_ts is None:
            return False
        try:
            local = bar_ts.tz_convert("America/New_York")
        except Exception:
            local = bar_ts
        h, m = local.hour, local.minute
        open_h, open_m = (
            int(self._session.session_open_et.split(":")[0]),
            int(self._session.session_open_et.split(":")[1]),
        )
        ext_h, ext_m = (
            int(self._session.extended_session_open_et.split(":")[0]),
            int(self._session.extended_session_open_et.split(":")[1]),
        )
        bar_hm = h * 60 + m
        rth_open = open_h * 60 + open_m
        ext_open = ext_h * 60 + ext_m
        # Extended = after ext_open (e.g. 18:00) OR before rth_open (e.g. 08:20)
        return bar_hm >= ext_open or bar_hm < rth_open

    def _classify_session_bucket(self, bar_ts: Optional[pd.Timestamp]) -> GoldSessionBucket:
        """Map a bar timestamp to the appropriate session bucket.

        All bucket boundaries are specified in ET. The classification order is:
        1. OVERNIGHT:  18:00 ET → 03:00 ET  (crosses midnight)
        2. PRE_COMEX:  03:00 ET → COMEX_OPEN start
        3. COMEX_OPEN: COMEX open → MIDDAY start
        4. MIDDAY:     midday start → PRE_CLOSE start
        5. PRE_CLOSE:  pre-close start → session_close_et
        6. MAINTENANCE: checked separately (CT-based, handled by caller)
        7. UNKNOWN:     anything in the gap between close and overnight open

        Returns GoldSessionBucket.COMEX_OPEN as the default if bar_ts is None
        (most permissive fallback — do not restrict signals when timestamp unknown).
        """
        if bar_ts is None:
            return GoldSessionBucket.COMEX_OPEN

        try:
            local = bar_ts.tz_convert("America/New_York")
        except Exception:
            local = bar_ts

        bar_hm = local.hour * 60 + local.minute

        def _parse_hm(s: str) -> int:
            parts = s.split(":")
            return int(parts[0]) * 60 + int(parts[1])

        overnight_start = _parse_hm(self._session.bucket_overnight_start_et)
        pre_comex_start = _parse_hm(self._session.bucket_pre_comex_start_et)
        comex_open_start = _parse_hm(self._session.bucket_comex_open_start_et)
        midday_start = _parse_hm(self._session.bucket_midday_start_et)
        pre_close_start = _parse_hm(self._session.bucket_pre_close_start_et)
        session_close = _parse_hm(self._session.session_close_et)

        # OVERNIGHT wraps midnight: 18:00+ OR before pre_comex
        if bar_hm >= overnight_start or bar_hm < pre_comex_start:
            return GoldSessionBucket.OVERNIGHT

        if bar_hm < comex_open_start:
            return GoldSessionBucket.PRE_COMEX

        if bar_hm < midday_start:
            return GoldSessionBucket.COMEX_OPEN

        if bar_hm < pre_close_start:
            return GoldSessionBucket.MIDDAY

        if bar_hm < session_close:
            return GoldSessionBucket.PRE_CLOSE

        # After session close but before overnight open (13:30–18:00 ET)
        return GoldSessionBucket.UNKNOWN

    def _get_bucket_config(self, bucket: GoldSessionBucket) -> GoldSessionBucketConfig:
        """Return the per-bucket config, falling back to neutral defaults."""
        cfg = self._session.buckets.get(bucket.value)
        if cfg is not None:
            return cfg
        # MAINTENANCE, UNKNOWN, or missing — return neutral config (all 1.0 / no offset)
        return GoldSessionBucketConfig()

    def _sl_tp(
        self,
        action: str,
        entry_price: float,
        atr: float,
        bar_ts: Optional[pd.Timestamp] = None,
    ) -> Tuple[float, float]:
        """Compute SL and TP for a given action, ATR-based, snapped to tick.

        Uses wider multipliers during extended/overnight hours if enabled.
        """
        if self._is_extended_hours(bar_ts):
            sl_mult = self._exit.extended_atr_sl_multiplier
            tp_mult = self._exit.extended_atr_tp_multiplier
        else:
            sl_mult = self._exit.atr_sl_multiplier
            tp_mult = self._exit.atr_tp_multiplier

        sl_distance = atr * sl_mult
        sl_distance = max(sl_distance, self._exit.sl_floor_points)
        sl_distance = min(sl_distance, self._exit.sl_ceiling_points)

        tp_distance = atr * tp_mult

        if action == "BUY":
            stop_loss = _snap_to_tick(entry_price - sl_distance, self._tick_size, -1)
            take_profit = _snap_to_tick(entry_price + tp_distance, self._tick_size, +1)
        else:  # SELL
            stop_loss = _snap_to_tick(entry_price + sl_distance, self._tick_size, +1)
            take_profit = _snap_to_tick(entry_price - tp_distance, self._tick_size, -1)

        return stop_loss, take_profit

    def _check_rr(self, action: str, entry: float, sl: float, tp: float) -> bool:
        """Return True if R:R meets the minimum threshold."""
        if action == "BUY":
            risk = entry - sl
            reward = tp - entry
        else:
            risk = sl - entry
            reward = entry - tp
        if risk <= 0:
            return False
        return (reward / risk) >= self._exit.min_rr_ratio

    def _check_pullback(
        self,
        features: pd.DataFrame,
        close: float,
        ema9: float,
        ema21: float,
        atr: float,
        vwap: float,
        regime: GoldRegime,
        bar_ts: Optional[pd.Timestamp] = None,
        bucket_cfg: Optional[GoldSessionBucketConfig] = None,
    ) -> GoldSignal:
        """VWAP or EMA21 pullback signal."""
        row = features.iloc[-1]
        open_price = float(row.get("open", close))
        bar_high = float(row.get("high", close))
        bar_low = float(row.get("low", close))
        bar_range = max(bar_high - bar_low, self._tick_size)
        body_fraction = abs(close - open_price) / bar_range
        best_rejection: Optional[Tuple[str, Dict]] = None

        def remember(reason: str, **meta) -> None:
            nonlocal best_rejection
            if best_rejection is None:
                best_rejection = (reason, meta)

        if regime == GoldRegime.TRENDING_BULL:
            action = "BUY"
            signal_types_to_try = [
                (GoldSignalType.VWAP_PB_LONG, vwap),
                (GoldSignalType.EMA_PB_LONG, ema21),
            ]
        else:  # TRENDING_BEAR
            action = "SELL"
            signal_types_to_try = [
                (GoldSignalType.VWAP_PB_SHORT, vwap),
                (GoldSignalType.EMA_PB_SHORT, ema21),
            ]

        for sig_type, level in signal_types_to_try:
            if level <= 0:
                continue
            touch_pct = self._entry.vwap_touch_pct if "VWAP" in sig_type.value else self._entry.ema_touch_pct
            touch_band = level * touch_pct

            if atr > 0:
                # Use per-bucket extension strictness (Phase 5), falling back to
                # the legacy extended_hours_extension_strictness_mult for backward compat
                if bucket_cfg is not None:
                    ext_mult = bucket_cfg.extension_strictness_mult
                elif self._is_extended_hours(bar_ts):
                    ext_mult = self._entry.extended_hours_extension_strictness_mult
                else:
                    ext_mult = 1.0
                max_vwap_ext = self._entry.pullback_max_vwap_extension_atr * ext_mult
                max_ema_ext = self._entry.pullback_max_ema_extension_atr * ext_mult

                vwap_extension_atr = abs(close - vwap) / atr if vwap > 0 else 0.0
                ema_extension_atr = abs(close - ema21) / atr if ema21 > 0 else 0.0
                if vwap_extension_atr > max_vwap_ext:
                    remember(
                        "pullback_vwap_extension_exceeded",
                        signal_candidate=sig_type.value,
                        vwap_extension_atr=vwap_extension_atr,
                        max_allowed=max_vwap_ext,
                        extended_hours=self._is_extended_hours(bar_ts),
                    )
                    continue
                if ema_extension_atr > max_ema_ext:
                    remember(
                        "pullback_ema_extension_exceeded",
                        signal_candidate=sig_type.value,
                        ema_extension_atr=ema_extension_atr,
                        max_allowed=max_ema_ext,
                        extended_hours=self._is_extended_hours(bar_ts),
                    )
                    continue

            # Is the current close within the touch zone?
            in_zone: bool
            if action == "BUY":
                # Long: price pulled back to within touch_band of level
                in_zone = close <= level + touch_band and close >= level - touch_band * 2
            else:
                # Short: price pulled back to within touch_band of level
                in_zone = close >= level - touch_band and close <= level + touch_band * 2

            if not in_zone:
                remember("pullback_not_in_touch_zone", signal_candidate=sig_type.value)
                continue

            if self._entry.pullback_reclaim_required:
                if action == "BUY":
                    touched = bar_low <= level + touch_band
                    reclaimed = close >= level
                else:
                    touched = bar_high >= level - touch_band
                    reclaimed = close <= level
                if not (touched and reclaimed):
                    remember(
                        "pullback_reclaim_failed",
                        signal_candidate=sig_type.value,
                        touched=touched,
                        reclaimed=reclaimed,
                    )
                    continue

            # Confirmation: close is on the correct side of the level
            confirmed: bool
            if action == "BUY":
                confirmed = close >= level
            else:
                confirmed = close <= level

            if not confirmed:
                remember("pullback_level_not_confirmed", signal_candidate=sig_type.value)
                continue

            if self._entry.pullback_confirm_with_bar_direction:
                if action == "BUY" and close <= open_price:
                    remember("pullback_bar_direction_failed", signal_candidate=sig_type.value)
                    continue
                if action == "SELL" and close >= open_price:
                    remember("pullback_bar_direction_failed", signal_candidate=sig_type.value)
                    continue

            if body_fraction < self._entry.pullback_min_body_fraction:
                remember(
                    "pullback_body_too_small",
                    signal_candidate=sig_type.value,
                    body_fraction=body_fraction,
                )
                continue

            # Build signal
            sl, tp = self._sl_tp(action, close, atr, bar_ts=bar_ts)
            if not self._check_rr(action, close, sl, tp):
                logger.debug(
                    "GoldSignalGenerator: %s skipped — R:R below %.2f",
                    sig_type.value,
                    self._exit.min_rr_ratio,
                )
                remember("pullback_rr_below_minimum", signal_candidate=sig_type.value)
                continue

            confidence = self._pullback_confidence(features, regime, atr, close, level)
            return GoldSignal(
                action=action,
                signal_type=sig_type,
                confidence=confidence,
                entry_ref_price=close,
                stop_loss=sl,
                take_profit=tp,
                atr=atr,
                regime=regime,
                metadata={
                    "level": level,
                    "touch_band": touch_band,
                    "body_fraction": body_fraction,
                    "ema9": ema9,
                    "ema21": ema21,
                    "vwap": vwap,
                },
            )

        if best_rejection is not None:
            reason, meta = best_rejection
            return self._hold_signal(regime, reason, **meta)
        return self._hold_signal(regime, "pullback_no_candidate")

    def _check_orb(
        self,
        features: pd.DataFrame,
        close: float,
        atr: float,
        vwap: float,
        regime: GoldRegime,
        bar_ts: Optional[pd.Timestamp] = None,
        bucket_cfg: Optional[GoldSessionBucketConfig] = None,
    ) -> GoldSignal:
        """Opening range breakout / breakdown signal."""
        if self._or_high is None or self._or_low is None:
            return self._hold_signal(regime, "orb_missing_range")

        row = features.iloc[-1]
        open_price = float(row.get("open", close))
        bar_high = float(row.get("high", close))
        bar_low = float(row.get("low", close))
        volume = float(row.get("volume", 0.0))
        bar_range = max(bar_high - bar_low, 0.0)
        breakout_buffer = atr * self._entry.orb_breakout_min_atr_fraction if atr > 0 else 0.0

        # Long ORB: close breaks above OR high
        if close > self._or_high + breakout_buffer and regime in (
            GoldRegime.TRENDING_BULL,
            GoldRegime.RANGING,  # ORB valid even in ranging; regime just provides direction
        ):
            action = "BUY"
            sig_type = GoldSignalType.ORB_LONG
        # Short ORB: close breaks below OR low
        elif close < self._or_low - breakout_buffer and regime in (
            GoldRegime.TRENDING_BEAR,
            GoldRegime.RANGING,
        ):
            action = "SELL"
            sig_type = GoldSignalType.ORB_SHORT
        else:
            return self._hold_signal(
                regime,
                "orb_breakout_buffer_not_cleared",
                breakout_buffer=breakout_buffer,
                or_high=self._or_high,
                or_low=self._or_low,
            )

        lookback = max(1, self._entry.orb_volume_lookback_bars)
        if "volume" in features.columns and len(features) > 1:
            prior_window = features["volume"].iloc[max(0, len(features) - 1 - lookback): len(features) - 1]
            if len(prior_window) > 0:
                baseline_volume = float(prior_window.median())
                required_volume = baseline_volume * self._entry.orb_volume_min_multiple
                if baseline_volume > 0 and volume < required_volume:
                    return self._hold_signal(
                        regime,
                        "orb_volume_below_threshold",
                        baseline_volume=baseline_volume,
                        required_volume=required_volume,
                        volume=volume,
                    )

        if atr > 0 and self._entry.orb_max_breakout_candle_atr > 0:
            if bar_range > atr * self._entry.orb_max_breakout_candle_atr:
                return self._hold_signal(
                    regime,
                    "orb_breakout_candle_too_large",
                    bar_range=bar_range,
                    atr=atr,
                )

        if atr > 0 and self._entry.orb_max_extension_atr > 0:
            if action == "BUY":
                extension_atr = (close - self._or_high) / atr
            else:
                extension_atr = (self._or_low - close) / atr
            if extension_atr > self._entry.orb_max_extension_atr:
                return self._hold_signal(
                    regime,
                    "orb_extension_too_large",
                    extension_atr=extension_atr,
                )

        # ORB VWAP extension guard — block ORB when price has run too far from VWAP
        if atr > 0 and vwap > 0 and self._entry.orb_max_vwap_extension_atr > 0:
            orb_vwap_ext = abs(close - vwap) / atr
            if orb_vwap_ext > self._entry.orb_max_vwap_extension_atr:
                return self._hold_signal(
                    regime,
                    "orb_vwap_extension_too_large",
                    vwap_extension_atr=orb_vwap_ext,
                    max_allowed=self._entry.orb_max_vwap_extension_atr,
                )

        if action == "BUY" and close <= open_price:
            return self._hold_signal(regime, "orb_bar_direction_failed")
        if action == "SELL" and close >= open_price:
            return self._hold_signal(regime, "orb_bar_direction_failed")

        sl, tp = self._sl_tp(action, close, atr, bar_ts=bar_ts)
        if not self._check_rr(action, close, sl, tp):
            return self._hold_signal(regime, "orb_rr_below_minimum")

        return GoldSignal(
            action=action,
            signal_type=sig_type,
            confidence=0.75,   # ORB has slightly elevated confidence by design
            entry_ref_price=close,
            stop_loss=sl,
            take_profit=tp,
            atr=atr,
            regime=regime,
            metadata={
                "or_high": self._or_high,
                "or_low": self._or_low,
                "or_range": self._or_high - self._or_low,
                "breakout_buffer": breakout_buffer,
            },
        )

    @staticmethod
    def _pullback_confidence(
        features: pd.DataFrame,
        regime: GoldRegime,
        atr: float,
        close: float,
        level: float,
    ) -> float:
        """Heuristic confidence for pullback setups, capped in [0.5, 0.95]."""
        # Base confidence by regime alignment
        base = 0.60

        # Closer to the level = higher confidence (tighter pullback)
        distance_pct = abs(close - level) / level if level > 0 else 1.0
        proximity_bonus = max(0.0, 0.10 * (1.0 - distance_pct / 0.005))

        # ADX bonus: stronger trend = more confident
        adx = float(features["adx"].iloc[-1]) if "adx" in features.columns else 25.0
        adx_bonus = min(0.10, (adx - 20.0) / 300.0)   # Caps at +0.10 around ADX=50

        return min(0.95, max(0.50, base + proximity_bonus + adx_bonus))

    # ── State accessors (for testing and manager inspection) ──────────────────

    @property
    def or_high(self) -> Optional[float]:
        return self._or_high

    @property
    def or_low(self) -> Optional[float]:
        return self._or_low

    @property
    def or_formed(self) -> bool:
        return self._or_formed
