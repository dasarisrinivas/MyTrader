"""Gold intraday strategy — BaseStrategy implementation.

This class wires together:
  - GoldRegimeDetector     (market regime)
  - GoldSignalGenerator    (entry signals)
  - Indicator computation  (EMA, ATR, ADX, VWAP — via pandas rolling)

It conforms to ``BaseStrategy`` so it can be used wherever BaseStrategy
is expected, though the Gold trading manager typically calls it directly
to access the richer ``GoldSignal`` rather than the plain ``Signal``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ...config.gold import GoldStrategyConfig
from ...strategies.base import BaseStrategy, Signal
from ...utils.logger import logger
from .regime import GoldRegime, GoldRegimeDetector
from .signals import GoldSignal, GoldSignalGenerator


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average with min_periods=1 so initial bars have a value."""
    return series.ewm(span=period, adjust=False, min_periods=1).mean()


def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothed moving average (used in ATR/ADX computation)."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def compute_indicators(df: pd.DataFrame, cfg: "GoldStrategyConfig") -> pd.DataFrame:
    """Add Gold-relevant indicators to a OHLCV DataFrame.

    All computations are causal (use only past bars).

    Expected input columns: open, high, low, close, volume.

    Added columns:
        ema9, ema21, atr, adx, vwap
    """
    df = df.copy()

    ind = cfg.indicators

    # ── EMAs ────────────────────────────────────────────────────────────────
    df["ema9"] = _ema(df["close"], ind.ema_fast)
    df["ema21"] = _ema(df["close"], ind.ema_slow)

    # ── EMA slope (Phase 2) — rate of change of fast EMA per bar ────────────
    slope_lb = ind.ema_slope_lookback_bars
    if slope_lb > 0:
        df["ema9_slope"] = (df["ema9"] - df["ema9"].shift(slope_lb)) / slope_lb
    else:
        df["ema9_slope"] = 0.0

    # ── ATR (True Range → Wilder smooth) ────────────────────────────────────
    high = df["high"] if "high" in df.columns else df["close"]
    low = df["low"] if "low" in df.columns else df["close"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = _rma(tr, ind.atr_period)

    # ── ADX (Wilder's +DI / -DI / DX / ADX) ────────────────────────────────
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = (high - prev_high).clip(lower=0.0)
    minus_dm = (prev_low - low).clip(lower=0.0)
    overlap = (plus_dm > 0) & (minus_dm > 0)
    # When both move, only the larger one counts
    keep_plus = plus_dm >= minus_dm
    plus_dm = plus_dm.where(~overlap | keep_plus, 0.0)
    minus_dm = minus_dm.where(~overlap | ~keep_plus, 0.0)

    smooth_tr = _rma(tr, ind.adx_period)
    smooth_plus = _rma(plus_dm, ind.adx_period)
    smooth_minus = _rma(minus_dm, ind.adx_period)

    plus_di = 100.0 * smooth_plus / smooth_tr.replace(0, np.nan)
    minus_di = 100.0 * smooth_minus / smooth_tr.replace(0, np.nan)

    dx_denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / dx_denom
    df["adx"] = _rma(dx.fillna(0.0), ind.adx_period)

    # ── Session VWAP (reset at configured anchor time) ──────────────────────
    vwap_anchor_hour, vwap_anchor_min = _parse_time(ind.vwap_session_anchor_et)
    typical_price = (high + low + df["close"]) / 3.0
    volume = df.get("volume", pd.Series(1, index=df.index)).replace(0, 1)

    # Identify session boundaries (reset at anchor time)
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            local_idx = df.index.tz_convert("America/New_York")
        except Exception:
            local_idx = df.index
        session_id = (
            (local_idx.hour == vwap_anchor_hour) & (local_idx.minute == vwap_anchor_min)
        ).cumsum()
    else:
        # No timezone info — treat all bars as one session
        session_id = pd.Series(1, index=df.index)

    cum_tp_vol = (typical_price * volume).groupby(session_id).cumsum()
    cum_vol = volume.groupby(session_id).cumsum()
    df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)

    # ── Higher-timeframe confirmation indicators (optional) ──────────────────
    if cfg.indicators.mtf_enabled:
        df = compute_htf_indicators(df, cfg)

    return df


def compute_htf_indicators(df: pd.DataFrame, cfg: "GoldStrategyConfig") -> pd.DataFrame:
    """Resample 1-min bars to higher TF and merge ADX/EMA back (forward-filled, causal).

    Adds columns: htf_ema9, htf_ema21, htf_adx
    Only called when cfg.indicators.mtf_enabled is True.
    """
    period = cfg.indicators.mtf_timeframe_minutes
    agg_spec = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg_spec["volume"] = "sum"

    htf = df.resample(f"{period}min").agg(agg_spec).dropna(subset=["close"])
    if len(htf) < 2:
        return df

    htf["htf_ema9"] = _ema(htf["close"], cfg.indicators.ema_fast)
    htf["htf_ema21"] = _ema(htf["close"], cfg.indicators.ema_slow)

    # ATR for HTF ADX
    h = htf["high"]
    lo = htf["low"]
    pc = htf["close"].shift(1)
    tr_htf = pd.concat([(h - lo).abs(), (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    prev_h = h.shift(1)
    prev_l = lo.shift(1)
    plus_dm = (h - prev_h).clip(lower=0.0)
    minus_dm = (prev_l - lo).clip(lower=0.0)
    ov = (plus_dm > 0) & (minus_dm > 0)
    kp = plus_dm >= minus_dm
    plus_dm = plus_dm.where(~ov | kp, 0.0)
    minus_dm = minus_dm.where(~ov | ~kp, 0.0)
    p = cfg.indicators.adx_period
    smt = _rma(tr_htf, p)
    sp = _rma(plus_dm, p)
    sm = _rma(minus_dm, p)
    pdi = 100.0 * sp / smt.replace(0, np.nan)
    mdi = 100.0 * sm / smt.replace(0, np.nan)
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    htf["htf_adx"] = _rma(dx.fillna(0.0), p)

    # Forward-fill HTF values onto 1-min index (causal: only past HTF bars affect current 1m bar)
    out = df.copy()
    for col in ("htf_ema9", "htf_ema21", "htf_adx"):
        out[col] = htf[col].reindex(df.index, method="ffill")

    return out


def _parse_time(t: str) -> tuple[int, int]:
    """Parse 'HH:MM' string into (hour, minute) ints."""
    try:
        h, m = t.split(":")
        return int(h), int(m)
    except Exception:
        return 8, 20   # Fallback: COMEX open


class GoldIntradayStrategy(BaseStrategy):
    """Rules-based intraday gold strategy.

    Implements ``BaseStrategy.generate()`` for ensemble compatibility.
    Call ``generate_gold()`` directly for the richer ``GoldSignal`` type.
    """

    name: str = "gold_intraday"

    def __init__(self, config: GoldStrategyConfig) -> None:
        self._cfg = config
        from ...risk.trade_math import get_contract_spec
        spec = get_contract_spec(config.symbol)
        self._tick_size = spec.tick_size

        self._regime_detector = GoldRegimeDetector(
            indicators=config.indicators,
            entry=config.entry,
        )
        self._signal_gen = GoldSignalGenerator(
            session=config.session,
            indicators=config.indicators,
            entry=config.entry,
            exit_cfg=config.exit,
            tick_size=self._tick_size,
        )

    # ── BaseStrategy interface ────────────────────────────────────────────────

    def generate(self, features: pd.DataFrame) -> Signal:
        """Thin adapter: compute indicators, run gold logic, return plain Signal."""
        gold_signal = self.generate_gold(features)
        return Signal(
            action=gold_signal.action,
            confidence=gold_signal.confidence,
            metadata={
                "signal_type": gold_signal.signal_type.value,
                "stop_loss": gold_signal.stop_loss,
                "take_profit": gold_signal.take_profit,
                "atr": gold_signal.atr,
                "regime": gold_signal.regime.value,
            },
        )

    # ── Gold-specific interface ───────────────────────────────────────────────

    def generate_gold(
        self,
        raw_features: pd.DataFrame,
        bar_timestamp: Optional[pd.Timestamp] = None,
    ) -> GoldSignal:
        """Full signal pipeline: indicators → regime → signal.

        Args:
            raw_features:  OHLCV DataFrame (may or may not have indicators yet).
            bar_timestamp: Override bar timestamp (used in tests).

        Returns:
            GoldSignal with action, SL/TP, and regime metadata.
        """
        if len(raw_features) < self._cfg.indicators.warmup_bars:
            logger.debug(
                "GoldIntradayStrategy: warming up ({}/{} bars)",
                len(raw_features),
                self._cfg.indicators.warmup_bars,
            )
            return GoldSignal(action="HOLD", regime=GoldRegime.WARMING_UP)

        # Session gate: block signals outside the configured trading window
        # unless extended_hours_enabled permits overnight bars
        if bar_timestamp is not None or (
            raw_features is not None and isinstance(raw_features.index, pd.DatetimeIndex)
        ):
            ts = bar_timestamp or raw_features.index[-1]
            if not self._in_tradeable_session(ts):
                return GoldSignal(action="HOLD", regime=GoldRegime.NO_TRADE)

        features = compute_indicators(raw_features, self._cfg)
        regime = self._regime_detector.detect(features)
        signal = self._signal_gen.generate(
            features,
            regime=regime,
            bar_timestamp=bar_timestamp or (
                features.index[-1] if isinstance(features.index, pd.DatetimeIndex)
                else None
            ),
        )

        if signal.is_actionable:
            logger.info(
                "GoldIntradayStrategy: {} signal — {} conf={:.2f} "
                "entry={:.2f} sl={:.2f} tp={:.2f} atr={:.2f} regime={}",
                signal.action,
                signal.signal_type.value,
                signal.confidence,
                signal.entry_ref_price,
                signal.stop_loss,
                signal.take_profit,
                signal.atr,
                regime.value,
            )

        return signal

    def notify_loss(self, signal_type=None, direction=None) -> None:
        """Propagate a stop-loss outcome to the signal generator."""
        self._signal_gen.notify_loss(signal_type=signal_type, direction=direction)

    # ── Session helpers ───────────────────────────────────────────────────────

    def _in_tradeable_session(self, ts: pd.Timestamp) -> bool:
        """Return True if the timestamp falls within a configured trading window."""
        sess = self._cfg.session
        try:
            local = ts.tz_convert("America/New_York")
        except Exception:
            local = ts
        hm = local.hour * 60 + local.minute

        def _parse(t: str) -> int:
            h, m = t.split(":")
            return int(h) * 60 + int(m)

        rth_open = _parse(sess.session_open_et)
        rth_close = _parse(sess.session_close_et) - sess.flatten_before_close_minutes

        in_rth = rth_open <= hm < rth_close
        if in_rth:
            return True

        if not sess.extended_hours_enabled:
            return False

        ext_open = _parse(sess.extended_session_open_et)
        # Extended = after ext_open (e.g. 18:00) OR before rth_open (e.g. 08:20)
        return hm >= ext_open or hm < rth_open
