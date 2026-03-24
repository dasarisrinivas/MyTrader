"""Gold market-regime classifier.

Inputs:  a pandas DataFrame of completed 1-min OHLCV bars with pre-computed
         indicator columns (ema9, ema21, adx, atr, vwap).
Output:  GoldRegime enum value.

Deliberately *stateless* — call `detect()` on each bar; no side-effects.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import pandas as pd

import numpy as np

from ...config.gold import GoldEntryConfig, GoldIndicatorConfig
from ...utils.logger import logger


class GoldRegime(str, Enum):
    """Market regime classification for the Gold intraday strategy."""

    TRENDING_BULL = "TRENDING_BULL"   # ADX in range, EMA9>EMA21, price>VWAP
    TRENDING_BEAR = "TRENDING_BEAR"   # ADX in range, EMA9<EMA21, price<VWAP
    RANGING = "RANGING"               # ADX below trend threshold
    NO_TRADE = "NO_TRADE"             # Chaotic volatility, outside session, or maintenance
    WARMING_UP = "WARMING_UP"         # Insufficient bars for reliable indicators


class GoldRegimeDetector:
    """Classify the current gold market regime from indicator columns.

    Expected DataFrame columns (all pre-computed, no lookahead):
        close, volume, ema9, ema21, adx, atr, vwap
    """

    # Required indicator columns
    _REQUIRED = {"close", "ema9", "ema21", "adx", "atr", "vwap"}

    def __init__(
        self,
        indicators: GoldIndicatorConfig,
        entry: GoldEntryConfig,
    ) -> None:
        self._ind = indicators
        self._entry = entry

    def detect(self, features: pd.DataFrame) -> GoldRegime:
        """Return the regime for the latest completed bar.

        Args:
            features: DataFrame of completed bars with indicator columns.
                      The *last row* is the most recent completed bar.

        Returns:
            GoldRegime value.
        """
        if len(features) < self._ind.warmup_bars:
            return GoldRegime.WARMING_UP

        missing = self._REQUIRED - set(features.columns)
        if missing:
            logger.warning("GoldRegimeDetector: missing columns {} — NO_TRADE", missing)
            return GoldRegime.NO_TRADE

        row = features.iloc[-1]

        # Sanity check for NaN indicators (can happen at indicator startup)
        if pd.isna(row[["ema9", "ema21", "adx", "atr", "vwap"]]).any():
            return GoldRegime.WARMING_UP

        close: float = float(row["close"])
        ema9: float = float(row["ema9"])
        ema21: float = float(row["ema21"])
        adx: float = float(row["adx"])
        atr: float = float(row["atr"])
        vwap: float = float(row["vwap"])

        # ── Volatility guard ─────────────────────────────────────────────────
        if close > 0:
            atr_ratio = atr / close
            if atr_ratio < self._entry.atr_min_ratio:
                logger.debug(
                    "GoldRegime: NO_TRADE — ATR ratio %.5f < min %.5f (flat/illiquid)",
                    atr_ratio,
                    self._entry.atr_min_ratio,
                )
                return GoldRegime.NO_TRADE
            if atr_ratio > self._entry.atr_max_ratio:
                logger.debug(
                    "GoldRegime: NO_TRADE — ATR ratio %.5f > max %.5f (chaotic)",
                    atr_ratio,
                    self._entry.atr_max_ratio,
                )
                return GoldRegime.NO_TRADE

        # ── ADX: ranging vs trending ──────────────────────────────────────────
        if adx < self._entry.adx_trend_min:
            return GoldRegime.RANGING
        if adx > self._entry.adx_trend_max:
            return GoldRegime.NO_TRADE   # Extreme trend / exhaustion spike

        # ── Trend direction: EMA alignment + VWAP relationship ───────────────
        bull_ema = ema9 > ema21
        bull_vwap = close > vwap

        # ── Phase 2 Gate 1: EMA spread ───────────────────────────────────────
        # Reject "trending" when the two EMAs are practically on top of each
        # other — ADX alone can be elevated during choppy expansion.
        ema_spread_ratio = abs(ema9 - ema21) / close if close > 0 else 0.0
        if ema_spread_ratio < self._ind.ema_spread_min_ratio:
            logger.debug(
                "GoldRegime: RANGING — EMA spread {:.6f} < min {:.6f}",
                ema_spread_ratio,
                self._ind.ema_spread_min_ratio,
            )
            return GoldRegime.RANGING

        # ── Phase 2 Gate 2: EMA slope ────────────────────────────────────────
        # The fast EMA must be *moving* in the expected direction.
        if self._ind.ema_slope_enabled and "ema9_slope" in features.columns:
            ema9_slope = float(features["ema9_slope"].iloc[-1])
            if not pd.isna(ema9_slope):
                min_slope = self._ind.ema_slope_min_per_bar
                if bull_ema and ema9_slope < min_slope:
                    logger.debug(
                        "GoldRegime: RANGING — bull EMA slope {:.4f} < min {:.4f}",
                        ema9_slope,
                        min_slope,
                    )
                    return GoldRegime.RANGING
                if not bull_ema and ema9_slope > -min_slope:
                    logger.debug(
                        "GoldRegime: RANGING — bear EMA slope {:.4f} > -{:.4f}",
                        ema9_slope,
                        min_slope,
                    )
                    return GoldRegime.RANGING

        # ── Phase 2 Gate 3: Price structure ──────────────────────────────────
        # Optional higher-high/higher-low (bull) or lower-high/lower-low (bear)
        # confirmation over recent bars.  Avoids labelling as trending when
        # price is actually making counter-trend swings.
        if self._ind.price_structure_enabled:
            lb = self._ind.price_structure_lookback_bars
            if len(features) > lb and lb >= 2:
                recent = features.iloc[-(lb + 1):]
                highs_arr = recent["high"].values
                lows_arr = recent["low"].values
                if bull_ema:
                    # Bull: latest high > earliest high AND latest low > earliest low
                    if highs_arr[-1] <= highs_arr[0] and lows_arr[-1] <= lows_arr[0]:
                        logger.debug(
                            "GoldRegime: RANGING — bull but no HH/HL structure"
                        )
                        return GoldRegime.RANGING
                else:
                    # Bear: latest high < earliest high AND latest low < earliest low
                    if highs_arr[-1] >= highs_arr[0] and lows_arr[-1] >= lows_arr[0]:
                        logger.debug(
                            "GoldRegime: RANGING — bear but no LH/LL structure"
                        )
                        return GoldRegime.RANGING

        # ── Multi-timeframe confirmation gate (runs before direction return) ─
        if self._ind.mtf_enabled and "htf_adx" in features.columns:
            htf_adx = row.get("htf_adx", np.nan)
            if not np.isnan(float(htf_adx) if htf_adx is not None else np.nan):
                if float(htf_adx) < self._ind.mtf_adx_min:
                    logger.debug(
                        "GoldRegime: RANGING — HTF ADX %.1f < min %.1f",
                        float(htf_adx), self._ind.mtf_adx_min,
                    )
                    return GoldRegime.RANGING
                if self._ind.mtf_ema_alignment_required:
                    htf_ema9 = row.get("htf_ema9", np.nan)
                    htf_ema21 = row.get("htf_ema21", np.nan)
                    if not (np.isnan(float(htf_ema9)) or np.isnan(float(htf_ema21))):
                        htf_bull = float(htf_ema9) > float(htf_ema21)
                        if htf_bull != bull_ema:
                            logger.debug(
                                "GoldRegime: RANGING — HTF EMA alignment disagrees with 1m"
                            )
                            return GoldRegime.RANGING

        if bull_ema and bull_vwap:
            return GoldRegime.TRENDING_BULL
        if not bull_ema and not bull_vwap:
            return GoldRegime.TRENDING_BEAR

        # EMA and VWAP disagree → treat as ranging / wait for alignment
        return GoldRegime.RANGING

    def is_tradeable(self, regime: GoldRegime) -> bool:
        """Return True if the regime allows entry signals to fire."""
        return regime in (GoldRegime.TRENDING_BULL, GoldRegime.TRENDING_BEAR)
