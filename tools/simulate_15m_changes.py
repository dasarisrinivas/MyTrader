#!/usr/bin/env python3
"""
Simulation Preview — 15m Strategy Parameter Changes
====================================================

Replays historical 1m bars → 15m resampled bars through the EsFifteenMinStrategy
under two configurations:

  A) CURRENT: ft_ema_touch_pct=0.001, 3-bar ascending, full hybrid oppose
  B) PROPOSED: ft_ema_touch_pct=0.0015, 2-bar ascending, hybrid oppose cap −0.10

Outputs:
  - Signals per day (by type)
  - Simulated trades (fixed SL/TP hit from subsequent bars)
  - Win rate, avg win, avg loss, total P&L, max drawdown
  - Side-by-side comparison table

Usage:
    python3 tools/simulate_15m_changes.py
"""

from __future__ import annotations
import sys, os, copy, json
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import pandas as pd
import numpy as np

# ── Add project root to path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shree.features.feature_engineer import engineer_features

# ── Resampler: 1m → 15m ─────────────────────────────────────────────

def resample_1m_to_15m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute bars to 15-minute bars."""
    df = df_1m.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    ohlcv = df[["open", "high", "low", "close", "volume"]].resample("15min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return ohlcv


# ── Lightweight signal checker (mirrors es_fifteen_min.py logic) ─────

@dataclass
class SimConfig:
    """Mirror of the subset of EsFifteenMinStrategy params we care about."""
    ema_touch_pct: float = 0.001       # Signal A touch tolerance
    ema9_touch_pct: float = 0.0015     # Signal C touch tolerance
    adx_min: float = 20.0
    adx_max: float = 35.0
    rth_start: time = field(default_factory=lambda: time(9, 30))
    rth_end: time = field(default_factory=lambda: time(16, 0))
    or_minutes: int = 30
    entry_start: time = field(default_factory=lambda: time(0, 0))
    entry_end: time = field(default_factory=lambda: time(23, 59))
    shorts_enabled: bool = True
    trend_cont_enabled: bool = True
    trend_cont_adx_min: float = 18.0
    trend_cont_max_per_day: int = 2
    trend_cont_max_ext_pts: float = 30.0
    trend_cont_gap_adx_min: float = 25.0
    # SL/TP
    fixed_sl_ab: float = 6.0
    fixed_tp_ab: float = 8.0
    fixed_sl_c: float = 8.0          # Legacy fallback (overridden by ATR-adaptive)
    fixed_tp_c: float = 10.0         # Legacy fallback (overridden by ATR-adaptive)
    fixed_sl_f: float = 8.0
    fixed_tp_f: float = 12.0
    # FEB 20 2026: ATR-adaptive stops for Signal C
    ema9_sl_atr_mult: float = 1.0    # ATR multiplier for Signal C stop
    ema9_sl_floor: float = 8.0       # Minimum SL (pts)
    ema9_sl_ceiling: float = 20.0    # Maximum SL (pts)
    ema9_rr_ratio: float = 1.25      # TP = SL × rr_ratio
    # ── Changed params ──
    ascending_bars: int = 3           # 3 for current, 2 for proposed
    hybrid_oppose_cap: float = -0.33  # current: unlimited; proposed: -0.10
    label: str = "CURRENT"


@dataclass
class SimSignal:
    bar_time: Any
    action: str           # BUY / SELL
    signal_type: str      # A / B / C / D / E / F_long / F_short
    entry_price: float
    stop_loss: float
    take_profit: float
    base_confidence: float = 0.70


@dataclass
class SimTrade:
    signal: SimSignal
    exit_price: float
    exit_reason: str      # PROFIT_TARGET / STOP_LOSS / TIMEOUT
    pnl: float
    bars_held: int


# ── Indicator computation (uses shree feature_engineer) ──────────────

def compute_indicators(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Add EMA9/21/50, RSI14, ATR14, ADX14, MACD to 15m bars."""
    from shree.features.feature_engineer import _ema, _atr, _adx, _rsi
    enriched = df_15m.copy()
    c = enriched["close"]
    h = enriched["high"]
    lo = enriched["low"]
    enriched["EMA_9"] = _ema(c, 9)
    enriched["EMA_21"] = _ema(c, 21)
    enriched["EMA_50"] = _ema(c, 50)
    enriched["RSI_14"] = _rsi(c, 14)
    enriched["ATR_14"] = _atr(h, lo, c, 14)
    enriched["ADX_14"] = _adx(h, lo, c, 14)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    enriched["MACDhist_12_26_9"] = macd_line - signal_line

    for col in ["EMA_9", "EMA_21", "EMA_50", "RSI_14", "ATR_14", "ADX_14", "MACDhist_12_26_9"]:
        enriched[col] = enriched[col].ffill().bfill()

    return enriched


# ── Signal evaluation engine ──────────────────────────────────────────

class SignalEngine:
    """Evaluates 15m bars for signals — parameterized by SimConfig."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self._or_high = 0.0
        self._or_low = 0.0
        self._or_computed = False
        self._or_break_long_count = 0
        self._or_break_short_count = 0
        self._or_break_max_per_day = 2
        self._opening_bars: list = []
        self._prev_close = 0.0
        self._session_date = None
        self._trend_cont_long_count = 0
        self._trend_cont_short_count = 0

    def _reset_session(self, date):
        self._session_date = date
        self._or_high = 0.0
        self._or_low = 0.0
        self._or_computed = False
        self._opening_bars = []
        self._or_break_long_count = 0
        self._or_break_short_count = 0
        self._prev_close = 0.0
        self._trend_cont_long_count = 0
        self._trend_cont_short_count = 0

    def evaluate(self, enriched: pd.DataFrame, idx: int) -> Optional[SimSignal]:
        """Evaluate bar at position idx in enriched DataFrame. Return signal or None."""
        if idx < 60:
            return None

        latest = enriched.iloc[idx]
        current_time = enriched.index[idx]

        # Convert to ET
        try:
            if current_time.tz is None:
                et_time = pd.Timestamp(current_time).tz_localize("UTC").tz_convert("US/Eastern")
            else:
                et_time = current_time.tz_convert("US/Eastern")
        except Exception:
            return None

        et_date = et_time.date()

        # Daily reset
        if self._session_date != et_date:
            self._reset_session(et_date)

        # OR collection
        or_end_dt = datetime.combine(datetime.today(), self.cfg.rth_start) + timedelta(minutes=self.cfg.or_minutes)
        or_end = or_end_dt.time()

        if self.cfg.rth_start <= et_time.time() < or_end:
            self._opening_bars.append({
                "high": float(latest["high"]),
                "low": float(latest["low"]),
                "close": float(latest["close"]),
            })
            self._prev_close = float(latest["close"])
            return None

        # Compute OR once
        if not self._or_computed and self._opening_bars:
            self._or_high = max(b["high"] for b in self._opening_bars)
            self._or_low = min(b["low"] for b in self._opening_bars)
            self._or_computed = True

        # RTH check
        if not (self.cfg.rth_start <= et_time.time() < self.cfg.rth_end):
            self._prev_close = float(latest["close"])
            return None

        # Entry window
        if not (self.cfg.entry_start <= et_time.time() < self.cfg.entry_end):
            self._prev_close = float(latest["close"])
            return None

        # Extract indicators
        close = float(latest["close"])
        open_p = float(latest.get("open", close))
        low = float(latest["low"])
        high = float(latest["high"])
        ema9 = float(latest.get("EMA_9", close))
        ema21 = float(latest.get("EMA_21", close))
        ema50 = float(latest.get("EMA_50", close))
        atr = float(latest.get("ATR_14", 10.0))
        adx = float(latest.get("ADX_14", 0.0))
        rsi = float(latest.get("RSI_14", 50.0))
        macd_hist = float(latest.get("MACDhist_12_26_9", 0.0))

        # Seed prev_close
        if self._prev_close == 0.0 and idx >= 1:
            self._prev_close = float(enriched.iloc[idx - 1]["close"])

        # Fix NaN
        for name in ["atr", "adx", "rsi", "macd_hist"]:
            val = locals()[name]
            if np.isnan(val):
                if name == "atr": atr = 10.0
                elif name == "adx": adx = 0.0
                elif name == "rsi": rsi = 50.0
                elif name == "macd_hist": macd_hist = 0.0

        if atr <= 0:
            self._prev_close = close
            return None

        df_slice = enriched.iloc[:idx + 1]

        # ── Signal A: EMA21 Pullback Long ──
        sig_a = self._check_a(close, open_p, low, ema21, ema50, atr, adx, rsi, macd_hist, et_time)
        # ── Signal C: EMA9 Pullback Long ──
        sig_c = self._check_c(close, open_p, low, ema9, ema21, ema50, atr, adx, rsi, macd_hist, et_time)
        # ── Signal B: OR Breakout Long ──
        sig_b = self._check_b(close, high, ema9, ema21, atr, adx, macd_hist, et_time)
        # ── Signal F Long: Trend Continuation ──
        sig_f_long = self._check_f_long(df_slice, close, open_p, low, high, ema9, ema21, ema50, atr, adx, rsi, macd_hist, et_time)
        # ── Signal D: EMA21 Pullback Short ──
        sig_d = self._check_d(close, open_p, high, ema21, ema50, atr, adx, rsi, macd_hist, et_time) if self.cfg.shorts_enabled else None
        # ── Signal E: OR Breakdown Short ──
        sig_e = self._check_e(close, low, ema9, ema21, atr, adx, macd_hist, et_time) if self.cfg.shorts_enabled else None
        # ── Signal F Short: Trend Continuation ──
        sig_f_short = None
        if self.cfg.shorts_enabled and self.cfg.trend_cont_enabled and sig_f_long is None:
            sig_f_short = self._check_f_short(df_slice, close, open_p, low, high, ema9, ema21, ema50, atr, adx, rsi, macd_hist, et_time)

        # Priority chain
        chosen = sig_a or sig_c or sig_b or sig_f_long or sig_d or sig_e or sig_f_short
        self._prev_close = close
        return chosen

    # ── Signal A ──
    def _check_a(self, close, open_p, low, ema21, ema50, atr, adx, rsi, macd_hist, et_time):
        if ema21 <= ema50: return None
        touch = ema21 * (1 + self.cfg.ema_touch_pct)
        if low > touch: return None
        if close <= ema21: return None
        if close <= open_p: return None
        if adx < self.cfg.adx_min or adx > self.cfg.adx_max: return None
        if macd_hist <= 0: return None
        if rsi > 70 or rsi < 35: return None
        sl = close - self.cfg.fixed_sl_ab
        tp = close + self.cfg.fixed_tp_ab
        return SimSignal(et_time, "BUY", "A", close, sl, tp)

    # ── Signal B ──
    def _check_b(self, close, high, ema9, ema21, atr, adx, macd_hist, et_time):
        if not self._or_computed or self._or_high <= 0: return None
        if self._or_break_long_count >= self._or_break_max_per_day: return None
        if not (close > self._or_high and self._prev_close <= self._or_high): return None
        if ema9 <= ema21: return None
        if adx < self.cfg.adx_min or adx > self.cfg.adx_max: return None
        if macd_hist <= 0: return None
        sl = close - self.cfg.fixed_sl_ab
        tp = close + self.cfg.fixed_tp_ab
        self._or_break_long_count += 1
        return SimSignal(et_time, "BUY", "B", close, sl, tp)

    # ── Signal C ──
    def _check_c(self, close, open_p, low, ema9, ema21, ema50, atr, adx, rsi, macd_hist, et_time):
        if not (ema9 > ema21 > ema50): return None
        touch = ema9 * (1 + self.cfg.ema9_touch_pct)
        if low > touch: return None
        if close <= ema9: return None
        if close <= open_p: return None
        if adx < 22 or adx > self.cfg.adx_max: return None
        if rsi > 70 or rsi < 40: return None
        if macd_hist <= 0: return None
        # Don't overlap with Signal A territory
        touch_ema21 = ema21 * (1 + self.cfg.ema_touch_pct)
        if low <= touch_ema21: return None
        # FEB 20 2026: ATR-adaptive stops
        sl_pts = min(self.cfg.ema9_sl_ceiling,
                     max(self.cfg.ema9_sl_floor, atr * self.cfg.ema9_sl_atr_mult))
        tp_pts = sl_pts * self.cfg.ema9_rr_ratio
        sl = close - sl_pts
        tp = close + tp_pts
        return SimSignal(et_time, "BUY", "C", close, sl, tp)

    # ── Signal D ──
    def _check_d(self, close, open_p, high, ema21, ema50, atr, adx, rsi, macd_hist, et_time):
        if ema21 >= ema50: return None
        touch = ema21 * (1 - self.cfg.ema_touch_pct)
        if high < touch: return None
        if close >= ema21: return None
        if close >= open_p: return None
        if adx < self.cfg.adx_min or adx > self.cfg.adx_max: return None
        if macd_hist >= 0: return None
        if rsi < 30 or rsi > 65: return None
        sl = close + self.cfg.fixed_sl_ab
        tp = close - self.cfg.fixed_tp_ab
        return SimSignal(et_time, "SELL", "D", close, sl, tp)

    # ── Signal E ──
    def _check_e(self, close, low, ema9, ema21, atr, adx, macd_hist, et_time):
        if not self._or_computed or self._or_low <= 0: return None
        if self._or_break_short_count >= self._or_break_max_per_day: return None
        if not (close < self._or_low and self._prev_close >= self._or_low): return None
        if ema9 >= ema21: return None
        if adx < self.cfg.adx_min or adx > self.cfg.adx_max: return None
        if macd_hist >= 0: return None
        sl = close + self.cfg.fixed_sl_ab
        tp = close - self.cfg.fixed_tp_ab
        self._or_break_short_count += 1
        return SimSignal(et_time, "SELL", "E", close, sl, tp)

    # ── Signal F Long ──
    def _check_f_long(self, df, close, open_p, low, high, ema9, ema21, ema50, atr, adx, rsi, macd_hist, et_time):
        if not self.cfg.trend_cont_enabled: return None
        if self._trend_cont_long_count >= self.cfg.trend_cont_max_per_day: return None
        if not (ema9 > ema21 > ema50): return None
        if close <= ema9: return None
        if close <= open_p: return None
        if adx < self.cfg.trend_cont_adx_min: return None
        if macd_hist <= 0: return None
        if rsi < 45 or rsi > 78: return None

        # Extension check
        spread = close - ema9
        max_ext = max(atr * 1.5, self.cfg.trend_cont_max_ext_pts)
        ema9_zone = ema9 * 0.999  # 0.1% proximity
        if low < ema9_zone:
            if spread > max_ext: return None
            if atr > 0 and spread > atr * 3 and adx < self.cfg.trend_cont_gap_adx_min: return None

        # Ascending closes — parameterized!
        n = self.cfg.ascending_bars
        if len(df) < n + 1:
            return None
        closes = [float(df.iloc[-(i + 1)]["close"]) for i in range(n)]  # [latest, latest-1, ...]
        ascending = all(closes[i] > closes[i + 1] for i in range(n - 1))
        if not ascending:
            return None

        sl = close - self.cfg.fixed_sl_f
        tp = close + self.cfg.fixed_tp_f
        self._trend_cont_long_count += 1
        return SimSignal(et_time, "BUY", "F_long", close, sl, tp)

    # ── Signal F Short ──
    def _check_f_short(self, df, close, open_p, low, high, ema9, ema21, ema50, atr, adx, rsi, macd_hist, et_time):
        if not self.cfg.trend_cont_enabled: return None
        if self._trend_cont_short_count >= self.cfg.trend_cont_max_per_day: return None
        if not (ema9 < ema21 < ema50): return None
        if close >= ema9: return None
        if close >= open_p: return None
        if adx < self.cfg.trend_cont_adx_min: return None
        if macd_hist >= 0: return None
        if rsi > 55 or rsi < 22: return None

        # Extension check (inverted)
        spread = ema9 - close
        max_ext = max(atr * 1.5, self.cfg.trend_cont_max_ext_pts)
        ema9_zone = ema9 * 1.001
        if high > ema9_zone:
            if spread > max_ext: return None
            if atr > 0 and spread > atr * 3 and adx < self.cfg.trend_cont_gap_adx_min: return None

        # Descending closes — parameterized
        n = self.cfg.ascending_bars
        if len(df) < n + 1:
            return None
        closes = [float(df.iloc[-(i + 1)]["close"]) for i in range(n)]
        descending = all(closes[i] < closes[i + 1] for i in range(n - 1))
        if not descending:
            return None

        sl = close + self.cfg.fixed_sl_f
        tp = close - self.cfg.fixed_tp_f
        self._trend_cont_short_count += 1
        return SimSignal(et_time, "SELL", "F_short", close, sl, tp)


# ── Trade simulator: walk-forward SL/TP resolution ──────────────────

def simulate_trades(
    signals: List[SimSignal],
    enriched: pd.DataFrame,
    slippage_pts: float = 0.50,    # 2 ticks slippage
    commission: float = 2.48,       # RT commission
    timeout_bars: int = 20,         # Exit after 20 bars (5 hours) if neither SL/TP hit
    hybrid_oppose_cap: float = -0.33, # Max confidence dampening (current=-0.20, proposed=-0.10)
    oppose_rate: float = 0.50,      # Fraction of signals that get hybrid opposition
    vx_multiplier: float = 0.88,    # Average VX scaling factor
    conf_threshold: float = 0.50,   # Minimum confidence to trade
) -> List[SimTrade]:
    """Walk forward from each signal to resolve SL/TP hit."""
    trades = []

    # Deterministic "oppose" assignment based on signal index
    # Every other signal gets opposed (simulates 50% oppose rate)
    for sig_idx, sig in enumerate(signals):
        base_conf = sig.base_confidence
        vx_scaled = base_conf * vx_multiplier

        # Determine if this signal gets opposed
        is_opposed = (sig_idx % int(1.0 / oppose_rate) == 0) if oppose_rate > 0 else False

        if hybrid_oppose_cap != 0.0 and is_opposed:
            final_conf = vx_scaled + hybrid_oppose_cap
        else:
            final_conf = vx_scaled

        if final_conf < conf_threshold:
            trades.append(SimTrade(
                signal=sig, exit_price=sig.entry_price,
                exit_reason="BLOCKED_HYBRID", pnl=0.0, bars_held=0,
            ))
            continue

        # Find bar index for this signal's timestamp
        # Find nearest index
        bar_idx = None
        for i, ts in enumerate(enriched.index):
            try:
                if ts.tz is not None:
                    et_ts = ts.tz_convert("US/Eastern")
                else:
                    et_ts = ts
                if et_ts == sig.bar_time or (hasattr(sig.bar_time, 'tz') and ts == sig.bar_time):
                    bar_idx = i
                    break
            except:
                pass

        if bar_idx is None:
            # Try fuzzy match (within 1 minute)
            for i, ts in enumerate(enriched.index):
                try:
                    diff = abs((ts - sig.bar_time).total_seconds()) if hasattr(sig.bar_time, 'total_seconds') else 999
                except:
                    try:
                        ts_et = ts.tz_convert("US/Eastern") if ts.tz else ts.tz_localize("UTC").tz_convert("US/Eastern")
                        sig_ts = sig.bar_time if hasattr(sig.bar_time, 'date') else pd.Timestamp(sig.bar_time)
                        if hasattr(sig_ts, 'tz') and sig_ts.tz is not None:
                            diff = abs((ts_et - sig_ts).total_seconds())
                        else:
                            diff = 999
                    except:
                        diff = 999
                if diff < 120:
                    bar_idx = i
                    break

        if bar_idx is None:
            continue

        # Apply slippage
        is_long = sig.action == "BUY"
        entry = sig.entry_price + (slippage_pts if is_long else -slippage_pts)
        sl = sig.stop_loss
        tp = sig.take_profit

        # Walk forward
        exit_price = entry
        exit_reason = "TIMEOUT"
        bars_held = 0

        for j in range(bar_idx + 1, min(bar_idx + 1 + timeout_bars, len(enriched))):
            bar = enriched.iloc[j]
            bars_held += 1
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])

            if is_long:
                # Check SL first (worst-case)
                if bar_low <= sl:
                    exit_price = sl - slippage_pts  # Slippage on exit
                    exit_reason = "STOP_LOSS"
                    break
                if bar_high >= tp:
                    exit_price = tp - slippage_pts  # Conservative fill
                    exit_reason = "PROFIT_TARGET"
                    break
            else:
                # Short
                if bar_high >= sl:
                    exit_price = sl + slippage_pts
                    exit_reason = "STOP_LOSS"
                    break
                if bar_low <= tp:
                    exit_price = tp + slippage_pts
                    exit_reason = "PROFIT_TARGET"
                    break

        # P&L calculation
        if is_long:
            pnl = (exit_price - entry) * 5.0  # $5 per point MES
        else:
            pnl = (entry - exit_price) * 5.0
        pnl -= commission  # Round-trip commission

        trades.append(SimTrade(signal=sig, exit_price=exit_price, exit_reason=exit_reason, pnl=pnl, bars_held=bars_held))

    return trades


# ── Reporting ──────────────────────────────────────────────────────────

def analyze_results(trades: List[SimTrade], label: str, total_days: int) -> Dict:
    """Compute statistics from simulated trades."""
    executed = [t for t in trades if t.exit_reason != "BLOCKED_HYBRID"]
    blocked = [t for t in trades if t.exit_reason == "BLOCKED_HYBRID"]
    wins = [t for t in executed if t.pnl > 0]
    losses = [t for t in executed if t.pnl <= 0]

    total_pnl = sum(t.pnl for t in executed)
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
    win_rate = len(wins) / len(executed) * 100 if executed else 0

    # Max drawdown
    equity = []
    running = 0
    for t in executed:
        running += t.pnl
        equity.append(running)
    peak = 0
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = peak - e
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    # Per signal type
    by_type = {}
    for t in trades:
        st = t.signal.signal_type
        if st not in by_type:
            by_type[st] = {"signals": 0, "executed": 0, "blocked": 0, "wins": 0, "losses": 0, "pnl": 0.0}
        by_type[st]["signals"] += 1
        if t.exit_reason == "BLOCKED_HYBRID":
            by_type[st]["blocked"] += 1
        else:
            by_type[st]["executed"] += 1
            if t.pnl > 0:
                by_type[st]["wins"] += 1
            else:
                by_type[st]["losses"] += 1
            by_type[st]["pnl"] += t.pnl

    return {
        "label": label,
        "total_signals": len(trades),
        "blocked": len(blocked),
        "executed": len(executed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_pnl": total_pnl,
        "max_drawdown": max_dd,
        "profit_factor": pf,
        "trades_per_day": len(executed) / total_days if total_days > 0 else 0,
        "signals_per_day": len(trades) / total_days if total_days > 0 else 0,
        "by_type": by_type,
    }


def print_comparison(current: Dict, proposed: Dict):
    """Print side-by-side comparison."""
    print("\n" + "=" * 80)
    print("  SIMULATION PREVIEW — 15m Strategy Parameter Changes")
    print("=" * 80)
    print(f"\n  Data: October 2025 MES 1m bars → 15m resampled")
    print(f"  Slippage: 0.50 pts (2 ticks) per side | Commission: $2.48 RT")
    print(f"  Timeout: 20 bars (5 hours) if SL/TP not hit")

    print(f"\n  {'Metric':<30} {'CURRENT':>15} {'PROPOSED':>15} {'CHANGE':>15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*15}")

    def row(name, cur, pro, fmt=".1f", suffix="", better_higher=True):
        cur_s = f"{cur:{fmt}}{suffix}"
        pro_s = f"{pro:{fmt}}{suffix}"
        delta = pro - cur
        sign = "+" if delta >= 0 else ""
        if fmt == ".0f":
            delta_s = f"{sign}{delta:.0f}{suffix}"
        elif fmt == ".1f":
            delta_s = f"{sign}{delta:.1f}{suffix}"
        else:
            delta_s = f"{sign}{delta:{fmt}}{suffix}"
        # Color hint
        good = (delta > 0 and better_higher) or (delta < 0 and not better_higher)
        marker = " ✅" if good and delta != 0 else (" ⚠️" if delta != 0 else "")
        print(f"  {name:<30} {cur_s:>15} {pro_s:>15} {delta_s:>12}{marker}")

    row("Total signals generated", current["total_signals"], proposed["total_signals"], ".0f")
    row("Blocked by hybrid", current["blocked"], proposed["blocked"], ".0f", better_higher=False)
    row("Trades executed", current["executed"], proposed["executed"], ".0f")
    row("Trades per day", current["trades_per_day"], proposed["trades_per_day"], ".2f")
    row("Win rate", current["win_rate"], proposed["win_rate"], ".1f", "%")
    row("Avg win ($)", current["avg_win"], proposed["avg_win"], ".2f")
    row("Avg loss ($)", current["avg_loss"], proposed["avg_loss"], ".2f", better_higher=False)
    row("Total P&L ($)", current["total_pnl"], proposed["total_pnl"], ".2f")
    row("Max drawdown ($)", current["max_drawdown"], proposed["max_drawdown"], ".2f", better_higher=False)
    row("Profit factor", current["profit_factor"], proposed["profit_factor"], ".2f")

    # Win rate degradation check
    wr_delta = proposed["win_rate"] - current["win_rate"]
    print(f"\n  📊 Win rate change: {wr_delta:+.1f}%", end="")
    if abs(wr_delta) <= 5:
        print(f"  ✅ Within acceptable range (±5%)")
    else:
        print(f"  ⚠️ Exceeds 5% threshold — review before deploying")

    # Per signal type
    print(f"\n  {'Signal':<12} {'CURRENT':>36} {'PROPOSED':>36}")
    print(f"  {'':12} {'sig  exec  blk  W   L    P&L':>36} {'sig  exec  blk  W   L    P&L':>36}")
    print(f"  {'-'*12} {'-'*36} {'-'*36}")
    all_types = sorted(set(list(current["by_type"].keys()) + list(proposed["by_type"].keys())))
    for st in all_types:
        c = current["by_type"].get(st, {"signals": 0, "executed": 0, "blocked": 0, "wins": 0, "losses": 0, "pnl": 0})
        p = proposed["by_type"].get(st, {"signals": 0, "executed": 0, "blocked": 0, "wins": 0, "losses": 0, "pnl": 0})
        c_s = f"{c['signals']:3d}  {c['executed']:3d}  {c['blocked']:3d}  {c['wins']:2d}  {c['losses']:2d}  ${c['pnl']:>7.2f}"
        p_s = f"{p['signals']:3d}  {p['executed']:3d}  {p['blocked']:3d}  {p['wins']:2d}  {p['losses']:2d}  ${p['pnl']:>7.2f}"
        print(f"  {st:<12} {c_s:>36} {p_s:>36}")

    print(f"\n{'='*80}")
    print(f"  PARAMETER CHANGES:")
    print(f"    1. ft_ema_touch_pct:  0.001  → 0.0015  (Signal A tolerance +50%)")
    print(f"    2. ascending_bars:    3      → 2       (Signal F relaxed)")
    print(f"    3. hybrid_oppose_cap: -0.33  → -0.10   (Hybrid dampen capped)")
    print(f"{'='*80}\n")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    csv_path = ROOT / "data" / "es_october_2025.csv"
    if not csv_path.exists():
        # Fallback
        csv_path = ROOT / "data" / "es_2025-10-27_to_2025-10-31.csv"
    df_1m = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")

    print(f"  Loaded {len(df_1m)} 1m bars: {df_1m.index[0]} → {df_1m.index[-1]}")

    print("Resampling to 15m...")
    df_15m = resample_1m_to_15m(df_1m)
    print(f"  {len(df_15m)} 15m bars")

    print("Computing indicators...")
    enriched = compute_indicators(df_15m)
    print(f"  Indicators computed (EMA9/21/50, RSI14, ATR14, ADX14, MACD)")

    # Count trading days
    trading_dates = set()
    for ts in enriched.index:
        try:
            et = ts.tz_convert("US/Eastern") if ts.tz else ts.tz_localize("UTC").tz_convert("US/Eastern")
            if time(9, 30) <= et.time() < time(16, 0):
                trading_dates.add(et.date())
        except:
            pass
    total_days = len(trading_dates)
    print(f"  Trading days: {total_days}")

    # ── Config A: CURRENT params ──
    cfg_current = SimConfig(
        ema_touch_pct=0.001,
        ascending_bars=3,
        hybrid_oppose_cap=-0.33,  # Current: full oppose
        label="CURRENT",
    )

    # ── Config B: PROPOSED params ──
    cfg_proposed = SimConfig(
        ema_touch_pct=0.0015,     # Widened from 0.001 → 0.0015
        ascending_bars=2,          # Relaxed from 3 → 2
        hybrid_oppose_cap=-0.10,   # Capped oppose
        label="PROPOSED",
    )

    # ── Run CURRENT (raw signals, no hybrid blocking — to see raw quality) ──
    print("\n▶ Running CURRENT config simulation...")
    engine_cur = SignalEngine(cfg_current)
    signals_cur = []
    for i in range(len(enriched)):
        sig = engine_cur.evaluate(enriched, i)
        if sig:
            signals_cur.append(sig)
    print(f"  → {len(signals_cur)} raw signals generated")

    # Model hybrid blocking:
    # In live, ~40% of signals get hybrid opposition (observed from today: 1/1 = 100%, 
    # but historically it's variable). We'll model the oppose deterministically:
    #   - VX scaling: 0.70 × 0.81 = 0.567 (worst), 0.70 × 0.95 = 0.665 (normal VIX)
    #   - Hybrid oppose: ~50% of signals get opposed (pipeline disagrees)
    #   - Current oppose penalty: -0.198 (observed today, midpoint ~-0.20)
    #   - If VX_scaled + oppose < 0.50 threshold → BLOCKED
    # 
    # Current worst case: 0.70 × 0.81 = 0.567 - 0.20 = 0.367 → BLOCKED
    # Current normal VIX: 0.70 × 0.95 = 0.665 - 0.20 = 0.465 → BLOCKED  
    # Current calm VIX:   0.70 × 1.00 = 0.700 - 0.20 = 0.500 → BORDERLINE
    #
    # Proposed:           0.70 × 0.81 = 0.567 - 0.10 = 0.467 → BLOCKED (high VIX)
    # Proposed normal:    0.70 × 0.95 = 0.665 - 0.10 = 0.565 → ALLOWED ✅
    # Proposed calm:      0.70 × 1.00 = 0.700 - 0.10 = 0.600 → ALLOWED ✅

    # For realistic modeling, we'll simulate three scenarios:
    #   1. RAW (no hybrid blocking) — pure signal quality
    #   2. CURRENT blocking (50% oppose rate × -0.20 penalty; VX mix of 0.85 avg)
    #   3. PROPOSED blocking (50% oppose rate × -0.10 penalty cap; same VX)

    # Scenario 1: RAW — no blocking at all
    trades_cur_raw = simulate_trades(
        signals_cur, enriched,
        hybrid_oppose_cap=0.0,  # No oppose → all pass
    )

    # Scenario 2: Current blocking — realistic oppose model
    # VX avg multiplier ~0.88, oppose rate ~50%, oppose penalty ~-0.20
    # conf = 0.70 × 0.88 = 0.616; opposed: 0.616 - 0.20 = 0.416 → BLOCKED
    # un-opposed: 0.616 → ALLOWED
    # Net: ~50% pass (un-opposed), ~50% blocked (opposed)
    trades_cur = simulate_trades(
        signals_cur, enriched,
        hybrid_oppose_cap=-0.20,  # Current observed oppose penalty
    )

    # ── Run PROPOSED ──
    print("\n▶ Running PROPOSED config simulation...")
    engine_pro = SignalEngine(cfg_proposed)
    signals_pro = []
    for i in range(len(enriched)):
        sig = engine_pro.evaluate(enriched, i)
        if sig:
            signals_pro.append(sig)
    print(f"  → {len(signals_pro)} raw signals generated")

    # Scenario 3: Proposed blocking — oppose capped at -0.10
    # conf = 0.70 × 0.88 = 0.616; opposed: 0.616 - 0.10 = 0.516 → ALLOWED ✅
    # Net: nearly all pass (even opposed signals survive)
    trades_pro = simulate_trades(
        signals_pro, enriched,
        hybrid_oppose_cap=-0.10,  # Proposed: capped
    )

    # Also run proposed raw for signal count comparison
    trades_pro_raw = simulate_trades(
        signals_pro, enriched,
        hybrid_oppose_cap=0.0,
    )

    # ── Analyze & compare ──
    results_cur_raw = analyze_results(trades_cur_raw, "CURRENT (raw)", total_days)
    results_cur = analyze_results(trades_cur, "CURRENT (w/ hybrid)", total_days)
    results_pro = analyze_results(trades_pro, "PROPOSED (w/ hybrid cap)", total_days)
    results_pro_raw = analyze_results(trades_pro_raw, "PROPOSED (raw)", total_days)

    # ── Print raw signal quality first ──
    print("\n" + "=" * 80)
    print("  PART 1: RAW SIGNAL QUALITY (no hybrid blocking)")
    print("=" * 80)
    print_comparison(results_cur_raw, results_pro_raw)

    # ── Print realistic comparison ──
    print("\n" + "=" * 80)
    print("  PART 2: REALISTIC (with hybrid oppose modeled)")
    print(f"    Current:  50% oppose rate × -0.20 penalty → most signals BLOCKED at 0.50 threshold")
    print(f"    Proposed: 50% oppose rate × -0.10 cap     → most signals PASS")
    print("=" * 80)
    print_comparison(results_cur, results_pro)

    # ── Print per-signal trade list for proposed ──
    print("\n" + "-" * 80)
    print("  PROPOSED — Individual Trade Outcomes (with hybrid cap)")
    print("-" * 80)
    executed_pro = [t for t in trades_pro if t.exit_reason != "BLOCKED_HYBRID"]
    for t in executed_pro:
        action = t.signal.action
        sig_type = t.signal.signal_type
        entry = t.signal.entry_price
        exit_p = t.exit_price
        reason = t.exit_reason
        pnl = t.pnl
        marker = "✅" if pnl > 0 else ("❌" if pnl < 0 else "➖")
        print(f"  {t.signal.bar_time} | {action} {sig_type:7s} @ {entry:.2f} → {exit_p:.2f} | {reason:14s} | ${pnl:>7.2f} {marker}")

    # Save detailed results
    out_path = ROOT / "artifacts" / "sim_preview_15m_changes.json"
    out_path.parent.mkdir(exist_ok=True)
    out = {
        "generated_at": datetime.now().isoformat(),
        "data_source": str(csv_path),
        "trading_days": total_days,
        "bars_15m": len(enriched),
        "current_raw": {k: v for k, v in results_cur_raw.items() if k != "by_type"},
        "current_hybrid": {k: v for k, v in results_cur.items() if k != "by_type"},
        "proposed_hybrid": {k: v for k, v in results_pro.items() if k != "by_type"},
        "proposed_raw": {k: v for k, v in results_pro_raw.items() if k != "by_type"},
        "current_by_type": results_cur_raw["by_type"],
        "proposed_by_type": results_pro_raw["by_type"],
        "changes": {
            "ema_touch_pct": "0.001 → 0.0015",
            "ascending_bars": "3 → 2",
            "hybrid_oppose_cap": "-0.33 → -0.10",
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Detailed results saved to {out_path}")


if __name__ == "__main__":
    main()
