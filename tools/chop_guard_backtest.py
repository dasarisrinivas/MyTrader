#!/usr/bin/env python3
"""
CHOP Guard — 12-Month Strategy Comparison Backtest
====================================================

Compares three CHOP guard strategies across 12 months of 15m data:

  A) Block-All (current default)
     → Block ALL pullback/trend_cont in CHOP. PnL = $0 (no trades taken).

  B) Exception Framework (5-gate filter, OFF by default)
     → LONG only + ADX >= 25 + BULLISH bias + conf >= 0.70 + ATR expanding.
     → Dampened by 0.05.

  C) Direction-Aware Baseline (REVERTED — the old MAR 2 2026 approach)
     → Allow ALL LONGs in CHOP (dampen 0.10), block SHORTs.
     → Shown to be net negative in year-long data.

Metrics (per strategy):
  1. Net PnL ($)
  2. Expectancy ($/trade)
  3. Max drawdown ($)
  4. Trade count
  5. Sharpe ratio (on per-trade PnL)
  6. Win rate (%)
  7. % of CHOP trades taken (vs total blocked)
  8. Stability (quarterly edge count ≥ 3/4 profitable quarters)

Validation Rules (for any strategy to be considered production-ready):
  - Expectancy > 0
  - Net PnL improvement > 1.5R vs block-all
  - Max DD increase < 10% of total capital ($5K)
  - Edge persists in ≥ 3 independent quarters
  - Sample ≥ 50 trades

Data: data/ib/ES_15m_1y.parquet (25,646 bars, Dec 2024 – Jan 2026)
"""

from __future__ import annotations

import sys
from datetime import datetime, time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shree.features.feature_engineer import add_technical_indicators
from shree.config import OneMinuteStrategyConfig
from shree.strategies.es_fifteen_min import EsFifteenMinStrategy


# ──────────────────────────────────────────────────────────────
# 1. Hybrid trend label (same as chop_guard_year_analysis.py)
# ──────────────────────────────────────────────────────────────

def compute_hybrid_trend(row: pd.Series, chop_ema_spread_min_pct: float = 0.0005) -> Tuple[str, float]:
    price = row["close"]
    ema_9 = row["EMA_9"]
    ema_20 = row["EMA_20"]
    ema_50 = row["EMA_50"]
    rsi = row["RSI_14"]
    macd_hist = row["MACDhist_12_26_9"]
    adx = row["ADX_14"]

    ema_spread = abs(ema_9 - ema_50) / ema_50 if ema_50 > 0 else 0
    if ema_spread < chop_ema_spread_min_pct:
        return "CHOP_RANGE", 0.0

    is_trending = adx > 20

    ema_diff_pct = (ema_9 - ema_20) / ema_20 * 100 if ema_20 > 0 else 0
    price_vs_ema20_pct = (price - ema_20) / ema_20 * 100 if ema_20 > 0 else 0
    price_vs_ema50_pct = (price - ema_50) / ema_50 * 100 if ema_50 > 0 else 0

    trend_score = 0.0
    # EMA alignment (40%)
    if price > ema_9 > ema_20 > ema_50:
        trend_score += 40
    elif price < ema_9 < ema_20 < ema_50:
        trend_score -= 40
    elif price > ema_20 and ema_diff_pct > 0:
        trend_score += 20
    elif price < ema_20 and ema_diff_pct < 0:
        trend_score -= 20

    # EMA_50 anchor (20%)
    if price_vs_ema50_pct > 0.1:
        trend_score += 20
    elif price_vs_ema50_pct < -0.1:
        trend_score -= 20

    # MACD (15%)
    if macd_hist > 0:
        trend_score += 15
    elif macd_hist < 0:
        trend_score -= 15

    # RSI (10%)
    if rsi > 60:
        trend_score += 10
    elif rsi < 40:
        trend_score -= 10
    elif rsi > 50:
        trend_score += 5
    elif rsi < 50:
        trend_score -= 5

    if trend_score >= 60 and is_trending:
        return "UPTREND", trend_score
    elif trend_score <= -60 and is_trending:
        return "DOWNTREND", trend_score
    elif trend_score >= 30 and is_trending:
        return "MICRO_UP", trend_score
    elif trend_score <= -30 and is_trending:
        return "MICRO_DOWN", trend_score
    elif trend_score >= 10 and is_trending:
        return "WEAK_UP", trend_score
    elif trend_score <= -10 and is_trending:
        return "WEAK_DOWN", trend_score
    elif abs(ema_diff_pct) < 0.02 and abs(price_vs_ema20_pct) < 0.05:
        return "RANGE", trend_score
    else:
        return "CHOP", trend_score


# ──────────────────────────────────────────────────────────────
# 2. Trade simulation
# ──────────────────────────────────────────────────────────────

@dataclass
class SimTrade:
    bar_idx: int
    timestamp: str
    signal_type: str
    action: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    adx: float
    rsi: float
    trend_label: str
    atr_current: float = 0.0
    atr_5ago: float = 0.0
    daily_bias: str = "NEUTRAL"  # simulated for backtest
    quarter: str = ""
    outcome: str = ""
    exit_price: float = 0.0
    pnl_points: float = 0.0
    pnl_dollars: float = 0.0
    bars_held: int = 0
    strategy_a: bool = False  # would be taken by strategy A (block-all: always False)
    strategy_b: bool = False  # would be taken by strategy B (exception framework)
    strategy_b_conf: float = 0.0
    strategy_c: bool = False  # would be taken by strategy C (direction-aware)
    strategy_c_conf: float = 0.0


def simulate_trade(trade: SimTrade, df: pd.DataFrame, max_hold_bars: int = 6) -> SimTrade:
    start_idx = trade.bar_idx + 1
    end_idx = min(start_idx + max_hold_bars, len(df))

    for i in range(start_idx, end_idx):
        bar = df.iloc[i]
        trade.bars_held = i - trade.bar_idx

        if trade.direction == "LONG":
            if bar["low"] <= trade.stop_loss:
                trade.outcome = "SL_HIT"
                trade.exit_price = trade.stop_loss
                break
            if bar["high"] >= trade.take_profit:
                trade.outcome = "TP_HIT"
                trade.exit_price = trade.take_profit
                break
        else:
            if bar["high"] >= trade.stop_loss:
                trade.outcome = "SL_HIT"
                trade.exit_price = trade.stop_loss
                break
            if bar["low"] <= trade.take_profit:
                trade.outcome = "TP_HIT"
                trade.exit_price = trade.take_profit
                break

    if not trade.outcome:
        trade.outcome = "TIMEOUT"
        if start_idx < len(df):
            last_idx = min(end_idx - 1, len(df) - 1)
            trade.exit_price = df.iloc[last_idx]["close"]
        else:
            trade.exit_price = trade.entry_price
        trade.bars_held = max_hold_bars

    if trade.direction == "LONG":
        trade.pnl_points = trade.exit_price - trade.entry_price
    else:
        trade.pnl_points = trade.entry_price - trade.exit_price
    trade.pnl_dollars = trade.pnl_points * 5.0

    return trade


# ──────────────────────────────────────────────────────────────
# 3. Strategy filter simulation
# ──────────────────────────────────────────────────────────────

def simulate_daily_bias(df: pd.DataFrame, bar_idx: int) -> str:
    """
    Simulate a daily sentiment bias from price context.
    Uses prior-day close vs 20-bar EMA as a rough proxy:
      - close > EMA_20 + 0.2% → BULLISH
      - close < EMA_20 - 0.2% → BEARISH
      - else → NEUTRAL
    This is a simplification since we don't have live sentiment
    feeds in backtesting. Errs conservative (fewer BULLISH labels).
    """
    if bar_idx < 20:
        return "NEUTRAL"
    row = df.iloc[bar_idx]
    ema20 = row["EMA_20"]
    close = row["close"]
    if ema20 <= 0:
        return "NEUTRAL"
    pct = (close - ema20) / ema20 * 100
    if pct > 0.2:
        return "BULLISH"
    elif pct < -0.2:
        return "BEARISH"
    return "NEUTRAL"


def classify_trade(trade: SimTrade) -> None:
    """
    Apply strategy filters to determine which strategies would take this trade.
    Strategy A: Block-all → never take any CHOP-blocked trade
    Strategy B: Exception framework → 5-gate filter
    Strategy C: Direction-aware → LONGs pass (dampen 0.10), SHORTs block
    """
    is_long = trade.direction == "LONG"

    # Strategy A: always blocked
    trade.strategy_a = False

    # Strategy C: direction-aware (old approach)
    if is_long:
        trade.strategy_c = True
        trade.strategy_c_conf = max(0.15, trade.confidence - 0.10)
    else:
        trade.strategy_c = False
        trade.strategy_c_conf = 0.0

    # Strategy B: 5-gate exception framework
    gate_1_direction = is_long
    gate_2_adx = trade.adx >= 25.0
    gate_3_bias = trade.daily_bias == "BULLISH"
    gate_4_conf = trade.confidence >= 0.70
    gate_5_atr = trade.atr_current > trade.atr_5ago and trade.atr_5ago > 0

    trade.strategy_b = (gate_1_direction and gate_2_adx and gate_3_bias
                        and gate_4_conf and gate_5_atr)
    if trade.strategy_b:
        trade.strategy_b_conf = max(0.15, trade.confidence - 0.05)
    else:
        trade.strategy_b_conf = 0.0


# ──────────────────────────────────────────────────────────────
# 4. Metrics computation
# ──────────────────────────────────────────────────────────────

@dataclass
class StrategyMetrics:
    name: str
    net_pnl: float = 0.0
    expectancy: float = 0.0
    max_drawdown: float = 0.0
    trade_count: int = 0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    pct_chop_trades_taken: float = 0.0
    profitable_quarters: int = 0
    total_quarters: int = 0
    quarterly_detail: Dict[str, float] = field(default_factory=dict)
    # Validation
    passes_expectancy: bool = False
    passes_pnl_improvement: bool = False
    passes_max_dd: bool = False
    passes_quarterly: bool = False
    passes_sample_size: bool = False
    recommendation: str = ""


def compute_metrics(trades: List[SimTrade], strategy_key: str, name: str,
                    total_blocked: int) -> StrategyMetrics:
    m = StrategyMetrics(name=name)

    # Filter to trades taken by this strategy
    if strategy_key == "a":
        taken = []  # Block-all: no trades
    elif strategy_key == "b":
        taken = [t for t in trades if t.strategy_b]
    elif strategy_key == "c":
        taken = [t for t in trades if t.strategy_c]
    else:
        taken = []

    m.trade_count = len(taken)
    m.pct_chop_trades_taken = (len(taken) / total_blocked * 100) if total_blocked > 0 else 0.0

    if len(taken) == 0:
        m.recommendation = "No trades taken — $0 baseline"
        return m

    pnls = [t.pnl_dollars for t in taken]
    pnl_arr = np.array(pnls)

    m.net_pnl = float(pnl_arr.sum())
    m.expectancy = float(pnl_arr.mean())
    m.win_rate = sum(1 for t in taken if t.pnl_dollars > 0) / len(taken) * 100

    # Max drawdown
    cum_pnl = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(cum_pnl)
    drawdowns = cum_pnl - peak
    m.max_drawdown = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0

    # Sharpe ratio (annualized from per-trade returns)
    if len(pnl_arr) > 1 and pnl_arr.std() > 0:
        # Approximate: assume ~4 trades/week, ~52 weeks
        trades_per_year = min(len(taken), 208)
        m.sharpe_ratio = float(pnl_arr.mean() / pnl_arr.std() * np.sqrt(trades_per_year))
    else:
        m.sharpe_ratio = 0.0

    # Quarterly breakdown
    quarters = {}
    for t in taken:
        q = t.quarter
        if q not in quarters:
            quarters[q] = []
        quarters[q].append(t.pnl_dollars)

    m.total_quarters = len(quarters)
    m.profitable_quarters = 0
    for q_label in sorted(quarters.keys()):
        q_pnl = sum(quarters[q_label])
        m.quarterly_detail[q_label] = q_pnl
        if q_pnl > 0:
            m.profitable_quarters += 1

    return m


def apply_validation(m: StrategyMetrics, r_value: float = 40.0, capital: float = 5000.0):
    """Apply the 5 validation rules."""
    # R value: average risk per trade = SL x $5/pt ≈ 6pts x $5 = $30.
    # Using $40 as approximate (includes ATR-adaptive stops).
    m.passes_expectancy = m.expectancy > 0
    m.passes_pnl_improvement = m.net_pnl > 1.5 * r_value  # > $60
    m.passes_max_dd = abs(m.max_drawdown) < capital * 0.10  # < $500
    m.passes_quarterly = m.profitable_quarters >= 3
    m.passes_sample_size = m.trade_count >= 50

    all_pass = (m.passes_expectancy and m.passes_pnl_improvement
                and m.passes_max_dd and m.passes_quarterly
                and m.passes_sample_size)

    if m.trade_count == 0:
        m.recommendation = "BASELINE — $0 (no trades)"
    elif all_pass:
        m.recommendation = "✅ VALIDATED — all 5 criteria met"
    else:
        failed = []
        if not m.passes_expectancy:
            failed.append(f"Expectancy={m.expectancy:+.2f}≤0")
        if not m.passes_pnl_improvement:
            failed.append(f"PnL=${m.net_pnl:+.0f}<1.5R(${1.5*40:.0f})")
        if not m.passes_max_dd:
            failed.append(f"DD=${m.max_drawdown:.0f}>10%cap")
        if not m.passes_quarterly:
            failed.append(f"Quarters={m.profitable_quarters}/{m.total_quarters}<3")
        if not m.passes_sample_size:
            failed.append(f"n={m.trade_count}<50")
        m.recommendation = f"❌ FAILED — {', '.join(failed)}"


# ──────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────

def get_quarter(ts_str: str) -> str:
    """Convert timestamp string to quarter label like '2025-Q1'."""
    try:
        if isinstance(ts_str, str):
            dt = pd.Timestamp(ts_str)
        else:
            dt = ts_str
        q = (dt.month - 1) // 3 + 1
        return f"{dt.year}-Q{q}"
    except Exception:
        return "UNKNOWN"


def run_backtest():
    data_path = Path("data/ib/ES_15m_1y.parquet")
    if not data_path.exists():
        print(f"ERROR: {data_path} not found")
        sys.exit(1)

    print("=" * 75)
    print("CHOP GUARD — 12-MONTH STRATEGY COMPARISON BACKTEST")
    print("=" * 75)

    # Load and prepare data
    df = pd.read_parquet(data_path)
    print(f"\nData: {len(df)} bars, {df.index[0]} → {df.index[-1]}")

    df = add_technical_indicators(df)

    # Compute hybrid trend for every bar
    trends = []
    scores = []
    for i in range(len(df)):
        if i < 60:
            trends.append("WARMUP")
            scores.append(0.0)
        else:
            trend, score = compute_hybrid_trend(df.iloc[i])
            trends.append(trend)
            scores.append(score)
    df["hybrid_trend"] = trends
    df["trend_score"] = scores

    # Trend distribution
    valid_df = df[df["hybrid_trend"] != "WARMUP"]
    trend_counts = valid_df["hybrid_trend"].value_counts()
    chop_labels = {"CHOP", "CHOP_RANGE", "RANGE"}
    chop_bars = sum(trend_counts.get(t, 0) for t in chop_labels)
    print(f"\nTrend distribution ({len(valid_df)} bars):")
    for t, c in trend_counts.items():
        marker = " ← blocked" if t in chop_labels else ""
        print(f"  {t:15s}: {c:5d} ({c/len(valid_df)*100:.1f}%){marker}")
    print(f"  {'CHOP+RANGE':15s}: {chop_bars:5d} ({chop_bars/len(valid_df)*100:.1f}%)")

    # Run strategy to find all CHOP-blocked signals
    config = OneMinuteStrategyConfig()
    config.ft_shorts_enabled = True
    config.ft_ema9_pb_enabled = True
    config.ft_trend_cont_enabled = True
    config.ft_trend_cont_adx_min = 25.0
    config.ft_adx_min = 20.0
    config.ft_adx_max = 35.0

    strategy = EsFifteenMinStrategy(config)
    all_trades: List[SimTrade] = []
    total_signals = 0

    print(f"\nRunning strategy on {len(df) - 60} bars...")
    for i in range(60, len(df)):
        trend = df.iloc[i]["hybrid_trend"]
        is_chop = trend in chop_labels

        window_start = max(0, i - 100)
        features_slice = df.iloc[window_start:i + 1].copy()
        try:
            signal = strategy.generate(features_slice)
        except Exception:
            continue

        if signal.action in ("HOLD", ""):
            continue

        total_signals += 1
        reason = signal.metadata.get("reason", "") if isinstance(signal.metadata, dict) else ""
        is_pullback = "_PB_" in reason
        is_trend_cont = "TREND_CONT" in reason
        is_or_break = "OR_BREAK" in reason

        would_be_blocked = is_chop and (is_pullback or is_trend_cont) and not is_or_break
        if not would_be_blocked:
            continue

        row = df.iloc[i]
        is_long = signal.action in ("BUY", "SCALP_BUY")
        direction = "LONG" if is_long else "SHORT"
        entry_price = row["close"]
        atr = row.get("ATR_14", 10.0)

        # SL/TP
        meta = signal.metadata if isinstance(signal.metadata, dict) else {}
        stop_loss = meta.get("stop_loss", 0.0)
        take_profit = meta.get("take_profit", 0.0)
        if stop_loss == 0.0 or take_profit == 0.0:
            if "EMA21_PB" in reason or "OR_BREAK" in reason:
                sl_pts, tp_pts = 6.0, 8.0
            elif "EMA9_PB" in reason:
                sl_pts = min(20.0, max(8.0, atr * 1.0))
                tp_pts = sl_pts * 1.25
            elif "TREND_CONT" in reason:
                sl_pts = min(20.0, max(6.0, atr * 1.0))
                tp_pts = sl_pts * 1.25
            else:
                sl_pts, tp_pts = 6.0, 8.0
            if is_long:
                stop_loss = entry_price - sl_pts
                take_profit = entry_price + tp_pts
            else:
                stop_loss = entry_price + sl_pts
                take_profit = entry_price - tp_pts

        # ATR values for gate 5
        atr_current = float(row.get("ATR_14", 0))
        atr_5ago = float(df.iloc[max(0, i - 5)].get("ATR_14", 0)) if i >= 5 else 0.0

        ts = df.index[i]
        ts_str = str(ts)

        trade = SimTrade(
            bar_idx=i,
            timestamp=ts_str,
            signal_type=reason.split("|")[0].strip() if reason else "UNKNOWN",
            action=signal.action,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=signal.confidence,
            adx=float(row.get("ADX_14", 0)),
            rsi=float(row.get("RSI_14", 50)),
            trend_label=trend,
            atr_current=atr_current,
            atr_5ago=atr_5ago,
            daily_bias=simulate_daily_bias(df, i),
            quarter=get_quarter(ts),
        )

        # Simulate outcome
        trade = simulate_trade(trade, df, max_hold_bars=6)

        # Classify for each strategy
        classify_trade(trade)

        all_trades.append(trade)

    total_blocked = len(all_trades)
    print(f"\n✅ Scan complete:")
    print(f"   Total signals: {total_signals}")
    print(f"   CHOP-blocked:  {total_blocked}")

    if total_blocked == 0:
        print("\n⚠️  No CHOP-blocked trades found!")
        return

    # ──────────────────────────────────────────────────────────
    # 6. Compute metrics for all 3 strategies
    # ──────────────────────────────────────────────────────────

    metrics_a = compute_metrics(all_trades, "a", "A: Block-All (default)", total_blocked)
    metrics_b = compute_metrics(all_trades, "b", "B: Exception Framework (5-gate)", total_blocked)
    metrics_c = compute_metrics(all_trades, "c", "C: Direction-Aware (REVERTED)", total_blocked)

    apply_validation(metrics_a)
    apply_validation(metrics_b)
    apply_validation(metrics_c)

    # ──────────────────────────────────────────────────────────
    # 7. Output: Performance Table
    # ──────────────────────────────────────────────────────────

    print("\n" + "=" * 75)
    print("PERFORMANCE COMPARISON — 3 STRATEGIES")
    print("=" * 75)

    header = f"{'Metric':<28} {'A: Block-All':>14} {'B: Exception':>14} {'C: Dir-Aware':>14}"
    print(f"\n{header}")
    print("─" * len(header))

    def fmt_money(v):
        return f"${v:+,.2f}" if v != 0 else "$0.00"

    def fmt_pct(v):
        return f"{v:.1f}%"

    def fmt_sharpe(v):
        return f"{v:.2f}" if v != 0 else "N/A"

    rows = [
        ("Net PnL", fmt_money(metrics_a.net_pnl), fmt_money(metrics_b.net_pnl), fmt_money(metrics_c.net_pnl)),
        ("Expectancy ($/trade)", fmt_money(metrics_a.expectancy), fmt_money(metrics_b.expectancy), fmt_money(metrics_c.expectancy)),
        ("Max Drawdown", fmt_money(metrics_a.max_drawdown), fmt_money(metrics_b.max_drawdown), fmt_money(metrics_c.max_drawdown)),
        ("Trade Count", str(metrics_a.trade_count), str(metrics_b.trade_count), str(metrics_c.trade_count)),
        ("Sharpe Ratio", fmt_sharpe(metrics_a.sharpe_ratio), fmt_sharpe(metrics_b.sharpe_ratio), fmt_sharpe(metrics_c.sharpe_ratio)),
        ("Win Rate", fmt_pct(metrics_a.win_rate), fmt_pct(metrics_b.win_rate), fmt_pct(metrics_c.win_rate)),
        ("% CHOP Trades Taken", fmt_pct(metrics_a.pct_chop_trades_taken), fmt_pct(metrics_b.pct_chop_trades_taken), fmt_pct(metrics_c.pct_chop_trades_taken)),
        ("Profitable Quarters", f"{metrics_a.profitable_quarters}/{metrics_a.total_quarters}",
         f"{metrics_b.profitable_quarters}/{metrics_b.total_quarters}",
         f"{metrics_c.profitable_quarters}/{metrics_c.total_quarters}"),
    ]

    for label, va, vb, vc in rows:
        print(f"  {label:<26} {va:>14} {vb:>14} {vc:>14}")

    # ──────────────────────────────────────────────────────────
    # 8. Validation Rules
    # ──────────────────────────────────────────────────────────

    print("\n" + "=" * 75)
    print("VALIDATION RULES")
    print("=" * 75)

    def check(b):
        return "✅" if b else "❌"

    val_header = f"{'Rule':<35} {'A':>5} {'B':>5} {'C':>5}"
    print(f"\n{val_header}")
    print("─" * len(val_header))
    print(f"  {'Expectancy > 0':<33} {check(metrics_a.passes_expectancy):>5} {check(metrics_b.passes_expectancy):>5} {check(metrics_c.passes_expectancy):>5}")
    print(f"  {'Net PnL > 1.5R ($60)':<33} {check(metrics_a.passes_pnl_improvement):>5} {check(metrics_b.passes_pnl_improvement):>5} {check(metrics_c.passes_pnl_improvement):>5}")
    print(f"  {'Max DD < 10% capital ($500)':<33} {check(metrics_a.passes_max_dd):>5} {check(metrics_b.passes_max_dd):>5} {check(metrics_c.passes_max_dd):>5}")
    print(f"  {'Edge in ≥ 3 quarters':<33} {check(metrics_a.passes_quarterly):>5} {check(metrics_b.passes_quarterly):>5} {check(metrics_c.passes_quarterly):>5}")
    print(f"  {'Sample ≥ 50 trades':<33} {check(metrics_a.passes_sample_size):>5} {check(metrics_b.passes_sample_size):>5} {check(metrics_c.passes_sample_size):>5}")

    for m in [metrics_a, metrics_b, metrics_c]:
        print(f"\n  {m.name}: {m.recommendation}")

    # ──────────────────────────────────────────────────────────
    # 9. Quarterly Breakdown
    # ──────────────────────────────────────────────────────────

    print("\n" + "=" * 75)
    print("QUARTERLY BREAKDOWN")
    print("=" * 75)

    all_quarters = sorted(set(
        list(metrics_b.quarterly_detail.keys()) +
        list(metrics_c.quarterly_detail.keys())
    ))

    if all_quarters:
        q_header = f"  {'Quarter':<12} {'A: PnL':>10} {'B: PnL':>10} {'C: PnL':>10}  {'B trades':>8} {'C trades':>8}"
        print(f"\n{q_header}")
        print("  " + "─" * 60)

        for q in all_quarters:
            a_pnl = 0.0  # always $0
            b_pnl = metrics_b.quarterly_detail.get(q, 0.0)
            c_pnl = metrics_c.quarterly_detail.get(q, 0.0)
            b_n = sum(1 for t in all_trades if t.strategy_b and t.quarter == q)
            c_n = sum(1 for t in all_trades if t.strategy_c and t.quarter == q)
            marker_b = " ✅" if b_pnl > 0 else " ❌" if b_n > 0 else ""
            marker_c = " ✅" if c_pnl > 0 else " ❌" if c_n > 0 else ""
            print(f"  {q:<12} {fmt_money(a_pnl):>10} {fmt_money(b_pnl):>10}{marker_b} {fmt_money(c_pnl):>10}{marker_c}  {b_n:>8} {c_n:>8}")

    # ──────────────────────────────────────────────────────────
    # 10. Equity Curves (text-based)
    # ──────────────────────────────────────────────────────────

    print("\n" + "=" * 75)
    print("EQUITY CURVES (cumulative PnL)")
    print("=" * 75)

    b_trades = sorted([t for t in all_trades if t.strategy_b], key=lambda t: t.bar_idx)
    c_trades = sorted([t for t in all_trades if t.strategy_c], key=lambda t: t.bar_idx)

    if b_trades or c_trades:
        # Build combined timeline
        max_trades = max(len(b_trades), len(c_trades))
        width = 50  # chart width in chars

        print(f"\n  Strategy B ({len(b_trades)} trades):")
        if b_trades:
            b_cum = np.cumsum([t.pnl_dollars for t in b_trades])
            b_min, b_max = float(b_cum.min()), float(b_cum.max())
            b_range = max(abs(b_min), abs(b_max), 1)
            for i, (trade, cum) in enumerate(zip(b_trades, b_cum)):
                pos = int((cum + b_range) / (2 * b_range) * width)
                pos = max(0, min(width - 1, pos))
                zero_pos = int(b_range / (2 * b_range) * width)
                bar = [" "] * width
                bar[zero_pos] = "│"
                if pos > zero_pos:
                    for j in range(zero_pos + 1, pos + 1):
                        bar[j] = "█"
                elif pos < zero_pos:
                    for j in range(pos, zero_pos):
                        bar[j] = "█"
                indicator = "+" if trade.pnl_dollars > 0 else "-" if trade.pnl_dollars < 0 else "="
                if i < 30 or i >= len(b_trades) - 5:  # show first 30 + last 5
                    print(f"    {i+1:3d} {indicator} {''.join(bar)} ${cum:+.0f}")
                elif i == 30:
                    print(f"    ... ({len(b_trades) - 35} trades omitted) ...")

        print(f"\n  Strategy C ({len(c_trades)} trades):")
        if c_trades:
            c_cum = np.cumsum([t.pnl_dollars for t in c_trades])
            c_min, c_max = float(c_cum.min()), float(c_cum.max())
            c_range = max(abs(c_min), abs(c_max), 1)
            for i, (trade, cum) in enumerate(zip(c_trades, c_cum)):
                pos = int((cum + c_range) / (2 * c_range) * width)
                pos = max(0, min(width - 1, pos))
                zero_pos = int(c_range / (2 * c_range) * width)
                bar = [" "] * width
                bar[zero_pos] = "│"
                if pos > zero_pos:
                    for j in range(zero_pos + 1, pos + 1):
                        bar[j] = "█"
                elif pos < zero_pos:
                    for j in range(pos, zero_pos):
                        bar[j] = "█"
                indicator = "+" if trade.pnl_dollars > 0 else "-" if trade.pnl_dollars < 0 else "="
                if i < 30 or i >= len(c_trades) - 5:
                    print(f"    {i+1:3d} {indicator} {''.join(bar)} ${cum:+.0f}")
                elif i == 30:
                    print(f"    ... ({len(c_trades) - 35} trades omitted) ...")

    # ──────────────────────────────────────────────────────────
    # 11. Signal Type Breakdown by Strategy
    # ──────────────────────────────────────────────────────────

    print("\n" + "=" * 75)
    print("SIGNAL TYPE × STRATEGY BREAKDOWN")
    print("=" * 75)

    sig_types = sorted(set(t.signal_type for t in all_trades))
    for sig in sig_types:
        sig_trades = [t for t in all_trades if t.signal_type == sig]
        b_sig = [t for t in sig_trades if t.strategy_b]
        c_sig = [t for t in sig_trades if t.strategy_c]
        print(f"\n  {sig} (total blocked: {len(sig_trades)}):")
        if b_sig:
            b_wr = sum(1 for t in b_sig if t.pnl_dollars > 0) / len(b_sig) * 100
            b_pnl = sum(t.pnl_dollars for t in b_sig)
            print(f"    B (exception): n={len(b_sig):3d}, WR={b_wr:.0f}%, PnL=${b_pnl:+.2f}")
        else:
            print(f"    B (exception): n=  0 (all filtered)")
        if c_sig:
            c_wr = sum(1 for t in c_sig if t.pnl_dollars > 0) / len(c_sig) * 100
            c_pnl = sum(t.pnl_dollars for t in c_sig)
            print(f"    C (dir-aware): n={len(c_sig):3d}, WR={c_wr:.0f}%, PnL=${c_pnl:+.2f}")
        else:
            print(f"    C (dir-aware): n=  0 (all filtered)")

    # ──────────────────────────────────────────────────────────
    # 12. Exception Framework Gate Analysis
    # ──────────────────────────────────────────────────────────

    print("\n" + "=" * 75)
    print("EXCEPTION FRAMEWORK — GATE ANALYSIS")
    print("=" * 75)

    longs_in_chop = [t for t in all_trades if t.direction == "LONG"]
    if longs_in_chop:
        gate_adx = [t for t in longs_in_chop if t.adx >= 25]
        gate_bias = [t for t in longs_in_chop if t.daily_bias == "BULLISH"]
        gate_conf = [t for t in longs_in_chop if t.confidence >= 0.70]
        gate_atr = [t for t in longs_in_chop if t.atr_current > t.atr_5ago and t.atr_5ago > 0]

        total_l = len(longs_in_chop)
        print(f"\n  LONGs in CHOP: {total_l} total")
        print(f"  Gate 1 (Direction=LONG): {total_l:3d} ({total_l/total_l*100:.0f}%) — auto-pass")
        print(f"  Gate 2 (ADX ≥ 25):      {len(gate_adx):3d} ({len(gate_adx)/total_l*100:.0f}%)")
        print(f"  Gate 3 (Bias=BULLISH):   {len(gate_bias):3d} ({len(gate_bias)/total_l*100:.0f}%)")
        print(f"  Gate 4 (Conf ≥ 0.70):    {len(gate_conf):3d} ({len(gate_conf)/total_l*100:.0f}%)")
        print(f"  Gate 5 (ATR expanding):  {len(gate_atr):3d} ({len(gate_atr)/total_l*100:.0f}%)")
        all_gates = [t for t in longs_in_chop if t.strategy_b]
        print(f"  All 5 gates pass:        {len(all_gates):3d} ({len(all_gates)/total_l*100:.0f}%)")

        # Gate selectivity: which gate eliminates the most?
        gate_pass_rates = {
            "ADX≥25": len(gate_adx) / total_l,
            "BULLISH bias": len(gate_bias) / total_l,
            "Conf≥0.70": len(gate_conf) / total_l,
            "ATR expanding": len(gate_atr) / total_l,
        }
        tightest = min(gate_pass_rates, key=gate_pass_rates.get)
        print(f"\n  Tightest gate: {tightest} ({gate_pass_rates[tightest]*100:.0f}% pass rate)")

    # ──────────────────────────────────────────────────────────
    # 13. ADX Micro-Edge Analysis (Strategy B focus)
    # ──────────────────────────────────────────────────────────

    print("\n" + "=" * 75)
    print("ADX MICRO-EDGE (LONGs in CHOP)")
    print("=" * 75)

    if longs_in_chop:
        adx_bins = [(20, 25), (25, 30), (30, 35), (35, 50)]
        for lo, hi in adx_bins:
            bin_trades = [t for t in longs_in_chop if lo <= t.adx < hi]
            if bin_trades:
                wr = sum(1 for t in bin_trades if t.pnl_dollars > 0) / len(bin_trades) * 100
                pnl = sum(t.pnl_dollars for t in bin_trades)
                avg = pnl / len(bin_trades)
                print(f"  ADX [{lo:2d}-{hi:2d}): n={len(bin_trades):3d}, WR={wr:.0f}%, PnL=${pnl:+,.2f}, Avg=${avg:+.2f}")

    # ──────────────────────────────────────────────────────────
    # 14. Final Recommendation
    # ──────────────────────────────────────────────────────────

    print("\n" + "=" * 75)
    print("FINAL RECOMMENDATION")
    print("=" * 75)

    strategies = [
        ("A", metrics_a),
        ("B", metrics_b),
        ("C", metrics_c),
    ]

    best = max(strategies, key=lambda x: x[1].net_pnl)
    print(f"\n  Best Net PnL:     {best[0]} ({best[1].name}) = ${best[1].net_pnl:+,.2f}")

    # Check if any non-baseline passes all validation
    validated = [s for s in strategies if "VALIDATED" in s[1].recommendation]
    if validated:
        print(f"\n  Validated strategies: {', '.join(s[0] for s in validated)}")
        print(f"\n  → Consider enabling: {validated[0][1].name}")
    else:
        print(f"\n  ⚠️  No strategy passes all 5 validation rules.")
        print(f"  → KEEP BLOCK-ALL (Strategy A) as production default.")
        print(f"  → Exception framework stays OFF until data supports it.")

    # Specific B vs A comparison
    if metrics_b.trade_count > 0:
        print(f"\n  Strategy B vs A:")
        print(f"    PnL delta:     ${metrics_b.net_pnl - metrics_a.net_pnl:+,.2f}")
        print(f"    DD impact:     ${metrics_b.max_drawdown:+,.2f} (vs $0 for A)")
        if metrics_b.net_pnl > 0:
            print(f"    PnL/DD ratio:  {metrics_b.net_pnl / abs(metrics_b.max_drawdown):.2f}" if metrics_b.max_drawdown != 0 else "    PnL/DD ratio:  ∞")
        print(f"    Risk-adjusted: {'WORTH PURSUING' if metrics_b.net_pnl > 0 and abs(metrics_b.max_drawdown) < 300 else 'NOT YET JUSTIFIED'}")

    # Save results
    results_df = pd.DataFrame([{
        "timestamp": t.timestamp,
        "signal_type": t.signal_type,
        "direction": t.direction,
        "entry_price": t.entry_price,
        "stop_loss": t.stop_loss,
        "take_profit": t.take_profit,
        "confidence": t.confidence,
        "adx": t.adx,
        "rsi": t.rsi,
        "atr_current": t.atr_current,
        "atr_5ago": t.atr_5ago,
        "daily_bias": t.daily_bias,
        "quarter": t.quarter,
        "outcome": t.outcome,
        "pnl_dollars": t.pnl_dollars,
        "strategy_a": t.strategy_a,
        "strategy_b": t.strategy_b,
        "strategy_c": t.strategy_c,
    } for t in all_trades])
    output_path = Path("tools/chop_guard_backtest_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n  Detailed results saved to {output_path}")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    run_backtest()
