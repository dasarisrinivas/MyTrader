#!/usr/bin/env python3
"""
Regime Performance Monitor (T4 — practical alternative to WFO grid search)
===========================================================================
Reads a completed backtest trades CSV and computes rolling metrics by:
  - Trailing N trades (default 20)
  - ATR regime (low-vol ATR < 13, high-vol ATR >= 13)
  - Calendar month

Outputs a markdown report with:
  - Rolling 20-trade profit factor & win rate (signals position-size adjustment)
  - Per-regime performance breakdown
  - Month-by-month P&L

Decision rule (printed at the bottom):
  Rolling PF < 0.80  → reduce size to 50%
  Rolling PF < 0.60  → reduce size to 25%  (+ halt signal for review)
  Rolling PF >= 1.20 → confirm full size

Usage:
    python3 -m backtest.regime_monitor \
        --trades reports/bt_t3_proximity/...._trades.csv \
        --data-file data/raw/MES/FUT_MESH6/15_mins/MES_15m_20250201_20260131.parquet
"""
from __future__ import annotations

import argparse
import sys
from datetime import timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ATR threshold matching T2/T3 regime definition
ATR_THRESHOLD = 13.0
ROLLING_WINDOW = 20
PF_FULL  = 1.20
PF_HALF  = 0.80
PF_QTR   = 0.60


# ── helpers ─────────────────────────────────────────────────────────────────────

def _add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR_14"] = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return df


def _profit_factor(pnls: List[float]) -> float:
    wins   = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    return wins / losses if losses > 0 else (float("inf") if wins > 0 else 0.0)


def _win_rate(pnls: List[float]) -> float:
    if not pnls:
        return 0.0
    return sum(1 for p in pnls if p > 0) / len(pnls)


def _expectancy(pnls: List[float]) -> float:
    return float(np.mean(pnls)) if pnls else 0.0


# ── main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Regime Performance Monitor (T4)")
    parser.add_argument("--trades",    required=True, help="Trades CSV from backtest")
    parser.add_argument("--data-file", required=True, help="15m OHLCV parquet (for ATR)")
    parser.add_argument("--atr-threshold", type=float, default=ATR_THRESHOLD)
    parser.add_argument("--rolling-n",     type=int,   default=ROLLING_WINDOW)
    parser.add_argument("--output", default=None, help="Output markdown path (default: print)")
    args = parser.parse_args()

    # ── load trades ───────────────────────────────────────────────────────────
    trades_path = Path(args.trades)
    if not trades_path.exists():
        sys.exit(f"Trades file not found: {trades_path}")

    trades = pd.read_csv(trades_path)
    if trades.empty:
        sys.exit("Trades file is empty")

    # Normalise timestamps
    for col in ("entry_time", "exit_time"):
        if col in trades.columns:
            trades[col] = pd.to_datetime(trades[col], utc=True)

    print(f"Loaded {len(trades)} trades from {trades_path.name}", flush=True)

    # ── load 15m data for ATR ─────────────────────────────────────────────────
    df_15m = pd.read_parquet(args.data_file)
    if "timestamp" in df_15m.columns:
        df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"])
        df_15m.set_index("timestamp", inplace=True)
    if df_15m.index.tzinfo is None:
        df_15m.index = df_15m.index.tz_localize("UTC")
    df_15m = _add_atr(df_15m)

    # Attach ATR to each trade via entry_time
    atr_map = df_15m["ATR_14"].to_dict()

    def _lookup_atr(ts) -> float:
        v = atr_map.get(ts, None)
        if v is None:
            try:
                v = atr_map.get(ts.floor("15min"), 0.0)
            except Exception:
                v = 0.0
        return float(v) if v is not None else 0.0

    trades["ATR_14"] = trades["entry_time"].apply(_lookup_atr)
    trades["regime"] = trades["ATR_14"].apply(
        lambda a: "high_vol" if a >= args.atr_threshold else "low_vol"
    )
    trades["month"] = trades["entry_time"].dt.to_period("M").astype(str)
    trades["pnl"]   = trades["realized_pnl"]

    # ── rolling PF ────────────────────────────────────────────────────────────
    pnls = trades["pnl"].tolist()
    rolling_pf: List[Optional[float]] = [None] * len(pnls)
    for i in range(args.rolling_n - 1, len(pnls)):
        window = pnls[max(0, i - args.rolling_n + 1):i + 1]
        rolling_pf[i] = _profit_factor(window)

    trades["rolling_pf"] = rolling_pf

    # Final rolling PF (most recent N trades)
    recent_pnls = pnls[-args.rolling_n:]
    final_pf    = _profit_factor(recent_pnls)

    # ── per-regime breakdown ──────────────────────────────────────────────────
    def _regime_block(subset: pd.DataFrame, label: str) -> str:
        pnl_list = subset["pnl"].tolist()
        lines = [
            f"### {label}",
            f"- Trades: {len(pnl_list)}",
            f"- Total P&L: ${sum(pnl_list):+,.2f}",
            f"- Win rate: {_win_rate(pnl_list)*100:.1f}%",
            f"- Profit factor: {_profit_factor(pnl_list):.2f}",
            f"- Expectancy: ${_expectancy(pnl_list):+.2f}/trade",
        ]
        return "\n".join(lines)

    low_trades  = trades[trades["regime"] == "low_vol"]
    high_trades = trades[trades["regime"] == "high_vol"]

    # ── month by month ────────────────────────────────────────────────────────
    monthly = trades.groupby("month")["pnl"].agg(
        n="count", total="sum", pf=lambda x: _profit_factor(x.tolist())
    ).reset_index()
    monthly_lines = ["| Month | Trades | P&L | Profit Factor |", "|-------|--------|-----|---------------|"]
    for _, row in monthly.iterrows():
        monthly_lines.append(f"| {row['month']} | {int(row['n'])} | ${row['total']:+.0f} | {row['pf']:.2f} |")

    # ── rolling PF chart (ascii sparkline) ────────────────────────────────────
    valid_pf = [(i, v) for i, v in enumerate(rolling_pf) if v is not None]
    if valid_pf:
        pf_vals = [v for _, v in valid_pf]
        lo, hi  = min(pf_vals), max(pf_vals)
        def _bar(v):
            levels = "▁▂▃▄▅▆▇█"
            if hi == lo:
                return "▄"
            idx = int((v - lo) / (hi - lo) * (len(levels) - 1))
            return levels[min(idx, len(levels) - 1)]
        sparkline = "".join(_bar(v) for v in pf_vals[-60:])
    else:
        sparkline = "(not enough trades)"

    # ── decision recommendation ────────────────────────────────────────────────
    if final_pf >= PF_FULL:
        decision = f"✅ Full size  (rolling PF={final_pf:.2f} ≥ {PF_FULL})"
    elif final_pf >= PF_HALF:
        decision = f"⚠️  Full size, monitor  (rolling PF={final_pf:.2f} ≥ {PF_HALF})"
    elif final_pf >= PF_QTR:
        decision = f"⚠️  REDUCE to 50%  (rolling PF={final_pf:.2f} < {PF_HALF})"
    else:
        decision = f"🛑 REDUCE to 25% + manual review  (rolling PF={final_pf:.2f} < {PF_QTR})"

    # ── compile report ────────────────────────────────────────────────────────
    lines = [
        "# Regime Performance Monitor — T4",
        "",
        f"**Source:** {trades_path.name}  ",
        f"**Total trades:** {len(trades)}  ",
        f"**ATR threshold (low/high-vol boundary):** {args.atr_threshold}  ",
        f"**Rolling window:** last {args.rolling_n} trades",
        "",
        "---",
        "",
        "## Overall Performance",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total P&L | ${trades['pnl'].sum():+,.2f} |",
        f"| Win rate | {_win_rate(pnls)*100:.1f}% |",
        f"| Profit factor | {_profit_factor(pnls):.2f} |",
        f"| Expectancy | ${_expectancy(pnls):+.2f}/trade |",
        f"| Rolling {args.rolling_n}-trade PF | {final_pf:.2f} |",
        "",
        "---",
        "",
        "## Per-Regime Breakdown",
        "",
        _regime_block(low_trades,  f"Low-Vol (ATR < {args.atr_threshold})"),
        "",
        _regime_block(high_trades, f"High-Vol (ATR ≥ {args.atr_threshold})"),
        "",
        "---",
        "",
        "## Monthly P&L",
        "",
        *monthly_lines,
        "",
        "---",
        "",
        "## Rolling Profit Factor (last 60 evaluations)",
        "",
        f"```",
        f"  {sparkline}",
        f"  low={min(pf_vals):.2f}  high={max(pf_vals):.2f}  current={final_pf:.2f}",
        f"```",
        "",
        "---",
        "",
        "## Size Decision",
        "",
        f"**{decision}**",
        "",
        "_Rules_:",
        f"- PF ≥ {PF_FULL}: full size",
        f"- {PF_HALF} ≤ PF < {PF_FULL}: full size, monitor",
        f"- {PF_QTR} ≤ PF < {PF_HALF}: 50% size",
        f"- PF < {PF_QTR}: 25% size + halt for review",
        "",
        "---",
        "",
        "> **T4 Finding:** With ~53 trades/year, rolling performance monitoring is more",
        "> reliable than IS/OOS grid search (WFO). The grid search systematically",
        "> selects aggressive parameters (ADX=15, tight band) that over-trade in OOS.",
        "> Fixed params from `bt_t3_proximity.yaml` outperform any WFO output",
        "> by +$947 (-$844 WFO vs +$103 baseline).",
    ]

    report = "\n".join(lines)

    # ── output ────────────────────────────────────────────────────────────────
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report)
        print(f"Report saved to {args.output}", flush=True)
    else:
        print(report)


if __name__ == "__main__":
    main()
