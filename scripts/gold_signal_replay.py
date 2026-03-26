#!/usr/bin/env python3
"""Replay blocked gold signals from Mar 23–24 logs to determine outcomes.

Downloads 1-minute MGC data from IB for the relevant windows and simulates
each entry to see if it would have hit TP, SL, or timed out.

Usage:
    python3 scripts/gold_signal_replay.py
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ib_insync import IB, Contract, util


# ── Signal reconstruction from logs ──────────────────────────────────────────
# We know:
# - RTH SL mult = 1.5, TP mult = 2.5
# - Extended SL mult = 2.5, TP mult = 4.0
# - SL floor = 2.0, SL ceiling = 30.0
# - MGC tick_size = 0.10, multiplier = $10/point

# Known signal: 11:46 CT → BUY @ 4428.00, SL=4422.40, TP=4437.40
# SL distance = 4428.00 - 4422.40 = 5.60 → ATR × 1.5 = 5.60 → ATR ≈ 3.73 (RTH)
# TP distance = 4437.40 - 4428.00 = 9.40 → ATR × 2.5 = 9.40 → ATR ≈ 3.76 (RTH) ✓

# For overnight signals (extended hours): SL mult=2.5, TP mult=4.0

@dataclass
class ReplaySignal:
    """A signal to replay."""
    timestamp_ct: str       # CT timestamp from log
    timestamp_utc: str      # UTC timestamp for IB query
    entry_price: float      # Close price at signal time
    action: str             # BUY or SELL (inferred from regime context)
    sl: Optional[float] = None
    tp: Optional[float] = None
    is_extended: bool = True  # Most signals are overnight/extended
    note: str = ""

@dataclass
class ReplayResult:
    """Outcome of replaying a signal."""
    signal: ReplaySignal
    outcome: str            # TP_HIT, SL_HIT, TIME_STOP
    exit_price: float
    pnl_points: float
    pnl_usd: float          # MGC = $10/point
    bars_held: int
    exit_time: str


# Reconstruct signals from log data.
# The %s formatting bug means we don't have SL/TP for the first 16 signals,
# but we know the close prices and can check the regime context to determine
# direction. From the log context, the overnight session on Mar 23-24 was
# in a TRENDING_BEAR regime (prices falling from ~4340 to ~4330 area).
# The signals around 06:16-07:57 are around 4410-4380 (post-rebound, then sell-off).

# We'll download 1m data and use the strategy's actual SL/TP logic.

SIGNALS = [
    # Mar 23 overnight — TRENDING_BEAR regime → likely SELL signals
    ReplaySignal("2026-03-23 23:33", "2026-03-24 04:33:00", 4338.70, "SELL", is_extended=True, note="Overnight TRENDING_BEAR"),
    # Mar 24 early morning cluster — price ~4335–4344, still TRENDING
    ReplaySignal("2026-03-24 00:14", "2026-03-24 05:14:00", 4337.80, "SELL", is_extended=True, note="Overnight cluster"),
    ReplaySignal("2026-03-24 00:15", "2026-03-24 05:15:00", 4335.90, "SELL", is_extended=True, note="Overnight cluster"),
    ReplaySignal("2026-03-24 00:24", "2026-03-24 05:24:00", 4343.90, "SELL", is_extended=True, note="Overnight cluster"),
    ReplaySignal("2026-03-24 00:26", "2026-03-24 05:26:00", 4342.70, "SELL", is_extended=True, note="Overnight cluster"),
    ReplaySignal("2026-03-24 00:37", "2026-03-24 05:37:00", 4338.30, "SELL", is_extended=True, note="Overnight cluster"),
    # Mar 24 pre-RTH — price moving from 4420 down to 4380
    ReplaySignal("2026-03-24 06:16", "2026-03-24 11:16:00", 4416.70, "SELL", is_extended=True, note="Pre-RTH sell-off"),
    ReplaySignal("2026-03-24 06:17", "2026-03-24 11:17:00", 4413.30, "SELL", is_extended=True, note="Pre-RTH sell-off"),
    ReplaySignal("2026-03-24 06:21", "2026-03-24 11:21:00", 4415.50, "SELL", is_extended=True, note="Pre-RTH sell-off"),
    ReplaySignal("2026-03-24 06:24", "2026-03-24 11:24:00", 4414.50, "SELL", is_extended=True, note="Pre-RTH sell-off"),
    ReplaySignal("2026-03-24 06:26", "2026-03-24 11:26:00", 4411.80, "SELL", is_extended=True, note="Pre-RTH sell-off"),
    ReplaySignal("2026-03-24 07:08", "2026-03-24 12:08:00", 4409.80, "SELL", is_extended=True, note="Pre-RTH continuation"),
    ReplaySignal("2026-03-24 07:49", "2026-03-24 12:49:00", 4379.60, "SELL", is_extended=True, note="Deep sell-off"),
    ReplaySignal("2026-03-24 07:52", "2026-03-24 12:52:00", 4378.50, "SELL", is_extended=True, note="Deep sell-off"),
    ReplaySignal("2026-03-24 07:54", "2026-03-24 12:54:00", 4383.20, "SELL", is_extended=True, note="Deep sell-off"),
    ReplaySignal("2026-03-24 07:57", "2026-03-24 12:57:00", 4380.60, "SELL", is_extended=True, note="Deep sell-off"),
    # Mar 24 RTH — the __enter__ bug signal (known BUY)
    ReplaySignal("2026-03-24 11:46", "2026-03-24 16:46:00", 4428.00, "BUY", sl=4422.40, tp=4437.40, is_extended=False, note="RTH BUY - __enter__ bug"),
]


def compute_sl_tp(entry: float, action: str, atr: float, is_extended: bool) -> tuple:
    """Replicate the gold strategy's SL/TP logic."""
    if is_extended:
        sl_mult, tp_mult = 2.5, 4.0
    else:
        sl_mult, tp_mult = 1.5, 2.5

    sl_dist = atr * sl_mult
    sl_dist = max(sl_dist, 2.0)   # floor
    sl_dist = min(sl_dist, 30.0)  # ceiling
    tp_dist = atr * tp_mult

    if action == "BUY":
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist

    # Snap to 0.10 tick
    import math
    sl = round(round(sl / 0.10) * 0.10, 2)
    tp = round(round(tp / 0.10) * 0.10, 2)
    return sl, tp


async def download_mgc_bars(ib: IB, start_utc: str, duration: str = "2 D") -> list:
    """Download 1-minute MGC bars from IB."""
    contract = Contract(
        symbol="MGC",
        secType="FUT",
        exchange="COMEX",
        currency="USD",
    )
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        # Try with explicit month
        contract.lastTradeDateOrContractMonth = "202604"
        qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        print("ERROR: Could not qualify MGC contract")
        return []

    contract = qualified[0]
    print(f"Qualified: {contract.localSymbol} (conId={contract.conId})")

    bars = await ib.reqHistoricalDataAsync(
        contract,
        endDateTime=start_utc,
        durationStr=duration,
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=2,
    )
    print(f"Downloaded {len(bars)} 1-minute bars")
    return bars


def replay_signal(signal: ReplaySignal, bars: list, atr_value: float) -> ReplayResult:
    """Simulate a single trade against 1-minute bars.
    
    Walks forward from signal entry time checking each bar's high/low
    against SL and TP. Returns after max 120 bars (2 hours) as time stop.
    """
    # Compute SL/TP if not provided
    sl = signal.sl
    tp = signal.tp
    if sl is None or tp is None:
        sl, tp = compute_sl_tp(signal.entry_price, signal.action, atr_value, signal.is_extended)

    entry_price = signal.entry_price
    max_bars = 120  # 2-hour time stop

    # Find the bar at or after the signal time
    signal_dt = datetime.fromisoformat(signal.timestamp_utc).replace(tzinfo=timezone.utc)
    started = False
    bars_held = 0

    for bar in bars:
        bar_time = bar.date if hasattr(bar, 'date') else bar['date']
        if hasattr(bar_time, 'timestamp'):
            pass  # already datetime-like
        else:
            bar_time = datetime.fromisoformat(str(bar_time))

        if not bar_time.tzinfo:
            bar_time = bar_time.replace(tzinfo=timezone.utc)

        if bar_time < signal_dt:
            continue
        started = True
        bars_held += 1

        high = bar.high
        low = bar.low

        if signal.action == "BUY":
            # Check SL first (conservative)
            if low <= sl:
                pnl_pts = sl - entry_price
                return ReplayResult(signal, "SL_HIT", sl, pnl_pts, pnl_pts * 10, bars_held, str(bar_time))
            if high >= tp:
                pnl_pts = tp - entry_price
                return ReplayResult(signal, "TP_HIT", tp, pnl_pts, pnl_pts * 10, bars_held, str(bar_time))
        else:  # SELL
            # Check SL first (conservative)
            if high >= sl:
                pnl_pts = entry_price - sl
                return ReplayResult(signal, "SL_HIT", sl, pnl_pts, pnl_pts * 10, bars_held, str(bar_time))
            if low <= tp:
                pnl_pts = entry_price - tp
                return ReplayResult(signal, "TP_HIT", tp, pnl_pts, pnl_pts * 10, bars_held, str(bar_time))

        if bars_held >= max_bars:
            close = bar.close
            if signal.action == "BUY":
                pnl_pts = close - entry_price
            else:
                pnl_pts = entry_price - close
            return ReplayResult(signal, "TIME_STOP", close, pnl_pts, pnl_pts * 10, bars_held, str(bar_time))

    # Ran out of bars — use last available
    if bars and started:
        last = bars[-1]
        close = last.close
        if signal.action == "BUY":
            pnl_pts = close - entry_price
        else:
            pnl_pts = entry_price - close
        return ReplayResult(signal, "DATA_END", close, pnl_pts, pnl_pts * 10, bars_held, str(last.date))

    return ReplayResult(signal, "NO_DATA", entry_price, 0, 0, 0, "N/A")


async def main():
    print("=" * 70)
    print("Gold Signal Replay — Mar 23–24 2026 Blocked Signals")
    print("=" * 70)

    ib = IB()
    try:
        await ib.connectAsync("127.0.0.1", 4001, clientId=99, timeout=10)
        print("Connected to IB Gateway")
    except Exception as e:
        print(f"Cannot connect to IB: {e}")
        print("Running with estimated ATR values only (no bar data)")
        # Run with estimates anyway
        _run_without_ib()
        return

    # Download bars for the full period (Mar 23 evening through Mar 25 morning)
    try:
        bars = await download_mgc_bars(ib, "20260325 15:00:00 US/Eastern", "3 D")
    except Exception as e:
        print(f"Error downloading bars: {e}")
        ib.disconnect()
        _run_without_ib()
        return

    if not bars:
        print("No bars downloaded")
        ib.disconnect()
        _run_without_ib()
        return

    # Estimate ATR from downloaded bars
    import pandas as pd
    df = util.df(bars)
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate ATR(14) on the 1-min bars
    df['tr'] = df.apply(
        lambda r: max(r['high'] - r['low'], abs(r['high'] - r['close']), abs(r['low'] - r['close'])),
        axis=1,
    )
    df['atr14'] = df['tr'].rolling(14).mean()

    # Get ATR at various timestamps for signal replay
    print(f"\nBar data range: {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
    print(f"ATR(14) range: {df['atr14'].min():.2f} – {df['atr14'].max():.2f}")
    print(f"ATR(14) mean: {df['atr14'].mean():.2f}")

    # Find ATR at each signal time
    results: List[ReplayResult] = []
    for sig in SIGNALS:
        sig_dt = datetime.fromisoformat(sig.timestamp_utc).replace(tzinfo=timezone.utc)
        # Find nearest bar
        nearest_idx = None
        for i, row in df.iterrows():
            bar_dt = row['date']
            if hasattr(bar_dt, 'tzinfo') and bar_dt.tzinfo:
                pass
            else:
                bar_dt = pd.Timestamp(bar_dt, tz='UTC')
            if bar_dt >= sig_dt:
                nearest_idx = i
                break
        
        atr = 3.5  # default estimate
        if nearest_idx is not None and nearest_idx >= 14:
            atr = float(df.loc[nearest_idx, 'atr14'])
            if pd.isna(atr):
                atr = 3.5

        result = replay_signal(sig, bars, atr)
        results.append(result)
        
        sl_used = sig.sl if sig.sl else compute_sl_tp(sig.entry_price, sig.action, atr, sig.is_extended)[0]
        tp_used = sig.tp if sig.tp else compute_sl_tp(sig.entry_price, sig.action, atr, sig.is_extended)[1]
        
        icon = "✅" if result.outcome == "TP_HIT" else "❌" if result.outcome == "SL_HIT" else "⏱️"
        print(f"\n{icon} {sig.timestamp_ct} | {sig.action} @ {sig.entry_price:.2f} | "
              f"SL={sl_used:.2f} TP={tp_used:.2f} | ATR={atr:.2f} | "
              f"{result.outcome} after {result.bars_held} bars → P&L: ${result.pnl_usd:+.2f}")
        if sig.note:
            print(f"   Note: {sig.note}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    winners = [r for r in results if r.outcome == "TP_HIT"]
    losers = [r for r in results if r.outcome == "SL_HIT"]
    time_stops = [r for r in results if r.outcome == "TIME_STOP"]
    other = [r for r in results if r.outcome not in ("TP_HIT", "SL_HIT", "TIME_STOP")]

    total_pnl = sum(r.pnl_usd for r in results)
    print(f"Total signals: {len(results)}")
    print(f"  TP_HIT (winners):  {len(winners)}  →  ${sum(r.pnl_usd for r in winners):+.2f}")
    print(f"  SL_HIT (losers):   {len(losers)}  →  ${sum(r.pnl_usd for r in losers):+.2f}")
    print(f"  TIME_STOP:         {len(time_stops)}  →  ${sum(r.pnl_usd for r in time_stops):+.2f}")
    if other:
        print(f"  Other:             {len(other)}  →  ${sum(r.pnl_usd for r in other):+.2f}")
    print(f"  NET P&L:           ${total_pnl:+.2f}")
    win_rate = len(winners) / len(results) * 100 if results else 0
    print(f"  Win Rate:          {win_rate:.0f}%")

    ib.disconnect()


def _run_without_ib():
    """Fallback: use estimated ATR and the known SL/TP for the 11:46 signal."""
    print("\n⚠️  Cannot connect to IB — showing signal reconstruction only")
    print("The 11:46 signal is the only one with full SL/TP data:")
    print("  BUY @ 4428.00, SL=4422.40, TP=4437.40")
    print("  SL distance = 5.60 pts, TP distance = 9.40 pts")
    print("  Implied ATR ≈ 3.73 (RTH mult 1.5/2.5)")
    print("\nFor the 16 overnight signals, estimated using ATR ≈ 3.5:")
    print("  Extended hours: SL mult=2.5, TP mult=4.0")
    print("  SL distance ≈ 8.75 pts ($87.50), TP distance ≈ 14.0 pts ($140)")
    print("\nTo get actual outcomes, run this script while IB Gateway is connected.")


if __name__ == "__main__":
    asyncio.run(main())
