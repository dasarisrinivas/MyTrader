#!/usr/bin/env python3
"""
Download multi-year 1-minute ES futures data from Interactive Brokers by
ROLLING THROUGH THE DATED QUARTERLY CONTRACTS (H/M/U/Z) that were actually
trading in each period.

Why not backtest.run / download_continuous?
    The library's `download_continuous` picks today's front month (e.g. ESM6,
    expiry Jun-2026) and then requests EVERY historical chunk against that one
    contract. A 2026 contract has no 2023 data, so it returns empty and aborts.
    This script instead asks each quarter's own contract for its own data and
    stitches them with clean rolls at expiry.

Run on your Mac with IB Gateway LIVE (port 4001):
    cd /Users/svss/Documents/code/ShreeBot
    python scripts/download_es_multiyr_1m.py --start-year 2023

Notes:
- client_id 98 (won't collide with the running bot).
- 1m downloads in ~6-day chunks under IB pacing -> a multi-year pull is slow
  (expect 1-2 hours). Safe to leave running.
- IB does NOT keep 1m history forever. Once it stops serving an old contract,
  the script logs it and stops walking further back (after 2 dead contracts),
  keeping everything it did get. The printed span is what IB actually served.
- We pull ES (not MES): deeper/more-liquid 1m, identical index points, and the
  backtest prices PnL at $5/pt MES regardless.
"""

import argparse
import asyncio
import calendar
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from ib_insync import IB, Future
from loguru import logger

QUARTERLY_MONTHS = [3, 6, 9, 12]  # H, M, U, Z


def third_friday(year: int, month: int) -> datetime:
    """ES quarterly contracts settle on the 3rd Friday of the contract month."""
    fridays = [
        d for d in calendar.Calendar().itermonthdates(year, month)
        if d.month == month and d.weekday() == 4
    ]
    return datetime(fridays[2].year, fridays[2].month, fridays[2].day, tzinfo=timezone.utc)


def build_contract_calendar(start_year: int, now: datetime):
    """List of (expiry_dt, prev_expiry_dt, yyyymm) for ES quarterlies covering the span."""
    exps = []
    for y in range(start_year, now.year + 2):
        for m in QUARTERLY_MONTHS:
            exps.append((third_friday(y, m), f"{y}{m:02d}"))
    exps.sort(key=lambda t: t[0])
    out = []
    for i, (exp, yyyymm) in enumerate(exps):
        if exp < third_friday(start_year, 3) - timedelta(days=95):
            continue
        if exp - timedelta(days=95) > now:  # not trading yet
            continue
        prev_exp = exps[i - 1][0] if i > 0 else exp - timedelta(days=95)
        out.append((exp, prev_exp, yyyymm))
    return out


async def download_one_contract(ib, yyyymm, prev_exp, exp, now, chunk_days=6):
    """Pull a single dated ES contract over its active window (prev_exp, exp]."""
    contract = Future(symbol="ES", lastTradeDateOrContractMonth=yyyymm,
                      exchange="CME", currency="USD", includeExpired=True)
    try:
        q = await ib.qualifyContractsAsync(contract)
        if not q:
            logger.warning(f"  ES{yyyymm}: could not qualify")
            return pd.DataFrame()
        contract = q[0]
    except Exception as e:
        logger.warning(f"  ES{yyyymm}: qualify failed: {e}")
        return pd.DataFrame()

    win_start = prev_exp
    win_end = min(exp + timedelta(days=2), now)
    frames = []
    cur_end = win_end
    while cur_end > win_start:
        cur_start = max(cur_end - timedelta(days=chunk_days), win_start)
        days = max((cur_end - cur_start).days, 1)
        try:
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime=cur_end.strftime("%Y%m%d %H:%M:%S"),
                durationStr=f"{days} D",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
                timeout=120,
            )
            if bars:
                frames.append(pd.DataFrame([{
                    "timestamp": pd.Timestamp(b.date).tz_localize("UTC")
                    if b.date.tzinfo is None else pd.Timestamp(b.date).tz_convert("UTC"),
                    "open": float(b.open), "high": float(b.high),
                    "low": float(b.low), "close": float(b.close),
                    "volume": int(b.volume),
                } for b in bars]))
        except Exception as e:
            logger.warning(f"  ES{yyyymm} chunk ending {cur_end.date()}: {e}")
        await asyncio.sleep(1.1)  # gentle pacing
        cur_end = cur_start

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).set_index("timestamp")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    # clip to this contract's active window so rolls are clean (no overlap)
    df = df[(df.index > win_start) & (df.index <= win_end)]
    return df


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2023)
    ap.add_argument("--port", type=int, default=4001)
    ap.add_argument("--client-id", type=int, default=98)
    ap.add_argument("--out", default="data/ib/ES_1m_multiyr.parquet")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    cal = build_contract_calendar(args.start_year, now)
    logger.info(f"Will roll through {len(cal)} ES contracts back to ~{args.start_year}")

    ib = IB()
    await ib.connectAsync("127.0.0.1", args.port, clientId=args.client_id)
    logger.info(f"Connected to IB on port {args.port} (client_id {args.client_id})")

    collected, dead_streak = [], 0
    try:
        # newest -> oldest, stop after 2 contracts IB won't serve (depth wall)
        for exp, prev_exp, yyyymm in reversed(cal):
            logger.info(f"Contract ES{yyyymm} (expiry {exp.date()}) "
                        f"window {prev_exp.date()}..{min(exp, now).date()}")
            df = await download_one_contract(ib, yyyymm, prev_exp, exp, now)
            if df.empty:
                dead_streak += 1
                logger.warning(f"  -> no data ({dead_streak} dead in a row)")
                if dead_streak >= 2:
                    logger.warning("  -> hit IB's 1m depth wall, stopping.")
                    break
                continue
            dead_streak = 0
            logger.info(f"  -> {len(df):,} bars {df.index.min()} .. {df.index.max()}")
            collected.append(df)
    finally:
        ib.disconnect()

    if not collected:
        logger.error("Nothing downloaded. Is IB Gateway running on the port?")
        return

    full = pd.concat(collected)
    full = full[~full.index.duplicated(keep="first")].sort_index()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out, engine="pyarrow")

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Bars:     {len(full):,}")
    print(f"Span got: {full.index.min()}  ->  {full.index.max()}")
    print(f"Saved to: {out}")
    print("\nTell Claude this file is ready; it'll run the live-faithful "
          "backtest + ADX re-validation on it.")


if __name__ == "__main__":
    asyncio.run(main())
