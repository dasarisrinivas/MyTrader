#!/usr/bin/env python3
"""
Fetch the Jan-2026 -> present ES 15m forward slice from Interactive Brokers,
matching the training pipeline byte-for-byte so the forward test (see
FORWARD_TEST_PROTOCOL.md) cannot be contaminated by source mismatch.

MUST be run on the machine where IB Gateway is reachable (127.0.0.1:4001) — the
research sandbox cannot reach your local Gateway. Run from the repo root with the
venv that has ib_insync installed:

    python tools/expansion_research/fetch_forward_slice.py

Output: data/ib/ES_15m_fwd_2026.parquet   (NEW file; training parquet untouched)
"""
from __future__ import annotations
import asyncio, sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backtest.data.ib_downloader import IBHistoricalDownloader, DownloadConfig

TRAIN = ROOT / "data/ib/ES_15m_1y.parquet"
OUT = ROOT / "data/ib/ES_15m_fwd_2026.parquet"
FORWARD_START = datetime(2026, 1, 17, 0, 0, tzinfo=timezone.utc)  # day after training ends
TZ = "US/Central"
BAR = "15 mins"


async def fetch() -> pd.DataFrame:
    cfg = DownloadConfig(host="127.0.0.1", port=4001, client_id=71)
    dl = IBHistoricalDownloader(cfg)
    try:
        await dl.connect()
        end = datetime.now(timezone.utc)
        print(f"Fetching ES {BAR} continuous front-month {FORWARD_START.date()} -> {end.date()} ...")
        df = await dl.download_continuous(
            symbol="ES", start=FORWARD_START, end=end,
            bar_size=BAR, what_to_show="TRADES", use_cache=False,
        )
        return df
    finally:
        dl.disconnect()


def main():
    if not TRAIN.exists():
        sys.exit(f"Training parquet not found: {TRAIN}")
    df = asyncio.run(fetch())
    if df is None or df.empty:
        sys.exit("No data returned from IB. Is Gateway up on 4001 and the ES contract subscribed?")

    # UTC (downloader output) -> US/Central to match training file exactly
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(TZ)
    df.index.name = "timestamp"
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # ---- pre-flight assertions vs training pipeline --------------------
    tr = pd.read_parquet(TRAIN)
    assert list(df.columns) == list(tr.columns), f"column mismatch {list(df.columns)} vs {list(tr.columns)}"
    assert str(df.index.tz) == str(tr.index.tz) == TZ, f"tz mismatch {df.index.tz} vs {tr.index.tz}"
    assert df.index.min() > tr.index.max(), (
        f"forward start {df.index.min()} not strictly after training end {tr.index.max()}")
    gap_days = (df.index.min() - tr.index.max()).total_seconds() / 86400
    bars_per_day = df.groupby(df.index.date).size()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, engine="pyarrow")

    print("\n" + "=" * 64)
    print("FORWARD SLICE SAVED")
    print("=" * 64)
    print(f"  file       : {OUT.relative_to(ROOT)}")
    print(f"  rows       : {len(df)}")
    print(f"  range      : {df.index.min()}  ->  {df.index.max()}")
    print(f"  tz         : {df.index.tz}")
    print(f"  training end: {tr.index.max()}  (seam gap {gap_days:.1f} days)")
    print(f"  bars/day   : median {int(bars_per_day.median())}, "
          f"min {bars_per_day.min()}, max {bars_per_day.max()}")
    print(f"  trading days: {bars_per_day.size}")
    print("\nNext: python tools/expansion_research/forward_test.py")


if __name__ == "__main__":
    main()
