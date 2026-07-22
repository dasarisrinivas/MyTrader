"""One-off: pull REAL MES 15m history from IB (read-only) for clean structure analysis.

ContFuture forbids endDateTime paging, so request the max single-shot duration.
Spare client id (17) so the live signal bot (cid=1) is untouched.
"""
import sys
import pandas as pd
from ib_insync import IB, ContFuture

ib = IB()
ib.connect("127.0.0.1", 4001, clientId=17, timeout=30)
ib.RequestTimeout = 120
print("connected cid=17 (read-only historical)")
cf = ContFuture("MES", "CME", currency="USD")
[c] = ib.qualifyContracts(cf)
print("qualified:", c.localSymbol, c.lastTradeDateOrContractMonth)


def fetch(duration, use_rth, tag):
    bars = ib.reqHistoricalData(
        c, endDateTime="", durationStr=duration,
        barSizeSetting="15 mins", whatToShow="TRADES",
        useRTH=use_rth, formatDate=2,
    )
    if not bars:
        print(f"  {tag} {duration}: 0 bars")
        return 0
    df = pd.DataFrame(
        {"open": [b.open for b in bars], "high": [b.high for b in bars],
         "low": [b.low for b in bars], "close": [b.close for b in bars],
         "volume": [b.volume for b in bars]},
        index=pd.DatetimeIndex([pd.Timestamp(b.date) for b in bars]),
    )
    df = df[~df.index.duplicated(keep="first")].sort_index()
    out = f"/tmp/mes_15m_{tag}.parquet"
    df.to_parquet(out)
    print(f"  {tag}: {len(df)} bars  {df.index[0]} -> {df.index[-1]}  -> {out}")
    return len(df)


for use_rth, tag in ((True, "rth"), (False, "eth")):
    for dur in ("1 Y", "6 M", "3 M", "2 M"):
        try:
            if fetch(dur, use_rth, tag):
                break
        except Exception as e:
            print(f"  {tag} {dur} failed: {e}")
ib.disconnect()
print("done")
