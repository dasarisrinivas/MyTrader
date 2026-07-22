"""Pull REAL NQ (Nasdaq-100 future) history at multiple timeframes from IB.

Read-only, cid=17. Live signal bot (cid=1) untouched. Also QQZ/QQQ, VIX, VXN
for cross-asset. For the NQ edge-hunt — mirrors the MES protocol exactly.
"""
import pandas as pd
from ib_insync import IB, ContFuture, Stock, Index

ib = IB()
ib.connect("127.0.0.1", 4001, clientId=17, timeout=30)
ib.RequestTimeout = 180
print("connected cid=17 (read-only)")


def save(bars, tag):
    if not bars:
        print(f"  {tag}: 0 bars"); return 0
    df = pd.DataFrame(
        {"open": [b.open for b in bars], "high": [b.high for b in bars],
         "low": [b.low for b in bars], "close": [b.close for b in bars],
         "volume": [b.volume for b in bars]},
        index=pd.DatetimeIndex([pd.Timestamp(b.date) for b in bars]))
    df = df[~df.index.duplicated(keep="first")].sort_index()
    out = f"/tmp/{tag}.parquet"; df.to_parquet(out)
    print(f"  {tag}: {len(df)} bars {df.index[0]} -> {df.index[-1]}")
    return len(df)


def hist(contract, dur, bar, rth, tag, what="TRADES"):
    try:
        bars = ib.reqHistoricalData(contract, endDateTime="", durationStr=dur,
                                    barSizeSetting=bar, whatToShow=what,
                                    useRTH=rth, formatDate=2)
        return save(bars, tag)
    except Exception as e:
        print(f"  {tag} failed: {e}"); return 0


nq = ContFuture("NQ", "CME", currency="USD")
[nq] = ib.qualifyContracts(nq)
print("NQ:", nq.localSymbol)

jobs = [
    ("1 min",  "5 D",  True,  "nq_1m_rth"),
    ("1 min",  "5 D",  False, "nq_1m_eth"),
    ("2 mins", "1 M",  True,  "nq_2m_rth"),
    ("3 mins", "1 M",  True,  "nq_3m_rth"),
    ("5 mins", "2 M",  True,  "nq_5m_rth"),
    ("5 mins", "2 M",  False, "nq_5m_eth"),
    ("10 mins","6 M",  True,  "nq_10m_rth"),
    ("15 mins","1 Y",  True,  "nq_15m_rth"),
    ("15 mins","1 Y",  False, "nq_15m_eth"),
    ("30 mins","1 Y",  True,  "nq_30m_rth"),
    ("1 hour", "2 Y",  True,  "nq_60m_rth"),
]
for bar, dur, rth, tag in jobs:
    hist(nq, dur, bar, rth, tag)

qqq = Stock("QQQ", "SMART", "USD"); [qqq] = ib.qualifyContracts(qqq)
hist(qqq, "1 Y", "15 mins", True, "qqq_15m_rth")
for sym in ("VIX", "VXN"):
    try:
        ix = Index(sym, "CBOE"); [ix] = ib.qualifyContracts(ix)
        hist(ix, "1 Y", "15 mins", True, f"{sym.lower()}_15m_rth")
    except Exception as e:
        print(f"  {sym} failed: {e}")

ib.disconnect()
print("done")
