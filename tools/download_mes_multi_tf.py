"""Pull REAL MES history at MULTIPLE timeframes from IB (read-only, cid=17).

For the cross-timeframe edge hunt. Live signal bot (cid=1) untouched.
Also pulls SPY (lead) and VIX for cross-asset tests.
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


mes = ContFuture("MES", "CME", currency="USD")
[mes] = ib.qualifyContracts(mes)
print("MES:", mes.localSymbol)

# (bar, duration, rth-flag, tag) — durations tuned to IB caps per bar size
jobs = [
    ("1 min",  "5 D",  True,  "mes_1m_rth"),
    ("1 min",  "5 D",  False, "mes_1m_eth"),
    ("5 mins", "2 M",  True,  "mes_5m_rth"),
    ("5 mins", "2 M",  False, "mes_5m_eth"),
    ("30 mins","1 Y",  True,  "mes_30m_rth"),
    ("30 mins","1 Y",  False, "mes_30m_eth"),
    ("1 hour", "2 Y",  True,  "mes_60m_rth"),
]
for bar, dur, rth, tag in jobs:
    hist(mes, bar, dur, rth, tag) if False else hist(mes, dur, bar, rth, tag)

# Cross-asset
spy = Stock("SPY", "SMART", "USD"); [spy] = ib.qualifyContracts(spy)
hist(spy, "1 Y", "15 mins", True, "spy_15m_rth")
try:
    vix = Index("VIX", "CBOE"); [vix] = ib.qualifyContracts(vix)
    hist(vix, "1 Y", "15 mins", True, "vix_15m_rth", what="TRADES")
except Exception as e:
    print("  vix failed:", e)

ib.disconnect()
print("done")
