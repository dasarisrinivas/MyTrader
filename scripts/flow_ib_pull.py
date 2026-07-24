#!/usr/bin/env python3
"""Pull REAL SPY option prints from IB and load them into the flow-research DB.

Read-only, observation-only proof that the pipeline ingests real data.

  * Connects to IB Gateway with a FRESH client id and readonly=True
    (order placement is impossible on a readonly session).
  * For a small set of near-ATM SPY option contracts on the nearest expiry,
    pulls historical TRADES ticks + BID_ASK ticks (NBBO) for the RTH session.
  * Merges each trade with the prevailing NBBO, computes a Black-Scholes IV/delta
    from the trade, maps to `Print`, classifies, and inserts into
    spy_flow_prints.

HONEST LIMITATIONS of IB as a flow source (vs ThetaData/Databento OPRA):
  * IB reports a CONSOLIDATED last, not simultaneous multi-venue prints —
    so multi-exchange sweep detection is NOT reliable here. is_sweep will be
    mostly false; treat that measure as unavailable from IB.
  * Historical ticks are capped (1000/request) and paced — this pulls a small
    sample, enough to prove ingestion, not a full-session tape.
  * Open/close (oc_estimate) needs OI deltas we don't fetch here -> UNKNOWN.

Nothing here trades. It imports NOTHING from the executor/manager/governor.
"""
from __future__ import annotations

import argparse
import math
import sys
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

from ib_insync import IB, Stock, Option  # noqa: E402

from shree.flow_research.models import Print  # noqa: E402
from shree.flow_research.classify import classify_all, mark_blocks, mark_sweeps  # noqa: E402
from shree.flow_research.schema import open_db, insert_prints  # noqa: E402

ET = ZoneInfo("America/New_York")


# ── tiny Black-Scholes (IV inversion + delta) — for real dw_flow ─────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(S, K, T, sigma, right, r=0.045):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if right == "C" else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if right == "C":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _bs_delta(S, K, T, sigma, right, r=0.045):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1) if right == "C" else _norm_cdf(d1) - 1.0


def _implied_vol(price, S, K, T, right) -> Optional[float]:
    if price <= 0 or T <= 0 or S <= 0:
        return None
    lo, hi = 1e-4, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _bs_price(S, K, T, mid, right) > price:
            hi = mid
        else:
            lo = mid
    iv = 0.5 * (lo + hi)
    return iv if 1e-3 < iv < 4.99 else None


# ── IB pull ──────────────────────────────────────────────────────────────────

def _to_et_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ET).replace(tzinfo=None).isoformat(timespec="milliseconds")


def _page_bidask(ib, contract, end_str, per_page, cover_until, max_pages):
    """Page BID_ASK historical ticks backward until coverage reaches
    `cover_until` (a tz-aware datetime) or `max_pages` requests are spent.
    Returns quotes sorted ascending by time, de-duplicated."""
    seen = {}
    end = end_str
    for _ in range(max_pages):
        batch = ib.reqHistoricalTicks(contract, "", end, per_page,
                                      "BID_ASK", useRth=True)
        if not batch:
            break
        for q in batch:
            seen[q.time] = q
        earliest = batch[0].time
        if cover_until is not None and earliest <= cover_until:
            break
        # walk the window back: next request ends 1s before this batch's start
        end = earliest.astimezone(ET).strftime("%Y%m%d %H:%M:%S US/Eastern")
    return [seen[t] for t in sorted(seen)]


def pull(flow_db: str, client_id: int, n_strikes: int, max_ticks: int,
         host: str, port: int, quote_pages: int = 6) -> int:
    ib = IB()
    print(f"[ib] connecting {host}:{port} clientId={client_id} readonly=True ...")
    ib.connect(host, port, clientId=client_id, timeout=20, readonly=True)
    try:
        spy = Stock("SPY", "SMART", "USD")
        ib.qualifyContracts(spy)

        bars = ib.reqHistoricalData(spy, "", "2 D", "1 day", "TRADES",
                                    useRTH=True, formatDate=1)
        if not bars:
            print("[ib] no SPY daily bar — aborting")
            return 0
        close = float(bars[-1].close)
        last_session = bars[-1].date  # date object
        print(f"[ib] SPY close={close} last RTH session={last_session}")

        params = ib.reqSecDefOptParams(spy.symbol, "", spy.secType, spy.conId)
        # IB returns several SMART param sets: the real tradingClass 'SPY'
        # (481 strikes, near-dated expiries incl. 0DTE) plus a decoy '2SPY'
        # (2 strikes). Pick the real one = SMART + class 'SPY' + most strikes.
        cand = [p for p in params
                if p.exchange == "SMART" and p.tradingClass == "SPY"]
        if not cand:
            print("[ib] no SMART/SPY option params — aborting")
            return 0
        chain = max(cand, key=lambda p: len(p.strikes))
        expirations = sorted(chain.expirations)
        strikes = sorted(float(s) for s in chain.strikes)

        # nearest expiry on/after the last session
        sess_str = last_session.strftime("%Y%m%d")
        future_exp = [e for e in expirations if e >= sess_str]
        expiry = future_exp[0] if future_exp else expirations[-1]

        atm = min(strikes, key=lambda s: abs(s - close))
        atm_idx = strikes.index(atm)
        half = n_strikes // 2
        picks = strikes[max(0, atm_idx - half): atm_idx + half + 1]
        print(f"[ib] expiry={expiry} ATM={atm} strikes={picks}")

        exp_dt = datetime.strptime(expiry, "%Y%m%d").date()
        # session RTH close as the historical-ticks end anchor
        end_str = f"{sess_str} 16:00:00 US/Eastern"

        # SPY underlying trade ticks, for per-print underlying + BS inputs
        spy_trades = ib.reqHistoricalTicks(spy, "", end_str, max_ticks,
                                           "TRADES", useRth=True)
        spy_times = [t.time for t in spy_trades]
        spy_px = [float(t.price) for t in spy_trades]

        def underlying_at(t) -> float:
            if not spy_times:
                return close
            i = bisect_right(spy_times, t) - 1
            return spy_px[i] if i >= 0 else close

        prints: List[Print] = []
        for strike in picks:
            for right in ("C", "P"):
                opt = Option("SPY", expiry, strike, right, "SMART",
                             tradingClass="SPY")
                try:
                    ib.qualifyContracts(opt)
                except Exception as e:
                    print(f"[ib]   qualify fail {strike}{right}: {e}")
                    continue

                trades = ib.reqHistoricalTicks(opt, "", end_str, max_ticks,
                                               "TRADES", useRth=True)
                # NBBO is far denser than trades for 0DTE, so a single 1000-tick
                # quote pull covers only the last minutes. Page BID_ASK backward
                # until quote coverage reaches the earliest trade (or page cap).
                earliest_trade = trades[0].time if trades else None
                quotes = _page_bidask(ib, opt, end_str, max_ticks,
                                      earliest_trade, quote_pages)
                q_times = [q.time for q in quotes]
                cover_start = q_times[0] if q_times else None

                def nbbo_at(t):
                    if not q_times:
                        return None, None
                    i = bisect_right(q_times, t) - 1
                    if i < 0:
                        return None, None
                    q = quotes[i]
                    b, a = float(q.priceBid), float(q.priceAsk)
                    if b <= 0 or a <= 0:
                        return None, None
                    return b, a

                made = 0
                for tk in trades:
                    px = float(tk.price)
                    sz = int(tk.size)
                    if sz <= 0 or px <= 0:
                        continue
                    # only keep trades inside NBBO coverage so aggressor is real
                    if cover_start is not None and tk.time < cover_start:
                        continue
                    bid, ask = nbbo_at(tk.time)
                    S = underlying_at(tk.time)
                    T = max((exp_dt - tk.time.astimezone(ET).date()).days, 0) / 252.0
                    if T == 0:
                        # intraday 0DTE: use fraction of a day so BS is defined
                        T = 0.5 / 252.0
                    iv = _implied_vol(px, S, strike, T, right)
                    delta = _bs_delta(S, strike, T, iv, right) if iv else None
                    ts_et = _to_et_iso(tk.time)
                    prints.append(Print(
                        ts_utc=tk.time.astimezone(timezone.utc).isoformat(
                            timespec="milliseconds"),
                        ts_et=ts_et,
                        session_date=ts_et[:10],
                        root="SPY", expiry=exp_dt.isoformat(),
                        strike=float(strike), right=right,
                        trade_px=px, size=sz,
                        exchange=getattr(tk, "exchange", None) or None,
                        condition_codes=[],
                        underlying_px=S,
                        dte=(exp_dt - tk.time.astimezone(ET).date()).days,
                        bid=bid, ask=ask,
                        delta=delta, iv=iv,
                        greeks_src="COMPUTED" if delta is not None else None,
                        data_source="ibkr_hist_ticks",
                    ))
                    made += 1
                print(f"[ib]   {strike}{right}: trades={len(trades)} "
                      f"quotes={len(quotes)} -> prints={made}")

        if not prints:
            print("[ib] NO option prints returned. Likely no historical OPRA "
                  "tick subscription, or no RTH trades at these strikes.")
            return 0

        classify_all(prints)
        mark_blocks(prints)
        mark_sweeps(prints)  # unreliable on IB consolidated tape (see docstring)

        conn = open_db(flow_db)
        n = insert_prints(conn, prints)
        conn.close()
        print(f"[ib] inserted {n} real prints into {flow_db}")
        return n
    finally:
        ib.disconnect()
        print("[ib] disconnected")


def main(argv=None):
    p = argparse.ArgumentParser(description="Pull real SPY option prints from IB")
    p.add_argument("--flow-db", default="data/flow_research.db")
    p.add_argument("--client-id", type=int, default=137)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4001)
    p.add_argument("--strikes", type=int, default=5, help="# strikes around ATM")
    p.add_argument("--max-ticks", type=int, default=1000)
    p.add_argument("--quote-pages", type=int, default=6,
                   help="max BID_ASK pages per contract (coverage vs pacing)")
    args = p.parse_args(argv)
    pull(args.flow_db, args.client_id, args.strikes, args.max_ticks,
         args.host, args.port, args.quote_pages)


if __name__ == "__main__":
    main()
