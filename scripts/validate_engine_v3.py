#!/usr/bin/env python3
"""V1-V7 validation of Replay Engine v3.0. Must pass before ANY research use.

Test cases: existing debit spread (logs), long straddle, long strangle,
synthetic credit spread. No strategy conclusions.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))
from shree.flow_research.thetadata import ThetaClient
from shree.research.replay_engine import QuoteCache, _prevailing
from shree.research.replay_engine_v3 import (
    parse_structure, synthetic_vertical, replay_multileg, provenance,
    RESEARCH_ENGINE_V3_VERSION, V3_LIMITATION, Leg,
)

ET = ZoneInfo("America/New_York")
DB = "/Users/svss/Documents/code/ShreeBot/data/spy_options_signals.db"
HOLD_MIN = 90
PASS, FAIL = "PASS", "FAIL"
results = {}


def et(t):
    s = t.split(".")[0].rstrip("Z")
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).astimezone(ET)


def k(d):
    return d.replace(tzinfo=None).isoformat(timespec="milliseconds")


def main():
    c = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    c.row_factory = sqlite3.Row
    rows = c.execute(
        """SELECT id, sent_at, exit_at, signal_type, strike, right, expiry_date,
                  suggested_trade FROM spy_signals
           WHERE suggested_trade IS NOT NULL AND expiry_date IS NOT NULL
             AND expiry_date!='' AND signal_type IN
             ('BULL_CALL_SPREAD','BEAR_PUT_SPREAD','LONG_STRADDLE')
           ORDER BY sent_at""").fetchall()
    c.close()
    print(f"multi-leg signals available: {len(rows)}")

    # ── V1 parser / known-spread reconstruction ─────────────────────────────
    print("\n=== V1 leg reconstruction (parser; strike COLUMN must be ignored) ===")
    parsed, v1_bad, col_mismatch = [], 0, 0
    for r in rows:
        st = parse_structure(r["suggested_trade"], r["expiry_date"])
        if st is None:
            v1_bad += 1
            continue
        parsed.append((r, st))
        if all(l.strike != r["strike"] for l in st.legs):
            col_mismatch += 1
    print(f"  parsed {len(parsed)}/{len(rows)}   unparseable {v1_bad}")
    print(f"  rows where strike COLUMN matches NO leg: {col_mismatch} "
          f"(proves column is not a leg)")
    for r, st in parsed[:4]:
        legs = " , ".join(f"{l.side} {l.strike:.0f}{l.right}" for l in st.legs)
        print(f"    id={r['id']} {r['signal_type']:<17} col_strike={r['strike']:<7}"
              f" -> {st.kind}: {legs} exp={st.legs[0].expiry}")
        print(f"       text: {r['suggested_trade'].splitlines()[0][:70]}")
    results["V1_parser"] = PASS if (len(parsed) > 0 and v1_bad == 0) else FAIL

    client = ThetaClient()
    cache = QuoteCache(client)

    def run(struct, entry, exit_, label):
        res = replay_multileg(cache, struct, entry, exit_)
        print(f"  [{label}] ok={res.ok} {res.reason}")
        if res.ok:
            print(f"     kind={res.kind} entry_net={res.entry_net} "
                  f"exit_net={res.exit_net}")
            print(f"     gross=${res.gross_dollar} fees=${res.fees} "
                  f"NET=${res.net_dollar}  MFE=${res.mfe} MAE=${res.mae}")
            print(f"     max_loss=${res.max_loss} max_profit=${res.max_profit} "
                  f"invariant_ok={res.invariant_ok}")
        return res

    # ── test cases ──────────────────────────────────────────────────────────
    print("\n=== TEST CASES ===")
    cases, ids = [], []
    # 1. existing debit spread from logs (most recent parseable vertical)
    for r, st in reversed(parsed):
        if st.kind == "vertical":
            e = et(r["sent_at"])
            x = et(r["exit_at"]) if r["exit_at"] else e + timedelta(minutes=HOLD_MIN)
            if x.date() != e.date():
                x = e.replace(hour=15, minute=55, second=0)
            res = run(st, e, x, f"1 debit-spread id={r['id']}")
            if res.ok:
                cases.append(("existing_debit_spread", st, e, x, res))
                ids.append(r["id"])
                break
    # 2. long straddle from logs
    for r, st in reversed(parsed):
        if st.kind == "straddle":
            e = et(r["sent_at"])
            x = et(r["exit_at"]) if r["exit_at"] else e + timedelta(minutes=HOLD_MIN)
            if x.date() != e.date():
                x = e.replace(hour=15, minute=55, second=0)
            res = run(st, e, x, f"2 straddle id={r['id']}")
            if res.ok:
                cases.append(("long_straddle", st, e, x, res))
                ids.append(r["id"])
                break
    # 3. long strangle (synthetic from a known liquid session)
    ref = cases[0] if cases else None
    sess_e = ref[2] if ref else None
    if sess_e:
        exp = ref[1].legs[0].expiry
        base = round(ref[1].legs[0].strike)
        from shree.research.replay_engine_v3 import Structure
        strangle = Structure(kind="strangle", legs=[
            Leg("buy", "C", base + 3, exp), Leg("buy", "P", base - 3, exp)],
            source="synthetic")
        res = run(strangle, sess_e, ref[3], "3 strangle (synthetic)")
        if res.ok:
            cases.append(("long_strangle", strangle, sess_e, ref[3], res))
    # 4. synthetic CREDIT spread (no historical credit signals exist)
    if sess_e:
        exp = ref[1].legs[0].expiry
        base = round(ref[1].legs[0].strike)
        credit = synthetic_vertical("C", base + 5, base, exp)  # long far/short near
        res = run(credit, sess_e, ref[3], "4 credit-spread (synthetic)")
        if res.ok:
            cases.append(("synthetic_credit_spread", credit, sess_e, ref[3], res))

    results["TESTCASES"] = PASS if len(cases) >= 4 else FAIL

    # ── V2 leg synchronization ──────────────────────────────────────────────
    print("\n=== V2 leg synchronization (both legs quoted at same instant) ===")
    sync_ok = True
    for name, st, e, x, res in cases:
        for leg in st.legs:
            q, kk = cache.get(leg.expiry, leg.strike, leg.right, e.strftime("%Y%m%d"))
            b, a = _prevailing(q, kk, k(e))
            if not b or not a:
                sync_ok = False
        print(f"  {name}: legs synchronized at entry = {sync_ok}")
    results["V2_sync"] = PASS if sync_ok else FAIL

    # ── V3/V4 entry & exit pricing verified independently ───────────────────
    print("\n=== V3/V4 entry & exit pricing (independent recompute) ===")
    v34 = True
    for name, st, e, x, res in cases:
        ent = exi = 0.0
        for leg in st.legs:
            q, kk = cache.get(leg.expiry, leg.strike, leg.right, e.strftime("%Y%m%d"))
            eb, ea = _prevailing(q, kk, k(e))
            xb, xa = _prevailing(q, kk, k(x))
            ent += leg.sign * (ea if leg.side == "buy" else eb)
            exi += leg.sign * (xb if leg.side == "buy" else xa)
        de, dx = abs(ent - res.entry_net), abs(exi - res.exit_net)
        ok = de < 1e-6 and dx < 1e-6
        v34 &= ok
        print(f"  {name}: entry {ent:.4f} vs {res.entry_net} | "
              f"exit {exi:.4f} vs {res.exit_net} -> {'OK' if ok else 'MISMATCH'}")
    results["V3_entry_pricing"] = PASS if v34 else FAIL
    results["V4_exit_pricing"] = PASS if v34 else FAIL

    # ── V5 P&L math + invariant bounds ──────────────────────────────────────
    print("\n=== V5 P&L math manually validated + invariant bounds ===")
    v5 = True
    for name, st, e, x, res in cases:
        manual = (res.exit_net - res.entry_net) * 100.0 - res.fees
        ok = abs(manual - res.net_dollar) < 0.01 and res.invariant_ok
        v5 &= ok
        print(f"  {name}: manual ${manual:.2f} vs engine ${res.net_dollar} "
              f"| invariant_ok={res.invariant_ok} -> {'OK' if ok else 'FAIL'}")
    results["V5_pnl_math"] = PASS if v5 else FAIL

    # ── V6 determinism ──────────────────────────────────────────────────────
    print("\n=== V6 determinism (same input -> same output) ===")
    det = True
    for name, st, e, x, res in cases:
        again = replay_multileg(cache, st, e, x)
        same = (again.net_dollar == res.net_dollar and
                again.entry_net == res.entry_net and again.mfe == res.mfe)
        det &= same
        print(f"  {name}: reproducible={same}")
    results["V6_determinism"] = PASS if det else FAIL

    # ── V7 provenance ───────────────────────────────────────────────────────
    print("\n=== V7 provenance ===")
    prov = provenance(ids)
    for kk2, vv in prov.items():
        print(f"  {kk2}: {vv}")
    results["V7_provenance"] = PASS if prov["engine_v3_hash"] else FAIL

    print("\n=== SUMMARY ===")
    for kk2, vv in results.items():
        print(f"  {kk2:<22} {vv}")
    allpass = all(v == PASS for v in results.values())
    print(f"\nKNOWN LIMITATION: {V3_LIMITATION}")
    print(f"\nv3 is {'MEASUREMENT-READY' if allpass else 'NOT READY'}")
    Path("data").mkdir(exist_ok=True)
    with open("data/v3_validation.json", "w") as f:
        json.dump({"version": RESEARCH_ENGINE_V3_VERSION, "results": results,
                   "provenance": prov, "limitation": V3_LIMITATION,
                   "measurement_ready": allpass}, f, indent=2)


if __name__ == "__main__":
    main()
