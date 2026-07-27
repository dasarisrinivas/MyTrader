#!/usr/bin/env python3
"""Analyze logs/greeks_quality.jsonl — the Phase-1 IB greek-timing instrumentation.

Answers the timing-vs-permanent question for missing greeks once ~30-50 sessions
have accrued:
  * how many missing-greek events are 'no_quote' (dead/illiquid — unrecoverable)
    vs 'quote_no_greek' (real contract, greeks absent — candidate for a retry)?
  * of the 'quote_no_greek' events, did greeks EVER arrive within the wait
    (first_greek_ms not null)? If a quote arrived early but greeks never did,
    a longer wait / retry is the plausible fix — NOT ThetaData.

Read-only. No trading.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

PATH = sys.argv[1] if len(sys.argv) > 1 else "logs/greeks_quality.jsonl"


def main():
    rows = []
    try:
        with open(PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except FileNotFoundError:
        print(f"no data yet at {PATH} — run the bot with greeks_quality_log on")
        return
    n = len(rows)
    if not n:
        print("file empty")
        return
    cls = Counter(r.get("class") for r in rows)
    print(f"missing-greek events logged: {n}")
    for k, v in cls.most_common():
        print(f"  {k}: {v} ({100*v//n}%)")

    qng = [r for r in rows if r.get("class") == "quote_no_greek"]
    print(f"\n=== quote_no_greek (the retry candidates): {len(qng)} ===")
    if qng:
        got = [r for r in qng if r.get("first_greek_ms") is not None]
        print(f"  greeks DID arrive within window but after read? {len(got)}"
              f" ({100*len(got)//len(qng)}%)  <- these a longer read recovers")
        print(f"  greeks NEVER arrived in window: {len(qng)-len(got)}"
              f"  <- need a LONGER wait to know; Phase-3 retry test")
        fq = [r["first_quote_ms"] for r in qng if r.get("first_quote_ms") is not None]
        if fq:
            fq.sort()
            print(f"  first_quote_ms: median={fq[len(fq)//2]} "
                  f"p90={fq[int(len(fq)*0.9)]} (quotes arrive fast => greeks lag)")
        fg = [r["first_greek_ms"] for r in got]
        if fg:
            fg.sort()
            print(f"  first_greek_ms (when seen): median={fg[len(fg)//2]} "
                  f"max={fg[-1]}  vs greeks_wait_s={rows[0].get('greeks_wait_s')}s")
    print("\nInterpretation:")
    print("  high quote_no_greek + quotes-fast/greeks-late => retry/longer wait is")
    print("  the fix (infrastructure), NOT ThetaData. high no_quote => dead")
    print("  contracts, nothing recovers them.")


if __name__ == "__main__":
    main()
