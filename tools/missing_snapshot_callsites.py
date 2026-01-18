#!/usr/bin/env python3
"""Summarize root-order snapshot-missing warnings by callsite.

This parses MyTrader logs for warnings emitted by OrderTracker when a ROOT order
is recorded with a trade_cycle_id but missing features/rationale snapshots.

Expected log snippet (single line):
  Order placement missing features/rationale (root order). ... caller=/path/file.py:123 func

Usage:
  python3 tools/missing_snapshot_callsites.py --log logs/bot.log

Options:
  --json  prints structured JSON output for piping.
  --limit limits number of callsites printed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


WARNING_NEEDLE = "Order placement missing features/rationale (root order)."

CALLER_RE = re.compile(r"\bcaller=(?P<file>[^:\s]+):(?P<line>\d+)\s+(?P<func>\S+)")
TRADE_CYCLE_RE = re.compile(r"\btrade_cycle_id=(?P<tc>[a-zA-Z0-9_-]+)")
ORDER_ID_RE = re.compile(r"\border_id=(?P<oid>\d+)")
SYMBOL_RE = re.compile(r"\bsymbol=(?P<sym>[A-Z0-9._-]+)")


@dataclass(frozen=True)
class WarningHit:
    callsite: str
    file: str
    line: int
    func: str
    trade_cycle_id: Optional[str]
    order_id: Optional[int]
    symbol: Optional[str]


def iter_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            yield line.rstrip("\n")


def parse_warning(line: str) -> Optional[WarningHit]:
    if WARNING_NEEDLE not in line:
        return None

    cm = CALLER_RE.search(line)
    if not cm:
        # Callsite tracing not enabled or warning line got wrapped.
        return None

    file = cm.group("file")
    line_no = int(cm.group("line"))
    func = cm.group("func")
    callsite = f"{file}:{line_no} {func}"

    tc = None
    om = ORDER_ID_RE.search(line)
    sm = SYMBOL_RE.search(line)
    tm = TRADE_CYCLE_RE.search(line)

    order_id = int(om.group("oid")) if om else None
    symbol = sm.group("sym") if sm else None
    if tm:
        tc = tm.group("tc")

    return WarningHit(
        callsite=callsite,
        file=file,
        line=line_no,
        func=func,
        trade_cycle_id=tc,
        order_id=order_id,
        symbol=symbol,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/bot.log", help="Path to bot.log")
    ap.add_argument("--limit", type=int, default=25, help="Max callsites to print")
    ap.add_argument("--json", action="store_true", help="Emit JSON")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"Log not found: {log_path}")

    hits: list[WarningHit] = []
    by_callsite: Counter[str] = Counter()

    for line in iter_lines(log_path):
        hit = parse_warning(line)
        if not hit:
            continue
        hits.append(hit)
        by_callsite[hit.callsite] += 1

    if args.json:
        payload = {
            "log": str(log_path),
            "total_hits": len(hits),
            "unique_callsites": len(by_callsite),
            "callsites": [
                {"callsite": cs, "count": cnt}
                for cs, cnt in by_callsite.most_common(args.limit)
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"log: {log_path}")
    print(f"hits: {len(hits)}")
    print(f"unique callsites: {len(by_callsite)}")
    print("")

    for callsite, count in by_callsite.most_common(args.limit):
        print(f"{count:5d}  {callsite}")

    if len(by_callsite) > args.limit:
        print(f"... ({len(by_callsite) - args.limit} more)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
