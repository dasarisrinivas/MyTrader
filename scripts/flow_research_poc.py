#!/usr/bin/env python3
"""Flow-research POC driver — observation only, no trading.

Pipeline:
  init      create the flow-research DB (data/flow_research.db)
  ingest    load raw prints from a CSV export into spy_flow_prints
  snapshot  attach flow snapshots to signals / rejected / intervals
  validate  run the battery and print the report

Typical zero-cost POC run (CSV replay of a historical OPRA export):
  python3 scripts/flow_research_poc.py init
  python3 scripts/flow_research_poc.py ingest --csv path/to/prints.csv
  python3 scripts/flow_research_poc.py snapshot \
      --spy-db /Users/.../data/spy_options_signals.db \
      --blocked logs/blocked_signals.jsonl
  python3 scripts/flow_research_poc.py validate \
      --spy-db /Users/.../data/spy_options_signals.db

Nothing here imports the executor, manager, governor, or signal engine.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shree.flow_research.schema import open_db, insert_prints
from shree.flow_research.ingest import CsvReplaySource
from shree.flow_research.thetadata import (
    ThetaClient, ThetaDataSource, ThetaSubscriptionError,
)
from shree.flow_research.snapshotter import (
    PrintStore,
    snapshot_signals,
    snapshot_rejected,
    snapshot_intervals,
)
from shree.flow_research.validate import run_battery


def cmd_init(args):
    conn = open_db(args.flow_db)
    conn.close()
    print(f"[init] flow-research DB ready at {args.flow_db}")


def cmd_ingest(args):
    conn = open_db(args.flow_db)
    src = CsvReplaySource(args.csv)
    n = insert_prints(conn, src.prints())
    conn.close()
    print(f"[ingest] inserted {n} prints from {args.csv}")


def cmd_ingest_theta(args):
    import datetime as _dt
    start = _dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    end = _dt.datetime.strptime(args.end, "%Y-%m-%d").date()
    client = ThetaClient(args.base_url)
    src = ThetaDataSource(client, start, end, symbol=args.symbol,
                          dte_max=args.dte_max,
                          strikes_around_atm=args.strikes)
    conn = open_db(args.flow_db)
    try:
        n = insert_prints(conn, src.prints())
        print(f"[ingest-theta] inserted {n} prints {args.symbol} "
              f"{args.start}..{args.end}")
    except ThetaSubscriptionError as e:
        print(f"[ingest-theta] SUBSCRIPTION REQUIRED — {str(e)[:120]}")
        print("  trade_quote needs the Standard tier. Subscribe, restart the "
              "terminal with your real credentials, and re-run.")
    finally:
        conn.close()


def cmd_snapshot(args):
    conn = open_db(args.flow_db)
    store = PrintStore(conn)
    n_sig = n_rej = n_int = 0
    if args.spy_db:
        n_sig = snapshot_signals(store, conn, args.spy_db, window_s=args.window_s)
    if args.blocked:
        n_rej = snapshot_rejected(store, conn, args.blocked, window_s=args.window_s)
    if not args.no_intervals:
        n_int = snapshot_intervals(store, conn, window_s=args.window_s)
    conn.close()
    print(f"[snapshot] SIGNAL={n_sig} REJECTED={n_rej} INTERVAL={n_int}")


def cmd_validate(args):
    report = run_battery(args.flow_db, args.spy_db)
    print(json.dumps(report, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description="Flow-research POC (observation only)")
    p.add_argument("--flow-db", default="data/flow_research.db")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init").set_defaults(func=cmd_init)

    pi = sub.add_parser("ingest")
    pi.add_argument("--csv", required=True)
    pi.set_defaults(func=cmd_ingest)

    pt = sub.add_parser("ingest-theta", help="pull real prints from ThetaData")
    pt.add_argument("--start", required=True, help="YYYY-MM-DD")
    pt.add_argument("--end", required=True, help="YYYY-MM-DD")
    pt.add_argument("--symbol", default="SPY")
    pt.add_argument("--dte-max", type=int, default=2)
    pt.add_argument("--strikes", type=int, default=5)
    pt.add_argument("--base-url", default="http://127.0.0.1:25503")
    pt.set_defaults(func=cmd_ingest_theta)

    ps = sub.add_parser("snapshot")
    ps.add_argument("--spy-db", default=None)
    ps.add_argument("--blocked", default=None)
    ps.add_argument("--window-s", type=int, default=1800)
    ps.add_argument("--no-intervals", action="store_true")
    ps.set_defaults(func=cmd_snapshot)

    pv = sub.add_parser("validate")
    pv.add_argument("--spy-db", required=True)
    pv.set_defaults(func=cmd_validate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
