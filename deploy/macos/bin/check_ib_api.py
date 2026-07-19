#!/usr/bin/env python3
"""Deep IB API health probe: TCP connect, API handshake, account summary.

Exit codes: 0 = healthy, 1 = unhealthy, 2 = probe misconfigured.
Used by wait_for_api.sh, watchdog.sh and boot_verify.sh. Uses a dedicated
clientId so it never collides with the trading bot's connection.
"""
import argparse
import socket
import sys


def main() -> int:
    p = argparse.ArgumentParser(description="IB API health probe")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4002)
    p.add_argument("--client-id", type=int, default=97)
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--summary", action="store_true",
                   help="print account summary lines on success")
    args = p.parse_args()

    # Cheap TCP check first so we fail fast when the port is closed.
    try:
        with socket.create_connection((args.host, args.port), timeout=5):
            pass
    except OSError as exc:
        print(f"UNHEALTHY: tcp connect {args.host}:{args.port} failed: {exc}")
        return 1

    try:
        from ib_insync import IB, util  # noqa: F401
    except ImportError:
        print("PROBE ERROR: ib_insync not installed in this interpreter")
        return 2

    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id,
                   timeout=args.timeout, readonly=True)
        rows = ib.accountSummary()
        accounts = ib.managedAccounts()
        if not accounts:
            print("UNHEALTHY: connected but no managed accounts reported")
            return 1
        print(f"HEALTHY: account(s) {','.join(accounts)}, "
              f"{len(rows)} summary rows")
        if args.summary:
            for r in rows:
                if r.tag in ("NetLiquidation", "AvailableFunds", "BuyingPower"):
                    print(f"  {r.account} {r.tag}={r.value} {r.currency}")
        return 0
    except Exception as exc:
        print(f"UNHEALTHY: api handshake/summary failed: {exc}")
        return 1
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
