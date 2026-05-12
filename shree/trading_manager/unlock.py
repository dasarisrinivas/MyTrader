"""Trading Manager unlock CLI.

The Layer 1.5 health monitor will transition the TM to LOCKED posture when
3+ multi-day health triggers fire (weekly drawdown, rolling WR collapse,
W/L ratio collapse, multi-day red streak). In LOCKED state every entry signal
is rejected — the TM has decided the strategy looks broken, not just unlucky.

Manual unlock is the human-in-the-loop checkpoint. To clear LOCKED:

    cd ~/Documents/code/ShreeBot
    python -m shree.trading_manager.unlock --reason "checked the strategy, ATR widened"

This writes a marker file (`logs/trading_manager_unlock.marker`) that the TM
daemon picks up on its next refresh cycle (within 5 seconds) and transitions
LOCKED → PROBATION. In PROBATION:

  • Exactly 1 probe trade is allowed
  • Probe must have confidence >= 0.85
  • Probe must use 'small' size
  • If the probe wins:  posture → NORMAL, health → HEALTHY
  • If the probe loses: posture → LOCKED again, requires another unlock

This pattern prevents an immediate rip-cord-pull retry after a clear strategy
breakdown. You unlock when you've reasoned about what's happening; you don't
unlock just to "see if it works now."

Usage:
    python -m shree.trading_manager.unlock --reason "<your reason>"
    python -m shree.trading_manager.unlock --status   # print current state

The --reason is recorded in the marker file and the manager log, so you have
a paper trail of every unlock decision.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

from .config import CONFIG


def _print_status() -> int:
    """Print current TM state — health, posture, lock info."""
    state_path = CONFIG.state_file
    if not os.path.exists(state_path):
        print(f"No state file at {state_path}. Is the TM running?")
        return 1
    try:
        with open(state_path, "r") as f:
            s = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Could not read state: {e}")
        return 1

    print("=" * 70)
    print("  Trading Manager — Current State")
    print("=" * 70)
    print(f"Session date:      {s.get('session_date', '?')}")
    print(f"Posture:           {s.get('posture', '?')}")
    print(f"Health status:     {s.get('health_status', '?')}")
    triggers = s.get("health_triggers", [])
    print(f"Health triggers:   {', '.join(triggers) if triggers else '(none)'}")
    print(f"Health summary:    {s.get('health_summary', '?')}")
    print(f"Last health check: {s.get('health_last_check', '?')}")
    print(f"PnL today:         ${s.get('realized_pnl_today', 0):+.2f}")
    print(f"Trades today:      {s.get('trades_today', 0)}")
    print(f"Streak (session):  {s.get('consec_wins', 0)}W / {s.get('consec_losses', 0)}L")
    if s.get("posture") == "LOCKED":
        print(f"\n🔒 LOCKED since:  {s.get('lock_since', '?')}")
        print(f"   Lock reason:   {s.get('lock_reason', '?')}")
        print(f"\nTo unlock:")
        print(f'   python -m shree.trading_manager.unlock --reason "your reason"')
    elif s.get("posture") == "PROBATION":
        print(f"\n⚠️  PROBATION since:  {s.get('probation_started', '?')}")
        print(f"   Trades attempted: {s.get('probation_trade_count', 0)} / 1")
        print(f"\n   Awaiting probe-trade outcome. Win → NORMAL. Loss → LOCKED.")
    return 0


def _unlock(reason: str) -> int:
    """Write the unlock marker file. The TM daemon picks it up automatically."""
    if not reason or not reason.strip():
        print("ERROR: --reason is required and must be non-empty.")
        print("Example: python -m shree.trading_manager.unlock --reason 'verified ATR widened — soft pause is now too tight'")
        return 1

    state_path = CONFIG.state_file
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                s = json.load(f)
            posture = s.get("posture", "?")
            if posture != "LOCKED":
                print(f"⚠️  Current posture is {posture}, not LOCKED.")
                print(f"   An unlock marker is only meaningful when posture=LOCKED.")
                print(f"   The TM will see the marker and ignore it.")
                resp = input("Write marker anyway? [y/N]: ").strip().lower()
                if resp not in ("y", "yes"):
                    print("Cancelled.")
                    return 0
        except (OSError, json.JSONDecodeError):
            print("Could not read state — proceeding anyway.")

    marker_path = CONFIG.unlock_marker_file
    os.makedirs(os.path.dirname(marker_path) or ".", exist_ok=True)
    payload = f"{datetime.now().astimezone().isoformat(timespec='seconds')} | {reason.strip()}\n"
    try:
        with open(marker_path, "w") as f:
            f.write(payload)
    except OSError as e:
        print(f"ERROR: could not write marker file {marker_path}: {e}")
        return 2

    print(f"✓ Unlock marker written: {marker_path}")
    print(f"   Reason: {reason.strip()}")
    print()
    print("The TM will detect this on its next refresh (~5s) and transition")
    print("LOCKED → PROBATION. Watch the log:")
    print(f"   tail -f {CONFIG.log_file}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m shree.trading_manager.unlock",
        description="Unlock the Trading Manager from LOCKED posture and enter PROBATION.",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default=None,
        help="Required: why are you unlocking? Recorded in the marker and the log.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current TM state instead of writing a marker.",
    )
    args = parser.parse_args(argv)

    if args.status:
        return _print_status()
    if args.reason is None:
        parser.print_help()
        print()
        print("Tip: run with --status to see current state first.")
        return 1
    return _unlock(args.reason)


if __name__ == "__main__":
    sys.exit(main())
