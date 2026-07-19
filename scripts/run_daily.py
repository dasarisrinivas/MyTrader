#!/usr/bin/env python3
"""launchd entry point for the SPY daily start/stop jobs (JUL 17 2026).

WHY THIS EXISTS: macOS TCC blocks /bin/bash under launchd from reading
~/Documents ("Operation not permitted", exit 126) — the SPY daily start/stop
jobs silently died this way while the MES/TM jobs kept working because their
plists invoke the venv python3 directly, and THAT binary holds the Documents
grant. This wrapper gives the SPY jobs the same identity: launchd runs
python3 (granted), which then runs the existing shell script as a child of a
permitted responsible process.

Usage (from the plists): run_daily.py start | stop
"""

import subprocess
import sys

ROOT = "/Users/svss/Documents/code/ShreeBot"
SCRIPTS = {
    "start": ["/bin/bash", f"{ROOT}/scripts/start_trading_day.sh"],
    "stop": ["/bin/bash", f"{ROOT}/scripts/stop_trading_day.sh"],
    # Weekly deterministic portfolio governor (Saturdays; tier review +
    # Telegram report + data/family_tiers.json for the executor).
    "weekly": [sys.executable, f"{ROOT}/scripts/portfolio_governor.py"],
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in SCRIPTS:
        print("usage: run_daily.py start|stop|weekly", file=sys.stderr)
        return 2
    return subprocess.run(SCRIPTS[sys.argv[1]], cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
