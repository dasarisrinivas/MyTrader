"""Kill-switch — pause the bot processes when daily-loss hard stop trips.

We do this the simple, robust way: send SIGTERM to the PID in logs/bot.pid
and logs/spy_options.pid. The bots have signal handlers that flatten and
shutdown cleanly. If TM_DRY_RUN=1 we log instead of killing.
"""
from __future__ import annotations

import logging
import os
import signal
from typing import List, Tuple

logger = logging.getLogger("trading_manager.bot_control")


def _read_pid(path: str) -> int:
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r") as f:
            s = f.read().strip()
        return int(s) if s else 0
    except (ValueError, OSError):
        return 0


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_bots(pid_files: List[str], dry_run: bool = False) -> List[Tuple[str, int, str]]:
    """Send SIGTERM to each PID. Returns list of (file, pid, status)."""
    results: List[Tuple[str, int, str]] = []
    for pf in pid_files:
        pid = _read_pid(pf)
        if pid == 0:
            results.append((pf, 0, "no_pid"))
            continue
        if not _alive(pid):
            results.append((pf, pid, "already_dead"))
            continue
        if dry_run:
            results.append((pf, pid, "dry_run_would_sigterm"))
            logger.warning("DRY RUN: would SIGTERM pid=%d (file=%s)", pid, pf)
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            results.append((pf, pid, "sigterm_sent"))
            logger.warning("KILL SWITCH: SIGTERM sent to pid=%d (file=%s)", pid, pf)
        except OSError as e:
            results.append((pf, pid, f"error:{e}"))
            logger.error("KILL SWITCH FAILED: pid=%d file=%s err=%s", pid, pf, e)
    return results


def bots_alive(pid_files: List[str]) -> List[Tuple[str, int, bool]]:
    out = []
    for pf in pid_files:
        pid = _read_pid(pf)
        out.append((pf, pid, _alive(pid)))
    return out
