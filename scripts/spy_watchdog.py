"""SPY options bot watchdog — restarts the bot if it dies mid-session.

AUG 4 2026 (Phase 4). Two consecutive sessions lost trading time to a bot that
was down with nothing watching: 2026-08-03 (launchd start died at 08:00:13,
recovered only by a manual 08:48 restart) and 2026-08-04 (died at 08:00:06,
then hung twice — ~108 minutes of session missing).

Deliberately conservative. It only ever ADDS a start; it never stops, kills or
reconfigures anything, and it cannot place an order.

Guards:
  * Only inside the session window, weekdays.
  * Only when no SPY bot process exists (pgrep, same pattern as the launcher).
  * Honours a manual-stop sentinel so it never fights a human who stopped the
    bot on purpose.
  * Rate-limited: at most MAX_RESTARTS_PER_DAY, and never twice within
    MIN_RESTART_GAP_S — a bot that crashes on boot must not be relaunched in a
    tight loop.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, time as dtime

ROOT = "/Users/svss/Documents/code/ShreeBot"
STATE = os.path.join(ROOT, "logs", "spy_watchdog_state.json")
LOG = os.path.join(ROOT, "logs", "spy_watchdog.log")
SENTINEL = os.path.join(ROOT, "logs", "spy_watchdog.disabled")

# Machine runs Central Time; the bot's own session gate is ET.
SESSION_START = dtime(8, 5)      # after the 08:00 launchd start has settled
SESSION_END = dtime(14, 55)      # before the 15:05 daily-stop job

# AUG 6 2026 — FALSE-POSITIVE FIX.
# The bot stops polling at config `session.rth_stop_et` = 15:45 ET = 14:45 CDT,
# but stall checking ran until SESSION_END (14:55 CDT). That left a 10-minute
# window in which the bot's legitimate post-RTH idle looked identical to a hang.
# On 2026-08-06 the watchdog killed a perfectly healthy bot:
#   14:45:35  last poll (normal end of RTH)
#   14:51:16  "bot ALIVE but log silent for 341s (> 300s) — treating as STALLED"
#   14:51:36  stalled bot stopped (pids ['61721']) — restarting
# Nothing was open at the time, but the same event during a live position would
# have SIGKILLed the process managing it. Stall detection now stops BEFORE the
# bot legitimately goes quiet.
STALL_CHECK_END = dtime(14, 40)   # < rth_stop_et (14:45 CDT) with 5-min margin
# Bot polls every 60s during RTH, so a log that has not grown in this long
# means the poll loop is wedged (see stalled()). 5x the poll interval.
POLL_WINDOW_START = dtime(8, 40)   # first poll due 08:35 CDT + margin
STALL_AFTER_S = 300
BOTLOG = os.path.join(ROOT, "logs", "spy_options.log")
MAX_RESTARTS_PER_DAY = 3
MIN_RESTART_GAP_S = 600          # 10 min


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line)
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def load_state() -> dict:
    try:
        with open(STATE, "r", encoding="utf-8") as fh:
            s = json.load(fh)
        if s.get("day") == datetime.now().date().isoformat():
            return s
    except Exception:
        pass
    return {"day": datetime.now().date().isoformat(), "restarts": 0, "last_ts": 0}


def save_state(s: dict) -> None:
    try:
        with open(STATE, "w", encoding="utf-8") as fh:
            json.dump(s, fh)
    except Exception:
        pass


def bot_running() -> bool:
    """Same detection the launcher uses, so the two can never disagree.

    Anchored on "--config" and using [.] so a process whose own command
    line mentions the script cannot self-match (false-positived during
    Phase 4 testing).
    """
    r = subprocess.run(["pgrep", "-f", "run_spy_options[.]py --config"],
                       capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def stalled() -> bool:
    """True if the bot is alive but has stopped making progress.

    AUG 5 2026 — liveness != health. On 2026-08-04 (x2) and 2026-08-05 the bot
    froze on an unbounded IB request: the process stayed alive and responsive to
    signals, so the pgrep liveness check saw nothing wrong, while polling,
    signals and exits had all stopped. Today that cost the session opening.

    During the polling window the log is written at least once per 60s poll, so
    a log that has not grown in STALL_AFTER_S means the poll loop is wedged.
    Only evaluated after the first poll is due — before RTH the bot is
    legitimately idle and silent.
    """
    try:
        # Window ends at STALL_CHECK_END, before the bot's own rth_stop_et —
        # after that a silent log is expected, not a stall (2026-08-06 fix).
        if not (POLL_WINDOW_START <= datetime.now().time() <= STALL_CHECK_END):
            return False
        age = datetime.now().timestamp() - os.path.getmtime(BOTLOG)
        if age > STALL_AFTER_S:
            log(f"bot ALIVE but log silent for {age:.0f}s "
                f"(> {STALL_AFTER_S}s) — treating as STALLED")
            return True
    except Exception as exc:
        log(f"stall check failed (ignoring): {exc}")
    return False


def main() -> int:
    now = datetime.now()

    if now.weekday() > 4:
        return 0
    if not (SESSION_START <= now.time() <= SESSION_END):
        return 0
    if os.path.exists(SENTINEL):
        log("watchdog disabled by sentinel — no action")
        return 0
    if bot_running():
        if not stalled():
            return 0
        # Stalled: stop the wedged process so the normal restart path applies.
        # SIGTERM first; a wedged bot often cannot finish shutdown, so escalate.
        try:
            pids = subprocess.run(
                ["pgrep", "-f", "run_spy_options[.]py --config"],
                capture_output=True, text=True).stdout.split()
            for sig in ("-TERM", "-KILL"):
                for p in pids:
                    subprocess.run(["kill", sig, p], capture_output=True)
                if sig == "-TERM":
                    import time as _t
                    _t.sleep(20)
                    if not bot_running():
                        break
            log(f"stalled bot stopped (pids {pids}) — restarting")
        except Exception as exc:
            log(f"failed to stop stalled bot: {exc}")
            return 1

    state = load_state()
    if state["restarts"] >= MAX_RESTARTS_PER_DAY:
        log(f"bot DOWN but restart budget exhausted "
            f"({state['restarts']}/{MAX_RESTARTS_PER_DAY}) — manual attention needed")
        return 1
    if now.timestamp() - float(state.get("last_ts") or 0) < MIN_RESTART_GAP_S:
        log("bot DOWN but within restart cooldown — waiting")
        return 0

    log("bot DOWN inside session window — restarting")
    try:
        r = subprocess.run(["/bin/bash", "start_spy_options.sh"],
                           cwd=ROOT, capture_output=True, text=True, timeout=120)
        ok = r.returncode == 0
        log(f"start_spy_options.sh returned {r.returncode}")
        if not ok:
            log((r.stdout or "")[-400:])
    except Exception as exc:
        log(f"restart failed: {exc}")
        ok = False

    state["restarts"] += 1
    state["last_ts"] = now.timestamp()
    save_state(state)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
