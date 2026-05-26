#!/bin/bash
# Recovery script for the hung MES bot (PID written to logs/bot.pid).
# Generated 2026-05-19 — restart after the bot went silent post-IB-reconnect.
#
# Sequence:
#   1. Capture a `sample` stack dump of the hung PID for post-mortem
#   2. Send SIGTERM, wait 5s, escalate to SIGKILL if still alive
#   3. Clean the stale PID file
#   4. Re-launch via start_bot.sh
#
# Safe to double-click in Finder — it always returns to the project dir
# and never executes a trade or moves money.

set -u
cd "$(dirname "$0")"
echo "=============================================================="
echo " Shree MES-bot recovery — $(date +'%Y-%m-%d %H:%M:%S %Z')"
echo "=============================================================="

PID_FILE="logs/bot.pid"
DUMP_DIR="logs/diagnostics"
mkdir -p "$DUMP_DIR"

if [ ! -f "$PID_FILE" ]; then
    echo "  ℹ️  No PID file at $PID_FILE — nothing to kill."
    HUNG_PID=""
else
    HUNG_PID="$(cat "$PID_FILE" 2>/dev/null)"
fi

if [ -n "${HUNG_PID:-}" ] && kill -0 "$HUNG_PID" 2>/dev/null; then
    DUMP_FILE="$DUMP_DIR/mes_hang_$(date +'%Y-%m-%dT%H-%M-%S')_pid${HUNG_PID}.txt"
    echo ""
    echo "  📸 Capturing 3-second sample of PID $HUNG_PID → $DUMP_FILE"
    /usr/bin/sample "$HUNG_PID" 3 -file "$DUMP_FILE" >/dev/null 2>&1 || \
        echo "  ⚠️  sample(1) failed — continuing anyway"
    echo "     (saved $(wc -l < "$DUMP_FILE" 2>/dev/null || echo 0) lines)"

    echo ""
    echo "  🛑 Sending SIGTERM to $HUNG_PID"
    kill -TERM "$HUNG_PID" 2>/dev/null || true
    for i in 1 2 3 4 5; do
        sleep 1
        if ! kill -0 "$HUNG_PID" 2>/dev/null; then
            echo "  ✅ Process exited after SIGTERM ($i s)"
            break
        fi
    done
    if kill -0 "$HUNG_PID" 2>/dev/null; then
        echo "  ⚠️  Still alive after 5s — escalating to SIGKILL"
        kill -KILL "$HUNG_PID" 2>/dev/null || true
        sleep 1
    fi
else
    echo "  ℹ️  PID $HUNG_PID is not running — already dead, just cleaning up"
fi

# Stale pid file → remove so start_bot.sh doesn't refuse to start
rm -f "$PID_FILE"
echo ""
echo "=============================================================="
echo " Restarting MES bot via start_bot.sh"
echo "=============================================================="
echo ""

# start_bot.sh expects to be sourced (`. start_bot.sh`) — it returns
# instead of exit so the wrapper terminal stays open. We run it under
# bash so it can spawn the background nohup process and we still get
# its output.
bash ./start_bot.sh
RC=$?

echo ""
echo "=============================================================="
echo " start_bot.sh exited with code $RC"
echo "=============================================================="
if [ -f "$PID_FILE" ]; then
    echo "  New PID: $(cat "$PID_FILE")"
fi
echo ""
echo "Tailing logs/live_trading.log for 10s — Ctrl+C to exit early"
echo "--------------------------------------------------------------"
( tail -F logs/live_trading.log 2>/dev/null & TAIL_PID=$!; sleep 10; kill "$TAIL_PID" 2>/dev/null ) || true

echo ""
echo "Done. Press Return to close this window."
read -r _
