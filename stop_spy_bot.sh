#!/usr/bin/env bash
# SPY Options Bot — graceful stop
#
# Usage:
#   ./stop_spy_bot.sh           # Graceful SIGTERM (waits up to 10s)
#   ./stop_spy_bot.sh --force   # Immediate SIGKILL

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PID="$SCRIPT_DIR/logs/spy_options_bot.pid"
FORCE=false

[[ "${1:-}" == "--force" || "${1:-}" == "-f" ]] && FORCE=true

echo ""
echo "Stopping SPY Options Bot..."

_kill() {
    local pid=$1
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "   PID $pid — already stopped"
        return 1
    fi
    if [[ "$FORCE" == "true" ]]; then
        kill -9 "$pid" 2>/dev/null
        echo "SPY Options Bot stopped (PID $pid) [forced]"
        return 0
    fi
    echo "   Sending SIGTERM to PID $pid..."
    kill -SIGTERM "$pid" 2>/dev/null
    local c=0
    while (( c < 10 )); do
        kill -0 "$pid" 2>/dev/null || { echo "SPY Options Bot stopped (PID $pid)"; return 0; }
        sleep 1
        (( c++ ))
    done
    kill -9 "$pid" 2>/dev/null
    echo "SPY Options Bot stopped (PID $pid) [forced after 10s]"
}

stopped=false

# Try PID file first
if [[ -f "$LOG_PID" ]]; then
    _kill "$(cat "$LOG_PID")" && stopped=true
    rm -f "$LOG_PID"
fi

# Fallback: find by process name (isolated to spy_options_bot/main.py)
if [[ "$stopped" == "false" ]]; then
    for p in $(pgrep -f 'python.*spy_options_bot/main\.py' 2>/dev/null); do
        _kill "$p" && stopped=true
    done
fi

[[ "$stopped" == "false" ]] && echo "   SPY Options Bot — not running"

echo ""
