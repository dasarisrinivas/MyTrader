#!/usr/bin/env zsh
#
# Shree — Stop Paper Trading Bot
#
# Usage:
#   ./stop_paper_bot.sh
#   ./stop_paper_bot.sh --force
#
# Stops only the paper trading bot started by `start_paper_bot.sh`.
# Does not touch the live bot or shared services outside paper-specific ports/PIDs.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS_DIR="${PAPER_BOT_LOG_DIR:-$SCRIPT_DIR/logs}"
PID_FILE="$LOGS_DIR/paper_bot.pid"
FORCE=false

while (( $# > 0 )); do
    case "$1" in
        --force|-f) FORCE=true ;;
    esac
    shift
done

echo ""
echo "🛑 Stopping Shree paper bot..."
echo ""

_graceful_kill() {
    local name=$1 pid=$2 timeout=${3:-5}
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "   $name (PID $pid) — already stopped"
        return 1
    fi
    if [[ "$FORCE" == "true" ]]; then
        kill -9 "$pid" 2>/dev/null
        echo "✅ $name stopped (PID $pid) [forced]"
        return 0
    fi
    echo "   Stopping $name (PID $pid)..."
    kill -SIGTERM "$pid" 2>/dev/null
    local c=0
    while (( c < timeout )); do
        kill -0 "$pid" 2>/dev/null || { echo "✅ $name stopped (PID $pid)"; return 0; }
        sleep 1
        (( c++ ))
    done
    kill -9 "$pid" 2>/dev/null
    echo "✅ $name stopped (PID $pid) [forced after ${timeout}s]"
}

paper_done=false

# Prefer the paper PID file created by start_paper_bot.sh
if [[ -f "$PID_FILE" ]]; then
    if _graceful_kill "Paper Trading Bot" "$(cat "$PID_FILE")" 5; then
        paper_done=true
        rm -f "$PID_FILE"
    elif ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        rm -f "$PID_FILE"
    fi
fi

# Fallback: find processes explicitly marked as paper or using paper port/config
if [[ "$paper_done" == "false" ]]; then
    for p in $(pgrep -f 'DEPLOY_ENV=paper.*run_bot.py|IBKR_PORT=4002.*run_bot.py|python.*run_bot.py.*--paper' 2>/dev/null); do
        _graceful_kill "Paper Trading Bot" "$p" 3 && paper_done=true
    done
fi

[[ "$paper_done" == "false" ]] && echo "   Paper Trading Bot — not running"

# Free the paper metrics port only; leave live metrics alone.
lsof -ti:8001 2>/dev/null | xargs kill -9 2>/dev/null && echo "✅ Freed paper metrics port 8001"

echo ""
echo "✨ Paper bot stop complete"
echo ""
