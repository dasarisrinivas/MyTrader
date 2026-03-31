#!/bin/bash

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#          SPY Options Signal Bot - Stop Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Usage:
#   . stop_spy_options.sh           # graceful SIGTERM (waits up to 10s)
#   . stop_spy_options.sh --force   # immediate SIGKILL
#
# Signal-only bot — no positions to flatten, so SIGTERM exits cleanly.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FORCE=false
for _ARG in "$@"; do
    case "$_ARG" in --force|-f) FORCE=true ;; esac
done

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

LOGS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"

echo ""
echo -e "${BLUE}Stopping SPY Options bot...${NC}"
echo ""

_graceful_kill() {
    local pid=$1 timeout=${2:-10}
    if ! kill -0 "$pid" 2>/dev/null; then
        echo -e "   SPY Options bot (PID $pid) — already stopped"
        return 1
    fi
    if [ "$FORCE" = "true" ]; then
        kill -9 "$pid" 2>/dev/null
        echo -e "${GREEN}✅ SPY Options bot stopped (PID $pid) [forced]${NC}"
        return 0
    fi
    echo -e "${BLUE}   Sending SIGTERM to PID $pid...${NC}"
    kill -SIGTERM "$pid" 2>/dev/null
    local c=0
    while [ "$c" -lt "$timeout" ]; do
        kill -0 "$pid" 2>/dev/null || {
            echo -e "${GREEN}✅ SPY Options bot stopped gracefully (PID $pid)${NC}"
            return 0
        }
        sleep 1
        c=$((c + 1))
    done
    echo -e "${YELLOW}   SIGTERM timeout — sending SIGKILL...${NC}"
    kill -9 "$pid" 2>/dev/null
    echo -e "${GREEN}✅ SPY Options bot stopped (PID $pid) [forced after ${timeout}s]${NC}"
}

# ── PID file ──────────────────────────────────────────────────────────────────
stopped=false
if [ -f "$LOGS_DIR/spy_options.pid" ]; then
    PID=$(cat "$LOGS_DIR/spy_options.pid" 2>/dev/null)
    if [ -n "$PID" ]; then
        _graceful_kill "$PID" 10 && stopped=true
    fi
    rm -f "$LOGS_DIR/spy_options.pid"
fi

# ── Fallback: scan by process name ───────────────────────────────────────────
if [ "$stopped" = "false" ]; then
    for p in $(pgrep -f "python.*run_spy_options.py" 2>/dev/null); do
        _graceful_kill "$p" 5 && stopped=true
    done
fi

if [ "$stopped" = "false" ]; then
    echo -e "   SPY Options bot — not running"
fi

echo ""
echo -e "${GREEN}✅ Done${NC}"
echo ""
