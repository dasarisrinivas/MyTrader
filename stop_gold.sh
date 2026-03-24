#!/bin/bash

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#          Gold Futures Strategy - Stop Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Usage:
#   . stop_gold.sh            # graceful SIGTERM (waits up to 10s)
#   . stop_gold.sh --force    # immediate SIGKILL
#
# The bot handles SIGTERM by flattening any open position before exit.
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
echo -e "${BLUE}Stopping Gold bot...${NC}"
echo ""

_graceful_kill() {
    local pid=$1 timeout=${2:-10}
    if ! kill -0 "$pid" 2>/dev/null; then
        echo -e "   Gold bot (PID $pid) — already stopped"
        return 1
    fi
    if [ "$FORCE" = "true" ]; then
        kill -9 "$pid" 2>/dev/null
        echo -e "${GREEN}✅ Gold bot stopped (PID $pid) [forced]${NC}"
        return 0
    fi
    echo -e "${BLUE}   Sending SIGTERM to PID $pid (will flatten open position)...${NC}"
    kill -SIGTERM "$pid" 2>/dev/null
    local c=0
    while [ "$c" -lt "$timeout" ]; do
        kill -0 "$pid" 2>/dev/null || {
            echo -e "${GREEN}✅ Gold bot stopped gracefully (PID $pid)${NC}"
            return 0
        }
        sleep 1
        c=$((c + 1))
    done
    echo -e "${YELLOW}   SIGTERM timeout — sending SIGKILL...${NC}"
    kill -9 "$pid" 2>/dev/null
    echo -e "${GREEN}✅ Gold bot stopped (PID $pid) [forced after ${timeout}s]${NC}"
}

# ── PID file ──────────────────────────────────────────────────────────────────
stopped=false
if [ -f "$LOGS_DIR/gold.pid" ]; then
    PID=$(cat "$LOGS_DIR/gold.pid" 2>/dev/null)
    if [ -n "$PID" ]; then
        _graceful_kill "$PID" 10 && stopped=true
    fi
    rm -f "$LOGS_DIR/gold.pid"
fi

# ── Fallback: scan by process name ───────────────────────────────────────────
if [ "$stopped" = "false" ]; then
    for p in $(pgrep -f "python.*run_gold.py" 2>/dev/null); do
        _graceful_kill "$p" 5 && stopped=true
    done
fi

if [ "$stopped" = "false" ]; then
    echo -e "   Gold bot — not running"
fi

echo ""
echo -e "${GREEN}✅ Done${NC}"
echo ""
