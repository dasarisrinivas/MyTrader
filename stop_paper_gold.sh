#!/bin/bash

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#          Gold Futures Strategy - Stop Script (PAPER)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Usage:
#   . stop_paper_gold.sh             # graceful SIGTERM (waits up to 10s)
#   . stop_paper_gold.sh --force     # immediate SIGKILL
#
# Only stops the paper gold bot (logs/paper_gold.pid).
# Does NOT touch the live gold bot or MES bots.
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

LOGS_DIR="${PAPER_GOLD_LOG_DIR:-logs}"
PID_FILE="${LOGS_DIR}/paper_gold.pid"

echo ""
echo -e "${BLUE}Stopping paper gold bot...${NC}"
echo ""

_graceful_kill() {
    local pid=$1 timeout=${2:-10}
    if ! kill -0 "$pid" 2>/dev/null; then
        echo -e "   Paper gold bot (PID $pid) — already stopped"
        return 1
    fi
    if [ "$FORCE" = "true" ]; then
        kill -9 "$pid" 2>/dev/null
        echo -e "${GREEN}✅ Paper gold bot stopped (PID $pid) [forced]${NC}"
        return 0
    fi
    echo -e "${BLUE}   Sending SIGTERM to PID $pid ...${NC}"
    kill -SIGTERM "$pid" 2>/dev/null
    local c=0
    while [ "$c" -lt "$timeout" ]; do
        kill -0 "$pid" 2>/dev/null || {
            echo -e "${GREEN}✅ Paper gold bot stopped gracefully (PID $pid)${NC}"
            return 0
        }
        sleep 1
        c=$((c + 1))
    done
    echo -e "${YELLOW}   SIGTERM timeout — sending SIGKILL...${NC}"
    kill -9 "$pid" 2>/dev/null
    echo -e "${GREEN}✅ Paper gold bot stopped (PID $pid) [forced after ${timeout}s]${NC}"
}

stopped=false

# ── PID file ──────────────────────────────────────────────────────────
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$PID" ]; then
        _graceful_kill "$PID" 10 && stopped=true
    fi
    rm -f "$PID_FILE"
fi

if [ "$stopped" = "false" ]; then
    echo -e "   Paper gold bot — not running"
fi

echo ""
echo -e "${GREEN}✅ Done${NC}"
echo ""
