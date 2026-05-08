#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#       🛡️  Shree - Start Trading Manager Daemon 🛡️
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Autonomous oversight layer that watches MES + SPY Options bots,
# applies the institutional decision framework, vetoes bad signals,
# and force-pauses bots if the daily loss hard-stop is hit.
#
# Usage: ./start_trading_manager.sh
#
# Env vars (optional):
#   TM_ACCOUNT_EQUITY  — current account equity in USD (default 4499.77)
#   TM_DAILY_LOSS_PCT  — hard daily loss % (default 0.03)
#   TM_DRY_RUN=1       — log kill events without actually SIGTERM'ing bots
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

cd "$(dirname "$0")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Pick interpreter
if [ -z "$PYTHON_BIN" ]; then
    if [ -x ".venv/bin/python3" ]; then
        PYTHON_BIN=".venv/bin/python3"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo -e "${RED}❌ python3 not found${NC}"
        exit 1
    fi
fi

mkdir -p logs

# Already running?
if [ -f "logs/trading_manager.pid" ]; then
    EXISTING_PID=$(cat logs/trading_manager.pid 2>/dev/null || echo "")
    if [ -n "$EXISTING_PID" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Trading Manager is already running (PID $EXISTING_PID)${NC}"
        echo "   Stop with: ./stop_trading_manager.sh"
        exit 1
    fi
    rm -f logs/trading_manager.pid
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}          🛡️  Trading Manager — Starting${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Equity:        ${TM_ACCOUNT_EQUITY:-4499.77}"
echo "Daily stop:    ${TM_DAILY_LOSS_PCT:-0.03} of equity"
echo "Dry run:       ${TM_DRY_RUN:-0}"
echo "Log file:      logs/trading_manager.log"
echo "Decisions out: logs/manager_decisions.jsonl"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

nohup "$PYTHON_BIN" run_trading_manager.py >> logs/trading_manager_nohup.log 2>&1 &
TM_PID=$!
echo "$TM_PID" > logs/trading_manager.pid

sleep 1

if kill -0 "$TM_PID" 2>/dev/null; then
    echo -e "${GREEN}✓ Trading Manager started (PID $TM_PID)${NC}"
    echo ""
    echo "Tail logs:    tail -f logs/trading_manager.log"
    echo "Decisions:    tail -f logs/manager_decisions.jsonl"
    echo "Stop:         ./stop_trading_manager.sh"
    echo ""
else
    echo -e "${RED}❌ Trading Manager failed to start. Check logs/trading_manager_nohup.log${NC}"
    rm -f logs/trading_manager.pid
    exit 1
fi
