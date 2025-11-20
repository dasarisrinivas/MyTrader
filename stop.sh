#!/bin/bash

################################################################################
# MyTrader - Stop Script
# Gracefully stops all MyTrader services
################################################################################

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="$PROJECT_ROOT/logs"

echo ""
echo -e "${BLUE}🛑 Stopping MyTrader services...${NC}"
echo ""

# Kill processes by PID if available
if [ -f "$LOGS_DIR/backend.pid" ]; then
    BACKEND_PID=$(cat "$LOGS_DIR/backend.pid")
    if kill -0 $BACKEND_PID 2>/dev/null; then
        kill $BACKEND_PID
        echo -e "${GREEN}✅ Stopped backend (PID: $BACKEND_PID)${NC}"
    fi
    rm "$LOGS_DIR/backend.pid"
fi

if [ -f "$LOGS_DIR/frontend.pid" ]; then
    FRONTEND_PID=$(cat "$LOGS_DIR/frontend.pid")
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        kill $FRONTEND_PID
        echo -e "${GREEN}✅ Stopped frontend (PID: $FRONTEND_PID)${NC}"
    fi
    rm "$LOGS_DIR/frontend.pid"
fi

if [ -f "$LOGS_DIR/bot.pid" ]; then
    BOT_PID=$(cat "$LOGS_DIR/bot.pid")
    if kill -0 $BOT_PID 2>/dev/null; then
        echo -e "${YELLOW}Gracefully stopping trading bot...${NC}"
        kill -SIGTERM $BOT_PID
        sleep 3
        # Force kill if still running
        if kill -0 $BOT_PID 2>/dev/null; then
            kill -9 $BOT_PID
            echo -e "${GREEN}✅ Stopped bot (PID: $BOT_PID) [forced]${NC}"
        else
            echo -e "${GREEN}✅ Stopped bot (PID: $BOT_PID)${NC}"
        fi
    fi
    rm "$LOGS_DIR/bot.pid"
fi

# Fallback: kill by process name
pkill -f "dashboard_api.py" 2>/dev/null && echo -e "${GREEN}✅ Stopped dashboard_api.py${NC}"
pkill -f "vite" 2>/dev/null && echo -e "${GREEN}✅ Stopped vite${NC}"
pkill -f "main.py live" 2>/dev/null && echo -e "${GREEN}✅ Stopped main.py${NC}"
pkill -f "run_bot.py" 2>/dev/null && echo -e "${GREEN}✅ Stopped run_bot.py${NC}"
pkill -f "run_autonomous_trading.py" 2>/dev/null && echo -e "${GREEN}✅ Stopped autonomous trading${NC}"
pkill -f "run_llm_trading.py" 2>/dev/null && echo -e "${GREEN}✅ Stopped LLM trading${NC}"

# Force kill any remaining Python processes related to MyTrader
MYTRADER_PROCS=$(ps aux | grep -i "mytrader" | grep -v "grep" | awk '{print $2}')
if [ ! -z "$MYTRADER_PROCS" ]; then
    for PID in $MYTRADER_PROCS; do
        kill -9 $PID 2>/dev/null && echo -e "${GREEN}✅ Stopped MyTrader process (PID: $PID)${NC}"
    done
fi

# Kill by port if needed (backend, frontend, and any other services)
lsof -ti:8000 | xargs kill -9 2>/dev/null && echo -e "${GREEN}✅ Freed port 8000 (backend)${NC}"
lsof -ti:5173 | xargs kill -9 2>/dev/null && echo -e "${GREEN}✅ Freed port 5173 (frontend)${NC}"
lsof -ti:8001 | xargs kill -9 2>/dev/null && echo -e "${GREEN}✅ Freed port 8001${NC}"
lsof -ti:4001 | xargs kill -9 2>/dev/null && echo -e "${GREEN}✅ Freed port 4001 (IB Gateway)${NC}"
lsof -ti:7497 | xargs kill -9 2>/dev/null && echo -e "${GREEN}✅ Freed port 7497 (TWS API)${NC}"

# Clean up
rm -f "$LOGS_DIR/services.info"
rm -f "$LOGS_DIR/all_services.info"

echo ""
echo -e "${GREEN}✨ All services stopped${NC}"
echo -e "${BLUE}Dashboard and trading bot are no longer running${NC}"
echo ""
