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

# Fallback: kill by process name
pkill -f "dashboard_api.py" 2>/dev/null && echo -e "${GREEN}✅ Stopped dashboard_api.py${NC}"
pkill -f "vite" 2>/dev/null && echo -e "${GREEN}✅ Stopped vite${NC}"

# Kill by port if needed
lsof -ti:8000 | xargs kill -9 2>/dev/null && echo -e "${GREEN}✅ Freed port 8000${NC}"
lsof -ti:5173 | xargs kill -9 2>/dev/null && echo -e "${GREEN}✅ Freed port 5173${NC}"

# Clean up
rm -f "$LOGS_DIR/services.info"

echo ""
echo -e "${GREEN}✨ All services stopped${NC}"
echo ""
