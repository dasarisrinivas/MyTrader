#!/bin/bash

################################################################################
# MyTrader - Dashboard Only Start
# 
# This script starts ONLY the dashboard (backend API and frontend)
# NOTE: Trading bot must be started separately with: python3 main.py live --config config.yaml
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/dashboard/backend"
FRONTEND_DIR="$PROJECT_ROOT/dashboard/frontend"
VENV_PATH="$PROJECT_ROOT/.venv"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}          � MyTrader - Dashboard Only �                      ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check virtual environment
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}❌ Virtual environment not found${NC}"
    echo "Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if backend dependencies are installed
echo -e "${BLUE}[INFO]${NC} Checking backend dependencies..."
source "$VENV_PATH/bin/activate"

if ! python3 -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  FastAPI not installed, installing...${NC}"
    pip install -q fastapi uvicorn websockets
fi

# Check if frontend dependencies are installed
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo -e "${YELLOW}⚠️  Frontend dependencies not installed${NC}"
    echo -e "${BLUE}[INFO]${NC} Installing frontend dependencies..."
    cd "$FRONTEND_DIR"
    npm install
    cd "$PROJECT_ROOT"
fi

echo -e "${GREEN}✅ All dependencies installed${NC}"
echo ""

# Check if trading bot is running
if pgrep -f "python3 main.py live" > /dev/null; then
    echo -e "${GREEN}✅ Trading bot is already running${NC}"
else
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}                        ⚠️  WARNING ⚠️                           ${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Trading bot is NOT running!${NC}"
    echo ""
    echo "Dashboard will start, but you won't see live trading data."
    echo ""
    echo "To start the trading bot:"
    echo -e "  ${GREEN}python3 main.py live --config config.yaml${NC}"
    echo ""
    read -p "Continue with dashboard only? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled. Start the trading bot first."
        exit 0
    fi
fi

# Check IB Gateway (informational only)
IB_PORT=4002
if ! lsof -Pi :$IB_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo ""
    echo -e "${YELLOW}ℹ️  IB Gateway/TWS not detected on port $IB_PORT${NC}"
    echo "   (This is OK if the trading bot is handling the connection)"
else
    echo -e "${GREEN}✅ IB Gateway/TWS is running${NC}"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}                   🎯 Starting Services 🎯                      ${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Kill any existing processes
pkill -f "dashboard_api.py" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
sleep 1

# Start backend
echo -e "${BLUE}[INFO]${NC} Starting backend API on http://localhost:8000..."
cd "$BACKEND_DIR"
source "$VENV_PATH/bin/activate"
nohup python3 dashboard_api.py > "$PROJECT_ROOT/logs/backend.log" 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > "$PROJECT_ROOT/logs/backend.pid"
echo -e "${GREEN}✅ Backend started (PID: $BACKEND_PID)${NC}"

# Wait for backend to be ready
echo -e "${BLUE}[INFO]${NC} Waiting for backend to be ready..."
for i in {1..10}; do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend is ready${NC}"
        break
    fi
    sleep 1
    if [ $i -eq 10 ]; then
        echo -e "${RED}❌ Backend failed to start. Check logs/backend.log${NC}"
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi
done

# Start frontend
echo -e "${BLUE}[INFO]${NC} Starting frontend on http://localhost:5173..."
cd "$FRONTEND_DIR"
nohup npm run dev > "$PROJECT_ROOT/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > "$PROJECT_ROOT/logs/frontend.pid"
echo -e "${GREEN}✅ Frontend started (PID: $FRONTEND_PID)${NC}"

# Wait a bit for frontend to start
sleep 3

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}                   ✅ Dashboard Running! ✅                      ${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "📊 ${BLUE}Frontend:${NC} http://localhost:5173"
echo -e "🔌 ${BLUE}Backend API:${NC} http://localhost:8000"
echo -e "📡 ${BLUE}WebSocket:${NC} ws://localhost:8000/ws"
echo -e "📝 ${BLUE}API Docs:${NC} http://localhost:8000/docs"
echo ""
echo -e "${BLUE}Logs:${NC}"
echo -e "  Backend:  logs/backend.log"
echo -e "  Frontend: logs/frontend.log"
echo ""
echo -e "${YELLOW}To stop services, run: ./stop.sh${NC}"
echo ""
echo -e "${GREEN}Opening dashboard in your browser...${NC}"

# Save process info
cat > "$PROJECT_ROOT/logs/services.info" << EOF
BACKEND_PID=$BACKEND_PID
FRONTEND_PID=$FRONTEND_PID
STARTED_AT=$(date)
EOF

# Open browser (works on macOS)
sleep 2
if command -v open &> /dev/null; then
    open http://localhost:5173
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:5173
fi

echo ""
echo -e "${BLUE}[INFO]${NC} Dashboard is running. Press Ctrl+C to view logs (services will continue running)"
echo -e "${BLUE}[INFO]${NC} To stop all services: ./stop.sh"
echo ""

# Optionally tail logs
read -p "Show live logs? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    tail -f "$PROJECT_ROOT/logs/backend.log" "$PROJECT_ROOT/logs/frontend.log"
fi
