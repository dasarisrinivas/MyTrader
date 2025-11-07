#!/bin/bash

################################################################################
# MyTrader - Complete System Startup
# 
# This script starts ALL components:
# 1. Dashboard Backend API (FastAPI on port 8000)
# 2. Dashboard Frontend (React on port 5173)
# 3. Live Trading Bot (main.py)
#
# Prerequisites:
# - IB Gateway or TWS must be running on port 4002
# - config.yaml must be properly configured
# - Virtual environment must be set up
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/dashboard/backend"
FRONTEND_DIR="$PROJECT_ROOT/dashboard/frontend"
VENV_PATH="$PROJECT_ROOT/.venv"
CONFIG_FILE="$PROJECT_ROOT/config.yaml"
IB_PORT=4002

# Banner
clear
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}║         ${BOLD}🚀 MyTrader - Complete System Startup 🚀${NC}${CYAN}           ║${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Parse arguments
SKIP_CONFIRM=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --yes|-y)
            SKIP_CONFIRM=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [--yes] [--config <config_file>]"
            echo "  --yes, -y    Skip confirmation prompts"
            echo "  --config     Specify custom config file"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}                    Step 1: Pre-flight Checks                   ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check virtual environment
echo -ne "${BLUE}[1/6]${NC} Checking virtual environment... "
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}FAILED${NC}"
    echo -e "${RED}❌ Virtual environment not found at $VENV_PATH${NC}"
    echo "Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi
echo -e "${GREEN}OK${NC}"

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Check config file
echo -ne "${BLUE}[2/6]${NC} Checking config file... "
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}FAILED${NC}"
    echo -e "${RED}❌ Config file not found: $CONFIG_FILE${NC}"
    echo "Please create config.yaml from config.example.yaml"
    exit 1
fi
echo -e "${GREEN}OK${NC}"

# Check IB Gateway
echo -ne "${BLUE}[3/6]${NC} Checking IB Gateway/TWS... "
if ! lsof -Pi :$IB_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}NOT RUNNING${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  WARNING: IB Gateway/TWS is NOT running on port $IB_PORT${NC}"
    echo ""
    echo "To enable live trading, you must:"
    echo "  1. Start IB Gateway or TWS"
    echo "  2. Select 'Paper Trading' mode (recommended)"
    echo "  3. Configure API settings:"
    echo "     - Edit > Global Configuration > API > Settings"
    echo "     - Enable ActiveX and Socket Clients"
    echo "     - Set Socket port to $IB_PORT"
    echo "     - Add 127.0.0.1 to Trusted IPs"
    echo "     - ${BOLD}UNCHECK 'Read-Only API'${NC}"
    echo "  4. Login with your credentials"
    echo ""
    if [ "$SKIP_CONFIRM" = false ]; then
        read -p "Continue anyway? (Dashboard will work, but trading bot will fail) [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled. Start IB Gateway/TWS and try again."
            exit 0
        fi
    fi
    echo -e "${YELLOW}⚠️  Proceeding without IB Gateway (bot will not be able to trade)${NC}"
else
    echo -e "${GREEN}OK${NC}"
fi

# Check Python dependencies
echo -ne "${BLUE}[4/6]${NC} Checking Python dependencies... "
MISSING_DEPS=()
for dep in fastapi uvicorn ib_insync websockets; do
    if ! python3 -c "import ${dep//-/_}" 2>/dev/null; then
        MISSING_DEPS+=("$dep")
    fi
done
if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${YELLOW}INSTALLING${NC}"
    echo -e "${BLUE}[INFO]${NC} Installing missing dependencies: ${MISSING_DEPS[*]}"
    pip install -q ${MISSING_DEPS[@]}
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${GREEN}OK${NC}"
fi

# Check Node dependencies
echo -ne "${BLUE}[5/6]${NC} Checking Node dependencies... "
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo -e "${YELLOW}INSTALLING${NC}"
    echo -e "${BLUE}[INFO]${NC} Installing frontend dependencies (this may take a minute)..."
    cd "$FRONTEND_DIR"
    npm install --silent
    cd "$PROJECT_ROOT"
    echo -e "${GREEN}✅ Frontend dependencies installed${NC}"
else
    echo -e "${GREEN}OK${NC}"
fi

# Check if services are already running
echo -ne "${BLUE}[6/6]${NC} Checking for existing services... "
EXISTING_SERVICES=false
if pgrep -f "dashboard_api.py" > /dev/null 2>&1; then
    EXISTING_SERVICES=true
fi
if pgrep -f "main.py live" > /dev/null 2>&1; then
    EXISTING_SERVICES=true
fi
if pgrep -f "vite" > /dev/null 2>&1; then
    EXISTING_SERVICES=true
fi

if [ "$EXISTING_SERVICES" = true ]; then
    echo -e "${YELLOW}FOUND${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  Some services are already running!${NC}"
    if [ "$SKIP_CONFIRM" = false ]; then
        read -p "Stop existing services and restart? [Y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            echo -e "${BLUE}[INFO]${NC} Stopping existing services..."
            pkill -f "dashboard_api.py" 2>/dev/null || true
            pkill -f "main.py live" 2>/dev/null || true
            pkill -f "vite" 2>/dev/null || true
            sleep 2
            echo -e "${GREEN}✅ Existing services stopped${NC}"
        fi
    else
        echo -e "${BLUE}[INFO]${NC} Stopping existing services..."
        pkill -f "dashboard_api.py" 2>/dev/null || true
        pkill -f "main.py live" 2>/dev/null || true
        pkill -f "vite" 2>/dev/null || true
        sleep 2
    fi
else
    echo -e "${GREEN}OK${NC}"
fi

echo ""
echo -e "${GREEN}✅ All pre-flight checks passed${NC}"
echo ""

# Final confirmation
if [ "$SKIP_CONFIRM" = false ]; then
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}                        ⚠️  FINAL WARNING ⚠️                        ${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "This will start:"
    echo "  1. ${BOLD}Dashboard Backend${NC} - API server on port 8000"
    echo "  2. ${BOLD}Dashboard Frontend${NC} - React UI on port 5173"
    echo "  3. ${BOLD}Trading Bot${NC} - Live trading with real/paper account"
    echo ""
    echo -e "${RED}The trading bot will execute REAL trades if configured!${NC}"
    echo -e "${YELLOW}Make sure you are using PAPER TRADING mode.${NC}"
    echo ""
    read -p "Start all components now? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}                    Step 2: Starting Services                   ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Start backend
echo -e "${CYAN}[1/3]${NC} ${BOLD}Starting Dashboard Backend API...${NC}"
cd "$BACKEND_DIR"
nohup python3 dashboard_api.py > "$PROJECT_ROOT/logs/backend.log" 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > "$PROJECT_ROOT/logs/backend.pid"
echo -e "      ${GREEN}✅ Backend started (PID: $BACKEND_PID)${NC}"
echo -e "      ${BLUE}URL:${NC} http://localhost:8000"
echo -e "      ${BLUE}Logs:${NC} logs/backend.log"

# Wait for backend to be ready
echo -ne "      ${BLUE}Waiting for backend to be ready${NC}"
for i in {1..15}; do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo -e " ${GREEN}OK${NC}"
        break
    fi
    echo -n "."
    sleep 1
    if [ $i -eq 15 ]; then
        echo -e " ${RED}FAILED${NC}"
        echo -e "${RED}❌ Backend failed to start. Check logs/backend.log${NC}"
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi
done

echo ""

# Start frontend
echo -e "${CYAN}[2/3]${NC} ${BOLD}Starting Dashboard Frontend...${NC}"
cd "$FRONTEND_DIR"
nohup npm run dev > "$PROJECT_ROOT/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > "$PROJECT_ROOT/logs/frontend.pid"
echo -e "      ${GREEN}✅ Frontend started (PID: $FRONTEND_PID)${NC}"
echo -e "      ${BLUE}URL:${NC} http://localhost:5173"
echo -e "      ${BLUE}Logs:${NC} logs/frontend.log"

# Wait for frontend to be ready
echo -ne "      ${BLUE}Waiting for frontend to be ready${NC}"
for i in {1..10}; do
    echo -n "."
    sleep 1
done
echo -e " ${GREEN}OK${NC}"

echo ""

# Start trading bot
echo -e "${CYAN}[3/3]${NC} ${BOLD}Starting Trading Bot...${NC}"
cd "$PROJECT_ROOT"
nohup python3 main.py live > "$PROJECT_ROOT/logs/bot.log" 2>&1 &
BOT_PID=$!
echo $BOT_PID > "$PROJECT_ROOT/logs/bot.pid"
echo -e "      ${GREEN}✅ Bot started (PID: $BOT_PID)${NC}"
echo -e "      ${BLUE}Config:${NC} $CONFIG_FILE"
echo -e "      ${BLUE}Logs:${NC} logs/bot.log"

# Wait a moment for bot to initialize
echo -ne "      ${BLUE}Initializing trading bot${NC}"
for i in {1..5}; do
    echo -n "."
    sleep 1
done
echo -e " ${GREEN}OK${NC}"

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}                  ✅ All Services Running! ✅                    ${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BOLD}Service Status:${NC}"
echo -e "  📊 ${CYAN}Dashboard Frontend:${NC}  http://localhost:5173"
echo -e "  🔌 ${CYAN}Backend API:${NC}         http://localhost:8000"
echo -e "  📡 ${CYAN}WebSocket:${NC}           ws://localhost:8000/ws"
echo -e "  📚 ${CYAN}API Docs:${NC}            http://localhost:8000/docs"
echo -e "  🤖 ${CYAN}Trading Bot:${NC}         Active (PID: $BOT_PID)"
echo ""
echo -e "${BOLD}Process IDs:${NC}"
echo -e "  Backend:  $BACKEND_PID"
echo -e "  Frontend: $FRONTEND_PID"
echo -e "  Bot:      $BOT_PID"
echo ""
echo -e "${BOLD}Log Files:${NC}"
echo -e "  Backend:  logs/backend.log"
echo -e "  Frontend: logs/frontend.log"
echo -e "  Bot:      logs/bot.log"
echo -e "  Trading:  logs/live_trading.log"
echo ""

# Save process info
cat > "$PROJECT_ROOT/logs/all_services.info" << EOF
BACKEND_PID=$BACKEND_PID
FRONTEND_PID=$FRONTEND_PID
BOT_PID=$BOT_PID
STARTED_AT=$(date)
CONFIG_FILE=$CONFIG_FILE
EOF

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}                       Quick Commands                          ${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${CYAN}Stop all services:${NC}        ./stop.sh"
echo -e "  ${CYAN}View bot logs:${NC}            tail -f logs/bot.log"
echo -e "  ${CYAN}View backend logs:${NC}        tail -f logs/backend.log"
echo -e "  ${CYAN}View trading logs:${NC}        tail -f logs/live_trading.log"
echo -e "  ${CYAN}Check service status:${NC}     ps aux | grep -E 'dashboard_api|main.py live|vite'"
echo ""

# Open browser (macOS)
if command -v open &> /dev/null; then
    sleep 2
    echo -e "${GREEN}Opening dashboard in your browser...${NC}"
    open http://localhost:5173
elif command -v xdg-open &> /dev/null; then
    sleep 2
    echo -e "${GREEN}Opening dashboard in your browser...${NC}"
    xdg-open http://localhost:5173
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}                        Monitoring Options                      ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
read -p "What would you like to do? [1=Watch bot logs, 2=Watch all logs, 3=Exit] " -n 1 -r
echo
echo ""

case $REPLY in
    1)
        echo -e "${CYAN}Showing bot logs (Ctrl+C to exit):${NC}"
        echo ""
        tail -f "$PROJECT_ROOT/logs/bot.log"
        ;;
    2)
        echo -e "${CYAN}Showing all logs (Ctrl+C to exit):${NC}"
        echo ""
        tail -f "$PROJECT_ROOT/logs/backend.log" "$PROJECT_ROOT/logs/frontend.log" "$PROJECT_ROOT/logs/bot.log"
        ;;
    *)
        echo -e "${GREEN}All services are running in the background.${NC}"
        echo -e "${YELLOW}Remember to run ./stop.sh when you're done!${NC}"
        echo ""
        ;;
esac
