#!/bin/bash

################################################################################
# MyTrader - All-in-One Startup Script
# 
# This script starts everything needed for the trading system:
# - Checks IB Gateway connection (optional, warns if not running)
# - Starts FastAPI Backend (Dashboard API)
# - Starts React Frontend (Web UI)
# - Opens browser automatically
# 
# Usage:
#   ./start.sh              # Start all services
#   ./start.sh --no-browser # Skip opening browser
################################################################################

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
BACKEND_DIR="$PROJECT_ROOT/dashboard/backend"
FRONTEND_DIR="$PROJECT_ROOT/dashboard/frontend"
BACKEND_PORT=8000
FRONTEND_PORT=5173
IB_GATEWAY_PORT=4002
OPEN_BROWSER=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-browser)
            OPEN_BROWSER=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-browser]"
            exit 1
            ;;
    esac
done

# Logging function
log() {
    echo -e "${BLUE}[$1]${NC} $2"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Banner
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}          🚀 MyTrader - Automated Trading System 🚀              ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

################################################################################
# PRE-FLIGHT CHECKS
################################################################################

log "CHECK" "Running pre-flight checks..."

# Check if Python virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    error "Virtual environment not found at $VENV_PATH"
    echo "Please run: python -m venv .venv"
    exit 1
fi
success "Virtual environment found"

# Check Node.js
if ! command -v node &> /dev/null; then
    # Try Homebrew location
    if [ -f "/opt/homebrew/bin/node" ]; then
        export PATH="/opt/homebrew/bin:$PATH"
        success "Node.js found at /opt/homebrew/bin/node ($(node --version))"
    else
        error "Node.js not found"
        echo "Please install Node.js: brew install node"
        exit 1
    fi
else
    success "Node.js found ($(node --version))"
fi

# Check npm
if ! command -v npm &> /dev/null; then
    error "npm not found"
    exit 1
fi
success "npm found ($(npm --version))"

# Check if ports are available
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 1
    else
        return 0
    fi
}

if ! check_port $BACKEND_PORT; then
    warn "Port $BACKEND_PORT is already in use (killing process...)"
    lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null || true
    sleep 1
fi

if ! check_port $FRONTEND_PORT; then
    warn "Port $FRONTEND_PORT is already in use (killing process...)"
    lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Check IB Gateway (optional)
log "CHECK" "Checking IB Gateway connection..."
if check_port $IB_GATEWAY_PORT; then
    warn "IB Gateway doesn't appear to be running on port $IB_GATEWAY_PORT"
    warn "Trading functionality will not work without IB Gateway"
    echo ""
    echo "To start IB Gateway:"
    echo "  1. Open IB Gateway application"
    echo "  2. Select 'Paper Trading' mode"
    echo "  3. Configure to use port $IB_GATEWAY_PORT"
    echo "  4. Enable API connections in settings"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Start IB Gateway and try again."
        exit 1
    fi
else
    success "IB Gateway is running on port $IB_GATEWAY_PORT"
fi

################################################################################
# DEPENDENCIES INSTALLATION
################################################################################

log "SETUP" "Installing/updating dependencies..."

# Python dependencies
source "$VENV_PATH/bin/activate"
pip install -q -r "$PROJECT_ROOT/requirements.txt"
success "Python dependencies installed"

# Node.js dependencies
cd "$FRONTEND_DIR"
if [ ! -d "node_modules" ]; then
    log "SETUP" "Installing npm packages (this may take a minute)..."
    npm install
else
    log "SETUP" "npm packages already installed (skipping)"
fi
success "Node.js dependencies installed"

################################################################################
# START SERVICES
################################################################################

echo ""
log "START" "Starting MyTrader services..."
echo ""

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Kill any existing processes
pkill -f "dashboard_api.py" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
sleep 1

# Start Backend
log "BACKEND" "Starting FastAPI server..."
cd "$PROJECT_ROOT"
source "$VENV_PATH/bin/activate"
nohup python "$BACKEND_DIR/dashboard_api.py" > "$PROJECT_ROOT/logs/backend.log" 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > "$PROJECT_ROOT/logs/backend.pid"

# Wait for backend to start
sleep 3
if kill -0 $BACKEND_PID 2>/dev/null; then
    success "Backend started (PID: $BACKEND_PID) - http://localhost:$BACKEND_PORT"
else
    error "Backend failed to start. Check logs/backend.log"
    exit 1
fi

# Start Frontend
log "FRONTEND" "Starting React development server..."
cd "$FRONTEND_DIR"
nohup npm run dev > "$PROJECT_ROOT/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > "$PROJECT_ROOT/logs/frontend.pid"

# Wait for frontend to start
sleep 4
if kill -0 $FRONTEND_PID 2>/dev/null; then
    success "Frontend started (PID: $FRONTEND_PID) - http://localhost:$FRONTEND_PORT"
else
    error "Frontend failed to start. Check logs/frontend.log"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

################################################################################
# SUCCESS MESSAGE
################################################################################

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}                    ✨ All Systems Running! ✨                   ${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${BLUE}📊 Dashboard UI:${NC}     http://localhost:$FRONTEND_PORT"
echo -e "  ${BLUE}🔌 Backend API:${NC}      http://localhost:$BACKEND_PORT"
echo -e "  ${BLUE}📚 API Docs:${NC}         http://localhost:$BACKEND_PORT/docs"
echo -e "  ${BLUE}💾 Backend Logs:${NC}     tail -f logs/backend.log"
echo -e "  ${BLUE}💾 Frontend Logs:${NC}    tail -f logs/frontend.log"
echo ""
echo -e "${YELLOW}Control:${NC}"
echo -e "  • Stop all:          ./stop.sh"
echo -e "  • View logs:         tail -f logs/*.log"
echo -e "  • Restart:           ./stop.sh && ./start.sh"
echo ""

# Open browser
if [ "$OPEN_BROWSER" = true ]; then
    log "BROWSER" "Opening dashboard in browser..."
    sleep 2
    if command -v open &> /dev/null; then
        open "http://localhost:$FRONTEND_PORT"
    elif command -v xdg-open &> /dev/null; then
        xdg-open "http://localhost:$FRONTEND_PORT"
    else
        warn "Could not open browser automatically"
        echo "Please open: http://localhost:$FRONTEND_PORT"
    fi
fi

echo ""
echo -e "${GREEN}🎯 MyTrader is ready! Press Ctrl+C or run './stop.sh' to stop${NC}"
echo ""

# Save process info
cat > "$PROJECT_ROOT/logs/services.info" <<EOF
BACKEND_PID=$BACKEND_PID
FRONTEND_PID=$FRONTEND_PID
BACKEND_PORT=$BACKEND_PORT
FRONTEND_PORT=$FRONTEND_PORT
STARTED_AT=$(date)
EOF

# Keep script running and wait for Ctrl+C
trap "echo '' && echo 'Stopping services...' && $PROJECT_ROOT/stop.sh && exit 0" INT TERM

# Monitor services
while true; do
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        error "Backend crashed! Check logs/backend.log"
        kill $FRONTEND_PID 2>/dev/null || true
        exit 1
    fi
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        error "Frontend crashed! Check logs/frontend.log"
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi
    sleep 5
done
