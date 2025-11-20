#!/bin/bash

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#          🤖 MyTrader - Start Bot Only 🤖                      
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}          🤖 MyTrader - Starting Bot Only 🤖${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if bot is already running
if pgrep -f "python.*run_bot.py" > /dev/null; then
    echo -e "${YELLOW}⚠️  Trading bot is already running!${NC}"
    echo ""
    echo "To restart the bot:"
    echo "  1. Run: ./stop.sh"
    echo "  2. Then run: ./start_bot.sh"
    exit 1
fi

# Check if IB Gateway/TWS is running
echo -e "${BLUE}[INFO]${NC} Checking IB Gateway/TWS..."
if lsof -i:4002 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ IB Gateway/TWS is running${NC}"
else
    echo -e "${RED}❌ IB Gateway/TWS is NOT running on port 4002${NC}"
    echo ""
    echo "Please start IB Gateway or TWS before running the bot."
    echo ""
    echo "IB Gateway Setup:"
    echo "  1. Download: https://www.interactivebrokers.com/en/trading/ibgateway-stable.php"
    echo "  2. Login with paper trading account"
    echo "  3. Configure API: Settings > API > Enable ActiveX and Socket Clients"
    echo "  4. Set port: 4002 (paper trading) or 4001 (live)"
    exit 1
fi

# Check if config file exists
if [ ! -f "config.yaml" ]; then
    echo -e "${RED}❌ config.yaml not found!${NC}"
    echo ""
    echo "Copy config.example.yaml to config.yaml and configure it:"
    echo "  cp config.example.yaml config.yaml"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}                   🚀 Starting Bot 🚀${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment if it exists
VENV_PATH="$PWD/.venv"
if [ -d "$VENV_PATH" ]; then
    echo -e "${BLUE}[INFO]${NC} Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo -e "${YELLOW}⚠️  Virtual environment not found at .venv${NC}"
    echo -e "${YELLOW}   Using system Python${NC}"
fi

# Set environment variables for safety
export MAX_CONTRACTS=${MAX_CONTRACTS:-5}
export IBKR_HOST=${IBKR_HOST:-"127.0.0.1"}
export IBKR_PORT=${IBKR_PORT:-4002}

# Start the bot in the background
echo -e "${BLUE}[INFO]${NC} Starting trading bot (MAX_CONTRACTS=$MAX_CONTRACTS)..."
nohup python run_bot.py > logs/bot.log 2>&1 &
BOT_PID=$!

# Wait a bit and check if it's still running
sleep 3
if ps -p $BOT_PID > /dev/null; then
    echo -e "${GREEN}✅ Bot started successfully (PID: $BOT_PID)${NC}"
else
    echo -e "${RED}❌ Bot failed to start. Check logs/bot.log for details${NC}"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}                   ✅ Bot Running! ✅${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${BLUE}📊 Bot PID:${NC} $BOT_PID"
echo -e "${BLUE}📝 Logs:${NC} logs/bot.log"
echo ""
echo -e "${YELLOW}To view live logs:${NC}"
echo "  tail -f logs/bot.log"
echo ""
echo -e "${YELLOW}To stop the bot:${NC}"
echo "  ./stop.sh"
echo ""
