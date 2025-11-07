#!/bin/bash

# MyTrader Live Trading Startup Script
# Usage: ./start_live_trading.sh

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   MyTrader Live Trading Startup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if IB Gateway is running (optional check)
echo -e "${YELLOW}Checking IB Gateway connection...${NC}"
if ! nc -z localhost 4002 2>/dev/null; then
    echo -e "${RED}Warning: IB Gateway doesn't appear to be running on port 4002${NC}"
    echo -e "${YELLOW}Please start IB Gateway before continuing.${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Startup cancelled.${NC}"
        exit 1
    fi
fi

# Kill any existing trading bot processes
echo -e "${YELLOW}Checking for existing bot processes...${NC}"
if pgrep -f "main.py live" > /dev/null; then
    echo -e "${YELLOW}Found existing bot process. Stopping it...${NC}"
    pkill -9 -f "main.py live"
    sleep 5
    echo -e "${GREEN}✓ Existing process stopped${NC}"
    echo -e "${YELLOW}Waiting for IB Gateway to release client ID...${NC}"
    sleep 3
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found at .venv${NC}"
    echo -e "${YELLOW}Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt${NC}"
    exit 1
fi

source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Check if required packages are installed
echo -e "${YELLOW}Checking dependencies...${NC}"
python -c "import ib_insync, boto3, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Missing required packages${NC}"
    echo -e "${YELLOW}Installing requirements...${NC}"
    pip install -r requirements.txt
fi
echo -e "${GREEN}✓ Dependencies OK${NC}"

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p data

# Start the live trading bot
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Starting Live Trading Bot...${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  - IB Gateway: localhost:4002"
echo -e "  - LLM: AWS Bedrock Claude 3 Sonnet"
echo -e "  - Mode: Consensus (Traditional + AI)"
echo -e "  - Min Confidence: 0.7"
echo ""
echo -e "${YELLOW}Logs will be saved to: logs/live_trading.log${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the bot${NC}"
echo ""

# Run the bot
python main.py live

# Cleanup message (only shows if bot exits normally)
echo ""
echo -e "${YELLOW}Bot stopped.${NC}"
