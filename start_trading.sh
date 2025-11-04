#!/bin/bash

################################################################################
# MyTrader - Live Trading Startup Script
# 
# This script starts LIVE TRADING (not the dashboard).
# The trading bot will connect to IBKR and execute trades based on signals.
#
# Prerequisites:
# - IB Gateway or TWS must be running
# - config.yaml must be properly configured
# - Paper trading or live account credentials set up
#
# Usage:
#   ./start_trading.sh              # Start live trading with default config
#   ./start_trading.sh --config my_config.yaml  # Use custom config
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
CONFIG_FILE="$PROJECT_ROOT/config.yaml"
IB_PORT=4002

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config <config_file>]"
            exit 1
            ;;
    esac
done

# Banner
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}          🤖 MyTrader - Starting Live Trading Bot 🤖            ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check virtual environment
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}❌ Virtual environment not found at $VENV_PATH${NC}"
    echo "Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi
echo -e "${GREEN}✅ Virtual environment found${NC}"

# Check config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Config file not found: $CONFIG_FILE${NC}"
    echo "Please create config.yaml from config.example.yaml"
    exit 1
fi
echo -e "${GREEN}✅ Config file found: $CONFIG_FILE${NC}"

# Check IB Gateway
if ! lsof -Pi :$IB_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo ""
    echo -e "${RED}❌ IB Gateway/TWS is NOT running on port $IB_PORT${NC}"
    echo ""
    echo "You must start IB Gateway or Trader Workstation first:"
    echo "  1. Open IB Gateway or TWS"
    echo "  2. Select 'Paper Trading' mode (recommended) or Live"
    echo "  3. Configure API settings:"
    echo "     - Edit > Global Configuration > API > Settings"
    echo "     - Enable ActiveX and Socket Clients"
    echo "     - Set Socket port to $IB_PORT"
    echo "     - Add 127.0.0.1 to Trusted IPs"
    echo "     - ** IMPORTANT: Uncheck 'Read-Only API' **"
    echo "  4. Login with your credentials"
    echo ""
    read -p "Continue anyway? (NOT RECOMMENDED) [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Start IB Gateway/TWS and try again."
        exit 1
    fi
    echo -e "${YELLOW}⚠️  Proceeding without IB Gateway (trades will fail)${NC}"
else
    echo -e "${GREEN}✅ IB Gateway/TWS is running on port $IB_PORT${NC}"
    echo -e "${YELLOW}⚠️  Make sure 'Read-Only API' is UNCHECKED in IB Gateway settings${NC}"
    echo -e "${YELLOW}    (Edit > Global Configuration > API > Settings)${NC}"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Check dependencies
echo ""
echo -e "${BLUE}[INFO]${NC} Checking dependencies..."
if ! python3 -c "import ib_insync" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  ib_insync not installed, installing now...${NC}"
    pip install -q ib_insync
fi
echo -e "${GREEN}✅ All dependencies installed${NC}"

# Warning
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}                          ⚠️  WARNING ⚠️                           ${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}This will start LIVE TRADING with REAL money if configured!${NC}"
echo -e "${YELLOW}Make sure you are using PAPER TRADING mode unless you know what you're doing.${NC}"
echo ""
echo "Config file: $CONFIG_FILE"
echo ""
echo -e "${BLUE}Trading will:${NC}"
echo "  • Connect to Interactive Brokers"
echo "  • Monitor market data in real-time"
echo "  • Execute trades automatically based on signals"
echo "  • Manage risk and positions according to config"
echo ""
read -p "Are you ABSOLUTELY SURE you want to start live trading? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Start trading
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}                   🚀 Starting Live Trading 🚀                   ${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}[INFO]${NC} Trading logs will appear below..."
echo -e "${BLUE}[INFO]${NC} Press Ctrl+C to stop trading"
echo ""

# Run trading bot
if [ "$CONFIG_FILE" == "$PROJECT_ROOT/config.yaml" ]; then
    python3 "$PROJECT_ROOT/main.py" live
else
    python3 "$PROJECT_ROOT/main.py" live --config "$CONFIG_FILE"
fi
