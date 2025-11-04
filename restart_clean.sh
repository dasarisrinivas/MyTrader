#!/bin/bash

################################################################################
# Clean Restart Script - Kills all connections and prepares for fresh start
################################################################################

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}          🧹 MyTrader - Clean Restart Script 🧹                  ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Step 1: Kill all Python trading processes
echo -e "${YELLOW}Step 1: Killing all Python trading processes...${NC}"
PYTHON_PIDS=$(lsof -i :4002 2>/dev/null | grep Python | awk '{print $2}' | sort -u)
if [ -n "$PYTHON_PIDS" ]; then
    for PID in $PYTHON_PIDS; do
        echo "  Killing Python process $PID..."
        kill -9 "$PID" 2>/dev/null || true
    done
    echo -e "${GREEN}✅ Killed Python processes${NC}"
else
    echo -e "${GREEN}✅ No Python processes to kill${NC}"
fi

sleep 2

# Step 2: Verify connections cleared
echo ""
echo -e "${YELLOW}Step 2: Verifying connections cleared...${NC}"
REMAINING=$(lsof -i :4002 2>/dev/null | grep ESTABLISHED | grep -v JavaAppli || true)
if [ -z "$REMAINING" ]; then
    echo -e "${GREEN}✅ All client connections cleared${NC}"
else
    echo -e "${RED}⚠️  Some connections remain:${NC}"
    echo "$REMAINING"
fi

# Step 3: Clear Python cache
echo ""
echo -e "${YELLOW}Step 3: Clearing Python bytecode cache...${NC}"
cd "$(dirname "${BASH_SOURCE[0]}")"
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
echo -e "${GREEN}✅ Python cache cleared${NC}"

# Step 4: Check IB Gateway
echo ""
echo -e "${YELLOW}Step 4: Checking IB Gateway status...${NC}"
IB_RUNNING=$(lsof -i :4002 2>/dev/null | grep JavaAppli | grep LISTEN || true)
if [ -n "$IB_RUNNING" ]; then
    echo -e "${GREEN}✅ IB Gateway is running on port 4002${NC}"
else
    echo -e "${RED}❌ IB Gateway is NOT running${NC}"
    echo ""
    echo "Please start IB Gateway:"
    echo "  1. Open IB Gateway application"
    echo "  2. Login to Paper Trading account"
    echo "  3. Configure API: Edit > Global Configuration > API > Settings"
    echo "     - Enable ActiveX and Socket Clients"
    echo "     - Set Socket port to 4002"
    echo "     - UNCHECK 'Read-Only API'"
    echo "     - Add 127.0.0.1 to Trusted IPs"
    echo ""
    exit 1
fi

# Step 5: Final status
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Clean restart complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Current connections to IB Gateway:"
lsof -i :4002 2>/dev/null | head -5 || echo "  None"
echo ""
echo -e "${GREEN}Ready to start trading!${NC}"
echo "Run: ${BLUE}./start_trading.sh${NC}"
echo ""
