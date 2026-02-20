#!/bin/bash

################################################################################
# IB Gateway Configuration Checker
################################################################################

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}       🔍 IB Gateway Configuration Checker 🔍                     ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check 1: IB Gateway Running
echo -e "${YELLOW}Check 1: IB Gateway Status${NC}"
IB_PROCESS=$(ps aux | grep -i "IB Gateway" | grep -v grep || true)
if [ -n "$IB_PROCESS" ]; then
    echo -e "${GREEN}✅ IB Gateway is running${NC}"
else
    echo -e "${RED}❌ IB Gateway is NOT running${NC}"
    echo "   Please start IB Gateway and login to Live Trading account"
    exit 1
fi

# Check 2: Port Listening
echo ""
echo -e "${YELLOW}Check 2: Port 4001 Status${NC}"
PORT_LISTEN=$(lsof -i :4001 2>/dev/null | grep LISTEN || true)
if [ -n "$PORT_LISTEN" ]; then
    echo -e "${GREEN}✅ Port 4001 is listening${NC}"
else
    echo -e "${RED}❌ Port 4001 is NOT listening${NC}"
    echo "   Check IB Gateway API settings (Edit > Global Configuration > API)"
    exit 1
fi

# Check 3: Active Connections
echo ""
echo -e "${YELLOW}Check 3: Active Connections${NC}"
CONNECTIONS=$(lsof -i :4001 2>/dev/null | grep ESTABLISHED | grep -v JavaAppli || true)
if [ -z "$CONNECTIONS" ]; then
    echo -e "${GREEN}✅ No active client connections (clean state)${NC}"
else
    echo -e "${YELLOW}⚠️  Active connections detected:${NC}"
    echo "$CONNECTIONS"
    echo ""
    echo "   If experiencing Error 162, run: ./restart_clean.sh"
fi

# Check 4: Configuration File
echo ""
echo -e "${YELLOW}Check 4: IB Gateway Config File${NC}"
IB_CONFIG="$HOME/Jts/jts.ini"
if [ -f "$IB_CONFIG" ]; then
    echo -e "${GREEN}✅ Found config at: $IB_CONFIG${NC}"
    
    # Check for read-only API setting
    if grep -q "readOnlyApi=0" "$IB_CONFIG" 2>/dev/null; then
        echo -e "${GREEN}✅ Read-Only API is disabled (good)${NC}"
    else
        echo -e "${YELLOW}⚠️  Read-Only API setting not found or enabled${NC}"
        echo "   Manual check required:"
        echo "   Edit > Global Configuration > API > Settings"
        echo "   Make sure 'Read-Only API' is UNCHECKED"
    fi
else
    echo -e "${YELLOW}⚠️  Config file not found (using default location)${NC}"
fi

# Check 5: Python processes
echo ""
echo -e "${YELLOW}Check 5: Python Trading Processes${NC}"
PYTHON_PROCS=$(lsof -i :4001 2>/dev/null | grep Python || true)
if [ -z "$PYTHON_PROCS" ]; then
    echo -e "${GREEN}✅ No Python processes connected${NC}"
else
    echo -e "${YELLOW}⚠️  Python processes connected:${NC}"
    echo "$PYTHON_PROCS"
fi

# Summary and Recommendations
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Configuration Check Complete${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}If you're seeing Error 162:${NC}"
echo ""
echo -e "  1. ${YELLOW}Verify API Settings in IB Gateway:${NC}"
echo "     Edit > Global Configuration > API > Settings"
echo "     - Socket port: 4001"
echo "     - Enable: ActiveX and Socket Clients"
echo "     - UNCHECK: Read-Only API"
echo "     - Trusted IPs: Add 127.0.0.1"
echo ""
echo -e "  2. ${YELLOW}Restart IB Gateway:${NC}"
echo "     File > Exit, wait 30 seconds, restart and login"
echo ""
echo -e "  3. ${YELLOW}Clean connections:${NC}"
echo "     ./restart_clean.sh"
echo ""
echo -e "  4. ${YELLOW}Start trading:${NC}"
echo "     ./start_trading.sh"
echo ""
