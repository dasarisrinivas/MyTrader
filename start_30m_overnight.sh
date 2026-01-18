#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#          🌙 30-MINUTE OVERNIGHT TRADING BOT                      
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Usage:
#   ./start_30m_overnight.sh          # Start in simulation mode
#   ./start_30m_overnight.sh --live   # Start in LIVE mode (careful!)
#   ./start_30m_overnight.sh status   # Check bot status
#   ./start_30m_overnight.sh stop     # Stop the bot
#   ./start_30m_overnight.sh logs     # View latest logs
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Create logs directory if needed
mkdir -p logs

# Get the PID file
PID_FILE="logs/30m_bot.pid"
LOG_FILE="logs/overnight_30m_$(date +%Y%m%d_%H%M%S).log"

case "${1:-start}" in
    status)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo -e "${GREEN}✅ 30m Bot is RUNNING (PID: $PID)${NC}"
                echo ""
                echo "Latest log entries:"
                tail -10 logs/overnight_30m_*.log 2>/dev/null | tail -10
            else
                echo -e "${YELLOW}⚠️ 30m Bot is NOT running (stale PID file)${NC}"
                rm -f "$PID_FILE"
            fi
        else
            if pgrep -f "run_30m_overnight.py" > /dev/null; then
                PID=$(pgrep -f "run_30m_overnight.py")
                echo -e "${GREEN}✅ 30m Bot is RUNNING (PID: $PID)${NC}"
            else
                echo -e "${YELLOW}⚠️ 30m Bot is NOT running${NC}"
            fi
        fi
        ;;
        
    stop)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo -e "${YELLOW}🛑 Stopping 30m Bot (PID: $PID)...${NC}"
                kill "$PID"
                sleep 2
                if ps -p "$PID" > /dev/null 2>&1; then
                    echo -e "${RED}Force killing...${NC}"
                    kill -9 "$PID"
                fi
                rm -f "$PID_FILE"
                echo -e "${GREEN}✅ Bot stopped${NC}"
            else
                echo -e "${YELLOW}⚠️ Bot was not running${NC}"
                rm -f "$PID_FILE"
            fi
        else
            # Try to find and kill by process name
            if pgrep -f "run_30m_overnight.py" > /dev/null; then
                pkill -f "run_30m_overnight.py"
                echo -e "${GREEN}✅ Bot stopped${NC}"
            else
                echo -e "${YELLOW}⚠️ No 30m bot process found${NC}"
            fi
        fi
        ;;
        
    logs)
        echo -e "${BLUE}📜 Latest 30m Bot Logs:${NC}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        tail -50 logs/overnight_30m_*.log 2>/dev/null | tail -50
        ;;
        
    start|--simulation|--live)
        # Check if already running
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo -e "${YELLOW}⚠️ 30m Bot is already running (PID: $PID)${NC}"
                echo "Use './start_30m_overnight.sh stop' to stop it first"
                exit 1
            fi
        fi
        
        # Determine mode
        if [ "$1" = "--live" ]; then
            MODE=""
            echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo -e "${RED}⚠️  LIVE TRADING MODE - REAL MONEY AT RISK!${NC}"
            echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            read -p "Type 'YES' to confirm: " CONFIRM
            if [ "$CONFIRM" != "YES" ]; then
                echo "Cancelled"
                exit 1
            fi
        else
            MODE="--simulation"
        fi
        
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}🌙 Starting 30-Minute Overnight Trading Bot${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        echo "Strategy: TREND_CONTINUATION (ADX > 30)"
        echo "Stop: 1x ATR (~6-8 pts)"  
        echo "Target: 2R"
        if [ -n "$MODE" ]; then
            echo -e "Mode: ${YELLOW}SIMULATION${NC}"
        else
            echo -e "Mode: ${RED}LIVE${NC}"
        fi
        echo ""
        echo "Log file: $LOG_FILE"
        echo ""
        
        # Start the bot
        nohup python3 run_30m_overnight.py $MODE > "$LOG_FILE" 2>&1 &
        PID=$!
        echo $PID > "$PID_FILE"
        
        sleep 2
        
        # Verify it started
        if ps -p "$PID" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Bot started successfully (PID: $PID)${NC}"
            echo ""
            echo "Commands:"
            echo "  ./start_30m_overnight.sh status  - Check status"
            echo "  ./start_30m_overnight.sh logs    - View logs"
            echo "  ./start_30m_overnight.sh stop    - Stop bot"
            echo ""
            echo "Initial log output:"
            sleep 1
            tail -15 "$LOG_FILE"
        else
            echo -e "${RED}❌ Bot failed to start${NC}"
            echo "Check logs: tail -50 $LOG_FILE"
            cat "$LOG_FILE"
            exit 1
        fi
        ;;
        
    *)
        echo "Usage: $0 {start|status|stop|logs|--simulation|--live}"
        exit 1
        ;;
esac
