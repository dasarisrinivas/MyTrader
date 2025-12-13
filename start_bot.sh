#!/bin/bash

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#          🤖 MyTrader - Start Bot 🤖                      
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Usage: . start_bot.sh
#
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
echo -e "${BLUE}          🤖 MyTrader - Starting Bot 🤖${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if bot is already running
if pgrep -f "python.*run_bot.py" > /dev/null; then
    echo -e "${YELLOW}⚠️  Trading bot is already running!${NC}"
    echo ""
    echo "To restart the bot:"
    echo "  1. Run: . stop.sh"
    echo "  2. Then run: . start_bot.sh"
    return 1 2>/dev/null || exit 1
fi

# Check if IB Gateway/TWS is running
echo -e "${BLUE}[INFO]${NC} Checking IB Gateway/TWS..."
if lsof -i:4002 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ IB Gateway/TWS is running${NC}"
else
    echo -e "${RED}❌ IB Gateway/TWS is NOT running on port 4002${NC}"
    echo ""
    echo "Please start IB Gateway or TWS before running the bot."
    return 1 2>/dev/null || exit 1
fi

# Check if config file exists
if [ ! -f "config.yaml" ]; then
    echo -e "${RED}❌ config.yaml not found!${NC}"
    echo ""
    echo "Copy config.example.yaml to config.yaml and configure it:"
    echo "  cp config.example.yaml config.yaml"
    return 1 2>/dev/null || exit 1
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
fi

# Set environment variables and guardrails
export MAX_CONTRACTS=${MAX_CONTRACTS:-5}
export IBKR_HOST=${IBKR_HOST:-"127.0.0.1"}
export IBKR_PORT=${IBKR_PORT:-4002}
ENABLE_GUARDRAILS=${ENABLE_GUARDRAILS:-1}

if [ "$ENABLE_GUARDRAILS" = "1" ]; then
    echo -e "${BLUE}[INFO]${NC} Guardrails enabled – running targeted tests..."
    GUARD_LOG=$(mktemp)
    if python -m pytest tests/test_execution_guards.py >"$GUARD_LOG" 2>&1; then
        echo -e "${GREEN}✅ Guardrail tests passed${NC}"
    else
        if grep -q "No module named pytest" "$GUARD_LOG"; then
            echo -e "${YELLOW}⚠️ pytest not installed – skipping guardrail tests${NC}"
        else
            cat "$GUARD_LOG"
            rm -f "$GUARD_LOG"
            echo -e "${RED}❌ Guardrail tests failed – aborting startup${NC}"
            return 1 2>/dev/null || exit 1
        fi
    fi
    rm -f "$GUARD_LOG"

    # Replay critical trade if logs are available
    if [ -f "logs/bot.log" ]; then
        GUARD_ORDER=${GUARDRAIL_REPLAY_ORDER:-14812}
        echo -e "${BLUE}[INFO]${NC} Verifying guardrails against log order ${GUARD_ORDER}..."
        if ! python scripts/replay_trade_from_logs.py --log logs/bot.log --order-id "$GUARD_ORDER" >/tmp/guardrail_replay.log 2>&1; then
            cat /tmp/guardrail_replay.log
            rm -f /tmp/guardrail_replay.log
            echo -e "${RED}❌ Guardrail replay failed${NC}"
            return 1 2>/dev/null || exit 1
        fi
        cat /tmp/guardrail_replay.log
        rm -f /tmp/guardrail_replay.log
    fi

    export FF_ENTRY_RISK_GUARDS=1
    export FF_WAIT_BLOCKING=1
    export FF_EXIT_GUARDS=1
    export FF_LEARNING_HOOKS=${FF_LEARNING_HOOKS:-1}
else
    export FF_ENTRY_RISK_GUARDS=${FF_ENTRY_RISK_GUARDS:-0}
    export FF_WAIT_BLOCKING=${FF_WAIT_BLOCKING:-0}
    export FF_EXIT_GUARDS=${FF_EXIT_GUARDS:-0}
    export FF_LEARNING_HOOKS=${FF_LEARNING_HOOKS:-0}
fi

# Start the bot in the background
# Determine additional bot arguments
BOT_ARGS=${BOT_ARGS:-}
if [ "${MYTRADER_SIMULATION:-0}" = "1" ]; then
    echo -e "${BLUE}[INFO]${NC} Simulation mode requested via MYTRADER_SIMULATION=1"
    BOT_ARGS="--simulation $BOT_ARGS"
fi

echo -e "${BLUE}[INFO]${NC} Starting trading bot (MAX_CONTRACTS=$MAX_CONTRACTS)..."
nohup python run_bot.py $BOT_ARGS > logs/bot.log 2>&1 &
BOT_PID=$!

# Wait a bit and check if it's still running
sleep 3
if kill -0 "$BOT_PID" 2>/dev/null; then
    echo -e "${GREEN}✅ Bot started successfully (PID: $BOT_PID)${NC}"
else
    echo -e "${RED}❌ Bot failed to start. Check logs/bot.log for details${NC}"
    return 1 2>/dev/null || exit 1
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
echo "  . stop.sh"
echo ""
