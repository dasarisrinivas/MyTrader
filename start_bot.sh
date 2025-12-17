#!/bin/bash

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#          🤖 MyTrader - Start Bot 🤖                      
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Usage: . start_bot.sh
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

# Determine Python interpreter
if [ -z "$PYTHON_BIN" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo -e "${RED}❌ Python interpreter not found (install python3 or set PYTHON_BIN)${NC}"
        return 1 2>/dev/null || exit 1
    fi
fi

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

# Check if config file exists
CONFIG_FILE=${CONFIG_FILE:-"config.yaml"}
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ ${CONFIG_FILE} not found!${NC}"
    echo ""
    echo "Copy config.example.yaml to config.yaml and configure it:"
    echo "  cp config.example.yaml config.yaml"
    return 1 2>/dev/null || exit 1
fi

export MYTRADER_CONFIG_FILE="$CONFIG_FILE"
CONFIG_VALUES=$("$PYTHON_BIN" - <<'PY'
import os
import yaml

cfg_path = os.environ.get("MYTRADER_CONFIG_FILE", "config.yaml")
try:
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
except FileNotFoundError:
    data = {}

def get_value(obj, path, default):
    current = obj
    for key in path.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current

values = [
    str(get_value(data, "data.ibkr_host", "127.0.0.1")),
    str(get_value(data, "data.ibkr_port", 4002)),
    str(get_value(data, "rag.min_similar_trades", 2)),
    str(get_value(data, "rag.min_weighted_win_rate", 0.45)),
    str(get_value(data, "trading.confidence_threshold", 0.7)),
]
print("|".join(values))
PY
)
IFS='|' read -r CFG_IBKR_HOST CFG_IBKR_PORT CFG_MIN_SIMILAR_TRADES CFG_MIN_WEIGHTED_WIN_RATE CFG_CONFIDENCE_THRESHOLD <<< "$CONFIG_VALUES"

export IBKR_HOST=${IBKR_HOST:-$CFG_IBKR_HOST}
export IBKR_PORT=${IBKR_PORT:-$CFG_IBKR_PORT}
export MIN_SIMILAR_TRADES=${MIN_SIMILAR_TRADES:-$CFG_MIN_SIMILAR_TRADES}
export MIN_WEIGHTED_WIN_RATE=${MIN_WEIGHTED_WIN_RATE:-$CFG_MIN_WEIGHTED_WIN_RATE}
export CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-$CFG_CONFIDENCE_THRESHOLD}

# Check if IB Gateway/TWS is running
echo -e "${BLUE}[INFO]${NC} Checking IB Gateway/TWS on port ${IBKR_PORT}..."
if lsof -i:"$IBKR_PORT" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ IB Gateway/TWS is running${NC}"
else
    echo -e "${RED}❌ IB Gateway/TWS is NOT running on port ${IBKR_PORT}${NC}"
    echo ""
    echo "Please start IB Gateway or TWS before running the bot."
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

if [ "${ENABLE_GUARDRAILS:-0}" = "1" ]; then
    echo -e "${BLUE}[INFO]${NC} Guardrails enabled – running targeted tests..."
    GUARD_LOG=$(mktemp)
    if "${PYTHON_BIN}" -m pytest tests/test_execution_guards.py >"$GUARD_LOG" 2>&1; then
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
        if ! "$PYTHON_BIN" scripts/replay_trade_from_logs.py --log logs/bot.log --order-id "$GUARD_ORDER" >/tmp/guardrail_replay.log 2>&1; then
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
nohup "$PYTHON_BIN" run_bot.py $BOT_ARGS > logs/bot.log 2>&1 &
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
