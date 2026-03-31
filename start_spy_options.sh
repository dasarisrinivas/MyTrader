#!/bin/bash

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#          SPY Options Signal Bot - Start Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Usage:
#   . start_spy_options.sh
#   CONFIG_FILE=config.yaml . start_spy_options.sh
#
# Prerequisites:
#   IB Client Portal Gateway must be running and authenticated on
#   localhost (default port 5000).  TWS Gateway is NOT used here.
#
# Signals are always sent via Telegram (configure telegram section in config).
# Runs independently of MES and Gold bots — no shared state.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

# ── Python interpreter ────────────────────────────────────────────────────────
if [ -z "$PYTHON_BIN" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "❌ Python interpreter not found (install python3 or set PYTHON_BIN)"
        return 1 2>/dev/null || exit 1
    fi
fi

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}       SPY Options Signal Bot - Starting${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

mkdir -p logs

# ── Duplicate-process guard ───────────────────────────────────────────────────
if [ -f "logs/spy_options.pid" ]; then
    EXISTING_PID=$(cat logs/spy_options.pid 2>/dev/null)
    if [ -n "$EXISTING_PID" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  SPY Options bot is already running!${NC}"
        echo ""
        echo "   PID: $EXISTING_PID"
        echo "To restart:"
        echo "  1. . stop_spy_options.sh"
        echo "  2. . start_spy_options.sh"
        return 1 2>/dev/null || exit 1
    fi
    rm -f logs/spy_options.pid
fi

if pgrep -f "python.*run_spy_options.py" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  SPY Options bot process already detected!${NC}"
    echo "To restart: . stop_spy_options.sh && . start_spy_options.sh"
    return 1 2>/dev/null || exit 1
fi

# ── Config file ───────────────────────────────────────────────────────────────
CONFIG_FILE=${CONFIG_FILE:-"config.yaml"}
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ ${CONFIG_FILE} not found!${NC}"
    echo ""
    echo "Copy config.example.yaml to config.yaml and set spy_options.enabled: true"
    return 1 2>/dev/null || exit 1
fi

export SHREE_CONFIG_FILE="$CONFIG_FILE"

# ── Read spy_options config values ───────────────────────────────────────────
CONFIG_VALUES=$("$PYTHON_BIN" - <<'PY'
import os, yaml

cfg_path = os.environ.get("SHREE_CONFIG_FILE", "config.yaml")
try:
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
except FileNotFoundError:
    data = {}

def get(obj, path, default):
    cur = obj
    for k in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur

s = data.get("spy_options", {})
ib = s.get("ib", {})
values = [
    str(get(ib, "host",     "127.0.0.1")),
    str(get(ib, "port",     5000)),
    str(get(s,  "enabled",  False)),
    str(get(s,  "log_file", "logs/spy_options.log")),
]
print("|".join(values))
PY
)
IFS='|' read -r CFG_HOST CFG_PORT CFG_ENABLED CFG_LOG_FILE <<< "$CONFIG_VALUES"

# ── Enabled check ─────────────────────────────────────────────────────────────
if [ "$CFG_ENABLED" = "False" ]; then
    echo -e "${RED}❌ SPY Options bot is disabled in ${CONFIG_FILE}${NC}"
    echo ""
    echo "Set  spy_options.enabled: true  in your config to allow startup."
    return 1 2>/dev/null || exit 1
fi

IB_HOST=${IB_HOST:-$CFG_HOST}
IB_PORT=${IB_PORT:-$CFG_PORT}

# ── IB Client Portal check ────────────────────────────────────────────────────
echo -e "${BLUE}[INFO]${NC} Checking IB Client Portal Gateway on ${IB_HOST}:${IB_PORT}..."
if lsof -i:"$IB_PORT" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ IB Client Portal Gateway is running on port ${IB_PORT}${NC}"
else
    echo -e "${RED}❌ IB Client Portal Gateway is NOT running on port ${IB_PORT}${NC}"
    echo ""
    echo "Start steps:"
    echo "  1. Download IB Client Portal Gateway from IBKR website"
    echo "  2. Run: bin/run.sh root/conf.yaml"
    echo "  3. Authenticate via browser at https://localhost:${IB_PORT}"
    echo "  4. Re-run this script"
    return 1 2>/dev/null || exit 1
fi

# ── Virtual environment ───────────────────────────────────────────────────────
VENV_PATH="$PWD/.venv"
if [ -d "$VENV_PATH" ]; then
    echo -e "${BLUE}[INFO]${NC} Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
fi

# ── Build argument list ───────────────────────────────────────────────────────
SPY_ARGS=(--config "$CONFIG_FILE" --log-level "${SPY_LOG_LEVEL:-INFO}")

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}         Launching SPY Options Signal Bot${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "  IB CP port  : ${IB_PORT}"
echo -e "  Config      : ${CONFIG_FILE}"
echo -e "  Log file    : ${CFG_LOG_FILE}"
echo -e "  Log level   : ${SPY_LOG_LEVEL:-INFO}"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
nohup "$PYTHON_BIN" run_spy_options.py "${SPY_ARGS[@]}" \
    >> "$CFG_LOG_FILE" 2>&1 &
SPY_PID=$!
echo "$SPY_PID" > logs/spy_options.pid

sleep 3
if kill -0 "$SPY_PID" 2>/dev/null; then
    echo -e "${GREEN}✅ SPY Options bot started successfully (PID: $SPY_PID)${NC}"
else
    echo -e "${RED}❌ SPY Options bot failed to start — check ${CFG_LOG_FILE}${NC}"
    rm -f logs/spy_options.pid
    return 1 2>/dev/null || exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}         SPY Options Bot Running!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${BLUE}PID         :${NC} $SPY_PID"
echo -e "${BLUE}Signal log  :${NC} $CFG_LOG_FILE"
echo ""
echo -e "${YELLOW}Live log:${NC}"
echo "  tail -f $CFG_LOG_FILE"
echo ""
echo -e "${YELLOW}Stop the bot:${NC}"
echo "  . stop_spy_options.sh"
echo ""
