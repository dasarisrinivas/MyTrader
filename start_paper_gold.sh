#!/bin/bash

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#          Gold Futures Strategy - Start Script (PAPER TRADING)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Usage:
#   . start_paper_gold.sh
#   GOLD_RESET_STATE=1 . start_paper_gold.sh  # clear daily P&L / cooldowns
#
# Overrides applied vs start_gold.sh:
#   Config         → config.paper.yaml (gold.ibkr_port=4002, simulation=true)
#   Logs           → logs/paper_gold_trading.log + logs/paper_gold_bot.log
#   PID file       → logs/paper_gold.pid
#   simulation     → controlled by config (false = real paper fills)
#
# Runs independently of the live gold bot and the MES paper bot.
# Uses IB client_id=3 (same as live gold — do not run both simultaneously
# unless you override GOLD_CLIENT_ID to a unique value).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

# ── Determine Python interpreter ──────────────────────────────────────
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

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${CYAN}       📄 Gold Futures Strategy - Starting (PAPER) 📄${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${YELLOW}⚠️  PAPER TRADING MODE — No real money at risk${NC}"
echo ""

# ── Paths ─────────────────────────────────────────────────────────────
LOGS_DIR="${PAPER_GOLD_LOG_DIR:-logs}"
PID_FILE="${LOGS_DIR}/paper_gold.pid"
TRADING_LOG="${LOGS_DIR}/paper_gold_trading.log"
AUDIT_LOG="${LOGS_DIR}/paper_gold_bot.log"

mkdir -p "$LOGS_DIR"

# ── Duplicate-process guard ───────────────────────────────────────────
if [ -f "$PID_FILE" ]; then
    EXISTING_PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$EXISTING_PID" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Paper gold bot is already running!${NC}"
        echo "   PID: $EXISTING_PID"
        echo "   Stop it first: . stop_paper_gold.sh"
        return 1 2>/dev/null || exit 1
    fi
    rm -f "$PID_FILE"
fi

# ── Config file ───────────────────────────────────────────────────────
PAPER_GOLD_CONFIG=${PAPER_GOLD_CONFIG:-"config.paper.yaml"}
if [ ! -f "$PAPER_GOLD_CONFIG" ]; then
    echo -e "${RED}❌ Config file not found: ${PAPER_GOLD_CONFIG}${NC}"
    echo "   Create config.paper.yaml with a gold: section, or set PAPER_GOLD_CONFIG=<path>"
    return 1 2>/dev/null || exit 1
fi

export SHREE_CONFIG_FILE="$PAPER_GOLD_CONFIG"

# ── Read Gold config values for display ───────────────────────────────
CONFIG_VALUES=$("$PYTHON_BIN" - <<'PY'
import os, yaml

cfg_path = os.environ.get("SHREE_CONFIG_FILE", "config.paper.yaml")
try:
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
except FileNotFoundError:
    data = {}

gold = data.get("gold", {})
values = [
    str(gold.get("ibkr_host", "127.0.0.1")),
    str(gold.get("ibkr_port", 4002)),
    str(gold.get("symbol", "MGC")),
    str(gold.get("enabled", False)),
]
print("|".join(values))
PY
)
IFS='|' read -r CFG_HOST CFG_PORT CFG_SYMBOL CFG_ENABLED <<< "$CONFIG_VALUES"

# ── Enabled check ─────────────────────────────────────────────────────
if [ "$CFG_ENABLED" = "False" ] || [ "$CFG_ENABLED" = "false" ]; then
    echo -e "${RED}❌ Gold strategy is disabled in ${PAPER_GOLD_CONFIG}${NC}"
    echo "   Set  gold.enabled: true  in the config to allow startup."
    return 1 2>/dev/null || exit 1
fi

# ── IB Gateway check (paper port) ────────────────────────────────────
IBKR_PORT=${CFG_PORT:-4002}
echo -e "${BLUE}[INFO]${NC} Checking IB Gateway on paper port ${IBKR_PORT}..."
if lsof -i:"$IBKR_PORT" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ IB Gateway/TWS (paper) is running on port ${IBKR_PORT}${NC}"
else
    echo -e "${RED}❌ IB Gateway/TWS is NOT running on port ${IBKR_PORT}${NC}"
    echo "   Start IB Gateway in paper trading mode on port ${IBKR_PORT}."
    return 1 2>/dev/null || exit 1
fi

# ── Conflict check: live gold bot using same client_id ────────────────
if [ -f "logs/gold.pid" ]; then
    LIVE_PID=$(cat logs/gold.pid 2>/dev/null)
    if [ -n "$LIVE_PID" ] && kill -0 "$LIVE_PID" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Live gold bot is running (PID $LIVE_PID) with the same client_id=3.${NC}"
        echo "   IB will reject the paper connection."
        echo "   Either stop the live bot (. stop_gold.sh) or set GOLD_CLIENT_ID."
        return 1 2>/dev/null || exit 1
    fi
fi

# ── Virtual environment ───────────────────────────────────────────────
VENV_PATH="$PWD/.venv"
if [ -d "$VENV_PATH" ]; then
    echo -e "${BLUE}[INFO]${NC} Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
fi

# ── Build argument list ───────────────────────────────────────────
GOLD_ARGS=(--config "$PAPER_GOLD_CONFIG")

if [ "${GOLD_RESET_STATE:-0}" = "1" ]; then
    echo -e "${YELLOW}[WARN]${NC} --reset-state: daily P&L and cooldown state will be cleared"
    GOLD_ARGS+=("--reset-state")
fi

GOLD_LOG_LEVEL=${GOLD_LOG_LEVEL:-INFO}
GOLD_ARGS+=(--log-level "$GOLD_LOG_LEVEL")

# ── Override log paths via env so the gold manager writes to paper logs ──
export GOLD_TRADING_LOG="$TRADING_LOG"
export GOLD_AUDIT_LOG="$AUDIT_LOG"

# ── Banner ────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${CYAN}            Launching Paper Gold Bot${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "  Symbol      : ${BLUE}${CFG_SYMBOL}${NC}"
echo -e "  Mode        : ${YELLOW}PAPER / SIMULATION${NC}"
echo -e "  IB port     : ${IBKR_PORT}"
echo -e "  Config      : ${PAPER_GOLD_CONFIG}"
echo -e "  Log level   : ${GOLD_LOG_LEVEL}"
echo -e "  Trading log : ${TRADING_LOG}"
echo ""

# ── Launch ────────────────────────────────────────────────────────────
nohup "$PYTHON_BIN" run_gold.py "${GOLD_ARGS[@]}" \
    > "$AUDIT_LOG" 2>&1 &
GOLD_PID=$!

echo "$GOLD_PID" > "$PID_FILE"

sleep 3
if kill -0 "$GOLD_PID" 2>/dev/null; then
    echo -e "${GREEN}✅ Paper gold bot started successfully (PID: $GOLD_PID)${NC}"
else
    echo -e "${RED}❌ Paper gold bot failed to start — check ${AUDIT_LOG}${NC}"
    rm -f "$PID_FILE"
    return 1 2>/dev/null || exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}              Paper Gold Bot Running!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${BLUE}PID          :${NC} $GOLD_PID"
echo -e "${BLUE}Trading log  :${NC} ${TRADING_LOG}"
echo -e "${BLUE}Audit log    :${NC} ${AUDIT_LOG}"
echo ""
echo -e "${YELLOW}Live log:${NC}"
echo "  tail -f ${TRADING_LOG}"
echo ""
echo -e "${YELLOW}Stop the bot:${NC}"
echo "  . stop_paper_gold.sh"
echo ""
