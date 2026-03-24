#!/bin/bash

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#          Gold Futures Strategy - Start Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Usage:
#   . start_gold.sh                    # paper trading (port 4002)
#   GOLD_SIMULATION=0 . start_gold.sh  # live trading (requires port 4001)
#   GOLD_RESET_STATE=1 . start_gold.sh # clear daily P&L / cooldowns
#   CONFIG_FILE=config.gold.yaml . start_gold.sh
#
# Runs independently of start_bot.sh — Gold uses IB client_id=3
# and writes to separate logs/state files from the MES bot.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

# Determine Python interpreter
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
NC='\033[0m'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}        Gold Futures Strategy - Starting${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Duplicate-process guard ───────────────────────────────────────────────────
mkdir -p logs

if [ -f "logs/gold.pid" ]; then
    EXISTING_PID=$(cat logs/gold.pid 2>/dev/null)
    if [ -n "$EXISTING_PID" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Gold bot is already running!${NC}"
        echo ""
        echo "   PID: $EXISTING_PID"
        echo "To restart:"
        echo "  1. Run: . stop_gold.sh"
        echo "  2. Then run: . start_gold.sh"
        return 1 2>/dev/null || exit 1
    fi
    rm -f logs/gold.pid
fi

if pgrep -f "python.*run_gold.py" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Gold bot process already detected!${NC}"
    echo "To restart: . stop_gold.sh && . start_gold.sh"
    return 1 2>/dev/null || exit 1
fi

# ── Config file ───────────────────────────────────────────────────────────────
CONFIG_FILE=${CONFIG_FILE:-"config.yaml"}
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ ${CONFIG_FILE} not found!${NC}"
    echo ""
    echo "Copy config.example.yaml to config.yaml and add a [gold] section,"
    echo "or point to a dedicated gold config:"
    echo "  CONFIG_FILE=config.gold.yaml . start_gold.sh"
    return 1 2>/dev/null || exit 1
fi

export SHREE_CONFIG_FILE="$CONFIG_FILE"

# ── Read Gold config values ───────────────────────────────────────────────────
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

gold = data.get("gold", {})
values = [
    str(get(gold, "ibkr_host",  "127.0.0.1")),
    str(get(gold, "ibkr_port",  4002)),
    str(get(gold, "symbol",     "MGC")),
    str(get(gold, "simulation", True)),
    str(get(gold, "enabled",    False)),
]
print("|".join(values))
PY
)
IFS='|' read -r CFG_IBKR_HOST CFG_IBKR_PORT CFG_SYMBOL CFG_SIMULATION CFG_ENABLED <<< "$CONFIG_VALUES"

# ── Enabled check ─────────────────────────────────────────────────────────────
if [ "$CFG_ENABLED" = "False" ]; then
    echo -e "${RED}❌ Gold strategy is disabled in ${CONFIG_FILE}${NC}"
    echo ""
    echo "Set  gold.enabled: true  in your config to allow startup."
    return 1 2>/dev/null || exit 1
fi

# ── Simulation / live mode ────────────────────────────────────────────────────
# GOLD_SIMULATION env var overrides config value (0 = live, 1 = paper).
# Default is whatever the config says (usually simulation: true).
if [ "${GOLD_SIMULATION:-}" = "0" ]; then
    SIMULATION_FLAG=""
    IBKR_PORT=${IBKR_PORT:-4001}    # Live gateway
    MODE_LABEL="LIVE"
    MODE_COLOR="$RED"
elif [ "${GOLD_SIMULATION:-}" = "1" ]; then
    SIMULATION_FLAG="--simulation"
    IBKR_PORT=${IBKR_PORT:-4002}    # Paper gateway
    MODE_LABEL="PAPER / SIMULATION"
    MODE_COLOR="$YELLOW"
else
    # Respect config value
    if [ "$CFG_SIMULATION" = "True" ] || [ "$CFG_SIMULATION" = "true" ]; then
        SIMULATION_FLAG="--simulation"
        IBKR_PORT=${IBKR_PORT:-$CFG_IBKR_PORT}
        MODE_LABEL="PAPER / SIMULATION"
        MODE_COLOR="$YELLOW"
    else
        SIMULATION_FLAG=""
        IBKR_PORT=${IBKR_PORT:-$CFG_IBKR_PORT}
        MODE_LABEL="LIVE"
        MODE_COLOR="$RED"
    fi
fi

SYMBOL=${GOLD_SYMBOL:-$CFG_SYMBOL}
IBKR_HOST=${IBKR_HOST:-$CFG_IBKR_HOST}

# ── GC safety prompt ─────────────────────────────────────────────────────────
if [ "$SYMBOL" = "GC" ] && [ "$MODE_LABEL" = "LIVE" ]; then
    echo -e "${RED}⚠️  GC (Full Gold, \$100/pt) LIVE mode requested.${NC}"
    echo ""
    echo "   This will trade full-size Gold contracts with real money."
    echo "   Make sure gold.allow_gc: true is set in ${CONFIG_FILE}."
    echo ""
    printf "   Type YES to continue: "
    read -r _CONFIRM
    if [ "$_CONFIRM" != "YES" ]; then
        echo "Aborted."
        return 1 2>/dev/null || exit 1
    fi
fi

# ── Economic calendar check ───────────────────────────────────────────────────
# Gold is highly sensitive to macro releases; warn before entry.
_TODAY_DOW=$("$PYTHON_BIN" -c "import datetime; print(datetime.date.today().strftime('%u %d %m'))")
_DAY_NUM=$(echo "$_TODAY_DOW" | awk '{print $2}')
_MONTH_NUM=$(echo "$_TODAY_DOW" | awk '{print $3}')
_DOW=$(echo "$_TODAY_DOW" | awk '{print $1}')   # 1=Mon … 7=Sun

_ECON_WARNINGS=()

# CPI: day 10–15 of month
if [ "$_DAY_NUM" -ge 10 ] 2>/dev/null && [ "$_DAY_NUM" -le 15 ] 2>/dev/null; then
    _ECON_WARNINGS+=("📰 CPI window (day ${_DAY_NUM}) — set HIGH_IMPACT_DATES if today is the release")
fi
# NFP: first Friday of month
if [ "$_DOW" -eq 5 ] 2>/dev/null && [ "$_DAY_NUM" -le 7 ] 2>/dev/null; then
    _ECON_WARNINGS+=("📰 NFP likely today (first Friday) — Gold typically spikes at 8:30 AM ET")
fi
# Core PCE: last Friday of month (day 25–31)
if [ "$_DOW" -eq 5 ] 2>/dev/null && [ "$_DAY_NUM" -ge 25 ] 2>/dev/null; then
    _ECON_WARNINGS+=("📰 Core PCE likely today (last Friday) — check lockout windows")
fi
# FOMC: Wednesday in key months
if [ "$_DOW" -eq 3 ] 2>/dev/null; then
    case "$_MONTH_NUM" in 01|03|05|06|07|09|11|12)
        _ECON_WARNINGS+=("📰 Possible FOMC Wednesday — Gold reacts sharply; 1:30–2:45 PM ET blackout recommended")
    ;; esac
fi

if [ ${#_ECON_WARNINGS[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠️  ECONOMIC RELEASE WARNING${NC}"
    for _W in "${_ECON_WARNINGS[@]}"; do
        echo -e "   ${_W}"
    done
    echo ""
    echo -e "   Add lockout windows to your config or let the news_calendar auto-fetch:"
    echo -e "   ${BLUE}gold.session.news_lockout_windows_et: [[\"08:25\",\"08:40\"]]${NC}"
    echo ""
fi
# ── End economic calendar check ───────────────────────────────────────────────

# ── IB Gateway / TWS check ───────────────────────────────────────────────────
echo -e "${BLUE}[INFO]${NC} Checking IB Gateway/TWS on ${IBKR_HOST}:${IBKR_PORT}..."
if lsof -i:"$IBKR_PORT" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ IB Gateway/TWS is running on port ${IBKR_PORT}${NC}"
else
    echo -e "${RED}❌ IB Gateway/TWS is NOT running on port ${IBKR_PORT}${NC}"
    echo ""
    if [ "$IBKR_PORT" = "4002" ]; then
        echo "Start IB Paper Gateway on port 4002, or switch to live:"
        echo "  GOLD_SIMULATION=0 . start_gold.sh   (uses port 4001)"
    else
        echo "Start IB Gateway (live) on port 4001 before starting the Gold bot."
    fi
    return 1 2>/dev/null || exit 1
fi

# ── Virtual environment ───────────────────────────────────────────────────────
VENV_PATH="$PWD/.venv"
if [ -d "$VENV_PATH" ]; then
    echo -e "${BLUE}[INFO]${NC} Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
fi

# ── Build argument list ───────────────────────────────────────────────────────
GOLD_ARGS=(--config "$CONFIG_FILE")

if [ -n "$SIMULATION_FLAG" ]; then
    GOLD_ARGS+=("$SIMULATION_FLAG")
fi

if [ "${GOLD_RESET_STATE:-0}" = "1" ]; then
    echo -e "${YELLOW}[WARN]${NC} --reset-state: daily P&L and cooldown state will be cleared"
    GOLD_ARGS+=("--reset-state")
fi

GOLD_LOG_LEVEL=${GOLD_LOG_LEVEL:-INFO}
GOLD_ARGS+=(--log-level "$GOLD_LOG_LEVEL")

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}            Launching Gold Strategy${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "  Symbol      : ${BLUE}${SYMBOL}${NC}"
echo -e "  Mode        : ${MODE_COLOR}${MODE_LABEL}${NC}"
echo -e "  IB port     : ${IBKR_PORT}"
echo -e "  Config      : ${CONFIG_FILE}"
echo -e "  Log level   : ${GOLD_LOG_LEVEL}"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
nohup "$PYTHON_BIN" run_gold.py "${GOLD_ARGS[@]}" \
    > logs/gold_bot.log 2>&1 &
GOLD_PID=$!

echo "$GOLD_PID" > logs/gold.pid

sleep 3
if kill -0 "$GOLD_PID" 2>/dev/null; then
    echo -e "${GREEN}✅ Gold bot started successfully (PID: $GOLD_PID)${NC}"
else
    echo -e "${RED}❌ Gold bot failed to start — check logs/gold_bot.log${NC}"
    rm -f logs/gold.pid
    return 1 2>/dev/null || exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}              Gold Bot Running!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${BLUE}PID          :${NC} $GOLD_PID"
echo -e "${BLUE}Trading log  :${NC} logs/gold_trading.log"
echo -e "${BLUE}Audit log    :${NC} logs/gold_bot.log"
echo ""
echo -e "${YELLOW}Live log:${NC}"
echo "  tail -f logs/gold_trading.log"
echo ""
echo -e "${YELLOW}Stop the bot:${NC}"
echo "  . stop_gold.sh"
echo ""
