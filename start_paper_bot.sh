#!/bin/bash

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#          🤖 Shree - Start Bot (PAPER TRADING) 🤖
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Usage: . start_paper_bot.sh
#
# Overrides applied vs start_bot.sh:
#   IBKR_PORT=4002        (IB Gateway paper trading port)
#   DEPLOY_ENV=paper
#   Logs written to logs/paper_trading.log + logs/paper_bot.log
#
# All other env vars (MAX_CONTRACTS, CONFIDENCE_THRESHOLD, etc.)
# can still be overridden before sourcing this script.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

# ── Paper trading overrides (must be set before start_bot.sh reads them) ──
export IBKR_PORT=${IBKR_PORT:-4002}
export DEPLOY_ENV=paper

# Redirect logs so paper and live logs never mix
export PAPER_BOT_LOG_DIR=${PAPER_BOT_LOG_DIR:-logs}
_PAPER_LOG="${PAPER_BOT_LOG_DIR}/paper_bot.log"
_PAPER_LIVE_LOG="${PAPER_BOT_LOG_DIR}/paper_trading.log"

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
echo -e "${CYAN}       📄 Shree - Starting Bot (PAPER TRADING) 📄${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${YELLOW}⚠️  PAPER TRADING MODE — No real money at risk${NC}"
echo -e "   IBKR port: ${IBKR_PORT}  (IB Gateway paper)"
echo ""

# Prefer the PID file created by this script before pattern matching.
if [ -f "${PAPER_BOT_LOG_DIR}/paper_bot.pid" ] && kill -0 "$(cat ${PAPER_BOT_LOG_DIR}/paper_bot.pid 2>/dev/null)" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Paper bot PID file exists and process is alive.${NC}"
    echo "   PID: $(cat ${PAPER_BOT_LOG_DIR}/paper_bot.pid)"
    echo "   Stop it first: ./stop_paper_bot.sh"
    return 1 2>/dev/null || exit 1
fi
rm -f "${PAPER_BOT_LOG_DIR}/paper_bot.pid"

# ── Guard: don't start if a paper bot is already running ─────────────
if pgrep -f "run_bot.py.*config.paper.yaml|SHREE_CONFIG_FILE=config.paper.yaml.*run_bot.py" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  A paper trading bot is already running!${NC}"
    echo ""
    echo "To stop it:  ./stop_paper_bot.sh"
    return 1 2>/dev/null || exit 1
fi

# ── Config file ───────────────────────────────────────────────────────
# Allow a separate paper config; default to config.paper.yaml, then fall back to shared config.yaml
PAPER_CONFIG_FILE=${PAPER_CONFIG_FILE:-${CONFIG_FILE:-"config.paper.yaml"}}
if [ ! -f "$PAPER_CONFIG_FILE" ] && [ "$PAPER_CONFIG_FILE" = "config.paper.yaml" ] && [ -f "config.yaml" ]; then
    PAPER_CONFIG_FILE="config.yaml"
fi
if [ ! -f "$PAPER_CONFIG_FILE" ]; then
    echo -e "${RED}❌ Config file not found: ${PAPER_CONFIG_FILE}${NC}"
    echo "   Set PAPER_CONFIG_FILE=<path> or ensure config.paper.yaml/config.yaml exists."
    return 1 2>/dev/null || exit 1
fi
export SHREE_CONFIG_FILE="$PAPER_CONFIG_FILE"
echo -e "${BLUE}[INFO]${NC} Using config: ${PAPER_CONFIG_FILE}"

# ── Read config values (port overridden to paper port) ────────────────
CONFIG_VALUES=$("$PYTHON_BIN" - <<'PY'
import os, yaml

cfg_path = os.environ.get("SHREE_CONFIG_FILE", "config.yaml")
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
    str(get_value(data, "rag.min_similar_trades", 2)),
    str(get_value(data, "rag.min_weighted_win_rate", 0.45)),
    str(get_value(data, "trading.confidence_threshold", 0.7)),
]
print("|".join(values))
PY
)
IFS='|' read -r CFG_IBKR_HOST CFG_MIN_SIMILAR_TRADES CFG_MIN_WEIGHTED_WIN_RATE CFG_CONFIDENCE_THRESHOLD <<< "$CONFIG_VALUES"

export IBKR_HOST=${IBKR_HOST:-$CFG_IBKR_HOST}
export MIN_SIMILAR_TRADES=${MIN_SIMILAR_TRADES:-$CFG_MIN_SIMILAR_TRADES}
export MIN_WEIGHTED_WIN_RATE=${MIN_WEIGHTED_WIN_RATE:-$CFG_MIN_WEIGHTED_WIN_RATE}
export CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-$CFG_CONFIDENCE_THRESHOLD}

# ── IB Gateway check (paper port) ────────────────────────────────────
echo -e "${BLUE}[INFO]${NC} Checking IB Gateway/TWS on paper port ${IBKR_PORT}..."
if lsof -i:"$IBKR_PORT" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ IB Gateway/TWS (paper) is running on port ${IBKR_PORT}${NC}"
else
    echo -e "${RED}❌ IB Gateway/TWS is NOT running on port ${IBKR_PORT}${NC}"
    echo ""
    echo "  Make sure IB Gateway or TWS is started in paper trading mode."
    echo "  Paper port is typically 4002 (Gateway) or 7497 (TWS)."
    echo "  Override: IBKR_PORT=7497 . start_paper_bot.sh"
    return 1 2>/dev/null || exit 1
fi

# ── Environment ───────────────────────────────────────────────────────
mkdir -p "$PAPER_BOT_LOG_DIR"

export MAX_CONTRACTS=${MAX_CONTRACTS:-5}
export PROMETHEUS_ENABLED=${PROMETHEUS_ENABLED:-true}
export PROMETHEUS_ADDR=${PROMETHEUS_ADDR:-0.0.0.0}
export PROMETHEUS_PORT=${PROMETHEUS_PORT:-8001}   # different port so paper+live can coexist
export VIX_FEED_IB_HOST=${VIX_FEED_IB_HOST:-$IBKR_HOST}
export VIX_FEED_IB_PORT=${VIX_FEED_IB_PORT:-$IBKR_PORT}
export OBSERVABILITY_ENV_LABEL=${OBSERVABILITY_ENV_LABEL:-paper}
export ENABLE_CHOP_EXCEPTION=${ENABLE_CHOP_EXCEPTION:-1}
export SHREE_ORDER_TRACKER_CALLSITE=${SHREE_ORDER_TRACKER_CALLSITE:-0}

# Feature flags: off by default in paper (mirror live defaults)
export FF_ENTRY_RISK_GUARDS=${FF_ENTRY_RISK_GUARDS:-0}
export FF_WAIT_BLOCKING=${FF_WAIT_BLOCKING:-0}
export FF_EXIT_GUARDS=${FF_EXIT_GUARDS:-0}
export FF_LEARNING_HOOKS=${FF_LEARNING_HOOKS:-1}

# ── Virtual environment ───────────────────────────────────────────────
VENV_PATH="$PWD/.venv"
if [ -d "$VENV_PATH" ]; then
    echo -e "${BLUE}[INFO]${NC} Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
fi

# ── Launch ────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${CYAN}            📄 Launching Paper Bot 📄${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${BLUE}[INFO]${NC} Starting paper trading bot (MAX_CONTRACTS=$MAX_CONTRACTS, PORT=$IBKR_PORT)..."

nohup "$PYTHON_BIN" run_bot.py > "$_PAPER_LOG" 2>&1 &
BOT_PID=$!
echo "$BOT_PID" > "${PAPER_BOT_LOG_DIR}/paper_bot.pid"

sleep 3
if kill -0 "$BOT_PID" 2>/dev/null; then
    echo -e "${GREEN}✅ Paper bot started successfully (PID: $BOT_PID)${NC}"
else
    echo -e "${RED}❌ Paper bot failed to start. Check ${_PAPER_LOG}${NC}"
    rm -f "${PAPER_BOT_LOG_DIR}/paper_bot.pid"
    return 1 2>/dev/null || exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}              ✅ Paper Bot Running! ✅${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${BLUE}📊 Bot PID:${NC}   $BOT_PID"
echo -e "${BLUE}📝 Log:${NC}       ${_PAPER_LOG}"
echo -e "${BLUE}🌐 Metrics:${NC}   http://localhost:${PROMETHEUS_PORT}/metrics"
echo -e "${BLUE}🔌 IBKR port:${NC} ${IBKR_PORT} (paper)"
echo ""
echo -e "${YELLOW}To view live logs:${NC}"
echo "  tail -f ${_PAPER_LOG}"
echo ""
echo -e "${YELLOW}To stop the paper bot:${NC}"
echo "  kill \$(cat ${PAPER_BOT_LOG_DIR}/paper_bot.pid)"
echo ""
