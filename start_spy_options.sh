#!/bin/bash

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#          SPY Options Bot - Start Script (signals + execution)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Usage:
#   . start_spy_options.sh
#   CONFIG_FILE=config.yaml . start_spy_options.sh
#
# Prerequisites:
#   IB Gateway (TWS API, ib_insync) running and authenticated:
#     - Market data connection:  spy_options.ib.ibkr_port        (4001 live)
#     - Order execution (JUL 2026, when spy_options.execution.enabled):
#       spy_options.execution.ibkr_port  (4002 PAPER by default; a paper
#       gateway session must be running or execution stays offline and the
#       bot falls back to signal-only).
#
# Signals are always sent via Telegram (configure telegram section in config).
# When execution is enabled, gated signals are ALSO traded as IB bracket
# orders (limit entry + stop-loss + take-profit resting at IB).
# Runs independently of MES and Gold bots — no shared state.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

# ── Python interpreter ────────────────────────────────────────────────────────
# AUG 4 2026 (Phase 4): PREFER THE VENV EXPLICITLY.
# This block runs BEFORE the venv is activated further down, so a bare
# `python3` resolved from $PATH. Under launchd the PATH is minimal, which
# silently selected system Python 3.9 instead of the venv's 3.12 — a different
# interpreter with a different dependency set from the one the bot is tested
# against. Verified 2026-08-04: a launcher-started bot was running under
# /Library/Developer/CommandLineTools/.../Python 3.9.
SPY_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [ -z "$PYTHON_BIN" ] && [ -x "$SPY_REPO_DIR/.venv/bin/python3" ]; then
    PYTHON_BIN="$SPY_REPO_DIR/.venv/bin/python3"
fi
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

# AUG 4 2026 (Phase 4): pattern was "python.*run_spy_options.py", which is
# case-sensitive and does NOT match interpreter paths ending in capital
# "Python" (e.g. CommandLineTools .../Python.app/Contents/MacOS/Python).
# Verified 2026-08-04: a live bot (pid 29334) was invisible to this guard,
# so a second bot could be started on the same IB client ids 6/7.
# The replacement is anchored on "--config" (how the bot is always launched)
# and uses [.] so a shell whose own command line mentions the script cannot
# self-match — a plain "run_spy_options.py" pattern false-positived on the
# invoking harness during Phase 4 testing.
if pgrep -f "run_spy_options[.]py --config" > /dev/null 2>&1; then
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
ex = s.get("execution", {})
values = [
    str(get(ib, "ibkr_host",    "127.0.0.1")),
    str(get(ib, "ibkr_port",    4001)),
    str(get(s,  "enabled",      False)),
    str(get(s,  "log_file",     "logs/spy_options.log")),
    str(get(ex, "enabled",      False)),
    str(get(ex, "ibkr_port",    4002)),
]
print("|".join(values))
PY
)
IFS='|' read -r CFG_HOST CFG_PORT CFG_ENABLED CFG_LOG_FILE CFG_EXEC_ENABLED CFG_EXEC_PORT <<< "$CONFIG_VALUES"

# ── Enabled check ─────────────────────────────────────────────────────────────
if [ "$CFG_ENABLED" = "False" ]; then
    echo -e "${RED}❌ SPY Options bot is disabled in ${CONFIG_FILE}${NC}"
    echo ""
    echo "Set  spy_options.enabled: true  in your config to allow startup."
    return 1 2>/dev/null || exit 1
fi

IB_HOST=${IB_HOST:-$CFG_HOST}
IB_PORT=${IB_PORT:-$CFG_PORT}

# ── IB Gateway check ─────────────────────────────────────────────────────────
echo -e "${BLUE}[INFO]${NC} Checking IB Gateway on ${IB_HOST}:${IB_PORT}..."
if lsof -i:"$IB_PORT" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ IB Gateway is running on port ${IB_PORT}${NC}"
else
    echo -e "${RED}❌ IB Gateway is NOT running on port ${IB_PORT}${NC}"
    echo ""
    echo "Start steps:"
    echo "  1. Launch IB Gateway (ibgateway) and log in"
    echo "  2. Verify API port is set to ${IB_PORT} in IB Gateway settings"
    echo "  3. Re-run this script"
    return 1 2>/dev/null || exit 1
fi

# ── Execution gateway check (JUL 2026) ───────────────────────────────────────
if [ "$CFG_EXEC_ENABLED" = "True" ]; then
    if [ "$CFG_EXEC_PORT" = "4001" ]; then
        EXEC_MODE="⚠️  LIVE — REAL ORDERS"
    else
        EXEC_MODE="PAPER"
    fi
    echo -e "${BLUE}[INFO]${NC} Order execution ENABLED (${EXEC_MODE}) — checking port ${CFG_EXEC_PORT}..."
    if lsof -i:"$CFG_EXEC_PORT" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Execution gateway reachable on port ${CFG_EXEC_PORT}${NC}"
    else
        echo -e "${YELLOW}⚠️  No gateway on port ${CFG_EXEC_PORT} — bot will run SIGNAL-ONLY${NC}"
        echo "   (Start the paper IB Gateway session to enable order placement.)"
    fi
else
    echo -e "${BLUE}[INFO]${NC} Order execution disabled — signal-only mode"
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
echo -e "  IB data port: ${IB_PORT}"
echo -e "  Execution   : $([ "$CFG_EXEC_ENABLED" = "True" ] && echo "ENABLED (port ${CFG_EXEC_PORT})" || echo "disabled (signal-only)")"
echo -e "  Config      : ${CONFIG_FILE}"
echo -e "  Log file    : ${CFG_LOG_FILE}"
echo -e "  Log level   : ${SPY_LOG_LEVEL:-INFO}"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
nohup "$PYTHON_BIN" run_spy_options.py "${SPY_ARGS[@]}" \
    > logs/spy_options_nohup.log 2>&1 &
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
