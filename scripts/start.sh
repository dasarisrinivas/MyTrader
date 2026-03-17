#!/usr/bin/env zsh
#
# Shree — Start All Services
#
# Usage:
#   ./start.sh                         # Bot + Agent (production)
#   ./start.sh --bot-only              # Bot only, no agent
#   ./start.sh --agent-only            # Agent only, no bot
#   ./start.sh --dry-run               # Agent in dry-run mode
#   SHREE_SIMULATION=1 ./start.sh      # Simulation mode

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

START_BOT=true
START_AGENT=true
AGENT_EXTRA_ARGS=""

while (( $# > 0 )); do
    case "$1" in
        --bot-only)   START_AGENT=false ;;
        --agent-only) START_BOT=false ;;
        --dry-run)    AGENT_EXTRA_ARGS="$AGENT_EXTRA_ARGS --dry-run" ;;
    esac
    shift
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "          🤖 Shree — Starting Services 🤖"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Python ──
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python3 &>/dev/null; then
        PYTHON_BIN="python3"
    elif command -v python &>/dev/null; then
        PYTHON_BIN="python"
    else
        echo "❌ Python interpreter not found"; exit 1
    fi
fi

# ── Venv ──
if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    echo "[INFO] Activating virtual environment..."
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

mkdir -p logs agent/logs

# ━━━━━━━━━━━━  TRADING BOT  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [[ "$START_BOT" == "true" ]]; then
    if pgrep -f "python.*run_bot.py" > /dev/null 2>&1; then
        echo "⚠️  Trading bot is already running — skipping"
    else
        CONFIG_FILE=${CONFIG_FILE:-"config.yaml"}
        if [[ ! -f "$CONFIG_FILE" ]]; then
            echo "❌ ${CONFIG_FILE} not found!"; exit 1
        fi
        export SHREE_CONFIG_FILE="$CONFIG_FILE"

        CONFIG_VALUES=$("$PYTHON_BIN" - <<'PY'
import os, yaml
cfg_path = os.environ.get("SHREE_CONFIG_FILE", "config.yaml")
try:
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
except FileNotFoundError:
    data = {}
def g(obj, path, default):
    cur = obj
    for k in path.split("."):
        if not isinstance(cur, dict): return default
        cur = cur.get(k)
        if cur is None: return default
    return cur
vals = [
    str(g(data, "data.ibkr_host", "127.0.0.1")),
    str(g(data, "data.ibkr_port", 4001)),
    str(g(data, "rag.min_similar_trades", 2)),
    str(g(data, "rag.min_weighted_win_rate", 0.45)),
    str(g(data, "trading.confidence_threshold", 0.7)),
]
print("|".join(vals))
PY
)
        IFS='|' read -r CFG_IBKR_HOST CFG_IBKR_PORT CFG_MIN_SIMILAR CFG_MIN_WR CFG_CONF <<< "$CONFIG_VALUES"
        export IBKR_HOST=${IBKR_HOST:-$CFG_IBKR_HOST}
        export IBKR_PORT=${IBKR_PORT:-$CFG_IBKR_PORT}
        export MIN_SIMILAR_TRADES=${MIN_SIMILAR_TRADES:-$CFG_MIN_SIMILAR}
        export MIN_WEIGHTED_WIN_RATE=${MIN_WEIGHTED_WIN_RATE:-$CFG_MIN_WR}
        export CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-$CFG_CONF}
        export SHREE_ORDER_TRACKER_CALLSITE=${SHREE_ORDER_TRACKER_CALLSITE:-0}

        echo "[INFO] Checking IB Gateway/TWS on port ${IBKR_PORT}..."
        if lsof -i:"$IBKR_PORT" > /dev/null 2>&1; then
            echo "✅ IB Gateway/TWS is running"
        else
            echo "❌ IB Gateway/TWS is NOT running on port ${IBKR_PORT}"
            exit 1
        fi

        export MAX_CONTRACTS=${MAX_CONTRACTS:-5}
        export PROMETHEUS_ENABLED=${PROMETHEUS_ENABLED:-true}
        export PROMETHEUS_ADDR=${PROMETHEUS_ADDR:-0.0.0.0}
        export PROMETHEUS_PORT=${PROMETHEUS_PORT:-8000}
        export DEPLOY_ENV=${DEPLOY_ENV:-prod}

        if [[ "${ENABLE_GUARDRAILS:-0}" == "1" ]]; then
            echo "[INFO] Guardrails enabled — running tests..."
            GUARD_LOG=$(mktemp)
            if "${PYTHON_BIN}" -m pytest tests/test_execution_guards.py >"$GUARD_LOG" 2>&1; then
                echo "✅ Guardrail tests passed"
            else
                if grep -q "No module named pytest" "$GUARD_LOG"; then
                    echo "⚠️ pytest not installed — skipping"
                else
                    cat "$GUARD_LOG"; rm -f "$GUARD_LOG"
                    echo "❌ Guardrail tests failed — aborting"; exit 1
                fi
            fi
            rm -f "$GUARD_LOG"

            if [[ -f "logs/bot.log" ]]; then
                GUARD_ORDER=${GUARDRAIL_REPLAY_ORDER:-14812}
                echo "[INFO] Verifying guardrails against log order ${GUARD_ORDER}..."
                if ! "$PYTHON_BIN" scripts/replay_trade_from_logs.py --log logs/bot.log --order-id "$GUARD_ORDER" >/tmp/guardrail_replay.log 2>&1; then
                    cat /tmp/guardrail_replay.log; rm -f /tmp/guardrail_replay.log
                    echo "❌ Guardrail replay failed"; exit 1
                fi
                cat /tmp/guardrail_replay.log; rm -f /tmp/guardrail_replay.log
            fi

            export FF_ENTRY_RISK_GUARDS=1
            export FF_WAIT_BLOCKING=1
            export FF_EXIT_GUARDS=1
            export FF_LEARNING_HOOKS=${FF_LEARNING_HOOKS:-1}
        else
            export FF_ENTRY_RISK_GUARDS=${FF_ENTRY_RISK_GUARDS:-0}
            export FF_WAIT_BLOCKING=${FF_WAIT_BLOCKING:-0}
            export FF_EXIT_GUARDS=${FF_EXIT_GUARDS:-0}
            export FF_LEARNING_HOOKS=${FF_LEARNING_HOOKS:-1}
        fi

        BOT_ARGS=${BOT_ARGS:-}
        if [[ "${SHREE_SIMULATION:-0}" == "1" ]]; then
            echo "[INFO] Simulation mode (SHREE_SIMULATION=1)"
            BOT_ARGS="--simulation $BOT_ARGS"
        fi

        echo "[INFO] Starting trading bot (MAX_CONTRACTS=$MAX_CONTRACTS)..."
        if [[ "${SHREE_ORDER_TRACKER_CALLSITE:-0}" != "0" ]]; then
            echo "[WARN] SHREE_ORDER_TRACKER_CALLSITE enabled"
        fi
        nohup "$PYTHON_BIN" run_bot.py ${=BOT_ARGS} > logs/bot.log 2>&1 &
        BOT_PID=$!
        echo "$BOT_PID" > logs/bot.pid

        sleep 3
        if kill -0 "$BOT_PID" 2>/dev/null; then
            echo "✅ Trading bot started (PID: $BOT_PID)"
        else
            echo "❌ Bot failed to start — check logs/bot.log"
            rm -f logs/bot.pid; exit 1
        fi
    fi
else
    echo "[SKIP] Trading bot (--agent-only)"
fi

echo ""

# ━━━━━━━━━━━━  ANALYST AGENT  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [[ "$START_AGENT" == "true" ]]; then
    if pgrep -f "agent/agent.py" > /dev/null 2>&1; then
        echo "⚠️  Analyst agent is already running — skipping"
    else
        if ! command -v gh &>/dev/null; then
            echo "⚠️  GitHub CLI (gh) not found — Copilot sessions will fail"
        fi

        echo "[INFO] Starting analyst agent..."
        nohup "$PYTHON_BIN" agent/agent.py ${=AGENT_EXTRA_ARGS} > /dev/null 2>&1 &
        AGENT_PID=$!
        echo "$AGENT_PID" > agent/agent.pid

        sleep 2
        if kill -0 "$AGENT_PID" 2>/dev/null; then
            echo "✅ Analyst agent started (PID: $AGENT_PID)"
        else
            echo "❌ Agent failed to start — check agent/logs/agent.log"
            rm -f agent/agent.pid
        fi
    fi
else
    echo "[SKIP] Analyst agent (--bot-only)"
fi

# ━━━━━━━━━━━━  SUMMARY  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "                   ✅ Shree Running ✅"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [[ "$START_BOT" == "true" ]]; then
    _BOT_PID=$(cat logs/bot.pid 2>/dev/null || echo "?")
    echo "📊 Bot PID:    $_BOT_PID"
    echo "📝 Bot Logs:   logs/bot.log"
fi
if [[ "$START_AGENT" == "true" ]]; then
    _AGENT_PID=$(cat agent/agent.pid 2>/dev/null || echo "?")
    echo "🔍 Agent PID:  $_AGENT_PID"
    echo "📝 Agent Logs: agent/logs/agent.log"
fi

echo ""
echo "Live logs:  tail -f logs/bot.log agent/logs/agent.log"
echo "Stop all:   ./stop.sh"
echo ""
