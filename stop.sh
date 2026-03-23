#!/usr/bin/env zsh
#
# Shree — Stop All Services
#
# Usage:
#   ./stop.sh                # Stop bot + agent
#   ./stop.sh --bot-only     # Stop bot only
#   ./stop.sh --agent-only   # Stop agent only
#   ./stop.sh --force        # Immediate SIGKILL

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS_DIR="$SCRIPT_DIR/logs"
AGENT_DIR="$SCRIPT_DIR/agent"

STOP_BOT=true
STOP_AGENT=true
FORCE=false

while (( $# > 0 )); do
    case "$1" in
        --bot-only)   STOP_AGENT=false ;;
        --agent-only) STOP_BOT=false ;;
        --force|-f)   FORCE=true ;;
    esac
    shift
done

echo ""
echo "🛑 Stopping Shree services..."
echo ""

_graceful_kill() {
    local name=$1 pid=$2 timeout=${3:-5}
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "   $name (PID $pid) — already stopped"
        return 1
    fi
    if [[ "$FORCE" == "true" ]]; then
        kill -9 "$pid" 2>/dev/null
        echo "✅ $name stopped (PID $pid) [forced]"
        return 0
    fi
    echo "   Stopping $name (PID $pid)..."
    kill -SIGTERM "$pid" 2>/dev/null
    local c=0
    while (( c < timeout )); do
        kill -0 "$pid" 2>/dev/null || { echo "✅ $name stopped (PID $pid)"; return 0; }
        sleep 1
        (( c++ ))
    done
    kill -9 "$pid" 2>/dev/null
    echo "✅ $name stopped (PID $pid) [forced after ${timeout}s]"
}

# ── Trading Bot ──
if [[ "$STOP_BOT" == "true" ]]; then
    bot_done=false
    if [[ -f "$LOGS_DIR/bot.pid" ]]; then
        if _graceful_kill "Trading Bot" "$(cat "$LOGS_DIR/bot.pid")" 5; then
            bot_done=true
            rm -f "$LOGS_DIR/bot.pid"
        elif ! kill -0 "$(cat "$LOGS_DIR/bot.pid")" 2>/dev/null; then
            rm -f "$LOGS_DIR/bot.pid"
        fi
    fi
    if [[ "$bot_done" == "false" ]]; then
        for p in $(pgrep -f 'python.*run_bot.py' 2>/dev/null); do
            _graceful_kill "Trading Bot" "$p" 3 && bot_done=true
        done
    fi
    [[ "$bot_done" == "false" ]] && echo "   Trading Bot — not running"

    pkill -f "dashboard_api.py" 2>/dev/null && echo "✅ Stopped dashboard_api.py"
    pkill -f "vite" 2>/dev/null && echo "✅ Stopped vite"

    pkill -f "run_autonomous_trading.py" 2>/dev/null && echo "✅ Stopped autonomous trading"
    pkill -f "run_llm_trading.py" 2>/dev/null && echo "✅ Stopped LLM trading"
    rm -f "$LOGS_DIR/backend.pid" "$LOGS_DIR/frontend.pid"
    lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null && echo "✅ Freed port 8000"
    lsof -ti:5173 2>/dev/null | xargs kill -9 2>/dev/null && echo "✅ Freed port 5173"
    lsof -ti:8001 2>/dev/null | xargs kill -9 2>/dev/null && echo "✅ Freed port 8001"
fi

# ── Analyst Agent ──
if [[ "$STOP_AGENT" == "true" ]]; then
    agent_done=false
    if [[ -f "$AGENT_DIR/agent.pid" ]]; then
        if _graceful_kill "Analyst Agent" "$(cat "$AGENT_DIR/agent.pid")" 5; then
            agent_done=true
            rm -f "$AGENT_DIR/agent.pid"
        elif ! kill -0 "$(cat "$AGENT_DIR/agent.pid")" 2>/dev/null; then
            rm -f "$AGENT_DIR/agent.pid"
        fi
    fi
    if [[ "$agent_done" == "false" ]]; then
        for p in $(pgrep -f 'agent/agent.py' 2>/dev/null); do
            _graceful_kill "Analyst Agent" "$p" 3 && agent_done=true
        done
    fi
    [[ "$agent_done" == "false" ]] && echo "   Analyst Agent — not running"
fi

rm -f "$LOGS_DIR/services.info" "$LOGS_DIR/all_services.info"
echo ""
echo "✨ All services stopped"
echo ""
