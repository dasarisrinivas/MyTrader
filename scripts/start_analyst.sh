#!/usr/bin/env zsh
# start_analyst.sh — Launch the Autonomous Trading Analyst Agent
#
# Usage:
#   ./start_analyst.sh              # Run in foreground
#   ./start_analyst.sh --background # Run in background with nohup
#   ./start_analyst.sh --dry-run    # Detect anomalies but skip Copilot
#   ./start_analyst.sh --once       # Single check and exit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Activate venv if available ──
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# ── Check prerequisites ──
if ! command -v python3 &>/dev/null; then
    echo "❌ python3 not found"
    exit 1
fi

if ! command -v gh &>/dev/null; then
    echo "⚠️  GitHub CLI (gh) not found — Copilot sessions will fail"
    echo "   Install: brew install gh && gh auth login && gh extension install github/gh-copilot"
fi

# ── Parse arguments ──
BACKGROUND=false
EXTRA_ARGS=""

for arg in "$@"; do
    case "$arg" in
        --background|-b)
            BACKGROUND=true
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $arg"
            ;;
    esac
done

# ── Launch ──
if [ "$BACKGROUND" = true ]; then
    echo "🚀 Starting Trading Analyst Agent in background..."
    # Agent's FileHandler already writes to agent/logs/agent.log,
    # so redirect nohup output to /dev/null to avoid duplicate lines.
    nohup python3 agent/agent.py $EXTRA_ARGS > /dev/null 2>&1 &
    PID=$!
    echo "$PID" > agent/agent.pid
    echo "   PID: $PID"
    echo "   Logs: agent/logs/agent.log"
    echo "   Stop: kill \$(cat agent/agent.pid)"
else
    echo "🚀 Starting Trading Analyst Agent..."
    echo "   Press Ctrl+C to stop"
    echo ""
    python3 agent/agent.py $EXTRA_ARGS
fi
