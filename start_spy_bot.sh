#!/usr/bin/env bash
# SPY Options Bot — live trading startup script
# Connects to IB Gateway live port 4001 with clientId=20 (isolated from MES bot)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# IB Gateway live port (4001)
export IBKR_LIVE_PORT=4001
export LIVE_TRADING=true
export SPY_BOT_CLIENT_ID=20

echo "========================================"
echo " SPY Options Bot — LIVE TRADING"
echo " IBKR: 127.0.0.1:${IBKR_LIVE_PORT} clientId=${SPY_BOT_CLIENT_ID}"
echo " Mode: LIVE (LIVE_TRADING=${LIVE_TRADING})"
echo " Log:  logs/spy_options_bot.log"
echo "========================================"
echo ""

mkdir -p logs
python3 spy_options_bot/main.py "$@" &
SPY_PID=$!
echo $SPY_PID > logs/spy_options_bot.pid
echo "   PID $SPY_PID written to logs/spy_options_bot.pid"
wait $SPY_PID
