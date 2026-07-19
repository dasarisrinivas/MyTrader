#!/bin/bash
# start_bot_service.sh - launchd entrypoint for the trading bot.
# Waits for the IB API, then execs run_bot.py in the foreground so launchd
# tracks the real bot process. The kill switch (trading.enabled flag) gates
# startup: if trading is disabled we exit 0 and launchd leaves us alone
# (KeepAlive PathState only restarts while the flag exists).

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if ! trading_enabled; then
    log bot "trading flag absent ($TRADING_FLAG) — kill switch engaged, not starting"
    exit 0
fi

PY="$(find_python)" || {
    log bot "FATAL: no python interpreter found"
    "$MYTRADER_DEPLOY_DIR/bin/notify.sh" "❌ Bot cannot start: python not found"
    sleep 60; exit 1
}

log bot "waiting for IB API before starting bot"
if ! "$MYTRADER_DEPLOY_DIR/bin/wait_for_api.sh"; then
    log bot "IB API unavailable after ${API_WAIT_TIMEOUT}s; exiting (launchd will retry)"
    "$MYTRADER_DEPLOY_DIR/bin/notify.sh" "⚠️ Bot start delayed: IB API not ready after ${API_WAIT_TIMEOUT}s"
    exit 1
fi

cd "$MYTRADER_ROOT" || exit 1

CONFIG_FILE="${CONFIG_FILE:-config.yaml}"
if [ ! -f "$CONFIG_FILE" ]; then
    log bot "FATAL: $CONFIG_FILE not found in $MYTRADER_ROOT"
    "$MYTRADER_DEPLOY_DIR/bin/notify.sh" "❌ Bot cannot start: $CONFIG_FILE missing"
    sleep 60; exit 1
fi

export IBKR_HOST IBKR_PORT
export MAX_CONTRACTS="${MAX_CONTRACTS:-5}"
export PROMETHEUS_ENABLED="${PROMETHEUS_ENABLED:-true}"
export PROMETHEUS_PORT="${PROMETHEUS_PORT:-8000}"

ARGS=(--config "$CONFIG_FILE")
if [ "$MYTRADER_SIMULATION" = "1" ]; then
    ARGS+=(--simulation)
    log bot "SIMULATION mode enabled"
fi
# Extra args (e.g. --reset-state) via BOT_ARGS in ~/.mytrader/env
# shellcheck disable=SC2086
ARGS+=(${BOT_ARGS:-})

log bot "starting run_bot.py (python=$PY, mode=${MYTRADER_SIMULATION/1/simulation})"
"$MYTRADER_DEPLOY_DIR/bin/notify.sh" "🤖 Trading bot starting"

exec "$PY" run_bot.py "${ARGS[@]}" >> "$MYTRADER_ROOT/logs/bot.log" 2>&1
