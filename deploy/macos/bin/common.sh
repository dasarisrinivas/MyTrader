#!/bin/bash
# common.sh - shared helpers for MyTrader macOS automation.
# Sourced by every service script. Never executed directly.

set -o pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MYTRADER_DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MYTRADER_ROOT="${MYTRADER_ROOT:-$(cd "$MYTRADER_DEPLOY_DIR/../.." && pwd)}"
MYTRADER_HOME="${MYTRADER_HOME:-$HOME/.mytrader}"
MYTRADER_ENV_FILE="${MYTRADER_ENV_FILE:-$MYTRADER_HOME/env}"
MYTRADER_STATE_DIR="$MYTRADER_HOME/state"
MYTRADER_RUNTIME_DIR="$MYTRADER_HOME/runtime"
MYTRADER_LOG_DIR="$MYTRADER_ROOT/logs/launchd"
TRADING_FLAG="$MYTRADER_HOME/trading.enabled"

mkdir -p "$MYTRADER_HOME" "$MYTRADER_STATE_DIR" "$MYTRADER_LOG_DIR"
mkdir -p "$MYTRADER_RUNTIME_DIR"
chmod 700 "$MYTRADER_HOME" "$MYTRADER_RUNTIME_DIR" 2>/dev/null || true

# User-tunable settings (non-secret). Created by install.sh from env.example.
if [ -f "$MYTRADER_ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$MYTRADER_ENV_FILE"
fi

# ---------------------------------------------------------------------------
# Defaults (override in ~/.mytrader/env)
# ---------------------------------------------------------------------------
IBKR_HOST="${IBKR_HOST:-127.0.0.1}"
IBKR_PORT="${IBKR_PORT:-4002}"
TRADING_MODE="${TRADING_MODE:-paper}"          # paper | live
IBC_PATH="${IBC_PATH:-/opt/ibc}"
TWS_VERSION="${TWS_VERSION:-}"                  # e.g. 1030; auto-detected if empty
IB_GATEWAY_DIR="${IB_GATEWAY_DIR:-}"            # auto-detected if empty
API_WAIT_TIMEOUT="${API_WAIT_TIMEOUT:-600}"     # seconds to wait for port 4002 at boot
HEALTHCHECK_CLIENT_ID="${HEALTHCHECK_CLIENT_ID:-97}"
BOOTVERIFY_CLIENT_ID="${BOOTVERIFY_CLIENT_ID:-98}"
API_PROBE_EVERY_N="${API_PROBE_EVERY_N:-5}"     # watchdog does deep API probe every Nth run
BOT_PATTERN="${BOT_PATTERN:-run_bot.py}"
MYTRADER_SIMULATION="${MYTRADER_SIMULATION:-0}"
LAUNCHD_DOMAIN="gui/$(id -u)"

LABEL_GATEWAY="com.mytrader.ibgateway"
LABEL_BOT="com.mytrader.bot"
LABEL_WATCHDOG="com.mytrader.watchdog"
LABEL_BOOTVERIFY="com.mytrader.bootverify"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log() {
    # log <component> <message...>
    local component="$1"; shift
    printf '%s [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$component" "$*" \
        | tee -a "$MYTRADER_LOG_DIR/automation.log"
}

# ---------------------------------------------------------------------------
# Keychain access (secrets never touch the repo or env files)
# ---------------------------------------------------------------------------
keychain_get() {
    # keychain_get <service>  -> prints secret or returns 1
    security find-generic-password -s "$1" -a "$USER" -w 2>/dev/null
}

# ---------------------------------------------------------------------------
# Python interpreter (prefer project venv)
# ---------------------------------------------------------------------------
find_python() {
    if [ -n "$PYTHON_BIN" ] && command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "$PYTHON_BIN"; return 0
    fi
    if [ -x "$MYTRADER_ROOT/.venv/bin/python" ]; then
        echo "$MYTRADER_ROOT/.venv/bin/python"; return 0
    fi
    command -v python3 2>/dev/null && return 0
    return 1
}

# ---------------------------------------------------------------------------
# Health primitives
# ---------------------------------------------------------------------------
gateway_running() {
    # IBC launches gateway with main class ibcalpha.ibc.IbcGateway; a manually
    # started gateway shows up as install4j JavaApplicationStub with ibgateway path.
    pgrep -f "ibcalpha.ibc" >/dev/null 2>&1 && return 0
    pgrep -f "ibgateway" >/dev/null 2>&1
}

port_open() {
    nc -z -G 3 "$IBKR_HOST" "$IBKR_PORT" >/dev/null 2>&1 && return 0
    # -G (connect timeout) is BSD nc; fall back for other nc builds
    nc -z -w 3 "$IBKR_HOST" "$IBKR_PORT" >/dev/null 2>&1
}

bot_running() {
    pgrep -f "$BOT_PATTERN" >/dev/null 2>&1
}

trading_enabled() {
    [ -f "$TRADING_FLAG" ]
}

api_probe() {
    # Deep check: real IB API handshake + account summary. $1 = clientId
    local cid="${1:-$HEALTHCHECK_CLIENT_ID}"
    local py; py="$(find_python)" || return 2
    "$py" "$MYTRADER_DEPLOY_DIR/bin/check_ib_api.py" \
        --host "$IBKR_HOST" --port "$IBKR_PORT" --client-id "$cid" --timeout 20
}

kickstart() {
    # kickstart <label> : force-restart a launchd job
    launchctl kickstart -k "$LAUNCHD_DOMAIN/$1"
}
