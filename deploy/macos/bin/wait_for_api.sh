#!/bin/bash
# wait_for_api.sh - block until IB Gateway's API is genuinely usable.
# Stage 1: TCP port open. Stage 2: API handshake + account summary.
# Usage: wait_for_api.sh [timeout_seconds] [client_id]
# Exit 0 when healthy, 1 on timeout.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TIMEOUT="${1:-$API_WAIT_TIMEOUT}"
CLIENT_ID="${2:-$HEALTHCHECK_CLIENT_ID}"
DEADLINE=$(( $(date +%s) + TIMEOUT ))

log wait_for_api "waiting for $IBKR_HOST:$IBKR_PORT (timeout ${TIMEOUT}s)"

while ! port_open; do
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        log wait_for_api "TIMEOUT: port $IBKR_PORT never opened"
        exit 1
    fi
    sleep 5
done
log wait_for_api "port $IBKR_PORT open; probing API"

while true; do
    if api_probe "$CLIENT_ID" >>"$MYTRADER_LOG_DIR/automation.log" 2>&1; then
        log wait_for_api "API healthy (account summary received)"
        exit 0
    fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        log wait_for_api "TIMEOUT: port open but API probe kept failing"
        exit 1
    fi
    sleep 10
done
