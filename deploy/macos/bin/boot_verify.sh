#!/bin/bash
# boot_verify.sh - runs once at login/boot. Walks the recovery chain and
# reports the outcome to Telegram:
#   BOOT COMPLETE -> gateway PID -> API 4002 -> account summary -> bot -> System Ready

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
NOTIFY="$MYTRADER_DEPLOY_DIR/bin/notify.sh"

DEADLINE=$(( $(date +%s) + API_WAIT_TIMEOUT + 120 ))
STEPS=""

wait_until() {
    # wait_until <label> <check-fn>
    local label="$1" fn="$2"
    while true; do
        if "$fn"; then
            STEPS="$STEPS✔ $label"$'\n'
            log bootverify "OK: $label"
            return 0
        fi
        if [ "$(date +%s)" -ge "$DEADLINE" ]; then
            log bootverify "TIMEOUT waiting for: $label"
            "$NOTIFY" "❌ Boot verification FAILED at step: $label
$STEPS
Check logs/launchd/ on the Mac."
            exit 1
        fi
        sleep 10
    done
}

api_summary_ok() {
    api_probe "$BOOTVERIFY_CLIENT_ID" >>"$MYTRADER_LOG_DIR/bootverify.log" 2>&1
}

bot_ok() {
    if trading_enabled; then bot_running; else return 0; fi
}

log bootverify "boot verification started"
sleep 15   # let launchd bring the other jobs up first

wait_until "IB Gateway process detected" gateway_running
wait_until "API port $IBKR_PORT accepting connections" port_open
wait_until "Account summary received" api_summary_ok
wait_until "Trading bot running" bot_ok

if trading_enabled; then
    BOT_LINE="✔ Trading bot running"
else
    BOT_LINE="⏸ Trading disabled (kill switch) — gateway only"
fi

BOOT_SEC=$(sysctl -n kern.boottime 2>/dev/null | sed -n 's/.*sec = \([0-9]*\),.*/\1/p')
if [ -n "$BOOT_SEC" ]; then
    UPTIME_MIN=$(( ( $(date +%s) - BOOT_SEC ) / 60 ))
else
    UPTIME_MIN="?"
fi

"$NOTIFY" "🟢 System Ready after boot
✔ IB Gateway PID detected
✔ API $IBKR_PORT OK
✔ Account summary received
$BOT_LINE
(boot +${UPTIME_MIN}min, mode=$TRADING_MODE)"

log bootverify "system ready"
exit 0
