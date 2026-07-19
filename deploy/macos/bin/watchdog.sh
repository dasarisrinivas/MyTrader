#!/bin/bash
# watchdog.sh - health monitor, run by launchd every 60s.
# Checks: gateway process, API port, deep API probe (every Nth run),
# bot process. Restarts failed components via launchctl and sends Telegram
# alerts with de-duplication (one alert per state change, plus recovery).

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
NOTIFY="$MYTRADER_DEPLOY_DIR/bin/notify.sh"

# --- state helpers: remember previous health to alert only on transitions ---
state_get() { cat "$MYTRADER_STATE_DIR/$1" 2>/dev/null || echo "unknown"; }
state_set() { echo "$2" > "$MYTRADER_STATE_DIR/$1"; }

transition() {
    # transition <key> <new_state> <down_msg> <up_msg>
    local key="$1" new="$2" down_msg="$3" up_msg="$4"
    local old; old="$(state_get "$key")"
    if [ "$old" != "$new" ]; then
        state_set "$key" "$new"
        if [ "$new" = "down" ]; then
            log watchdog "$key: $old -> down"
            [ -n "$down_msg" ] && "$NOTIFY" "$down_msg"
        elif [ "$old" = "down" ]; then
            log watchdog "$key: recovered"
            [ -n "$up_msg" ] && "$NOTIFY" "$up_msg"
        else
            state_set "$key" "$new"   # unknown -> up at first boot: silent
        fi
    fi
}

# --- 1. IB Gateway process --------------------------------------------------
if gateway_running; then
    transition gateway up "" "✅ IB Gateway recovered"
else
    transition gateway down "🔴 IB Gateway process not running — restarting" ""
    log watchdog "kickstarting $LABEL_GATEWAY"
    kickstart "$LABEL_GATEWAY" || log watchdog "kickstart gateway failed"
    exit 0   # give it time; next run re-evaluates port/API
fi

# --- 2. API port ------------------------------------------------------------
if port_open; then
    transition port up "" "✅ IB API port $IBKR_PORT accepting connections again"
    state_set port_fail_count 0
else
    fails=$(( $(state_get port_fail_count | grep -E '^[0-9]+$' || echo 0) + 1 ))
    state_set port_fail_count "$fails"
    log watchdog "port $IBKR_PORT closed (consecutive fails: $fails)"
    transition port down "🔴 IB Gateway disconnected (port $IBKR_PORT closed)" ""
    if [ "$fails" -ge 3 ]; then
        log watchdog "port down ${fails}x — restarting gateway"
        "$NOTIFY" "♻️ Restarting IB Gateway (API port dead ${fails} checks)"
        state_set port_fail_count 0
        kickstart "$LABEL_GATEWAY" || true
    fi
    exit 0
fi

# --- 3. Deep API probe (every Nth run, own clientId) ------------------------
run_no=$(( $(state_get probe_counter | grep -E '^[0-9]+$' || echo 0) + 1 ))
state_set probe_counter "$run_no"
if [ $(( run_no % API_PROBE_EVERY_N )) -eq 0 ]; then
    if api_probe >>"$MYTRADER_LOG_DIR/watchdog_probe.log" 2>&1; then
        transition api up "" "✅ IB API connection healthy again"
        state_set api_fail_count 0
    else
        afails=$(( $(state_get api_fail_count | grep -E '^[0-9]+$' || echo 0) + 1 ))
        state_set api_fail_count "$afails"
        transition api down "🟠 IB API probe failing (port open, handshake dead)" ""
        if [ "$afails" -ge 2 ]; then
            log watchdog "API probe failed ${afails}x — restarting gateway"
            "$NOTIFY" "♻️ Restarting IB Gateway (API handshake dead)"
            state_set api_fail_count 0
            kickstart "$LABEL_GATEWAY" || true
            exit 0
        fi
    fi
fi

# --- 4. Trading bot ---------------------------------------------------------
if ! trading_enabled; then
    # Kill switch engaged: bot intentionally down, don't restart or alert.
    state_set bot disabled
    exit 0
fi
if bot_running; then
    transition bot up "" "✅ Trading bot recovered"
else
    prev="$(state_get bot)"
    transition bot down "" ""
    log watchdog "bot not running — kickstarting $LABEL_BOT"
    kickstart "$LABEL_BOT" || log watchdog "kickstart bot failed"
    if [ "$prev" = "up" ]; then
        "$NOTIFY" "♻️ Bot restarted (process was dead)"
    fi
fi
exit 0
