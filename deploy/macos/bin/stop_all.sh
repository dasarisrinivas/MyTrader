#!/bin/bash
# stop_all.sh - FULL STOP. Stops everything: bot, IB Gateway, watchdog.
# Unloads the launchd jobs first so KeepAlive/watchdog cannot restart
# anything, then kills the processes. Plists stay installed on disk.
#
# Everything (including auto-restart) comes back with:
#   deploy/macos/bin/start_all.sh     (or a reboot, or install.sh)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

log stopall "FULL STOP requested by $USER"

# 1. Unload jobs (watchdog first so it can't fight the shutdown).
for label in "$LABEL_WATCHDOG" "$LABEL_BOOTVERIFY" "$LABEL_BOT" "$LABEL_GATEWAY"; do
    launchctl bootout "$LAUNCHD_DOMAIN/$label" 2>/dev/null \
        && log stopall "unloaded $label"
done

# 2. Graceful bot shutdown (SIGTERM -> manager.stop()), escalate after 10s.
if bot_running; then
    pkill -TERM -f "$BOT_PATTERN"
    for _ in $(seq 1 10); do bot_running || break; sleep 1; done
    bot_running && pkill -KILL -f "$BOT_PATTERN"
    log stopall "bot stopped"
fi

# 3. Stop IB Gateway (IBC-launched or manually started).
if gateway_running; then
    pkill -TERM -f "ibcalpha.ibc" 2>/dev/null
    pkill -TERM -f "ibgateway" 2>/dev/null
    for _ in $(seq 1 15); do gateway_running || break; sleep 1; done
    if gateway_running; then
        log stopall "gateway ignored SIGTERM, sending SIGKILL"
        pkill -KILL -f "ibcalpha.ibc" 2>/dev/null
        pkill -KILL -f "ibgateway" 2>/dev/null
    fi
    log stopall "IB Gateway stopped"
fi

# 4. Remove rendered IBC config (contains credentials) while stack is down.
rm -f "$MYTRADER_RUNTIME_DIR/ibc-config.ini"

"$MYTRADER_DEPLOY_DIR/bin/notify.sh" "⛔ FULL STOP: bot + IB Gateway + watchdog all stopped. Restart with start_all.sh"
echo "Full stack stopped. Nothing will auto-restart until start_all.sh (or reboot)."
