#!/bin/bash
# stop_trading.sh - KILL SWITCH.
# Stops the trading bot and prevents launchd/watchdog from restarting it.
# IB Gateway stays alive. Resume with resume_trading.sh.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

log killswitch "KILL SWITCH engaged by $USER"

# 1. Remove the flag: launchd KeepAlive(PathState) and the watchdog both
#    stop restarting the bot once this file is gone.
rm -f "$TRADING_FLAG"

# 2. Gracefully stop the running bot (SIGTERM -> manager.stop()), escalate
#    after 10s.
if bot_running; then
    pkill -TERM -f "$BOT_PATTERN"
    for _ in $(seq 1 10); do
        bot_running || break
        sleep 1
    done
    if bot_running; then
        log killswitch "bot ignored SIGTERM, sending SIGKILL"
        pkill -KILL -f "$BOT_PATTERN"
    fi
    log killswitch "bot stopped"
else
    log killswitch "bot was not running"
fi

"$MYTRADER_DEPLOY_DIR/bin/notify.sh" "🛑 KILL SWITCH: trading bot stopped. IB Gateway still running. Resume with resume_trading.sh"
echo "Trading stopped. IB Gateway untouched. Resume: deploy/macos/bin/resume_trading.sh"
