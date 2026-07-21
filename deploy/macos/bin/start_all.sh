#!/bin/bash
# start_all.sh - reload the full stack after stop_all.sh.
# Bootstraps the installed plists; gateway comes up, bot follows once the
# API is healthy (unless the kill switch flag is absent), watchdog resumes.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

AGENTS_DIR="$HOME/Library/LaunchAgents"
log startall "full stack start requested by $USER"

for label in "$LABEL_GATEWAY" "$LABEL_BOT" "$LABEL_WATCHDOG" "$LABEL_BOOTVERIFY"; do
    plist="$AGENTS_DIR/$label.plist"
    if [ ! -f "$plist" ]; then
        echo "MISSING $plist — run deploy/macos/install.sh" >&2
        exit 1
    fi
    if launchctl print "$LAUNCHD_DOMAIN/$label" >/dev/null 2>&1; then
        log startall "$label already loaded"
    else
        launchctl bootstrap "$LAUNCHD_DOMAIN" "$plist" && log startall "loaded $label"
    fi
done

trading_enabled || echo "NOTE: kill switch engaged — bot stays down (resume_trading.sh to enable)"
"$MYTRADER_DEPLOY_DIR/bin/notify.sh" "🔄 Full stack starting (gateway → API wait → bot → watchdog)"
echo "Stack loading. Watch: deploy/macos/bin/status.sh"
