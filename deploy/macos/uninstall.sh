#!/bin/bash
# uninstall.sh - full rollback of the MyTrader launchd automation.
# Stops all managed services and removes the plists. Leaves the repo,
# ~/.mytrader/env and Keychain secrets untouched (add --purge-credentials
# to also delete the Keychain items, --purge-state for ~/.mytrader).

set -uo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTS_DIR="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"
LABELS=(com.mytrader.bootverify com.mytrader.watchdog com.mytrader.bot com.mytrader.ibgateway)

echo "== MyTrader automation uninstall =="
for label in "${LABELS[@]}"; do
    launchctl bootout "$DOMAIN/$label" 2>/dev/null && echo "stopped  $label"
    rm -f "$AGENTS_DIR/$label.plist" "$AGENTS_DIR/$label.plist.bak" && echo "removed  $label.plist"
done

# Stop any orphaned processes the jobs left behind
pkill -TERM -f "run_bot.py" 2>/dev/null && echo "stopped bot process"
pkill -TERM -f "ibcalpha.ibc" 2>/dev/null && echo "stopped IB Gateway (IBC)"

for arg in "$@"; do
    case "$arg" in
        --purge-credentials) "$DEPLOY_DIR/bin/credentials.sh" delete ;;
        --purge-state)       rm -rf "$HOME/.mytrader" && echo "removed ~/.mytrader" ;;
    esac
done

echo "Done. Repo untouched; manual scripts (start_bot.sh etc.) still work."
