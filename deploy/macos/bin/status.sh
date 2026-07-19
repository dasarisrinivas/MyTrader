#!/bin/bash
# status.sh - one-shot health overview of the whole stack.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ok()   { printf '  ✅ %s\n' "$1"; }
bad()  { printf '  ❌ %s\n' "$1"; }
info() { printf '  ℹ️  %s\n' "$1"; }

echo "MyTrader stack status ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "─────────────────────────────────────────────"

gateway_running && ok "IB Gateway process running" || bad "IB Gateway process NOT running"
port_open && ok "API port $IBKR_PORT open" || bad "API port $IBKR_PORT closed"

if trading_enabled; then
    bot_running && ok "Trading bot running" || bad "Trading bot NOT running (watchdog will restart)"
else
    info "Kill switch ENGAGED — bot intentionally stopped"
fi

echo
echo "launchd jobs:"
for label in "$LABEL_GATEWAY" "$LABEL_BOT" "$LABEL_WATCHDOG" "$LABEL_BOOTVERIFY"; do
    if launchctl print "$LAUNCHD_DOMAIN/$label" >/dev/null 2>&1; then
        pid=$(launchctl print "$LAUNCHD_DOMAIN/$label" 2>/dev/null | sed -n 's/^[[:space:]]*pid = \([0-9]*\)$/\1/p')
        printf '  loaded  %-28s pid=%s\n' "$label" "${pid:--}"
    else
        printf '  MISSING %-28s (run install.sh)\n' "$label"
    fi
done

echo
echo "Deep API probe (clientId $HEALTHCHECK_CLIENT_ID):"
if PY="$(find_python)"; then
    "$PY" "$MYTRADER_DEPLOY_DIR/bin/check_ib_api.py" \
        --host "$IBKR_HOST" --port "$IBKR_PORT" \
        --client-id "$HEALTHCHECK_CLIENT_ID" --summary | sed 's/^/  /'
else
    bad "python not found"
fi

echo
echo "Recent automation log:"
tail -5 "$MYTRADER_LOG_DIR/automation.log" 2>/dev/null | sed 's/^/  /'
