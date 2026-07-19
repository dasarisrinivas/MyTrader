#!/bin/bash
# credentials.sh - manage MyTrader secrets in the macOS login Keychain.
#
# Secrets stored (service names):
#   mytrader.ib.username        IB Gateway login
#   mytrader.ib.password        IB Gateway password
#   mytrader.telegram.bot_token Telegram bot token
#   mytrader.telegram.chat_id   Telegram chat id
#
# Usage:
#   credentials.sh setup     # interactive: prompt and store everything
#   credentials.sh check     # verify all secrets are present (no values printed)
#   credentials.sh get <service>
#   credentials.sh delete    # remove all MyTrader secrets

set -euo pipefail

SERVICES=(mytrader.ib.username mytrader.ib.password mytrader.telegram.bot_token mytrader.telegram.chat_id)

store() {
    # -U updates in place if the item already exists
    security add-generic-password -U -s "$1" -a "$USER" -w "$2"
}

case "${1:-}" in
    setup)
        echo "MyTrader credential setup — secrets go to your login Keychain only."
        printf 'IB username: '
        read -r IB_USER
        printf 'IB password: '
        read -rs IB_PASS; echo
        printf 'Telegram bot token (empty to skip): '
        read -r TG_TOKEN
        TG_CHAT=""
        if [ -n "$TG_TOKEN" ]; then
            printf 'Telegram chat id: '
            read -r TG_CHAT
        fi
        store mytrader.ib.username "$IB_USER"
        store mytrader.ib.password "$IB_PASS"
        [ -n "$TG_TOKEN" ] && store mytrader.telegram.bot_token "$TG_TOKEN"
        [ -n "$TG_CHAT" ]  && store mytrader.telegram.chat_id "$TG_CHAT"
        echo "Stored. Verify with: credentials.sh check"
        ;;
    check)
        rc=0
        for svc in "${SERVICES[@]}"; do
            if security find-generic-password -s "$svc" -a "$USER" >/dev/null 2>&1; then
                echo "OK      $svc"
            else
                echo "MISSING $svc"
                case "$svc" in mytrader.ib.*) rc=1;; esac  # telegram is optional
            fi
        done
        exit $rc
        ;;
    get)
        [ -n "${2:-}" ] || { echo "usage: credentials.sh get <service>" >&2; exit 2; }
        security find-generic-password -s "$2" -a "$USER" -w
        ;;
    delete)
        for svc in "${SERVICES[@]}"; do
            security delete-generic-password -s "$svc" -a "$USER" >/dev/null 2>&1 \
                && echo "deleted $svc" || true
        done
        ;;
    *)
        echo "usage: credentials.sh {setup|check|get <service>|delete}" >&2
        exit 2
        ;;
esac
