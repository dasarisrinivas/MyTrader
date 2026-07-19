#!/bin/bash
# notify.sh - best-effort Telegram notification. Never fails the caller.
# Usage: notify.sh "message text"
# Token/chat id come from Keychain (mytrader.telegram.*) with env fallback
# (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in ~/.mytrader/env).

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MSG="${1:-}"
[ -n "$MSG" ] || exit 0

TOKEN="$(keychain_get mytrader.telegram.bot_token || true)"
CHAT="$(keychain_get mytrader.telegram.chat_id || true)"
TOKEN="${TOKEN:-${TELEGRAM_BOT_TOKEN:-}}"
CHAT="${CHAT:-${TELEGRAM_CHAT_ID:-}}"

if [ -z "$TOKEN" ] || [ -z "$CHAT" ]; then
    log notify "telegram not configured; message dropped: $MSG"
    exit 0
fi

HOSTTAG="$(hostname -s 2>/dev/null || echo mac)"
if curl -sS --max-time 10 \
        --data-urlencode "chat_id=$CHAT" \
        --data-urlencode "text=[$HOSTTAG] $MSG" \
        "https://api.telegram.org/bot$TOKEN/sendMessage" >/dev/null 2>&1; then
    log notify "sent: $MSG"
else
    log notify "FAILED to send: $MSG"
fi
exit 0
