#!/bin/bash
# start_ibgateway.sh - launchd entrypoint for headless IB Gateway via IBC.
#
# Renders the IBC config from the template with credentials from the Keychain,
# then runs IBC's ibcstart.sh in the FOREGROUND so launchd owns the process
# tree and can restart it on crash (KeepAlive=true in the plist).

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

log ibgateway "starting (mode=$TRADING_MODE port=$IBKR_PORT)"

# --- Preconditions ---------------------------------------------------------
if [ ! -d "$IBC_PATH" ] || [ ! -f "$IBC_PATH/scripts/ibcstart.sh" ]; then
    log ibgateway "FATAL: IBC not found at $IBC_PATH (install from https://github.com/IbcAlpha/IBC)"
    "$MYTRADER_DEPLOY_DIR/bin/notify.sh" "❌ IB Gateway cannot start: IBC missing at $IBC_PATH"
    sleep 60   # keep launchd from thrashing
    exit 1
fi

IB_USER="$(keychain_get mytrader.ib.username || true)"
IB_PASS="$(keychain_get mytrader.ib.password || true)"
if [ -z "$IB_USER" ] || [ -z "$IB_PASS" ]; then
    log ibgateway "FATAL: IB credentials missing from Keychain (run credentials.sh setup)"
    "$MYTRADER_DEPLOY_DIR/bin/notify.sh" "❌ IB Gateway cannot start: credentials missing from Keychain"
    sleep 60
    exit 1
fi

# --- Locate IB Gateway install / version -----------------------------------
if [ -z "$IB_GATEWAY_DIR" ]; then
    for d in "$HOME/Applications" "/Applications"; do
        cand=$(ls -d "$d"/ibgateway*/ 2>/dev/null | sort -V | tail -1)
        if [ -n "$cand" ]; then IB_GATEWAY_DIR="${cand%/}"; break; fi
    done
fi
if [ -z "$TWS_VERSION" ]; then
    # Directory is like ~/Applications/ibgateway/1030 or /Applications/IB Gateway 10.30
    TWS_VERSION=$(ls "$HOME/Applications/ibgateway" 2>/dev/null | sort -V | tail -1)
fi
if [ -z "$TWS_VERSION" ]; then
    log ibgateway "FATAL: cannot determine IB Gateway version; set TWS_VERSION in ~/.mytrader/env"
    "$MYTRADER_DEPLOY_DIR/bin/notify.sh" "❌ IB Gateway cannot start: TWS_VERSION unknown"
    sleep 60
    exit 1
fi

# --- Render IBC config with secrets (0600, private runtime dir) ------------
IBC_INI="$MYTRADER_RUNTIME_DIR/ibc-config.ini"
umask 077
sed -e "s|__IB_USER__|$IB_USER|" \
    -e "s|__IB_PASSWORD__|$IB_PASS|" \
    -e "s|__TRADING_MODE__|$TRADING_MODE|" \
    -e "s|__API_PORT__|$IBKR_PORT|" \
    "$MYTRADER_DEPLOY_DIR/ibc/config.ini.template" > "$IBC_INI"
chmod 600 "$IBC_INI"
unset IB_PASS

"$MYTRADER_DEPLOY_DIR/bin/notify.sh" "🚀 IB Gateway starting (mode=$TRADING_MODE, v$TWS_VERSION)"

# --- Run IBC in the foreground ---------------------------------------------
# ibcstart.sh runs the gateway JVM directly (unlike gatewaystartmacos.sh which
# detaches via Terminal). launchd therefore tracks the real process.
exec "$IBC_PATH/scripts/ibcstart.sh" "$TWS_VERSION" --gateway \
    "--mode=$TRADING_MODE" \
    "--ibc-path=$IBC_PATH" \
    "--ibc-ini=$IBC_INI" \
    ${IB_GATEWAY_DIR:+"--tws-path=$(dirname "$IB_GATEWAY_DIR")"}
