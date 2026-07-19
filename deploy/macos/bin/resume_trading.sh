#!/bin/bash
# resume_trading.sh - disengage the kill switch and start the bot again.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

touch "$TRADING_FLAG"
log killswitch "kill switch released by $USER — restarting bot"
kickstart "$LABEL_BOT" || {
    echo "launchctl kickstart failed — is $LABEL_BOT loaded? (install.sh)" >&2
    exit 1
}
"$MYTRADER_DEPLOY_DIR/bin/notify.sh" "▶️ Trading resumed (kill switch released)"
echo "Trading resumed."
