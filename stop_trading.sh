#!/bin/bash
# Kill switch: stop the trading bot, keep IB Gateway alive.
# Thin wrapper around the automation stack's real kill switch.
exec "$(dirname "${BASH_SOURCE[0]}")/deploy/macos/bin/stop_trading.sh" "$@"
