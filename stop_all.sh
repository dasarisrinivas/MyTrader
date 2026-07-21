#!/bin/bash
# Full stop: bot + IB Gateway + watchdog, no auto-restart until start_all.sh.
exec "$(dirname "${BASH_SOURCE[0]}")/deploy/macos/bin/stop_all.sh" "$@"
