#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#       🛡️  Install Trading Manager as a macOS Launch Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# After this runs once, the Trading Manager:
#   • starts automatically at user login
#   • restarts automatically if it crashes
#   • survives reboots
#
# This installs to ~/Library/LaunchAgents/ (per-user, not system-wide).
# No sudo required.
#
# Uninstall with: ./deploy/launchd/uninstall.sh
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

LABEL="com.shree.trading-manager"
SRC_PLIST="$(cd "$(dirname "$0")" && pwd)/${LABEL}.plist"
DEST_DIR="$HOME/Library/LaunchAgents"
DEST_PLIST="$DEST_DIR/${LABEL}.plist"

if [ ! -f "$SRC_PLIST" ]; then
    echo -e "${RED}❌ Source plist not found: $SRC_PLIST${NC}"
    exit 1
fi

mkdir -p "$DEST_DIR"

# If already loaded, unload first so the new plist takes effect.
if launchctl list | grep -q "$LABEL"; then
    echo -e "${YELLOW}↺  Unloading existing agent first…${NC}"
    launchctl bootout "gui/$(id -u)" "$DEST_PLIST" 2>/dev/null || \
        launchctl unload -w "$DEST_PLIST" 2>/dev/null || true
fi

# Copy the plist into LaunchAgents (this is the install).
cp "$SRC_PLIST" "$DEST_PLIST"
chmod 644 "$DEST_PLIST"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}      🛡️  Trading Manager Launch Agent — Installing${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "Label:        $LABEL"
echo "Plist:        $DEST_PLIST"
echo ""

# Bootstrap: modern macOS (11+) prefers `bootstrap`, but `load -w` still works.
if launchctl bootstrap "gui/$(id -u)" "$DEST_PLIST" 2>/dev/null; then
    echo -e "${GREEN}✓ Loaded via launchctl bootstrap${NC}"
elif launchctl load -w "$DEST_PLIST" 2>/dev/null; then
    echo -e "${GREEN}✓ Loaded via launchctl load -w (legacy)${NC}"
else
    echo -e "${RED}❌ Failed to load the launch agent. Check Console.app for details.${NC}"
    exit 1
fi

# Give launchd ~2s to spawn the process.
sleep 2

# Verify it's running.
if launchctl list | grep -q "$LABEL"; then
    PID_LINE=$(launchctl list | grep "$LABEL")
    PID=$(echo "$PID_LINE" | awk '{print $1}')
    if [ "$PID" != "-" ] && [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo -e "${GREEN}✓ Trading Manager is running (PID $PID)${NC}"
    else
        echo -e "${YELLOW}⚠  Loaded but no PID yet — check logs/trading_manager_launchd.err.log${NC}"
    fi
else
    echo -e "${RED}❌ Agent not visible in launchctl list — install failed${NC}"
    exit 1
fi

echo ""
echo "Operational notes:"
echo "  • Auto-start at login: yes"
echo "  • Auto-restart on crash: yes (30s throttle)"
echo "  • Survives reboot: yes"
echo ""
echo "Logs:"
echo "  Daemon:   tail -f logs/trading_manager.log"
echo "  Stdout:   tail -f logs/trading_manager_launchd.out.log"
echo "  Stderr:   tail -f logs/trading_manager_launchd.err.log"
echo "  Verdicts: tail -f logs/manager_decisions.jsonl"
echo ""
echo "Manage:"
echo "  Stop:     launchctl bootout gui/\$(id -u) $DEST_PLIST"
echo "  Start:    launchctl bootstrap gui/\$(id -u) $DEST_PLIST"
echo "  Status:   launchctl list | grep $LABEL"
echo "  Remove:   ./deploy/launchd/uninstall.sh"
echo ""
echo "If you change account equity (e.g. account moved to \$5,200):"
echo "  edit $DEST_PLIST → TM_ACCOUNT_EQUITY"
echo "  ./deploy/launchd/install.sh   # re-runs to reload"
echo ""
