#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#       🛑  Uninstall the Trading Manager Launch Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

LABEL="com.shree.trading-manager"
DEST_PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"

if [ ! -f "$DEST_PLIST" ]; then
    echo -e "${YELLOW}⚠  $DEST_PLIST not found — nothing to uninstall.${NC}"
    exit 0
fi

echo "Stopping and unloading $LABEL…"

# Try modern syntax first, fall back to legacy.
if launchctl bootout "gui/$(id -u)" "$DEST_PLIST" 2>/dev/null; then
    echo -e "${GREEN}✓ Unloaded via launchctl bootout${NC}"
elif launchctl unload -w "$DEST_PLIST" 2>/dev/null; then
    echo -e "${GREEN}✓ Unloaded via launchctl unload -w (legacy)${NC}"
else
    echo -e "${YELLOW}⚠  Unload command failed; agent may already be stopped${NC}"
fi

rm -f "$DEST_PLIST"
echo -e "${GREEN}✓ Removed $DEST_PLIST${NC}"

if launchctl list | grep -q "$LABEL"; then
    echo -e "${RED}⚠  Agent still showing in launchctl list — try logging out and back in${NC}"
    exit 1
fi

echo ""
echo "The plist is also still in deploy/launchd/${LABEL}.plist if you want to reinstall."
