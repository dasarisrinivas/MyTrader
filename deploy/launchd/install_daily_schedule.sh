#!/bin/bash
# Install/refresh the SPY Options daily start/stop launchd schedule.
#   Start: Mon–Fri 08:00 CT   Stop: Mon–Fri 14:00 CT (sends summary first)
# Re-run any time to pick up plist changes. Uninstall: pass 'uninstall'.
set -e
UID_=$(id -u)
DOMAIN="gui/$UID_"
SRC="/Users/svss/Documents/code/ShreeBot/deploy/launchd"
DEST="$HOME/Library/LaunchAgents"
JOBS=("com.shree.spy-daily-start" "com.shree.spy-daily-stop")

for j in "${JOBS[@]}"; do
  # Bootout if already loaded (ignore errors when not loaded).
  launchctl bootout "$DOMAIN/$j" 2>/dev/null || true
done

if [ "$1" = "uninstall" ]; then
  for j in "${JOBS[@]}"; do rm -f "$DEST/$j.plist"; done
  echo "✅ Uninstalled SPY daily schedule."
  exit 0
fi

mkdir -p "$DEST"
for j in "${JOBS[@]}"; do
  cp "$SRC/$j.plist" "$DEST/$j.plist"
  launchctl bootstrap "$DOMAIN" "$DEST/$j.plist"
  echo "✅ Installed + loaded $j"
done

echo ""
echo "Scheduled (Central Time, Mon–Fri):"
echo "  08:00  start_trading_day.sh  → start bot"
echo "  14:00  stop_trading_day.sh   → send summary, then stop bot"
echo ""
echo "Verify:   launchctl print $DOMAIN/com.shree.spy-daily-start | grep -A2 'run interval'"
echo "Logs:     logs/daily_scheduler.log"
