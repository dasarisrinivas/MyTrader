#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Daily auto-start for the SPY Options bot (Mon–Fri 08:00 CT via launchd).
# Ensures the Trading Manager is up, checks the IB Gateway, then starts the bot.
# ─────────────────────────────────────────────────────────────────────────────
ROOT="/Users/svss/Documents/code/ShreeBot"
cd "$ROOT" || exit 1
LOG="$ROOT/logs/daily_scheduler.log"
mkdir -p "$ROOT/logs"
echo "════════ $(date '+%Y-%m-%d %H:%M:%S %Z') START_TRADING_DAY ════════" >> "$LOG"

UID_=$(id -u)

# Ensure the Trading Manager daemon is running (launchd-managed).
launchctl kickstart "gui/$UID_/com.shree.trading-manager" >> "$LOG" 2>&1 \
  && echo "$(date '+%H:%M:%S') TM daemon kickstarted" >> "$LOG"

# IB Gateway must be up + authenticated on 4001 (it needs manual/auto login).
if lsof -i:4001 >/dev/null 2>&1; then
  echo "$(date '+%H:%M:%S') IB Gateway reachable on 4001" >> "$LOG"
else
  echo "$(date '+%H:%M:%S') ❌ IB Gateway NOT on 4001 — cannot start bot" >> "$LOG"
  python3 "$ROOT/scripts/daily_summary.py" --message \
    "⚠️ <b>SPY bot auto-start FAILED</b> — IB Gateway not running on port 4001. Log in to IB Gateway, then run start_spy_options.sh manually." >> "$LOG" 2>&1
  exit 1
fi

# Start the SPY options bot (guards against double-start internally).
bash "$ROOT/start_spy_options.sh" >> "$LOG" 2>&1
echo "$(date '+%H:%M:%S') start_spy_options.sh returned $?" >> "$LOG"

# Confirm + notify.
if pgrep -f "run_spy_options.py" >/dev/null 2>&1; then
  echo "$(date '+%H:%M:%S') ✅ SPY bot running (PID $(pgrep -f run_spy_options.py | head -1))" >> "$LOG"
  python3 "$ROOT/scripts/daily_summary.py" --message \
    "🟢 <b>SPY Options bot STARTED</b> for the trading day ($(date '+%a %Y-%m-%d')). Signals begin at 09:35 ET." >> "$LOG" 2>&1
else
  echo "$(date '+%H:%M:%S') ❌ SPY bot did NOT start" >> "$LOG"
  python3 "$ROOT/scripts/daily_summary.py" --message \
    "⚠️ <b>SPY bot auto-start FAILED</b> — process not running after start. Check logs/daily_scheduler.log." >> "$LOG" 2>&1
fi
