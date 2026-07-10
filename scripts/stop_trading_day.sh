#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Daily auto-stop for the SPY Options bot (Mon–Fri 14:00 CT via launchd).
# Sends the day's summary to Telegram FIRST, then stops the bot.
# 14:00 CT = 15:00 ET = the executor's no-new-entries cutoff, so no entries are
# lost. Any open position's protective bracket remains resting AT IB.
# ─────────────────────────────────────────────────────────────────────────────
ROOT="/Users/svss/Documents/code/ShreeBot"
cd "$ROOT" || exit 1
LOG="$ROOT/logs/daily_scheduler.log"
mkdir -p "$ROOT/logs"
echo "════════ $(date '+%Y-%m-%d %H:%M:%S %Z') STOP_TRADING_DAY ════════" >> "$LOG"

# 1) Send the daily summary BEFORE stopping (so it reflects the full session).
python3 "$ROOT/scripts/daily_summary.py" >> "$LOG" 2>&1
echo "$(date '+%H:%M:%S') daily_summary sent" >> "$LOG"

# 2) Warn if a position is still open (its IB bracket stays live, but the bot
#    won't manage discretionary exits once stopped).
if pgrep -f "run_spy_options.py" >/dev/null 2>&1; then
  if grep -q "$(date '+%Y-%m-%d')" "$ROOT/logs/spy_options.log" 2>/dev/null \
     && [ "$(grep "$(date '+%Y-%m-%d')" "$ROOT/logs/spy_options.log" | grep -c 'EXEC FILL')" -gt \
          "$(grep "$(date '+%Y-%m-%d')" "$ROOT/logs/spy_options.log" | grep -c 'EXEC CLOSE')" ]; then
    echo "$(date '+%H:%M:%S') ⚠️ possible open position at stop — IB bracket still protects it" >> "$LOG"
  fi
fi

# 3) Stop the SPY options bot.
bash "$ROOT/stop_spy_options.sh" >> "$LOG" 2>&1
sleep 2
if pgrep -f "run_spy_options.py" >/dev/null 2>&1; then
  echo "$(date '+%H:%M:%S') ❌ SPY bot still running — force killing" >> "$LOG"
  pkill -f "run_spy_options.py"
fi
echo "$(date '+%H:%M:%S') ✅ SPY bot stopped for the day" >> "$LOG"
