#!/bin/bash
# Auto-ingest today's trade journal data into SQLite DB.
# Runs daily at 3:00 PM CT via launchd (com.shreebot.journal-ingest.plist).
#
# Manual run: bash scripts/auto_journal_ingest.sh
# Logs to: logs/journal_ingest.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/journal_ingest.log"
VENV="$PROJECT_ROOT/.venv/bin/python3"

# Use venv python if available, else system python3
if [ -f "$VENV" ]; then
    PYTHON="$VENV"
else
    PYTHON="python3"
fi

TODAY=$(date +%Y-%m-%d)

echo "" >> "$LOG_FILE"
echo "=== Journal ingest: $TODAY at $(date '+%H:%M:%S %Z') ===" >> "$LOG_FILE"

cd "$PROJECT_ROOT"
$PYTHON scripts/daily_journal.py "$TODAY" >> "$LOG_FILE" 2>&1

echo "✅ Journal ingest complete for $TODAY" >> "$LOG_FILE"
