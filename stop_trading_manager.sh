#!/bin/bash
# Stop the Trading Manager daemon (graceful SIGTERM).
set -e
cd "$(dirname "$0")"

if [ ! -f "logs/trading_manager.pid" ]; then
    echo "No PID file. Trading Manager not running (or PID file lost)."
    exit 0
fi

PID=$(cat logs/trading_manager.pid)
if ! kill -0 "$PID" 2>/dev/null; then
    echo "PID $PID not alive. Cleaning up stale PID file."
    rm -f logs/trading_manager.pid
    exit 0
fi

echo "Stopping Trading Manager (PID $PID)..."
kill -TERM "$PID"
for i in {1..10}; do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "Stopped."
        rm -f logs/trading_manager.pid
        exit 0
    fi
    sleep 1
done

echo "Did not stop after 10s — sending SIGKILL."
kill -KILL "$PID" 2>/dev/null || true
rm -f logs/trading_manager.pid
