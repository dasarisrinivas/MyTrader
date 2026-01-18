#!/bin/bash
# Run all available backtests and show progress with timestamps

set -e

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

cd "$(dirname "$0")"

log "Starting 1-min (60D) backtest..."
python3 -m backtest.run --symbol ES --start 2025-10-20 --end 2026-01-09 --bar 1m --data-source file --data-file data/raw/ES/ES_1min_60D.parquet --capital 50000 --no-mtf --report html &> reports/backtest_ES_1min_60D.log &
PID1=$!

log "Starting daily (2Y) backtest..."
python3 -m backtest.run --symbol ES --start 2024-01-11 --end 2026-01-09 --bar 1d --data-source file --data-file data/raw/ES/ES_daily_2Y.parquet --capital 50000 --no-mtf --report html &> reports/backtest_ES_daily_2Y.log &
PID2=$!

log "Starting hourly (1Y) backtest..."
python3 -m backtest.run --symbol ES --start 2025-01-31 --end 2026-01-09 --bar 1h --data-source file --data-file data/raw/ES/ES_hourly_1Y.parquet --capital 50000 --no-mtf --report html &> reports/backtest_ES_hourly_1Y.log &
PID3=$!

wait $PID1
log "1-min (60D) backtest complete. Log: reports/backtest_ES_1min_60D.log"
wait $PID2
log "Daily (2Y) backtest complete. Log: reports/backtest_ES_daily_2Y.log"
wait $PID3
log "Hourly (1Y) backtest complete. Log: reports/backtest_ES_hourly_1Y.log"

log "All backtests finished!"
