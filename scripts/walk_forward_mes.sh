#!/usr/bin/env bash
#
# walk_forward_mes.sh — MES walk-forward validation of the adaptive-bucket
# loss-learning (out-of-sample test).
#
#   Step 1  TRAIN   : year-1 accrues learning (bleeding buckets hit suppress threshold)
#   Step 2a TEST cold: year-2 with NO prior learning   (baseline)
#   Step 2b TEST warm: year-2 with year-1 learning kept (suppression pre-active)
#
# Compares 2a vs 2b: if warm P&L > cold P&L, the learned suppression generalises.
#
# Usage:
#   bash scripts/walk_forward_mes.sh
#   SPLIT=2025-03-15 START=2024-03-15 END=2026-05-26 bash scripts/walk_forward_mes.sh
#
set -u

# ── Config (override via env) ────────────────────────────────────────────────
SYMBOL="${SYMBOL:-MES}"
START="${START:-2024-03-15}"          # full sample start
SPLIT="${SPLIT:-2025-03-15}"          # train/test boundary
END="${END:-2026-05-26}"              # full sample end
DATA="${DATA:-data/ib/ES_1m_multiyr.parquet}"
SESSION="${SESSION:-rth}"
PY="${PY:-python3}"                   # override with PY=python if needed

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p reports logs
LOG="logs/walk_forward_${SYMBOL}_${TS}.log"
TMP="$(mktemp -d)"
DB_TRAIN="${TMP}/wf_learn.db"         # year-1 learning, reused warm
DB_COLD="${TMP}/wf_cold.db"           # fresh year-2 learning

# ── Helpers ──────────────────────────────────────────────────────────────────
log()  { echo "$@" | tee -a "$LOG" ; }
hr()   { log "============================================================" ; }

# Run one backtest; args: <label> <start> <end> <learning_db> <keep:0|1>
run_bt() {
  local label="$1" s="$2" e="$3" db="$4" keep="$5"
  hr ; log "[$label]  ${s} → ${e}   (learning_db=$(basename "$db"), keep=${keep})" ; hr
  local out="${TMP}/${label}.out"
  local keepenv=""
  [ "$keep" = "1" ] && keepenv="BT_KEEP_DB=1"
  env BT_WITH_MANAGER=1 BT_LEARNING_DB="$db" $keepenv \
    $PY -m backtest.run --symbol "$SYMBOL" --start "$s" --end "$e" \
        --bar 1m --bar2 15m --session "$SESSION" \
        --data-source file --data-file "$DATA" > "$out" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "  ⚠️  run exited rc=$rc — last lines:" ; tail -5 "$out" | tee -a "$LOG"
  fi
  # capture the summary block + learned-rejection count
  grep -E "Total Return|Total P&L|Sharpe Ratio|Max Drawdown|Win Rate|Profit Factor|Total Trades" "$out" | tee -a "$LOG"
  local lr ; lr=$(grep -c "LEARNED REJECTION" "$out")
  log "  LEARNED REJECTION events: ${lr}"
  # expose P&L for the final comparison (handles "$-156.50" / "$474.80")
  grep -E "Total P&L" "$out" | sed -E 's/[^0-9.-]//g' | tail -1
}

# ── Run ──────────────────────────────────────────────────────────────────────
log "Walk-forward MES — ${START}..${SPLIT}..${END}  (data=${DATA})   ${TS}"
run_bt "1_TRAIN_y1"     "$START" "$SPLIT" "$DB_TRAIN" "0" >/dev/null
PNL_COLD=$(run_bt "2a_TEST_cold" "$SPLIT" "$END" "$DB_COLD"  "0")
PNL_WARM=$(run_bt "2b_TEST_warm" "$SPLIT" "$END" "$DB_TRAIN" "1")

# ── Verdict ──────────────────────────────────────────────────────────────────
hr ; log "WALK-FORWARD RESULT (year-2 out-of-sample)" ; hr
log "  cold (no prior learning) P&L : ${PNL_COLD:-n/a}"
log "  warm (year-1 learning)   P&L : ${PNL_WARM:-n/a}"
log ""
log "  Interpretation:"
log "   warm > cold  → learned suppression generalises out-of-sample (ship it)"
log "   warm ≈ cold  → midday leak real but not stably bucketable (needs structural rule)"
log "   warm < cold  → suppression overfit year-1 (raise MIN_SAMPLE / loosen threshold)"
hr
log "Full log: ${LOG}"
rm -rf "$TMP"
