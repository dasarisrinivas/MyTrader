#!/usr/bin/env bash
#
# rr_ablation_mes.sh — controlled 3-config ablation to separate RR-floor
# miscalibration from learning-layer overfitting (Year-2 out-of-sample).
#
#   Config 1  COLD baseline   : adaptive fully OFF      (BT_NO_ADAPTIVE=1)
#   Config 2  WARM full       : year-1 learning + RR floor active (current impl)
#   Config 3  WARM no-RR-floor: year-1 learning ON, RR floor capped <= 1.33
#
# Decision rule (auto-applied):
#   CASE 1 RR MISCALIBRATION : Config3 ≈ Config1  AND  Config2 << both
#   CASE 2 LEARNING DEGRADE  : Config3 still << Config1
#   CASE 3 MIXED             : Config3 partially recovers
#
# Usage:  bash scripts/rr_ablation_mes.sh
#   override: START/SPLIT/END/DATA/SESSION/RRCAP/PY
set -u

SYMBOL="${SYMBOL:-MES}"
START="${START:-2024-03-15}"      # year-1 train start
SPLIT="${SPLIT:-2025-03-15}"      # year-1/year-2 boundary
END="${END:-2026-05-26}"          # year-2 test end
DATA="${DATA:-data/ib/ES_1m_multiyr.parquet}"
SESSION="${SESSION:-rth}"
RRCAP="${RRCAP:-1.33}"            # strategy median R:R
PY="${PY:-python3}"

TS="$(date +%Y%m%d_%H%M%S)"; mkdir -p logs reports
LOG="logs/rr_ablation_${SYMBOL}_${TS}.log"
TMP="$(mktemp -d)"
log(){ echo "$@" | tee -a "$LOG"; }
hr(){ log "============================================================"; }

# extract one metric (numeric) from a run's stdout file
m(){ grep -E "$2" "$1" | head -1 | grep -oE '\-?[0-9]+\.?[0-9]*%?' | head -1; }

# run year-2 with given extra env; args: <label> <learn_db> <extra_env...>
run(){
  local label="$1" db="$2"; shift 2
  local out="${TMP}/${label}.out"
  hr; log "[$label]  year-2 ${SPLIT} → ${END}   env: $*"; hr
  env BT_WITH_MANAGER=1 BT_LEARNING_DB="$db" "$@" \
    $PY -m backtest.run --symbol "$SYMBOL" --start "$SPLIT" --end "$END" \
      --bar 1m --bar2 15m --session "$SESSION" \
      --data-source file --data-file "$DATA" > "$out" 2>&1
  local pnl tr wr pf rej
  pnl=$(m "$out" "Total P&L"); tr=$(m "$out" "Total Trades")
  wr=$(m "$out" "Win Rate"); pf=$(m "$out" "Profit Factor")
  rej=$(grep -c "TM_REJECT" "$out")
  grep -E "Total Return|Total P&L|Sharpe|Max Draw|Win Rate|Profit Factor|Total Trades" "$out" | tee -a "$LOG"
  log "  TM_REJECT events: ${rej}   LEARNED REJECTION: $(grep -c 'LEARNED REJECTION' "$out")"
  echo "${pnl:-0}|${tr:-0}|${wr:-0}|${pf:-0}|${rej:-0}"
}

log "RR-floor ablation — ${SYMBOL}  year1=${START}..${SPLIT}  year2=${SPLIT}..${END}  RRcap=${RRCAP}  ${TS}"

# ── Pre-train year-1 into a learning DB, clone for warm configs ──────────────
hr; log "[TRAIN] year-1 ${START} → ${SPLIT} (accrue learning)"; hr
env BT_WITH_MANAGER=1 BT_LEARNING_DB="${TMP}/y1.db" \
  $PY -m backtest.run --symbol "$SYMBOL" --start "$START" --end "$SPLIT" \
    --bar 1m --bar2 15m --session "$SESSION" \
    --data-source file --data-file "$DATA" > "${TMP}/train.out" 2>&1
grep -E "Total P&L|Total Trades|Win Rate|Profit Factor" "${TMP}/train.out" | tee -a "$LOG"
cp "${TMP}/y1.db" "${TMP}/warm2.db"; cp "${TMP}/y1.db" "${TMP}/warm3.db"

# ── 3 configs ───────────────────────────────────────────────────────────────
C1=$(run "1_COLD_baseline"   "${TMP}/cold.db"  BT_NO_ADAPTIVE=1)
C2=$(run "2_WARM_full"       "${TMP}/warm2.db" BT_KEEP_DB=1)
C3=$(run "3_WARM_noRRfloor"  "${TMP}/warm3.db" BT_KEEP_DB=1 BT_ADAPTIVE_RR_CAP="$RRCAP")

IFS='|' read p1 t1 w1 f1 r1 <<< "$C1"
IFS='|' read p2 t2 w2 f2 r2 <<< "$C2"
IFS='|' read p3 t3 w3 f3 r3 <<< "$C3"

hr; log "SUMMARY TABLE (year-2)"; hr
log "$(printf '%-22s %10s %8s %8s %8s %8s' config P\&L trades WR% PF TM_rej)"
log "$(printf '%-22s %10s %8s %8s %8s %8s' 1_COLD_baseline   "$p1" "$t1" "$w1" "$f1" "$r1")"
log "$(printf '%-22s %10s %8s %8s %8s %8s' 2_WARM_full       "$p2" "$t2" "$w2" "$f2" "$r2")"
log "$(printf '%-22s %10s %8s %8s %8s %8s' 3_WARM_noRRfloor  "$p3" "$t3" "$w3" "$f3" "$r3")"

# ── Differentials + decision rule ───────────────────────────────────────────
hr; log "DIFFERENTIALS"; hr
$PY - "$p1" "$t1" "$p2" "$t2" "$p3" "$t3" <<'PYEOF' | tee -a "$LOG"
import sys
p1,t1,p2,t2,p3,t3=[float(x) for x in sys.argv[1:7]]
A_dpnl=p3-p2; A_dtr=t3-t2                  # RR-floor impact (warm full vs warm no-floor)
B_dpnl=p3-p1; B_dtr=t3-t1                  # learning impact isolated (warm no-floor vs cold)
print(f"A. RR-floor impact  : dP&L(C3-C2)={A_dpnl:+.2f}  dTrades={A_dtr:+.0f}")
if A_dtr: print(f"   EV of marginal trades recovered = ${ (p3-p2)/A_dtr:+.2f}/trade" if A_dtr>0 else f"   (C3 took {-A_dtr:.0f} fewer trades)")
print(f"B. Learning impact  : dP&L(C3-C1)={B_dpnl:+.2f}  dTrades={B_dtr:+.0f}")
print()
# tolerance: 'approx equal' if within 15% of cold P&L magnitude
tol=max(50.0, abs(p1)*0.15)
c3_near_c1 = abs(p3-p1) <= tol
c2_far     = (p1-p2) > tol and (p3-p2) > tol
if c3_near_c1 and c2_far:
    print("DECISION → CASE 1: RR MISCALIBRATION — the RR floor is the sole destructive mechanism.")
elif (p1-p3) > tol:
    print("DECISION → CASE 2: LEARNING DEGRADATION — adaptive learning hurts even without the RR floor.")
else:
    print("DECISION → CASE 3: MIXED — RR floor + learning interact; C3 recovers partially.")
PYEOF
hr; log "Full log: ${LOG}"
rm -rf "$TMP"
