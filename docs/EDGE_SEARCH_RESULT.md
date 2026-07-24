# Exploratory Edge Search — RESULT

**Verdict: NOTHING SURVIVES. Do NOT build a new strategy.** No repeatable,
out-of-sample, economically-independent edge exists in the historical dataset.

**Date:** 2026-07-24. Observation only. Production/TC/allocation/governor/
confidence/exits/risk untouched. Script: `scripts/edge_search.py`.

## 1. What was searched
426 dispatched signals with a per-signal outcome (`pnl_pct`), spanning 32
sessions Apr 1–Jul 23 2026 (mean −0.011, win 47.7% — coin-flip). Mined the
STRUCTURAL/CALENDAR/CONTEXT dimensions the failed price-indicators never
covered: day-of-week, time-of-day, month, DTE bucket, VIX regime, signal_type,
confidence tier, market regime, sentiment — and all 2- and 3-way interactions.
Price indicators / flow / GEX / max-pain / mean-reversion / ORB / TC variants
were NOT re-tested (already failed).

## 2. Methodology
Subgroup discovery (conditional expectancy over every value-combination, min
n=20) → permutation significance → then aggressive destruction: (a) multiple-
comparisons (BH-FDR + expected-false-positive count), (b) **session-clustered
bootstrap** (collapse to per-session means, resample across sessions — the true
independent unit), (c) out-of-sample split (train Apr–May / test Jun–Jul),
(d) RandomForest permutation importance on the meta-features.

## 3. Candidate structures found (apparent)
502 subgroups tested; 237 positive-expectancy; 133 with raw p<0.05 (vs 25
expected by chance); 75 "survived" BH-FDR. Top pockets looked strong: e.g.
PUT_SWEEP in mid-session (n=24, mean +0.118, win 0.88), CALL_SWEEP mornings
(n=29, mean +0.163, win 0.90).

## 4. Candidates rejected — and why
All of them. The apparent significance is an artifact of **session clustering**:
- Every top pocket spans only 2–7 sessions (median **4**), median **95% in July**.
  426 signals are really ~32 sessions, heavily concentrated in ~8 July days.
  Per-signal tests treat correlated intra-session signals as independent →
  massively overstated significance.
- **Session-clustered bootstrap dissolves them**: the best per-signal pocket's
  per-session CI = (−0.037, 0.106), includes 0 → noise.
- **No out-of-sample**: pockets live entirely in Jun–Jul (best pocket train n=1).
- Survivors of session-bootstrap = CALL_SWEEP mornings on **4 July days** — a
  4-session sample, single regime, and CALL_SWEEP is the already-accepted
  direction-confound (beta, not alpha). Rejected on sample size + regime + known
  confound.
- **RF permutation importance = 0.015 AUC** (noise level) — meta-features carry
  no win/loss information.

## 5. Candidate surviving
**None.** Every pocket fails at least one of: independent sample size (sessions,
not signals), out-of-sample survival, regime diversity, or economic
independence from the known CALL/PUT_SWEEP beta confound.

## 6. Confidence
High that no edge is demonstrable in THIS dataset. Driven by a hard data
limitation as much as by the null: 32 sessions with ~95% of the mineable
outcomes clustered in ~8 July days gives no temporal breadth — any pocket lands
in a single regime with no hold-out. The data cannot support edge discovery even
if edge existed.

## 7. Next research
The blocker is not method — it is DATA BREADTH. To make edge discovery even
possible would require many more independent sessions across regimes (months of
additional signal/outcome history), and ideally REAL fills (only 11 exist;
`pnl_pct` is a simulated barrier/stop proxy). No new feature or model fixes a
32-session, one-regime sample. Do not spend on more feature engineering.

## 8. Should a new strategy be built?
**No.** The evidence does not support it. Nothing repeatable survived. Consistent
with every prior audit. Keep the current system exactly as-is; continue
collecting live outcomes so that, in time, a genuinely broad multi-regime sample
exists to search — until then there is nothing to build on.
