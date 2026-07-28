# New-Alpha Research Log

Every direction tested with the FROZEN replay engine (v2.0) under the FROZEN
promotion framework (v2.1). Negative results are recorded, not discarded.

---

## #2 Holding horizon — VERDICT: NEGATIVE (holding period is not the lever)
**Date:** 2026-07-27. Exploratory (sampled, n=250, G9 — cannot promote).
**Method:** signal held fixed; ONLY the exit horizon varied across
30m / 90m / EOD / +1 / +3 / +5 sessions. Horizons landing past expiration settle
at INTRINSIC using the SPY close on expiry (not a stale quote).

| Horizon | n | EV$ | Win% |
|---|---|---|---|
| 30m | 249 | −9.34 | 37% |
| **90m** | 249 | **−7.45** (least bad) | 42% |
| EOD | 249 | −27.95 | 31% |
| +1 sess | 246 | −61.00 | 24% |
| +3 sess | 238 | −62.91 | 25% |
| +5 sess | 234 | −78.16 | 25% |

**Finding:** every horizon negative; EV degrades MONOTONICALLY with holding time;
win rate 42% → 25%. Mechanism: long premium + time = theta bleed; 168–187
multi-day positions expired and settled at (mostly zero) intrinsic. Production's
existing ~90-minute exit already sits at the optimum of the tested set.

**Exploratory only (do NOT act):** ORB_BREAKOUT inverts with time
(−5.2 → +199.9 at +5sess) but is 100% puts settled at intrinsic in a falling
window = beta, already classified. BEAR_PUT_SPREAD positive at all horizons but
is mis-graded single-leg (G10 ineligible until engine v3).

**Consequence:** this strengthens the case for direction #1 (short premium). Time
decay is a *reliable, monotonic* force — and every tested family is on the losing
side of it.

---

## Backlog status
1. Short-premium (defined-risk) inversion — **BLOCKED on engine v3 multi-leg**. Highest EV.
2. Holding horizon — **DONE, negative (above).**
3. Event-conditioned behavior — not started (engine-compatible).
4. Overnight/globex context — not started (engine-compatible).
5. Cross-asset lead/lag — not started (engine-compatible).
