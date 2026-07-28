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

## #1 Short premium (defined-risk credit spreads) — VERDICT: NO EVIDENCE
**Date:** 2026-07-27. Instrument: v3.0 (V1-V7 passed). Research only.
**Method:** 966 SYNTHETIC historical credit spreads (bull put / bear call) across
92 configs — structure x moneyness (0.3%/0.7%) x width (2/5) x DTE (0/1) x entry
(10:00/12:00/14:00) x exit (90m/EOD), 20 sessions. Landscape mapped BEFORE any
tuning. Conservative fills, $2.60 RT, no assignment modeling.

| Filter | Count |
|---|---|
| configs mapped | 92 |
| EV > 0 | 35 (38%) -> **62% NEGATIVE** |
| + 95% CI excludes 0 | 6 |
| + \|beta\|<0.5 and alpha>0 | 6 |

**6 survivors vs ~4.6 expected by chance at alpha=0.05 over 92 tests = the null.**
All survivors n=6-14 (G1 needs >=50); coverage 20 sessions (needs >=30);
replay completeness 91.5% (G8 needs >=99%). Constitution applied unchanged, no
exceptions -> nothing promotable, nothing shadow-worthy.

**Mechanism fails independently:** best bear-call configs carry beta -0.58..-0.82
(they profited because SPY drifted down, not from decay); theta capture is
erratic (+0.71 .. -0.85), not systematically positive as a decay-harvesting
hypothesis requires.

**VERDICT: NO — hypothesis CLOSED.** Defined-risk short premium shows no evidence
of regime-independent positive expectancy under observed conditions.

**Bounded claim:** per-config n is small (6-14); a much larger session count could
sharpen the estimate. But the landscape is majority-negative and survivors match
chance, so there is no positive signal to pursue.

**LIMITATION carried:** American-style SPY options — early assignment of short
legs is NOT modeled. Material for short legs; would only make results WORSE.

---

## Backlog status
1. Short-premium (defined-risk) inversion — **DONE, no evidence (above).**
2. Holding horizon — **DONE, negative (above).**
3. Event-conditioned behavior — not started (engine-compatible).
4. Overnight/globex context — not started (engine-compatible).
5. Cross-asset lead/lag — not started (engine-compatible).
