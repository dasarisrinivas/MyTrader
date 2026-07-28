# Shadow → Production Promotion Framework

**Purpose:** convert replay evidence into an operational promotion decision, so
promotion is never a reaction to one good day. Binding for every shadow family.

**Canonical evaluator:** frozen Research Engine v2 (real option dollar P&L).
SPY-barrier metrics are retired and may not be cited in a promotion argument.

## The lesson that created this framework (2026-07-27)
Replay of that session showed **+$2,363** if every dispatched signal had traded
(+$13.82/trade). It was **not edge**:

| Right | n | EV | Total |
|---|---|---|---|
| Puts | 105 | +$57.16 | +$6,001 |
| Calls | 66 | −$55.13 | −$3,639 |

SPY fell −0.89%. The net was positive only because the signal mix was put-heavy
on a down day — and **101 of 171 signals still lost**. Any framework that would
have promoted on that number is broken. Hence gate G5 below.

## Promotion gates — ALL must pass (SHADOW → PILOT, 1 contract)

| Gate | Requirement | Why |
|---|---|---|
| **G1 SAMPLE** | ≥ 50 replayed signals | single-day/small-n results are noise |
| **G2 SESSIONS** | ≥ 10 distinct sessions | one session ≠ evidence |
| **G3 CONSISTENCY** | ≥ 3 **consecutive** sessions with positive replay EV | filters one-off spikes |
| **G4 REGIME** | positive EV on **both** up-SPY and down-SPY days | must work when the market disagrees |
| **G5 INDEPENDENCE** | EV not explained by SPY direction (beta-adjusted) | the −0.89%-day trap above |
| **G6 REAL P&L** | graded on real option $ via frozen engine, after spread + commission | barrier metrics are retired |

Fail any gate → `WATCH` (if EV>0) or `NO`. Never promote on a single session.

## Promotion ladder
```
SHADOW  ──all gates pass──▶  PILOT (1 contract)  ──sustained──▶  PRODUCTION
   ▲                              │
   └──────── any gate fails ──────┘
```
Demotion is automatic and immediate on gate failure; promotion is never automatic.

## Mandatory audit section
Every session audit must end with:

```
PROMOTION REVIEW
Any shadow strategy eligible for live tomorrow?  YES / NO
If YES: strategy, reason, expected contracts/day, risk, confidence, evidence
If NO:  what evidence is still missing
```

## Known blockers to running this framework honestly
1. **Blocked-signal replayability** — until 2026-07-27 `blocked_signals.jsonl`
   stored only a month code (`AUG26`), so 0/508 blocked signals were replayable
   and **gate effectiveness was unmeasurable**. Fixed by recording `expiry_date`
   (exact YYYYMMDD). Blocked-signal evidence only accrues from that date forward.
2. **Multi-leg** — spreads/straddles are mis-graded single-leg by engine v2; no
   spread family may be promoted until v3 multi-leg replay exists.
3. **Confidence calibration** needs 20–30 sessions across bull/bear/flat before
   confidence can be used as a production lever. On 2026-07-27 HIGH confidence
   showed +$79.61 EV / 68% win — but it was dominated by ORB puts on a down day,
   i.e. the same beta artifact, not validation.

## Tool
`scripts/promotion_scorecard.py` — computes every gate per family and prints the
PROMOTION REVIEW block. It never promotes anything; it reports the decision.

---

## v2 gates (implemented 2026-07-27)
Beyond G1–G3 the scorecard now enforces:
- **G4 MULTI-WINDOW** — +EV at 10d, 30d AND 90d; lifetime non-negative. A family
  that works in only one window is not stable.
- **G5 STATISTICAL CONFIDENCE** — bootstrap 95% CI of EV must EXCLUDE zero, and
  Wilson lower bound on win rate > 0.40. Prevents promoting noise.
- **G6 ALPHA vs BETA (measurable)** — +EV on both up and down days,
  |corr(trade P&L, SPY session return)| < 0.50, AND direction-residualized
  alpha > 0. Turns the qualitative "that's beta" observation into a gate.
- **G7 ELIGIBILITY** — multi-leg families INELIGIBLE until engine v3.
- **Lifecycle stages**: research → shadow → candidate → pilot → active;
  archived (enough evidence, negative) / ineligible (engine limitation).
- **Machine-enforceable** — emits `data/promotion_scorecard.json`
  (`{strategy, eligible, stage, failed[], ev, ev_ci95, beta_corr, alpha_residual_ev}`)
  so production consumes the verdict directly instead of a human reading a report.

## KNOWN METHODOLOGY LIMITATION (must fix before any real promotion)
`--cap-per-session` SUBSAMPLES signals per session, which makes results vary
run-to-run: a 30-day run showed CALL_SWEEP EV −$20.53 while a 120-day run's 30d
window showed +$15.79 — same data, opposite sign. **Any binding promotion
decision must replay EVERY signal (no cap).** The gates are sound; capped
sampling makes the inputs unstable. Treat capped runs as indicative only.

## Standing evidence statement (wording discipline)
Do NOT write "every family is a directional bet, not an edge." The supportable
claim is: **over the evaluated window, no tested family demonstrated
regime-independent positive expectancy, and performance is strongly dependent on
market direction.** Future data in different volatility regimes could change
this — consistent with "archived under observed conditions."

## Current standing (120d, capped — indicative)
No family eligible. CALL_SWEEP is the only one with positive residual alpha
(+$37.42) and positive EV across all windows, but fails sample (44<50), EV CI
crosses zero, Wilson win-rate, and regime dependence → stays `shadow`, watch only.
