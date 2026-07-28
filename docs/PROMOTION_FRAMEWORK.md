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

---

# Promotion Framework v2.0 — FROZEN (effective 2026-07-27)

Constitution: `shree/research/promotion_constitution.py` (single source of truth).
Amendments require a **version bump + migration notes + historical re-comparison** —
changing a gate changes history, because prior decisions were made under different
rules.

## Gates G1–G10 (deterministic, no per-run tuning)
G1 n>=50 · G2 EV>0 · G3 95% CI lower>0 · G4 Wilson lower>0.40 ·
G5 +EV at 10d/30d/90d + lifetime non-negative · G6 |beta corr|<0.50 AND
|slope|<5000 · G7 residual alpha>0 · G8 replay completeness==100% ·
G9 deterministic replay only · G10 multi-leg ineligible until engine v3.
COVERAGE: >=30 sessions spanning bull, bear, range, high_vix, low_vix.

## Lifecycle
research → shadow → candidate → pilot → active → **probation** → archived;
ineligible (engine limitation). Probation = a live strategy whose evidence
deteriorated but which does not yet meet archival criteria.

## Invariant (proven necessary 2026-07-27)
**Promotion decisions must use COMPLETE replay datasets. Sampling is exploratory
only, never binding.** Evidence: a capped run scored CALL_SWEEP at n=44 / EV
+$20.13 ("watch"); the uncapped binding run scored the same family at n=204 /
EV **-$11.73** (archived). Subsampling reversed the sign. `--cap` now forces
G9 failure so a sampled run can never promote anything.

## Provenance (every report)
replay engine version + file hash, constitution hash, scorecard hash, dataset
hash, signals attempted/replayed, completeness, binding flag. Guarantees a
verdict is reproducible years later.

## Metric separation
`promotion_metrics` (binding) vs `exploratory_metrics` (interesting, NOT
actionable: residual Sharpe, flat-day EV, consecutive sessions, win rate, hold
time). Prevents overweighting exploratory findings.

## Binding result 2026-07-27 (120d, 1031/1035 replayed)
No family eligible. CALL_SWEEP -11.73 (archived), PUT_SWEEP +1.79 but residual
alpha -33.43, ORB -10.96, PC_RATIO -28.63, TC -24.77; spreads/straddle ineligible.
Beta SLOPE caught what correlation missed: CALL_SWEEP corr 0.34 (passes) but
slope 8855 (fails) — the asymmetric relationship correlation alone hides.

## OPEN CONSTITUTION DEFECT (needs a versioned amendment decision)
G8 demands exactly 100% completeness, but the binding run reached 99.6%
(1031/1035) — 4 signals had no NBBO (contracts that never traded). As written,
G8 can never pass, since unquotable contracts are unavoidable. Proposed
amendment (NOT applied): `G8 >= 0.99` with every unreplayable signal itemized in
the report. Requires version bump to v2.1 + migration notes.

## Wording standard
CALL_SWEEP satisfies one EXPLORATORY criterion (positive residual alpha) but has
negative EV under complete replay and fails multiple promotion gates; it remains
Shadow/archived. Do not describe any family as "the one to watch."

---

# AMENDMENT v2.1 — APPLIED 2026-07-27 (framework now FROZEN)

**Change:** G8 completeness `1.00 → 0.99`; denominator redefined to ALL in-window
signals (no pre-count exclusion); itemization of every unreplayable signal made
mandatory, with a standing upward-bias note.

**Evidence:** 4/1048 signals unreplayable — ALL four from 2026-04-03 (market
closed, Good Friday; vol=0, oi=0; vendor HTTP 472). Zero vendor gaps on tradeable
sessions, zero replay-engine defects. v2.0's 100% rule was therefore unsatisfiable
and blocked every promotion on a technicality.

**Denominator defect also fixed:** v2.0 reported 99.6% (1031/1035) because its
per-session filter silently dropped 13 signals before counting; true completeness
was 98.4% (1031/1048). v2.1 counts them — sessions lacking a computable SPY return
are still replayed and counted, and are excluded from beta/regime math only.

**Historical impact: NONE.** Verified by re-run: every family still fails 4–6
gates besides G8; verdict unchanged (no strategy promotable).

## v2.1 binding result (120d, completeness 99.6% = 1044/1048, G8 PASSES)
| Strategy | Stage | n | Sess | EV$ | CI95 | beta_r | slope | alpha$ |
|---|---|---|---|---|---|---|---|---|
| PUT_SWEEP | shadow | 209 | 19 | +1.52 | [−14.5, +18.3] | −0.27 | −8034 | −33.43 |
| CALL_SWEEP | archived | 206 | 14 | −11.99 | [−27.6, +4.9] | 0.34 | 8855 | +22.66 |
| ORB_BREAKOUT | archived | 163 | 18 | −10.96 | [−23.7, +1.6] | −0.09 | −1314 | −13.80 |
| PC_RATIO_EXTREME | archived | 160 | 18 | −28.33 | [−48.7, −7.2] | −0.38 | −8782 | −39.73 |
| TREND_CONTINUATION | shadow | 49 | 15 | −23.12 | [−37.9, −8.2] | 0.28 | 3280 | −19.33 |
| spreads/straddle | ineligible | — | — | — | — | — | — | — |

**PROMOTION REVIEW: NO strategy eligible.** G8 now passes for all; they fail on
merit (CI crosses zero, Wilson, beta slope, residual alpha, session coverage).

## FREEZE DECLARATION
Replay Engine — **FROZEN** (only sanctioned change: v3 multi-leg).
Promotion Framework — **FROZEN at v2.1**. No new gates, metrics, wording, or
visualizations. Version bumps only for a verified defect that changes a
promotion decision.
Production — **FROZEN**.
Current strategies — **DONE under observed conditions**.
**Next work — NEW ALPHA.**
