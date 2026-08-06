# MES Production Readiness & Controlled Live Validation Audit — 2026-08-06

**Research/audit exercise. No production code was modified.** Per the stated
assumptions: broker execution capability and Cash-vs-Margin are treated as
solved/ignored for this design exercise — the question answered here is "if
execution existed, what is the safest possible path, and does the evidence
support even a maximally constrained trial?"

> **Revision note (post-review, three rounds):** the original version of this
> report answered readiness as one binary verdict. That conflated questions that
> answer differently: engineering readiness is deterministic (the code exists or
> it doesn't), strategy readiness is probabilistic (evidence accumulates toward
> a threshold), whether *production execution behaves like the research
> environment it was validated in* is continuous once live (round two), and —
> added in round three — whether the shadow signal that everything else is
> built on is itself **observable and verified in real time**, since a broken or
> silently-dropped signal pipeline would invalidate every later live-vs-shadow
> comparison without anyone knowing. §7 is now four checks: 7.1 engineering
> (checklist), 7.2 strategy (staged checkpoints at 25–30/~50/~100 trades,
> replacing one hard n=100 bar), 7.5 shadow observability & alerting
> (pre-activation, verifies the bridge between research and future live
> execution is trustworthy), and 7.4 shadow-vs-live divergence (continuous from
> the first live trade). §7.5 sits physically after §7.4 in this document (added
> later) but is logically a **pre-activation** gate alongside 7.1/7.2, not a
> continuous one like 7.4 — see the ordering note at the top of §7. §4.4's build
> list is an explicit, ordered engineering roadmap. Findings and numbers are
> unchanged throughout.

---

## Executive Summary

This audit answers **four separate questions**, because they have different
answer types, different owners, and — critically — a strategy can pass any
subset of them and still fail on the rest:

- **Engineering readiness — deterministic, checklist-based.** ❌ **NOT
  PRODUCTION READY.** The live-facing code (`shree/signal_bot/bot.py`, 278 lines)
  has **zero** execution layer, position manager, watchdog/heartbeat, restart
  recovery, live order reconciliation, or dead-man switch. This is not a matter
  of degree — each item is present or absent, and today they are all absent by
  design (deleted at the Cash-account conversion). Fixable by building; scored
  in §1 and §7.1.
- **Strategy readiness — probabilistic, evidence-based.** ⚠️ **INCONCLUSIVE,
  not proven positive and not proven zero.** n=29 actionable signals since
  2026-07-21. Win rate 58.6%, 95% CI **[40.7%, 74.5%]**. Mean R +0.24, 95% CI
  **[−0.19, +0.66]** — includes zero. This means the sample is too small to
  distinguish real edge from variance, in either direction. It does **not** mean
  the strategy has no edge. Consistent with six prior structural audits showing
  MES 15m is a random walk on real IB data. Resolved by staged checkpoints, not
  a single hard gate — see §7.2.
- **Execution-fidelity readiness — continuous, not a one-time gate.** Not yet
  applicable — no live trade has occurred. Once §7.1 clears and trading starts,
  §7.4 checks every single trade for slippage-band violations, duplicate orders,
  and position mismatches against the broker. This exists because a strategy
  with real shadow-measured edge can still lose live for reasons that have
  nothing to do with the strategy — because execution reality (fills, timing,
  reconciliation) diverges from what the research environment assumed. Neither
  §7.1 nor §7.2 alone would catch that; §7.4 is what proves the first two gates'
  results actually transfer to production.
- **Shadow observability readiness — pre-activation, verified today by reading
  the code.** ❌ **NOT YET RELIABLE.** The current Telegram alert (verified in
  `shree/signal_bot/bot.py:225-231`) sends on every actionable signal, but: no
  `signal_id` exists anywhere in the emitted record (§7.5 needs one to link
  alert → replay → future live execution → P&L), delivery is fire-and-forget
  (`send_message_background` discards the success/failure result — even
  *successful* sends go unconfirmed, not just failures), and there is
  **documented precedent for silent failure in this exact codebase**: a
  2026-08-06 code comment in `telegram_notifier.py` records that an HTML
  parse error silently dropped both alerts for the first live fill of an
  observation period. This is not a hypothetical risk — it already happened
  once with this infrastructure. Scored in §7.5.
- **What CAN be done today, safely:** nothing changes about *whether* to go
  live. What changes is *how to find out*, on two independent tracks that should
  run in parallel without being conflated: (1) build and de-risk the execution
  layer in paper/simulation, (2) keep accumulating shadow evidence toward the
  first checkpoint. A maximally constrained 1-contract, 1-trade/day, RTH-only
  protocol with hard circuit breakers is designed below (§4), reusing this
  repo's own proven prior art (the deleted `RiskGateConfig`, the live Gold
  `GoldRiskManager`) rather than inventing new mechanisms.
- **One rule that governs the whole rollout:** change one variable at a time.
  Full-ETH shadow, live execution, any broker change, and any risk-rule change
  must each be isolated so that if performance moves, the cause is
  attributable. See §4.0.

---

## 1. Production Readiness Review

### 1.1 What exists (verified by reading the code)

| Safeguard | Status | Evidence |
|---|---|---|
| Stop-loss / take-profit **computation** | ✅ Present | `es_fifteen_min.py` computes `SL = clamp(ATR × mult, floor, ceiling)`, `TP = SL × R:R` at 7 call sites (lines 378, 406, 449, 1881, 2453, 2574, 2688). Floor 6pt, ceilings 12–20pt depending on family. |
| Stop-loss / take-profit **sanity** | ✅ Clean historically | 0 of 29 actionable signals had a missing or degenerate (stop==entry) stop/target. |
| Duplicate **bar** processing prevention | ✅ Present | `bot.py:250-251` — `_last_bar_ts` check skips re-evaluating an already-processed bar. |
| Warmup / insufficient-data guard | ✅ Present | `bot.py:105-107` — requires ≥60 bars before evaluating; logs and skips otherwise. |
| In-progress-bar exclusion | ✅ Present | `bot.py:121-124` — drops the still-forming bar so indicators never compute on partial data. |
| IB reconnect on data-fetch failure | ✅ Present | `bot.py:255-262` — catches `ConnectionError`/`TimeoutError`, reconnects next cycle. |
| Catch-all cycle failure isolation | ✅ Present | `bot.py:263-264` — one bad cycle logs and continues; doesn't crash the process. |
| Process supervision / auto-restart | ✅ Present (OS-level) | `deploy/launchd/com.shree.mes-bot.plist` — `KeepAlive` + `ThrottleInterval`. Respawns on crash. |
| Graceful, prompt shutdown | ✅ Present | `bot.py:265-270` — interruptible 1-second poll loop so SIGTERM exits within ~1s instead of blocking up to 15 min. |

### 1.2 What does NOT exist (the actual gap for live)

| Safeguard | Status | Why it matters |
|---|---|---|
| **Order placement / execution** | ❌ **Does not exist** | Zero occurrences of `placeOrder`, `bracketOrder`, `reqExecutions` anywhere in `shree/signal_bot/`. This is not a bug — it was deliberately deleted (commit `f43bfed`, "convert to signal-only bot, delete execution stack"). **Going live requires writing this layer, not flipping a flag.** |
| **Max position size enforcement** | ❌ Does not exist | No position-tracking object anywhere in the live path. Nothing would stop a duplicate signal from stacking contracts. |
| **Duplicate SIGNAL/ORDER prevention** (order-ID level) | ❌ Does not exist | The `_last_bar_ts` check (§1.1) prevents double-*evaluating* a bar, but there is no order-lock, no "position already open" check. `shree/execution/order_lock.py` (281 lines) still exists in the repo — built for exactly this, still used by the Gold bot — but is **not wired into MES** at all. |
| **Stale-signal-before-execution check** | ❌ Does not exist | No "is this signal older than N seconds, don't act on it" gate. Irrelevant today because nothing acts on signals; becomes mandatory the moment an order layer exists (IB latency, slow fills). |
| **Market-data integrity checks** | ⚠️ Partial | Warmup and in-progress-bar checks exist (§1.1), but there is **no check for**: gaps in the bar sequence, OHLC sanity (high<low, zero/negative price), or a stale-feed timeout (if IB stops sending new bars, the bot silently keeps polling with no alert). |
| **Dead-man's switch / heartbeat alerting** | ❌ Does not exist | Grep for `heartbeat|watchdog|dead.?man` in the signal bot and session manager: **zero matches.** If the bot process is alive but stuck (e.g., IB connected but no bars arriving), nothing detects or alerts on it. Launchd only restarts on process *death*, not on a *hung* process. |
| **Daily loss cap / weekly loss cap enforcement** | ❌ Does not exist for MES today | **Existed before deletion** — `RiskGateConfig` (deleted in `f43bfed`) had `daily_max_loss_usd=$150`, `weekly_max_loss_usd=$500`, `max_consecutive_losses=3`, `margin_buffer_usd=$1000`. This is proven, tuned prior art for this exact strategy and should be the starting point for any live risk gate, not a fresh design. |
| **Consecutive-loss circuit breaker** | ❌ Does not exist for MES today | Same source. The **live Gold bot** (`shree/execution/gold/risk.py`) has an active, working pattern: a `DailyState` object (`realized_pnl`, `trades_today`, `consecutive_losses`, `cooldown_until`) evaluated stateless against a `GoldRiskManager` on every proposed trade. This is the template to reuse for MES, not reinvent. |
| **Kill switch (manual + automatic)** | ❌ Does not exist | No mechanism to halt live trading independent of killing the whole process. |

### 1.3 Operational risk identified

The single highest-risk fact in this review: **the signal-generation code and the
(nonexistent) execution code have zero coupling today.** That is a *feature* for
the current shadow bot (a bug in signal logic can't cause a bad trade), but it
means a live version is not "the same bot with orders turned on" — it is a new
execution layer bolted onto a proven signal generator, and that new layer carries
100% of the operational risk (duplicate fills, runaway position, silent hang).
**Treat it as new, unproven infrastructure requiring its own test cycle**, separate
from and in addition to the strategy's statistical readiness.

---

## 2. Historical Performance Review

Source: `logs/mes_signals.jsonl`, 1,049 rows, 2026-07-21 → 2026-08-06. Replayed
against real IB 1-minute bars (next-bar-open fill, 1 tick slippage/side, $1.70
round-turn fees, stop→target→120min time stop). 3 of 29 actionable signals
predate the available 1-minute bar window (IB caps 1m history ~10 days); their
previously-measured outcomes (all losers, from the 2026-07-22 daily audit) are
included below so the sample isn't survivorship-biased.

### Observed results (n=29, ALL actionable signals)

| Metric | Value | 95% Confidence Interval |
|---|---|---|
| Sample size | 29 | — |
| Win rate | 58.6% (17W/12L) | **[40.7%, 74.5%]** (Wilson) |
| Mean R per trade | +0.238 | **[−0.188, +0.664]** |
| Mean net $ per trade | +$9.07 | **[−$13.83, +$31.97]** |
| Net P&L (total) | +$263.13 | — |
| Profit factor | 1.38 | (gross +$958.68 / −$695.55) |
| t-statistic (mean net) | +0.81 | (need ≈2.0 for 95% significance) |
| Max drawdown | −$392.16 | — |

### RTH-only subset (n=19) vs ETH-only subset (n=7 replayed, 5 total)

| Slice | n | Win rate | Mean R | 95% CI | PF | t |
|---|---|---|---|---|---|---|
| RTH-only | 19 | 57.9% | +0.306 | [−0.270, +0.881] | 1.51 | +1.14 |
| ETH-only | 7 | 85.7% | +0.589 | [−0.136, +1.314] | 3.50 | +1.74 |

### Statistical limitations — observed vs established

**Every confidence interval above spans zero.** That is the load-bearing fact of
this section:

- **Observed:** the engine has made hypothetical money in shadow, at a 58.6% win
  rate, PF 1.38, over 29 signals.
- **NOT established:** that this reflects a real, repeatable edge rather than
  variance. The 95% CI for mean R is [−0.19, +0.66] — a system with **zero true
  edge** would produce a sample exactly like this one with regularity. The
  RTH-only slice (the one directly relevant to "go live on RTH") is *worse*, not
  better: n=19, CI [−0.27, +0.88], t=1.14.
- **This is consistent with, not contradicted by**, six prior structural audits in
  this repo showing MES 15m is a random walk on real IB data (variance ratio
  ≈0.93, Hurst ≈0.55, lag-1 autocorrelation ≈−0.017) and that no price-only rule
  (RSI, EMA cross, Donchian, ORB) survives transaction costs at any timeframe from
  1m to 60m.
- **Standing threshold in this research program:** n≈100 non-overlapping trades
  before treating a result as evidence rather than noise. **Current n=29 is 29%
  of that bar.** Widening the shadow engine to full ETH (done 2026-08-05) roughly
  doubles the accumulation rate going forward.

**One honest caveat in the other direction:** absence of proof is not proof of
absence. A CI spanning zero at n=29 does not mean the edge is zero — it means the
sample is too small to tell. That is precisely the argument for *more shadow data*,
not for skipping ahead to live capital on an unresolved question.

---

## 3. Session Analysis — ETH vs RTH

*(Full detail already produced in `MES_FULL_SESSION_AUDIT_2026-08-06.md`; key
figures reproduced here for this report's self-containment. No new signals have
been emitted since that audit — 1,049 rows / 29 actionable is still current.)*

**Coverage confirmed: the shadow engine evaluates the full CME Globex session.**
~44–48 evaluations every ET hour, 24 hours a day (hour 17 ET is thin — 13
evaluations — because that is the CME daily maintenance halt, not a filter).
`OUTSIDE_RTH` and `OUTSIDE_ENTRY_WINDOW` gates: **0 occurrences each** in the logs
(both disarmed by config). The overnight family allowlist was opened to `[]`
(no restriction) on 2026-08-05 at user request — 0 `OVERNIGHT_ALLOWLIST` blocks since.

### Hourly signal distribution (ET)

| ET Hr | Signals | Net $ | PF | Win % |
|---|---|---|---|---|
| 00–07 | 0 | — | — | — |
| 08 | 2 | +$106.93 | ∞ | 100% |
| 09 | 0 | — | — | — |
| 10 | 1 | −$83.82 | 0.00 | 0% |
| 11 | 3 | +$67.49 | 1.91 | 66.7% |
| **12** | **7 (peak)** | −$10.74 | 0.94 | 42.9% |
| 13 | 5 | +$82.10 | 1.62 | 60% |
| 14 | 2 | +$137.91 | ∞ | 100% |
| 15 | 1 | +$53.59 | ∞ | 100% |
| 16–18 | 0 | — | — | — (incl. CME halt) |
| 19 | 3 | −$2.19 | 0.97 | 66.7% |
| 20 | 2 | +$57.85 | ∞ | 100% |
| 21–23 | 0 | — | — | — |

**No hour has n>7.** No time-of-day conclusion is statistically usable; none
should be gated on.

### RTH vs ETH — do not over-read this

| Slice | n | PF | t |
|---|---|---|---|
| RTH | 19 | 1.51 | +1.14 |
| ETH | 7 | 3.50 | +1.74 |

ETH *looks* better. This should **not** be believed: 4 of the 7 ETH trades are one
`EMA9_PB_LONG` setup firing on consecutive bars during a single move (08-02); this
is the opposite sign from a controlled 2-month replay study
(`ETH_SUPPRESSION_RESEARCH_2026-08-05.md`) that measured the incremental ETH set at
**−$135.20, PF 0.78** with a byte-identical RTH control. Two events dressed as
seven observations. **Trust the controlled study over this live slice until n grows.**

---

## 4. Live Validation Plan

Designed against the user's stated constraints (1 contract, ≤1 trade/day, one open
position at a time, shadow continues in parallel). Built on this repo's own proven
prior art rather than new invention.

### 4.0 One variable at a time (governing principle)

The signal engine has already had one variable changed recently: shadow was
widened to full ETH on 2026-08-05 (§3). That change is **isolated and already in
progress** — its effect is being measured on its own, with a byte-identical RTH
control (`ETH_SUPPRESSION_RESEARCH_2026-08-05.md`). Everything from here forward
must preserve that isolation:

- **Do not** introduce live execution and expand session coverage in the same
  step. ETH expansion is a *shadow-only, already-isolated* experiment; live
  execution is a *separate* experiment that should start on the
  already-better-evidenced RTH slice (§2), independent of how the ETH question
  resolves.
- **Do not** change broker/execution adapter and risk-rule parameters in the
  same rollout step. If a broker changes, hold risk rules fixed and vice versa.
- **Do not** change the signal engine itself while validating execution.
  §4.4's build is scoped to *execution*, not *strategy* — no threshold,
  indicator, or confidence change should ship alongside it.
- Practically: each of {session coverage, execution layer, broker/adapter,
  risk parameters} gets its own before/after comparison before the next one
  changes. If live performance moves, this is what makes it possible to say
  *why*.

### 4.1 Position & frequency limits

| Control | Value | Source |
|---|---|---|
| Contracts | 1 (hard cap, code-enforced, not config-only) | User requirement |
| Trades per day | 1 maximum | User requirement |
| Concurrent positions | 1 (no new entry while one is open) | User requirement; requires porting `order_lock.py` |
| Trading window | **RTH only to start** (09:30–16:00 ET) | RTH is the better-evidenced slice (§2); defer ETH live until RTH validation completes |
| Signal selection when >1 fires in a day | Take the **first** qualifying signal only; log all others as shadow-only "would have been 2nd/3rd today" | Avoids cherry-picking the "best-looking" signal after the fact |

### 4.2 Risk gate (reuses the deleted `RiskGateConfig`, tightened for validation)

| Parameter | Prior production value | Recommended validation value | Rationale |
|---|---|---|---|
| `max_contracts` | 1 | **1** | unchanged |
| `daily_max_loss_usd` | $150 | **$150** (≈ 1 stop-out at max stop distance) | unchanged — already conservative |
| `weekly_max_loss_usd` | $500 | **$300** | tighter — this is validation, not production |
| `max_consecutive_losses` | 3 | **2** | halt sooner during validation |
| `min_stop_points` / `max_stop_points` | 6 / 12 | unchanged | already-validated bounds |
| `avoid_close_window_minutes` | 60 | unchanged | avoid entries into the maintenance halt |
| **Review cadence** | — | **staged checkpoints at 25–30 / ~50 / ~100 live trades (§7.2)** | replaces a single hard gate — see revision note |

### 4.3 Automatic halt conditions (circuit breakers)

1. **2 consecutive losses** → halt live entries, continue shadow, require manual review to resume.
2. **Daily loss ≥ $150** → halt for the remainder of the session.
3. **Weekly loss ≥ $300** → halt for the remainder of the week.
4. **Any execution anomaly**: fill price >2 ticks from expected, order rejected, unexpected position size, IB disconnect during an open position → **immediate halt**, alert, manual review before any further live entry.
5. **Data integrity failure**: bar gap detected, stale feed (no new bar within 20 min during RTH), OHLC sanity failure → halt live entries (shadow may continue logging with a flag).
6. **Checkpoint reached (25–30 / 50 / 100 live trades, §7.2)** → mandatory stop for review regardless of P&L.

### 4.4 What must be built before any of this can activate — ordered engineering roadmap

This plan cannot be switched on today — per §1.2, the execution layer does not
exist. Five components, in dependency order. Each is independently testable in
paper/simulation before the next depends on it.

**1. Execution adapter (broker abstraction layer)**
Submit / cancel / status APIs behind a broker-agnostic interface — not written
against one broker's SDK directly. IB is the current live data connection and
the natural first implementation; keep the interface broker-agnostic so a
different execution venue (the door is open, nothing decided) doesn't require
touching the strategy or risk layers. Reuses the IB contract-resolution logic
already in `signal_bot/bot.py`.

**2. Persistent position state**
Must survive a process crash or restart — today the bot has no state at all
across restarts. On startup, **reconcile against actual broker-reported
state** before doing anything else; never assume in-memory state is correct
after any restart. This is the component most likely to cause a real loss if
skipped (a restart that "forgets" an open position).

**3. Watchdog / heartbeat — three independent heartbeats**
- **Process heartbeat**: is the loop still iterating (catches a hang, which
  launchd's death-only restart does not).
- **Data heartbeat**: are new bars still arriving on schedule (catches a stale
  IB feed the bot would otherwise poll silently forever).
- **Broker heartbeat**: is the execution connection alive and authenticated.
Any one failing → automatic halt of new entries. This is §1.2's
highest-priority gap — build it before anything that can hold a live position.

**4. Kill switch — five independent triggers**
Manual stop, daily-loss stop, consecutive-loss stop, disconnect stop, data-quality
stop (§4.3) — each must be able to halt trading **independently** of killing the
whole process, and independently of each other (one tripping doesn't require
diagnosing all five).

**5. Execution journal**
Every trade records, as a first-class deliverable (not an afterthought): signal
timestamp, decision timestamp, order submission timestamp, broker acknowledgement
timestamp, fill(s), latency, slippage, exit reason. This becomes the primary
input to §5's live-vs-shadow comparison and is typically the most-used
debugging tool once live validation starts — build it alongside the adapter
(#1), not after.

---

## 5. Execution Validation (methodology — no live trades exist yet)

Since no live trade has occurred, this section defines the **comparison
methodology** to apply once §4.4 is built and the first live trades happen. It
does not report fabricated results.

For every live trade, capture and compare against the shadow signal that
triggered it:

| Metric | Definition | Shadow proxy used |
|---|---|---|
| Entry price difference | live fill − shadow-assumed fill (next-bar-open + 1 tick) | shadow replay entry |
| Exit price difference | live fill − shadow-assumed exit (stop/target − 1 tick) | shadow replay exit |
| Slippage | live fill − signal price at emission | signal's `entry` field |
| Missed fills | live order not filled within N seconds / at all | count as a shadow "signal without a trade" |
| Latency | signal-emission timestamp → order-ack timestamp → fill timestamp | n/a (new instrumentation) |
| Execution quality score | (target-adjusted R achieved) ÷ (shadow R for the same signal) | ratio, tracked per trade and rolling 10-trade average |

**Minimum sample before drawing conclusions from this comparison: Checkpoint 1
(25–30 live trades, §7.2)** — this is specifically what Checkpoint 1 is for:
validating execution quality before the sample is large enough to say anything
about the strategy itself. Below that, report descriptively only.

---

## 6. Risk Controls Summary

| Layer | Control | Status |
|---|---|---|
| Position | 1 contract hard cap | Design only — not yet code-enforced |
| Frequency | 1 trade/day, 1 concurrent position | Design only |
| Daily | $150 max loss halt | Spec exists (prior `RiskGateConfig`); not wired to MES |
| Weekly | $300 max loss halt (validation-tightened) | New spec |
| Streak | 2 consecutive losses halt | New spec (tighter than prior 3) |
| Execution | fill-price sanity check | Not built |
| Data | feed staleness / gap detection | Not built |
| Process | heartbeat / dead-man's switch | Not built — **highest-priority gap** |
| Kill switch | 5 independent triggers: manual, daily-loss, consecutive-loss, disconnect, data-quality (§4.4.4) | Not built |
| Execution journal | signal→decision→order→ack→fill→exit, with latency and slippage (§4.4.5) | Not built — build alongside the adapter, not after |
| Review cadence | staged checkpoints: 25–30 / ~50 / ~100 trades (§7.2) | New — replaces single hard gate |

---

## 7. Go / No-Go Decision Framework

Four checks, because they answer different questions on different timelines.
Logical order (not document order — §7.5 was added after §7.4 and is placed
after it below, but belongs here in the sequence):

1. **§7.1 Engineering Readiness** — can it operate safely?
2. **§7.2 Strategy Readiness** — is there evidence of edge?
3. **§7.5 Shadow Observability & Alerting** — can every decision be seen and verified, right now, in shadow?
4. **§7.4 Shadow vs Live Divergence** — once live, does execution still match research?

**§7.1, §7.2, and §7.5 are pre-activation gates — all three must pass before the
first live trade**, and each can be tracked and worked on independently. §7.5 in
particular must clear *before* live trading precisely because it validates the
research-to-live bridge itself: if shadow alerts are unreliable, later
live-vs-shadow comparisons (§7.4) cannot be trusted, because you would not know
whether a discrepancy is a real divergence or a signal that was never correctly
observed in the first place. **§7.4 is the only continuous check; it runs from
the first live trade onward**, because it answers a question none of the other
three can: does the built, edge-carrying, observable system actually behave in
production the way it did in research?

### 7.1 Engineering Readiness Gate (deterministic — pass/fail checklist)

Not a maturity scale. Every item below is present or absent today (§1), and
every item must flip to present before activation:

- [ ] Execution adapter built and broker-connectivity-tested (§4.4.1)
- [ ] Persistent position state built, crash/restart-tested, reconciles against broker state on startup (§4.4.2)
- [ ] All three heartbeats (process, data, broker) operational and independently tested — simulated failure of each triggers halt within 5 min (§4.4.3)
- [ ] All five kill-switch triggers wired and independently testable (§4.4.4)
- [ ] Execution journal capturing every field in §4.4.5, verified on paper trades
- [ ] Risk gate (§4.2) fail-closed behavior verified — gate unavailable → no trade, not a silent bypass
- [ ] Full build run in paper/simulated environment for ≥ 2 weeks with **zero** execution anomalies
- [ ] Account supports MES execution (Margin, or otherwise capable) — *excluded from this analysis per stated assumptions, but a real gate in practice*

**Status today: 0 of 8 complete.** This gate is entirely within engineering
control and does not depend on how the strategy question resolves.

### 7.2 Strategy Readiness Checkpoints (probabilistic — staged, not a single hard gate)

Each checkpoint has its own question. Passing a checkpoint means "proceed to
the next stage of accumulating evidence," not "the strategy is proven."

**Checkpoint 1 — 25 to 30 live trades**
Question: *does execution work?* Not yet a strategy verdict.
- Validate execution quality (§5): fill slippage, latency, missed-fill rate.
- Compare live fills directly against the shadow signal that triggered each trade.
- Confirm zero engineering failures (§7.1 items did not regress once live).
- Recalculate win rate / mean R / CI on the live sample as a first look — not a decision input yet, just a health check.

**Checkpoint 2 — ~50 live trades**
Question: *is the evidence moving in a usable direction?*
- Recalculate expectancy and its 95% CI on the full live sample.
- Recalculate slippage and operational reliability trends since Checkpoint 1.
- Compare live CI against the pre-live shadow CI (§2) — are they consistent, or diverging?

**Checkpoint 3 — ~100 live trades**
Question: *does the evidence support scaling?*
- Full statistical review: does the 95% CI for mean R exclude zero?
- Decide continue-at-current-size / scale / stop, using the same evidence
  standard applied throughout this research program (n≈100 non-overlapping
  trades before treating a result as more than noise).

This staged structure avoids two failure modes at once: waiting months (a
single n=100 gate) before learning anything, and scaling prematurely off a
small, noisy early sample.

### 7.3 Live Operation Rules (apply only once all pre-activation gates — §7.1, §7.2, §7.5 — have cleared and trading is live)

**CONTINUE**
- Fewer than 2 consecutive live losses, AND
- Daily loss < $150, weekly loss < $300, AND
- No execution anomaly (§4.3.4) in the last 5 trades, AND
- Live results track shadow within the expected slippage band (§5).

**PAUSE and investigate** (halt new entries, keep existing position management active)
- 1 consecutive loss combined with any execution-quality metric (§5) more than 1
  standard deviation worse than the shadow proxy, OR
- Any single data-integrity flag (§4.3.5), OR
- Live win rate over the last 5 trades diverges from the shadow win rate for the
  same signals by >20 percentage points (suggests execution, not strategy, problem).

**STOP** (return to shadow-only)
- 2 consecutive losses (§4.3.1), OR
- Daily loss ≥ $150 or weekly loss ≥ $300 (§4.3.2/3), OR
- Any execution anomaly confirmed as a code defect (not just a data blip), OR
- At any checkpoint (§7.2), the live-trade mean-R 95% CI is materially worse
  than the pre-live shadow CI — i.e., live execution is destroying edge that
  shadow measurement said existed.

**Consider INCREASING trade frequency** (only after Checkpoint 3 clears with a
sustained CONTINUE status)
- Checkpoint 3 passed with mean-R 95% CI excluding zero, AND
- Zero STOP-triggering events in the trailing 30 trades, AND
- Execution-quality ratio (§5) ≥ 0.9 sustained.

**No frequency increase is justified by the current n=29 shadow sample under any
reading of this data — Checkpoint 1 has not even started, because §7.1 has not
cleared.**

### 7.4 Shadow vs Live Divergence Gate

§7.1 and §7.2 answer "is it built" and "is there edge." Neither answers a third,
separate question: **does the built system actually behave like the environment
the edge was measured in?** A strategy can carry positive expectancy in shadow
and still lose live for reasons that have nothing to do with the strategy —
because production execution diverges from the research assumptions it was
validated under (fill model, timing, reconciliation). This gate exists to catch
that failure mode specifically, so it is never mistaken for "no edge."

**Baseline reference (binding for every comparison in this section):** every
live execution comparison must reference the **original frozen shadow signal
generated before order submission** — the exact record written to
`logs/mes_signals.jsonl` at signal-emission time. Live results must never be
compared against a modified, re-computed, or hindsight-adjusted signal. This is
what makes §7.4 an integrity check rather than a narrative: if the comparison
baseline itself can drift (e.g., because the signal engine changed after the
trade), a real divergence could be explained away instead of caught. The frozen
record is the one and only source of truth for what was "expected."

**Applies continuously once live, starting with the very first trade** — it does
not wait for a checkpoint, because divergence is a real-time signal, not a
sample-size question.

**Checked after every single live trade:**

| Metric | Acceptable | Escalation if violated |
|---|---|---|
| Entry slippage vs shadow-assumed fill (§5) | within the modeled band (1 tick + fees) | 3 consecutive violations → PAUSE (§7.3) |
| Exit slippage vs shadow-assumed exit (§5) | within the modeled band | 3 consecutive violations → PAUSE |
| Signal-to-order timestamp drift | minimal, stable trade-to-trade | growing trend (not just one outlier) → investigate before next trade |
| Missing trades (signal fired, no live order resulted) | zero, or explained by a known halt condition (§4.3) | any unexplained miss → investigate before next trade |
| Duplicate trades (same signal, >1 live order) | **zero, always** | any occurrence → immediate STOP (§7.3) — this is an order-lock defect (§4.4.4), not a market outcome |
| Position mismatch (bot's believed position ≠ broker-reported position) | **zero, always** | any occurrence → immediate STOP — this is exactly what §4.4.2's startup reconciliation exists to prevent; a mismatch appearing mid-session means reconciliation is incomplete |

**Position mismatch handling — broker state is the only source of truth.** The
bot's internal position record is an *expectation*, not a fact; the broker's
reported position is *reality*. On any disagreement: (1) stop new entries, (2)
reconcile internal state to the broker's reported position (never the reverse),
(3) alert, (4) preserve the journal entries around the mismatch for post-hoc
review. This ordering is deliberate — reconciling *toward* the broker is what
makes §4.4.2's startup check and this mid-session check the same mechanism
applied at two different moments, instead of two different sources of truth
that could disagree with each other.

**Signal-to-fill integrity — latency broken out by stage, not just totaled.**
Every trade's journal entry (§4.4.5) already captures the five timestamps below;
this is the derived diagnostic that makes them useful for attribution rather
than just record-keeping:

| Interval | Computed as | Isolates |
|---|---|---|
| Signal → decision latency | `decision_time − shadow_signal_time` | strategy/decision-logic timing |
| Decision → order latency | `order_submit_time − decision_time` | software/adapter latency (§4.4.1) |
| Order → fill latency | `fill_time − broker_ack_time` | broker/execution-venue latency |
| **Total signal → fill latency** | `fill_time − shadow_signal_time` | end-to-end — the number that actually matters for slippage exposure |

Tracking these as **separate** intervals (not one blended total) is what lets a
slippage problem be attributed correctly: a slow *decision* stage points back at
the strategy process, a slow *order* stage points at the adapter (§4.4.1), and a
slow *fill* stage points at the broker — three different fixes for what would
otherwise look like one undifferentiated "execution is slow" symptom.

**Why this sits apart from §7.1/§7.2:** §7.1 can be fully checked off (adapter
built, journal built, tested clean in paper) and §7.2 can show a healthy CI, and
this gate can *still* fail — paper trading and live trading are not the same
environment (real fills, real latency, real broker state). §7.4 is what proves
the first two gates' results actually transfer to production, trade by trade,
not just once at rollout.

**Operating model — the frozen signal record is the single source both sides compare against:**

```
                 Shadow Engine
                      |
             Frozen Signal Record
           (logs/mes_signals.jsonl —
          the binding baseline, above)
                      |
        +-------------+-------------+
        |                           |
 Research Replay              Live Execution
   (§2, §3 — the                (once §7.1
  evidence this report            clears)
    is built on)
        |                           |
 Strategy Metrics          Execution Metrics
  (win rate, PF,           (slippage, latency,
   mean R, CI)              fills, position)
        |                           |
        +-------------+-------------+
                      |
             §7.4 Divergence Monitor
                      |
          Continue / Pause / Stop  (§7.3)
```

Both branches read the *same* frozen record; neither is allowed to compare
against a version of "what the signal said" that has drifted from what was
actually written at emission time.

### 7.5 Shadow Signal Observability & Alerting Gate

**Purpose:** ensure every actionable shadow signal is visible in real time
*before* any live execution is enabled. Shadow mode must behave as a
production-grade signal system even though it places no orders — because it is
the bridge every later live-vs-shadow comparison (§7.4) depends on. If shadow
alerts are unreliable, that dependency is broken and nobody would know it.

**Current state (verified against the code, not designed in the abstract):**

| Requirement | Status today | Evidence |
|---|---|---|
| Alert fires on every actionable signal | ✅ Present | `bot.py:225-231` — `send_message_background` called on every non-HOLD signal |
| Unique `signal_id` linking alert → replay → future live record → P&L | ❌ **Does not exist** | The emitted record's field set (`ts, bar_ts, signal, confidence, entry, stop, target, expected_r, regime, adx, atr, htf_30m_trend, supporting_evidence, blocking_evidence, strategy`) has no ID field. Nothing today links a Telegram message back to a specific `logs/mes_signals.jsonl` row programmatically — only by eyeballing timestamps. |
| Delivery confirmation captured per signal | ❌ **Does not exist** | `send_message_background` (`telegram_notifier.py:137-165`) schedules `_send_message_safe` as a fire-and-forget task; the underlying `send_message`'s `True`/`False` result (`telegram_notifier.py:70-79`) is never read or persisted. A *successful* send is exactly as unconfirmed as a failed one from the log's point of view. |
| Failure is logged in a way tied to the signal | ⚠️ Partial | `_send_message_safe` logs `❌ Background Telegram send failed: {e}` on exception (`telegram_notifier.py:181-182`), but this is a bare log line — not linked to a `signal_id`, not queryable, not counted anywhere. |
| **Documented precedent for silent delivery failure** | ⚠️ **Already occurred once** | `telegram_notifier.py:100-112` (comment dated AUG 6 2026): an HTML parse error (`400 Unsupported start tag`) from an unescaped `<` in gate/reason text **silently dropped both alerts for the first live fill of an observation period**. A plain-text retry fallback was added after the fact. This is direct evidence the failure mode §7.5 exists to catch is not hypothetical. |

**Required alert content** (design target — build alongside §4.4's execution
adapter, since both need the same `signal_id`):

*Signal information:* strategy/family name, instrument, direction, signal
timestamp, market session (ETH/RTH), entry price assumption, stop level, target
level, expected R:R, confidence score, signal tier *(note: "tier" does not
exist as a concept today — confidence is currently a flat 0.70 constant on
every signal, per §2; a real tier system is a prerequisite for this field to
mean anything, not just a display change)*, reason codes / contributing factors.

*Market context:* current price, trend state, volatility state, volume/flow
information if available, time remaining in session, current open shadow
position state.

*Replay tracking:* the `signal_id` linking:
```
signal_id
 |
 +-- shadow alert
 +-- replay result
 +-- future live execution record (if enabled)
 +-- P&L attribution
```

**Shadow Alert Reliability Gate (verify before live activation):**
- [ ] 100% of actionable shadow signals generate a Telegram alert (measured over the full pre-activation shadow window, not spot-checked)
- [ ] Zero duplicate alerts for the same `signal_id`
- [ ] Zero missing alerts (cross-reference `logs/mes_signals.jsonl` actionable rows against sent-alert records — this cross-reference does not exist today and must be built)
- [ ] Alert timestamp matches signal timestamp within a defined tolerance
- [ ] Delivery latency is measured, not assumed

**Latency tracked per alert** (mirrors the §7.4 signal-to-fill breakdown — same
principle, applied to the alert pipeline instead of the order pipeline):

| Timestamp | Captured today? |
|---|---|
| `signal_generated_time` | ✅ (the record's `ts` field) |
| `telegram_created_time` | ❌ not captured |
| `telegram_sent_time` | ❌ not captured |
| `telegram_delivery_time` | ❌ not captured (Telegram Bot API confirms send, not end-user delivery — capture what the API confirms and label it accordingly) |

Derived: signal→alert-creation latency, alert-creation→send latency, send→confirmed-delivery latency, failed-alert count. None of these four are measurable today because none of the three missing timestamps are logged.

**Alert failures are operational events, not cosmetic ones:**

| Event | Severity |
|---|---|
| Missing shadow alert | Warning / investigate |
| Duplicate shadow alert | Warning |
| Incorrect signal details in the alert (mismatch vs the frozen record) | **Critical** |
| Alert system unavailable while live trading is active | **STOP condition** (§7.3) — if the observability layer that later proves execution-fidelity is itself down, live trading has no verification path and must not continue |

**Why this is a separate gate from §7.4, not folded into it:** §7.4 asks "did
execution match research?" — a question that only exists once live trades
exist. §7.5 asks "did humans/systems actually see the signal in the first
place?" — a question that must already be answered **during shadow-only
operation**, before there is anything to execute. A signal pipeline that is
silently broken (per the documented 2026-08-06 precedent above) could sit
undetected through the entirety of §7.2's checkpoint accumulation, since a
missing alert does not stop the signal from being written to
`logs/mes_signals.jsonl` and used in replay — the *research* evidence stays
intact even while the *observability* layer is broken. §7.5 is what catches that
specific gap, which neither §7.1, §7.2, nor §7.4 would.

**Complete operating model** (supersedes the narrower diagram in §7.4, which
remains valid for illustrating the frozen-record baseline rule specifically):

```
                    Shadow Engine
                         |
                 Frozen Signal Record
             (logs/mes_signals.jsonl —
            the binding baseline, §7.4)
                         |
             +-----------+-----------+
             |                       |
       Telegram Alert          Research Replay
        (§7.5 — must be            (§2, §3)
       100% reliable pre-
          activation)
             |                       |
     Human Observability       Strategy Metrics
      (signal_id links               |
      alert->replay->live)           |
             |                       |
             +-----------+-----------+
                         |
                 Future Live Execution
                    (once §7.1, §7.2,
                     §7.5 all clear)
                         |
              §7.4 Divergence Monitor
                    (continuous)
                         |
                  Continue / Pause / Stop
                        (§7.3)
```

---

## 8. Monitoring Dashboard — daily metrics to track

| Metric | Purpose |
|---|---|
| Trades taken (live) | frequency-limit compliance |
| Shadow opportunities (signals not taken, incl. days with >1 signal) | quantifies what the 1-trade/day cap is leaving on the table |
| Win rate (live, rolling 10 & all-time) | core performance |
| Profit factor (live, rolling 10 & all-time) | core performance |
| Expectancy / mean R (live, with running 95% CI) | **the single most important number** — watch the CI narrow and where it sits relative to zero |
| Drawdown (current & max) | risk exposure |
| Slippage (mean, by direction) | execution quality |
| Execution errors / anomalies (count, by type) | infrastructure health |
| Data quality flags (gaps, stale feed, OHLC failures) | infrastructure health |
| Signal timing by session (RTH count vs ETH count, shadow) | tracks progress toward the n≈100 threshold, RTH and ETH separately |
| Heartbeat status — process / data / broker, last-successful timestamp for each | process health — the currently-missing safeguard, §4.4.3's three independent heartbeats |
| Live-vs-shadow R divergence (per trade and rolling) | isolates execution problems from strategy problems |
| Checkpoint progress (trades toward 25–30 / 50 / 100) | makes the staged review cadence (§7.2) visible without waiting for the checkpoint to arrive |
| Execution journal completeness (any trade missing a required field) | the journal (§4.4.5) is only useful if every trade is fully captured — track gaps as their own health metric |
| Divergence gate status (§7.4): duplicate trades, position mismatches, slippage-band violations | the one gate that runs every trade, not just at checkpoints — surfaces execution-reality problems before they're mistaken for "no edge" |
| Signal-to-fill latency, broken into 4 stages (§7.4) | attributes slowness to strategy decision, adapter, or broker separately instead of one blended "execution is slow" number |
| Shadow alert delivery rate (%), by day (§7.5) | must be 100% before activation; a silent drop here invalidates the research-to-live bridge without anyone noticing |
| Shadow alert latency, 3 stages: creation / send / delivery (§7.5) | mirrors the execution-latency breakdown but for the observability pipeline — same attribution logic |
| Alert-vs-frozen-record content mismatches (§7.5) | a Critical-severity event even pre-live — the alert must always agree with the record it's reporting on |

---

## 9. Prioritized Recommendations

Grouped by track (§7.1 engineering / §7.2 strategy), since they proceed in
parallel and are owned/paced independently.

### Engineering track

| # | Recommendation | Priority | Why |
|---|---|---|---|
| 1 | Add `signal_id` to the emitted record and delivery-confirmation logging to `TelegramNotifier` (§7.5) | **Critical, and start immediately** | Unlike every other item in this track, this does **not** depend on execution capability or account type — it can be built and verified today, against the shadow bot that is already running, with zero blockers. Every day this is delayed, checkpoint-accumulating signals (§7.2) go unverified against the documented 2026-08-06 silent-delivery-failure precedent. |
| 2 | Build the three-way heartbeat (process/data/broker, §4.4.3) | **Critical** | Currently zero detection if the bot hangs while (hypothetically) holding a live position. Worst failure mode in the whole design. |
| 3 | Build persistent position state with startup broker-reconciliation (§4.4.2) | **Critical** | Second-worst failure mode: a restart that "forgets" an open position. |
| 4 | Build the execution adapter + execution journal together, broker-agnostic (§4.4.1, §4.4.5) | High | The journal is the primary debugging tool once live validation starts — build it alongside the adapter, not bolted on after. |
| 5 | Wire all five kill-switch triggers independently (§4.4.4) | High | Each must halt trading without requiring the others to be diagnosed first. |
| 6 | Run the full build in paper/simulated mode for ≥2 weeks, zero anomalies, before touching real capital | High | Separates "does the execution layer work" from "does the strategy have edge" — never test both with real money at once. |
| 7 | Wire the §7.4 divergence gate (slippage-band, duplicate-order, position-mismatch checks) as part of the same build, not a later add-on | High | It's the check that catches "edge is real but production doesn't behave like research" — needs to be live from trade 1, not retrofitted after a problem is already suspected. |

### Strategy track

| # | Recommendation | Priority | Why |
|---|---|---|---|
| 1 | Continue shadow (now full ETH) — this is the single highest-value, zero-cost action available today | High | Directly resolves the open statistical question; no capital risk. |
| 2 | Do NOT activate live trading until §7.1, §7.5, and Checkpoint 1's prerequisites are all ready | **Critical** | CI for mean R includes zero; execution layer doesn't exist yet; shadow alerting isn't yet reliable. None of these three is a judgment call — all are checklist/threshold-based. |
| 3 | Hold the strategy engine fixed while the engineering track builds (§4.0) | High | Prevents conflating "execution problem" with "strategy problem" once live trades start. |
| 4 | When all pre-activation gates clear, start RTH-only (§4.1) | Medium | RTH slice has the larger, less-clustered sample (§2); ETH's apparent edge is not trusted yet (§3). |
| 5 | Run Checkpoints 1/2/3 as scheduled (§7.2), not ad hoc | Medium | Pre-registered checkpoints prevent "peeking and stopping early" bias in either direction. |

---

## Assumptions & limitations (stated)

- Per the task brief: Cash-vs-Margin and broker execution capability are excluded
  from this analysis. In reality, MES cannot be traded live on the account's
  current (Cash) type — this report designs the *protocol*, not an activation.
- All P&L is hypothetical shadow replay; no live trade has occurred, so §5 is a
  methodology, not a result.
- Confidence intervals use a t-approximation appropriate for n<30; treat them as
  indicative, not exact, at this sample size — which itself reinforces the core
  finding that n=29 is too small for firm conclusions.
- The RiskGateConfig values cited (§1.2, §4.2) are read from the pre-deletion
  version of `shree/risk/risk_gate.py` (git history, commit `f43bfed^`) — cited as
  documented prior art, not re-implemented in this audit.

---

## Caveman verdict

Four different question, four different gate. Not one "not ready."

**Engine gate — yes or no, no gray.** Bot only look and log, no hand to trade.
No order arm, no memory across restart, no watchdog, no reconcile with broker,
no kill switch. Eight box to check, zero check today. This gate not about luck
or sample — it about build or not build. Build it, test in paper two week clean,
gate flip.

**Number gate — not yes, not no, just not enough yet.** 29 signal, win 58.6%,
but math box stretch **40% to 74%**. Edge box stretch **negative to positive** —
cross zero. That NOT same as "no edge." Mean "not enough bone to tell." RTH
slice, the one ask about, box even wider: **−0.27 to +0.88**.

Don't wait to hundred before learn anything — learn in stage. First
twenty-five-thirty trade: check execution work, not check edge yet. Next fifty:
recheck box, recheck slip. Near hundred: real answer on edge, real answer on
scale.

**See gate — can eye actually watch every signal, right now, today, no live
money need.** Look in code: bot DOES send message on every real signal — good.
But no ID tag on signal, so cannot chain alert → replay → later live trade →
money. And send is fire-forget: even GOOD send never confirm back. Worse — comb
find real proof already: one day (AUG 6) bad character in message text made
Telegram **silently eat two alert whole**, nobody see, code comment only place
it written down. Ghost signal already walk once in this exact camp. This gate
fix cheap and fix now — no wait for other gate, no wait for live, shadow bot
already run today, fix beside it same day.

**Trip-wire gate — the one that never sleep once live start.** Other three gate
all pass clean. And STILL — real trade in real world not match paper trade in
shadow world. Fill different, timing different, broker count different from
what bot think. This gate watch EVERY live trade, not just at checkpoint: any
duplicate order, any position bot-count not match broker-count — **stop same
second, no waiting for pattern.** Broker count always win over bot count, never
other way — bot memory just guess, broker book is truth. Small slip drift —
watch three-in-row before act. Compare only against frozen signal wrote at
birth, never against signal someone touch up after the fact.

Move all four gate, don't tangle: fix see-gate now in shadow, build engine arm
in paper, shadow keep counting bone same time. But when finally go live —
touch **one** thing at a time. Don't open night AND flip live switch AND change
broker AND change risk number all same day. Tangle wire, tangle blame — good
result or bad, cannot tell which wire did it.

Journal every live trade start to finish — signal time, decision time, order
time, ack time, fill time, slip, exit why. Same idea feed the see-gate too:
signal-made time, alert-made time, alert-sent time, alert-land time. Both
journal save more headache than any other piece once live start.

Bottom line: **build gate is checklist, flip when built. Number gate is
staircase, climb at twenty-five, fifty, hundred. See gate is cheap fix, do
today, no excuse to wait. Trip-wire gate never off once live. Walk all four
together, keep wire separate, trade real money only after first three stand
clear — and watch trip-wire every single trade after that.**
