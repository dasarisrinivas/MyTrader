# MES Adaptive Architecture Redesign — Conviction-Weighted, Not Existence-Gated

**Date:** 2026-05-30
**Status:** Proposal
**Root cause (established):** the adaptive layer's only live control variable is the
manager R:R floor (`effective_min_rr`). The strategy emits a fixed, razor-thin R:R band
(1.24–1.36, median 1.255). Any adaptive raise of the floor above ~1.26 mass-rejects
profitable trades (67 of 70 lost cold trades had R:R < 1.33; $717 of $867 P&L lives in
R:R < 1.33 trades). Because the system is path-dependent (single position + evolving
manager state), each rejection forks the trajectory and never reconverges.

**Design principle:** *Learning must influence **conviction (size)**, never **existence
(eligibility)**.* Adapt the bet, not the universe.

---

## 1. Current architecture weaknesses

| # | Weakness | Evidence |
|---|---|---|
| W1 | Adaptive control variable is the **R:R floor**, incompatible with a fixed-R:R strategy | floor 1.2→1.33→1.5 ⇒ trades 135→80→60, P&L $867→$189→$110 (monotonic) |
| W2 | `confidence_floor` is **computed but never read** in `rules.evaluate` — dead code | learning.py derives it; rules.py only uses `min_rr_required` |
| W3 | `auto_suppress` is a **binary hard REJECT** — trade deletion, not degradation | LEARNED REJECTION returns REJECT at conf=90 |
| W4 | Learning changes **trade existence**, creating **path-dependent starvation** | one rejection forks occupancy; 15 warm-unique trades, 43% universe overlap |
| W5 | Bucket cold-start deadlock: suppression stops trades → no new data → permanent suppression | prior RAG/learning zeroing incident |

Net: the adaptive layer is wired to the one axis (R:R) the strategy cannot vary, and on
the wrong action (reject vs. size).

---

## 2. New adaptive architecture

### 2.1 Invariant: R:R is owned by the signal engine
`es_fifteen_min` fixes SL/TP (hence R:R ≈ 1.25). The manager keeps **one static** R:R
sanity floor (`cfg.min_rr_ratio = 1.2`) as a hard risk gate. **The adaptive layer may
never modify R:R, SL, or TP.**

### 2.2 The single adaptive output becomes a size multiplier
Replace `AdaptiveThresholds.min_rr_required` (and the dead `confidence_floor`) with a
continuous **`size_multiplier`** and a **`conviction_delta`**:

```
size_mult =  bucket_quality_mult   # 0.5 – 1.25  (from bucket expectancy / edge_ratio)
           × confidence_mult       # 0.8 – 1.2   (from adaptive conviction, §2.4)
           × regime_mult           # 0.6 – 1.0   (from regime, §2.6)
size_mult = clamp(size_mult, 0.25, 1.50)
final_qty = max(1, round(base_qty × size_mult))     # never 0 except catastrophic
```

Mapping from bucket stats (replaces `derive_thresholds`’s R:R logic):

| Bucket evidence (n ≥ MIN_SAMPLE) | size_mult | Note |
|---|---|---|
| WR ≥ 65% OR expectancy strongly + | **1.25×** | press winners |
| WR 55–65% | 1.10× | |
| WR 45–55% / neutral | 1.00× | |
| WR 35–45% | 0.70× | trim, don't delete |
| WR < 35% AND expectancy < 0 AND edge < 1.5 | **0.50×** | graceful degrade |
| **Catastrophic:** n ≥ 30 AND WR < 10% AND expectancy ≤ −1R | 0.0× (skip) | the *only* deletion, near-never |

This preserves participation (so buckets keep producing data → no cold-start deadlock) and
applies the learning where it's mathematically safe: the **bet size**, where a 0.5×
on a bleeding bucket halves its drawdown contribution without forking the trade universe.

### 2.3 Position-occupancy decoupling (path-dependence mitigation)
Because sizing never rejects, the **trade universe is stable across regimes** —
occupancy timelines stop forking on learning state. (The hard risk gates in §2.5 still
gate existence, but those are deterministic risk rules, not learning.)

### 2.4 Confidence as the conviction channel (revives W2 dead code)
Route the previously-dead confidence path into `confidence_mult` (0.8–1.2):
- **Penalty:** bucket bleeding → conviction −, `confidence_mult` toward 0.8.
- **Boost:** bucket strong + aligned regime → `confidence_mult` toward 1.2.
- **Decay:** confidence adjustments decay toward 1.0 with age (see §2.7 memory).
- **Recovery:** a winning trade in a previously-poor bucket lifts conviction.
Confidence affects **size**, never eligibility. (If a soft confidence gate is ever wanted,
it must be a size taper to the 0.25 floor — never a hard 0.)

### 2.5 Hard gates remain — but only for *risk*, never *learning*
Keep as existence gates (deterministic, not adaptive): per-trade risk cap, daily-loss
cap, drawdown halt, consecutive-loss cooldown, LOCKED/PROBATION posture, startup-bar
gate. These are circuit breakers, independent of bucket stats.

### 2.6 Regime adaptation = size, not suppression
Replace CHOP/bucket suppression with a `regime_mult`:

| Regime | regime_mult | Rationale |
|---|---|---|
| Trending (ADX ≥ 22) | 1.0× | full participation |
| Mixed (15 < ADX < 22) | 0.85× | |
| Choppy (ADX ≤ 15) | 0.6× | reduce, don't delete |
| High vol (ATR percentile high) | 0.7× + stop normalized | survive noise |
| Low vol | 0.9× | tighter participation |

The CHOP Block-All Guard (current hard HOLD) becomes a `regime_mult = 0.6` taper.

### 2.7 Bounded, decaying memory (prevents stale-regime poisoning)
The learning store must **forget**: weight bucket stats by recency (exponential decay,
~60-day half-life) and/or cap to the last N trades per bucket. This addresses the
year-1-poisons-year-2 effect at the source — old regimes fade rather than dictate.

---

## 3. Expected impact (directional — requires walk-forward validation)

| Metric | Current (warm/full) | Target (redesign) | Basis |
|---|---|---|---|
| Trade count | 60 | **~135** (cold level) | no R:R rejection → universe restored |
| Win rate | 50% | **~55–57%** | cold baseline; sizing concentrates on good buckets |
| Profit factor | 1.08 | **≥ 1.29, likely higher** | EV-weighting amplifies good buckets, shrinks bad |
| Max drawdown | −0.80% | **≤ −1.3% (controlled)** | 0.5× on bleeders halves their DD; risk gates unchanged |
| Path divergence | 57% | **≈ 0%** | sizing doesn't fork the universe |

Cold (base floor, no adaptation) is the *floor* of expected performance: **135 trades,
+$867, PF 1.29, −1.32% DD**. The redesign should meet or exceed it because conviction
weighting is strictly more information than a flat book, applied without deleting trades.
*Exact numbers must be confirmed by the walk-forward harness (`scripts/walk_forward_mes.sh`).*

---

## 4. Migration plan (incremental, each step independently backtestable)

1. **Phase 1 — Neutralize R:R adaptation (1-line, immediate). ✅ IMPLEMENTED 2026-05-30**
   (awaiting backtest validation). Cap `adaptive.min_rr_required` at `cfg.min_rr_ratio`
   in `rules.evaluate`, gated by `cfg.adaptive_modifies_rr` (default False = fix on).
   MES-only (`evaluate()`); SPY (`evaluate_spy()`) unaffected. See
   `docs/adaptive_architecture_implementation.md`.
2. **Phase 2 — Size multiplier from bucket quality.** Add `size_multiplier` to
   `AdaptiveThresholds`; map bucket stats per §2.2; consume in the position sizer.
   Convert `auto_suppress` → 0.5× (catastrophic → 0×).
3. **Phase 3 — Confidence conviction channel.** Wire confidence into `confidence_mult`;
   delete or repurpose the dead `confidence_floor`.
4. **Phase 5 — Regime sizing.** Replace CHOP Block-All Guard with `regime_mult`; add
   vol-based normalization.
5. **Memory decay (§2.7).** Add recency weighting / capped memory to the learning store.
6. **Phase 4 — MTF (deferred, evaluate last; see §6).**

Validate each phase with cold-vs-new walk-forward before proceeding. Do not stack phases
without a fold-level check.

---

## 5. Specific code modules to change

| Module | Change |
|---|---|
| `shree/trading_manager/learning.py` → `derive_thresholds()` | Stop emitting `min_rr_required` as a tightening lever; emit `size_multiplier` + `conviction_delta` per §2.2. Add recency decay to `get_bucket*`. |
| `shree/trading_manager/learning.py` → `AdaptiveThresholds` | Add `size_multiplier: float`, `conviction_delta: float`; deprecate `confidence_floor`, `min_rr_required`. |
| `shree/trading_manager/rules.py` → `evaluate()` | Remove adaptive R:R override (cap at base); apply `size_multiplier` to `position_size`; convert `auto_suppress` REJECT → 0.5× size (catastrophic-only 0×). |
| Position sizer (`shree/execution/...risk` / `order_coordinator`) | Consume the multiplier: `final_qty = max(1, round(base_qty × size_mult))`. |
| Regime path (`signal_processor` CHOP Block-All Guard) | Convert hard HOLD → `regime_mult` taper. |
| `shree/strategies/es_fifteen_min.py` | **Unchanged** — R:R/SL/TP stay fixed (the invariant). |

---

## 6. Phase 4 — Multi-timeframe (evaluate, do not implement yet)

Current: 15m signal + 30m HTF trend. Recommendation:

- **Option C (15m unchanged) — adopt now.** The strategy is already profitable at the
  base floor (PF 1.29). MTF is an enhancement, not a fix; don't add it while the sizing
  redesign is unvalidated.
- **Option A (15m ORB + 5m confirmation) — best later experiment.** A 5m confirmation
  inside the 15m signal can improve entry timing (fewer SL hits) and modestly lift
  participation, with bounded noise. Run it as an isolated A/B *after* Phases 1–3.
- **Option B (15m regime + 1m execution) — avoid.** 1m execution adds noise and has a
  history of breakage in this codebase; the marginal timing gain doesn't justify the risk.

Decision: **C now, A as a later isolated experiment, B rejected.** Do not change timeframe
assumptions until the conviction-weighting redesign (Phases 1–3) is validated.

---

## Summary

The strategy is sound (cold: 135 trades, +$867, PF 1.29). The adaptive layer fails because
it adapts the **one variable the strategy can't vary (R:R)** via the **wrong action
(reject)**. Re-point it to **size** and **conviction**, keep R:R fixed, make risk gates the
only existence gates, decay old memory, and adapt regimes by sizing. This restores
participation, preserves drawdown control, adds graceful regime/quality weighting, and
eliminates path-dependent starvation.
