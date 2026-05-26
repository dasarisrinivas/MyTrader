# Forward-Test Protocol — `box_rng_atr` gate on relaxed continuation arming

**Status: PRE-REGISTERED. Frozen 2026-05-21, before any 2026-01-17 → present data was observed or fetched.**
**Instrument: ES 15m, longs only. Scope: research validation, no live routing.**

This document exists so the Jan–May 2026 forward run *cannot be interpreted into success
afterward*. Every rule, threshold, metric, and pass/fail band below is fixed now. If any of
them is changed after the forward data is seen, the run is void and must be re-pre-registered.

The forward test is the only test in this chain that answers: **did we find a reusable rule,
or a historical compression artifact?** A null result is an accepted, informative outcome — see
§8.

---

## 1. Hypothesis (falsifiable)

A single, frozen, entry-time feature — `box_rng_atr` (prior 6-bar range ÷ ATR) — distinguishes
**re-timed** continuation setups (real, already-expanded trends the EMA stack confirms late)
from **net-new** setups (rotational drift that never matures), and a gate on it preserves the
relaxed rule's edge over baseline **on data generated after this protocol was written**.

Falsified if the separation and/or the gated edge fail to persist out-of-sample per §6.

## 2. Frozen rule definition (no parameter may change)

All definitions reference the existing, unmodified prototype code in `tools/expansion_research/`.

| Element | Definition (frozen) | Source |
|---|---|---|
| Indicators | EMA9/21/50 (`adjust=False`), Wilder ATR-14, Wilder ADX-14, RSI-14; box = prior 6-bar high-low, `shift(1)` | `expansion_proto.enrich` |
| RTH session | 08:30–15:00 CT (`US/Central`) | `expansion_proto.in_rth`, `P` |
| Episode | contiguous RTH bars, same day, `ema9>ema21 & close>ema50`, ≥2 bars | `continuation_timing.episodes` |
| BASE arm | `full_stack (ema9>ema21 & ema21>ema50) & touch(ema9) & close>ema9 & bull & rsi∈[40,70] & adx≥22` | `continuation_timing.arm_masks` BASE |
| RELAXED arm (the rule under test) | `loose_stack (ema9>ema21 & close>ema50) & touch & close>ema9 & bull & rsi∈[40,70] & (adx≥22 OR adx_slope≥3 over 3 bars with adx≥15)` | `arm_masks` V3 |
| Feature (FROZEN) | `box_rng_atr = box_rng / atr` at the relaxed-arm bar | `entry_filter_proto.entry_features` |
| **Gate threshold (FROZEN)** | **keep entry iff `box_rng_atr ≥ 3.01`** (the block-0 frozen value from the in-sample walk-forward; chosen by fixed median rule, never tuned to test) | `walk_forward.py` strict variant |
| Risk model (FROZEN) | 1×ATR stop, 1.5×ATR target, hold 8 bars (2h), stop-wins-ties | `continuation_timing.sim` |
| Side | longs only | — |

**Cohort labels** (re-timed / net-new / no-change) are computed exactly as
`timing_attribution.classify` — using whether BASE arms later in the episode. These labels use
look-ahead and are for *measurement only*; they are never inputs to the gate.

## 3. Data acquisition (must match the training pipeline exactly)

The training data `data/ib/ES_15m_1y.parquet` is **Interactive Brokers**, 15-min bars, tz
`US/Central`, columns `[open, high, low, close, volume]`, index `timestamp`. The forward slice
**must** be fetched the same way. The Yahoo path (`scripts/download_data.py`) is **forbidden**
(1-min only, ~30-day limit, different timestamping/volume conventions → source mismatch would
masquerade as regime degradation).

1. Fetch via `backtest.data.ib_downloader.IBHistoricalDownloader`, `symbol="ES"`,
   `bar_size="15 mins"`, from IB Gateway (port 4001), range **2026-01-17 → fetch date**.
2. Save to a **new** file `data/ib/ES_15m_fwd_2026.parquet`. **Do not overwrite or append to
   the training parquet.**
3. Pre-flight assertions (run before any analysis; abort on failure):
   - schema identical: columns `[open,high,low,close,volume]`, index tz `US/Central`;
   - no calendar gap vs training data beyond normal market closures at the 2026-01-16/17 seam;
   - bar count per RTH day in the expected 26 (±) range; no duplicate timestamps.
4. Indicators are computed by `enrich()` on the forward slice **with enough lead-in bars**
   (prepend the final ~60 bars of training data so EMA/ATR/ADX are warm at 2026-01-17 open;
   discard those lead-in bars from all metrics). Document the warm-up handling in the run log.

## 4. Causality

Every feature uses only data up to and including the arm bar (`enrich` is causal: `ewm
adjust=False`, trailing rolling, box `shift(1)`). The trend/rotational and session-segment
labels are post-hoc *analysis* strata only and are never inputs to the gate.

## 5. What is measured (frozen metric set)

Computed on the forward slice, longs only, at the 1.5R target:

- **Separation:** AUC(re-timed vs net-new) on `box_rng_atr` — pooled **and** within session
  segment (open 08:30–10:00 / midday 10:00–13:30 / close 13:30–15:00) and within ATR tercile.
- **Gated edge:** blended decided-WR expectancy and mark-to-market mean-R for: base, ungated
  relaxed, relaxed+gate. Edge = gated − base.
- **Regime split:** gated vs base within trend vs rotational days (directional-efficiency
  median split, computed on the forward slice's own days).
- **Mechanism:** dropped-cohort mean-R and loser-rate (suppression) vs kept-re-timed MFE/MAE
  asymmetry (structural).
- **Composition controls (mandatory):** the time-of-day distribution and ATR-regime
  distribution of forward entries, reported beside the in-sample distribution
  (open/midday/close = 47% / 35% / 18% of relaxed entries in-sample; reproduce the table).

## 6. Pre-registered acceptance criteria (fixed bands)

In-sample reference (frozen): pooled AUC 0.78; **within-segment AUC open 0.68 / midday 0.67**;
gated edge over base +0.24R (full-sample) / +0.38R (pooled OOS); base ≈ +0.06R.

| # | Criterion | PASS band |
|---|---|---|
| F1 | **Within-segment separation** (the real test, TOD-robust) | AUC ≥ **0.62** in BOTH open and midday segments |
| F2 | Pooled separation (composition-sensitive, secondary) | AUC ≥ 0.60 |
| F3 | Gated edge over base | gated expectancy > base AND ≥ ungated relaxed, with edge ≥ **+0.12R** (≥ 50% of in-sample full-sample edge) |
| F4 | Regime robustness | gated > base in BOTH trend and rotational days (no sign reversal) |
| F5 | Mechanism | kept re-timed MFE/MAE asymmetry preserved (MFE ≥ 1.8, MAE ≤ 0.8 ×ATR) — i.e. survival is not pure loser-suppression |
| F6 | Composition | forward open/midday/close mix within ±20pp of in-sample; if not, F1 (not F2) governs the verdict |

**OVERALL CONFIRMED** = F1 AND F3 AND F4 (the load-bearing three). F2/F5/F6 are diagnostic
context that determines *why* a pass or fail occurred, not the verdict itself.

Minimum sample: if the forward slice yields **< 8 re-timed AND < 8 net-new** marginal events,
the test is **inconclusive** (underpowered), not pass/fail — extend the window or wait for more
data. Report event counts first.

## 7. Forbidden actions (any one voids the run)

- Changing the threshold (3.01), the feature, the target/stop/hold, the RSI/ADX/EMA params, or
  the RTH window.
- Adding, removing, or combining features (no `stack_gap`/`vol_ratio` stacking, no AND/OR, no
  adaptive/per-regime gating).
- Re-deriving the threshold from the forward distribution ("re-thresholding").
- Dropping days as "outliers", trimming the date range, or selecting a favorable sub-window.
- Switching data source, bar size, or session definition.
- Re-running with a different seed/segmentation until the verdict flips.

If the rule genuinely needs to change, that is a **new** hypothesis requiring a **new**
pre-registration and a **new** forward window — not a revision of this run.

## 8. Decision rule (what each outcome means)

- **CONFIRMED (F1+F3+F4):** the discriminator is a reusable rule, not an artifact. Proceed to a
  default-off, shadow/paper deployment (Phase 1) with the risk model unchanged. Magnitude is
  still soft — size conservatively and confirm in live paper before capital.
- **FAIL on F1 (within-segment separation gone):** the in-sample 0.78 was substantially a
  time-of-day composition effect. The feature is **not** a market-state classifier on its own.
  Informative: stop trying to gate on it; the EMA21>EMA50 lag may be near the minimum viable
  price of trend discrimination at this timeframe (do not "optimize" the strategy into a
  lower-quality participation engine).
- **FAIL on F3/F4 (separation holds, edge does not / reverses in a regime):** conditional
  structure. Edge is regime-dependent; not tradeable as a static gate. Document the regime where
  it breaks; revisit only as an explicit regime-conditioned hypothesis (new pre-registration).
- **INCONCLUSIVE (underpowered):** widen the window or wait; do not soften criteria to force a
  verdict.

## 9. Execution

A single runner — `tools/expansion_research/forward_test.py` (to be written *after* this spec is
frozen) — will: load `ES_15m_fwd_2026.parquet` with warm-up lead-in, apply the frozen rule and
gate verbatim by importing the existing prototype functions, and print §5 metrics with the §6
PASS/FAIL table. It takes **no tunable arguments** other than the input parquet path and the
fetch-as-of date. Its output (with the input file's checksum and row count) is the run log of
record.

---

*Pre-registered by: research session 2026-05-21. Any change below this line requires a new
protocol version and a new, unseen forward window.*
