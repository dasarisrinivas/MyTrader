# Expectancy-Gate Proposal — replace the flat 2.0 R:R cold-start floor

**Status: PROPOSAL / pre-registration. NOT implemented. No `rules.py` logic changed by this document.**
**Date: 2026-05-22. Implement only on top of tag `shadow-baseline-2026-05-22`.**

This is a deliberate, pre-registered design — not a reactive live tweak. It is written to be
agreed *before* any code change, and rolled out shadow-first, consistent with the project's
existing evaluation discipline.

---

## 1. Problem (observed live, 2026-05-22)

Every directional MES entry today was rejected by gate `Q5_rr_ratio`:

| signal | R:R | required | result |
|---|---|---|---|
| TREND_CONT_LONG ×2 | 1.25 | 2.0 | REJECT |
| EMA21_PB_LONG ×2 | 1.33 | 2.0 | REJECT |

The strategy's natural continuation profile is ~1.25–1.33R at a 1×ATR stop (median MFE ≈1.5×ATR).
The manager mandates ≥2.0. This is the mismatch flagged in prior research ("expectancy gate
instead of flat 2.0 R:R floor").

## 2. Root cause — it's a COLD-START problem, not a missing feature

The expectancy logic already exists. `rules.py` (≈L276–311) calls `adaptive_lookup(...)`:

- bucket pays asymmetrically → **relaxes** `effective_min_rr` below the static floor;
- bucket bleeds (negative EV) → **auto-suppress** (hard REJECT);
- **no empirical data → falls back to `effective_min_rr = cfg.min_rr_ratio` (the static 2.0).**

Today's record shows the fallback firing: `adaptive_bucket: "no empirical data — using static
rules"`. So the flat 2.0 is purely the cold-start path.

**The deadlock:** static 2.0 rejects all directional entries → nothing executes → `learning_db`
never accrues MES bucket outcomes → `adaptive_lookup` keeps returning `None` → the floor stays
2.0 indefinitely. The adaptive layer can never "warm up" from live trading because the cold-start
floor prevents the trades that would populate it.

## 3. Proposed change (two parts, both default-OFF)

**3a. Seed the bucket priors (preferred, no logic change).** Populate `learning_db` with
per-bucket win-rate / expectancy from the strategy's own validated backtest (via the existing
`backfill_learning.run_backfill` path or a one-time import), keyed by `(signal_type, regime)` to
match `adaptive_lookup`'s key. Once seeded, the adaptive layer sets a data-driven
`min_rr_required` and the static 2.0 is no longer hit. This uses the existing machinery end-to-end.
Prerequisite: a trustworthy per-bucket WR prior (source: `backtest_results/`), documented and
versioned.

**3b. Expectancy-based cold-start floor (small, gated `rules.py` change).** For the
*no-empirical-data* branch only, replace the flat `cfg.min_rr_ratio` with a breakeven-plus-margin
R:R derived from a prior win rate:

```
E_R(rr) = WR_prior * rr − (1 − WR_prior) * 1        # risk leg fixed at 1×ATR
require:  E_R(rr) ≥ MIN_E
⇒  required_rr = (1 − WR_prior + MIN_E) / WR_prior
effective_min_rr = required_rr        (cold-start branch only)
```

Gated behind a new config flag `use_expectancy_coldstart` (default **False**). When False, behaviour
is byte-identical to today (static 2.0). When True, the cold-start floor reflects expectancy.

Proposed parameters (frozen for the trial):
- `MIN_E = +0.10R` (require positive expectancy with a margin).
- `WR_prior`: per-`signal_type` from the validated backtest; if absent, a conservative default of
  0.45. Worked examples (corrected 2026-05-22): `required_rr = (1 − WR + MIN_E)/WR`, so WR 0.50 →
  0.60/0.50 = **1.20**; WR 0.45 → 0.65/0.45 = **1.44**; WR 0.70 → 0.40/0.70 = **0.57**; WR 0.41 →
  0.69/0.41 = **1.69**. (An earlier draft mis-stated WR 0.45 → 1.33 by reusing the WR=0.50
  numerator over a 0.45 denominator.) Consequence with the **real** sourced priors below: a
  high-WR continuation bucket easily clears, but a low-WR pullback at 1.33R does **not** — the
  floor discriminates rather than blanket-passing "1.25–1.33R setups."
- Empirical override threshold: once a bucket has `n_trades ≥ 30`, the empirical
  `adaptive.min_rr_required` supersedes the prior (existing behaviour).

## 4. Guardrails preserved (explicitly NOT loosened)

- `auto_suppress` for negative-EV buckets stays — a bleeding bucket is still hard-rejected.
- Soft-pause `max()` floor stays — loss-streak still raises the bar.
- Risk-$ cap (`risk_within_cap`), Q1–Q4 gates, posture ladder all unchanged.
- This change *only* alters the cold-start R:R floor. It cannot admit a bucket the empirical layer
  would suppress, and it changes nothing about position sizing or risk.

## 5. Rollout — shadow first (matches the box_rng_atr discipline)

- **Phase 0 — SHADOW (no behavioural change).** Compute the expectancy-floor decision and log it
  *alongside* the live static decision in `manager_decisions.jsonl` (new field
  `shadow_expectancy: {required_rr, would_pass, wr_prior, e_r}`), without acting on it. Collect
  2–4 weeks. This reuses the now-rich decision log; it does not touch gating.
- **Phase 1 — enable behind `use_expectancy_coldstart=True`** only after Phase 0 evidence clears §6.
- Default-off until explicitly flipped. Implement on top of `shadow-baseline-2026-05-22` so the
  before/after is reproducible.

## 6. Pre-registered acceptance criteria (set before the trial; do not relax after seeing data)

1. **No suppression bypass:** the expectancy floor must never approve a `(signal_type, regime)`
   bucket that `auto_suppress` rejects. (Hard invariant; verify in shadow.)
2. **Out-of-sample positive EV:** on replayed historical signals, the *newly admitted* setups
   (those 2.0 rejected but the expectancy floor passes) must show realized expectancy ≥ `MIN_E` out
   of sample.
3. **Composition sanity:** newly-admitted volume must be dominated by the strategy's high-WR
   continuation buckets, not low-WR junk — i.e. the prior `WR_prior` per bucket must itself be
   backtest-validated, not assumed.
4. **Live paper:** over the Phase-0 shadow window, admitted-trade realized E ≥ 0 with the gate
   would-decisions tracked; flip default-on only if positive.
5. If criteria fail → keep the static floor; the conclusion (expectancy floor doesn't help here) is
   itself acceptable and informative.

## 7. Open prerequisite — RESOLVED for shadow (2026-05-22)

Item 3a/3b both need a **trustworthy per-bucket WR prior**. Both questions are now answered:

(a) **`learning_db` (`data/learning.db`) has ZERO `bot='mes'` rows** — 18 `bucket_stats` and 36
`trade_events`, all `bot='spy_options'`. This confirms the cold-start diagnosis at the data layer:
MES `adaptive_lookup` always returns `None`, so the static 2.0 fires and no MES bucket can ever
accrue — the deadlock in §2 is real, not hypothetical. (Note: `backtest_results/` is the SPY-options
backtest — wrong instrument — so it is **not** the MES prior source.)

(b) **MES per-`signal_type` WR sourced** from
`reports/bt_baseline/backtest_MES_2025-02-01_2026-01-31_20260309_111038_trades.csv` (full-year, real
exit mix: 20 take_profit / 19 stop_loss / 14 flatten):

| signal_type | n | WR | required_rr @ MIN_E=0.10 |
|---|---|---|---|
| EMA21_PB_LONG | 22 | 40.9% | 1.69 |
| TREND_CONT_LONG | 10 | 70.0% | 0.57 |
| OR_BREAK_LONG | 3 | 33.3% | (thin → default 0.45 → 1.44) |
| TREND_CONT_SHORT | 11 | 36.4% | 1.86 |
| EMA21_PB_SHORT | 7 | 42.9% | (thin → default 0.45 → 1.44) |

**Caveats (load-bearing):** samples are small (TREND_CONT_LONG n=10, OR_BREAK_LONG n=3) and pooled
across regime, whereas `adaptive_lookup` keys on `(signal_type, regime)`. These are adequate for a
shadow prior that is only logged and audited; they are **not** yet trustworthy enough to gate live
(criterion §6.3). Frozen prior table + provenance: `shree/trading_manager/expectancy_priors.py`
(`PRIORS_VERSION = "2026-05-22.bt_baseline"`), with rule: use backtest WR when `n ≥ 10`, else the
conservative 0.45 default.

**Phase 0 is IMPLEMENTED (no behavioural change).** `decision_log.append_decision` now attaches a
`shadow_expectancy` block to each MES decision: `{version, wr_prior, n_prior, prior_source,
backtest_wr, min_e, required_rr, e_r, would_pass, static_floor}`. The live decision from `rules.py`
is untouched and byte-identical (verified: today's TREND_CONT_LONG@1.25 would-pass=True,
EMA21_PB_LONG@1.33 would-pass=False, both still live-REJECT). SPY path = n/a (options, no R:R floor).
Takes effect on next trading-manager restart. Collect 2–4 weeks, then judge §6 before any Phase-1 flip.

---

*Pre-registered 2026-05-22. Any parameter change after data is observed requires a new version.*
