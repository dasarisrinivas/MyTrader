# Review: should RAG / historical-similarity hold decision authority at current scale?

**Reviewer stance:** skeptical prop-trading risk reviewer, live capital assumed. Complexity is
guilty until it has *earned* its degrees of freedom out-of-sample.
**Date:** 2026-05-22. Grounded in the actual code/data, not the architecture description.

---

## 0. Premise correction (this changes the question)

The brief assumes "RAG-style retrieval of similar historical setups" is a major trade-approval
factor. **It is not in the gating path.** Verified:

- `rules.py` and `manager.py` import **nothing** from `shree/llm/` or any vector/embedding/retrieval
  module. The RAG subsystem (`shree/llm/rag_storage.py`, `rag_data/{vectors,trades,docs_*}`) is
  LLM journaling / daily market-context narration. It has **zero veto authority today.** Good —
  leave it that way.
- The two things in the path that *resemble* "similarity" are neither RAG nor similarity:
  1. **`_similar_recent_lost` (Q4).** A last-N-closed-trades loser-count veto. Its own docstring
     concedes it has no `signal_type` and no direction — it vetoes a fresh long because *unrelated*
     recent trades lost. This is a crude regime/streak proxy wearing a "pattern match" label.
  2. **`adaptive_lookup` → `bucket_stats`.** **Exact-match contingency-table lookup** on the 5-tuple
     `(signal_type, bot, regime, time_bucket, vix_bucket)`, with a 4-level dimensional fallback
     (L0 full → L3 just `signal_type,bot`). This is stratified frequency estimation, *not*
     nearest-neighbor retrieval.
- For **MES specifically the adaptive layer is inert**: `learning.db` has zero `bot='mes'` rows, so
  every MES lookup returns `None` and the **static R:R floor does 100% of the rejecting.** The
  layer you're worried is over-powered is, for MES, switched off.

So the correct question is not "does RAG have too much authority" but: **is a sparse exact-match
expectancy table (plus a static conservative fallback) a defensible decision authority at n≈36 total
events?** That has a sharper answer, and worse failure modes than RAG would.

---

## 0a. The loss-history rejection ladder (what actually rejects MES on prior losses)

There *is* a live "previous trades lost → reject" path for MES — it just isn't RAG or similarity.
`recent = last_n_closed_trades(cfg.orders_db, n=20)` (manager.py:518) is **signal-type- and
direction-blind** (the Q4 docstring concedes the executions table has no `signal_type`). That pool
feeds four loss-based gates in `evaluate()`:

1. **Q4 `_similar_recent_lost` (rules.py:266).** If **≥3 of the last 5 closed trades lost**
   (`pattern_loss_threshold=3`, `last_n_for_pattern_check=5`) → Q4 fails. Alone → MODIFY (half size);
   combined with *any* other Q-fail → `n_q_fails>=2` → **hard REJECT** (rules.py:405). Blind to
   whether those 5 were the same setup, direction, or strategy. Its 5-trade window is **not**
   session-bounded.
2. **Soft-pause streak (rules.py:228), `consec_losses >= 2`.** Raises R:R floor to **2.5** + confidence
   to 0.70, forces small size, DEFENSIVE. At 2.5 the strategy's 1.25–1.33R setups are rejected
   wholesale; signal-conf < 0.70 → explicit REJECT (rules.py:427).
3. **Hard pause (rules.py:207), `consec_losses >= 5`.** Hard REJECT until session roll or a winner.
   `consec_losses` is computed **since session start** (intraday) via `streaks_from_recent`.
4. **Adaptive auto_suppress "LEARNED REJECTION" (rules.py:289).** The only *bucket-history* reject —
   hard-rejects a bleeding bucket. **Inert for MES** today (zero rows → `adaptive is None`).

The live loss-based rejection is gates 1–3, **not** the learned/retrieval layer. And gates 1–2 are
the real concern: blunt, signal-type-blind recent-loss counters that punish a fresh high-WR
TREND_CONT_LONG for *portfolio-level* roughness — conflating "this setup has a bad record"
(legitimate, expectancy-based) with "we've lost a few lately" (a tilt proxy). This is arguably worse
than similarity retrieval. It also *stacks* with the cold-start static floor: on a 2-loss day the
floor is 2.5, not 2.0, so the rejection of continuation setups is over-determined.

**Fix direction:** make the recent-loss check *conditional on the signal's own bucket* (expectancy
of THIS signal_type), not the undifferentiated last-N pool; keep the intraday streak ladder as a
deterministic *risk* throttle (that part is fine — capital preservation), but stop letting it and Q4
double-count as a *quality* signal about an unrelated fresh setup.

---

## 1. Is retrieval/bucket filtering statistically justified on sparse history?

No, not at the thresholds in the code. The system **adapts at `MIN_SAMPLE = 5` and "trusts" at
`TRUST_SAMPLE = 10`.** Those numbers are statistically indefensible for a win-rate gate.

Standard error of a win-rate estimate (worst case p=0.5), 95% CI ≈ ±1.96·√(p(1−p)/n):

| n | 95% CI on WR | what you can distinguish |
|---|---|---|
| 5  | ±0.44 | nothing — CI spans almost [0,1] |
| 10 | ±0.31 | cannot tell 45% from 65% |
| 22 | ±0.21 | cannot tell a coin flip from a real edge |
| 30 | ±0.18 | barely |
| 100 | ±0.10 | a *large* edge, maybe |
| 400 | ±0.05 | a modest edge |

To distinguish a genuine 58% bucket from a 50% coin flip (effect 0.08) at p<0.05 you need
**~150–400 trades in that bucket.** The gate moves on 5.

**Expectancy is worse, not better, at small n.** Per-trade R is a heavy-tailed Bernoulli payoff. At
p=0.5, rr=1.3 the per-trade SD ≈ 1.15R, so the SE of estimated expectancy at n=10 is ≈ **±0.36R** —
the interval around a "+0.15R" bucket comfortably includes large negatives. With *real* fat-tailed
losses (the SPY backtest shows avg_win +$62 vs avg_loss −$203) the estimate is dominated by whether
you happened to catch one tail loss yet. A point estimate of expectancy at n=10–30 is mostly noise.

**The 4-level fallback is self-defeating.** When the specific bucket is too sparse it pools up to
L3 `(signal_type, bot)` to clear MIN_SAMPLE. But the entire reason for conditioning on
regime/time/vix was that those *change the distribution*. So you either get a specific bucket (and
sparsity) or a pooled bucket (and you've thrown away the conditioning) — and pooling across regimes
is exactly the Simpson's-paradox trap the time-of-day/vol control research already flagged on this
strategy. There is no n at which both "specific" and "adequately sampled" hold here.

**Verdict:** below ~50–100 per *terminal* bucket this is noise-fitting / pseudo-confidence. The
current 5/10 thresholds manufacture confidence the data cannot support.

---

## 2. Is the architecture creating a cold-start deadlock?

**Yes — confirmed at the data layer, but the cause is misattributed.** The loop is real:
strict static floor (2.0 R:R) rejects the strategy's natural 1.25–1.33R setups → nothing fills →
`learning.db` never accrues MES buckets → `adaptive_lookup` returns `None` forever → the static floor
stays. `learning.db` has zero MES rows after live running; that is the smoking gun.

But note **the deadlock is caused by the *static fallback*, not by RAG/adaptive.** A learning layer
cannot deadlock you; the mis-calibrated conservative default it falls back to does. This reframes
the fix away from "reduce RAG authority" toward "make the cold-start floor expectancy-aware so it
permits positive-EV trades and lets data accrue" — which is the Phase-0 shadow work already in
flight (`expectancy_priors.py`, `docs/expectancy_gate_proposal.md`). A bounded *exploration budget*
(take N min-size below-threshold trades specifically to populate buckets) is the cleaner structural
cure; freezing is the disease.

---

## 3. Deterministic + light overlays (A) vs retrieval-heavy approval (B)

**A wins decisively at this scale, across all four conditions.** The deciding factor is
degrees of freedom vs. data. A retrieval/contingency system over a 5-dim bucket space has, in
effect, one free parameter *per bucket* — dozens to hundreds of DoF — fit against ~36 observations.
A deterministic strategy with a handful of walk-forward-validated parameters has few DoF and a known
OOS profile.

| condition | A: deterministic + light overlay | B: retrieval-heavy |
|---|---|---|
| MES intraday futures | Robust. Structure/EMA/ATR logic is the actual edge; walk-forward validated. | Sparse; buckets never populate; inert or noise. |
| SPY options flow | Robust if gates stay deterministic. | Worst case: option outcomes are path/IV/greeks-dependent, so "similar past setup" is a weak key; n=1–6 buckets mis-fire. |
| Changing vol regimes | A degrades gracefully (regime filter is explicit, rule-based). | B is fragile: cumulative buckets (no decay — see §Failure 5) weight a 2025-03 trade equally with yesterday. |
| Sparse live data | A doesn't need live data to function. | B *requires* the data it can't get → cold-start. |

Retrieval-heavy approval only starts to make sense at *thousands* of trades per regime, which a PDT-
constrained, ~50–70-trades/year strategy will not reach for years, by which point the regime has
turned over anyway.

---

## 4. Does RAG add genuine edge, or comfort / explainability / false precision / overfit?

Honest split, given RAG isn't in the gate:

- **Genuine edge (gating): none, currently.** It isn't in the path, and at n≈36 it couldn't add
  calibrated edge if it were.
- **Explainability / journaling: real and worth keeping.** LLM retrieval over `rag_data/` for daily
  review and post-hoc attribution is legitimately useful — *for humans, after the fact.*
- **False precision: the main risk if promoted.** A "this resembles past winners, confidence 0.78"
  score over sparse data is precision the sample size cannot justify. It launders a wide posterior
  into a crisp number.
- **Overfit / sparse-data illusion: high** if it ever gains gating authority at this scale,
  *especially* the dangerous direction — a lucky 4/5 bucket currently *eases* R:R to 1.5 and
  confidence to 0.40 (`derive_thresholds`). Sparse data here doesn't just fail to help; it
  actively loosens the gate toward a random bucket.

---

## 5. What authority should RAG / the adaptive layer have?

- **RAG (LLM/`rag_data`): advisory + journaling only. No sizing, no posture, no veto.** It is a
  human-facing explanation and research tool. Keep it out of the automated path.
- **Adaptive bucket layer: advisory → at most sizing, earning authority continuously with n.** The
  *one* hard-block it may keep is the asymmetric protective case: **auto-suppress a bucket with
  proven negative expectancy at adequate n.** That's defensible because it protects capital and the
  error is one-directional (a false suppress costs opportunity, not money). Today's suppress requires
  only `n≥10` — raise that bar; `n=10` cannot establish "bleeding."
- **It should never *loosen* a gate on small n.** Easing R:R/confidence on a 4/5 bucket is the
  inverse of prudent — remove the loosening branches until buckets are deep.

No learned or retrieved component should be able to override a deterministic hard risk stop.

---

## 6. Expectancy-based vs similarity-based gating

**Expectancy-based is superior — with one honest caveat.** Your example: a setup that historically
wins 58% at 1.3R has E_R = 0.58·1.3 − 0.42 = **+0.33R**. That is more trustworthy than "similar past
trades" because:

- It is a **sufficient statistic tied directly to the objective** (money), not a vague resemblance.
- It is **falsifiable and auditable** — you can pre-register a threshold and check it OOS.
- It admits a **natural prior and shrinkage** (Beta on WR, blend toward a backtest prior, weight
  ∝ n), so it **degrades gracefully** at low n. Similarity has no natural shrinkage — at n=3 it just
  confidently retrieves 3 neighbors.

Caveat: expectancy is *harder to estimate* than WR (it's WR × a fat-tailed payoff), so at small n the
expectancy point estimate is itself noisy (§1). The resolution is not to prefer similarity — it's to
use expectancy **with an explicit prior and credible interval, and act on the conservative bound.**
Similarity-based gating is the weakest of the three: small-n, no calibration, no shrinkage, and a
poor distance key for options especially.

---

## 7. Minimum architecture I would trust for a small-data live system

Prioritizing robustness, anti-overfit, operational simplicity, regime resilience, explainability:

1. **Deterministic strategy core** — structure logic, EMA stack, ATR risk model, regime filter,
   signal-quality gates. Frozen, versioned, walk-forward validated. **This carries the edge.**
2. **Deterministic risk/posture layer** — daily-loss kill switch, streak throttle, position caps,
   PDT handling. Non-negotiable, dominates everything, no learned override.
3. **One adaptive overlay — expectancy with a prior.** Per *coarse* bucket (signal_type, maybe
   regime — **not** a 5-way split). Beta/Normal posterior shrinking to a backtest prior; authority
   scales *continuously* with n (no cliff at 5). Advisory → sizing. Hard-block only on
   proven-negative-expectancy-at-adequate-n.
4. **Memory = append-only outcome log + human journal.** You already have the log. RAG narration
   sits here, read-only to humans.
5. **Confidence = a posterior with a credible interval, not a point score.** Wide interval ⇒ defer
   to the deterministic prior; don't adapt.

Everything past item 3 is optional and must earn its place with out-of-sample evidence.

---

## 8. If designing from scratch today

- **Deterministic rules:** the strategy + the risk layer. Versioned, frozen between validated
  releases, walk-forward gated before any change ships.
- **Adaptive learning:** a single Bayesian expectancy estimator on a *low-dimensional* bucket key.
  Posterior = prior (from backtest) updated by observed trades; influence grows with n. No
  hand-tuned breakpoint ladder.
- **Memory:** append-only trades + outcomes; immutable; the source of truth for later analysis.
- **RAG:** LLM context for human daily review and attribution. **Zero** automated authority.
- **Confidence:** report the expectancy posterior mean *and* its lower credible bound; gate on the
  lower bound (conservative), not the mean.
- **Risk throttling:** purely deterministic and supreme. A learned layer can shrink size; it can
  never grow size past the deterministic cap nor lift a stop.
- **Cold-start handling:** (a) seed priors from the backtest; (b) make the fallback floor
  expectancy-aware so positive-EV setups *can* trade; (c) a **capped exploration budget** — a small
  number of min-size below-threshold trades whose purpose is to populate buckets. Never let a
  conservative default create a no-fill deadlock.

---

## Failure modes (enumerated)

1. **Multiplicative bucket fragmentation.** 5-dim exact match × ~36 events ⇒ chronic `None` ⇒ the
   "adaptive system" is theater; the static fallback silently does all the work.
2. **Fallback collapse destroys conditioning.** L3 pooling to escape sparsity averages across the
   very regimes the buckets existed to separate (Simpson's paradox).
3. **Cold-start deadlock** (§2) — confirmed.
4. **Small-n mis-fire toward looser gates.** A lucky 4/5 bucket eases R:R to 1.5 and confidence to
   0.40. Sparse data actively *loosens* the gate. This is the dangerous one.
5. **No recency decay.** `bucket_stats` is pure cumulative (`n_trades = n_trades + 1`, no half-life).
   A trade from over a year ago counts equally with yesterday — fatal under regime turnover.
6. **Unvalidated heuristic ladder.** `derive_thresholds` breakpoints (WR 0.35/0.45/0.55/0.65;
   edge_ratio 0.6/1.5/2.0; floors 0.40–0.75) are plausible-looking *free parameters*, none
   walk-forward validated — researcher degrees of freedom hard-coded as constants.
7. **Q4 is a mislabeled regime proxy.** `_similar_recent_lost` is direction- and signal-type-blind;
   it vetoes good fresh setups on unrelated recent losses.

---

## What stays deterministic / what stays adaptive / what should NOT exist yet

- **Deterministic (keep):** structure & EMA logic, ATR risk model, regime filter, signal-quality
  gates, posture/risk management, kill switches, walk-forward-validated overlays (e.g. `box_rng_atr`
  long-only, default-off shadow). Everything that protects capital or is OOS-validated.
- **Adaptive (allow, constrained):** a single shrinkage expectancy estimator, advisory→sizing,
  authority growing with n; protective negative-EV suppress at a *raised* sample bar.
- **Should NOT exist yet:** RAG/similarity as a gate; the 5-dimensional bucketing; the multi-
  breakpoint `derive_thresholds` ladder; *any* gate-loosening from small-n buckets; any hard-block
  from a learned component other than the protective suppress.

---

## Final verdict: does RAG deserve veto power at current scale?

**No — unambiguously.** And, correctly, it does not have it today. The actionable risks are (1) never
*granting* it veto power, and (2) recognizing that the component which *does* hold authority — the
sparse exact-match expectancy table behind a static fallback — is itself below the sample sizes that
justify gating, and is the true source of the cold-start deadlock. Fix the fallback (expectancy-aware,
default-off shadow first), shrink the bucket dimensionality, raise the trust thresholds, delete the
small-n loosening branches, and keep RAG advisory. The edge lives in the deterministic core; the
learned layer should earn authority slowly and asymmetrically, and RAG should stay a tool for humans.
