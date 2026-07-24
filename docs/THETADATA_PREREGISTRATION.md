# ThetaData Flow Research — PRE-REGISTRATION (locked)

**Locked:** 2026-07-23, BEFORE the ThetaData adapter or any battery code is
written. Changing anything below after data is seen invalidates the result.
This file is the referee. If a finding requires editing this file to pass, it
FAILS.

**Scope:** observation research only. TC, confidence, governor, allocation,
risk, exits, and all production behavior are FROZEN and untouched.

---

## 0. Confirmed data conventions (verified before lock)

- **`spy_signals.sent_at` is UTC** (naive, `datetime.utcnow()`). Verified
  2026-07-23: hour histogram clusters 13:00–19:00 = RTH+4h; `18:35` sent_at →
  ET-derived bucket `PRE_POWER` (14:35 ET). The join converts UTC→ET
  (`snapshotter._to_et_iso`, test-locked). A prior "naive ET" comment was wrong.
- **Flow prints `ts_et` are true ET** (converted from source UTC at ingest).
- **Window:** 30-minute trailing (`window_s=1800`) ending at the context timestamp.
- **Universe:** SPY options, 0–2 DTE (where day-trade flow lives).
- **Aggressor:** Lee-Ready vs synchronized NBBO; only `aggressor_src=QUOTE` +
  clean condition codes count in NET measures. Spread legs / auctions / late
  prints excluded. Mid prints excluded from net.
- **Outcomes (already existing, not recomputed):** `pnl_pct` (option-P&L proxy),
  `spy_price`→`spy_price_exit` (underlying move), `outcome` win/loss.

## 1. Primary hypothesis (ONE, pre-specified)

> **H1:** delta-weighted aggressor flow (`dw_flow`) over the 30-min pre-context
> window has **positive information lift** on the context's forward option P&L
> (`pnl_pct`), **after residualizing on the existing price features**
> {confidence, vix, iv_rank, volume_spike_mult, spread_pct, intraday_pc_ratio,
> external_composite, sentiment_score, flow_score}.

Direction predicted: higher `dw_flow` (bullish transacted delta) → higher
forward P&L on call-side signals; symmetric on puts. Test is two-sided at α=0.05.

## 2. Secondary / exploratory features (heavier penalty)

pc_prem_imbalance, net_call_prem, net_put_prem, sweep_intensity, block_prem,
oc_open_ratio, expiry_concentration, strike_repetition, atm_vs_wing,
iv_weighted_side. These are exploratory: reported under BH-FDR, cannot by
themselves trigger PASS. Note: on IB `sweep_intensity` is untrustworthy
(consolidated tape); it becomes valid ONLY with ThetaData venue+condition data.

## 3. Fixed analysis choices (no post-hoc tuning)

- Horizons: forward option P&L at exit (primary) + underlying return at
  1/5/15/30 min (secondary).
- Buckets: quintiles (q=5), computed on the pooled in-sample set, applied
  unchanged out-of-sample.
- Samples: (a) 657 dispatched signals, (b) rejected/blocked control, (c) all
  production fills, (d) unconditional interval snapshots (13×/session).
- No per-feature window tuning, no threshold search, no bucket-count search.

## 4. Out-of-sample split (pre-declared)

- **Train (in-sample):** 2026-04-01 → 2026-05-31.
- **Test (out-of-sample):** 2026-06-01 → 2026-07-23.
- **Purge/embargo:** drop any train/test pair whose 30-min windows overlap the
  split boundary; embargo 1 trading day around the cut.
- A feature discovered/estimated in-sample must reproduce out-of-sample with the
  **same sign** and magnitude **≥ ½** of in-sample.

## 5. Multiple-comparisons plan (pre-declared)

- Primary family = {H1}. Must clear **Bonferroni** at α=0.05 (i.e. raw p < 0.05,
  single pre-registered test) AND permutation/shuffle p < 0.05.
- Secondary family = the 10 exploratory features × horizons. Controlled by
  **Benjamini-Hochberg FDR** at q=0.10. Secondary "hits" are reported as
  hypotheses for a FUTURE pre-registration, never as this project's PASS.
- Every claimed effect also passes the label-shuffle control (effect vanishes
  under permutation).

## 6. PASS / FAIL (locked thresholds)

**PASS requires ALL six:**

1. **Significant incremental info** — H1 clears Bonferroni (raw p<0.05) AND
   shuffle p<0.05.
2. **Survives OOS** — H1 present on 2026-06-01→07-23 holdout, same sign,
   magnitude ≥ ½ in-sample, purged/embargoed.
3. **Not already explained** — information-lift partial correlation > 0 after
   residualizing on all existing features; max |Spearman corr| of `dw_flow` with
   any existing feature < 0.50.
4. **Improves the 657 signals** — top-vs-bottom quintile forward-P&L spread on
   the dispatched-signal set is significant (not intervals-only).
5. **Separates winners/losers** — barrier-win rate (P(+0.5% before −0.5%)) in the
   top quintile > 50% AND Wilson-lower-bound above the bottom quintile.
6. **Economically meaningful after costs** — top-minus-bottom quintile P&L spread
   exceeds realistic 0DTE spread+slippage under the executor-real fill model.

**Any single criterion fails → FAIL. Exact ties → FAIL. "Close" → FAIL.**

## 7. Decision rule (pre-committed)

- **FAIL →** stop options-flow research. Do not renew ThetaData. Do not invent a
  replacement hypothesis. Record the null in the audit docs. Done.
- **PASS →** authorize ONLY a shadow observation layer (log flow beside signals,
  influence nothing) that must prove itself forward ≥20 live sessions under a
  governor-style evidence tier before any separate proposal to touch production.

## 8. Power note (honest limitation, pre-stated)

657 signals ≈ 130/quintile. Underpowered for small effects — deliberately. If an
effect needs >657 observations to reach significance, it is too small to trade
and we treat undetectability as FAIL. The interval sample (~1500+) backstops
power for the unbiased (non-signal) test only.

---

**Signed off (pre-data):** 2026-07-23. Referee file — do not edit post-hoc.
